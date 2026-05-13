"""Tests for CUDA out-of-memory recovery in train_all_phases.

train_all_phases wraps each phase in a ``while True`` retry loop.  When
``train_one_epoch`` raises ``RuntimeError("CUDA out of memory")``, the loop
applies a compounding batch-size reduction factor (×0.7 per OOM event) and
retries the whole phase without aborting.

These tests use monkeypatch to:
- inject synthetic OOM errors via a patched ``train_one_epoch``
- capture the ``batch_size_reduction_factor`` passed to
  ``_create_datasets_and_dataloaders_for_job`` on each attempt
- suppress the 10-second ``time.sleep`` call so tests run quickly
"""

import pytest

from bnode_core.ode import trainer

from bnode_test_helpers import ode_training


# ---------------------------------------------------------------------------
# Common overrides for minimal, deterministic OOM tests
# ---------------------------------------------------------------------------

_OOM_TEST_OVERRIDES = [
    'use_cuda=false',
    'n_workers_train_loader=0',
    'n_workers_other_loaders=0',
    'prefetch_factor=null',
    'nn_model.training.test=false',
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_oom_retry_reduces_batch_size_with_compounding_factor(monkeypatch):
    """OOM in train_one_epoch retries with a compounding batch-size factor.

    Mechanism under test (see train_all_phases in trainer.py):
    1. Each phase is wrapped in a ``while True`` retry loop.
    2. When ``RuntimeError("CUDA out of memory")`` escapes ``train_one_phase``,
       the loop increments ``oom_reduction_count`` and multiplies
       ``batch_size_reduction_factor`` by 0.7 (starting from 1.0 on first OOM).
    3. ``_create_datasets_and_dataloaders_for_job`` is called again with the
       updated factor, which it applies to both ``batch_size_train`` and
       ``batch_size_valid_test`` (clamped to a minimum of 1).
    4. The phase is *not* aborted; training continues until it succeeds.
    5. The factor is reset to ``None`` at the start of every new phase.

    This test injects two consecutive OOMs then lets training proceed normally.
    It verifies:
    - the factor is ``None`` on the first attempt (no OOM yet)
    - the factor is ≈0.7 after the first OOM
    - the factor compounds to ≈0.49 after the second OOM
    - training completes and ``model.pt`` is written
    """
    # Suppress the 10-second OOM sleep so the test finishes quickly.
    monkeypatch.setattr(trainer.pyTime, 'sleep', lambda _: None)

    original_train_one_epoch = trainer.train_one_epoch
    original_create_dataloaders = trainer._create_datasets_and_dataloaders_for_job

    train_epoch_call_count = [0]
    captured_factors = []

    def _oom_on_first_two_calls(*args, **kwargs):
        """Raises a synthetic OOM on the first two calls; delegates afterwards."""
        train_epoch_call_count[0] += 1
        if train_epoch_call_count[0] <= 2:
            raise RuntimeError("CUDA out of memory: injected by test")
        return original_train_one_epoch(*args, **kwargs)

    def _capture_factor(*args, batch_size_reduction_factor=None, **kwargs):
        """Records the reduction factor and delegates to the real function."""
        captured_factors.append(batch_size_reduction_factor)
        return original_create_dataloaders(
            *args,
            batch_size_reduction_factor=batch_size_reduction_factor,
            **kwargs,
        )

    monkeypatch.setattr(trainer, 'train_one_epoch', _oom_on_first_two_calls)
    monkeypatch.setattr(trainer, '_create_datasets_and_dataloaders_for_job', _capture_factor)

    output_dir = ode_training('oom_retry', overrides=_OOM_TEST_OVERRIDES)

    # Training must not abort — model.pt must be present.
    assert (output_dir / 'model.pt').exists(), (
        "model.pt missing after OOM recovery — phase was likely aborted."
    )

    # Dataloader creation is called once per attempt before train_one_phase.
    # Expected sequence for the first phase:
    #   attempt 1 (no OOM yet)     → factor is None
    #   attempt 2 (after 1 OOM)    → factor ≈ 0.7
    #   attempt 3 (after 2 OOMs)   → factor ≈ 0.49  (0.7²)
    assert len(captured_factors) >= 3, (
        f"Expected ≥3 dataloader-creation calls (got {len(captured_factors)}); "
        "OOM retries may not have reached _create_datasets_and_dataloaders_for_job."
    )
    assert captured_factors[0] is None, (
        "First attempt must use config batch sizes unchanged (factor should be None)."
    )
    assert captured_factors[1] == pytest.approx(0.7), (
        "After 1 OOM the factor must be ≈0.7 (first compound step)."
    )
    assert captured_factors[2] == pytest.approx(0.7 ** 2), (
        "After 2 OOMs the factor must compound to ≈0.49 (0.7²)."
    )

    # The factor is reset to None at the start of every new phase.
    # If the config has more than one training phase, the next phase's first
    # captured factor must be None again.
    phase_boundaries = [i for i, f in enumerate(captured_factors) if f is None]
    assert phase_boundaries[0] == 0, "First dataloader call in phase 1 must have factor=None."
    if len(phase_boundaries) > 1:
        # Each phase after the first also starts fresh.
        for boundary_idx in phase_boundaries[1:]:
            assert captured_factors[boundary_idx] is None, (
                f"Factor not reset to None at start of new phase (index {boundary_idx})."
            )
