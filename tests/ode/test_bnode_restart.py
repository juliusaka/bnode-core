"""Tests for the trainer restart/resume mechanism.

These tests verify that training can be interrupted mid-run and resumed from
the exact same Hydra output directory, reproducing identical final model weights
to a clean reference run.

See ``bnode_test_helpers.ode_training`` for how ``clear_output_before_start=False``
triggers the implicit reload: the restart-checkpoint files left on disk by the
interrupted leg are detected automatically by ``load_restart_checkpoint()`` inside
``train_all_phases`` at the start of the next ``trainer.main()`` call.
"""

import random
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from bnode_core.ode import trainer
from bnode_core.ode.trainer_utils.restart_checkpoint_store import RestartCheckpointStore
from bnode_core.ode.trainer_utils.restart_state import (
    CheckpointRequestedExit,
    RESTART_CHECKPOINT_FILENAME,
    TrainAllPhasesState,
    TrainOnePhaseState,
)

from bnode_test_helpers import ode_training


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------

def _resume_training_overrides(
    *,
    nn_model: str,
    max_epochs: tuple[int, ...],
    batches_per_epoch: int = 1,
    scheduler_phase: int | None = None,
    with_test_job: bool = False,
) -> list[str]:
    overrides = [
        f'nn_model={nn_model}',
        f'nn_model.training.test={"true" if with_test_job else "false"}',
        'use_cuda=false',
        'n_workers_train_loader=0',
        'n_workers_other_loaders=0',
        'prefetch_factor=null',
    ]
    for phase_idx, phase_max_epochs in enumerate(max_epochs):
        overrides.extend(
            [
                f'nn_model.training.main_training.{phase_idx}.max_epochs={phase_max_epochs}',
                f'nn_model.training.main_training.{phase_idx}.batches_per_epoch={batches_per_epoch}',
            ]
        )
    if scheduler_phase is not None:
        overrides.extend(
            [
                f'nn_model.training.main_training.{scheduler_phase}.use_lr_scheduler=true',
                f'nn_model.training.main_training.{scheduler_phase}.lr_scheduler_type=cosine',
                f'nn_model.training.main_training.{scheduler_phase}.cosine_T_max=2',
                f'nn_model.training.main_training.{scheduler_phase}.cosine_eta_min=1e-6',
            ]
        )
    return overrides


def _resume_mlflow_tracking_uri(scope: str) -> str:
    return f"file://{(Path('./tests/_results/ode/mlruns') / scope).absolute()}"


def _resume_mlflow_overrides(scope: str) -> list[str]:
    return [
        f'mlflow_tracking_uri={_resume_mlflow_tracking_uri(scope)}',
        f'mlflow_experiment_name={scope}',
    ]


def _fixed_seq_len_phase_overrides(phase_idx: int, seq_len_train: int = 3) -> list[str]:
    return [
        f'nn_model.training.main_training.{phase_idx}.seq_len_train={seq_len_train}',
        f'nn_model.training.main_training.{phase_idx}.seq_len_increase_in_batches=0',
    ]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _get_mlflow_run(tracking_uri: str, run_id: str):
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)


def _load_model_state(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def _set_training_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_model_states_equal(
    path_a: Path,
    path_b: Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    not_equal: bool = False,
) -> None:
    state_a = _load_model_state(path_a)
    state_b = _load_model_state(path_b)
    assert state_a.keys() == state_b.keys()
    if not_equal:
        with pytest.raises(AssertionError):
            for key in state_a:
                torch.testing.assert_close(state_a[key], state_b[key])
        return
    for key in state_a:
        assert_close_kwargs = {}
        if rtol is not None:
            assert_close_kwargs['rtol'] = rtol
        if atol is not None:
            assert_close_kwargs['atol'] = atol
        torch.testing.assert_close(state_a[key], state_b[key], **assert_close_kwargs)


def _assert_restart_state(
    outer_restart_state,
    inner_restart_state,
    *,
    expected_job_idx: int,
    deterministic_mode_active: bool | None = None,
) -> None:
    assert outer_restart_state.job_idx == expected_job_idx
    assert outer_restart_state.next_epoch_anchor >= inner_restart_state.phase_epoch
    if deterministic_mode_active is not None:
        assert inner_restart_state.deterministic_mode_active is deterministic_mode_active


def _assert_resumed_mlflow_run(
    output_dir: Path,
    *,
    mlflow_scope: str,
    outer_restart_state,
    expected_final_job_idx: int,
):
    resumed_run = _get_mlflow_run(
        _resume_mlflow_tracking_uri(mlflow_scope),
        outer_restart_state.mlflow_run_id,
    )
    assert resumed_run.data.params['mlflow_run_id'] == outer_restart_state.mlflow_run_id
    assert f'job_{expected_final_job_idx}_final_epoch' in resumed_run.data.metrics
    return resumed_run


# ---------------------------------------------------------------------------
# Interrupt helper
# ---------------------------------------------------------------------------

def _interrupt_after_n_epoch_saves(monkeypatch: pytest.MonkeyPatch, n_saves: int) -> None:
    """Interrupt training after *n_saves* epoch checkpoints have been written.

    Mechanism:
    1. ``monkeypatch.setattr`` replaces ``RestartCheckpointStore.save_epoch_checkpoint``
       with a counting wrapper, scoped to the enclosing ``monkeypatch.context()`` block
       so the real method is restored automatically when that block exits.
    2. The wrapper calls the *real* ``save_epoch_checkpoint`` first, so the restart-
       checkpoint files (outer state, inner state, LR schedulers, GradScaler) are fully
       written to disk before the exception is raised.
    3. On the Nth call the wrapper raises ``CheckpointRequestedExit``, a custom
       ``RuntimeError`` subclass.
    4. ``train_one_phase`` catches ``CheckpointRequestedExit`` explicitly, tags the
       MLflow run as ``"ended by checkpoint request"``, and returns cleanly — no
       exception escapes to the test.

    The checkpoint files being on disk before the exception is the key property that
    the resumed leg of the test relies on.
    """
    original_save = trainer.RestartCheckpointStore.save_epoch_checkpoint
    call_count = [0]

    def _patched_save(self, *args, **kwargs):
        # Real save runs first — files are on disk before we raise.
        result = original_save(self, *args, **kwargs)
        call_count[0] += 1
        if call_count[0] >= n_saves:
            raise CheckpointRequestedExit(
                f"Test-injected interrupt after {call_count[0]} epoch saves."
            )
        return result

    monkeypatch.setattr(
        trainer.RestartCheckpointStore,
        'save_epoch_checkpoint',
        _patched_save,
    )


# ---------------------------------------------------------------------------
# Reference fixtures (clean uninterrupted runs used as ground truth)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def resume_main_reference_dir():
    _set_training_seeds()
    return ode_training(
        'resume_main_reference',
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest',
            max_epochs=(3, 3),
            scheduler_phase=1,
        )
        + _fixed_seq_len_phase_overrides(1)
        + _resume_mlflow_overrides('resume_main_reference'),
    )


@pytest.fixture(scope='module')
def resume_deterministic_reference_dir():
    _set_training_seeds()
    return ode_training(
        'resume_deterministic_reference',
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest_det',
            max_epochs=(3, 3, 3),
        )
        + _fixed_seq_len_phase_overrides(1, seq_len_train=3)
        + _fixed_seq_len_phase_overrides(2, seq_len_train=3)
        + _resume_mlflow_overrides('resume_deterministic_reference'),
    )


# ---------------------------------------------------------------------------
# Restart / resume tests
# ---------------------------------------------------------------------------

def test_resume_from_same_hydra_output_dir_during_main_training(resume_main_reference_dir, monkeypatch):
    interrupted_case = 'resume_main_same_dir_interrupted'
    mlflow_scope = 'resume_main_same_dir'
    mlflow_overrides = _resume_mlflow_overrides(mlflow_scope)
    with monkeypatch.context() as ctx:
        _set_training_seeds()
        # Interrupt after two epoch-end checkpoint saves in the scheduler phase.
        _interrupt_after_n_epoch_saves(ctx, n_saves=4)
        interrupted_dir = ode_training(
            interrupted_case,
            overrides=_resume_training_overrides(
                nn_model='bnode_pytest',
                max_epochs=(3, 3),
                scheduler_phase=1,
            )
            + _fixed_seq_len_phase_overrides(1)
            + mlflow_overrides,
        )

    checkpoint_path = interrupted_dir / RESTART_CHECKPOINT_FILENAME
    assert checkpoint_path.exists()
    outer_restart_state, inner_restart_state, scheduler_states, scaler_state, _, _ = RestartCheckpointStore(checkpoint_path=checkpoint_path).load_checkpoint_if_available()
    _assert_restart_state(
        outer_restart_state,
        inner_restart_state,
        expected_job_idx=2,
        deterministic_mode_active=False,
    )
    assert 'cosine' in scheduler_states
    assert scheduler_states['cosine']['last_epoch'] > 0
    assert isinstance(scaler_state, dict)

    # Resume: keep the output directory so the restart-checkpoint files written
    # by the interrupted run remain on disk.  load_restart_checkpoint() inside
    # train_all_phases detects them automatically at startup and restores the
    # full training state without any explicit reload call in the test.
    _set_training_seeds()
    resumed_dir = ode_training(
        interrupted_case,
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest',
            max_epochs=(3, 3),
            scheduler_phase=1,
        )
        + _fixed_seq_len_phase_overrides(1)
        + mlflow_overrides,
        clear_output_before_start=False,
    )
    assert resumed_dir == interrupted_dir
    # Restart-checkpoint file is removed after a successful run.
    assert not checkpoint_path.exists()
    _assert_resumed_mlflow_run(
        resumed_dir,
        mlflow_scope=mlflow_scope,
        outer_restart_state=outer_restart_state,
        expected_final_job_idx=2,
    )
    _assert_model_states_equal(
        resumed_dir / 'model_phase_2.pt',
        resume_main_reference_dir / 'model_phase_2.pt',
    )


def test_resume_with_zeroed_model_weights_diverges_from_reference(
    resume_main_reference_dir, monkeypatch
):
    """Zeroing model weights in the restart checkpoint causes final weights to diverge.

    The model checkpoint (``model.pt``) saved alongside the restart-state files is
    the actual starting point for the resumed training leg.  This test corrupts it by
    setting every weight tensor to zero, then resumes.  The final model must differ
    from the clean reference run because the remaining epochs are trained from a
    zero-initialised model rather than from the properly trained epoch-2 weights.
    """
    interrupted_case = 'resume_corrupt_model_weights_interrupted'
    mlflow_overrides = _resume_mlflow_overrides('resume_corrupt_model_weights')

    with monkeypatch.context() as ctx:
        _set_training_seeds()
        _interrupt_after_n_epoch_saves(ctx, n_saves=2)
        interrupted_dir = ode_training(
            interrupted_case,
            overrides=_resume_training_overrides(
                nn_model='bnode_pytest',
                max_epochs=(3, 3),
            )
            + mlflow_overrides,
        )

    # Zero every weight tensor in the model state dict inside the restart bundle.
    checkpoint_path = interrupted_dir / RESTART_CHECKPOINT_FILENAME
    bundle = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    zeroed = {k: torch.zeros_like(v) for k, v in bundle['model'].items()}
    bundle['model'] = zeroed
    torch.save(bundle, checkpoint_path)

    # Resume: restart-checkpoint files are still on disk, so the trainer picks
    # them up automatically and resumes — but now from zeroed model weights.
    _set_training_seeds()
    resumed_dir = ode_training(
        interrupted_case,
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest',
            max_epochs=(3, 3),
        )
        + mlflow_overrides,
        clear_output_before_start=False,
    )
    assert resumed_dir == interrupted_dir

    # Final weights must differ from the clean reference because the remaining
    # training epochs started from zero instead of from epoch-2 weights.
    _assert_model_states_equal(
        resumed_dir / 'model_phase_2.pt',
        resume_main_reference_dir / 'model_phase_2.pt',
        not_equal=True,
    )


def test_resume_fails_when_rng_bytes_are_corrupted(monkeypatch):
    """A restart checkpoint with corrupt _rng_state_bytes cannot be loaded.

    This is a negative test: it verifies that _unpickle_rng_state propagates a
    clear error when the on-disk bytes are not valid pickle, rather than silently
    restoring garbage state into the running RNG.
    """
    with monkeypatch.context() as ctx:
        _set_training_seeds()
        _interrupt_after_n_epoch_saves(ctx, n_saves=2)
        interrupted_dir = ode_training(
            'resume_corrupt_rng_bytes_interrupted',
            overrides=_resume_training_overrides(
                nn_model='bnode_pytest',
                max_epochs=(3, 3),
            )
            + _resume_mlflow_overrides('resume_corrupt_rng_bytes'),
        )

    checkpoint_path = interrupted_dir / RESTART_CHECKPOINT_FILENAME

    # Overwrite _rng_state_bytes in the bundle's inner state with random garbage.
    bundle = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    bundle['inner']['_rng_state_bytes'] = torch.randint(0, 256, (64,), dtype=torch.uint8)
    torch.save(bundle, checkpoint_path)

    with pytest.raises(Exception):
        TrainOnePhaseState().load_from_state_dict(bundle['inner'])


def test_resume_from_same_hydra_output_dir_across_deterministic_activation(
    monkeypatch,
):
    interrupted_case = 'resume_deterministic_same_dir_interrupted'
    mlflow_scope = 'resume_deterministic_same_dir'
    mlflow_overrides = _resume_mlflow_overrides(mlflow_scope)
    with monkeypatch.context() as ctx:
        _set_training_seeds()
        # Interrupt after two epoch-end checkpoint saves in the first train phase, before
        # deterministic mode is activated in later phases.
        # We cannot interrupt after det-mode is active because the model
        # state-dict then contains zero-sized masked tensors that cannot be
        # loaded into a freshly-initialised model.
        _interrupt_after_n_epoch_saves(ctx, n_saves=2)
        interrupted_dir = ode_training(
            interrupted_case,
            overrides=_resume_training_overrides(
                nn_model='bnode_pytest_det',
                max_epochs=(3, 3, 3),
            )
            + _fixed_seq_len_phase_overrides(1, seq_len_train=3)
            + _fixed_seq_len_phase_overrides(2, seq_len_train=3)
            + mlflow_overrides,
        )

    checkpoint_path = interrupted_dir / RESTART_CHECKPOINT_FILENAME
    outer_restart_state, inner_restart_state, _, _, _, _ = RestartCheckpointStore(checkpoint_path=checkpoint_path).load_checkpoint_if_available()
    # Interrupted during phase 2 training before det-mode was applied.
    _assert_restart_state(
        outer_restart_state,
        inner_restart_state,
        expected_job_idx=1,
        deterministic_mode_active=False,
    )

    # Resume: keep the output directory so the restart-checkpoint files written
    # by the interrupted run remain on disk.  load_restart_checkpoint() inside
    # train_all_phases detects them automatically at startup and restores the
    # full training state without any explicit reload call in the test.
    _set_training_seeds()
    resumed_dir = ode_training(
        interrupted_case,
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest_det',
            max_epochs=(3, 3, 3),
        )
        + _fixed_seq_len_phase_overrides(1, seq_len_train=3)
        + _fixed_seq_len_phase_overrides(2, seq_len_train=3)
        + mlflow_overrides,
        clear_output_before_start=False,
    )

    assert resumed_dir == interrupted_dir
    # Restart state is removed after successful completion.
    assert not checkpoint_path.exists()
    # MLflow run ID is preserved and phase-3 metrics were logged.
    _assert_resumed_mlflow_run(
        resumed_dir,
        mlflow_scope=mlflow_scope,
        outer_restart_state=outer_restart_state,
        expected_final_job_idx=3,
    )
    # Verify deterministic mode was applied: the final model_phase_3.pt has the
    # masked tensor shapes (zero-sized latent dims) expected after det-mode.
    final_model_state = _load_model_state(resumed_dir / 'model_phase_3.pt')
    assert final_model_state['latent_ode_func.net.0.weight'].shape[1] == 0, (
        "Expected det-mode masked weights in final model_phase_3.pt"
    )


def test_resume_when_killed_during_test_job(monkeypatch):
    """Killing the process during the test job re-runs only the test job on resume.

    Sequence:
    1. Train two phases to completion.  Just before the test job runs,
       ``save_outer_restart_state_for_test_job`` advances ``job_idx`` to the
       test-job index and saves the outer restart state.
    2. A patched ``_run_test_job`` logs one metric and then raises
       ``CheckpointRequestedExit``, simulating a mid-test-job kill.
    3. The resume leg re-loads the outer state (job_idx = test job) and runs
       only the test job — no training epochs are repeated.
    4. The test verifies that MLflow double-logging (partial metric from the
       interrupted run + all metrics from the resume) raises no errors and
       that the metrics are present in the resumed run.
    """
    interrupted_case = 'resume_killed_during_test_job_interrupted'
    mlflow_scope = 'resume_killed_during_test_job'
    mlflow_overrides = _resume_mlflow_overrides(mlflow_scope)
    common_overrides = _resume_training_overrides(
        nn_model='bnode_pytest',
        max_epochs=(2, 2),
        with_test_job=True,
    ) + mlflow_overrides

    with monkeypatch.context() as ctx:
        _set_training_seeds()

        def _patched_run_test_job(*args, **kwargs):
            mlflow.log_metric('partial_test_metric', 1.0, step=0)
            raise CheckpointRequestedExit("Test-injected kill during test job.")

        ctx.setattr(trainer, '_run_test_job', _patched_run_test_job)
        interrupted_dir = ode_training(interrupted_case, overrides=common_overrides)

    checkpoint_path = interrupted_dir / RESTART_CHECKPOINT_FILENAME

    # Bundle must survive; outer state must point at the test job.
    outer_restart_state, _, _, _, _, _ = RestartCheckpointStore(checkpoint_path=checkpoint_path).load_checkpoint_if_available()
    # bnode_pytest job list: pre_train(skip)=0, phase0=1, phase1=2, test=3
    assert outer_restart_state.job_idx == 3, (
        f"Expected job_idx=3 (test job) after kill, got {outer_restart_state.job_idx}"
    )
    assert checkpoint_path.exists(), "Restart checkpoint must remain on disk after kill during test job."

    # Resume: outer state points at the test job, so only the test job re-runs.
    _set_training_seeds()
    resumed_dir = ode_training(
        interrupted_case,
        overrides=common_overrides,
        clear_output_before_start=False,
    )
    assert resumed_dir == interrupted_dir

    # Restart-checkpoint file is cleaned up after successful completion.
    assert not checkpoint_path.exists()

    # MLflow: double-logging the same step is safe — both runs' metrics are
    # present and no exception was raised during resume.
    tracking_uri = _resume_mlflow_tracking_uri(mlflow_scope)
    resumed_run = _get_mlflow_run(tracking_uri, outer_restart_state.mlflow_run_id)
    assert 'partial_test_metric' in resumed_run.data.metrics, (
        "partial_test_metric from the interrupted run must still be present in MLflow."
    )
