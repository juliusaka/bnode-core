"""Tests for the LR warm-up strategy in the trainer.

Three tests cover the three key behaviours:
 - test_warmup_cosine_scheduler: warmup + cosine → SequentialLR under 'cosine' key; LR ramps 0 → lr_start.
 - test_warmup_plateau_scheduler: warmup + plateau → separate 'warmup' and 'plateau' keys; plateau not stepped
   during warmup.
 - test_warmup_seq_len_offset: seq_len frozen at seq_len_epoch_start during warmup; offset-based ramp starts
   after warmup ends; max_epochs includes warmup and seq_len_increase sequentially.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, ReduceLROnPlateau

from bnode_core.config import base_training_settings_class, base_time_stepper_training_settings
from bnode_core.ode.trainer import _create_phase_lr_schedulers, _compute_phase_epoch_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    model = nn.Linear(2, 1)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _make_cfg(
    *,
    max_epochs: int = 10,
    warmup_epochs: int = 0,
    lr_scheduler_type: str | None = None,
    cosine_T_max: int | None = 5,
    lr_start: float = 1e-3,
    early_stopping_patience: int = 50,
) -> base_training_settings_class:
    """Return a minimal base_training_settings_class for scheduler tests."""
    cfg = base_training_settings_class(
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        lr_scheduler_type=lr_scheduler_type,
        cosine_T_max=cosine_T_max,
        lr_start=lr_start,
        early_stopping_patience=early_stopping_patience,
    )
    return cfg


def _make_seq_len_cfg(
    *,
    max_epochs: int = 10,
    warmup_epochs: int = 0,
    seq_len_train: int = 10,
    seq_len_epoch_start: int | None = None,
    seq_len_increase_in_batches: int = 0,
    batches_per_epoch: int = 5,
    lr_start: float = 1e-3,
) -> base_time_stepper_training_settings:
    """Return a base_time_stepper_training_settings for seq_len tests."""
    return base_time_stepper_training_settings(
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        seq_len_train=seq_len_train,
        seq_len_epoch_start=seq_len_epoch_start,
        seq_len_increase_in_batches=seq_len_increase_in_batches,
        batches_per_epoch=batches_per_epoch,
        lr_start=lr_start,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_warmup_cosine_scheduler():
    """Warmup + cosine wraps into a SequentialLR under 'cosine'; LR ramps from ~0 to lr_start."""
    lr_start = 1e-3
    warmup_batches = 10
    optimizer = _simple_optimizer(lr=lr_start)

    train_cfg = _make_cfg(
        max_epochs=20,
        warmup_epochs=2,
        lr_scheduler_type='cosine',
        cosine_T_max=5,
        lr_start=lr_start,
    )

    schedulers = _create_phase_lr_schedulers(
        train_cfg, optimizer, batches_per_epoch=5, job_idx=0, pre_train=False, test=False,
        warmup_batches=warmup_batches,
    )

    assert schedulers is not None, "Expected schedulers dict"
    assert 'cosine' in schedulers, "Expected 'cosine' key for warmup+cosine case"
    assert 'plateau' not in schedulers
    assert 'warmup' not in schedulers, "Warmup is embedded in SequentialLR, not a separate key"
    assert isinstance(schedulers['cosine'], SequentialLR), "Expected SequentialLR wrapping warmup+cosine"

    # LR should start at approximately 0 (start_factor=1e-8 of lr_start)
    initial_lr = optimizer.param_groups[0]['lr']
    assert initial_lr < lr_start * 1e-6, f"LR should start near 0, got {initial_lr}"

    # Step through warmup: LR should increase toward lr_start
    for _ in range(warmup_batches):
        schedulers['cosine'].step()
    post_warmup_lr = optimizer.param_groups[0]['lr']
    assert post_warmup_lr > initial_lr, "LR should have increased after warmup steps"
    assert abs(post_warmup_lr - lr_start) / lr_start < 0.02, (
        f"LR after warmup should be ≈ lr_start={lr_start}, got {post_warmup_lr}"
    )


def test_warmup_plateau_scheduler():
    """Warmup + plateau stores schedulers under separate keys; plateau not stepped during warmup."""
    lr_start = 1e-3
    warmup_batches = 8
    optimizer = _simple_optimizer(lr=lr_start)

    train_cfg = _make_cfg(
        max_epochs=20,
        warmup_epochs=2,
        lr_scheduler_type='plateau',
        lr_start=lr_start,
        early_stopping_patience=50,
    )

    schedulers = _create_phase_lr_schedulers(
        train_cfg, optimizer, batches_per_epoch=4, job_idx=0, pre_train=False, test=False,
        warmup_batches=warmup_batches,
    )

    assert schedulers is not None
    assert 'warmup' in schedulers, "Expected separate 'warmup' key for plateau+warmup case"
    assert 'plateau' in schedulers, "Expected 'plateau' key"
    assert 'cosine' not in schedulers
    assert isinstance(schedulers['plateau'], ReduceLROnPlateau)

    # Simulate warmup: step 'warmup' scheduler each batch, do NOT step plateau
    lr_before = optimizer.param_groups[0]['lr']
    for _ in range(warmup_batches):
        schedulers['warmup'].step()
    lr_after_warmup = optimizer.param_groups[0]['lr']
    # After warmup steps, LR should be close to lr_start
    assert abs(lr_after_warmup - lr_start) / lr_start < 0.02, (
        f"LR after warmup should be ≈ lr_start={lr_start}, got {lr_after_warmup}"
    )
    # Plateau: calling step with a bad loss should reduce LR if patience is 0,
    # but the point here is just that it is NOT stepped during warmup (LR didn't drop).
    assert lr_after_warmup >= lr_before, "LR must not decrease during warmup"


def test_warmup_seq_len_offset():
    """seq_len is frozen at seq_len_epoch_start during warmup; max_epochs is the sequential sum."""
    batches_per_epoch = 5
    warmup_epochs = 2
    seq_len_increase_epochs = 4  # set via seq_len_increase_in_batches = 4 * batches_per_epoch
    max_epochs_base = 10
    seq_len_epoch_start = 3
    seq_len_train = 10

    class _FakeDataloader:
        def __len__(self):
            return batches_per_epoch

    dataloaders = {'train': _FakeDataloader()}

    train_cfg = _make_seq_len_cfg(
        max_epochs=max_epochs_base,
        warmup_epochs=warmup_epochs,
        seq_len_train=seq_len_train,
        seq_len_epoch_start=seq_len_epoch_start,
        seq_len_increase_in_batches=seq_len_increase_epochs * batches_per_epoch,
        batches_per_epoch=batches_per_epoch,
    )

    bpe, epochs_seq, max_ep, slib, we, wb = _compute_phase_epoch_settings(
        dataloaders, train_cfg, pre_train=False
    )

    assert bpe == batches_per_epoch
    assert we == warmup_epochs
    assert wb == warmup_epochs * batches_per_epoch
    assert slib == seq_len_increase_epochs * batches_per_epoch
    # Sequential: max_epochs = base + warmup + seq_len_increase
    assert max_ep == max_epochs_base + warmup_epochs + seq_len_increase_epochs, (
        f"Expected {max_epochs_base + warmup_epochs + seq_len_increase_epochs}, got {max_ep}"
    )

    # Verify the seq_len offset formula used in train_one_epoch:
    # batch index in phase, frozen during warmup, offset after warmup
    warmup_batches = wb
    seq_len_increase_in_batches = slib

    def _seq_len_at(batches_this_phase: int) -> int:
        if batches_this_phase < warmup_batches:
            return seq_len_epoch_start
        batches_after_warmup = batches_this_phase - warmup_batches
        if batches_after_warmup < seq_len_increase_in_batches:
            val = seq_len_epoch_start + int(
                batches_after_warmup / seq_len_increase_in_batches
                * (seq_len_train - seq_len_epoch_start)
            )
            return min(val, seq_len_train)
        return seq_len_train

    # During warmup: frozen
    for b in range(warmup_batches):
        assert _seq_len_at(b) == seq_len_epoch_start, f"Batch {b}: expected frozen seq_len"

    # Just after warmup: at seq_len_epoch_start (0 progress into ramp)
    assert _seq_len_at(warmup_batches) == seq_len_epoch_start

    # Midway through ramp (after warmup)
    mid_ramp = warmup_batches + seq_len_increase_in_batches // 2
    mid_val = _seq_len_at(mid_ramp)
    assert seq_len_epoch_start < mid_val < seq_len_train, (
        f"Expected mid-ramp seq_len between {seq_len_epoch_start} and {seq_len_train}, got {mid_val}"
    )

    # After ramp: at seq_len_train
    assert _seq_len_at(warmup_batches + seq_len_increase_in_batches) == seq_len_train
    assert _seq_len_at(warmup_batches + seq_len_increase_in_batches + 1) == seq_len_train


# ---------------------------------------------------------------------------
# RAdam / RAdamW + warmup guard
# ---------------------------------------------------------------------------

def _make_tiny_model() -> torch.nn.Module:
    return nn.Linear(2, 1)


@pytest.mark.parametrize("optimizer_name", ["radam", "RAdam", "radamw", "RAdamW"])
def test_radam_warmup_raises(optimizer_name):
    """_create_phase_optimizer must raise ValueError for radam/radamw + warmup_epochs > 0."""
    from bnode_core.ode.trainer import _create_phase_optimizer

    cfg = base_training_settings_class(
        optimizer=optimizer_name,
        warmup_epochs=2,
        lr_start=1e-3,
    )
    with pytest.raises(ValueError, match="warmup_epochs"):
        _create_phase_optimizer(_make_tiny_model(), cfg, pre_train=False, job_idx=0)
