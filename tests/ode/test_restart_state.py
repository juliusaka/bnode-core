import random
from pathlib import Path

import numpy as np
import pytest
import torch

from bnode_core.nn.nn_utils.early_stopping import EarlyStopping
from bnode_core.ode.trainer_utils.restart_checkpoint_store import RestartCheckpointStore
from bnode_core.ode.trainer_utils.restart_state import (
    RESTART_CHECKPOINT_FILENAME,
    TrainAllPhasesState,
    TrainOnePhaseState,
    capture_rng_state,
    restore_rng_state,
)


def _sample_rng_triplet():
    return (
        torch.rand(4),
        np.random.rand(4),
        [random.random() for _ in range(4)],
    )


def test_train_all_phases_state_roundtrip(tmp_path):
    restart_path = tmp_path / "training_outer_restart.pt"
    state = TrainAllPhasesState(
        job_idx=3,
        next_epoch_anchor=17,
        mlflow_run_id="run-outer",
    )

    state.save(restart_path)
    loaded_state = TrainAllPhasesState().load(restart_path)

    assert loaded_state.job_idx == 3
    assert loaded_state.next_epoch_anchor == 17
    assert loaded_state.mlflow_run_id == "run-outer"


def test_train_one_phase_state_roundtrip_with_rng_and_early_stopping(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    restart_path = hydra_output_dir / "training_inner_restart.pt"

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    source_early_stopping = EarlyStopping(
        patience=9,
        verbose=False,
        threshold=0.25,
        threshold_mode="rel",
        path=str(hydra_output_dir / "best_model.pt"),
        optimizer_path=str(hydra_output_dir / "best_optimizer.pt"),
        trace_func=lambda *_args, **_kwargs: None,
    )
    source_early_stopping.counter = 3
    source_early_stopping.best_score = 0.4
    source_early_stopping.corresponding_score = 0.6
    source_early_stopping.early_stop = False
    source_early_stopping.score_last_save = 0.4

    state = TrainOnePhaseState(
        phase_epoch=4,
        nan_counter=4,
        grad_norm_last_reduced_counter=2,
        stable_epochs=5,
        deterministic_mode_active=True,
        seq_len_increase_in_batches=91,
        rng_state=capture_rng_state(use_cuda=False),
    )
    state.early_stopping = source_early_stopping
    state.save(restart_path)

    expected_torch, expected_numpy, expected_python = _sample_rng_triplet()
    _ = _sample_rng_triplet()

    restored_early_stopping = EarlyStopping(
        patience=1,
        verbose=True,
        threshold=0.0,
        threshold_mode="abs",
        path=Path("placeholder.pt"),
        optimizer_path=Path("placeholder_optimizer.pt"),
        trace_func=lambda *_args, **_kwargs: None,
    )
    loaded_state = TrainOnePhaseState()
    loaded_state.early_stopping = restored_early_stopping
    loaded_state.load(restart_path)

    assert loaded_state.phase_epoch == 4
    assert loaded_state.nan_counter == 4
    assert loaded_state.grad_norm_last_reduced_counter == 2
    assert loaded_state.stable_epochs == 5
    assert loaded_state.deterministic_mode_active is True
    assert loaded_state.seq_len_increase_in_batches == 91

    assert restored_early_stopping.patience == 9
    assert restored_early_stopping.threshold == 0.25
    assert restored_early_stopping.threshold_mode == "rel"
    assert restored_early_stopping.counter == 3
    assert restored_early_stopping.path == str(hydra_output_dir / "best_model.pt")
    assert restored_early_stopping.optimizer_path == str(
        hydra_output_dir / "best_optimizer.pt"
    )

    restore_rng_state(loaded_state.rng_state, use_cuda=False)
    restored_torch, restored_numpy, restored_python = _sample_rng_triplet()
    torch.testing.assert_close(restored_torch, expected_torch)
    np.testing.assert_allclose(restored_numpy, expected_numpy)
    assert restored_python == expected_python


def test_checkpoint_store_saves_epoch_checkpoint_syncs_effective_seq_len(tmp_path):
    train_all_phases_state = TrainAllPhasesState(
        job_idx=2,
        next_epoch_anchor=6,
    )
    train_one_phase_state = TrainOnePhaseState(
        phase_epoch=4,
        seq_len_increase_in_batches=12,
        rng_state=capture_rng_state(use_cuda=False),
    )
    checkpoint_store = RestartCheckpointStore(
        checkpoint_path=tmp_path / RESTART_CHECKPOINT_FILENAME
    )

    class DummyScaler:
        def state_dict(self):
            return {"scale": 1.0}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    class DummyOptimizer:
        def state_dict(self):
            return {"lr": 0.001}

    dummy_model = DummyModel()
    dummy_optimizer = DummyOptimizer()

    checkpoint_store.save_epoch_checkpoint(
        train_all_phases_state=train_all_phases_state,
        train_one_phase_state=train_one_phase_state,
        lr_schedulers=None,
        scaler=DummyScaler(),
        model=dummy_model,
        optimizer=dummy_optimizer,
    )

    outer_state, inner_state, scheduler_states, scaler_state, model_state, optimizer_state = checkpoint_store.load_checkpoint_if_available()

    assert outer_state.job_idx == 2
    assert outer_state.next_epoch_anchor == 6
    assert inner_state.phase_epoch == 4
    assert inner_state.seq_len_increase_in_batches == 12
    assert scheduler_states == {}
    assert scaler_state == {"scale": 1.0}
    assert model_state is not None
    assert "linear.weight" in model_state
    assert optimizer_state == {"lr": 0.001}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_train_one_phase_state_raises_for_missing_registry_attr(tmp_path, monkeypatch):
    restart_path = tmp_path / RESTART_CHECKPOINT_FILENAME
    state = TrainOnePhaseState()
    monkeypatch.setattr(
        TrainOnePhaseState,
        "FIELD_REGISTRY",
        [
            *TrainOnePhaseState.FIELD_REGISTRY,
            ("nonexistent_field", "_some_buf", torch.int64, int, int),
        ],
    )

    with pytest.raises(AttributeError):
        state.save(restart_path)
