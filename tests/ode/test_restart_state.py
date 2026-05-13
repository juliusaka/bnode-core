import random
from pathlib import Path

import numpy as np
import pytest
import torch

from bnode_core.nn.nn_utils.early_stopping import EarlyStopping
from bnode_core.ode.trainer_utils.restart_checkpoint_store import RestartCheckpointStore
from bnode_core.ode.trainer_utils.restart_state import (
    INNER_RESTART_STATE_FILENAME,
    OUTER_RESTART_STATE_FILENAME,
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
    restart_path = tmp_path / OUTER_RESTART_STATE_FILENAME
    state = TrainAllPhasesState(
        job_idx=3,
        next_epoch_anchor=17,
        mlflow_run_id="run-outer",
        checkpoint_uuid="checkpoint-uuid-1",
    )

    state.save(restart_path)
    loaded_state = TrainAllPhasesState().load(restart_path)

    assert loaded_state.job_idx == 3
    assert loaded_state.next_epoch_anchor == 17
    assert loaded_state.mlflow_run_id == "run-outer"
    assert loaded_state.checkpoint_uuid == "checkpoint-uuid-1"


def test_train_one_phase_state_roundtrip_with_rng_and_early_stopping(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    restart_path = hydra_output_dir / INNER_RESTART_STATE_FILENAME

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
        checkpoint_uuid="checkpoint-uuid-2",
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
    assert loaded_state.checkpoint_uuid == "checkpoint-uuid-2"

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
    outer_restart_path = tmp_path / OUTER_RESTART_STATE_FILENAME
    inner_restart_path = tmp_path / INNER_RESTART_STATE_FILENAME
    scheduler_restart_path = tmp_path / "lr_schedulers.pt"
    scaler_restart_path = tmp_path / "grad_scaler.pt"
    train_all_phases_state = TrainAllPhasesState(
        job_idx=2,
        next_epoch_anchor=6,
    )
    train_one_phase_state = TrainOnePhaseState(
        phase_epoch=4,
        seq_len_increase_in_batches=12,
        rng_state=capture_rng_state(use_cuda=False),
    )
    checkpoint_store = RestartCheckpointStore.from_paths(
        outer_path=outer_restart_path,
        inner_path=inner_restart_path,
        scheduler_path=scheduler_restart_path,
        scaler_path=scaler_restart_path,
    )

    class DummyScaler:
        def state_dict(self):
            return {"scale": 1.0}

    checkpoint_store.save_epoch_checkpoint(
        train_all_phases_state=train_all_phases_state,
        train_one_phase_state=train_one_phase_state,
        lr_schedulers=None,
        scaler=DummyScaler(),
    )

    loaded_outer_state = TrainAllPhasesState().load(outer_restart_path)
    loaded_inner_state = TrainOnePhaseState().load(inner_restart_path)

    assert loaded_outer_state.job_idx == 2
    assert loaded_outer_state.next_epoch_anchor == 6
    assert loaded_inner_state.phase_epoch == 4
    assert loaded_inner_state.seq_len_increase_in_batches == 12
    assert loaded_outer_state.checkpoint_uuid is not None
    assert loaded_inner_state.checkpoint_uuid == loaded_outer_state.checkpoint_uuid
    assert torch.load(scheduler_restart_path, weights_only=False) == {}
    assert torch.load(scaler_restart_path, weights_only=False) == {"scale": 1.0}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_checkpoint_store_rejects_mismatched_uuid_pair_on_load(tmp_path):
    outer_restart_path = tmp_path / OUTER_RESTART_STATE_FILENAME
    inner_restart_path = tmp_path / INNER_RESTART_STATE_FILENAME
    TrainAllPhasesState(checkpoint_uuid="uuid-outer").save(outer_restart_path)
    TrainOnePhaseState(checkpoint_uuid="uuid-inner").save(inner_restart_path)
    checkpoint_store = RestartCheckpointStore.from_paths(
        outer_path=outer_restart_path,
        inner_path=inner_restart_path,
        scheduler_path=tmp_path / "lr_schedulers.pt",
        scaler_path=tmp_path / "grad_scaler.pt",
    )

    with pytest.raises(ValueError, match="UUID mismatch"):
        checkpoint_store.load_state_pair_if_available()


def test_train_one_phase_state_raises_for_missing_registry_attr(tmp_path, monkeypatch):
    restart_path = tmp_path / INNER_RESTART_STATE_FILENAME
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
