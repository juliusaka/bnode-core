import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from bnode_core.nn.nn_utils.early_stopping import EarlyStopping
from bnode_core.ode.trainer_utils.restart_state import (
    INNER_RESTART_STATE_FILENAME,
    OUTER_RESTART_STATE_FILENAME,
    TrainAllPhasesState,
    TrainOnePhaseState,
    capture_rng_state,
    load_train_all_phases_state,
    load_train_one_phase_state,
    save_train_all_phases_state,
    save_train_one_phase_state,
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
    )

    save_train_all_phases_state(restart_path, state)
    loaded_state = load_train_all_phases_state(restart_path)

    assert loaded_state.job_idx == 3
    assert loaded_state.next_epoch_anchor == 17
    assert loaded_state.mlflow_run_id == "run-outer"
    assert loaded_state.metadata() == {
        "job_idx": 3,
        "next_epoch_anchor": 17,
        "mlflow_run_id": "run-outer",
    }


def test_train_one_phase_state_roundtrip(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    restart_path = hydra_output_dir / INNER_RESTART_STATE_FILENAME

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    early_stopping = EarlyStopping(
        patience=9,
        verbose=False,
        threshold=0.25,
        threshold_mode="rel",
        path=str(hydra_output_dir / "best_model.pt"),
        optimizer_path=str(hydra_output_dir / "best_optimizer.pt"),
        trace_func=lambda *_args, **_kwargs: None,
    )

    inputs = torch.randn(6, 3)
    loss = model(inputs).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    state = TrainOnePhaseState.from_runtime(
        phase_epoch=4,
        optimizer=optimizer,
        lr_schedulers={"cosine": scheduler},
        scaler=scaler,
        early_stopping=early_stopping,
        nan_counter=4,
        grad_norm_last_reduced_counter=2,
        stable_epochs=5,
        deterministic_mode_active=False,
        seq_len_increase_in_batches=84,
        use_cuda=False,
    )

    save_train_one_phase_state(restart_path, state)
    loaded_state = load_train_one_phase_state(restart_path)

    assert loaded_state.phase_epoch == 4
    assert loaded_state.nan_counter == 4
    assert loaded_state.grad_norm_last_reduced_counter == 2
    assert loaded_state.stable_epochs == 5
    assert loaded_state.seq_len_increase_in_batches == 84
    assert not hasattr(loaded_state, "model_state")
    assert loaded_state.metadata()["phase_epoch"] == 4


def test_train_one_phase_state_rejects_invalid_payload_type(tmp_path):
    restart_path = tmp_path / INNER_RESTART_STATE_FILENAME
    torch.save({"phase_epoch": 1}, restart_path)

    with pytest.raises(ValueError, match="Invalid train_one_phase_state payload"):
        load_train_one_phase_state(restart_path)


def test_train_one_phase_state_restores_runtime_objects_and_rng(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    restart_path = hydra_output_dir / INNER_RESTART_STATE_FILENAME

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.05)
    source_scheduler = CosineAnnealingLR(source_optimizer, T_max=5)
    source_scaler = torch.amp.GradScaler("cuda", enabled=False)
    source_early_stopping = EarlyStopping(
        patience=9,
        verbose=False,
        threshold=0.25,
        threshold_mode="rel",
        path=str(hydra_output_dir / "best_model.pt"),
        optimizer_path=str(hydra_output_dir / "best_optimizer.pt"),
        trace_func=lambda *_args, **_kwargs: None,
    )

    inputs = torch.randn(6, 3)
    loss = source_model(inputs).pow(2).mean()
    loss.backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    source_scheduler.step()

    source_early_stopping.counter = 3
    source_early_stopping.best_score = 0.4
    source_early_stopping.corresponding_score = 0.6
    source_early_stopping.early_stop = False
    source_early_stopping.score_last_save = 0.4

    state = TrainOnePhaseState(
        phase_epoch=4,
        optimizer_state=source_optimizer.state_dict(),
        scheduler_states={"cosine": source_scheduler.state_dict()},
        scaler_state=source_scaler.state_dict(),
        early_stopping_state=source_early_stopping.state_dict(),
        nan_counter=4,
        grad_norm_last_reduced_counter=2,
        stable_epochs=5,
        rng_state=capture_rng_state(use_cuda=False),
        deterministic_mode_active=True,
        seq_len_increase_in_batches=91,
    )
    save_train_one_phase_state(restart_path, state)

    expected_torch, expected_numpy, expected_python = _sample_rng_triplet()
    _ = _sample_rng_triplet()

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.99)
    restored_scheduler = CosineAnnealingLR(restored_optimizer, T_max=5)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_early_stopping = EarlyStopping(
        patience=1,
        verbose=True,
        threshold=0.0,
        threshold_mode="abs",
        path=Path("placeholder.pt"),
        optimizer_path=Path("placeholder_optimizer.pt"),
        trace_func=lambda *_args, **_kwargs: None,
    )

    loaded_state = load_train_one_phase_state(restart_path)
    loaded_state.restore_runtime_objects(
        optimizer=restored_optimizer,
        lr_schedulers={"cosine": restored_scheduler},
        scaler=restored_scaler,
        early_stopping=restored_early_stopping,
        use_cuda=False,
    )

    assert restored_scheduler.state_dict()["last_epoch"] == source_scheduler.state_dict()["last_epoch"]
    assert restored_early_stopping.patience == 9
    assert restored_early_stopping.threshold == 0.25
    assert restored_early_stopping.threshold_mode == "rel"
    assert restored_early_stopping.counter == 3
    assert restored_early_stopping.path == str(hydra_output_dir / "best_model.pt")
    assert restored_early_stopping.optimizer_path == str(hydra_output_dir / "best_optimizer.pt")
    assert loaded_state.deterministic_mode_active is True
    assert loaded_state.seq_len_increase_in_batches == 91

    restored_torch, restored_numpy, restored_python = _sample_rng_triplet()
    torch.testing.assert_close(restored_torch, expected_torch)
    np.testing.assert_allclose(restored_numpy, expected_numpy)
    assert restored_python == pytest.approx(expected_python)
