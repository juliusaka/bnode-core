import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from bnode_core.nn.nn_utils.early_stopping import EarlyStopping
from bnode_core.ode.trainer_utils.restart_state import (
    INNER_RESTART_STATE_FILENAME,
    OUTER_RESTART_STATE_FILENAME,
    InnerTrainingStateCheckpoint,
    LiveTrainingState,
    OuterTrainingStateCheckpoint,
    RESTART_STATE_FILENAME,
    RESTART_STATE_SCHEMA_VERSION,
    TrainingRestartState,
    apply_training_restart_state,
    capture_rng_state,
    load_inner_restart_state,
    load_outer_restart_state,
    load_restart_state,
    save_inner_restart_state,
    save_outer_restart_state,
    save_restart_state,
)


def _sample_rng_triplet():
    return (
        torch.rand(4),
        np.random.rand(4),
        [random.random() for _ in range(4)],
    )


def test_outer_restart_state_roundtrip(tmp_path):
    restart_path = tmp_path / OUTER_RESTART_STATE_FILENAME
    state = OuterTrainingStateCheckpoint(
        hydra_output_dir=str(tmp_path.resolve()),
        restart_state_path=str(restart_path.resolve()),
        checkpoint_reason="epoch_end",
        mlflow_run_id="run-outer",
        mlflow_tracking_uri="file:///mlruns",
        mlflow_experiment_name="restart-tests",
        job_idx=3,
        next_epoch_anchor=17,
        slurm_job_id="1234",
    )

    save_outer_restart_state(restart_path, state)
    loaded_state = load_outer_restart_state(restart_path)

    assert loaded_state.job_idx == 3
    assert loaded_state.next_epoch_anchor == 17
    assert loaded_state.metadata()["mlflow_run_id"] == "run-outer"
    assert loaded_state.restart_state_path == str(restart_path.resolve())


def test_inner_restart_state_roundtrip(tmp_path):
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

    state = InnerTrainingStateCheckpoint(
        hydra_output_dir=str(hydra_output_dir.resolve()),
        restart_state_path=str(restart_path.resolve()),
        checkpoint_reason="epoch_end",
        job_idx=2,
        phase_epoch=4,
        first_epoch_is_evaluation=False,
        current_model_path=str((hydra_output_dir / "model.pt").resolve()),
        current_optimizer_path=str((hydra_output_dir / "optimizer.pt").resolve()),
        best_model_path=str((hydra_output_dir / "model_phase_2.pt").resolve()),
        best_optimizer_path=str((hydra_output_dir / "optimizer_phase_2.pt").resolve()),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_states={"cosine": scheduler.state_dict()},
        scaler_state=scaler.state_dict(),
        early_stopping_state=early_stopping.state_dict(),
        nan_counter=4,
        grad_norm_last_reduced_counter=2,
        stable_epochs=5,
        flag_out_of_seq_len_increase=False,
        epoch_stop=21,
        rng_state=capture_rng_state(use_cuda=False),
        deterministic_mode_active=False,
        slurm_job_id=None,
    )

    save_inner_restart_state(restart_path, state)
    loaded_state = load_inner_restart_state(restart_path)

    assert loaded_state.job_idx == 2
    assert loaded_state.phase_epoch == 4
    assert not hasattr(loaded_state, "training_cfg_state")
    assert loaded_state.metadata()["phase_epoch"] == 4
    assert loaded_state.restart_state_path == str(restart_path.resolve())


def test_restart_state_roundtrip_restores_runtime_state(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    restart_path = hydra_output_dir / RESTART_STATE_FILENAME

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

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

    early_stopping.counter = 3
    early_stopping.best_score = 0.4
    early_stopping.corresponding_score = 0.6
    early_stopping.early_stop = False
    early_stopping.score_last_save = 0.4

    state = TrainingRestartState(
        hydra_output_dir=str(hydra_output_dir.resolve()),
        restart_state_path=str(restart_path.resolve()),
        checkpoint_reason="epoch_end",
        mlflow_run_id="run-123",
        mlflow_tracking_uri="file:///mlruns",
        mlflow_experiment_name="restart-tests",
        job_idx=2,
        epoch_0=10,
        next_epoch=14,
        phase_epoch=4,
        first_epoch_is_evaluation=False,
        current_model_path=str((hydra_output_dir / "model.pt").resolve()),
        current_optimizer_path=str((hydra_output_dir / "optimizer.pt").resolve()),
        best_model_path=str((hydra_output_dir / "model_phase_2.pt").resolve()),
        best_optimizer_path=str((hydra_output_dir / "optimizer_phase_2.pt").resolve()),
        training_cfg_state={"clip_grad_norm": 1.5, "seq_len_train": 17},
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_states={"cosine": scheduler.state_dict()},
        scaler_state=scaler.state_dict(),
        early_stopping_state=early_stopping.state_dict(),
        nan_counter=4,
        grad_norm_last_reduced_counter=2,
        stable_epochs=5,
        flag_out_of_seq_len_increase=False,
        epoch_stop=21,
        rng_state=capture_rng_state(use_cuda=False),
        deterministic_mode_active=False,
        slurm_job_id=None,
    )
    save_restart_state(restart_path, state)

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
        path="placeholder.pt",
        optimizer_path="placeholder_optimizer.pt",
        trace_func=lambda *_args, **_kwargs: None,
    )

    loaded_state = load_restart_state(restart_path)
    apply_training_restart_state(
        loaded_state,
        model=restored_model,
        optimizer=restored_optimizer,
        lr_schedulers={"cosine": restored_scheduler},
        scaler=restored_scaler,
        early_stopping=restored_early_stopping,
        use_cuda=False,
    )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[key], value)

    assert loaded_state.schema_version == RESTART_STATE_SCHEMA_VERSION
    assert loaded_state.phase_epoch == 4
    assert loaded_state.training_cfg_state["clip_grad_norm"] == 1.5
    assert loaded_state.mlflow_run_id == "run-123"
    assert loaded_state.metadata()["mlflow_tracking_uri"] == "file:///mlruns"
    assert loaded_state.metadata()["mlflow_experiment_name"] == "restart-tests"
    assert loaded_state.metadata()["checkpoint_reason"] == "epoch_end"
    assert loaded_state.restart_state_path == str(restart_path.resolve())
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert restored_early_stopping.patience == 9
    assert restored_early_stopping.threshold == 0.25
    assert restored_early_stopping.threshold_mode == "rel"
    assert restored_early_stopping.counter == 3
    assert restored_early_stopping.path == str(hydra_output_dir / "best_model.pt")
    assert restored_early_stopping.optimizer_path == str(hydra_output_dir / "best_optimizer.pt")

    restored_torch, restored_numpy, restored_python = _sample_rng_triplet()
    torch.testing.assert_close(restored_torch, expected_torch)
    np.testing.assert_allclose(restored_numpy, expected_numpy)
    assert restored_python == pytest.approx(expected_python)


def test_restart_state_rejects_schema_mismatch(tmp_path):
    restart_path = tmp_path / RESTART_STATE_FILENAME
    torch.save(
        {
            "schema_version": RESTART_STATE_SCHEMA_VERSION + 1,
            "hydra_output_dir": str(tmp_path),
            "restart_state_path": str(restart_path),
            "job_idx": 0,
            "epoch_0": 0,
            "next_epoch": 1,
            "phase_epoch": 1,
            "first_epoch_is_evaluation": False,
            "current_model_path": str(tmp_path / "model.pt"),
            "training_cfg_state": {},
            "model_state": {"weight": torch.ones(1)},
            "optimizer_state": {},
            "scheduler_states": {},
            "scaler_state": {},
            "early_stopping_state": {},
            "rng_state": {
                "torch_cpu": torch.random.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        restart_path,
    )

    with pytest.raises(ValueError, match="Unsupported restart state schema version"):
        load_restart_state(restart_path)


def test_live_training_state_bind_runtime_objects_restores_runtime_state(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    restart_path = hydra_output_dir / INNER_RESTART_STATE_FILENAME

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

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

    early_stopping.counter = 3
    early_stopping.best_score = 0.4
    early_stopping.corresponding_score = 0.6
    early_stopping.early_stop = False
    early_stopping.score_last_save = 0.4

    restart_state = InnerTrainingStateCheckpoint(
        hydra_output_dir=str(hydra_output_dir.resolve()),
        restart_state_path=str(restart_path.resolve()),
        checkpoint_reason="epoch_end",
        job_idx=2,
        phase_epoch=4,
        first_epoch_is_evaluation=False,
        current_model_path=str((hydra_output_dir / "model.pt").resolve()),
        current_optimizer_path=str((hydra_output_dir / "optimizer.pt").resolve()),
        best_model_path=str((hydra_output_dir / "model_phase_2.pt").resolve()),
        best_optimizer_path=str((hydra_output_dir / "optimizer_phase_2.pt").resolve()),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_states={"cosine": scheduler.state_dict()},
        scaler_state=scaler.state_dict(),
        early_stopping_state=early_stopping.state_dict(),
        nan_counter=4,
        grad_norm_last_reduced_counter=2,
        stable_epochs=5,
        flag_out_of_seq_len_increase=False,
        epoch_stop=21,
        rng_state=capture_rng_state(use_cuda=False),
        deterministic_mode_active=False,
        slurm_job_id=None,
    )

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.99)
    restored_scheduler = CosineAnnealingLR(restored_optimizer, T_max=5)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_early_stopping = EarlyStopping(
        patience=1,
        verbose=True,
        threshold=0.0,
        threshold_mode="abs",
        path="placeholder.pt",
        optimizer_path="placeholder_optimizer.pt",
        trace_func=lambda *_args, **_kwargs: None,
    )

    live_state = LiveTrainingState.create_uninitialized(
        cfg=SimpleNamespace(mlflow_experiment_name="restart-tests", use_cuda=False),
        train_cfg=SimpleNamespace(),
        job_idx=2,
        pre_train=False,
        device=torch.device("cpu"),
        phase_epoch_0=10,
        max_epochs=11,
        batches_per_epoch=7,
        epochs_for_seq_len_increase=0,
        path_best_model=Path(restart_state.best_model_path),
        path_optimizer_best_model=Path(restart_state.best_optimizer_path),
        path_current_model=Path(restart_state.current_model_path),
        path_current_optimizer=Path(restart_state.current_optimizer_path),
        hydra_output_dir=hydra_output_dir,
        restart_manager_path=restart_path,
        restart_state=restart_state,
        next_epoch_anchor=14,
    )

    assert live_state.model is None
    assert live_state.optimizer is None
    assert live_state.batches_per_epoch == 7
    assert live_state.phase_state.epoch_start == 14

    live_state.bind_runtime_objects(
        model=restored_model,
        optimizer=restored_optimizer,
        lr_schedulers={"cosine": restored_scheduler},
        scaler=restored_scaler,
        early_stopping=restored_early_stopping,
        restart_state=restart_state,
    )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(live_state.model.state_dict()[key], value)

    assert live_state.phase_state.epoch_start == 14
    assert live_state.phase_state.epoch_stop == restart_state.epoch_stop
    assert live_state.phase_state.nan_counter == restart_state.nan_counter
    assert live_state.phase_state.stable_epochs == restart_state.stable_epochs
    assert live_state.phase_state.flag_out_of_seq_len_increase is False
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert restored_early_stopping.counter == 3


def test_live_training_state_create_wrapper_restores_runtime_state(tmp_path):
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

    restart_state = InnerTrainingStateCheckpoint(
        hydra_output_dir=str(hydra_output_dir.resolve()),
        restart_state_path=str(restart_path.resolve()),
        checkpoint_reason="epoch_end",
        job_idx=1,
        phase_epoch=2,
        first_epoch_is_evaluation=False,
        current_model_path=str((hydra_output_dir / "model.pt").resolve()),
        current_optimizer_path=str((hydra_output_dir / "optimizer.pt").resolve()),
        best_model_path=str((hydra_output_dir / "model_phase_1.pt").resolve()),
        best_optimizer_path=str((hydra_output_dir / "optimizer_phase_1.pt").resolve()),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_states={"cosine": scheduler.state_dict()},
        scaler_state=scaler.state_dict(),
        early_stopping_state=early_stopping.state_dict(),
        epoch_stop=9,
        rng_state=capture_rng_state(use_cuda=False),
    )

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.99)
    restored_scheduler = CosineAnnealingLR(restored_optimizer, T_max=5)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_early_stopping = EarlyStopping(
        patience=1,
        verbose=True,
        threshold=0.0,
        threshold_mode="abs",
        path="placeholder.pt",
        optimizer_path="placeholder_optimizer.pt",
        trace_func=lambda *_args, **_kwargs: None,
    )

    live_state = LiveTrainingState.create(
        cfg=SimpleNamespace(mlflow_experiment_name="restart-tests", use_cuda=False),
        model=restored_model,
        optimizer=restored_optimizer,
        lr_schedulers={"cosine": restored_scheduler},
        scaler=restored_scaler,
        early_stopping=restored_early_stopping,
        train_cfg=SimpleNamespace(),
        job_idx=1,
        pre_train=False,
        device=torch.device("cpu"),
        phase_epoch_0=3,
        max_epochs=6,
        epochs_for_seq_len_increase=0,
        path_best_model=Path(restart_state.best_model_path),
        path_optimizer_best_model=Path(restart_state.best_optimizer_path),
        path_current_model=Path(restart_state.current_model_path),
        path_current_optimizer=Path(restart_state.current_optimizer_path),
        hydra_output_dir=hydra_output_dir,
        restart_manager_path=restart_path,
        restart_state=restart_state,
        batches_per_epoch=4,
        next_epoch_anchor=5,
    )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(live_state.model.state_dict()[key], value)
    assert live_state.phase_state.epoch_start == 5
    assert live_state.phase_state.phase_epoch_0 == 3
