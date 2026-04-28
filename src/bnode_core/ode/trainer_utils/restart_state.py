from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import logging
import os
import random
from typing import Any

import hydra
import numpy as np
import torch


RESTART_STATE_SCHEMA_VERSION = 1
RESTART_STATE_FILENAME = "training_restart.pt"


class CheckpointRequestedExit(RuntimeError):
    """Raised when training should stop after persisting a restart checkpoint."""


@dataclass
class TrainingRestartState:
    schema_version: int = RESTART_STATE_SCHEMA_VERSION
    hydra_output_dir: str = ""
    restart_state_path: str = ""
    checkpoint_reason: str = "epoch_end"
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    job_idx: int = 0
    epoch_0: int = 0
    next_epoch: int = 0
    phase_epoch: int = 0
    first_epoch_is_evaluation: bool = True
    current_model_path: str = ""
    current_optimizer_path: str = ""
    best_model_path: str = ""
    best_optimizer_path: str = ""
    training_cfg_state: dict[str, Any] = field(default_factory=dict)
    model_state: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    scheduler_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    scaler_state: dict[str, Any] = field(default_factory=dict)
    early_stopping_state: dict[str, Any] = field(default_factory=dict)
    nan_counter: int = 0
    grad_norm_last_reduced_counter: int = 0
    stable_epochs: int = 0
    flag_out_of_seq_len_increase: bool = True
    epoch_stop: int | None = None
    rng_state: dict[str, Any] = field(default_factory=dict)
    deterministic_mode_active: bool = False
    slurm_job_id: str | None = None

    def validate(self) -> None:
        if self.schema_version != RESTART_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported restart state schema version {self.schema_version}. "
                f"Expected {RESTART_STATE_SCHEMA_VERSION}."
            )
        if not self.hydra_output_dir:
            raise ValueError("restart state missing hydra_output_dir")
        if not self.restart_state_path:
            raise ValueError("restart state missing restart_state_path")
        if not self.current_model_path:
            raise ValueError("restart state missing current_model_path")
        if not self.model_state:
            raise ValueError("restart state missing model_state")
        if self.next_epoch < self.epoch_0:
            raise ValueError(
                f"restart state next_epoch {self.next_epoch} cannot be smaller than epoch_0 {self.epoch_0}"
            )
        if self.phase_epoch != self.next_epoch - self.epoch_0:
            raise ValueError(
                "restart state phase_epoch does not match next_epoch - epoch_0 "
                f"({self.phase_epoch} != {self.next_epoch - self.epoch_0})"
            )
        if not isinstance(self.training_cfg_state, dict):
            raise ValueError("restart state training_cfg_state must be a dict")
        if not isinstance(self.optimizer_state, dict):
            raise ValueError("restart state optimizer_state must be a dict")
        if not isinstance(self.scheduler_states, dict):
            raise ValueError("restart state scheduler_states must be a dict")
        if not isinstance(self.scaler_state, dict):
            raise ValueError("restart state scaler_state must be a dict")
        if not isinstance(self.early_stopping_state, dict):
            raise ValueError("restart state early_stopping_state must be a dict")
        if not isinstance(self.rng_state, dict):
            raise ValueError("restart state rng_state must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hydra_output_dir": self.hydra_output_dir,
            "restart_state_path": self.restart_state_path,
            "checkpoint_reason": self.checkpoint_reason,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "job_idx": self.job_idx,
            "next_epoch": self.next_epoch,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingRestartState":
        payload = dict(payload)
        if "phase_epoch" not in payload and "next_epoch" in payload and "epoch_0" in payload:
            payload["phase_epoch"] = payload["next_epoch"] - payload["epoch_0"]
        state = cls(**payload)
        state.validate()
        return state


def _move_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _move_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(value) for value in obj)
    return obj


def _cfg_state_to_dict(training_cfg_state: Any) -> dict[str, Any]:
    if training_cfg_state is None:
        return {}
    if is_dataclass(training_cfg_state):
        return asdict(training_cfg_state)
    if isinstance(training_cfg_state, dict):
        return dict(training_cfg_state)
    raise TypeError("training_cfg_state must be a dataclass or dict")


def capture_rng_state(use_cuda: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if use_cuda and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], use_cuda: bool) -> None:
    if not state:
        return
    torch.random.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if use_cuda and torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def restart_state_path_from_cfg(cfg: Any, hydra_output_dir: Path | None = None) -> Path:
    explicit = getattr(cfg, "restart_state_path", None)
    if explicit is not None:
        return Path(explicit)
    if hydra_output_dir is None:
        hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    return hydra_output_dir / RESTART_STATE_FILENAME


def load_restart_state(path: Path) -> TrainingRestartState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid restart state payload in {path}")
    payload = dict(payload)
    payload.setdefault("restart_state_path", str(path.resolve()))
    return TrainingRestartState.from_dict(payload)


def save_restart_state(path: Path, state: TrainingRestartState) -> None:
    state.restart_state_path = str(path.resolve())
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.to_dict(), path)
    logging.info("Saved trainer restart state to %s", path)


def load_restart_metadata(path: Path) -> dict[str, Any]:
    return load_restart_state(path).metadata()


def build_training_restart_state(
    *,
    hydra_output_dir: Path,
    restart_state_path: Path,
    model: torch.nn.Module,
    job_idx: int,
    epoch_0: int,
    next_epoch: int,
    first_epoch_is_evaluation: bool,
    current_model_path: Path,
    training_cfg_state: Any,
    optimizer: torch.optim.Optimizer | None = None,
    current_optimizer_path: Path | None = None,
    best_model_path: Path | None = None,
    best_optimizer_path: Path | None = None,
    lr_schedulers: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: Any = None,
    nan_counter: int = 0,
    grad_norm_last_reduced_counter: int = 0,
    stable_epochs: int = 0,
    flag_out_of_seq_len_increase: bool = True,
    epoch_stop: int | None = None,
    checkpoint_reason: str = "epoch_end",
    mlflow_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    deterministic_mode_active: bool = False,
    use_cuda: bool = False,
) -> TrainingRestartState:
    scheduler_states: dict[str, dict[str, Any]] = {}
    if lr_schedulers is not None:
        scheduler_states = {
            name: _move_to_cpu(scheduler.state_dict()) for name, scheduler in lr_schedulers.items()
        }
    return TrainingRestartState(
        hydra_output_dir=str(hydra_output_dir.resolve()),
        restart_state_path=str(restart_state_path.resolve()),
        checkpoint_reason=checkpoint_reason,
        mlflow_run_id=mlflow_run_id,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        job_idx=job_idx,
        epoch_0=epoch_0,
        next_epoch=next_epoch,
        phase_epoch=next_epoch - epoch_0,
        first_epoch_is_evaluation=first_epoch_is_evaluation,
        current_model_path=str(current_model_path.resolve()),
        current_optimizer_path=(
            str(current_optimizer_path.resolve()) if current_optimizer_path is not None else ""
        ),
        best_model_path=str(best_model_path.resolve()) if best_model_path is not None else "",
        best_optimizer_path=(
            str(best_optimizer_path.resolve()) if best_optimizer_path is not None else ""
        ),
        training_cfg_state=_cfg_state_to_dict(training_cfg_state),
        model_state=_move_to_cpu(model.state_dict()),
        optimizer_state=_move_to_cpu(optimizer.state_dict()) if optimizer is not None else {},
        scheduler_states=scheduler_states,
        scaler_state=_move_to_cpu(scaler.state_dict()) if scaler is not None else {},
        early_stopping_state=(
            _move_to_cpu(early_stopping.state_dict()) if early_stopping is not None else {}
        ),
        nan_counter=nan_counter,
        grad_norm_last_reduced_counter=grad_norm_last_reduced_counter,
        stable_epochs=stable_epochs,
        flag_out_of_seq_len_increase=flag_out_of_seq_len_increase,
        epoch_stop=epoch_stop,
        rng_state=_move_to_cpu(capture_rng_state(use_cuda)),
        deterministic_mode_active=deterministic_mode_active,
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
    )


def apply_training_restart_state(
    state: TrainingRestartState,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_schedulers: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: Any = None,
    use_cuda: bool = False,
) -> None:
    state.validate()
    model.load_state_dict(state.model_state)

    if state.optimizer_state:
        if optimizer is None:
            raise ValueError("restart state contains optimizer_state but no optimizer was provided")
        optimizer.load_state_dict(state.optimizer_state)

    scheduler_keys = set(state.scheduler_states.keys())
    provided_scheduler_keys = set(lr_schedulers.keys()) if lr_schedulers is not None else set()
    if scheduler_keys != provided_scheduler_keys:
        raise ValueError(
            "restart state scheduler keys do not match current schedulers: "
            f"saved={sorted(scheduler_keys)}, current={sorted(provided_scheduler_keys)}"
        )
    if lr_schedulers is not None:
        for name, scheduler in lr_schedulers.items():
            scheduler.load_state_dict(state.scheduler_states[name])

    if state.scaler_state:
        if scaler is None:
            raise ValueError("restart state contains scaler_state but no scaler was provided")
        scaler.load_state_dict(state.scaler_state)

    if state.early_stopping_state:
        if early_stopping is None:
            raise ValueError("restart state contains early_stopping_state but no early_stopping was provided")
        early_stopping.load_state_dict(state.early_stopping_state)

    restore_rng_state(state.rng_state, use_cuda=use_cuda)


