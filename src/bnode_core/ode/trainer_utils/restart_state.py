from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


OUTER_RESTART_STATE_FILENAME = "training_outer_restart.pt"
INNER_RESTART_STATE_FILENAME = "training_inner_restart.pt"


class CheckpointRequestedExit(RuntimeError):
    """Raised when training should stop after persisting a restart checkpoint."""


class TrainAllPhasesState(torch.nn.Module):
    """Minimal persisted state for ``train_all_phases()``."""

    def __init__(
        self,
        *,
        job_idx: int = 0,
        next_epoch_anchor: int = 0,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.job_idx = job_idx
        self.next_epoch_anchor = next_epoch_anchor
        self.mlflow_run_id = mlflow_run_id

    def validate(self) -> None:
        if self.job_idx < 0:
            raise ValueError("train_all_phases_state.job_idx must be non-negative")
        if self.next_epoch_anchor < 0:
            raise ValueError("train_all_phases_state.next_epoch_anchor must be non-negative")

    def metadata(self) -> dict[str, Any]:
        return {
            "job_idx": self.job_idx,
            "next_epoch_anchor": self.next_epoch_anchor,
            "mlflow_run_id": self.mlflow_run_id,
        }


class TrainOnePhaseState(torch.nn.Module):
    """Minimal persisted state for ``train_one_phase()``."""

    def __init__(
        self,
        *,
        phase_epoch: int = 0,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_states: dict[str, dict[str, Any]] | None = None,
        scaler_state: dict[str, Any] | None = None,
        early_stopping_state: dict[str, Any] | None = None,
        nan_counter: int = 0,
        grad_norm_last_reduced_counter: int = 0,
        stable_epochs: int = 0,
        rng_state: dict[str, Any] | None = None,
        deterministic_mode_active: bool = False,
        seq_len_increase_in_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.phase_epoch = phase_epoch
        self.optimizer_state = optimizer_state or {}
        self.scheduler_states = scheduler_states or {}
        self.scaler_state = scaler_state or {}
        self.early_stopping_state = early_stopping_state or {}
        self.nan_counter = nan_counter
        self.grad_norm_last_reduced_counter = grad_norm_last_reduced_counter
        self.stable_epochs = stable_epochs
        self.rng_state = rng_state or {}
        self.deterministic_mode_active = deterministic_mode_active
        self.seq_len_increase_in_batches = seq_len_increase_in_batches

    def validate(self) -> None:
        if self.phase_epoch < 0:
            raise ValueError("train_one_phase_state.phase_epoch must be non-negative")
        if self.nan_counter < 0:
            raise ValueError("train_one_phase_state.nan_counter must be non-negative")
        if self.grad_norm_last_reduced_counter < 0:
            raise ValueError(
                "train_one_phase_state.grad_norm_last_reduced_counter must be non-negative"
            )
        if self.stable_epochs < 0:
            raise ValueError("train_one_phase_state.stable_epochs must be non-negative")
        if not isinstance(self.optimizer_state, dict):
            raise ValueError("train_one_phase_state.optimizer_state must be a dict")
        if not isinstance(self.scheduler_states, dict):
            raise ValueError("train_one_phase_state.scheduler_states must be a dict")
        if not isinstance(self.scaler_state, dict):
            raise ValueError("train_one_phase_state.scaler_state must be a dict")
        if not isinstance(self.early_stopping_state, dict):
            raise ValueError("train_one_phase_state.early_stopping_state must be a dict")
        if not isinstance(self.rng_state, dict):
            raise ValueError("train_one_phase_state.rng_state must be a dict")
        if (
            self.seq_len_increase_in_batches is not None
            and self.seq_len_increase_in_batches < 0
        ):
            raise ValueError(
                "train_one_phase_state.seq_len_increase_in_batches must be non-negative"
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "phase_epoch": self.phase_epoch,
            "nan_counter": self.nan_counter,
            "grad_norm_last_reduced_counter": self.grad_norm_last_reduced_counter,
            "stable_epochs": self.stable_epochs,
            "deterministic_mode_active": self.deterministic_mode_active,
            "seq_len_increase_in_batches": self.seq_len_increase_in_batches,
        }

    @classmethod
    def from_runtime(
        cls,
        *,
        phase_epoch: int,
        optimizer: torch.optim.Optimizer | None,
        lr_schedulers: dict[str, Any] | None,
        scaler: torch.amp.GradScaler | None,
        early_stopping: Any,
        nan_counter: int,
        grad_norm_last_reduced_counter: int,
        stable_epochs: int,
        deterministic_mode_active: bool,
        seq_len_increase_in_batches: int | None,
        use_cuda: bool,
    ) -> "TrainOnePhaseState":
        scheduler_states: dict[str, dict[str, Any]] = {}
        if lr_schedulers is not None:
            scheduler_states = {
                name: _move_to_cpu(scheduler.state_dict())
                for name, scheduler in lr_schedulers.items()
            }
        return cls(
            phase_epoch=phase_epoch,
            optimizer_state=(
                _move_to_cpu(optimizer.state_dict()) if optimizer is not None else {}
            ),
            scheduler_states=scheduler_states,
            scaler_state=_move_to_cpu(scaler.state_dict()) if scaler is not None else {},
            early_stopping_state=(
                _move_to_cpu(early_stopping.state_dict())
                if early_stopping is not None
                else {}
            ),
            nan_counter=nan_counter,
            grad_norm_last_reduced_counter=grad_norm_last_reduced_counter,
            stable_epochs=stable_epochs,
            rng_state=_move_to_cpu(capture_rng_state(use_cuda)),
            deterministic_mode_active=deterministic_mode_active,
            seq_len_increase_in_batches=seq_len_increase_in_batches,
        )

    def restore_runtime_objects(
        self,
        *,
        optimizer: torch.optim.Optimizer | None,
        lr_schedulers: dict[str, Any] | None,
        scaler: torch.amp.GradScaler | None,
        early_stopping: Any,
        use_cuda: bool,
    ) -> None:
        self.validate()

        if self.optimizer_state:
            if optimizer is None:
                raise ValueError(
                    "train_one_phase_state contains optimizer_state but no optimizer was provided"
                )
            optimizer.load_state_dict(self.optimizer_state)

        saved_scheduler_keys = set(self.scheduler_states.keys())
        current_scheduler_keys = set(lr_schedulers.keys()) if lr_schedulers is not None else set()
        if saved_scheduler_keys != current_scheduler_keys:
            raise ValueError(
                "train_one_phase_state scheduler keys do not match current schedulers: "
                f"saved={sorted(saved_scheduler_keys)}, current={sorted(current_scheduler_keys)}"
            )
        if lr_schedulers is not None:
            for name, scheduler in lr_schedulers.items():
                scheduler.load_state_dict(self.scheduler_states[name])

        if self.scaler_state:
            if scaler is None:
                raise ValueError(
                    "train_one_phase_state contains scaler_state but no scaler was provided"
                )
            scaler.load_state_dict(self.scaler_state)

        if self.early_stopping_state:
            if early_stopping is None:
                raise ValueError(
                    "train_one_phase_state contains early_stopping_state but no early_stopping was provided"
                )
            early_stopping.load_state_dict(self.early_stopping_state)

        restore_rng_state(self.rng_state, use_cuda=use_cuda)


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


def load_train_all_phases_state(path: Path) -> TrainAllPhasesState:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, TrainAllPhasesState):
        raise ValueError(f"Invalid train_all_phases_state payload in {path}")
    state.validate()
    return state


def load_train_all_phases_state_metadata(path: Path) -> dict[str, Any]:
    metadata = load_train_all_phases_state(path).metadata()
    metadata["restart_state_path"] = str(path.resolve())
    metadata["hydra_output_dir"] = str(path.resolve().parent.resolve())
    return metadata


def save_train_all_phases_state(path: Path, state: TrainAllPhasesState) -> None:
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logging.info("Saved train_all_phases_state to %s", path)


def load_train_one_phase_state(path: Path) -> TrainOnePhaseState:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, TrainOnePhaseState):
        raise ValueError(f"Invalid train_one_phase_state payload in {path}")
    state.validate()
    return state


def save_train_one_phase_state(path: Path, state: TrainOnePhaseState) -> None:
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logging.info("Saved train_one_phase_state to %s", path)
