from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import logging
import os
import random
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch

if TYPE_CHECKING:
    from torch.amp import GradScaler


RESTART_STATE_SCHEMA_VERSION = 1
RESTART_STATE_FILENAME = "training_restart.pt"
OUTER_RESTART_STATE_FILENAME = "training_outer_restart.pt"
INNER_RESTART_STATE_FILENAME = "training_inner_restart.pt"


class CheckpointRequestedExit(RuntimeError):
    """Raised when training should stop after persisting a restart checkpoint."""


@dataclass
class OuterTrainingStateCheckpoint:
    """Checkpoint schema for outer orchestration state in ``train_all_phases()``.

    This is the target outer checkpoint shape for the restart redesign.  It keeps
    only orchestration progress and resume metadata, not model/runtime state.
    """

    schema_version: int = RESTART_STATE_SCHEMA_VERSION
    hydra_output_dir: str = ""
    restart_state_path: str = ""
    checkpoint_reason: str = "epoch_end"
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    job_idx: int = 0
    next_epoch_anchor: int = 0
    slurm_job_id: str | None = None

    def validate(self) -> None:
        if self.schema_version != RESTART_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported restart state schema version {self.schema_version}. "
                f"Expected {RESTART_STATE_SCHEMA_VERSION}."
            )
        if not self.hydra_output_dir:
            raise ValueError("outer restart state missing hydra_output_dir")
        if not self.restart_state_path:
            raise ValueError("outer restart state missing restart_state_path")
        if self.job_idx < 0:
            raise ValueError("outer restart state job_idx must be non-negative")
        if self.next_epoch_anchor < 0:
            raise ValueError("outer restart state next_epoch_anchor must be non-negative")

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
            "next_epoch_anchor": self.next_epoch_anchor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OuterTrainingStateCheckpoint":
        state = cls(**dict(payload))
        state.validate()
        return state


@dataclass
class InnerTrainingStateCheckpoint:
    """Checkpoint schema for phase-local runtime state in ``train_one_phase()``.

    This is the target inner checkpoint shape for the restart redesign.  It keeps
    model/runtime state dicts plus the minimum phase-control values needed for
    resume, but does not persist config or dataloaders.
    """

    schema_version: int = RESTART_STATE_SCHEMA_VERSION
    hydra_output_dir: str = ""
    restart_state_path: str = ""
    checkpoint_reason: str = "epoch_end"
    job_idx: int = 0
    phase_epoch: int = 0
    first_epoch_is_evaluation: bool = True
    current_model_path: str = ""
    current_optimizer_path: str = ""
    best_model_path: str = ""
    best_optimizer_path: str = ""
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
            raise ValueError("inner restart state missing hydra_output_dir")
        if not self.restart_state_path:
            raise ValueError("inner restart state missing restart_state_path")
        if not self.current_model_path:
            raise ValueError("inner restart state missing current_model_path")
        if not self.model_state:
            raise ValueError("inner restart state missing model_state")
        if self.job_idx < 0:
            raise ValueError("inner restart state job_idx must be non-negative")
        if self.phase_epoch < 0:
            raise ValueError("inner restart state phase_epoch must be non-negative")
        if not isinstance(self.optimizer_state, dict):
            raise ValueError("inner restart state optimizer_state must be a dict")
        if not isinstance(self.scheduler_states, dict):
            raise ValueError("inner restart state scheduler_states must be a dict")
        if not isinstance(self.scaler_state, dict):
            raise ValueError("inner restart state scaler_state must be a dict")
        if not isinstance(self.early_stopping_state, dict):
            raise ValueError("inner restart state early_stopping_state must be a dict")
        if not isinstance(self.rng_state, dict):
            raise ValueError("inner restart state rng_state must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hydra_output_dir": self.hydra_output_dir,
            "restart_state_path": self.restart_state_path,
            "checkpoint_reason": self.checkpoint_reason,
            "job_idx": self.job_idx,
            "phase_epoch": self.phase_epoch,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InnerTrainingStateCheckpoint":
        state = cls(**dict(payload))
        state.validate()
        return state


class OuterTrainingState:
    """Live orchestration state for ``train_all_phases()``.

    Keeps only long-lived outer-loop values and the optional restart metadata
    needed to resume exactly one main-training job.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        job_list: list[dict[str, Any]],
        outer_restart_state_path: Path,
        inner_restart_state_path: Path,
        outer_restart_state: OuterTrainingStateCheckpoint | None = None,
        inner_restart_state: InnerTrainingStateCheckpoint | None = None,
    ) -> None:
        self.cfg = cfg
        self.job_list = job_list
        self.outer_restart_state_path = outer_restart_state_path
        self.inner_restart_state_path = inner_restart_state_path
        self.outer_restart_state = outer_restart_state
        self.inner_restart_state = inner_restart_state
        self.job_start_idx = outer_restart_state.job_idx if outer_restart_state is not None else 0
        self.next_epoch_anchor = (
            outer_restart_state.next_epoch_anchor if outer_restart_state is not None else 0
        )
        self._validate_restart_target()

    def _validate_restart_target(self) -> None:
        if self.outer_restart_state is None and self.inner_restart_state is None:
            return
        if self.outer_restart_state is None or self.inner_restart_state is None:
            raise ValueError(
                "Trainer restart requires both outer and inner restart checkpoints; found only one."
            )
        if self.outer_restart_state.job_idx != self.inner_restart_state.job_idx:
            raise ValueError(
                "Outer and inner restart checkpoints disagree on the resumed job index: "
                f"{self.outer_restart_state.job_idx} != {self.inner_restart_state.job_idx}"
            )
        if self.job_start_idx >= len(self.job_list):
            raise ValueError(
                f"Restart state refers to job index {self.job_start_idx}, but only {len(self.job_list)} jobs exist."
            )
        target_job = self.job_list[self.job_start_idx]
        if target_job["test"] or target_job["pre_train"]:
            raise ValueError("Trainer restart currently supports main-training phases only.")

    def restart_state_for_job(self, job_idx: int) -> InnerTrainingStateCheckpoint | None:
        if self.inner_restart_state is None or job_idx != self.job_start_idx:
            return None
        return self.inner_restart_state

    def consume_restart_state(self) -> None:
        self.outer_restart_state = None
        self.inner_restart_state = None

    def advance_to_next_epoch_anchor(self, next_epoch_anchor: int) -> None:
        self.next_epoch_anchor = next_epoch_anchor

    def save_checkpoint(self, *, job_idx: int, next_epoch_anchor: int) -> None:
        checkpoint = OuterTrainingStateCheckpoint(
            hydra_output_dir=str(Path(self.outer_restart_state_path).resolve().parent.resolve()),
            restart_state_path=str(self.outer_restart_state_path.resolve()),
            checkpoint_reason="epoch_end",
            mlflow_run_id=(
                mlflow.active_run().info.run_id if mlflow.active_run() is not None else None
            ),
            mlflow_tracking_uri=mlflow.get_tracking_uri(),
            mlflow_experiment_name=self.cfg.mlflow_experiment_name,
            job_idx=job_idx,
            next_epoch_anchor=next_epoch_anchor,
            slurm_job_id=os.getenv("SLURM_JOB_ID"),
        )
        save_outer_restart_state(self.outer_restart_state_path, checkpoint)
        self.outer_restart_state = checkpoint
        self.next_epoch_anchor = next_epoch_anchor


@dataclass
class TrainingPhaseState:
    """Mutable counters and flags for a single training phase.

    Serves as the single source of truth for both the live training loop and
    the restart checkpoint, replacing scattered individual variables and a
    lengthy parameter list in ``_save_training_restart_state``.

    See ``docs/bnode_core/ode/restart_training.md`` for how this differs from
    ``LiveTrainingState`` and ``TrainingRestartState``.
    """

    phase_epoch_0: int
    """Global epoch index at which this phase started (the resume anchor)."""
    epoch_start: int
    """First epoch to execute in this run (equals phase_epoch_0 on a fresh
    start, or restart_state.next_epoch when resuming)."""
    epoch_stop: int
    """Exclusive upper bound for the epoch loop; may grow when seq-len
    increase ends earlier than the configured maximum."""
    first_epoch_is_evaluation: bool
    """True while the very first epoch of the phase is still pending
    (that epoch is eval-only, not a training epoch)."""
    nan_counter: int = 0
    """Cumulative NaN-loss events; training aborts when this exceeds 50."""
    grad_norm_last_reduced_counter: int = 0
    """Consecutive optimizer reloads after NaN; triggers grad-norm reduction
    after 2 consecutive reloads."""
    stable_epochs: int = 0
    """Epochs where validation loss < 2× training loss; used to exit the
    seq-len increase warm-up early."""
    flag_out_of_seq_len_increase: bool = True
    """True once the sequence-length warm-up phase has ended."""
    deterministic_mode_active: bool = False
    """True after the deterministic mask has been applied to the model."""

    @classmethod
    def fresh(
        cls,
        epoch_0: int,
        epoch_stop: int,
        flag_out_of_seq_len_increase: bool,
    ) -> "TrainingPhaseState":
        """Create a phase state for a new (non-resumed) training phase."""
        return cls(
            phase_epoch_0=epoch_0,
            epoch_start=epoch_0,
            epoch_stop=epoch_stop,
            first_epoch_is_evaluation=True,
            flag_out_of_seq_len_increase=flag_out_of_seq_len_increase,
        )

    @classmethod
    def from_restart(
        cls,
        restart_state: "TrainingRestartState",
        default_epoch_stop: int,
    ) -> "TrainingPhaseState":
        """Restore a phase state from a persisted restart checkpoint."""
        return cls(
            phase_epoch_0=restart_state.epoch_0,
            epoch_start=restart_state.next_epoch,
            epoch_stop=(
                restart_state.epoch_stop
                if restart_state.epoch_stop is not None
                else default_epoch_stop
            ),
            first_epoch_is_evaluation=restart_state.first_epoch_is_evaluation,
            nan_counter=restart_state.nan_counter,
            grad_norm_last_reduced_counter=restart_state.grad_norm_last_reduced_counter,
            stable_epochs=restart_state.stable_epochs,
            flag_out_of_seq_len_increase=restart_state.flag_out_of_seq_len_increase,
            deterministic_mode_active=restart_state.deterministic_mode_active,
        )

    @classmethod
    def from_inner_checkpoint(
        cls,
        restart_state: InnerTrainingStateCheckpoint,
        *,
        next_epoch_anchor: int,
        default_epoch_stop: int,
    ) -> "TrainingPhaseState":
        phase_epoch_0 = next_epoch_anchor - restart_state.phase_epoch
        return cls(
            phase_epoch_0=phase_epoch_0,
            epoch_start=next_epoch_anchor,
            epoch_stop=(
                restart_state.epoch_stop
                if restart_state.epoch_stop is not None
                else default_epoch_stop
            ),
            first_epoch_is_evaluation=restart_state.first_epoch_is_evaluation,
            nan_counter=restart_state.nan_counter,
            grad_norm_last_reduced_counter=restart_state.grad_norm_last_reduced_counter,
            stable_epochs=restart_state.stable_epochs,
            flag_out_of_seq_len_increase=restart_state.flag_out_of_seq_len_increase,
            deterministic_mode_active=restart_state.deterministic_mode_active,
        )


class LiveTrainingState:
    """All live training objects bundled into one place.

    Passed to the inner training functions (``train_one_epoch`` etc.) and used
    as the single source of truth for ``save_checkpoint``.  Unlike
    ``TrainingRestartState``, this object is **not serialized** — it holds real
    PyTorch objects whose ``.state_dict()`` is captured only when checkpointing.

    See ``docs/bnode_core/ode/restart_training.md`` for the full state model.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        model: torch.nn.Module | None,
        optimizer: torch.optim.Optimizer | None,
        lr_schedulers: dict | None,
        scaler: GradScaler | None,
        early_stopping: Any | None,
        train_cfg: Any,
        job_idx: int,
        pre_train: bool,
        device: torch.device,
        phase_state: TrainingPhaseState,
        path_best_model: Path,
        path_optimizer_best_model: Path,
        path_current_model: Path,
        path_current_optimizer: Path,
        hydra_output_dir: Path,
        restart_manager_path: Path | None,
        max_epochs: int,
        batches_per_epoch: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_schedulers = lr_schedulers
        self.scaler = scaler
        self.early_stopping = early_stopping
        self.train_cfg = train_cfg
        self.job_idx = job_idx
        self.pre_train = pre_train
        self.device = device
        self.phase_state = phase_state
        self.path_best_model = path_best_model
        self.path_optimizer_best_model = path_optimizer_best_model
        self.path_current_model = path_current_model
        self.path_current_optimizer = path_current_optimizer
        self.hydra_output_dir = hydra_output_dir
        self.restart_manager_path = restart_manager_path
        self.max_epochs = max_epochs
        self.batches_per_epoch = batches_per_epoch

    @classmethod
    def create_uninitialized(
        cls,
        *,
        cfg: Any,
        train_cfg: Any,
        job_idx: int,
        pre_train: bool,
        device: torch.device,
        phase_epoch_0: int,
        max_epochs: int,
        batches_per_epoch: int | None,
        epochs_for_seq_len_increase: int,
        path_best_model: Path,
        path_optimizer_best_model: Path,
        path_current_model: Path,
        path_current_optimizer: Path,
        hydra_output_dir: Path,
        restart_manager_path: Path | None,
        restart_state: InnerTrainingStateCheckpoint | None = None,
        next_epoch_anchor: int | None = None,
    ) -> "LiveTrainingState":
        default_epoch_stop = phase_epoch_0 + max_epochs
        if restart_state is not None:
            if next_epoch_anchor is None:
                raise ValueError("next_epoch_anchor is required when resuming from an inner checkpoint")
            phase_state = TrainingPhaseState.from_inner_checkpoint(
                restart_state,
                next_epoch_anchor=next_epoch_anchor,
                default_epoch_stop=default_epoch_stop,
            )
        else:
            flag_out_of_seq_len_increase = True if pre_train else epochs_for_seq_len_increase == 0
            phase_state = TrainingPhaseState.fresh(
                epoch_0=phase_epoch_0,
                epoch_stop=default_epoch_stop,
                flag_out_of_seq_len_increase=flag_out_of_seq_len_increase,
            )

        live_state = cls(
            cfg=cfg,
            model=None,
            optimizer=None,
            lr_schedulers=None,
            scaler=None,
            early_stopping=None,
            train_cfg=train_cfg,
            job_idx=job_idx,
            pre_train=pre_train,
            device=device,
            phase_state=phase_state,
            path_best_model=path_best_model,
            path_optimizer_best_model=path_optimizer_best_model,
            path_current_model=path_current_model,
            path_current_optimizer=path_current_optimizer,
            hydra_output_dir=hydra_output_dir,
            restart_manager_path=restart_manager_path,
            max_epochs=max_epochs,
            batches_per_epoch=batches_per_epoch,
        )
        return live_state

    def bind_runtime_objects(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        lr_schedulers: dict | None,
        scaler: GradScaler | None,
        early_stopping: Any | None,
        restart_state: InnerTrainingStateCheckpoint | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_schedulers = lr_schedulers
        self.scaler = scaler
        self.early_stopping = early_stopping
        if restart_state is not None:
            self.load_checkpoint(restart_state)

    @classmethod
    def create(
        cls,
        *,
        cfg: Any,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        lr_schedulers: dict | None,
        scaler: GradScaler | None,
        early_stopping: Any | None,
        train_cfg: Any,
        job_idx: int,
        pre_train: bool,
        device: torch.device,
        phase_epoch_0: int,
        max_epochs: int,
        epochs_for_seq_len_increase: int,
        path_best_model: Path,
        path_optimizer_best_model: Path,
        path_current_model: Path,
        path_current_optimizer: Path,
        hydra_output_dir: Path,
        restart_manager_path: Path | None,
        restart_state: InnerTrainingStateCheckpoint | None = None,
        batches_per_epoch: int | None = None,
        next_epoch_anchor: int | None = None,
    ) -> "LiveTrainingState":
        live_state = cls.create_uninitialized(
            cfg=cfg,
            train_cfg=train_cfg,
            job_idx=job_idx,
            pre_train=pre_train,
            device=device,
            phase_epoch_0=phase_epoch_0,
            max_epochs=max_epochs,
            batches_per_epoch=batches_per_epoch,
            epochs_for_seq_len_increase=epochs_for_seq_len_increase,
            path_best_model=path_best_model,
            path_optimizer_best_model=path_optimizer_best_model,
            path_current_model=path_current_model,
            path_current_optimizer=path_current_optimizer,
            hydra_output_dir=hydra_output_dir,
            restart_manager_path=restart_manager_path,
            restart_state=restart_state,
            next_epoch_anchor=next_epoch_anchor,
        )
        live_state.bind_runtime_objects(
            model=model,
            optimizer=optimizer,
            lr_schedulers=lr_schedulers,
            scaler=scaler,
            early_stopping=early_stopping,
            restart_state=restart_state,
        )
        return live_state

    def load_checkpoint(self, restart_state: InnerTrainingStateCheckpoint) -> None:
        if self.model is None:
            raise ValueError("Cannot load checkpoint before binding runtime model")
        if restart_state.job_idx != self.job_idx:
            raise ValueError(
                f"Restart state job_idx {restart_state.job_idx} does not match current job {self.job_idx}."
            )
        apply_inner_training_restart_state(
            restart_state,
            model=self.model,
            optimizer=self.optimizer,
            lr_schedulers=self.lr_schedulers,
            scaler=self.scaler,
            early_stopping=self.early_stopping,
            use_cuda=self.cfg.use_cuda,
        )
        logging.info(
            "Restored restart bundle for job %s at global epoch %s (phase epoch %s)",
            self.job_idx,
            self.phase_state.epoch_start,
            restart_state.phase_epoch,
        )

    def save_checkpoint(self, next_epoch: int) -> InnerTrainingStateCheckpoint | None:
        """Persist a restart checkpoint for the current epoch.

        Writes model and optimizer to their *current* paths, then builds and
        saves an ``InnerTrainingStateCheckpoint`` to ``restart_manager_path``.
        Does nothing when ``restart_manager_path`` is ``None``.
        """
        if self.restart_manager_path is None:
            return None
        if self.model is None:
            raise ValueError("Cannot save checkpoint before binding runtime model")
        self.model.save(self.path_current_model)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), self.path_current_optimizer)
        scheduler_states: dict[str, dict[str, Any]] = {}
        if self.lr_schedulers is not None:
            scheduler_states = {
                name: _move_to_cpu(sched.state_dict())
                for name, sched in self.lr_schedulers.items()
            }
        phase_state = self.phase_state
        restart_state = InnerTrainingStateCheckpoint(
            hydra_output_dir=str(self.hydra_output_dir.resolve()),
            restart_state_path=str(self.restart_manager_path.resolve()),
            checkpoint_reason='epoch_end',
            job_idx=self.job_idx,
            phase_epoch=next_epoch - phase_state.phase_epoch_0,
            first_epoch_is_evaluation=phase_state.first_epoch_is_evaluation,
            current_model_path=str(self.path_current_model.resolve()),
            current_optimizer_path=(
                str(self.path_current_optimizer.resolve())
                if self.optimizer is not None
                else ""
            ),
            best_model_path=(
                str(self.path_best_model.resolve()) if self.path_best_model is not None else ""
            ),
            best_optimizer_path=(
                str(self.path_optimizer_best_model.resolve())
                if self.path_optimizer_best_model is not None
                else ""
            ),
            model_state=_move_to_cpu(self.model.state_dict()),
            optimizer_state=(
                _move_to_cpu(self.optimizer.state_dict())
                if self.optimizer is not None
                else {}
            ),
            scheduler_states=scheduler_states,
            scaler_state=(
                _move_to_cpu(self.scaler.state_dict()) if self.scaler is not None else {}
            ),
            early_stopping_state=(
                _move_to_cpu(self.early_stopping.state_dict())
                if self.early_stopping is not None
                else {}
            ),
            nan_counter=phase_state.nan_counter,
            grad_norm_last_reduced_counter=phase_state.grad_norm_last_reduced_counter,
            stable_epochs=phase_state.stable_epochs,
            flag_out_of_seq_len_increase=phase_state.flag_out_of_seq_len_increase,
            epoch_stop=phase_state.epoch_stop,
            rng_state=_move_to_cpu(capture_rng_state(self.cfg.use_cuda)),
            deterministic_mode_active=phase_state.deterministic_mode_active,
            slurm_job_id=os.getenv("SLURM_JOB_ID"),
        )
        save_inner_restart_state(self.restart_manager_path, restart_state)
        return restart_state



@dataclass
class TrainingRestartState:
    """Legacy monolithic restart checkpoint schema kept during migration.

    See ``docs/bnode_core/ode/restart_training.md`` for how this persisted schema
    relates to ``TrainingPhaseState`` and ``LiveTrainingState``.
    """
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


def load_restart_state(path: Path) -> TrainingRestartState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid restart state payload in {path}")
    payload = dict(payload)
    payload.setdefault("restart_state_path", str(path.resolve()))
    return TrainingRestartState.from_dict(payload)


def load_outer_restart_state(path: Path) -> OuterTrainingStateCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid outer restart state payload in {path}")
    payload = dict(payload)
    payload.setdefault("restart_state_path", str(path.resolve()))
    return OuterTrainingStateCheckpoint.from_dict(payload)


def load_inner_restart_state(path: Path) -> InnerTrainingStateCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid inner restart state payload in {path}")
    payload = dict(payload)
    payload.setdefault("restart_state_path", str(path.resolve()))
    return InnerTrainingStateCheckpoint.from_dict(payload)


def save_restart_state(path: Path, state: TrainingRestartState) -> None:
    state.restart_state_path = str(path.resolve())
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.to_dict(), path)
    logging.info("Saved trainer restart state to %s", path)


def save_outer_restart_state(path: Path, state: OuterTrainingStateCheckpoint) -> None:
    state.restart_state_path = str(path.resolve())
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.to_dict(), path)
    logging.info("Saved outer trainer restart state to %s", path)


def save_inner_restart_state(path: Path, state: InnerTrainingStateCheckpoint) -> None:
    state.restart_state_path = str(path.resolve())
    state.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.to_dict(), path)
    logging.info("Saved inner trainer restart state to %s", path)


def load_restart_metadata(path: Path) -> dict[str, Any]:
    return load_restart_state(path).metadata()


def load_outer_restart_metadata(path: Path) -> dict[str, Any]:
    return load_outer_restart_state(path).metadata()


def _apply_runtime_checkpoint(
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_states: dict[str, dict[str, Any]],
    scaler_state: dict[str, Any],
    early_stopping_state: dict[str, Any],
    rng_state: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_schedulers: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: Any = None,
    use_cuda: bool = False,
) -> None:
    model.load_state_dict(model_state)

    if optimizer_state:
        if optimizer is None:
            raise ValueError("restart state contains optimizer_state but no optimizer was provided")
        optimizer.load_state_dict(optimizer_state)

    scheduler_keys = set(scheduler_states.keys())
    provided_scheduler_keys = set(lr_schedulers.keys()) if lr_schedulers is not None else set()
    if scheduler_keys != provided_scheduler_keys:
        raise ValueError(
            "restart state scheduler keys do not match current schedulers: "
            f"saved={sorted(scheduler_keys)}, current={sorted(provided_scheduler_keys)}"
        )
    if lr_schedulers is not None:
        for name, scheduler in lr_schedulers.items():
            scheduler.load_state_dict(scheduler_states[name])

    if scaler_state:
        if scaler is None:
            raise ValueError("restart state contains scaler_state but no scaler was provided")
        scaler.load_state_dict(scaler_state)

    if early_stopping_state:
        if early_stopping is None:
            raise ValueError("restart state contains early_stopping_state but no early_stopping was provided")
        early_stopping.load_state_dict(early_stopping_state)

    restore_rng_state(rng_state, use_cuda=use_cuda)


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
    _apply_runtime_checkpoint(
        model_state=state.model_state,
        optimizer_state=state.optimizer_state,
        scheduler_states=state.scheduler_states,
        scaler_state=state.scaler_state,
        early_stopping_state=state.early_stopping_state,
        rng_state=state.rng_state,
        model=model,
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
        scaler=scaler,
        early_stopping=early_stopping,
        use_cuda=use_cuda,
    )


def apply_inner_training_restart_state(
    state: InnerTrainingStateCheckpoint,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_schedulers: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: Any = None,
    use_cuda: bool = False,
) -> None:
    state.validate()
    _apply_runtime_checkpoint(
        model_state=state.model_state,
        optimizer_state=state.optimizer_state,
        scheduler_states=state.scheduler_states,
        scaler_state=state.scaler_state,
        early_stopping_state=state.early_stopping_state,
        rng_state=state.rng_state,
        model=model,
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
        scaler=scaler,
        early_stopping=early_stopping,
        use_cuda=use_cuda,
    )
