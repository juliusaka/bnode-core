from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import torch

import bnode_core.filepaths as filepaths
from bnode_core.ode.trainer_utils.restart_state import (
    TrainAllPhasesState,
    TrainOnePhaseState,
)


class RestartCheckpointStore:
    """Atomic persistence for trainer restart artifacts."""

    def __init__(
        self,
        *,
        outer_path: Path,
        inner_path: Path,
        scheduler_path: Path,
        scaler_path: Path,
    ) -> None:
        self.outer_path = outer_path
        self.inner_path = inner_path
        self.scheduler_path = scheduler_path
        self.scaler_path = scaler_path

    @classmethod
    def from_current_hydra_output(cls) -> "RestartCheckpointStore":
        return cls(
            outer_path=filepaths.filepath_training_outer_restart_state_current_hydra_output(),
            inner_path=filepaths.filepath_training_inner_restart_state_current_hydra_output(),
            scheduler_path=filepaths.filepath_lr_schedulers_current_hydra_output(),
            scaler_path=filepaths.filepath_grad_scaler_current_hydra_output(),
        )

    @classmethod
    def from_paths(
        cls,
        *,
        outer_path: Path,
        inner_path: Path,
        scheduler_path: Path | None = None,
        scaler_path: Path | None = None,
    ) -> "RestartCheckpointStore":
        return cls(
            outer_path=outer_path,
            inner_path=inner_path,
            scheduler_path=(
                scheduler_path
                if scheduler_path is not None
                else filepaths.filepath_lr_schedulers_current_hydra_output()
            ),
            scaler_path=(
                scaler_path
                if scaler_path is not None
                else filepaths.filepath_grad_scaler_current_hydra_output()
            ),
        )

    def load_state_pair_if_available(
        self,
    ) -> tuple[TrainAllPhasesState | None, TrainOnePhaseState | None]:
        outer_exists = self.outer_path.exists()
        inner_exists = self.inner_path.exists()
        if not outer_exists and not inner_exists:
            return None, None
        if outer_exists != inner_exists:
            raise ValueError(
                "Trainer restart requires both outer and inner restart checkpoints in the Hydra output directory."
            )
        outer_state = TrainAllPhasesState().load(self.outer_path)
        inner_state = TrainOnePhaseState().load(self.inner_path)
        self._validate_checkpoint_uuid_pair(outer_state, inner_state)
        return outer_state, inner_state

    def save_outer_for_test_job(self, train_all_phases_state: TrainAllPhasesState) -> None:
        """Re-save only the outer restart state when advancing to a test job."""
        self._atomic_state_save(train_all_phases_state, self.outer_path)
        logging.info("Updated outer restart state for test job at %s", self.outer_path)

    def save_epoch_checkpoint(
        self,
        *,
        train_all_phases_state: TrainAllPhasesState,
        train_one_phase_state: TrainOnePhaseState,
        lr_schedulers,
        scaler,
    ) -> None:
        self._ensure_checkpoint_uuid_pair(train_all_phases_state, train_one_phase_state)
        scheduler_states = (
            {name: scheduler.state_dict() for name, scheduler in lr_schedulers.items()}
            if lr_schedulers is not None
            else {}
        )
        self._atomic_torch_save(scheduler_states, self.scheduler_path)
        self._atomic_torch_save(scaler.state_dict(), self.scaler_path)
        self._atomic_state_save(train_one_phase_state, self.inner_path)
        self._atomic_state_save(train_all_phases_state, self.outer_path)

    def clear_restart_artifacts(self) -> None:
        for path in (
            self.outer_path,
            self.inner_path,
            self.scheduler_path,
            self.scaler_path,
        ):
            if path.exists():
                path.unlink()
                logging.info("Removed trainer restart state at %s", path)

    @staticmethod
    def _ensure_checkpoint_uuid_pair(
        outer_state: TrainAllPhasesState,
        inner_state: TrainOnePhaseState,
    ) -> None:
        outer_uuid = outer_state.checkpoint_uuid
        inner_uuid = inner_state.checkpoint_uuid
        if outer_uuid is None and inner_uuid is None:
            shared_uuid = str(uuid.uuid4())
            outer_state.checkpoint_uuid = shared_uuid
            inner_state.checkpoint_uuid = shared_uuid
            return
        if outer_uuid is None and inner_uuid is not None:
            outer_state.checkpoint_uuid = inner_uuid
            return
        if inner_uuid is None and outer_uuid is not None:
            inner_state.checkpoint_uuid = outer_uuid
            return
        if outer_uuid != inner_uuid:
            raise ValueError(
                "Restart checkpoint UUID mismatch while saving: "
                f"outer={outer_uuid}, inner={inner_uuid}."
            )

    @staticmethod
    def _validate_checkpoint_uuid_pair(
        outer_state: TrainAllPhasesState,
        inner_state: TrainOnePhaseState,
    ) -> None:
        if outer_state.checkpoint_uuid is None or inner_state.checkpoint_uuid is None:
            raise ValueError(
                "Restart checkpoint pair is missing checkpoint UUID metadata."
            )
        if outer_state.checkpoint_uuid != inner_state.checkpoint_uuid:
            raise ValueError(
                "Restart checkpoint UUID mismatch: "
                f"outer={outer_state.checkpoint_uuid}, inner={inner_state.checkpoint_uuid}."
            )

    @staticmethod
    def _atomic_torch_save(payload, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = RestartCheckpointStore._temporary_path(path)
        try:
            with tmp_path.open("wb") as tmp_file:
                torch.save(payload, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, path)
            RestartCheckpointStore._fsync_directory(path.parent)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _atomic_state_save(state, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = RestartCheckpointStore._temporary_path(path)
        try:
            state.save(tmp_path)
            with tmp_path.open("rb") as tmp_file:
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, path)
            RestartCheckpointStore._fsync_directory(path.parent)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
