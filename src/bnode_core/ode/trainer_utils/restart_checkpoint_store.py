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

    BUNDLE_VERSION = 2

    def __init__(self, *, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_current_hydra_output(cls) -> "RestartCheckpointStore":
        return cls(checkpoint_path=filepaths.filepath_restart_checkpoint_current_hydra_output())

    def load_checkpoint_if_available(
        self,
    ) -> tuple[
        TrainAllPhasesState | None,
        TrainOnePhaseState | None,
        dict | None,  # scheduler_states
        dict | None,  # scaler_state
        dict | None,  # model_state_dict
        dict | None,  # optimizer_state_dict
    ]:
        """Returns (outer_state, inner_state, scheduler_states, scaler_state, model_state_dict, optimizer_state_dict)
        or (None, None, None, None, None, None)."""
        if not self.checkpoint_path.exists():
            return None, None, None, None, None, None
        bundle = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self._validate_bundle_version(bundle)
        outer_state = TrainAllPhasesState().load_from_state_dict(bundle["outer"])
        inner_state = TrainOnePhaseState().load_from_state_dict(bundle["inner"])
        return outer_state, inner_state, bundle["scheduler"], bundle["scaler"], bundle["model"], bundle["optimizer"]

    def save_epoch_checkpoint(
        self,
        *,
        train_all_phases_state: TrainAllPhasesState,
        train_one_phase_state: TrainOnePhaseState,
        lr_schedulers,
        scaler,
        model,
        optimizer,
    ) -> None:
        scheduler_states = (
            {name: s.state_dict() for name, s in lr_schedulers.items()}
            if lr_schedulers is not None
            else {}
        )
        bundle = {
            "bundle_version": self.BUNDLE_VERSION,
            "outer": train_all_phases_state.to_state_dict(),
            "inner": train_one_phase_state.to_state_dict(),
            "scheduler": scheduler_states,
            "scaler": scaler.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        self._atomic_save(bundle)
        logging.info("Saved restart checkpoint to %s", self.checkpoint_path)

    def save_outer_for_test_job(self, train_all_phases_state: TrainAllPhasesState) -> None:
        """Re-save the bundle with an updated outer state when advancing to a test job."""
        if not self.checkpoint_path.exists():
            # No checkpoint exists (e.g. load_trained_model_for_test=True skipped all training).
            # The test job will simply re-run from the start on any subsequent restart.
            logging.info(
                "No restart checkpoint found at %s; skipping outer state update for test job.",
                self.checkpoint_path,
            )
            return
        bundle = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        bundle["outer"] = train_all_phases_state.to_state_dict()
        self._atomic_save(bundle)
        logging.info("Updated outer restart state for test job at %s", self.checkpoint_path)

    def clear_restart_artifacts(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logging.info("Removed trainer restart checkpoint at %s", self.checkpoint_path)

    @classmethod
    def _validate_bundle_version(cls, bundle: dict) -> None:
        version = bundle.get("bundle_version")
        if version != cls.BUNDLE_VERSION:
            raise ValueError(
                f"Restart checkpoint bundle version mismatch: got {version}, expected {cls.BUNDLE_VERSION}."
            )

    def _atomic_save(self, bundle: dict) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.parent / f".{self.checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with tmp_path.open("wb") as f:
                torch.save(bundle, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.checkpoint_path)
            self._fsync_directory(self.checkpoint_path.parent)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
