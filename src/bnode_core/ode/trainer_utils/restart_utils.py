"""Restart-state utility helpers for the trainer."""

import logging
import mlflow
from pathlib import Path

import bnode_core.filepaths as filepaths
from bnode_core.ode.trainer_utils.restart_state import (
    InnerTrainingStateCheckpoint,
    OuterTrainingStateCheckpoint,
    OuterTrainingState,
    load_inner_restart_state,
    load_outer_restart_state,
)


def _validate_restart_hydra_output(*, stored_hydra_output: str, current_hydra_output: Path) -> None:
    resolved_stored_output = Path(stored_hydra_output).resolve()
    if resolved_stored_output != current_hydra_output:
        raise ValueError(
            "Restart state hydra output directory does not match current Hydra output directory. "
            f"Expected {stored_hydra_output}, got {current_hydra_output}. "
            "Resume runs must reuse the same hydra.run.dir."
        )


def _validate_restart_run_id(mlflow_run_id: str | None) -> None:
    active_run = mlflow.active_run()
    if mlflow_run_id is not None and active_run is not None and active_run.info.run_id != mlflow_run_id:
        raise ValueError(
            f"Active MLflow run {active_run.info.run_id} does not match restart-state run {mlflow_run_id}."
        )


def _load_restart_checkpoints_if_available(
) -> tuple[
    OuterTrainingStateCheckpoint | None,
    InnerTrainingStateCheckpoint | None,
    Path,
    Path,
]:
    outer_restart_state_path = filepaths.filepath_training_outer_restart_state_current_hydra_output()
    inner_restart_state_path = filepaths.filepath_training_inner_restart_state_current_hydra_output()
    if not outer_restart_state_path.exists() and not inner_restart_state_path.exists():
        return None, None, outer_restart_state_path, inner_restart_state_path
    if outer_restart_state_path.exists() != inner_restart_state_path.exists():
        raise ValueError(
            "Trainer restart requires both outer and inner restart checkpoints in the Hydra output directory."
        )
    outer_restart_state = load_outer_restart_state(outer_restart_state_path)
    inner_restart_state = load_inner_restart_state(inner_restart_state_path)
    current_hydra_output = filepaths.dir_current_hydra_output().resolve()
    _validate_restart_hydra_output(
        stored_hydra_output=outer_restart_state.hydra_output_dir,
        current_hydra_output=current_hydra_output,
    )
    _validate_restart_hydra_output(
        stored_hydra_output=inner_restart_state.hydra_output_dir,
        current_hydra_output=current_hydra_output,
    )
    _validate_restart_run_id(outer_restart_state.mlflow_run_id)
    logging.info("Loaded outer trainer restart state from %s", outer_restart_state_path)
    logging.info("Loaded inner trainer restart state from %s", inner_restart_state_path)
    return (
        outer_restart_state,
        inner_restart_state,
        outer_restart_state_path,
        inner_restart_state_path,
    )


def _load_outer_training_state(
    *,
    cfg,
    job_list: list[dict],
) -> OuterTrainingState:
    (
        outer_restart_state,
        inner_restart_state,
        outer_restart_state_path,
        inner_restart_state_path,
    ) = _load_restart_checkpoints_if_available()
    return OuterTrainingState(
        cfg=cfg,
        job_list=job_list,
        outer_restart_state_path=outer_restart_state_path,
        inner_restart_state_path=inner_restart_state_path,
        outer_restart_state=outer_restart_state,
        inner_restart_state=inner_restart_state,
    )


def _clear_restart_state(outer_path: Path, inner_path: Path) -> None:
    for path in (outer_path, inner_path):
        if path.exists():
            path.unlink()
            logging.info("Removed trainer restart state at %s", path)
