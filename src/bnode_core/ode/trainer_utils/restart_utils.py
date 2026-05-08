"""Restart-state utility helpers for the trainer."""

import logging
from pathlib import Path

import mlflow

import bnode_core.filepaths as filepaths
from bnode_core.ode.trainer_utils.restart_state import (
    TrainAllPhasesState,
    TrainOnePhaseState,
    load_train_all_phases_state,
    load_train_one_phase_state,
)


def _validate_restart_run_id(mlflow_run_id: str | None) -> None:
    active_run = mlflow.active_run()
    if mlflow_run_id is not None and active_run is not None and active_run.info.run_id != mlflow_run_id:
        raise ValueError(
            f"Active MLflow run {active_run.info.run_id} does not match restart-state run {mlflow_run_id}."
        )


def _load_restart_states_if_available(
) -> tuple[
    TrainAllPhasesState | None,
    TrainOnePhaseState | None,
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
    outer_restart_state = load_train_all_phases_state(outer_restart_state_path)
    inner_restart_state = load_train_one_phase_state(inner_restart_state_path)
    _validate_restart_run_id(outer_restart_state.mlflow_run_id)
    logging.info("Loaded train_all_phases_state from %s", outer_restart_state_path)
    logging.info("Loaded train_one_phase_state from %s", inner_restart_state_path)
    return (
        outer_restart_state,
        inner_restart_state,
        outer_restart_state_path,
        inner_restart_state_path,
    )


def _validate_restart_target(
    *,
    job_list: list[dict],
    train_all_phases_state: TrainAllPhasesState | None,
    train_one_phase_state: TrainOnePhaseState | None,
) -> None:
    if train_all_phases_state is None and train_one_phase_state is None:
        return
    if train_all_phases_state is None or train_one_phase_state is None:
        raise ValueError(
            "Trainer restart requires both train_all_phases_state and train_one_phase_state."
        )
    if train_all_phases_state.job_idx >= len(job_list):
        raise ValueError(
            "Restart state refers to job index "
            f"{train_all_phases_state.job_idx}, but only {len(job_list)} jobs exist."
        )
    target_job = job_list[train_all_phases_state.job_idx]
    if target_job["test"] or target_job["pre_train"]:
        raise ValueError("Trainer restart currently supports main-training phases only.")


def load_restart_state_pair(
    *,
    job_list: list[dict],
) -> tuple[TrainAllPhasesState | None, TrainOnePhaseState | None, Path, Path]:
    (
        train_all_phases_state,
        train_one_phase_state,
        outer_restart_state_path,
        inner_restart_state_path,
    ) = _load_restart_states_if_available()
    _validate_restart_target(
        job_list=job_list,
        train_all_phases_state=train_all_phases_state,
        train_one_phase_state=train_one_phase_state,
    )
    return (
        train_all_phases_state,
        train_one_phase_state,
        outer_restart_state_path,
        inner_restart_state_path,
    )


def _clear_restart_state(outer_path: Path, inner_path: Path) -> None:
    for path in (
        outer_path,
        inner_path,
        filepaths.filepath_lr_schedulers_current_hydra_output(),
        filepaths.filepath_grad_scaler_current_hydra_output(),
    ):
        if path.exists():
            path.unlink()
            logging.info("Removed trainer restart state at %s", path)
