"""Restart-state utility helpers for the trainer."""

import logging

import mlflow

from bnode_core.ode.trainer_utils.restart_checkpoint_store import RestartCheckpointStore
from bnode_core.ode.trainer_utils.restart_state import (
    TrainAllPhasesState,
    TrainOnePhaseState,
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
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    RestartCheckpointStore,
]:
    checkpoint_store = RestartCheckpointStore.from_current_hydra_output()
    outer_restart_state, inner_restart_state, scheduler_states, scaler_state, model_state, optimizer_state = checkpoint_store.load_checkpoint_if_available()
    if outer_restart_state is None:
        return None, None, None, None, None, None, checkpoint_store
    _validate_restart_run_id(outer_restart_state.mlflow_run_id)
    logging.info("Loaded restart checkpoint from %s", checkpoint_store.checkpoint_path)
    return outer_restart_state, inner_restart_state, scheduler_states, scaler_state, model_state, optimizer_state, checkpoint_store


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
    if target_job["pre_train"]:
        raise ValueError("Trainer restart does not support resuming at pre-training phases.")


def load_restart_state_pair(
    *,
    job_list: list[dict],
) -> tuple[
    TrainAllPhasesState | None,
    TrainOnePhaseState | None,
    dict | None,
    dict | None,
    dict | None,
    dict | None,
    RestartCheckpointStore,
]:
    train_all_phases_state, train_one_phase_state, scheduler_states, scaler_state, model_state, optimizer_state, checkpoint_store = _load_restart_states_if_available()
    _validate_restart_target(
        job_list=job_list,
        train_all_phases_state=train_all_phases_state,
        train_one_phase_state=train_one_phase_state,
    )
    return train_all_phases_state, train_one_phase_state, scheduler_states, scaler_state, model_state, optimizer_state, checkpoint_store
