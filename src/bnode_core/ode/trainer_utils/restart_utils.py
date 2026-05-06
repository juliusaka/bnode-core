"""Restart-state utility helpers for the trainer."""

import copy
import logging
import mlflow
from pathlib import Path

import bnode_core.filepaths as filepaths
from bnode_core.config import base_training_settings_class
from bnode_core.ode.trainer_utils.restart_state import (
    OuterTrainingState,
    TrainingRestartState,
    load_restart_state,
)


def _load_restart_state_if_available(
) -> tuple[TrainingRestartState | None, Path]:
    restart_state_path = filepaths.filepath_training_restart_state_current_hydra_output()
    if not restart_state_path.exists():
        return None, restart_state_path
    restart_state = load_restart_state(restart_state_path)
    current_hydra_output = filepaths.dir_current_hydra_output().resolve()
    stored_hydra_output = Path(restart_state.hydra_output_dir).resolve()
    if stored_hydra_output != current_hydra_output:
        raise ValueError(
            'Restart state hydra output directory does not match current Hydra output directory. '
            f'Expected {restart_state.hydra_output_dir}, got {current_hydra_output}. '
            'Resume runs must reuse the same hydra.run.dir.'
        )
    active_run = mlflow.active_run()
    if (
        restart_state.mlflow_run_id is not None
        and active_run is not None
        and active_run.info.run_id != restart_state.mlflow_run_id
    ):
        raise ValueError(
            f"Active MLflow run {active_run.info.run_id} does not match restart-state run {restart_state.mlflow_run_id}."
        )
    logging.info('Loaded trainer restart state from %s', restart_state_path)
    return restart_state, restart_state_path


def _apply_saved_train_cfg(
    train_cfg: base_training_settings_class,
    saved_cfg_state: dict,
) -> base_training_settings_class:
    restored = copy.deepcopy(train_cfg)
    for key, value in saved_cfg_state.items():
        setattr(restored, key, value)
    return restored


def _load_outer_training_state(
    *,
    cfg,
    job_list: list[dict],
) -> OuterTrainingState:
    restart_state, restart_state_path = _load_restart_state_if_available()
    if restart_state is not None:
        job_list[restart_state.job_idx]["train_cfg"] = _apply_saved_train_cfg(
            job_list[restart_state.job_idx]["train_cfg"],
            restart_state.training_cfg_state,
        )
    return OuterTrainingState(
        cfg=cfg,
        job_list=job_list,
        restart_state_path=restart_state_path,
        restart_state=restart_state,
    )


def _clear_restart_state(path: Path) -> None:
    if path.exists():
        path.unlink()
        logging.info('Removed trainer restart state at %s', path)
