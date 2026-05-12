import os

import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra
from functools import wraps
from typing import Any, Callable
from pathlib import Path
import sys
import io
import shutil
import logging
import traceback
import subprocess

from bnode_core.ode.trainer_utils.restart_state import (
  OUTER_RESTART_STATE_FILENAME,
  load_train_all_phases_state_metadata,
)
from bnode_core.utils.mlflow_proxy import mlflow_proxy


def _normalize_tracking_uri(uri: str | None) -> str | None:
  if uri is None:
    return None
  return uri.rstrip('/')


def _resolve_resume_run_context(
  *,
  restart_metadata: dict[str, Any] | None,
  hydra_output_dir: Path,
  current_tracking_uri: str | None,
  current_experiment_name: str,
) -> str | None:
  if restart_metadata is None:
    return None

  stored_output_dir = Path(restart_metadata['hydra_output_dir']).resolve()
  current_output_dir = hydra_output_dir.resolve()
  if stored_output_dir != current_output_dir:
    raise ValueError(
      'Hydra output directory does not match the restart state. '
      f'Expected {stored_output_dir}, got {current_output_dir}. '
      'Resume runs must reuse the same hydra.run.dir.'
    )

  restart_run_id = restart_metadata.get('mlflow_run_id')
  if restart_run_id is None:
    raise ValueError(
      'Restart state is missing mlflow_run_id. '
      'Refusing to resume training into a different MLflow run.'
    )

  restart_tracking_uri = _normalize_tracking_uri(restart_metadata.get('mlflow_tracking_uri'))
  normalized_tracking_uri = _normalize_tracking_uri(current_tracking_uri)
  if (
    restart_tracking_uri is not None
    and normalized_tracking_uri is not None
    and restart_tracking_uri != normalized_tracking_uri
  ):
    raise ValueError(
      'Configured MLflow tracking URI does not match the restart state. '
      f'Expected {restart_tracking_uri}, got {normalized_tracking_uri}.'
    )

  restart_experiment_name = restart_metadata.get('mlflow_experiment_name')
  if (
    restart_experiment_name is not None
    and current_experiment_name != restart_experiment_name
  ):
    raise ValueError(
      'Configured MLflow experiment does not match the restart state. '
      f'Expected {restart_experiment_name}, got {current_experiment_name}.'
    )

  return restart_run_id


def _resolve_git_hash() -> str:
  """Return the current commit hash using git."""
  candidate_dirs = []
  try:
    candidate_dirs.append(Path(hydra.utils.get_original_cwd()))
  except Exception:
    pass
  candidate_dirs.append(Path.cwd())

  seen = set()
  for directory in candidate_dirs:
    dir_str = str(directory.resolve())
    if dir_str in seen:
      continue
    seen.add(dir_str)
    try:
      completed = subprocess.run(
        ['git', '-C', dir_str, 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
      )
      git_hash = completed.stdout.strip()
      if git_hash:
        return git_hash
    except Exception:
      continue

  return 'unknown'


def _start_mlflow_run(
  *,
  hydra_output_dir: Path,
  tracking_uri: str | None,
  experiment_name: str,
  run_name: str | None,
  git_hash: str,
  system_info,
) -> tuple[str | None, dict | None]:
  """Load restart state (if any), validate it, and start the MLflow run.

  Loads the outer restart-checkpoint metadata from *hydra_output_dir* (if it
  exists), validates that the configured tracking URI / experiment / output
  directory match the stored values, and then either resumes the existing
  MLflow run or opens a fresh one.

  Returns:
    (resolved_run_id, restart_metadata):
      - ``resolved_run_id`` is the run-ID of the resumed run, or ``None`` for
        a fresh run.
      - ``restart_metadata`` is the raw metadata dict, or ``None`` if no
        restart-checkpoint file was found.
  """
  def _load_restart_metadata() -> dict | None:
    restart_state_path = hydra_output_dir / OUTER_RESTART_STATE_FILENAME
    if not restart_state_path.exists():
      return None
    return load_train_all_phases_state_metadata(restart_state_path)

  def _assert_run_experiment_matches(run_id: str) -> None:
    existing_run = mlflow.get_run(run_id)
    existing_experiment = mlflow.get_experiment(existing_run.info.experiment_id)
    existing_experiment_name = existing_experiment.name if existing_experiment is not None else None
    if existing_experiment_name is not None and existing_experiment_name != experiment_name:
      raise ValueError(
        'Configured MLflow experiment does not match the existing run. '
        f'Run {run_id} belongs to experiment {existing_experiment_name}, '
        f'not {experiment_name}.'
      )

  restart_metadata = _load_restart_metadata()
  resolved_run_id = _resolve_resume_run_context(
    restart_metadata=restart_metadata,
    hydra_output_dir=hydra_output_dir,
    current_tracking_uri=tracking_uri,
    current_experiment_name=experiment_name,
  )

  if resolved_run_id is not None:
    _assert_run_experiment_matches(resolved_run_id)
    mlflow.start_run(run_id=resolved_run_id, log_system_metrics=True)
  else:
    mlflow.start_run(
      log_system_metrics=True,
      run_name=run_name,
      tags={
        'host': system_info.nodename,
        'os': system_info.sysname + ' ' + system_info.release + ' ' + system_info.version,
        'machine': system_info.machine,
        'git_hash': git_hash,
      },
    )

  return resolved_run_id, restart_metadata


def log_hydra_to_mlflow(func: Callable) -> Callable:
  '''
  Decorator to log hydra config to mlflow
  base on https://hydra.cc/docs/advanced/decorating_main/
  '''
  @wraps(func)
  def inner_decorator(cfg: DictConfig):
    
    from bnode_core.config import convert_cfg_to_dataclass, train_test_config_class

    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # set mlflow tracking uri and experiment name from config
    if cfg.mlflow_tracking_uri is not None:
      mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    else:
      logging.warning('mlflow_tracking_uri is None, using file-based mlflow in root directory')
      logging.warning('If the training is running here, you might have set an environment variable MLflow_TRACKING_URI that overrides the config value.')
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    resolved_run_id, restart_metadata = _start_mlflow_run(
      hydra_output_dir=hydra_output_dir,
      tracking_uri=mlflow.get_tracking_uri(),
      experiment_name=cfg.mlflow_experiment_name,
      run_name=cfg.mlflow_run_name,
      git_hash=_resolve_git_hash(),
      system_info=os.uname(),
    )

    active_run_id = mlflow.active_run().info.run_id
    mlflow_proxy.log_param('hydra_output_dir_rel', str(hydra_output_dir))
    mlflow_proxy.log_param('hydra_output_dir_absolute', str(hydra_output_dir.resolve()))
    mlflow_proxy.log_param('mlflow_run_id', active_run_id)
    if restart_metadata is not None:
      mlflow.set_tag('restart_state_path', restart_metadata['restart_state_path'])
      if restart_metadata.get('checkpoint_reason') is not None:
        mlflow.set_tag('restart_checkpoint_reason', restart_metadata['checkpoint_reason'])

    # make dataclass from config
    cfg = convert_cfg_to_dataclass(cfg)

    # save validated yaml in hydra folder
    OmegaConf.save(config=OmegaConf.structured(cfg), f=hydra_output_dir / '.hydra/config_validated.yaml')
    
    def convert_to_dict(obj):
      '''Convert a Pydantic dataclass / OmegaConf config to a flat dict safe for mlflow.log_params.'''
      return OmegaConf.to_container(OmegaConf.structured(obj), resolve=True)

    # log Network config to mlflow
    if isinstance(cfg, train_test_config_class):
      mlflow_proxy.log_params(convert_to_dict(cfg.nn_model.network))
      mlflow_proxy.log_params(convert_to_dict(cfg.nn_model.training))
      if hasattr(cfg.nn_model.training, 'pre_training') and cfg.nn_model.training.pre_train is True:
        mlflow_proxy.log_params({'pre_training_' + k: v for k, v in convert_to_dict(cfg.nn_model.training.pre_training).items()})
      if hasattr(cfg.nn_model.training, 'main_training'):
        for i, settings in enumerate(cfg.nn_model.training.main_training):
          mlflow_proxy.log_params({'main_training_' + str(i) + '_' + k: v for k, v in convert_to_dict(settings).items()})

    mlflow_proxy.log_param('dataset_name', cfg.dataset_name)
    
    # run function
    had_error = False
    res = None
    try:
      res = func(cfg) # pass cfg to decorated function
    except Exception as e:
      had_error = True
      mlflow_proxy.log_param('error', True)
      logging.error('Exception occured: {}'.format(e))
      logging.error(traceback.format_exc())
      if cfg.raise_exception:
          # Ensure no active run leaks into the next pytest case.
          mlflow.end_run()
          raise
    # if no exception, log error as False
    if not had_error:
      mlflow_proxy.log_param('error', False)
    
    # log hydra config as artifacts to mlflow, this includes all loggings
    # see https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    logging.info('Logging hydra outputs to mlflow')
    try:
      artifact_uri = mlflow.get_artifact_uri()
      tracking_uri = mlflow.get_tracking_uri() or ''
      logging.info(f" Artifact URI: {artifact_uri}")
      logging.info(f" Tracking URI: {tracking_uri}")
      is_file_based = (artifact_uri is not None and artifact_uri.startswith('file://')) or (tracking_uri.startswith('file:'))
      if is_file_based:
        # File-based tracking: copy directly into MLflow artifacts directory
        artifacts_dir = Path(artifact_uri.replace('file://', ''))
        logging.info(f"Detected file-based MLflow artifacts directory: {artifacts_dir}")
        errors = []
        for file in hydra_output_dir.rglob('*'):
          if not file.is_file():
            continue
          rel = file.relative_to(hydra_output_dir)
          dest = artifacts_dir / rel
          try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            logging.info(f"Copied artifact: {file} -> {dest}")
          except Exception as e:
            logging.warning(f"Failed to copy artifact {file} -> {dest}: {e}")
            errors.append(str(file))
        if errors:
          try:
            with open(hydra_output_dir / 'could_not_log_artifacts.txt', 'a') as f:
              name = hydra.utils.get_original_cwd().split('/')[-1]
              f.write('Computer: {}\n'.format(name))
              for ef in errors:
                f.write('File: {}\n'.format(ef))
            mlflow.log_artifact(hydra_output_dir / 'could_not_log_artifacts.txt')
          except Exception as e:
            logging.warning(f"Could not log could_not_log_artifacts.txt: {e}")
      else:
        # Remote tracking: log files one by one, but skip HDF5 (too large); log paths of HDF5 files instead
        logging.info('Remote MLflow tracking detected; logging artifacts file-by-file and skipping HDF5 content.')
        h5_paths = []
        for file in hydra_output_dir.rglob('*'):
          if not file.is_file():
            continue
          if file.suffix.lower() in ['.h5', '.hdf5']:
            h5_paths.append(str(file))
            logging.info(f"Skipping HDF5 artifact due to size: {file}")
            continue
          try:
            logging.info(f"\t logging file {file}")
            mlflow.log_artifact(file)
          except Exception as e:
            logging.warning(f"Could not log artifact {file}: {e}")
            try:
              with open(hydra_output_dir / 'could_not_log_artifacts.txt', 'a') as f:
                name = hydra.utils.get_original_cwd().split('/')[-1]
                f.write('Computer: {}\n'.format(name))
                f.write('File: {}\n'.format(file))
              mlflow.log_artifact(hydra_output_dir / 'could_not_log_artifacts.txt')
            except Exception as e2:
              logging.warning(f"Could not log could_not_log_artifacts.txt: {e2}")
        if h5_paths:
          # Log list of HDF5 paths as an artifact text file
          h5_list_file = hydra_output_dir / 'hdf5_artifacts_paths.txt'
          try:
            with open(h5_list_file, 'w') as f:
              f.write('\n'.join(h5_paths))
            mlflow.log_artifact(h5_list_file)
            logging.info(f"Logged HDF5 paths list: {h5_list_file}")
          except Exception as e:
            logging.warning(f"Could not log HDF5 paths list: {e}")
    except Exception as e:
      logging.warning(f"Unexpected error while logging artifacts: {e}")
    logging.info('Finished logging hydra config to mlflow')
    # Capture stdout from mlflow.end_run() and log it as well
    _buf = io.StringIO()
    _old_stdout = sys.stdout
    try:
      sys.stdout = _buf
      mlflow.end_run()
    finally:
      sys.stdout = _old_stdout
    _endrun_out = _buf.getvalue()
    if _endrun_out:
      # re-emit to stdout
      print(_endrun_out, end='')
      # and also log it line-by-line
      for _line in _endrun_out.splitlines():
        logging.info(f"mlflow.end_run(): {_line}")
    
    return res
  
  return inner_decorator
