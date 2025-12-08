import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra
from functools import wraps
from typing import Callable
from omegaconf import DictConfig
import json
from pathlib import Path
import sys
import io
import shutil
import logging
from pathlib import Path
import traceback

def log_hydra_to_mlflow(func: Callable) -> Callable:
  '''
  Decorator to log hydra config to mlflow
  base on https://hydra.cc/docs/advanced/decorating_main/
  '''
  @wraps(func)
  def inner_decorator(cfg: DictConfig):
    
    from bnode_core.config import convert_cfg_to_dataclass, train_test_config_class

    # set mlflow tracking uri and experiment name from config
    if cfg.mlflow_tracking_uri is not None:
      mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    else:
      logging.warning('mlflow_tracking_uri is None, using file-based mlflow in root directory')
      logging.warning('If the training is running here, you might have set an environment variable MLflow_TRACKING_URI that overrides the config value.')
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    mlflow.start_run(log_system_metrics=True)

    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # make dataclass from config
    cfg = convert_cfg_to_dataclass(cfg)

    # save validated yaml in hydra folder
    OmegaConf.save(config=OmegaConf.structured(cfg), f=hydra_output_dir / '.hydra/config_validated.yaml')
    
    def convert_to_dict(cfg):
      json_str = json.dumps(cfg, default=lambda o: o.__dict__, indent=4) # adapted with chatgpt
      return json.loads(json_str)

    # log Network config to mlflow
    if type(cfg) == train_test_config_class:
      mlflow.log_params(convert_to_dict(cfg.nn_model.network))
      mlflow.log_params(convert_to_dict(cfg.nn_model.training))
      if 'pre_training' in cfg.nn_model.training.__pydantic_fields__ and cfg.nn_model.training.pre_train is True: # check if pre_training is in training config
        # append pre_training to keys:
        mlflow.log_params({'pre_training_' + k: v for k,v in convert_to_dict(cfg.nn_model.training.pre_training).items()})
      if 'main_training' in cfg.nn_model.training.__pydantic_fields__: # check if main_training is in training config
        for i, settings in enumerate(cfg.nn_model.training.main_training):
          # append main_training to keys:
          mlflow.log_params({'main_training_' + str(i) + '_' + k: v for k,v in convert_to_dict(settings).items()})

    mlflow.log_param('dataset_name', cfg.dataset_name)  
    
    # run function
    try:
      res = func(cfg) # pass cfg to decorated function
    except Exception as e:
      mlflow.log_param('error', True)
      logging.error('Exception occured: {}'.format(e))
      logging.error(traceback.format_exc())
      if cfg.raise_exception:
          raise e
    
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
            # If HDF5 file copied successfully, delete from output to save disk space
            if file.suffix.lower() in ['.h5', '.hdf5']:
              try:
                file.unlink()
                logging.info(f"Removed HDF5 from output after copy: {file}")
              except Exception as e:
                logging.warning(f"Could not remove HDF5 file {file}: {e}")
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