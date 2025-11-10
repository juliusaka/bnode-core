import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra
from functools import wraps
from typing import Callable
from omegaconf import DictConfig
import json
from pathlib import Path
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
      mlflow.log_artifacts(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    except:
      # Fallback to logging each file individually and log name of file to a text file. It seems like log_artifacts
      # works now for big files (at least on linux)
      logging.info('running mlflow log artifacts and catching exception if it fails')
      for file in hydra_output_dir.rglob('*'):
        if file.is_file():
          try:
            mlflow.log_artifact(file)
          except:
            logging.warning('Could not log artifact: {}'.format(file))
            # log name of file instead to a text file
            with open(hydra_output_dir / 'could_not_log_artifacts.txt', 'a') as f:
              # add name of this computer to file
              name = hydra.utils.get_original_cwd().split('/')[-1]
              f.write('Computer: {}\n'.format(name))
              f.write('File: {}\n'.format(file))
            logging.info('Logged name of file to could_not_log_artifacts.txt')
            try:
              mlflow.log_artifact(hydra_output_dir / 'could_not_log_artifacts.txt')
            except:
              logging.warning('Could not even log file could_not_log_artifacts.txt')
    logging.info('Finished logging hydra config to mlflow')
    mlflow.end_run()
    
    return res
  
  return inner_decorator