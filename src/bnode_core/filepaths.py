"""
Utilities for constructing, naming, and resolving files and directories used by
bnode projects. This module centralizes path conventions and discovery for:

- Auto-recognition of the configuration directory based on the current working
  directory or CLI overrides. Using little "hint" files, the code can determine if it is
  running inside a bnode package repository or a standalone project.
- Creation and discovery of data folders for raw data and datasets.
- Canonical naming for raw-data and dataset artifacts derived from the data
  generation configuration. This includes naming conventions that reflect
  model name, version, if initial states, parameters, and controls are sampled,
  sampling strategies itself, and dataset sizes.
- Paths to Hydra runtime output artifacts (models, optimizers, datasets).
- Resolution of MLflow artifact URIs to local filesystem paths via the
  MLFLOW_ARTIFACTS_DESTINATION environment variable.

"""

import os
import sys
import hydra
import logging

from pathlib import Path
from bnode_core.config import data_gen_config, convert_cfg_to_dataclass


def config_dir_auto_recognize() -> Path:
    """Auto-recognize and return the project's configuration directory.

    The current working directory is inspected for known locations of the configuration directory. 
    CLI flags can be used to override discovery and delegate resolution to Hydra.

    Returns:
        Path | None: Path to the discovered configuration directory. Returns
        None when --help is requested, or when CLI flags indicate that a config
        path will be provided externally (so Hydra may handle it).

    Raises:
        ValueError: If no configuration directory can be found and no CLI flag
        suggests that the path will be provided manually.

    Notes:
        Search order:
        1. If "--help" or "-h" present, print help and return None.
        2. If ".bnode_package_repo" exists and "resources/config/" exists, return it.
        3. Else if "./config/" exists, return it.
        4. Else, if "-cp"/"--config-path" or "-cd"/"--config-dir" are present,
           return None to allow Hydra to handle the path.
        5. Otherwise, log an error and raise ValueError.
    """
    if '--help' in sys.argv or '-h' in sys.argv:
        print('The config directory is auto-recognized based on the current working directory.')
        print('You can override this behavior by providing the config path via --config-path or --config-dir CLI arguments.')
        return None
    msg = ''
    if Path('.bnode_package_repo').exists():
        if Path('resources/config/').exists():
            return Path('resources/config/')
        else:
            msg += 'Even though .bnode_package_repo file exists, no config directory found in resources/config/.\n'
    else:
        if Path('./config/').exists():
            return Path('./config/')
        else:
            msg += 'No config directory found in the standard ./config/ location.\n'
    # Check if user provided config path via CLI args
    raise_error = True
    if '-cp' in sys.argv or '--config-path' in sys.argv:
        raise_error = False
    elif '--config-dir' in sys.argv or '-cd' in sys.argv:
        # we assume that the user provided also a config name in this case
        raise_error = False
    if raise_error:
        msg += 'Please ensure you are in a correct working directory or provide the config path manually.'
        logging.error(msg)
        raise ValueError(msg)
    else:
        return None


def create_path(path: Path, log: bool) -> None:
    """Create a directory if it does not exist.

    Args:
        path (Path): Directory path to create.
        log (bool): Whether to log the creation/exists message.

    Returns:
        None
    """
    if not path.exists():
        path.mkdir(parents=True)
        if log:
            logging.info('Created path: {}'.format(path))
    else:
        if log:
            logging.info('Path already exists: {}'.format(path))


def log_overwriting_file(path: Path) -> None:
    """Log whether a file will be written or overwritten.

    Args:
        path (Path): File path for which to log the action.

    Returns:
        None
    """
    if not path.exists():
        logging.info('Writing on file: {}'.format(path))
    else:
        logging.info('Overwriting file: {}'.format(path))


def raw_data_name(cfg: data_gen_config) -> str:
    """Build the canonical base name for a raw-data artifact.

    The name is derived from the model name/version and optional sampling
    strategy flags present in the configuration.

    Behavior:
        - Always includes model name and version.
        - Appends initial states sampling strategy if initial states are included. (E.g., '_s-R')
        - Appends parameters sampling strategy if parameters are included. (E.g., '_p-R')
        - Appends controls sampling strategy if controls are included. (E.g., '_c-RROCS')
        - Appends '_c-Mo' suffix if controls are only for sampling and the actual used controls are extracted
        from the model.

    Args:
        cfg (data_gen_config): Data generation configuration.

    Returns:
        str: Base name for raw-data artifacts.
    """
    RawData = cfg.pModel.RawData
    name = RawData.modelName + '_' + RawData.versionName
    if RawData.initial_states_include:
        name = name + '_s-' + RawData.initial_states_sampling_strategy
    if RawData.parameters_include:
        name = name + '_p-' + RawData.parameters_sampling_strategy
    if RawData.controls_include:
        name = name + '_c-' + RawData.controls_sampling_strategy
    if RawData.controls_only_for_sampling_extract_actual_from_model:
        name = name + '_c-Mo'
    return name


def dataset_name(cfg: data_gen_config, n_samples: int) -> str:
    """Build the canonical dataset name for a given configuration and size.

    Args:
        cfg (data_gen_config): Data generation configuration.
        n_samples (int): Number of samples in the dataset.

    Returns:
        str: Dataset name including sample count and optional suffix.
    """
    name = raw_data_name(cfg) + '__n-' + str(n_samples)
    if cfg.pModel.dataset_prep.dataset_suffix is not None:
        name = name + '_' + cfg.pModel.dataset_prep.dataset_suffix
    return name


def dir_data(log: bool = False) -> Path:
    """Resolve the root data directory for the project.

    Search order:
    1) If ".bnode_package_repo" exists: "./resources/data".
    2) Else if "../../.surrogate_test_data_repo" exists: "../../data".
    3) Else: "./data".

    The directory is created if missing.

    Note:
        In the future, it might be a good idea to switch to a more flexible directory assignment mechanism,
        e.g., via environment variables or configuration files.

    Args:
        log (bool): Whether to log directory creation/existence.

    Returns:
        Path: Root data directory.
    """
    if Path('.bnode_package_repo').exists():
        path = Path('./resources/data')
    else:
        path = Path('./data')
    create_path(path, log)
    return path


def dir_raw_data(cfg: data_gen_config, log: bool = False) -> Path:
    """Return the directory in which raw data for the config is stored.

    The path includes a subdirectory named by ``raw_data_name(cfg)`` and is
    created on demand.

    Args:
        cfg (data_gen_config): Data generation configuration.
        log (bool): Whether to log directory creation/existence.

    Returns:
        Path: Directory for raw data artifacts.
    """
    path = dir_data() / 'raw_data' / raw_data_name(cfg)
    create_path(path, log)
    return path


def filepath_raw_data(cfg: data_gen_config) -> Path:
    """Return the path to the raw-data file for a configuration.

    If ``cfg.pModel.RawData.raw_data_from_external_source`` is True, the path is
    resolved inside the raw-data directory using the external file name. Otherwise,
    a default file name of the form ``<raw_data_name>_raw_data.hdf5`` is used.

    Args:
        cfg (data_gen_config): Data generation configuration.

    Returns:
        Path: Path to the raw-data file.
    """
    if cfg.pModel.RawData.raw_data_from_external_source:
        file = dir_raw_data(cfg) / cfg.pModel.RawData.raw_data_path
    else:
        file = dir_raw_data(cfg) / (raw_data_name(cfg) + '_raw_data.hdf5')
    return file


def filepath_raw_data_config(cfg: data_gen_config) -> Path:
    """Return the path to the RawData configuration YAML file.

    Args:
        cfg (data_gen_config): Data generation configuration.

    Returns:
        Path: Path to the YAML configuration stored alongside raw-data.
    """
    file = dir_raw_data(cfg) / (raw_data_name(cfg) + '_RawData_config.yaml')
    return file


def dir_datasets(log: bool = False) -> Path:
    """Return the root directory for datasets, creating it if missing.

    Args:
        log (bool): Whether to log directory creation/existence.

    Returns:
        Path: Root datasets directory.
    """
    path = dir_data() / 'datasets'
    create_path(path, log)
    return path


def dir_specific_dataset_from_cfg(cfg: data_gen_config, n_samples: int, log: bool = False) -> Path:
    """Return the directory for a specific dataset derived from a config.

    The directory name is computed via :func:`dataset_name` and created if missing.

    Args:
        cfg (data_gen_config): Data generation configuration.
        n_samples (int): Number of samples in the dataset.
        log (bool): Whether to log directory creation/existence.

    Returns:
        Path: Directory for the specific dataset.
    """
    path = dir_datasets() / dataset_name(cfg, n_samples)
    create_path(path, log)
    return path


def dir_specific_dataset_from_name(name: str, log: bool = False) -> Path:
    """Return the directory for a specific dataset by name.

    Args:
        name (str): Dataset name.
        log (bool): Unused flag kept for API symmetry; directory is not created here.

    Returns:
        Path: Directory path for the dataset name.
    """
    path = dir_datasets() / name
    return path


def filepath_dataset(cfg: data_gen_config, n_samples: int) -> Path:
    """Return the path to a dataset file for a given config and size.

    Args:
        cfg (data_gen_config): Data generation configuration.
        n_samples (int): Number of samples in the dataset.

    Returns:
        Path: HDF5 dataset file path.
    """
    file = dir_specific_dataset_from_cfg(cfg, n_samples) / (dataset_name(cfg, n_samples) + '_dataset.hdf5')
    return file


def filepath_dataset_config(cfg: data_gen_config, n_samples: int) -> Path:
    """Return the path to the pModel configuration YAML for the dataset.

    Args:
        cfg (data_gen_config): Data generation configuration.
        n_samples (int): Number of samples in the dataset.

    Returns:
        Path: YAML configuration file path stored with the dataset.
    """
    file = dir_specific_dataset_from_cfg(cfg, n_samples) / (dataset_name(cfg, n_samples) + '_pModel_config.yaml')
    return file


def filepath_dataset_from_name(name: str) -> Path:
    """Return the HDF5 dataset file path for a given dataset name.
    If the name is already a file path, it is returned as a Path object.

    Args:
        name (str): Dataset name.

    Returns:
        Path: HDF5 dataset file path.
    """
    if Path(name) .is_file():
        file = Path(name)
    else:
        try: 
            file = dir_specific_dataset_from_name(name) / (name + '_dataset.hdf5')
        except Exception as e:
            raise FileNotFoundError('Dataset file not found for dataset_name: {}. Tried to load file directly and from dataset filepaths.'.format(cfg.dataset_name)) from e
    return file


def filepath_dataset_config_from_name(name: str) -> Path:
    """Return the pModel configuration YAML path for a dataset name.

    Args:
        name (str): Dataset name.

    Returns:
        Path: YAML configuration file path.
    """
    file = dir_specific_dataset_from_name(name) / (name + '_pModel_config.yaml')
    return file


def dir_current_hydra_output() -> Path:
    """Return the current Hydra runtime output directory.

    Returns:
        Path: Path to Hydra's current run output directory.
    """
    return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def filepath_model_current_hydra_output(phase: int | None = None) -> Path:
    """Return the model checkpoint path in the current Hydra output directory.

    Args:
        phase (int | None): Optional training phase index. When provided,
            the filename is "model_phase_{phase}.pt"; otherwise "model.pt".

    Returns:
        Path: Model checkpoint file path.
    """
    if phase is not None:
        return dir_current_hydra_output() / 'model_phase_{}.pt'.format(phase)
    else:
        return dir_current_hydra_output() / 'model.pt'


def filepath_pretrained_model_current_hydra_output() -> Path:
    """Return the pretrained model file path in the current Hydra output dir.

    Returns:
        Path: "model_pretrained.pt" path.
    """
    return dir_current_hydra_output() / 'model_pretrained.pt'


def filepath_dataset_current_hydra_output() -> Path:
    """Return the dataset file path in the current Hydra output directory.

    Returns:
        Path: "dataset.hdf5" path.
    """
    return dir_current_hydra_output() / 'dataset.hdf5'


def filepath_optimizer_current_hydra_output(phase: int | None = None) -> Path:
    """Return the optimizer state dict path in the current Hydra output dir.

    Args:
        phase (int | None): Optional training phase index. When provided,
            the filename is "optimizer_phase_{phase}.pt"; otherwise "optimizer.pt".

    Returns:
        Path: Optimizer checkpoint file path.
    """
    if phase is not None:
        return dir_current_hydra_output() / 'optimizer_phase_{}.pt'.format(phase)
    else:
        return dir_current_hydra_output() / 'optimizer.pt'


def filepath_from_ml_artifacts_uri(mlflow_uri: str) -> Path:
    """Resolve an MLflow artifacts URI to a local filesystem path.

    The base directory is read from the ``MLFLOW_ARTIFACTS_DESTINATION``
    environment variable. The leading "file://" is stripped from the env var
    value if present. The "mlflow-artifacts:/" prefix in ``mlflow_uri`` is also
    removed before joining.

    Args:
        mlflow_uri (str): An MLflow artifacts URI (e.g.,
            "mlflow-artifacts:/<experiment>/<run>/artifacts/..."), or a relative
            path component under the artifacts root.

    Returns:
        Path: Resolved local filesystem path.

    Raises:
        ValueError: If ``MLFLOW_ARTIFACTS_DESTINATION`` is not set.
    """
    try:
        # try resolving the path from the environment variable
        _dir = os.environ['MLFLOW_ARTIFACTS_DESTINATION']
        _dir = _dir.replace('file://', '')
    except:
        logging.error('MLFLOW_ARTIFACTS_DESTINATION not set')
        logging.error('please set the environment variable MLFLOW_ARTIFACTS_DESTINATION to the path where the mlflow artifacts are stored')
        logging.error('or provide the full path to the dataset')
        raise ValueError('MLFLOW_ARTIFACTS_DESTINATION not set as environment variable')
    mlflow_uri = _dir + os.sep + mlflow_uri.replace('mlflow-artifacts:/', '')
    mlflow_uri = Path(mlflow_uri)
    return mlflow_uri


def filepath_from_local_or_ml_artifacts(mlflow_path: str) -> Path:
    """Return a local Path from either a local path or an MLflow artifacts URI.
    If the provided path starts with "mlflow-artifacts:", it is resolved via
    :func:`filepath_from_ml_artifacts_uri`. Otherwise, it is treated as a local
    filesystem path.

    Args:
        mlflow_path (str): Local filesystem path or an MLflow artifacts URI
            beginning with "mlflow-artifacts:".

    Returns:
        Path: Local filesystem path.
    """
    if mlflow_path.startswith('mlflow-artifacts:'):
        _path = filepath_from_ml_artifacts_uri(mlflow_path)
    else:
        _path = mlflow_path
    return Path(_path)

@hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen_JuliusThermalTestModel', version_base=None)
def test(cfg: data_gen_config):
    cfg = convert_cfg_to_dataclass(cfg)
    print(raw_data_name(cfg))
    print(dir_raw_data(cfg, log=True))
    print(filepath_raw_data(cfg))
    log_overwriting_file(filepath_raw_data(cfg))

if __name__ == '__main__':
    test()