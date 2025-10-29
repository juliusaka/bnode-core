"""

Pydantic dataclass configuration schema used by the B-NODE project.

This module provides:

- Typed, validated dataclasses for solver settings, raw-data generation,
    dataset preparation and multiple neural-network model families (base, VAE,
    neural-ODE, latent-ODE).
- Field- and model-level validation via pydantic functional and model validators.
- Helpers to register all config dataclasses with Hydra's ConfigStore.
- Small utilities to convert an OmegaConf DictConfig into the corresponding
    Python dataclass and to persist dataclasses as YAML.

Notes
-----
- Use get_config_store() at program startup to register the configuration
    schemas with Hydra. Example:
    
            from bnode_core.config import get_config_store
            cs = get_config_store()

- The dataclasses are intended to be composed/loaded via Hydra + OmegaConf and
    validated through pydantic validators. See each dataclass docstring for
    details about available fields and validation behaviour.

ConfigStore layout (high level)
-------------------------------
```
base_data_gen: data_gen_config(
        pModel = base_pModelClass(
                RawData = RawDataClass,
                dataset_prep = base_dataset_prep_class
        )
)

base_train_test: train_test_config_class(
        nn_model = one of:
            - base_nn_model: base_nn_model_class
            - pels_vae: base_nn_model_class(network=pels_vae_network_class, training=pels_vae_training_settings_class)
            - neural_ode_base: base_ode_nn_model_class(network=neural_ode_network_class, training=base_neural_ode_training_settings_class)
            - latent_ode_base: base_latent_ode_nn_model_class(network=latent_ode_network_class, training=base_latent_ode_training_settings_class)
)

base_onnx_export: onnx_export_config_class  (inherits load_latent_ode_config_class)
```

Utilities
---------
- ```convert_cfg_to_dataclass(cfg: DictConfig) -> dataclass```
    Convert an OmegaConf DictConfig (Hydra config) into the corresponding
    validated Python dataclass (uses OmegaConf.to_object and pydantic validation).

- ```save_dataclass_as_yaml(cfg: dataclass, path: str)```
    Persist a dataclass to a YAML file (uses pydantic/dataclasses.asdict then yaml.dump).

See individual dataclass definitions below for field-level documentation,
validation rules and "gotchas".
"""
from pydantic.dataclasses import dataclass
from dataclasses import asdict, field
from pydantic import ValidationInfo
from pydantic.functional_validators import field_validator
from hydra.core.config_store import ConfigStore
import hydra
import numpy as np
from omegaconf import MISSING, OmegaConf
from omegaconf import DictConfig
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
import logging
from pydantic import model_validator

########################################################################################################################
# Dataclasses
########################################################################################################################

"""pModel config dataclass definitions"""
@dataclass
class SolverClass:
    """
    Configuration for simulation timing and solver behavior.

    Notes:

    - sequence_length is auto-computed from start/end/timestep during model validation.
    - Changing simulationStartTime, simulationEndTime, or timestep will recompute sequence_length.
    - all times are in seconds
    - the timeout variable aborts the simulation if it runs longer than the specified time (in seconds) in the raw data generation. If its None, no timeout is set.

    Attributes:
        simulationStartTime (float): Start time of the simulation in seconds.
        simulationEndTime (float): End time of the simulation in seconds.
        timestep (float): Fixed integration step size in seconds.
        tolerance (float): Solver tolerance used to run the co-sim FMU.
        sequence_length (int): Number of steps including the initial state; computed after validation.
        timeout (Optional[float]): Max wall-clock seconds allowed for raw-data simulation; None disables.
    """
    simulationStartTime: float = 0.0
    simulationEndTime: float = 1.0
    timestep: float = 0.1
    tolerance: float = 1e-6
    sequence_length: int = None
    timeout: Optional[float] = None # in seconds, if None, no timeout is set
    
    @model_validator(mode='after')
    def calculate_sequence_length(self):
        # +1 because of the initial state, ceil to make sure that the last timestep is included
        v = np.ceil((self.simulationEndTime - self.simulationStartTime) / self.timestep) + 1 
        logging.info(f'sequence_length has been set to {int(v)}')
        self.sequence_length = int(v)
        return self

@dataclass
class RawDataClass:
    """
    Raw model and sampling configuration used to generate datasets.

    Gotchas:

    - fmuPath is required (and must end with .fmu) when raw_data_from_external_source is False.
    - parameters entries must be length-3 lists [min, max, default]; if scalar/len-1, bounds are auto-derived
      using parameters_default_lower_factor/parameters_default_upper_factor and default must lie within bounds.
    - states/controls entries must be length-2 lists [min, max]; if None, class defaults are injected; lower <= upper.
    - When controls_sampling_strategy is ROCS/RROCS, both controls_frequency_min_in_timesteps and
      controls_frequency_max_in_timesteps are required and min <= max.
    - When controls_sampling_strategy is 'file' or 'constantInput', controls_file_path must exist; otherwise it is ignored.
    - If initial_states_include is True, dataset_prep.start_time must equal Solver.simulationStartTime (checked in base_pModelClass). 
        This is because initial states are sampled at the simulation start time, and excluding these times would not make sense.

    Attributes:
        raw_data_from_external_source (bool): If True, use external raw data and ignore FMU-related fields.
        raw_data_path (Optional[str]): Path to an existing raw-data file to consume.
        modelName (str): Model identifier/name.
        fmuPath (Optional[str]): Path to a .fmu file; required if raw_data_from_external_source is False.
        versionName (str): Version tag for the model/dataset.
        states_default_lower_value (float): Default lower bound used when a state range is None.
        states_default_upper_value (float): Default upper bound used when a state range is None.
        states (Optional[Dict]): Mapping state-name -> [min, max] or None to inject defaults; list must have length 2 and min <= max. E.g. {'s1': [0, 10], 's2': None}.
        initial_states_include (bool): Whether to include initial state sampling in data generation.
        states_der_include (bool): Whether to save state derivatives from the fmu, found by searching for 'der(%state_name)'.
        initial_states_sampling_strategy (str): Strategy for initial-state sampling (e.g., 'R' for random). See raw_data_generation.py for options.
        parameters_default_lower_factor (float): Factor to derive default lower bound from a scalar parameter default.
        parameters_default_upper_factor (float): Factor to derive default upper bound from a scalar parameter default.
        parameters (Optional[Dict]): Mapping parameter-name -> [min, max, default] or scalar; list must have length 3 and default within [min, max].
        parameters_include (bool): Whether to include parameter sampling in data generation.
        parameters_sampling_strategy (str): Strategy for parameter sampling (e.g., 'R'). See raw_data_generation.py for options.
        controls_default_lower_value (float): Default lower bound for a control when unspecified.
        controls_default_upper_value (float): Default upper bound for a control when unspecified.
        controls (Optional[Dict]): Mapping control-name -> [min, max] or None to inject defaults; list must have length 2 and min <= max.
        controls_include (bool): Whether to include controls in data generation.
        controls_sampling_strategy (str): Strategy for control sampling (e.g., 'R', 'ROCS', 'RROCS', 'file', 'constantInput').
        controls_frequency_min_in_timesteps (Optional[int]): Minimum control hold-frequency in time steps (required with ROCS/RROCS).
        controls_frequency_max_in_timesteps (Optional[int]): Maximum control hold-frequency in time steps (required with ROCS/RROCS; must be >= min).
        controls_file_path (Optional[str]): CSV/Excel path with time and control columns when using 'file'/'constantInput'; must exist.
        controls_only_for_sampling_extract_actual_from_model (bool): Use controls only for sampling while extracting actual controls from the model.
        controls_from_model (Optional[List[str]]): Names of controls to read from the model if available.
        outputs (Optional[List[str]]): List of variable names to be treated as outputs.
        Solver (SolverClass): Nested solver configuration used for time grid and lengths.
        n_samples (int): Number of trajectories to generate.
        creation_date (Optional[str]): Optional creation date string for bookkeeping.
    """
    raw_data_from_external_source: bool = False # this sets the code to not use anything else from this class
    raw_data_path: Optional[str] = None # give path to raw data file in raw_data
    modelName: str = MISSING
    fmuPath: Optional[str] = None
    versionName: str = 'v1'
    states_default_lower_value: float = -1000.0
    states_default_upper_value: float = 1000.0
    states: Optional[Dict] = field(default_factory=dict)
    initial_states_include: bool = False
    states_der_include: bool = True
    initial_states_sampling_strategy: str = 'R'
    parameters_default_lower_factor: float = 0.2
    parameters_default_upper_factor: float = 5.0
    parameters: Optional[Dict] = None
    parameters_include: bool = False
    parameters_sampling_strategy: str = 'R'
    controls_default_lower_value: float = 0.2
    controls_default_upper_value: float = 5.0
    controls: Optional[Dict] = field(default_factory=dict)
    controls_include: bool = False
    controls_sampling_strategy: str = 'R'
    controls_frequency_min_in_timesteps: Optional[int] = None
    controls_frequency_max_in_timesteps: Optional[int] = None
    controls_file_path: Optional[str] = None # must contains headers with time, control variable names
    controls_only_for_sampling_extract_actual_from_model: bool = False
    controls_from_model: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    Solver: SolverClass = field(default_factory=SolverClass)
    n_samples: int = 2048
    creation_date: Optional[str] = None
    @field_validator('fmuPath')
    @classmethod
    def check_fmuPath(cls, v, info: ValidationInfo):
        if v is None:
            if info.data['raw_data_from_external_source'] == False:
                raise ValueError('fmuPath must be set if raw_data_from_external_source is False')
        else:
            path = Path(v)
            if path.suffix != '.fmu':
                raise ValueError('fmuPath must be a .fmu file')
            if not path.exists():
                Warning('fmuPath does not exist')
            return str(path.as_posix())
    @field_validator('parameters')
    @classmethod
    def check_parameters(cls, v, info: ValidationInfo):
        if v is None:
            logging.info('no parameters included.')
        else:
            for key, value in v.items():
                if not isinstance(value, list):
                    value = [value]
                if len(value) == 1:
                    v[key] = [value[0] * info.data['parameters_default_lower_factor'], value[0] * info.data['parameters_default_upper_factor'], value[0]]
                    logging.info(f'parameter ranges {key} has been set to \t {[round(x, 4) for x in v[key]]}')
                if not len(v[key]) == 3:
                    raise ValueError(f'parameter {key} must be a dictionary of lists with 3 elements')
                if v[key][2] < v[key][0] or v[key][2] > v[key][1]:
                    raise ValueError(f'parameter {key} default value must be within the lower and upper bounds')
        return v
    @field_validator('controls')
    @classmethod
    def check_controls(cls, v, info: ValidationInfo):
        for key, value in v.items():
            if value is None:
                v[key] = [info.data['controls_default_lower_value'], info.data['controls_default_upper_value']]
                logging.info(f'control ranges {key} has been set to \t {[round(x, 4) for x in v[key]]}')
            if not len(v[key]) == 2:
                raise ValueError(f'control {key} must be a dictionary of lists with 2 elements')
            if v[key][0] > v[key][1]:
                raise ValueError(f'control {key}: first element must be smaller than second element')
        return v
    @field_validator('states')
    @classmethod
    def check_states(cls,v, info: ValidationInfo):
        for key, value in v.items():
            if value is None:
                v[key] = [info.data['states_default_lower_value'], info.data['states_default_upper_value']]
                logging.info(f'state ranges {key} has been set to \t {[round(x, 4) for x in v[key]]}')
            if not len(v[key]) == 2:
                raise ValueError(f'state {key} must be a dictionary of lists with 2 elements')
            if v[key][0] > v[key][1]:
                raise ValueError(f'state {key}: first element must be smaller than second element')
        return v
    @field_validator('controls_frequency_max_in_timesteps')
    @classmethod
    def check_controls_frequency(cls, v, info: ValidationInfo):
        if info.data['controls_sampling_strategy'] == 'ROCS' or info.data['controls_sampling_strategy'] == 'RROCS':
            if v is None or info.data['controls_frequency_min_in_timesteps'] is None:
                raise ValueError('controls_frequency_min_in_timesteps and controls_frequency_max_in_timesteps must be set if controls_sampling_strategy is ROCS')
            if v < info.data['controls_frequency_min_in_timesteps']:
                raise ValueError('controls_frequency_min_in_timesteps must be smaller than controls_frequency_max_in_timesteps')
        return v
    @field_validator('controls_file_path')
    @classmethod
    def check_controls_file_path(cls, v, info: ValidationInfo):
        if info.data['controls_sampling_strategy'] == 'file' or info.data['controls_sampling_strategy'] == 'constantInput':
            path = Path(v)
            if not path.exists():
                raise ValueError('controls_file_path does not exist')
            return str(path.as_posix())
        else:
            return None
    
@dataclass
class base_dataset_prep_class:
    """
    Dataset preparation settings for slicing, filtering, transforming, and splitting data.

    Gotchas:

    - The lists for states/parameters/controls/outputs accept the sentinel value ['all'] to include all available items.
    - filter_trajectories_limits is a dictionary where keys are variable names and values are [min, max] lists to filter out trajectories
      that have variable values outside these limits.
    - filter_trajectories_expression contains Python expressions as strings; all expressions in a list must evaluate to True
      to filter out a trajectory. WARNING: This is is not tested yet, debug carefully before use.
    - transforms is a dictionary where keys are variable names and values are strings representing Python expressions to transform the data. 
        example: var1: 'np.log(# + 1)' where # is replaced by the variable data.
    - validation_fraction and test_fraction control split sizes; sequence_length can remain None to use model defaults.

    Attributes:
        dataset_suffix (Optional[str]): Suffix to append to dataset names/artifacts.
        n_samples (List[int]): Samples to include in the dataset, a list with multiple entries will generate multiple datasets, but with same "common_test" and "common_validation" sets.
        filter_trajectories_limits (Optional[Dict[str, List]]): Variable limits for filtering trajectories, each as [min, max].
        filter_trajectories_expression (Optional[Dict[str, List[str]]]): Expressions that, when True, mark trajectories for removal. Attention; not tested yet.
        transforms (Optional[Dict[str, str]]): Mapping variable -> Python expression where '#' is replaced by the variable data.
        states (Optional[List[str]]): Names of states to keep, or ['all'].
        parameters (Optional[List[str]]): Names of parameters to keep, or ['all'].
        parameters_remove (bool): If True, all parameters are removed even if present in source data.
        controls (Optional[List[str]]): Names of controls to keep, or ['all'].
        outputs (Optional[List[str]]): Names of outputs to keep, or ['all'].
        start_time (float): Start time for slicing trajectories. E.g. used for excluding initial transients of initialization.
        end_time (float): End time for slicing trajectories (inf keeps full length).
        sequence_length (Optional[int]): Desired sequence length; None keeps source/model default.
        validation_fraction (float): Fraction of data reserved for validation split. Must be between 0 and 1. validation_fraction + test_fraction must be < 1.
        test_fraction (float): Fraction of data reserved for test split. Must be between 0 and 1. validation_fraction + test_fraction must be < 1.
        validation_idx_start (Optional[int]): Index to start validation split, leave empty, is auto-computed in dataset_preperation.py
        test_idx_start (Optional[int]): Index to start test split, leave empty, is auto-computed in dataset_preperation.py
    """
    dataset_suffix: Optional[str] = None
    n_samples: List[int] = field(default_factory=list)
    filter_trajectories_limits: Optional[Dict[str, List]] = field(default_factory=dict) 
    # e.g. stratifiedHeatFlow.partition[1].heatCapacitor.T: [10, 30]. 
    filter_trajectories_expression: Optional[Dict[str, List[str]]] = field(default_factory=dict)
    # str can be used to define a python expression to be evaluated, 
    # the list can contain multiple expressions that all must hold true for the trajectory to be removed
    transforms: Optional[Dict[str, str]] = field(default_factory=dict)
    states: Optional[List[str]] = field(default_factory=lambda: ['all'])
    parameters: Optional[List[str]] = field(default_factory=lambda: ['all'])
    parameters_remove: bool = False
    controls: Optional[List[str]] = field(default_factory=lambda: ['all'])
    outputs: Optional[List[str]] = field(default_factory=lambda: ['all'])
    start_time: float = 0
    end_time: float = float('inf')
    sequence_length: Optional[int] = None
    validation_fraction: float = 0.12
    test_fraction: float = 0.12
    validation_idx_start: Optional[int] = None
    test_idx_start: Optional[int] = None



@dataclass
class base_pModelClass:
    """
    Composite configuration that couples raw data settings and dataset preparation for a physical model.

    Gotchas:

    - If RawData.initial_states_include is True, dataset_prep.start_time must equal RawData.Solver.simulationStartTime.

    Attributes:
        RawData (RawDataClass): Raw-data generation configuration (FMU, variables, sampling).
        dataset_prep (base_dataset_prep_class): Dataset slicing/filtering/transforms and split setup.
    """
    RawData: RawDataClass = field(default_factory=RawDataClass)
    dataset_prep: base_dataset_prep_class = field(default_factory=base_dataset_prep_class)

    @field_validator('dataset_prep')
    @classmethod
    def check_dataset_prep(cls, v, info: ValidationInfo):
        # check if initial_state_include is set to True, dataset_prep.start_time must be RawData.Solver.simulationStartTime
        if info.data['RawData'].initial_states_include == True:
            if v.start_time != info.data['RawData'].Solver.simulationStartTime:
                raise ValueError('dataset_prep.start_time must be RawData.Solver.simulationStartTime if RawData.initial_states_include is True')
        return v

"""data_gen config dataclass definitions"""
@dataclass
class data_gen_config:
    """
    Top-level configuration for dataset generation and worker setup.

    Notes:

    - memory_limit_per_worker expects strings like '2GiB'. A worker is restricted to this memory limit and restarts if exceeded, see https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster.
    - setting multiprocessing_processes to None lets the backend decide on the number of processes.

    Attributes:
        pModel (base_pModelClass): Physical-model configuration bundle to generate data for.
        multiprocessing_processes (Optional[int]): Number of worker processes; None lets backend choose.
        memory_limit_per_worker (str): Dask worker memory limit (e.g., '2GiB').
        overwrite (bool): Whether to overwrite existing generated artifacts. Should be False to avoid accidental data loss.
    """
    pModel: base_pModelClass = MISSING
    multiprocessing_processes: Optional[int] = None
    memory_limit_per_worker: str = "2GiB" # specifiy as 2GiB, or auto. see https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
    overwrite: bool = False

########################################################################################################################

"""base nn_model config dataclass definitions"""
@dataclass 
class base_network_class:
    """
    Common MLP backbone configuration shared by multiple models.

    Attributes:
        n_linear_layers (int): Number of linear layers in the backbone.
        linear_hidden_dim (int): Hidden dimension for linear layers.
        activation (str): String name of torch.nn activation module (e.g., 'torch.nn.ReLU'), is evaluated during model construction (using eval()).
    """
    n_linear_layers: int = 1
    linear_hidden_dim: int = 128
    activation: str = 'torch.nn.ReLU'

@dataclass
class base_training_settings_class:
    """
    Generic optimization and early-stopping hyperparameters.

    Attributes:
        batch_size (int): Training mini-batch size.
        max_epochs (int): Maximum number of epochs.
        lr_start (float): Initial learning rate.
        beta1_adam (float): Adam beta1 parameter.
        beta2_adam (float): Adam beta2 parameter.
        weight_decay (float): L2 weight decay.
        clip_grad_norm (float): Gradient clipping norm.
        early_stopping_patience (int): Patience before early stopping.
        early_stopping_threshold (float): Improvement threshold for early stopping.
        early_stopping_threshold_mode (str): Threshold mode, typically 'rel' or 'abs'.
        initialization_type (Optional[str]): Optional weight initialization scheme identifier.
    """
    batch_size: int = 64
    max_epochs: int = 30000
    lr_start: float = 0.001
    beta1_adam: float = 0.9
    beta2_adam: float = 0.999
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0
    early_stopping_patience: int = 2000
    early_stopping_threshold: float = 0.001
    early_stopping_threshold_mode: str = 'rel'
    initialization_type: Optional[str] = None

@dataclass
class abstract_nn_model_class:
    """
    Marker base class for neural-network configuration objects.
    """
    pass

@dataclass
class base_nn_model_class(abstract_nn_model_class):
    """
    Configuration for a simple feed-forward model with network and training settings.

    Attributes:
        network (base_network_class): Backbone network hyperparameters.
        training (base_training_settings_class): Training/optimization hyperparameters.
    """
    network: base_network_class = field(default_factory=base_network_class)
    training: base_training_settings_class = field(default_factory=base_training_settings_class)

"""pels vae config dataclass definitions"""
@dataclass
class pels_vae_network_class(base_network_class):
    """
    Network configuration for the PELS-VAE encoder/decoder.

    Gotchas:

    - activation must be the string of a valid torch.nn module (e.g., 'torch.nn.ReLU'); validation will fail otherwise.

    Attributes:
        n_latent (int): Size of latent space.
        n_linear_layers (int): Number of linear layers in encoder/decoder.
        linear_hidden_dim (int): Hidden dimension in linear layers.
        dropout_rate (float): Dropout rate applied within the network.
        activation (str): Activation module as a string (validated via eval).
        params_to_decoder (bool): Whether to pass parameters to the decoder.
        feed_forward_nn (bool): If True, use a feed-forward network instead of a VAE.
    """
    n_latent: int = 64
    n_linear_layers: int = 3
    linear_hidden_dim: int = 128
    dropout_rate: float = 0.0
    activation: str = 'torch.nn.ReLU'
    params_to_decoder: bool = False
    feed_forward_nn: bool = False # can switch to feed forward network instead of VAE

    @field_validator('activation')
    @classmethod
    def check_if_torch_module(cls,v):
        try:
            eval(v)
        except:
            raise ValueError('activation must be a torch.nn.Module, \
                             please use the string representation of the module, e.g. torch.nn.ReLU')
        return v

@dataclass
class pels_vae_training_settings_class(base_training_settings_class):
    """
    Training hyperparameters for the PELS-VAE model.

    Notes:

    - Capacity terms (capacity_*) are applied only when use_capacity is True.

    Attributes:
        lr_min (float): Minimum learning rate used by scheduler.
        lr_scheduler_plateau_gamma (float): Multiplicative factor of LR reduction on plateau.
        lr_scheduler_plateau_patience (int): Epochs to wait before reducing LR on plateau.
        lr_scheduler_threshold (float): Threshold for measuring new optimum, to only focus on significant changes.
        lr_scheduler_threshold_mode (str): 'rel' or 'abs' threshold mode for scheduler.
        init_bias (float): Initial bias for certain layers.
        beta_start (float): Starting beta for KL term annealing.
        clip_grad_norm (float): Larger clip norm typical for VAE stability.
        gamma (float): Extra scaling for reconstruction/regularization balance.
        use_capacity (bool): Enable capacity scheduling.
        capacity_patience (int): Patience before increasing capacity.
        capacity_start (float): Initial target capacity.
        capacity_increment (float): Step size for increasing capacity.
        capacity_increment_mode (str): 'abs' or 'rel' increment mode.
        capacity_threshold (float): Threshold to trigger capacity increase.
        capacity_threshold_mode (str): 'rel' or 'abs' threshold mode.
        capacity_max (float): Maximum allowed target capacity.
        count_populated_dimensions_threshold (float): Threshold to count active latent dimensions.
        n_passes_test (int): Number of stochastic passes during test.
        n_passes_train (int): Number of stochastic passes during training.
        test_from_regressor (bool): Evaluate using regressor pathway if available.
        test_with_zero_eps (bool): If True, set latent noise to zero during testing.
    """
    batch_size: int = 64
    lr_start: float = 1e-5
    lr_min: float = 1e-5
    lr_scheduler_plateau_gamma: float = 0.5
    lr_scheduler_plateau_patience: int = 1500
    lr_scheduler_threshold: float = 0.1
    lr_scheduler_threshold_mode: str = 'rel'
    weight_decay: float = 1e-4
    early_stopping_patience: int = 2000
    init_bias: float = 1e-3
    beta_start: float = 0.01
    clip_grad_norm: float = 1e6
    gamma: float = 1.0
    use_capacity: bool = False
    capacity_patience: int = 10
    capacity_start: float = 0.1
    capacity_increment: float = 0.05
    capacity_increment_mode: str = 'abs'
    capacity_threshold: float = 0.2
    capacity_threshold_mode: str = 'rel'
    capacity_max: float = 12.0
    count_populated_dimensions_threshold: float = 0.1
    n_passes_test: int = 1
    n_passes_train: int = 1
    test_from_regressor: bool = True
    test_with_zero_eps: bool = False

@dataclass
class neural_ode_network_class(base_network_class):
    """
    Network configuration for the neural ODE model.

    Attributes:
        linear_hidden_dim (int): Hidden dimension in dynamics MLP.
        hidden_dim_output_nn (int): Hidden dimension in the output head.
        n_layers_output_nn (int): Number of layers in the output head.
        activation (str): Activation module as a string (e.g., 'torch.nn.ELU').
        controls_to_output_nn (bool): If True, concatenate controls to the output head inputs.
    """
    linear_hidden_dim: int = 128
    hidden_dim_output_nn: int = 12
    n_layers_output_nn: int = 4
    activation: str = 'torch.nn.ELU'
    controls_to_output_nn: bool = False

@dataclass
class latent_ode_network_class(base_network_class):
    """
    Network configuration and linearization modes for the latent ODE model.

    Gotchas:

    - lat_ode_type must be one of ['variance_constant', 'variance_dynamic', 'vanilla'].
    - linear_mode must be one of [None, 'mpc_mode', 'mpc_mode_for_controls', 'deep_koopman'].
    - linear_mode choices set the *_linear flags as follows:
      'mpc_mode' -> all linear except state encoder; 'mpc_mode_for_controls' -> linear ODE/decoder and control encoder;
      'deep_koopman' -> only ODE linear.

    Attributes:
        n_linear_layers (int): Number of linear layers in encoders/decoder/ODE.
        linear_hidden_dim (int): Hidden dimension in encoders/decoder/ODE.
        activation (str): Activation module as a string (e.g., 'torch.nn.ELU'). ss evaluated during model construction (using eval()).
        controls_to_decoder (bool): Concatenate controls to decoder inputs.
        predict_states (bool): If True, predict states in addition to outputs. Default True.
        lat_ode_type (str): Variance handling mode for latent ODE. Must be in ['variance_constant', 'variance_dynamic', 'vanilla']. 'vanilla' generates a standard latent ODE. The other options implement BNODE as described in the paper.
        include_params_encoder (bool): Include parameter encoder.
        params_to_state_encoder (bool): Feed parameters to state encoder.
        params_to_control_encoder (bool): Feed parameters to control encoder.
        params_to_decoder (bool): Feed latent parameters to decoder.
        lat_state_mu_independent (bool): If True, state mean is independent from variance path. Only applicable for 'variance_dynamic' lat_ode_type.
        linear_mode (Optional[str]): Linearization preset; updates *_linear flags accordingly. Available options: None, 'mpc_mode', 'mpc_mode_for_controls', 'deep_koopman'. (see above)
        state_encoder_linear (bool): Force state encoder to be linear. Can be overridden by linear_mode.
        control_encoder_linear (bool): Force control encoder to be linear. Can be overridden by linear_mode.
        parameter_encoder_linear (bool): Force parameter encoder to be linear. Can be overridden by linear_mode.
        ode_linear (bool): Use a linear latent ODE. Can be overridden by linear_mode.
        decoder_linear (bool): Use a linear decoder. Can be overridden by linear_mode.
        lat_states_dim (int): Dimension of latent state space.
        lat_parameters_dim (int): Dimension of latent parameter space.
        lat_controls_dim (int): Dimension of latent control space.
    """
    n_linear_layers: int = 4
    linear_hidden_dim: int = 128
    activation: str = 'torch.nn.ELU'
    controls_to_decoder: bool = True
    predict_states: bool = True
    lat_ode_type: str = 'variance_constant' # must be in ['variance_constant', 'variance_dynamic', 'vanilla']
    include_params_encoder: bool = True
    params_to_state_encoder: bool = False
    params_to_control_encoder: bool = False
    params_to_decoder: bool = False
    
    lat_state_mu_independent: bool = False

    linear_mode: Optional[str] = None # must be one of None, 'mpc_mode', 'mpc_mode_for_controls', 'deep_koopman'
    state_encoder_linear: bool = False
    control_encoder_linear: bool = False
    parameter_encoder_linear: bool = False
    ode_linear: bool = False
    decoder_linear: bool = False

    lat_states_dim: int = 64
    lat_parameters_dim: int = 64
    lat_controls_dim: int = 64

    # check field lat_ode_type
    @field_validator('lat_ode_type')
    @classmethod
    def check_lat_ode_type(cls, v):
        types = ['variance_constant', 'variance_dynamic', 'vanilla']
        if v not in types:
            raise ValueError('lat_ode_type must be in ', types)
        return v
    
    @model_validator(mode='after')
    def model_validate_linear_mode(self):
        # This method will be called after field validation for the whole model
        # It should be registered as a model_validator in the class
        v = self.linear_mode
        if v is not None:
            if v == 'mpc_mode':
                self.state_encoder_linear = False
                self.control_encoder_linear = True
                self.parameter_encoder_linear = True
                self.ode_linear = True
                self.decoder_linear = True
                logging.info('Setting all linear modes to True except state_encoder for mpc_mode')
            elif v == 'mpc_mode_for_controls':
                self.state_encoder_linear = False
                self.control_encoder_linear = True
                self.parameter_encoder_linear = False
                self.ode_linear = True
                self.decoder_linear = True
                logging.info('Setting all linear modes to True except state_encoder and parameter_encoder for mpc_mode_for_controls')
            elif v == 'deep_koopman':
                self.state_encoder_linear = False
                self.control_encoder_linear = False
                self.parameter_encoder_linear = False
                self.ode_linear = True
                self.decoder_linear = False
                logging.info('Setting only ODE to linear for deep_koopman')
        return self

@dataclass
class base_neural_ode_pretraining_settings_class(base_training_settings_class):
    """
    Pretraining settings for neural ODE components prior to main training.

    Attributes:
        method (str): Pretraining method (e.g., 'collocation').
        batch_size (int): Mini-batch size during pretraining.
        batches_per_epoch (int): Number of batches per epoch during pretraining.
        max_epochs (int): Maximum pretraining epochs.
        lr_start (float): Initial learning rate for pretraining.
        weight_decay (float): Weight decay during pretraining.
        early_stopping_patience (int): Patience for early stopping in pretraining.
        early_stopping_threshold (float): Threshold for early stopping in pretraining.
        early_stopping_threshold_mode (str): Threshold mode ('rel' or 'abs').
        load_seq_len (Optional[int]): If provided, sequence length to load from dataset.
        seq_len_train (int): Sequence length used during pretraining.
    """
    method: str = 'collocation'
    batch_size: int = 1024
    batches_per_epoch: int = 12
    max_epochs: int = 100
    lr_start: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 50
    early_stopping_threshold: float = 0.001
    early_stopping_threshold_mode: str = 'rel'
    load_seq_len: Optional[int] = None
    seq_len_train: int = 10

@dataclass
class base_time_stepper_training_settings(base_training_settings_class):
    """
    Per-phase training settings for time-stepped training with ODE solvers.

    Gotchas:

    - If load_seq_len is not None (and not 0), then seq_len_train must be <= load_seq_len.
    - solver_norm must be either 'max' or 'mixed'; other values raise validation errors.
    - solver_step_size None means the solver uses its internal default step size.

    Attributes:
        evaluate_at_control_times (Optional[bool]): If True, solver is forced to restart at control change times.
        batches_per_epoch (Optional[int]): Number of batches per epoch for this phase. Default is 12.
        reload_optimizer (Optional[bool]): Recreate optimizer at phase start.
        load_seq_len (Optional[int]): Sequence length used to load model/optimizer state.
        seq_len_train (Optional[int]): Training sequence length for this phase.
        seq_len_increase_in_batches (Optional[int]): Increase sequence length every N batches.
        seq_len_increase_abort_after_n_stable_epochs (Optional[int]): Stop increasing after N stable epochs. A stable epoch is counted when loss_validation < 2 * loss_train
        use_adjoint (Optional[bool]): Use adjoint method for ODE gradients.
        solver (Optional[str]): ODE solver name (e.g., 'dopri5').
        solver_rtol (Optional[float]): Relative tolerance for adaptive step size solver.
        solver_atol (Optional[float]): Absolute tolerance for adaptive step size solver.
        solver_norm (Optional[str]): Norm used for adaptive step ('max' or 'mixed'). Default is 'mixed', that uses rmse per sample and max over batch.
        solver_step_size (Optional[float]): Fixed step size; None uses solver defaults. Only used for fixed-step solvers, should the dataset time step should be a multiple of this step size.
        break_after_loss_of (Optional[float]): Early break threshold on loss value.
        reload_model_if_loss_nan (bool): Reload last checkpoint if loss becomes NaN.
        activate_deterministic_mode_after_this_phase (bool): Activate deterministic latent dynamics after this phase.
        seq_len_epoch_start (Optional[int]): Internal tracker for the starting sequence length of this phase.
    """
    evaluate_at_control_times: Optional[bool] = False
    batches_per_epoch: Optional[int] = 12
    reload_optimizer: Optional[bool] = False
    load_seq_len: Optional[int] = None
    seq_len_train: Optional[int] = None
    seq_len_increase_in_batches: Optional[int] = 0
    seq_len_increase_abort_after_n_stable_epochs: Optional[int] = 10
    use_adjoint: Optional[bool] = True
    solver: Optional[str] = 'dopri5'
    solver_rtol: Optional[float] = 1e-3
    solver_atol: Optional[float] = 1e-4
    solver_norm: Optional[str] = 'mixed' # max or mixed
    solver_step_size: Optional[float] = None # if None, the solver will use default settings
    break_after_loss_of: Optional[float] = None
    reload_model_if_loss_nan: bool = True # should be always True, only set to false e.g. for writing iclr paper
    activate_deterministic_mode_after_this_phase: bool = False # is only used for B-NODE, but for compatibility reasons it is included here

    seq_len_epoch_start: Optional[int] = None # only used internally, does not need to be set. But this can be set
    @field_validator('seq_len_train')
    @classmethod
    def check_seq_len_train_start(cls, v, info: ValidationInfo):
        if info.data['load_seq_len'] != None:
            if v > info.data['load_seq_len']:
                if info.data['load_seq_len'] == 0:
                    pass
                else:
                    raise ValueError('seq_len_train must be smaller than load_seq_len')
        return v
    @field_validator('solver_norm')
    @classmethod
    def check_adaptive_step_size_norm(cls, v, info: ValidationInfo):
        if v not in ['max', 'mixed']:
            raise ValueError('solver_norm must be max or mixed')
        return v


@dataclass
class base_neural_ode_training_settings_class():
    """
    High-level training configuration and per-phase override mechanism for ODE models.

    Gotchas:

    - Any *_override set here is propagated into each phase in main_training and may override non-defaults (a warning is logged).
    - There is no override for fields that are intended to apply to single phases only (e.g., break_after_loss_of).
    - main_training holds a sequence of phases executed in order. These are validated by the dataclass "base_time_stepper_training_settings".

    Attributes:
        pre_train (bool): If True, run a pretraining stage before main training.
        load_pretrained_model (bool): Load a pretrained model before training.
        load_trained_model_for_test (bool): Load a fully trained model and run testing only.
        save_predictions_in_dataset (bool): Save predictions back into the dataset after testing.
        test (bool): Enable test pass after training.
        test_save_internal_variables (bool): Save internal variables to dataset during testing.
        test_save_internal_variables_for (str): Save internal variables for this dataset split during testing. (e.g., 'common_test')
        pre_trained_model_seq_len (Optional[int]): Sequence length of the pretrained checkpoint to load.
        path_pretrained_model (Optional[str]): Path to pretrained weights. Can also be copied from mlflow web UI.
        path_trained_model (Optional[str]): Path to trained model weights for testing. Can also be copied from mlflow web UI.
        batch_size_test (int): Batch size to use during testing.
        initialization_type (Optional[str]): Weight initialization scheme for NN. Options: 'xavier', none.
        initialization_type_ode (Optional[str]): Initialization scheme for ODE-specific components. Options: 'xavier', none.
        ***_override (various): Overrides for training hyperparameters applied to all phases in main_training.
        main_training (List[base_time_stepper_training_settings]): Sequence of per-phase settings.
    """
    pre_train: bool = False
    load_pretrained_model: bool = False
    load_trained_model_for_test: bool = False
    save_predictions_in_dataset: bool = True
    test: bool = True
    test_save_internal_variables: bool = False
    test_save_internal_variables_for: str = 'common_test'
    pre_trained_model_seq_len: Optional[int] = None
    path_pretrained_model: Optional[str] = None
    path_trained_model: Optional[str] = None 

    batch_size_test: int = 48
    initialization_type: Optional[str] = None
    initialization_type_ode: Optional[str] = None

    batch_size_override: Optional[int] = None
    batches_per_epoch_override: Optional[int] = None
    max_epochs_override: Optional[int] = None
    lr_start_override: Optional[float] = None
    beta1_adam_override: Optional[float] = None
    beta2_adam_override: Optional[float] = None
    weight_decay_override: Optional[float] = None
    clip_grad_norm_override: Optional[float] = None
    early_stopping_patience_override: Optional[int] = None
    early_stopping_threshold_override: Optional[float] = None
    early_stopping_threshold_mode_override: Optional[str] = None

    reload_optimizer_override: Optional[bool] = None    
    solver_override: Optional[str] = None
    load_seq_len_override: Optional[int] = None
    seq_len_train_override: Optional[int] = None
    seq_len_increase_in_batches_override: Optional[int] = None
    seq_len_increase_abort_after_n_stable_epochs_override: Optional[int] = None
    use_adjoint_override: Optional[bool] = None
    evaluate_at_control_times_override: Optional[bool] = None
    solver_rtol_override: Optional[float] = None
    solver_atol_override: Optional[float] = None
    solver_step_size_override: Optional[float] = None
    solver_norm_override: Optional[str] = None
    # no override for break_after_loss_of as this should only used for one training phase    pre_training: base_neural_ode_pretraining_settings_class = field(default_factory=base_neural_ode_pretraining_settings_class)
    main_training: List[base_time_stepper_training_settings] = field(default_factory=lambda: [base_time_stepper_training_settings()])

    @field_validator('main_training')
    @classmethod
    def set_overrides(cls, v, info: ValidationInfo):
        default_class = base_time_stepper_training_settings()
        for i, training_settings in enumerate(v):
            for key in ['batch_size_override', 'batches_per_epoch_override', 'max_epochs_override', 'lr_start_override',
                        'beta1_adam_override', 'beta2_adam_override', 'clip_grad_norm_override', 
                        'weight_decay_override', 'early_stopping_patience_override', 'early_stopping_threshold_override', 
                        'early_stopping_threshold_mode_override', 'reload_optimizer_override','solver_override', 'load_seq_len_override', 
                        'seq_len_train_override', 'use_adjoint_override', 'evaluate_at_control_times_override','solver_rtol_override', 'solver_atol_override', 'solver_step_size_override',
                        'seq_len_increase_in_batches_override', 'seq_len_increase_abort_after_n_stable_epochs_override', 'solver_norm_override']:
                if info.data[key] is not None:
                    # print warning if override is set and non-default value is used
                    if v[i].__getattribute__(key.split('_override')[0]) != default_class.__getattribute__(key.split('_override')[0]):
                        logging.warning(f'Overriding {key.split("_override")[0]} with value {info.data[key]}')
                    v[i].__setattr__(key.split('_override')[0], info.data[key])
        return v

@dataclass
class base_ode_nn_model_class(abstract_nn_model_class):
    """
    Wrapper that binds a network definition to neural ODE training settings.

    Attributes:
        network (base_network_class): NN backbone used for the ODE model.
        training (base_neural_ode_training_settings_class): ODE training schedule and overrides.
    """
    network: base_network_class = field(default_factory=base_network_class)
    training: base_neural_ode_training_settings_class = field(default_factory=base_neural_ode_training_settings_class)

#latent ode
@dataclass
class latent_timestepper_training_settings(base_time_stepper_training_settings):
    """
    Time-stepper training settings extended with latent-ODE-specific losses and counters.

    Notes:

    - multi_shooting_condition_multiplier controls the strength of multi-shooting consistency; 10.0 - 1.0 is a typical value.

    Attributes:
        beta_start (float): Initial beta for KL term.
        alpha_mu (float): Weight for addes std nois on mean (mu) term in model evaluation. Should be 1.0 in all cases.
        alpha_sigma (float): Weight for added std noise (sigma) term in model evaluation. Typically small (e.g., 0.001), and only used when lat_ode_type is 'variance_dynamic'.
        n_passes (int): Number of passes per batch/epoch for stochastic evaluation. Default 1.
        threshold_count_populated_dimensions (float): Threshold to count active latent dims.
        include_reconstruction_loss_state0 (bool): Include reconstruction loss at initial state.
        include_reconstruction_loss_outputs0 (bool): Include reconstruction loss at initial outputs.
        include_reconstruction_loss_state_der (bool): Include loss on state derivatives. Adds a "state_der_decoder". Deprecated.
        include_states_grad_loss (bool): Include gradient matching loss for states.
        include_outputs_grad_loss (bool): Include gradient matching loss for outputs.
        multi_shooting_condition_multiplier (float): Weight for multi-shooting consistency term. Should be somewhere between 1.0 and 10.0 if used.
    """
    beta_start: float = 0.001
    alpha_mu: float = 1.0
    alpha_sigma: float = 0.001
    n_passes: int = 1
    threshold_count_populated_dimensions: float = 0.1
    include_reconstruction_loss_state0: bool = False
    include_reconstruction_loss_outputs0: bool = False
    include_reconstruction_loss_state_der: bool = False
    include_states_grad_loss: bool = True
    include_outputs_grad_loss: bool = False
    multi_shooting_condition_multiplier: float = 0.0 # 10.0 seems like a good value

@dataclass
class base_latent_ode_training_settings_class:
    """
    Training configuration and overrides for latent ODE models with multiple training phases.

    Gotchas:

    - Overrides (e.g., beta_start_override, n_passes_override, solver_*) are copied into each phase of main_training. A warning is logged if
      the override changes a non-default value. Variables that are intended to apply to single phases only (e.g., break_after_loss_of) do not have overrides.
    - At most one phase may set activate_deterministic_mode_after_this_phase=True (enforced in validation);
      consider using alpha_mu >= 1.0 when enabling deterministic mode.

    Attributes:
        pre_train (bool): If True, perform a pretraining stage.
        load_pretrained_model (bool): Load a pretrained model before training.
        load_trained_model_for_test (bool): Load a trained model and run tests only.
        save_predictions_in_dataset (bool): Save predictions back into the dataset on test.
        test (bool): Enable a post-training test run.
        test_save_internal_variables (bool): Store internal variables during test for analysis.
        test_save_internal_variables_for (str): Label for the stored internal variables.
        pre_trained_model_seq_len (Optional[int]): Sequence length used by the pretrained checkpoint.
        path_pretrained_model (Optional[str]): Path to pretrained model. Can also be copied from mlflow web UI.
        path_trained_model (Optional[str]): Path to trained model for testing. Can also be copied from mlflow web UI.
        batch_size_test (int): Test-time batch size.
        initialization_type (Optional[str]): Weight initialization scheme for NN. Options: 'xavier', none.
        initialization_type_ode (Optional[str]): Initialization scheme for ODE parts. Options: 'xavier', 'move_eigvals_matrix' (only for linear ode), 'move_eigvals_net', none.
        initialization_type_ode_matrix (Optional[str]): Initialization for ODE matrices if applicable.
        ***_override (various): See *_override fields to broadcast settings into each main training phase.
        pre_training (base_neural_ode_pretraining_settings_class): Settings for the pretraining stage.
        main_training (List[latent_timestepper_training_settings]): Sequence of latent ODE training phases.
    """
    pre_train: bool = False
    load_pretrained_model: bool = False
    load_trained_model_for_test: bool = False
    save_predictions_in_dataset: bool = True
    test: bool = True
    test_save_internal_variables: bool = False
    test_save_internal_variables_for: str = 'common_test'
    pre_trained_model_seq_len: Optional[int] = None
    path_pretrained_model: Optional[str] = None
    path_trained_model: Optional[str] = None 

    batch_size_test: int = 48
    initialization_type: Optional[str] = None
    initialization_type_ode: Optional[str] = None
    initialization_type_ode_matrix: Optional[str] = None

    batch_size_override: Optional[int] = None
    batches_per_epoch_override: Optional[int] = None
    max_epochs_override: Optional[int] = None
    lr_start_override: Optional[float] = None
    beta1_adam_override: Optional[float] = None
    beta2_adam_override: Optional[float] = None
    weight_decay_override: Optional[float] = None
    clip_grad_norm_override: Optional[float] = None
    early_stopping_patience_override: Optional[int] = None
    early_stopping_threshold_override: Optional[float] = None
    early_stopping_threshold_mode_override: Optional[str] = None

    reload_optimizer_override: Optional[bool] = None  
    solver_override: Optional[str] = None
    load_seq_len_override: Optional[int] = None
    seq_len_train_override: Optional[int] = None
    seq_len_increase_in_batches_override: Optional[int] = None
    seq_len_increase_abort_after_n_stable_epochs_override: Optional[int] = None
    use_adjoint_override: Optional[bool] = None
    evaluate_at_control_times_override: Optional[bool] = None
    solver_rtol_override: Optional[float] = None
    solver_atol_override: Optional[float] = None
    solver_step_size_override: Optional[float] = None
    solver_norm_override: Optional[str] = None
    # no override for break_after_loss_of as this should only used for one training phase
    # no override for activate_deterministic_mode_after_this_phase as this should only used for one training phase

    # additional to base_neural_ode_training_settings_class
    beta_start_override: Optional[float] = None
    alpha_mu_override: Optional[float] = None
    alpha_sigma_override: Optional[float] = None
    n_passes_override: Optional[int] = None
    threshold_count_populated_dimensions_override: Optional[float] = None

    include_reconstruction_loss_state0_override: Optional[bool] = None
    include_reconstruction_loss_outputs0_override: Optional[bool] = None
    include_states_grad_loss_override: Optional[bool] = None
    include_outputs_grad_loss_override: Optional[bool] = None
    multi_shooting_condition_multiplier_override: Optional[float] = None

    pre_training: base_neural_ode_pretraining_settings_class = field(default_factory=base_neural_ode_pretraining_settings_class)
    main_training: List[latent_timestepper_training_settings] = field(
        default_factory=lambda: [latent_timestepper_training_settings()]
    )

    @field_validator('main_training')
    @classmethod
    def set_overrides(cls, v, info: ValidationInfo):
        default_class = latent_timestepper_training_settings()
        for i, training_settings in enumerate(v):
            for key in ['batch_size_override', 'batches_per_epoch_override', 'max_epochs_override', 'lr_start_override', 
                        'beta1_adam_override', 'beta2_adam_override', 'clip_grad_norm_override', 
                        'weight_decay_override', 'early_stopping_patience_override', 'early_stopping_threshold_override', 
                        'early_stopping_threshold_mode_override', 'reload_optimizer_override','solver_override', 'load_seq_len_override', 
                        'seq_len_train_override', 'use_adjoint_override', 'evaluate_at_control_times_override', 'solver_rtol_override', 'solver_atol_override', 'solver_step_size_override',
                        'beta_start_override', 'alpha_mu_override', 'alpha_sigma_override',
                        'n_passes_override', 'threshold_count_populated_dimensions_override',
                        'include_reconstruction_loss_state0_override', 'include_reconstruction_loss_outputs0_override',
                        'multi_shooting_condition_multiplier_override',
                        'seq_len_increase_in_batches_override', 'seq_len_increase_abort_after_n_stable_epochs_override', 'solver_norm_override',
                        'include_states_grad_loss_override', 'include_outputs_grad_loss_override']:
                if info.data[key] is not None:
                    # print warning if override is set and non-default value is used
                    if v[i].__getattribute__(key.split('_override')[0]) != default_class.__getattribute__(key.split('_override')[0]):
                        logging.warning(f'Overriding {key.split("_override")[0]} with value {info.data[key]}')
                    v[i].__setattr__(key.split('_override')[0], info.data[key])
        return v
    
    # make sure that activate_deterministic_mode_after_this_phase is only set for one phase
    @field_validator('main_training')
    @classmethod
    def check_deterministic_mode(cls, v, info: ValidationInfo):
        n = 0
        for i, training_settings in enumerate(v):
            if training_settings.activate_deterministic_mode_after_this_phase:
                n += 1
                if training_settings.alpha_mu < 1.0:
                    logging.warning('alpha_mu should be larger than some value, e.g. 1.0, when activating deterministic mode')
                    logging.warning('otherwise excess states in latent ode are not learned to be ignored and the model fails')
        if n > 1:
            raise ValueError('Only one phase can have activate_deterministic_mode_after_this_phase set to True')
        
        return v

@dataclass
class base_latent_ode_nn_model_class(abstract_nn_model_class):
    """
    Wrapper that binds the latent ODE network to its training configuration.

    Attributes:
        network (latent_ode_network_class): Latent ODE network hyperparameters.
        training (base_latent_ode_training_settings_class): Latent ODE training configuration.
    """
    network: latent_ode_network_class = field(default_factory=latent_ode_network_class)
    training: base_latent_ode_training_settings_class = field(default_factory=base_latent_ode_training_settings_class)


"""train config dataclass definition"""
@dataclass
class train_test_config_class:
    """
    Runtime configuration for training/testing, hardware usage, and MLflow tracking.

    Attributes:
        nn_model (abstract_nn_model_class): Model configuration (network + training) to run.
        dataset_name (str): Name of the dataset configuration to use.
        mlflow_tracking_uri (str): MLflow tracking server URI. Defaults to localhost.
        mlflow_experiment_name (str): MLflow experiment name.
        use_amp (bool): Enable automatic mixed precision. Should not be used for NODE/BNODE models.
        use_cuda (bool): Use CUDA if available.
        raise_exception (bool): If True, re-raise exceptions for debugging. Otherwise, log and continue.
        batch_print_interval (int): Interval (in batches) for logging training progress.
        verbose (bool): Enable verbose logging.
        n_workers_train_loader (int): Number of workers for the training dataloader.
        n_workers_other_loaders (int): Number of workers for validation/test loaders.
        prefetch_factor (int): Prefetch factor for dataloaders.
    """
    nn_model: abstract_nn_model_class = MISSING
    dataset_name: str = MISSING
        
    mlflow_tracking_uri: str = 'http://127.0.0.1:5000'
    mlflow_experiment_name: str = 'Default'
    
    use_amp: bool = False
    use_cuda: bool = True
    raise_exception: bool = True
    
    batch_print_interval: int = 5
    verbose: bool = False
    n_workers_train_loader: int = 5
    n_workers_other_loaders: int = 1
    prefetch_factor: int = 2

'''ONNX export dataclass definition'''
@dataclass
class load_latent_ode_config_class:
    """
    Configuration for loading a trained latent ODE model and its associated artifacts.

    Gotchas:

    - Provide either mlflow_run_id or model_directory; at least one must be set.

    Attributes:
        mlflow_tracking_uri (str): MLflow tracking server URI. Defaults to localhost.
        model_directory (Optional[str]): Local directory containing a model to load.
        mlflow_run_id (Optional[str]): MLflow run ID to fetch the model from tracking.
        model_checkpoint_path (Optional[str]): Direct path to a model checkpoint file.
        config_path (Optional[str]): Path to a saved Hydra config to reproduce settings.
        dataset_path (Optional[str]): Path to a dataset used during export/evaluation.
    """
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_directory: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    model_checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None # should be the dataset saved from hydra, as we do not validate the dataclass
    dataset_path: Optional[str] = None 

    @field_validator('mlflow_run_id')
    @classmethod
    def check_deterministic_mode(cls, v, info: ValidationInfo):
        if (v == None) and (info.data['model_directory'] == None):
            raise ValueError('Either the mlflow run id or the model directory must be provided!')
        return v
    pass

@dataclass
class onnx_export_config_class(load_latent_ode_config_class):
    """
    Settings for exporting a latent ODE model to ONNX format.

    Attributes:
        output_dir (Optional[str]): Output directory for the exported ONNX model and assets.
    """
    output_dir: Optional[str] = None
    pass


def get_config_store() -> ConfigStore:
    """
    Registers all configuration dataclasses with Hydra's ConfigStore.
    
    Args: 
        None

    Returns:

        cs: ConfigStore instance with registered configurations.    
    """
    cs = ConfigStore.instance()
    cs.store(name='base_data_gen', node=data_gen_config)
    cs.store(group='pModel', name='base_pModel', node=base_pModelClass)

    cs.store(name='base_train_test', node=train_test_config_class)
    cs.store(group='nn_model', name='base_nn_model', node=base_nn_model_class)
    #pels_vae_linear
    cs.store(group='nn_model', name='pels_vae', 
            node = base_nn_model_class(
                network=pels_vae_network_class(), 
                training=pels_vae_training_settings_class()
            ),
        )
    # neural_ode
    cs.store(group='nn_model', name='neural_ode_base',
                node = base_ode_nn_model_class(
                    network=neural_ode_network_class(),
                    training=base_neural_ode_training_settings_class()
                ),
            )
    cs.store(group='nn_model', name='latent_ode_base',
                node = base_latent_ode_nn_model_class(
                    network=latent_ode_network_class(),
                    training=base_latent_ode_training_settings_class()
                ),
            )

    # onnx export
    cs.store(name='base_onnx_export', node=onnx_export_config_class)
    return cs

########################################################################################################################
# Config utility functions
#########################################################################################################################


def convert_cfg_to_dataclass(cfg: DictConfig) -> dataclass:
    '''
    Converts a hydra config object to a dataclass
    
    Args:
        cfg: hydra config object / that is omegaconf.dictconfig.DictConfig
    
    Returns:
        cfg: dataclass
    '''
    logging.info('Validating config...')
    cfg = OmegaConf.to_object(cfg)
    logging.info('Validatied config and converted to dataclass')
    return cfg

def save_dataclass_as_yaml(cfg: dataclass, path: str):
    '''
    Saves a dataclass as a yaml file
    
    Args:
        cfg: dataclass
        path: path to save yaml file
    '''
    with open(path, 'w') as f:
        yaml.dump(asdict(cfg), f)