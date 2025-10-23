"""
This file contains all the dataclass definitions for the config files
and the config store.
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
import torch
from pydantic import model_validator

########################################################################################################################
# Dataclasses
########################################################################################################################

"""pModel config dataclass definitions"""
@dataclass
class SolverClass:
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
class base_pModel_test_class:
    plot_variables: List[str] = field(default_factory=list)

@dataclass
class base_pModelClass:
    RawData: RawDataClass = field(default_factory=RawDataClass)
    dataset_prep: base_dataset_prep_class = field(default_factory=base_dataset_prep_class)
    test_nn_model: Optional[base_pModel_test_class] = field(default_factory=base_pModel_test_class)

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
    pModel: base_pModelClass = MISSING
    multiprocessing_processes: Optional[int] = None
    memory_limit_per_worker: str = "2GiB" # specifiy as 2GiB, or auto. see https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
    overwrite: bool = False

########################################################################################################################

"""base nn_model config dataclass definitions"""
@dataclass 
class base_network_class:
    n_linear_layers: int = 1
    linear_hidden_dim: int = 128
    activation: str = 'torch.nn.ReLU'

@dataclass
class base_training_settings_class:
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
    pass

@dataclass
class base_nn_model_class(abstract_nn_model_class):
    network: base_network_class = field(default_factory=base_network_class)
    training: base_training_settings_class = field(default_factory=base_training_settings_class)

"""pels vae config dataclass definitions"""
@dataclass
class pels_vae_network_class(base_network_class):
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
    linear_hidden_dim: int = 128
    hidden_dim_output_nn: int = 12
    n_layers_output_nn: int = 4
    activation: str = 'torch.nn.ELU'
    controls_to_output_nn: bool = False

@dataclass
class latent_ode_network_class(base_network_class):
    n_linear_layers: int = 4
    linear_hidden_dim: int = 128
    hidden_dim_output_nn: int = 12
    activation: str = 'torch.nn.ELU'
    controls_to_decoder: bool = True
    predict_states: bool = True
    lat_ode_type: str = 'variance_constant' # must be in ['variance_constant', 'variance_dynamic', 'vanilla']
    include_params_encoder: bool = True
    params_to_state_encoder: bool = False
    params_to_control_encoder: bool = False
    params_to_decoder: bool = False
    
    lat_state_mu_independent: bool = False

    koopman_mpc_mode: Optional[bool] = None
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
        koopman_mpc_mode = self.koopman_mpc_mode
        if koopman_mpc_mode is not None:
            logging.warning('koopman_mpc_mode is deprecated, please use linear_mode instead')
            logging.warning('overwriting linear_mode with koopman_mpc_mode')
            if koopman_mpc_mode is True and v is not None:
                raise ValueError('linear_mode must be None if koopman_mpc_mode is True')
            elif koopman_mpc_mode is True and v is None:
                v = 'mpc_mode'
                self['linear_mode'] = v
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
    # as this is loaded as a list in neural_ode_training_settings_class, 
    # the type must be Optional for proper validation
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
    solver_norm: Optional[str] = 'max' # max or mixed
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
    network: base_network_class = field(default_factory=base_network_class)
    training: base_neural_ode_training_settings_class = field(default_factory=base_neural_ode_training_settings_class)

#latent ode
@dataclass
class latent_timestepper_training_settings(base_time_stepper_training_settings):
    beta_start: float = 0.001
    alpha_mu: float = 1.0
    alpha_sigma: float = 0.001
    n_passes: int = 1
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
    network: latent_ode_network_class = field(default_factory=latent_ode_network_class)
    training: base_latent_ode_training_settings_class = field(default_factory=base_latent_ode_training_settings_class)


"""train config dataclass definition"""
@dataclass
class train_test_config_class:
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
    output_dir: Optional[str] = None
    pass
    
    
    
########################################################################################################################
# Config Store
########################################################################################################################
def get_config_store():
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


def convert_cfg_to_dataclass(cfg: DictConfig):
    '''
    Converts a hydra config object to a dataclass
    
    Args:
        cfg: hydra config object / that is omegaconf.dictconfig.DictConfig
    
    Returns:
        cfg1: dataclass
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
    