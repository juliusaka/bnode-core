"""Neural ODE and Balanced Neural ODE Training Module.

This module provides the main training pipeline for Neural ODE (NODE) and Balanced
Neural ODE (BNODE) models. It handles model initialization, multi-phase training,
validation, testing, and MLflow experiment tracking.

Architecture Support
--------------------
The trainer automatically detects and supports two model architectures:

- **Neural ODE (NODE)**: Direct neural differential equation models.
- **Balanced Neural ODE (BNODE)**: Latent-space ODE models with encoder-decoder
  architecture for improved training stability and representation learning.

Training Pipeline Overview
--------------------------
The training process follows these stages:

1. **Model Instantiation**
    - Automatically detects NODE vs BNODE from config
    - Initializes normalization layers using dataset statistics
    - Sets up device (CPU/CUDA) based on availability and config

2. **Pre-training (Optional, NODE only)**
    - Can be enabled in config: ``nn_model.training.pre_train=true``
    - Trains on state derivatives (``state_der``) if present in dataset
    - Uses collocation method for initial parameter estimation
    - **Not supported for BNODE models** (No latent states gradients available, 
      but you can mock this behavior by using a short main training phase with 
      states_grad_loss)

3. **Multi-Phase Main Training**
    - Configured as a list in ``nn_model.training.main_training``
    - Each phase can have different hyperparameters:
        - Solver type (euler, rk4, dopri5, etc.)
        - Learning rate, batch size, sequence length
        - Early stopping patience and threshold
    - See ``resources/config/nn_model/bnode_pytest.yaml`` for an example

4. **Final Testing**
    - Evaluates model on all dataset splits (train/val/test)
    - Optionally saves predictions and internal variables to dataset
    - Logs final metrics to MLflow

Key Training Features
---------------------

***Compatibility with NODE and BNODE***

- Trainer auto-detects model type from config
- Both models provide a consistent training interface with 
  e.g. the `model_and_loss_evaluation` method.

**Adaptive Batch Processing**

Each epoch processes a specified number of batches (not entire dataset).
Configured via ``nn_model.training.main_training[i].batches_per_epoch``.

**NaN Recovery**

- If NaN loss detected, automatically reloads last checkpoint
- Reduces gradient clipping norm to stabilize training
- Note: LR scheduling might be a better long-term solution

**Reparameterization Control (BNODE)**

- Training uses active reparameterization (variational inference)
- When evaluating (validation/test, or at final test for all datasets), 
  reparameterization is disabled. Also for deterministic mode.
- Ensures consistent evaluation metrics

**Progressive Sequence Length Increase**

- When switching phases, sequence length gradually increases
- Initial test with final sequence length to assess extrapolation
- Training sequence length increases gradually (controlled by
    ``seq_len_increase_in_batches``)
- Validation/test always use full sequence length to monitor extrapolation performance
- Early abort if stable extrapolation achieved:
    ``loss_train < 2 * loss_validation`` for N consecutive epochs
    (``seq_len_increase_abort_after_n_stable_epochs``)

**MLflow Integration**

- Logs metrics at end of each phase: ``{metric}_{context}_job{phase}_final``
- Final test metrics logged as: ``{metric}_final``
- All Hydra outputs and trained models saved as artifacts
- Experiment tracking with run name, parameters, and tags

Typical Usage Examples
----------------------

As other modules of the ``bnode_core`` package, we use Hydra for configuration management.

Basic training with default config:

    uv run trainer nn_model=latent_ode_base dataset_name=myDataset

Training with custom model configuration:

    uv run trainer nn_model=myCustomModel dataset_name=myDataset \\
        mlflow_experiment_name=my_experiment \\
        nn_model.network.lat_states_dim=1024 \\

Hyperparameter sweep (multi-run mode):

    uv run trainer \\
        nn_model=latent_ode_base \\
        dataset_name=myDataset \\
        nn_model.training.beta_start_override=0.1,0.01,0.001 \\
        -m

Override specific training parameters:

    uv run trainer \\
        nn_model=latent_ode_base \\
        dataset_name=myDataset \\
        nn_model.training.lr_start_override=1e-4 \\
        nn_model.training.batch_size_override=512 \\
        use_cuda=false

View available configuration options (from Hydra):

    uv run trainer --help

Configuration
-------------
For detailed configuration options, see:

- **Config Documentation**: Consult the Config section of the documentation
- **Config Files**: examples in ``resources/config/nn_model/`` directory
- **Config Schema**: ``bnode_core.config`` module for all available parameters
- **Search Tip**: Use Ctrl+F in config files to find specific parameter behavior

Command Line Interface
----------------------
The trainer is registered as a UV script in ``pyproject.toml``, enabling direct
execution via ``uv run trainer``. All Hydra config parameters can be overridden
via command line using dot notation.

Notes
-----
- CUDA is automatically used if available (override with ``use_cuda=false``)
- Model checkpoints saved after each phase: ``model_phase_{i}.pt``
- Failed artifact logging tracked in ``could_not_log_artifacts.txt``
- Supports mixed precision training (AMP) when enabled
- Early stopping based on validation loss with configurable patience

See Also
--------

bnode_core.config : Configuration schemas and validation
bnode_core.ode.node.node_architecture : Neural ODE model implementation
bnode_core.ode.bnode.bnode_architecture : Balanced Neural ODE model implementation
bnode_core.nn.nn_utils.load_data : Dataset loading utilities




"""
import torch
import hydra
from pathlib import Path
import numpy as np
import logging
import shutil
import h5py
import time as pyTime
import copy

from h5py import Dataset as hdf5_dataset_class
from torch.nn.utils import clip_grad_norm_
from torch.optim import LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import bnode_core.filepaths as filepaths
from bnode_core.ode.node.node_architecture import NeuralODE
from bnode_core.ode.bnode.bnode_architecture import BalancedNeuralODE, build_feedthrough_mask
from bnode_core.nn.nn_utils.lr_scheduler import lr_on_plateau_iterations_to_min_lr

from bnode_core.nn.nn_utils.load_data import (
    load_dataset_and_config,
    make_stacked_dataset,
    TimeSeriesDataset,
    timeseries_collate_fn,
)
from bnode_core.nn.nn_utils.early_stopping import EarlyStopping
from bnode_core.config import train_test_config_class, base_training_settings_class, get_config_store

from bnode_core.utils.hydra_mlflow_decorator import log_hydra_to_mlflow
from bnode_core.utils.mlflow_proxy import mlflow_proxy
from bnode_core.ode.trainer_utils.restart_state import (
    CheckpointRequestedExit,
    InnerTrainingStateCheckpoint,
    LiveTrainingState,
    OuterTrainingState,
)
from bnode_core.ode.trainer_utils.restart_utils import (
    _clear_restart_state,
    _load_outer_training_state,
)


torch.backends.cudnn.benchmark = True


def _get_early_stopping_corresponding_metric(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    for metric_name in ('rmse_states_outputs', 'rmse_states', 'rmse_outputs'):
        metric_value = metrics.get(metric_name)
        if metric_value is not None:
            return metric_name, metric_value
    return None, None


def initialize_model(cfg: train_test_config_class, train_dataset: TimeSeriesDataset, hdf5_dataset: hdf5_dataset_class, 
                     initialize_normalization=True):
    """Initialize and configure NODE or BNODE model with dataset statistics.
    
    Automatically detects model type from config and initializes normalization
    layers using training dataset statistics. Handles device placement (CPU/CUDA)
    and copies model architecture file to Hydra output directory.
    
    Args:
        cfg (train_test_config_class): Validated Hydra configuration.
        train_dataset (TimeSeriesDataset): Training dataset for normalization.
        hdf5_dataset (hdf5_dataset_class): HDF5 dataset handle for statistics.
        initialize_normalization (bool, optional): Whether to initialize normalization
            layers from dataset statistics. Defaults to True.
        model_type (str, optional): Force specific model type ('node' or 'bnode').
            If None, auto-detects from config. Defaults to None.
    
    Returns:
        model (torch.nn.Module): Initialized model (NeuralODE or BalancedNeuralODE) moved
            to appropriate device.
    
    Side Effects:
        - Modifies cfg.use_cuda based on availability
        - Copies model architecture source file to Hydra output directory
        - Logs device and parameter count information
    
    Notes:
        - CUDA is used if available and cfg.use_cuda=True
        - Normalization uses training set statistics only
        - Model type detection based on network class in config
    """
    _cuda_available = torch.cuda.is_available()
    logging.info('CUDA available: {} | cfg.use_cuda: {}'.format(_cuda_available, cfg.use_cuda))
    if _cuda_available and cfg.use_cuda:
        cfg.use_cuda = True
    else:
        cfg.use_cuda = False
    logging.info("---> Training with cuda: {}".format(cfg.use_cuda))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
    # create model (insert specific creations here)
    if cfg.nn_model.model_type == 'node':
            model_type='node'
    elif cfg.nn_model.model_type == 'bnode':
            model_type='bnode'
    else: 
        raise ValueError(f"Unknown nn_model.model_type: '{cfg.nn_model.model_type}'. Expected 'node' or 'bnode'.")
    assert model_type in ['node', 'bnode']
    if model_type == 'node':
        model = NeuralODE(states_dim=train_dataset[0]['states'].shape[0],
                        controls_dim=train_dataset[0]['controls'].shape[0] if 'controls' in train_dataset[0].keys() else 0,
                        parameters_dim=train_dataset[0]['parameters'].shape[0] if 'parameters' in train_dataset[0].keys() else 0,
                        outputs_dim=train_dataset[0]['outputs'].shape[0] if 'outputs' in train_dataset[0].keys() else 0,
                        controls_to_output_nn=cfg.nn_model.network.controls_to_output_nn,
                        hidden_dim=cfg.nn_model.network.linear_hidden_dim, 
                        n_layers=cfg.nn_model.network.n_linear_layers,
                        hidden_dim_output_nn=cfg.nn_model.network.hidden_dim_output_nn,
                        n_layers_output_nn=cfg.nn_model.network.n_layers_output_nn,
                        activation=eval(cfg.nn_model.network.activation),
                        intialization=cfg.nn_model.training.pre_training.initialization_type,
                        initialization_ode=cfg.nn_model.training.initialization_type_ode,
                        use_input_smoother=cfg.nn_model.training.use_input_smoother,
                        )
        # initialize normalizations
        if initialize_normalization:
            model.normalization_init(hdf5_dataset)
    elif model_type == 'bnode':
        # Build feedthrough controls mask if configured
        if cfg.nn_model.network.feedthrough_controls:
            controls_dim = train_dataset[0]['controls'].shape[0] if 'controls' in train_dataset[0].keys() else 0
            if controls_dim > 0 and 'controls_names' in hdf5_dataset:
                control_names = [s.decode() if isinstance(s, bytes) else str(s) for s in hdf5_dataset['controls_names'][:]]
                feedthrough_mask = build_feedthrough_mask(
                    control_names, 
                    cfg.nn_model.network.feedthrough_controls, 
                    controls_dim
                )
            else:
                raise ValueError('feedthrough_controls configured but no controls_names found in dataset or no controls in dataset')
        else:
            feedthrough_mask = None
        
        model = BalancedNeuralODE(
                        states_dim=train_dataset[0]['states'].shape[0],
                        lat_states_mu_dim=cfg.nn_model.network.lat_states_dim,
                        parameters_dim=train_dataset[0]['parameters'].shape[0] if 'parameters' in train_dataset[0].keys() else 0,
                        lat_parameters_dim=cfg.nn_model.network.lat_parameters_dim,
                        controls_dim=train_dataset[0]['controls'].shape[0] if 'controls' in train_dataset[0].keys() else 0,
                        lat_controls_dim=cfg.nn_model.network.lat_controls_dim,
                        outputs_dim=train_dataset[0]['outputs'].shape[0] if 'outputs' in train_dataset[0].keys() else 0,
                        hidden_dim=cfg.nn_model.network.linear_hidden_dim,
                        n_layers=cfg.nn_model.network.n_linear_layers,
                        controls_to_decoder=cfg.nn_model.network.controls_to_decoder,
                        predict_states=cfg.nn_model.network.predict_states,
                        activation=eval(cfg.nn_model.network.activation),
                        initialization_type=cfg.nn_model.training.initialization_type,
                        initialization_type_ode=cfg.nn_model.training.initialization_type_ode,
                        initialization_type_ode_matrix=cfg.nn_model.training.initialization_type_ode_matrix,
                        lat_ode_type=cfg.nn_model.network.lat_ode_type,
                        include_params_encoder= cfg.nn_model.network.include_params_encoder,
                        params_to_state_encoder=cfg.nn_model.network.params_to_state_encoder,
                        params_to_control_encoder=cfg.nn_model.network.params_to_control_encoder,
                        params_to_decoder=cfg.nn_model.network.params_to_decoder,
                        controls_to_state_encoder=cfg.nn_model.network.controls_to_state_encoder,
                        state_encoder_linear = cfg.nn_model.network.state_encoder_linear,
                        control_encoder_linear = cfg.nn_model.network.control_encoder_linear,
                        parameter_encoder_linear = cfg.nn_model.network.parameter_encoder_linear,
                        ode_linear = cfg.nn_model.network.ode_linear,
                        decoder_linear = cfg.nn_model.network.decoder_linear,
                        lat_state_mu_independent = cfg.nn_model.network.lat_state_mu_independent,
                        use_input_smoother=cfg.nn_model.training.use_input_smoother,
                        feedthrough_controls_mask=feedthrough_mask,
                        )
        # initialize normalizations
        if initialize_normalization:
            model.normalization_init(hdf5_dataset)
    logging.info('Initialized model: {}'.format(model))
    logging.info('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.to(device)
    logging.info('moved model to {}'.format(device))
    return model


def _build_job_list(cfg: train_test_config_class) -> list[dict]:
    """Build the ordered outer training workflow."""
    job_list = []
    job_list.append({
        'skip': not cfg.nn_model.training.pre_train
        or cfg.nn_model.training.load_pretrained_model
        or cfg.nn_model.training.load_trained_model_for_test,
        'test': False,
        'train_cfg': cfg.nn_model.training.pre_training,
        'pre_train': True,
    })
    for main_train_cfg in cfg.nn_model.training.main_training:
        job_list.append({
            'skip': cfg.nn_model.training.load_trained_model_for_test,
            'test': False,
            'train_cfg': main_train_cfg,
            'pre_train': False,
        })
    if cfg.nn_model.training.test is True:
        job_list.append({
            'skip': False,
            'test': True,
            'train_cfg': cfg.nn_model.training.main_training[-1],
            'pre_train': False,
        })
    return job_list


def _log_job_start(idx: int, job: dict) -> None:
    if job['pre_train'] is True:
        logging.info('Starting Pre-Training with settings {}'.format(job['train_cfg']))
    elif job['test'] is True:
        logging.info('Starting Testing with settings {}'.format(job['train_cfg']))
    else:
        logging.info('Starting Train Job {} with settings {}'.format(idx, job['train_cfg']))


def _job_dataset_loading_settings(job: dict) -> tuple[int | None, int | None, int | None, int | None]:
    if job['pre_train'] is True:
        load_seq_len = job['train_cfg'].load_seq_len
        seq_len_batches = 1
        stride_valid_test = seq_len_batches if seq_len_batches is not None else None
        max_samples_valid = job['train_cfg'].batches_per_epoch * job['train_cfg'].batch_size
    elif job['test'] is True:
        load_seq_len = None
        seq_len_batches = None
        stride_valid_test = 1
        max_samples_valid = None
    else:
        load_seq_len = job['train_cfg'].load_seq_len
        seq_len_batches = job['train_cfg'].seq_len_train
        stride_valid_test = seq_len_batches if seq_len_batches is not None else None
        # we later set the batch size for validation and test to be 4 times
        # higher then for training (less memory as no backprop),
        # so this is a quarter in terms of batches of train.
        max_samples_valid = job['train_cfg'].batches_per_epoch * job['train_cfg'].batch_size
    return load_seq_len, seq_len_batches, stride_valid_test, max_samples_valid


def _create_datasets_and_dataloaders_for_job(
    cfg: train_test_config_class,
    job: dict,
    idx: int,
    hdf5_dataset: hdf5_dataset_class,
    hdf5_dataset_norm: hdf5_dataset_class | None,
    hdf5_dataset_ref: hdf5_dataset_class | None,
) -> tuple[dict, dict, int]:
    _log_job_start(idx, job)
    load_seq_len, seq_len_batches, stride_valid_test, max_samples_valid = _job_dataset_loading_settings(job)
    datasets = {}
    for context in ['train', 'test', 'validation', 'common_test']:
        stride = 1 if context == 'train' else stride_valid_test
        max_samples = None if context != 'validation' else max_samples_valid
        datasets[context] = make_stacked_dataset(
            hdf5_dataset,
            context,
            load_seq_len,
            seq_len_batches,
            stride=stride,
            max_samples=max_samples,
        )
    if hdf5_dataset_norm is not None:
        datasets['testnorm'] = make_stacked_dataset(
            hdf5_dataset_norm,
            'test',
            load_seq_len,
            seq_len_batches,
            stride=stride_valid_test,
        )
    else:
        datasets['testnorm'] = None
    if hdf5_dataset_ref is not None:
        datasets['ref'] = make_stacked_dataset(hdf5_dataset_ref, 'test', None, None)
    else:
        datasets['ref'] = None

    drop_last = job['test'] is False
    shuffle = job['test'] is False
    dataloaders = {}
    batch_size_train = job['train_cfg'].batch_size if job['test'] is False else cfg.nn_model.training.batch_size_test
    batch_size_valid_test = 4 * job['train_cfg'].batch_size if job['test'] is False else cfg.nn_model.training.batch_size_test
    for context in ['train', 'test', 'validation', 'common_test', 'testnorm']:
        batch_size = batch_size_valid_test if context in ['validation', 'test', 'testnorm'] else batch_size_train
        if context == 'testnorm' and datasets[context] is None:
            dataloaders[context] = None
            continue
        if job['test'] is True and len(datasets[context]) == 0:  # when only testing, datasets can be empty
            # TODO: I believe this is never reached
            dataloaders[context] = None
            logging.info('Only Testing: No data for context {} in dataset. Skipping loading dataloader for this context'.format(context))
            continue
        num_workers = cfg.n_workers_train_loader if context == 'train' else cfg.n_workers_other_loaders
        if batch_size > len(datasets[context]):
            batch_size_here = len(datasets[context])
            logging.warning('Batch size {} is larger than dataset size {} for context {}. Setting batch size to {}'.format(batch_size, len(datasets[context]), context, batch_size_here))
        else:
            batch_size_here = batch_size
        if len(datasets[context]) == 0:
            raise ValueError('While creating dataloaders, dataset for context {} is empty. Aborting.'.format(context))
        dataloaders[context] = torch.utils.data.DataLoader(
            datasets[context],
            batch_size=batch_size_here,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
            # multiprocessing_context='fork',
            drop_last=drop_last,
            prefetch_factor=cfg.prefetch_factor,
            collate_fn=timeseries_collate_fn,
        )
    if datasets['ref'] is not None:
        dataloaders['ref'] = torch.utils.data.DataLoader(
            datasets['ref'],
            batch_size=len(datasets['ref']),
            shuffle=False,
            num_workers=1 if cfg.n_workers_other_loaders > 0 else 0,
            persistent_workers=True if cfg.n_workers_other_loaders > 0 else False,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=cfg.prefetch_factor,
            collate_fn=timeseries_collate_fn,
        )
    else:
        dataloaders['ref'] = None
    if 'seq_len' in datasets['train'].__dict__.keys():  # for custom dataset (with map)
        job['train_cfg'].seq_len_train = datasets['train'].seq_len
    else:
        job['train_cfg'].seq_len_train = datasets['train'].datasets['time'].shape[2]
    return datasets, dataloaders, batch_size_valid_test


def _initialize_or_reload_model_for_job(
    cfg: train_test_config_class,
    job: dict,
    model,
    model_created: bool,
    datasets: dict,
    hdf5_dataset: hdf5_dataset_class,
    hdf5_dataset_norm: hdf5_dataset_class | None,
    device: torch.device,
):
    created_model_this_job = False
    if model_created is False:
        try:
            model = initialize_model(
                cfg,
                datasets['train'],
                hdf5_dataset_norm if hdf5_dataset_norm is not None else hdf5_dataset,
            )
        except Exception as e:
            logging.error('Error during model initialization: {}'.format(e))
            logging.error('Maybe dataset and dataset_norm are not compatible?')
            raise e
        model_created, created_model_this_job = True, True

    if cfg.nn_model.training.load_pretrained_model is True and created_model_this_job is True:
        path = filepaths.filepath_from_local_or_ml_artifacts(cfg.nn_model.training.path_pretrained_model)
        model.load(path=path, device=device)
        logging.info('Loaded pretrained model from {}'.format(path))
        if cfg.nn_model.training.pre_trained_model_seq_len is not None:
            job['train_cfg'].seq_len_epoch_start = cfg.nn_model.training.pre_trained_model_seq_len
            logging.info('Set seq_len_epoch_start for next job to {}'.format(cfg.nn_model.training.pre_trained_model_seq_len))
        else:
            job['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train
            logging.info('Set seq_len_epoch_start for this job to seq_len_train {} as no pre_trained_model_seq_len is given in config'.format(job['train_cfg'].seq_len_train))

    if cfg.nn_model.training.load_trained_model_for_test is True:
        path = filepaths.filepath_from_local_or_ml_artifacts(cfg.nn_model.training.path_trained_model)
        model.load(path=path, device=device)
        logging.info('Loaded trained model from {}'.format(path))
    return model, model_created


def _run_test_job(
    cfg: train_test_config_class,
    model: torch.nn.Module,
    dataloaders: dict,
    job: dict,
    device: torch.device,
    epoch_0: int,
    hdf5_dataset: hdf5_dataset_class,
    hdf5_dataset_norm: hdf5_dataset_class | None,
    hdf5_dataset_ref: hdf5_dataset_class | None,
) -> None:
    logging.info('Testing model')
    saved_predictions_to_dataset = False
    for context in ['train', 'test', 'validation', 'common_test', 'testnorm', 'ref']:
        if dataloaders[context] is None:
            logging.info('No data for context {} in dataset. Skipping.'.format(context))
            continue

        logging.info('Testing of dataset for context {}'.format(context))
        save_predictions = cfg.nn_model.training.save_predictions_in_dataset
        save_predictions = save_predictions and context in cfg.nn_model.training.save_predictions_for
        save_predictions = save_predictions or (context in cfg.nn_model.training.test_save_internal_variables_for)

        if save_predictions is True:
            if not filepaths.filepath_dataset_current_hydra_output().exists():
                logging.warning('Creating dataset file: {}'.format(filepaths.filepath_dataset_current_hydra_output()))
                hdf5_dataset_pred = h5py.File(filepaths.filepath_dataset_current_hydra_output(), 'w')
                for key in hdf5_dataset.keys():
                    if key not in ['train', 'test', 'validation', 'common_test', 'common_validation', 'time']:
                        hdf5_dataset_pred.copy(hdf5_dataset[key], key)
                        logging.info('Copying dataset key {} to hdf5 file for testing.'.format(key))
            else:
                hdf5_dataset_pred = h5py.File(filepaths.filepath_dataset_current_hydra_output(), 'a')

            logging.info('Copying dataset for context {} to hdf5 file for testing.'.format(context))

            if context in ['train', 'test', 'validation', 'common_test', 'common_validation']:
                hdf5_dataset_pred.create_group(context)
                for key in hdf5_dataset[context].keys():
                    data = hdf5_dataset[context][key][:]
                    hdf5_dataset_pred[context].create_dataset(key, data=data)
                hdf5_dataset_pred[context].create_dataset('time', data=hdf5_dataset['time'][:])
            elif context in ['testnorm']:
                hdf5_dataset_pred.create_group(context)
                for key in hdf5_dataset_norm['test'].keys():
                    data = hdf5_dataset_norm['test'][key][:]
                    hdf5_dataset_pred[context].create_dataset(key, data=data)
                hdf5_dataset_pred[context].create_dataset('time', data=hdf5_dataset_norm['time'][:])
            elif context in ['ref']:
                hdf5_dataset_pred.create_group(context)
                for key in hdf5_dataset_ref['test'].keys():
                    data = hdf5_dataset_ref['test'][key][:]
                    hdf5_dataset_pred[context].create_dataset(key, data=data)
                hdf5_dataset_pred[context].create_dataset('time', data=hdf5_dataset_ref['time'][:])
            else:
                raise ValueError('Context {} not recognized for copying dataset to hdf5 file for testing.'.format(context))

            logging.info('Adding model predictions to hdf5 file for context {}.'.format(context))

            total_len = len(dataloaders[context].dataset)
            data_iter = iter(dataloaders[context])
            created_dsets = False
            write_offset = 0
            metrics_sum = {}
            n_batches = 0
            keys_to_save = []
            while True:
                try:
                    data_batch = next(data_iter)
                except StopIteration:
                    break
                with torch.no_grad():
                    logging.info(f"\t Batch {n_batches+1}/{int(total_len/cfg.nn_model.training.batch_size_test)+1}")
                    ret_vals_batch, model_outputs_batch = model.model_and_loss_evaluation(
                        data_batch,
                        job['train_cfg'],
                        job['pre_train'],
                        device,
                        return_model_outputs=True,
                        test=True,
                    )
                if not created_dsets:
                    for key in model_outputs_batch.keys():
                        if key in ['states_hat', 'states_der_hat', 'outputs_hat']:
                            save_key = True
                        elif cfg.nn_model.training.test_save_internal_variables is True and context in cfg.nn_model.training.test_save_internal_variables_for:
                            save_key = True
                            logging.info('Saving internal variable {} for context {} according to config.'.format(key, context))
                        else:
                            save_key = False
                        if save_key:
                            keys_to_save.append(key)
                    for key in keys_to_save:
                        arr = model_outputs_batch[key]
                        shape_rest = arr.shape[1:]
                        dset_shape = (total_len,) + shape_rest
                        hdf5_dataset_pred.create_dataset(context + '/' + key, shape=dset_shape, dtype=arr.dtype)
                    created_dsets = True
                batch = next(iter(model_outputs_batch.values())).shape[0] if len(model_outputs_batch) > 0 else 0
                for key in keys_to_save:
                    arr = model_outputs_batch[key]
                    hdf5_dataset_pred[context + '/' + key][write_offset:write_offset + arr.shape[0], ...] = arr
                write_offset += batch
                if n_batches == 0:
                    metrics_sum = {k: float(v) for k, v in ret_vals_batch.items()}
                else:
                    for key, value in ret_vals_batch.items():
                        metrics_sum[key] += float(value)
                n_batches += 1
            ret_vals = {k: (metrics_sum[k] / max(n_batches, 1)) for k in metrics_sum.keys()}
        else:
            ret_vals = test_or_validate_one_epoch(
                model,
                dataloaders[context],
                job['train_cfg'],
                job['pre_train'],
                device,
                all_batches=True,
                return_model_outputs=False,
            )

        logging.info('Stats for context {}: {}'.format(context, ret_vals))
        mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals, context), step=epoch_0 + 1)
        mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals, '{}_final'.format(context)), step=epoch_0 + 1)
        if save_predictions is True:
            for key, value in ret_vals.items():
                hdf5_dataset_pred.create_dataset(context + '/' + key, data=value)
            hdf5_dataset_pred.close()
            saved_predictions_to_dataset = True

    if saved_predictions_to_dataset:
        shutil.copy(Path(__file__), filepaths.dir_current_hydra_output())
        logging.info('copied current trainer.py: {} \nto: \n{}'.format(Path(__file__), filepaths.dir_current_hydra_output()))


@log_hydra_to_mlflow
def train_all_phases(cfg: train_test_config_class):
    """Execute complete multi-phase training pipeline with MLflow tracking.
    
    Main orchestration function that coordinates:

    - Dataset loading
    - Model initialization  
    - Optional pre-training (NODE only)
    - Multi-phase main training
    - Final testing and evaluation
    - MLflow artifact logging
    
    The function processes a job list consisting of optional pre-training,
    multiple main training phases, and final testing. Each phase can have
    different hyperparameters and training strategies.
    
    Args:
        cfg (train_test_config_class): Validated Hydra configuration containing:
            - dataset_path, dataset_name: Dataset location and identifier
            - nn_model.training.pre_train: Enable pre-training (NODE only)
            - nn_model.training.main_training: List of training phase configs
            - nn_model.training.test: Enable final testing
            - use_cuda: Device preference
            - mlflow_experiment_name: MLflow experiment name
    
    Side Effects:
        - Creates/updates model checkpoints: model_phase_{i}.pt
        - Logs metrics, parameters, and artifacts to MLflow
        - Saves predictions to dataset if configured
        - Copies Hydra outputs to MLflow artifacts
        - Creates could_not_log_artifacts.txt on logging failures
    
    Training Flow:
        1. Load HDF5 dataset and log to MLflow
        2. Build job list (pre-train, main phases, test)
        3. For each job:
           - Initialize/reload dataloaders if needed
           - Initialize/load model if needed
           - Execute training or testing
           - Save checkpoint and log metrics
        4. Copy all outputs to MLflow artifacts
    
    Raises:
        RuntimeError: If CUDA memory errors occur repeatedly
        FileNotFoundError: If dataset or checkpoint files missing
        
    Notes:
        - Decorated with @log_hydra_to_mlflow for automatic config logging
        - Memory errors trigger dataloader recreation with adjusted settings
        - NaN losses trigger checkpoint reload and gradient clipping adjustment
        - Progressive sequence length increase during phase transitions
        - For the runtime-vs-checkpoint state split, see
          ``docs/bnode_core/ode/restart_training.md``
    
    See Also:
        train_one_phase : Single training phase execution
        initialize_model : Model instantiation and initialization
    """
    logging.info('Start training all phases....')
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
    logging.info('Using device: {}'.format(device))
    
    # load hdf5 dataset
    hdf5_dataset, _ = load_dataset_and_config(cfg.dataset_name, cfg.dataset_path)
    mlflow_proxy.log_param('dataset_name', cfg.dataset_name)

    if cfg.dataset_norm_name is not None or cfg.dataset_norm_path is not None:
        hdf5_dataset_norm, _ = load_dataset_and_config(cfg.dataset_norm_name, cfg.dataset_norm_path)
        mlflow_proxy.log_param('dataset_norm_name', cfg.dataset_norm_name)
    else: 
        hdf5_dataset_norm = None
    
    if cfg.dataset_ref_name is not None or cfg.dataset_ref_path is not None:
        hdf5_dataset_ref, _ = load_dataset_and_config(cfg.dataset_ref_name, cfg.dataset_ref_path)
        mlflow_proxy.log_param('dataset_ref_name', cfg.dataset_ref_name)
    else:
        hdf5_dataset_ref = None
    
    # collect jobs
    # job_list=[] filled with dict of style: {'skip': bool, 'test': bool, 'train_cfg': cfg, 'pre_train': bool}
    job_list = _build_job_list(cfg)
    logging.info('Created job list: {}'.format(job_list))

    # restart logic

    # We inspect the serialized restart checkpoint here only to decide where the
    # outer job loop resumes. Once the selected phase has its dataloaders and
    # checkpoint paths, we create the phase-local LiveTrainingState without
    # runtime objects. train_one_phase() then binds optimizer/scaler/schedulers
    # explicitly before any checkpoint restore happens.
    # See docs/bnode_core/ode/restart_training.md for the full state model.
    outer_state = _load_outer_training_state(cfg=cfg, job_list=job_list)

    # outer-loop runtime values
    model_created = False
    next_epoch_anchor = outer_state.next_epoch_anchor
    model = None
    for idx, job in enumerate(
        outer_state.job_list[outer_state.job_start_idx:],
        start=outer_state.job_start_idx,
    ):
        retry_batch_size = job['train_cfg'].batch_size if job['test'] is False else cfg.nn_model.training.batch_size_test
        while True: # loop to catch memory errors
            try:
                if job['skip'] is False: 
                    datasets, dataloaders, retry_batch_size = _create_datasets_and_dataloaders_for_job(
                        cfg,
                        job,
                        idx,
                        hdf5_dataset,
                        hdf5_dataset_norm,
                        hdf5_dataset_ref,
                    )
                    model, model_created = _initialize_or_reload_model_for_job(
                        cfg,
                        job,
                        model,
                        model_created,
                        datasets,
                        hdf5_dataset,
                        hdf5_dataset_norm,
                        device,
                    )

                if job['skip'] is True:
                    if job['pre_train'] is True:
                        logging.info('Skipping Pre-Training')
                    else:
                        logging.info('Skipping Train Job {} as trained model is loaded in following phases'.format(idx))
                else:
                    if job['test'] is False:
                        phase_restart_state = outer_state.restart_state_for_job(idx)
                        live_state = _create_uninitialized_phase_state(
                            cfg,
                            dataloaders,
                            job['train_cfg'],
                            job['pre_train'],
                            idx,
                            next_epoch_anchor,
                            phase_restart_state,
                            outer_state.inner_restart_state_path,
                        )
                        # train one phase
                        next_epoch_anchor = train_one_phase(
                            cfg,
                            model,
                            dataloaders,
                            job['train_cfg'],
                            job['test'],
                            job['pre_train'],
                            idx,
                            next_epoch_anchor,
                            restart_state=phase_restart_state,
                            restart_manager=outer_state.inner_restart_state_path,
                            live_state=live_state,
                            outer_state=outer_state,
                        )
                        outer_state.consume_restart_state()
                        outer_state.advance_to_next_epoch_anchor(next_epoch_anchor)
                        # set seq_len_epoch_start for next job
                        if len(outer_state.job_list) > idx+1:
                            # consequently, seq_len_epoch_start should be seq_len_train
                            outer_state.job_list[idx+1]['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train if job['pre_train'] is False else 1
                            logging.info('Set seq_len_epoch_start for next job to {}, the seq_len_train of this job'.format(outer_state.job_list[idx+1]['train_cfg'].seq_len_epoch_start))
                    else:
                        _run_test_job(
                            cfg,
                            model,
                            dataloaders,
                            job,
                            device,
                            next_epoch_anchor,
                            hdf5_dataset,
                            hdf5_dataset_norm,
                            hdf5_dataset_ref,
                        )
                if cfg.use_cuda:
                    torch.cuda.empty_cache() 
                break # break the exception loop
            except CheckpointRequestedExit as e:
                logging.info('Stopping after checkpoint request: {}'.format(e))
                mlflow_proxy.set_tag_if_active('ended by', 'checkpoint request')
                return
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e) or 'CUDA memory is almost full' in str(e):
                    logging.warning('CUDA out of memory error. Trying again in 10 seconds')
                    pyTime.sleep(10)
                    logging.info('Setting batch size to {}'.format(int(retry_batch_size * 0.7)))
                    if not job['test']:
                        job['train_cfg'].batch_size = int(retry_batch_size * 0.7)
                    else:
                        cfg.nn_model.training.batch_size_test = int(retry_batch_size * 0.7)
                    if cfg.use_cuda:
                        torch.cuda.empty_cache()
                else:
                    raise e
    _clear_restart_state(
        outer_state.outer_restart_state_path,
        outer_state.inner_restart_state_path,
    )


def _next_batch(data_loader, iterator):
    """Get next batch from a DataLoader using a (possibly persistent) iterator.

    The caller owns the iterator reference (e.g. stored in a dict per
    context) and is responsible for keeping it between calls. This helper
    simply advances the iterator, recreating it on exhaustion.
    """
    if data_loader is None:
        raise ValueError("No DataLoader provided to _next_batch")
    if iterator is None:
        iterator = iter(data_loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(data_loader)
        batch = next(iterator)
    return batch, iterator


# define train loop for one epoch
def train_one_epoch(live_state: LiveTrainingState, train_loader, train_iter, epoch):
    epoch_this_phase = epoch - live_state.phase_state.phase_epoch_0
    # train_cfg may be deep-copied below for epoch-specific overrides (e.g. evaluate_at_control_times)
    train_cfg = live_state.train_cfg
    live_state.model.train()
    _time_forward = 0
    _time_backward = 0
    _time_step = 0
    _time_loader = 0
    _time_l = pyTime.time()
    batches_per_epoch = len(train_loader) if train_cfg.batches_per_epoch is None else train_cfg.batches_per_epoch
    if epoch_this_phase in [0, 1] and live_state.pre_train is False: # evaluate at control times only in first epoch to get good estimate for memory usage
        logging.info('Evaluating at control times to get good estimate for memory usage')
        train_cfg = copy.deepcopy(train_cfg)
        train_cfg.evaluate_at_control_times = True
    _batches_this_phase = epoch_this_phase * batches_per_epoch
    for batch_idx in range(batches_per_epoch):
        # Use a persistent iterator for the training DataLoader so that
        # batches_per_epoch can be much larger than len(train_loader)
        # without repeatedly recreating iterators.
        data_batch, train_iter = _next_batch(train_loader, train_iter)
        # seq_len_increase_in_batches
        _batches_this_phase = epoch_this_phase * batches_per_epoch + batch_idx
        if live_state.pre_train is False:
            if _batches_this_phase < train_cfg.seq_len_increase_in_batches:
                _seq_len_now = train_cfg.seq_len_epoch_start + int(_batches_this_phase/train_cfg.seq_len_increase_in_batches * (train_cfg.seq_len_train - train_cfg.seq_len_epoch_start))
                _seq_len_now = min(_seq_len_now, train_cfg.seq_len_train)
                for keys in data_batch.keys():
                    if len(data_batch[keys].shape) == 3:
                        data_batch[keys] = data_batch[keys][:,:,:_seq_len_now]
                if batch_idx % live_state.cfg.batch_print_interval == 0:
                    logging.info('\t \t Increasing sequence length to {} in batch since phase start {}/{} of increase_in_batches'.format(_seq_len_now, _batches_this_phase, train_cfg.seq_len_increase_in_batches))
            else:
                _seq_len_now = train_cfg.seq_len_train
        else:
            _seq_len_now = 1
        _time_loader += pyTime.time() - _time_l
        _time = pyTime.time()

        # Branch on optimizer type: standard first-order optimizers vs LBFGS
        is_lbfgs = isinstance(live_state.optimizer, LBFGS)

        if not is_lbfgs:
            live_state.optimizer.zero_grad()
            # Standard optimizers (e.g., Adam): single forward/backward pass
            with torch.amp.autocast('cuda', enabled=live_state.cfg.use_amp and live_state.cfg.use_cuda):
                ret_vals_train = live_state.model.model_and_loss_evaluation(
                    data_batch,
                    train_cfg,
                    live_state.pre_train,
                    live_state.device,
                    return_model_outputs=False,
                    test=False,
                    last_batch=batch_idx == batches_per_epoch - 1,
                )
            loss = ret_vals_train['loss']
            _time_forward += pyTime.time() - _time
            _time = pyTime.time()
            live_state.scaler.scale(loss).backward()
        else:
            # LBFGS: closure-based optimization; disable AMP for simplicity
            ret_vals_train = {}

            def _closure():
                live_state.optimizer.zero_grad()
                out = live_state.model.model_and_loss_evaluation(
                    data_batch,
                    train_cfg,
                    live_state.pre_train,
                    live_state.device,
                    return_model_outputs=False,
                    test=False,
                    last_batch=batch_idx == batches_per_epoch - 1,
                )
                loss_closure = out['loss']
                loss_closure.backward()
                # optionally apply gradient clipping inside the closure
                clip_grad_norm_(live_state.model.parameters(), train_cfg.clip_grad_norm)
                # store last returned values for logging
                nonlocal ret_vals_train
                ret_vals_train = out
                logging.info(' LBFGS step closure: batch {}, loss {:.4f}'.format(batch_idx, loss_closure.item()))
                return loss_closure

            # Run a single LBFGS step for this batch; internal iterations
            # are controlled via train_cfg.lbfgs_max_iter.
            loss = live_state.optimizer.step(_closure)
            _time_forward += pyTime.time() - _time
            _time = pyTime.time()
        _flag_break_cuda_memory = False
        if live_state.cfg.use_cuda:
            mlflow_proxy.log_metric('CUDA_memory_reserved_GB', torch.cuda.memory_reserved()/(1024^3), step=epoch)
            if epoch_this_phase == 0:
                if torch.cuda.memory_reserved() > 0.6 * torch.cuda.get_device_properties(0).total_memory:
                    _flag_break_cuda_memory = True
        if live_state.pre_train is False and live_state.cfg.use_cuda:
            if epoch_this_phase == 0:
                if (train_cfg.seq_len_train/_seq_len_now) * torch.cuda.memory_reserved() > 0.6 * torch.cuda.get_device_properties(0).total_memory:
                    _flag_break_cuda_memory = True
            if torch.cuda.memory_reserved() > 0.98 * torch.cuda.get_device_properties(0).total_memory:
                _flag_break_cuda_memory = True
        if _flag_break_cuda_memory is True:
            logging.warning('CUDA memory is almost full. Raising exception to catch in train_all_phases')
            logging.info('Current number of batches for whole dataset: {}'.format(len(train_loader)))
            raise RuntimeError('CUDA memory is almost full')
        _ode_calls_backward = live_state.model.ode_fun_count if hasattr(live_state.model, 'ode_fun_count') else 0
        _time_backward += pyTime.time() - _time
        _time = pyTime.time()

        # For LBFGS, step and clipping are handled in the closure; for others,
        # unscale, clip and step via the GradScaler.
        if not is_lbfgs:
            live_state.scaler.unscale_(live_state.optimizer)
            _norm = clip_grad_norm_(live_state.model.parameters(), train_cfg.clip_grad_norm)
            if _norm > train_cfg.clip_grad_norm:
                logging.info('Gradient norm {} is larger than clip_grad_norm {}. Clipping Gradient.'.format(_norm, train_cfg.clip_grad_norm))
            live_state.scaler.step(live_state.optimizer)
            live_state.scaler.update()
        else:
            # For LBFGS, report the most recent gradient norm for logging.
            _norm = clip_grad_norm_(live_state.model.parameters(), train_cfg.clip_grad_norm)
        # step learning-rate scheduler(s) once per optimizer update (per batch)
        if live_state.lr_schedulers:
            # For now only cosine-type schedulers are stepped per batch; others
            # will typically be stepped at epoch level using validation metrics.
            for key in live_state.lr_schedulers.keys():
                if key == 'cosine':
                    live_state.lr_schedulers[key].step()
        _time_step += pyTime.time() - _time
        
        # print training stats for this batch
        if batch_idx % live_state.cfg.batch_print_interval == 0:
            _total_time = _time_forward + _time_backward + _time_step + _time_loader
            _total_time = _total_time
            _ode_calls_forward = ret_vals_train['ode_calls_forward'] if 'ode_calls_forward' in ret_vals_train.keys() else 0 
            try:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%) tot.: {}] Loss: {:.6f}, avg. time per batch: {:.3f} [load. {:.1f}%, forw. {:.1f}%, backw. {:.1f}%, step {:.1f}%], ODE calls forw/backw {}/{}'.format(
                    epoch+1, batch_idx+1, batches_per_epoch,
                    100. * batch_idx / batches_per_epoch, len(train_loader),
                    loss.item(), _total_time/(batch_idx+1),_time_loader/_total_time*100, _time_forward/_total_time*100, _time_backward/_total_time*100, _time_step/_total_time*100,
                    _ode_calls_forward, _ode_calls_backward))
            except Exception as e:
                logging.info('error in logging train epoch info: {}'.format(e))
        _time_l = pyTime.time()
    # return values for this epoch
    ret_vals_train = dict(
        {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in ret_vals_train.items()
        }
    )
    ret_vals_train['grad_norm'] = _norm
    ret_vals_train['clip_grad_norm'] = train_cfg.clip_grad_norm
    ret_vals_train['seq_len_now'] = _seq_len_now
    ret_vals_train['time_forward'] = _time_forward
    ret_vals_train['time_backward'] = _time_backward
    ret_vals_train['time_optimizer_step'] = _time_step
    ret_vals_train['time_loader'] = _time_loader
    ret_vals_train['time_total'] = _time_forward + _time_backward + _time_step + _time_loader
    ret_vals_train['time_per_batch'] = ret_vals_train['time_total'] / batches_per_epoch
    ret_vals_train['time_per_batch_forward'] = ret_vals_train['time_forward'] / batches_per_epoch
    ret_vals_train['time_per_batch_backward'] = ret_vals_train['time_backward'] / batches_per_epoch
    ret_vals_train['time_per_batch_optimizer_step'] = ret_vals_train['time_optimizer_step'] / batches_per_epoch
    ret_vals_train['time_per_batch_loader'] = ret_vals_train['time_loader'] / batches_per_epoch
    if live_state.pre_train is False:
        ret_vals_train['ode_calls_backward'] = _ode_calls_backward
    return ret_vals_train, train_iter  

def test_or_validate_one_epoch(model, data_loader, train_cfg, pre_train, device, all_batches=False, return_model_outputs=False,
                               activate_deterministic_mode=False, data_iter=None):
    model.eval()
    if all_batches is True:
        ret_vals = []
        for batch_idx, data_batch in enumerate(data_loader):
            logging.info('Testing batch {}/{}'.format(batch_idx+1, len(data_loader)))
            with torch.no_grad():
                ret_vals.append(model.model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, return_model_outputs=return_model_outputs, test=True))
        if return_model_outputs is True:
            model_outputs = {key: np.concatenate([x[1][key] for x in ret_vals], axis=0) for key in ret_vals[0][1].keys()}
            ret_vals = {key: np.mean([x[0][key] for x in ret_vals]) for key in ret_vals[0][0].keys()}
        else:
            ret_vals = {key: np.mean([x[key] for x in ret_vals]) for key in ret_vals[0].keys()}
        return ret_vals if return_model_outputs is False else (ret_vals, model_outputs)
    else:
        # Single-batch evaluation. Use a persistent iterator if provided;
        # otherwise create a one-off iterator.
        data_batch, data_iter = _next_batch(data_loader, data_iter)
        with torch.no_grad():
            ret_vals = model.model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, return_model_outputs=return_model_outputs, test=True, activate_deterministic_mode=activate_deterministic_mode)
        if return_model_outputs is True:
            model_outputs = ret_vals[1]
            ret_vals = ret_vals[0]
        if return_model_outputs is False:
            return ret_vals, data_iter
        else:
            return ret_vals, model_outputs, data_iter

def append_context_to_dict_keys(dictionary: dict, context: str, pre_train: bool = False):
        if pre_train is True:
            return dict({'pre_{}_{}'.format(key, context): value for key, value in dictionary.items()})
        else:
            return dict({'{}_{}'.format(key, context): value for key, value in dictionary.items()})


def _compute_phase_epoch_settings(
    dataloaders: dict,
    train_cfg: base_training_settings_class,
    pre_train: bool,
) -> tuple[int, int, int]:
    if pre_train is True:
        batches_per_epoch = len(dataloaders['train'])
        epochs_for_seq_len_increase = 0
    else:
        batches_per_epoch = len(dataloaders['train']) if train_cfg.batches_per_epoch is None else train_cfg.batches_per_epoch
        if train_cfg.seq_len_epoch_start is not None:
            if train_cfg.seq_len_epoch_start < train_cfg.seq_len_train:
                epochs_for_seq_len_increase = int(train_cfg.seq_len_increase_in_batches / batches_per_epoch)
            else:
                epochs_for_seq_len_increase = 0
                train_cfg.seq_len_increase_in_batches = 0
        else:
            epochs_for_seq_len_increase = 0
            train_cfg.seq_len_increase_in_batches = 0
    max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase
    return batches_per_epoch, epochs_for_seq_len_increase, max_epochs


def _build_phase_checkpoint_paths(pre_train: bool, job_idx: int) -> tuple[Path, Path, Path, Path]:
    path_best_model = filepaths.filepath_pretrained_model_current_hydra_output() if pre_train is True else filepaths.filepath_model_current_hydra_output(job_idx)
    path_optimizer_best_model = filepaths.filepath_optimizer_current_hydra_output() if pre_train is True else filepaths.filepath_optimizer_current_hydra_output(job_idx)
    path_current_model = filepaths.filepath_model_current_hydra_output()
    path_current_optimizer = filepaths.filepath_optimizer_current_hydra_output()
    return path_best_model, path_optimizer_best_model, path_current_model, path_current_optimizer


def _create_phase_optimizer(
    model: torch.nn.Module,
    train_cfg: base_training_settings_class,
    pre_train: bool,
    job_idx: int,
):
    optimizer_name_lower = train_cfg.optimizer.lower()
    if optimizer_name_lower == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.lr_start,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.beta1_adam, train_cfg.beta2_adam),
        )
    elif optimizer_name_lower == 'lbfgs':
        optimizer = LBFGS(
            model.parameters(),
            lr=train_cfg.lr_start,
            max_iter=train_cfg.lbfgs_max_iter,
            history_size=train_cfg.lbfgs_history_size,
            tolerance_grad=train_cfg.lbfgs_tolerance_grad,
            tolerance_change=train_cfg.lbfgs_tolerance_change,
            line_search_fn=train_cfg.lbfgs_line_search_fn,
        )
        logging.info('Using LBFGS optimizer')
    else:
        raise ValueError(f"Unknown optimizer type '{train_cfg.optimizer}'. Supported: 'adam', 'lbfgs'.")
    if pre_train is False and train_cfg.reload_optimizer is True:
        try:
            optimizer.load_state_dict(torch.load(filepaths.filepath_optimizer_current_hydra_output(job_idx-1)))
            logging.info('Reloaded optimizer from {}'.format(filepaths.filepath_optimizer_current_hydra_output(job_idx-1)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = train_cfg.lr_start
                logging.info('Set learning rate to {} after reloading optimizer'.format(train_cfg.lr_start))
        except Exception:
            logging.warning('Could not reload optimizer from {}'.format(filepaths.filepath_optimizer_current_hydra_output(job_idx-1)))
            logging.warning('Initializing optimizer with new parameters')
    return optimizer


def _create_phase_lr_schedulers(
    train_cfg: base_training_settings_class,
    optimizer,
    batches_per_epoch: int,
    job_idx: int,
    pre_train: bool,
    test: bool,
):
    lr_schedulers = {}
    if test is False and pre_train is False and train_cfg.use_lr_scheduler:
        if train_cfg.lr_scheduler_type == 'cosine':
            if train_cfg.cosine_T_max is not None:
                t_max_epochs = train_cfg.cosine_T_max
            else:
                t_max_epochs = max(1, train_cfg.max_epochs // 10)
            t_max_batches = max(1, int(t_max_epochs * batches_per_epoch))
            eta_min = train_cfg.cosine_eta_min
            lr_schedulers['cosine'] = CosineAnnealingLR(optimizer, T_max=t_max_batches, eta_min=eta_min)
            logging.info(f'Initialized cosine LR scheduler (per batch): T_max_batches={t_max_batches}, eta_min={eta_min}')
        elif train_cfg.lr_scheduler_type == 'plateau':
            if train_cfg.plateau_patience is None:
                iters = lr_on_plateau_iterations_to_min_lr(
                    lr_start=train_cfg.lr_start,
                    lr_min=train_cfg.plateau_min_lr,
                    factor=train_cfg.plateau_factor,
                    eps=train_cfg.plateau_eps
                )
                iters = max(iters, 1)
                patience = min(int(train_cfg.early_stopping_patience / 5), (train_cfg.max_epochs / 3) // iters)
            else:
                patience = train_cfg.plateau_patience
            mlflow_proxy.log_param('job {} LR scheduler patience'.format(job_idx), patience)
            lr_schedulers['plateau'] = ReduceLROnPlateau(
                optimizer,
                mode=train_cfg.plateau_mode,
                factor=train_cfg.plateau_factor,
                patience=patience,
                threshold=train_cfg.plateau_threshold,
                threshold_mode=train_cfg.plateau_threshold_mode,
                cooldown=train_cfg.plateau_cooldown,
                min_lr=train_cfg.plateau_min_lr,
                eps=train_cfg.plateau_eps,
            )
            logging.info('Initialized ReduceLROnPlateau LR scheduler: '
                         f"mode={train_cfg.plateau_mode}, factor={train_cfg.plateau_factor}, "
                         f"patience={patience}, threshold={train_cfg.plateau_threshold}, "
                         f"threshold_mode={train_cfg.plateau_threshold_mode}, cooldown={train_cfg.plateau_cooldown}, "
                         f"min_lr={train_cfg.plateau_min_lr}, eps={train_cfg.plateau_eps}")
        else:
            raise ValueError(f'LR scheduler type {train_cfg.lr_scheduler_type} not recognized')
    if len(lr_schedulers) == 0:
        return None
    return lr_schedulers


# Phase runtime ownership note for the refactor:
# - recreated each run/phase: cfg-driven setup, datasets/dataloaders, epoch bounds, and checkpoint paths
# - checkpoint-managed through LiveTrainingState: model/optimizer/lr_schedulers/scaler/early_stopping
#   plus phase_state counters and flags
def _prepare_phase_runtime(
    cfg: train_test_config_class,
    model: torch.nn.Module,
    dataloaders: dict,
    train_cfg: base_training_settings_class,
    test: bool,
    pre_train: bool,
    job_idx: int,
    epoch_0: int,
    restart_state: InnerTrainingStateCheckpoint | None,
    restart_manager: Path | None,
    live_state: LiveTrainingState | None = None,
) -> LiveTrainingState:
    if live_state is None:
        live_state = _create_uninitialized_phase_state(
            cfg,
            dataloaders,
            train_cfg,
            pre_train,
            job_idx,
            epoch_0,
            restart_state,
            restart_manager,
        )
    optimizer = _create_phase_optimizer(model, train_cfg, pre_train, job_idx)
    early_stopping = EarlyStopping(
        patience=train_cfg.early_stopping_patience,
        verbose=True,
        threshold=train_cfg.early_stopping_threshold,
        threshold_mode=train_cfg.early_stopping_threshold_mode,
        path=live_state.path_best_model,
        optimizer_path=live_state.path_optimizer_best_model,
        trace_func=logging.info,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_cuda and cfg.use_amp)
    logging.info('Training with automatic mixed precision: {}'.format(cfg.use_amp and cfg.use_cuda))
    lr_schedulers = _create_phase_lr_schedulers(
        train_cfg,
        optimizer,
        live_state.batches_per_epoch or 1,
        job_idx,
        pre_train,
        test,
    )
    live_state.bind_runtime_objects(
        model=model,
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
        scaler=scaler,
        early_stopping=early_stopping,
        restart_state=restart_state,
    )
    return live_state


def _create_uninitialized_phase_state(
    cfg: train_test_config_class,
    dataloaders: dict,
    train_cfg: base_training_settings_class,
    pre_train: bool,
    job_idx: int,
    epoch_0: int,
    restart_state: InnerTrainingStateCheckpoint | None,
    restart_manager: Path | None,
) -> LiveTrainingState:
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
    batches_per_epoch, epochs_for_seq_len_increase, max_epochs = _compute_phase_epoch_settings(
        dataloaders,
        train_cfg,
        pre_train,
    )
    path_best_model, path_optimizer_best_model, path_current_model, path_current_optimizer = _build_phase_checkpoint_paths(
        pre_train,
        job_idx,
    )
    return LiveTrainingState.create_uninitialized(
        cfg=cfg,
        train_cfg=train_cfg,
        job_idx=job_idx,
        pre_train=pre_train,
        device=device,
        phase_epoch_0=epoch_0,
        max_epochs=max_epochs,
        batches_per_epoch=batches_per_epoch,
        epochs_for_seq_len_increase=epochs_for_seq_len_increase,
        path_best_model=path_best_model,
        path_optimizer_best_model=path_optimizer_best_model,
        path_current_model=path_current_model,
        path_current_optimizer=path_current_optimizer,
        hydra_output_dir=filepaths.dir_current_hydra_output(),
        restart_manager_path=restart_manager,
        restart_state=restart_state,
        next_epoch_anchor=epoch_0 if restart_state is not None else None,
    )


def _compute_phase_stop_flags(
    phase_state,
    early_stopping,
    train_cfg: base_training_settings_class,
    epoch: int,
) -> dict[str, bool]:
    return {
        'max_epoch': epoch == phase_state.epoch_stop - 1,
        'early_stopping': early_stopping.early_stop and phase_state.flag_out_of_seq_len_increase is True,
        'break_after_loss': (
            early_stopping.best_score < train_cfg.break_after_loss_of
            if train_cfg.break_after_loss_of is not None and early_stopping.best_score is not None
            else False
        ),
        'nan_counter': phase_state.nan_counter > 50,
    }


def _apply_phase_stop_flags(
    stop_flags: dict[str, bool],
    job_idx: int,
    train_cfg: base_training_settings_class,
    model: torch.nn.Module,
    path_best_model: Path,
    device: torch.device,
) -> bool:
    if not any(stop_flags.values()):
        return False
    if stop_flags['max_epoch']:
        logging.info('Reached max epochs')
        mlflow_proxy.set_tag_if_active('job {} ended by'.format(job_idx), 'max epochs')
    elif stop_flags['early_stopping']:
        logging.info("Early stopping")
        mlflow_proxy.set_tag_if_active('job {} ended by'.format(job_idx), 'early stopping')
    elif stop_flags['break_after_loss']:
        logging.info('Break phase after reaching loss level of {}'.format(train_cfg.break_after_loss_of))
        mlflow_proxy.set_tag_if_active('job {} ended by'.format(job_idx), 'break after loss')
    elif stop_flags['nan_counter']:
        logging.info('Break phase after 50 NaNs in loss')
        mlflow_proxy.set_tag_if_active('job {} ended by'.format(job_idx), '4 NaNs in loss')
    else:
        raise ValueError('This should not happen')
    model.load(path=path_best_model, device=device)
    logging.info('loaded best model from {}'.format(path_best_model))
    return True


def _update_phase_epoch_stop_for_seq_len(
    live_state: LiveTrainingState,
    batches_per_epoch: int,
    epoch: int,
) -> None:
    phase_state = live_state.phase_state
    train_cfg = live_state.train_cfg
    if live_state.pre_train is False:
        if phase_state.stable_epochs > train_cfg.seq_len_increase_abort_after_n_stable_epochs and phase_state.flag_out_of_seq_len_increase is False:
            train_cfg.seq_len_increase_in_batches = batches_per_epoch * (epoch - phase_state.phase_epoch_0)
            phase_state.epoch_stop = phase_state.phase_epoch_0 + train_cfg.max_epochs + (epoch - phase_state.phase_epoch_0)


def _run_training_epoch_or_eval_epoch(
    live_state: LiveTrainingState,
    dataloaders: dict,
    dataloader_iters: dict,
    epoch: int,
    flag_break_after_epoch: bool,
):
    phase_state = live_state.phase_state
    train_cfg = live_state.train_cfg
    model = live_state.model
    optimizer = live_state.optimizer
    device = live_state.device
    path_best_model = live_state.path_best_model
    path_optimizer_best_model = live_state.path_optimizer_best_model
    path_current_model = live_state.path_current_model
    path_current_optimizer = live_state.path_current_optimizer

    if not flag_break_after_epoch and not phase_state.first_epoch_is_evaluation:
        ret_vals_train = {'loss': float('nan')}
        try:
            ret_vals_train, dataloader_iters['train'] = train_one_epoch(
                live_state,
                dataloaders['train'],
                dataloader_iters['train'],
                epoch,
            )
            reload_assertion_error = False
        except AssertionError as e:
            logging.error('Assertion error during training: {}'.format(e))
            logging.error('This is likely to happen because of the odeint integration in the model.')
            logging.error('Aborting training of this epoch and reloading last working model to continue with next epoch.')
            reload_assertion_error = True
        if np.isnan(ret_vals_train['loss']) or np.isinf(ret_vals_train['loss']) or reload_assertion_error:
            if train_cfg.reload_model_if_loss_nan:
                if not phase_state.nan_counter > 49:
                    try:
                        model.load(path=path_current_model, device=device)
                        optimizer.load_state_dict(torch.load(path_current_optimizer))
                        logging.warning('Loss is NaN. Loaded last model and corresponding optimizer from {}'.format(path_current_model))
                        mlflow_proxy.log_metric('loss_nan_reload', 1, step=epoch)
                        phase_state.grad_norm_last_reduced_counter += 1
                    except Exception:
                        logging.error('Loss is NaN. Could not load last model and corresponding optimizer from {}'.format(path_current_model))
                        logging.error('The reason for this is that not even the first epoch had stable resuls. Aborting.')
                        raise ValueError('Loss is NaN. First training epoch did not have stable results.')
                    if phase_state.grad_norm_last_reduced_counter > 2:
                        train_cfg.clip_grad_norm = train_cfg.clip_grad_norm * 0.7
                        logging.info('Reducing clip_grad_norm to {}'.format(train_cfg.clip_grad_norm))
                        phase_state.grad_norm_last_reduced_counter = 0
                else:
                    model.load(path=path_best_model, device=device)
                    optimizer.load_state_dict(torch.load(path_optimizer_best_model))
                    logging.warning('Loss is NaN. Loaded last best model and corresponding optimizer from {}'.format(path_best_model))
                    mlflow_proxy.log_metric('loss_nan_reload', 1, step=epoch)
                    if phase_state.nan_counter > 55:
                        logging.error('Loss is NaN for more than 55 epochs, even after reloading last best model. Aborting training.')
                        raise ValueError('Loss is NaN for more than 55 epochs, even after reloading last best model. Aborting training.')
            else:
                logging.warning('Loss is NaN. Continuing with current model and optimizer as reload_model_if_loss_nan is False')
            phase_state.nan_counter += 1
        else:
            mlflow_proxy.log_metric('loss_nan_reload', 0, step=epoch)
            phase_state.nan_counter = 0
            phase_state.grad_norm_last_reduced_counter = 0
            model.save(path=path_current_model)
            torch.save(optimizer.state_dict(), path_current_optimizer)
    else:
        if live_state.pre_train is False:
            activate_deterministic_mode = train_cfg.activate_deterministic_mode_after_this_phase and flag_break_after_epoch
        else:
            activate_deterministic_mode = False
        ret_vals_train, dataloader_iters['train'] = test_or_validate_one_epoch(
            model,
            dataloaders['train'],
            train_cfg,
            live_state.pre_train,
            device,
            all_batches=False,
            return_model_outputs=False,
            activate_deterministic_mode=activate_deterministic_mode,
            data_iter=dataloader_iters['train']
        )
        if activate_deterministic_mode:
            logging.info('Activated deterministic mode')
            phase_state.deterministic_mode_active = True
            model.save(path=path_best_model)
            logging.info('Saved model with deterministic mode activated to {}'.format(path_best_model))
        phase_state.first_epoch_is_evaluation = False
        ret_vals_train['ode_calls_backward'] = 0
        ret_vals_train['seq_len_now'] = train_cfg.seq_len_train
    return ret_vals_train, dataloader_iters


def _evaluate_phase_contexts(
    live_state: LiveTrainingState,
    dataloaders: dict,
    dataloader_iters: dict,
    ret_vals_train: dict,
    epoch: int,
    flag_break_after_epoch: bool,
    flag_max_epoch: bool,
):
    model = live_state.model
    train_cfg = live_state.train_cfg
    pre_train = live_state.pre_train
    device = live_state.device
    phase_state = live_state.phase_state
    optimizer = live_state.optimizer
    early_stopping = live_state.early_stopping
    lr_schedulers = live_state.lr_schedulers
    cfg = live_state.cfg

    mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_train, 'train', pre_train), step=epoch)
    early_stopping_metric_name = None
    try:
        ret_vals_validation = test_or_validate_one_epoch(
            model,
            dataloaders['validation'],
            train_cfg,
            pre_train,
            device,
            all_batches=True,
            return_model_outputs=False,
            data_iter=dataloader_iters['validation'],
        )
        if lr_schedulers and 'plateau' in lr_schedulers.keys():
            val_loss = ret_vals_validation.get('loss', None)
            if val_loss is not None and not (np.isnan(val_loss) or np.isinf(val_loss)):
                lr_schedulers['plateau'].step(val_loss)
        early_stopping_metric_name, corresponding_metric_value = _get_early_stopping_corresponding_metric(ret_vals_validation)
        early_stopping(
            ret_vals_validation['loss'],
            model,
            epoch,
            optimizer,
            corresponding_loss=corresponding_metric_value,
        )
        if ret_vals_validation['loss'] < 2 * ret_vals_train['loss']:
            phase_state.stable_epochs += 1
            if phase_state.flag_out_of_seq_len_increase is False and pre_train is False:
                logging.info('\t \t \t Stable seq_len_increase epochs: {}/{}'.format(phase_state.stable_epochs, train_cfg.seq_len_increase_abort_after_n_stable_epochs))
        else:
            phase_state.stable_epochs = 0
        mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_validation, 'validation', pre_train), step=epoch)
    except AssertionError as e:
        if 'non-finite values in' in str(e):
            logging.warning('Error in validation: {}'.format(e))
            ret_vals_validation = {key: float('nan') for key in ret_vals_train.keys()}
        else:
            raise e

    try:
        ret_vals_test, dataloader_iters['test'] = test_or_validate_one_epoch(
            model,
            dataloaders['test'],
            train_cfg,
            pre_train,
            device,
            all_batches=False,
            return_model_outputs=False,
            data_iter=dataloader_iters['test'],
        )
    except AssertionError as e:
        if 'non-finite values in' in str(e):
            logging.warning('Error in test: {}'.format(e))
            ret_vals_test = {key: float('nan') for key in ret_vals_train.keys()}
        else:
            raise e
    mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_test, 'test', pre_train), step=epoch)

    ret_vals_ref = None
    if dataloaders['ref'] is not None:
        if epoch % cfg.nn_model.training.ref_and_testnorm_every_n_epochs == 0 or flag_break_after_epoch or flag_max_epoch or phase_state.first_epoch_is_evaluation:
            try:
                logging.info('Testing ref dataset')
                ret_vals_ref, dataloader_iters['ref'] = test_or_validate_one_epoch(
                    model,
                    dataloaders['ref'],
                    train_cfg,
                    pre_train,
                    device,
                    all_batches=False,
                    return_model_outputs=False,
                    data_iter=dataloader_iters['ref'],
                )
            except AssertionError as e:
                if 'non-finite values in' in str(e):
                    logging.warning('Error in ref test: {}'.format(e))
                    ret_vals_ref = {key: float('nan') for key in ret_vals_train.keys()}
                else:
                    raise e
            res = append_context_to_dict_keys(ret_vals_ref, 'ref', pre_train)
            logging.info(res)
            mlflow_proxy.log_metrics(res, step=epoch)

    ret_vals_testnorm = None
    if dataloaders['testnorm'] is not None:
        if epoch % cfg.nn_model.training.ref_and_testnorm_every_n_epochs == 0 or flag_break_after_epoch or flag_max_epoch or phase_state.first_epoch_is_evaluation:
            try:
                logging.info('Testing testnorm dataset')
                ret_vals_testnorm, dataloader_iters['testnorm'] = test_or_validate_one_epoch(
                    model,
                    dataloaders['testnorm'],
                    train_cfg,
                    pre_train,
                    device,
                    all_batches=False,
                    return_model_outputs=False,
                    data_iter=dataloader_iters['testnorm'],
                )
            except AssertionError as e:
                if 'non-finite values in' in str(e):
                    logging.warning('Error in testnorm test: {}'.format(e))
                    ret_vals_testnorm = {key: float('nan') for key in ret_vals_train.keys()}
                else:
                    raise e
            res = append_context_to_dict_keys(ret_vals_testnorm, 'testnorm', pre_train)
            logging.info(res)
            mlflow_proxy.log_metrics(res, step=epoch)

    return {
        'validation': ret_vals_validation,
        'test': ret_vals_test,
        'ref': ret_vals_ref,
        'testnorm': ret_vals_testnorm,
        'early_stopping_metric_name': early_stopping_metric_name,
        'dataloader_iters': dataloader_iters,
    }


def _update_phase_control_state(
    live_state: LiveTrainingState,
    dataloaders: dict,
    ret_vals_train: dict,
    ret_vals_validation: dict,
    ret_vals_test: dict,
    ret_vals_ref: dict | None,
    ret_vals_testnorm: dict | None,
    early_stopping_metric_name: str | None,
    epoch: int,
    flag_break_after_epoch: bool,
    batches_per_epoch: int,
) -> bool:
    optimizer = live_state.optimizer
    early_stopping = live_state.early_stopping
    phase_state = live_state.phase_state
    train_cfg = live_state.train_cfg
    pre_train = live_state.pre_train
    job_idx = live_state.job_idx
    model = live_state.model

    mlflow_proxy.log_metric('lr', optimizer.param_groups[0]['lr'], step=epoch)
    mlflow_proxy.log_metric('Stable_epochs', phase_state.stable_epochs, step=epoch)
    progress_string = model.get_progress_string(ret_vals_train, ret_vals_validation, ret_vals_test, pre_train)
    logging.info('Epoch: {}/{} EarlyStopping: {}/{} |-| {}'.format(epoch+1, phase_state.epoch_stop, early_stopping.counter, early_stopping.patience, progress_string))
    if flag_break_after_epoch is True:
        mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_train, 'train_job_{}_final'.format(job_idx-1), pre_train), step=epoch)
        mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_validation, 'validation_job_{}_final'.format(job_idx-1), pre_train), step=epoch)
        mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_test, 'test_job_{}_final'.format(job_idx-1), pre_train), step=epoch)
        if dataloaders['ref'] is not None:
            mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_ref, 'ref_job_{}_final'.format(job_idx-1), pre_train), step=epoch)
        if dataloaders['testnorm'] is not None:
            mlflow_proxy.log_metrics(append_context_to_dict_keys(ret_vals_testnorm, 'testnorm_job_{}_final'.format(job_idx-1), pre_train), step=epoch)
        return True

    batches_this_phase = (epoch - phase_state.phase_epoch_0 + 1) * batches_per_epoch
    if pre_train is False:
        if batches_this_phase > train_cfg.seq_len_increase_in_batches and phase_state.flag_out_of_seq_len_increase is False:
            logging.info('Out of seq_len_increase_in_batches')
            phase_state.flag_out_of_seq_len_increase = True
            early_stopping.reset_counter()
    mlflow_proxy.log_metric('EarlyStopping_counter', early_stopping.counter, step=epoch)
    if early_stopping.counter == 0:
        mlflow_proxy.log_metric('EarlyStopping_best_loss', early_stopping.best_score, step=epoch)
        if early_stopping_metric_name is not None and early_stopping.corresponding_score is not None:
            mlflow_proxy.log_metric(f'best_{early_stopping_metric_name}', early_stopping.corresponding_score, step=epoch)
    return False


def _save_phase_restart_checkpoint(
    live_state: LiveTrainingState,
    outer_state: OuterTrainingState | None,
    epoch: int,
) -> None:
    if outer_state is None:
        return
    inner_checkpoint = live_state.save_checkpoint(epoch + 1)
    if inner_checkpoint is not None:
        outer_state.save_checkpoint(job_idx=live_state.job_idx, next_epoch_anchor=epoch + 1)

def train_one_phase(
    cfg: train_test_config_class,
    model: torch.nn.Module,
    dataloaders: dict,
    train_cfg: base_training_settings_class,
    test: bool,
    pre_train: bool,
    job_idx: int,
    epoch_0: int = 0,
    restart_state: InnerTrainingStateCheckpoint | None = None,
    restart_manager: Path | None = None,
    live_state: LiveTrainingState | None = None,
    outer_state: OuterTrainingState | None = None,
):
    logging.info('Start next training phase....')
    if test is False:
        if live_state is None:
            live_state = _prepare_phase_runtime(
                cfg,
                model,
                dataloaders,
                train_cfg,
                test,
                pre_train,
                job_idx,
                epoch_0,
                restart_state,
                restart_manager,
            )
        else:
            live_state = _prepare_phase_runtime(
                cfg,
                model,
                dataloaders,
                live_state.train_cfg,
                test,
                live_state.pre_train,
                live_state.job_idx,
                live_state.phase_state.phase_epoch_0,
                restart_state,
                restart_manager,
                live_state=live_state,
            )
        phase_state = live_state.phase_state
        device = live_state.device
        early_stopping = live_state.early_stopping
        best_model_path = live_state.path_best_model
        max_epochs = live_state.max_epochs
        batches_per_epoch = live_state.batches_per_epoch
        if early_stopping is None or batches_per_epoch is None:
            raise ValueError("Phase runtime objects were not fully initialized")
        '''Training'''
        try:
            # persistent iterators over dataloaders per context across epochs
            dataloader_iters = {ctx: None for ctx, dl in dataloaders.items() if dl is not None}
            for epoch in range(phase_state.epoch_start, phase_state.phase_epoch_0 + max_epochs): # the upper range is a maximum value, and can be changed during training and escaped with if...break
                if epoch == phase_state.epoch_stop:
                    break
                stop_flags = _compute_phase_stop_flags(phase_state, early_stopping, train_cfg, epoch)
                flag_break_after_epoch = _apply_phase_stop_flags(
                    stop_flags,
                    job_idx,
                    train_cfg,
                    model,
                    best_model_path,
                    device,
                )
                _update_phase_epoch_stop_for_seq_len(live_state, batches_per_epoch, epoch)
                ret_vals_train, dataloader_iters = _run_training_epoch_or_eval_epoch(
                    live_state,
                    dataloaders,
                    dataloader_iters,
                    epoch,
                    flag_break_after_epoch,
                )
                eval_results = _evaluate_phase_contexts(
                    live_state,
                    dataloaders,
                    dataloader_iters,
                    ret_vals_train,
                    epoch,
                    flag_break_after_epoch,
                    stop_flags['max_epoch'],
                )
                dataloader_iters = eval_results['dataloader_iters']
                should_break = _update_phase_control_state(
                    live_state,
                    dataloaders,
                    ret_vals_train,
                    eval_results['validation'],
                    eval_results['test'],
                    eval_results['ref'],
                    eval_results['testnorm'],
                    eval_results['early_stopping_metric_name'],
                    epoch,
                    flag_break_after_epoch,
                    batches_per_epoch,
                )
                if should_break:
                    break
                _save_phase_restart_checkpoint(live_state, outer_state, epoch)
        except KeyboardInterrupt:
            logging.info('Interrupted by user')
            mlflow_proxy.set_tag_if_active('ended by', 'keyboard interrupt')
            # load the last checkpoint with the best model
            try:
                model.load(path=best_model_path, device=device)
            except:
                logging.warning('Could not load best model from {}'.format(best_model_path))
                for i in range(job_idx, 0):
                    best_model_path = filepaths.filepath_model_current_hydra_output(i)
                    try:
                        model.load(path=best_model_path, device=device)
                        logging.info('loaded best model from {}'.format(best_model_path))
                        break
                    except:
                        logging.warning('Could not load best model from {}'.format(best_model_path))
            logging.info('loaded best model from {}'.format(best_model_path))
        mlflow_proxy.log_metric('job_{}_final_epoch'.format(job_idx), value=epoch)
    return epoch + 1

def main():
    """Entry point for (B)NODE training via Hydra CLI.
    
    Initializes Hydra configuration system and launches train_all_phases with
    validated config. Auto-detects config directory and uses 'train_test_ode'
    as the default config name.
    
    This function is registered as 'trainer' in pyproject.toml, enabling
    command-line execution via::
    
        uv run trainer [config_overrides]
    
    Examples:
        See module docstring for usage examples.
    
    Side Effects:
        - Registers config store with Hydra
        - Auto-detects config directory from filepaths
        - Launches Hydra-decorated train_all_phases
    """
    get_config_store()
    config_dir = filepaths.config_dir_auto_recognize()
    config_name = 'train_test_ode'
    hydra.main(config_path=str(config_dir.absolute()), config_name=config_name, version_base=None)(train_all_phases)()

if __name__ == '__main__':
    main()
