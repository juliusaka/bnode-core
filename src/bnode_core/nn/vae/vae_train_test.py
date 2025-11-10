"""VAE training and testing pipeline for timeseries modeling.

This module implements the complete training pipeline for Variational Autoencoders (VAE)
with parameter conditioning, supporting standard VAE, PELS-VAE, and feed-forward modes.

Attention:
    This documentation is AI generated. Be aware of possible inaccuricies.

### Command-line Usage

The module uses Hydra for configuration management and MLflow for experiment tracking.
Training is launched via the command line:

```bash
uv run python -m bnode_core.nn.vae.vae_train_test

uv run <path to vae_train_test.py>
```

Configuration files are loaded from `conf/train_test_vae.yaml` (or specified config path).

### Configuration

Key configuration parameters (via Hydra config):

- `dataset_name`: Name of HDF5 dataset to load
- `use_cuda`: Enable CUDA acceleration
- `use_amp`: Enable automatic mixed precision training
- `nn_model.network.*`: Model architecture parameters (hidden_dim, n_latent, activation, etc.)
- `nn_model.training.*`: Training hyperparameters (batch_size, lr, max_epochs, etc.)

### Training Workflow

1. Load HDF5 dataset and create train/validation/test/common_test dataloaders
2. Initialize VAE model with specified architecture
3. Initialize normalization layers on full training dataset
4. Train with:

   - Adam optimizer with learning rate scheduling (ReduceLROnPlateau)
   - Early stopping monitoring validation loss
   - Capacity scheduling for controlled KL divergence growth
   - Automatic mixed precision (AMP) support
   - Gradient clipping for stability
5. Save best model checkpoint based on validation loss
6. Evaluate on all dataset splits and save predictions to HDF5 file
7. Log all metrics and artifacts to MLflow

### Output Files

- `model.pth`: Best model checkpoint (state_dict)
- `dataset_with_predictions.h5`: Copy of input dataset with added model predictions
- `vae_train_test.py`, `vae_architecture.py`: Copies of source files for reproducibility

### Key Features

- **Multi-pass prediction**: Average multiple stochastic forward passes for robust predictions
- **Capacity scheduling**: Gradually increase KL divergence capacity to prevent posterior collapse
- **Early stopping**: Monitor validation loss with configurable patience and threshold
- **MLflow integration**: Automatic logging of metrics, parameters, and artifacts via decorator
- **Reproducibility**: Saves source code and full configuration to output directory
"""

import torch
import hydra
from pathlib import Path
import numpy as np
import sys
import os
import mlflow
import logging
import shutil
import h5py

from bnode_core import filepaths
from bnode_core.nn.nn_utils.kullback_leibler import kullback_leibler, count_populated_dimensions
from bnode_core.nn.vae.vae_architecture import VAE, loss_function
from bnode_core.config import train_test_config_class
from bnode_core.nn.nn_utils.load_data import load_dataset_and_config, make_stacked_dataset
from bnode_core.nn.nn_utils.early_stopping import EarlyStopping
from bnode_core.nn.nn_utils.capacity_scheduler import capacity_scheduler as CapacityScheduler
from bnode_core.utils.hydra_mlflow_decorator import log_hydra_to_mlflow

@log_hydra_to_mlflow
def train(cfg: train_test_config_class) -> float:
    """Train VAE model on timeseries dataset with MLflow tracking.
    
    Complete training pipeline including:

    - Dataset loading and preprocessing
    - Model initialization and normalization layer setup
    - Training loop with early stopping and capacity scheduling
    - Evaluation on all dataset splits
    - Model checkpoint saving and artifact logging
    
    Args:
        cfg: Hydra configuration object containing all training parameters.
            Key sections: dataset_name, use_cuda, use_amp, nn_model.network, nn_model.training
    
    Returns:
        Final MSE loss on test set (float).
    
    Notes:
        - Uses @log_hydra_to_mlflow decorator for automatic MLflow experiment tracking
        - Saves best model based on validation loss
        - Copies dataset to output directory with added model predictions
        - Logs metrics at each epoch: loss, mse_loss, kl_loss, regressor_loss, populated_dims
    """
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')

    # load dataset and config
    dataset, dataset_config = load_dataset_and_config(cfg.dataset_name, cfg.dataset_path)
    
    # make train and test torch tensor datasets
    train_dataset = make_stacked_dataset(dataset, 'train')
    test_dataset = make_stacked_dataset(dataset, 'test')
    validation_dataset = make_stacked_dataset(dataset, 'validation')
    common_test_dataset = make_stacked_dataset(dataset, 'common_test')

    # initialize data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.nn_model.training.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.nn_model.training.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.nn_model.training.batch_size, shuffle=True)
    common_test_loader = torch.utils.data.DataLoader(common_test_dataset, batch_size=cfg.nn_model.training.batch_size, shuffle=True)

    # initialize model
    model = VAE(
        n_states=dataset['train']['states'].shape[1],
        n_outputs=dataset['train']['outputs'].shape[1],
        seq_len=dataset['train']['states'].shape[2],
        parameter_dim=dataset['train']['parameters'].shape[1],
        hidden_dim=cfg.nn_model.network.linear_hidden_dim,
        bottleneck_dim=cfg.nn_model.network.n_latent,
        activation=eval(cfg.nn_model.network.activation),
        n_layers=cfg.nn_model.network.n_linear_layers,
        params_to_decoder=cfg.nn_model.network.params_to_decoder,
        feed_forward_nn=cfg.nn_model.network.feed_forward_nn,
    )
    model.to(device)

    # initialize timeseries_normalization layer on whole dataset
    _states = train_dataset.datasets['states'].to(device)
    _outputs = train_dataset.datasets['outputs'].to(device)
    _parameters = train_dataset.datasets['parameters'].to(device)
    _x = x = torch.cat((_states, _outputs), dim=1)
    model.timeseries_normalization.initialize_normalization(_x)
    model.Regressor.normalization(_parameters) if model.feed_forward_nn is False else None
    del _states, _outputs, _parameters, _x
    logging.info('Initialized timeseries_normalization layer on whole dataset')
    logging.info('Initialized Regressor normalization layer on whole dataset')

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.nn_model.training.lr_start)
    #optimizer = torch.optim.SGD(model.parameters(), lr=cfg.nn_model.training.lr_start)

    # initialize lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            factor=cfg.nn_model.training.lr_scheduler_plateau_gamma,
                                                            patience=cfg.nn_model.training.lr_scheduler_plateau_patience,
                                                            threshold=cfg.nn_model.training.lr_scheduler_threshold,
                                                            threshold_mode=cfg.nn_model.training.lr_scheduler_threshold_mode,
                                                            min_lr=cfg.nn_model.training.lr_min,
                                                            )
    # initialize early stopping
    early_stopping = EarlyStopping(patience=cfg.nn_model.training.early_stopping_patience,
                                      verbose=True,
                                      threshold=cfg.nn_model.training.early_stopping_threshold,
                                      threshold_mode=cfg.nn_model.training.early_stopping_threshold_mode,
                                      path=filepaths.filepath_model_current_hydra_output(),
                                      trace_func=logging.info)
    
    # initialize capacity scheduler
    capacity_scheduler = CapacityScheduler(
        patience = cfg.nn_model.training.capacity_patience,
        capacity_start = cfg.nn_model.training.capacity_start,
        capacity_max=cfg.nn_model.training.capacity_max,
        capacity_increment = cfg.nn_model.training.capacity_increment,
        capacity_increment_mode = cfg.nn_model.training.capacity_increment_mode,
        threshold = cfg.nn_model.training.capacity_threshold,
        threshold_mode = cfg.nn_model.training.capacity_threshold_mode,
        trace_func = logging.info,
        enabled=cfg.nn_model.training.use_capacity
    )
    # initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    logging.info('Training with automatic mixed precision: {}'.format(cfg.use_amp))
    
    # define one model and loss evaluation
    def model_and_loss_evaluation(model, states, outputs, parameters, train=True, n_passes: int = 1, return_model_outputs: bool = False, test_from_regressor: bool = True):
        """Evaluate model forward pass and compute all loss components.
        
        Args:
            model: VAE model instance.
            states: State timeseries tensor, shape (batch, n_states, seq_len).
            outputs: Output timeseries tensor, shape (batch, n_outputs, seq_len).
            parameters: System parameters tensor, shape (batch, parameter_dim).
            train: If True, use Encoder's latent distribution. If False, controlled by test_from_regressor.
            n_passes: Number of stochastic forward passes to average.
            return_model_outputs: If True, return model outputs (predictions, latent variables, raw losses).
            test_from_regressor: If True during testing, use Regressor's latent distribution instead of Encoder's.
        
        Returns:
            Dictionary with keys: loss, mse_loss, kl_loss, regressor_loss, populated_dims.
            If return_model_outputs=True, returns tuple (ret_val, model_outputs) where model_outputs
            contains: mse_loss_raw, kl_loss_raw, regressor_loss_raw, states_hat, outputs_hat, 
            mu, logvar, mu_hat, logvar_hat.
        """
        _train = train if train is True else test_from_regressor # if not training, do the test with either mu, logvar from regressor or from encoder
        x, x_hat, states_hat, outputs_hat, mu, logvar, mu_hat, logvar_hat, normed_values = model(states, outputs, parameters, train=_train, 
                                                                                                 predict = False, n_passes=n_passes, 
                                                                                                 test_with_zero_eps=cfg.nn_model.training.test_with_zero_eps,
                                                                                                 device=device)
        loss, mse_loss, kl_loss, regressor_loss = loss_function(
                    normed_values['x'], normed_values['x_hat'], mu, mu_hat, 
                    logvar, logvar_hat, 
                    beta=cfg.nn_model.training.beta_start, 
                    gamma=cfg.nn_model.training.gamma,
                    capacity= None if cfg.nn_model.training.use_capacity is False
                        else capacity_scheduler.get_capacity(),
                    device=device,
        )
        _populated_dimensions, _ = count_populated_dimensions(mu, logvar, cfg.nn_model.training.count_populated_dimensions_threshold)
        ret_val = {
            'loss': loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'regressor_loss': regressor_loss,
            'populated_dims': _populated_dimensions,
        }
        if return_model_outputs:
            # losses per dim
            _, mse_loss_raw, kl_loss_raw, regressor_loss_raw = loss_function(
                    x, x_hat, mu, mu_hat, 
                    logvar, logvar_hat, 
                    beta=cfg.nn_model.training.beta_start, 
                    gamma=cfg.nn_model.training.gamma,
                    capacity= None,
                    reduce=False
                    )   
            model_outputs = {
                'mse_loss_raw': mse_loss_raw,
                'kl_loss_raw': kl_loss_raw,
                'regressor_loss_raw': regressor_loss_raw,
                'states_hat': states_hat,
                'outputs_hat': outputs_hat,
                'mu': mu,
                'logvar': logvar,
                'mu_hat': mu_hat,
                'logvar_hat': logvar_hat,
            }
        if not train:
            # call value.item() for each value in return_value
            ret_val = dict({key: value.item() for key, value in ret_val.items()})
            if return_model_outputs:
                model_outputs = dict({key: value.cpu().detach().numpy() for key, value in model_outputs.items()})
        return ret_val if not return_model_outputs else (ret_val, model_outputs)
        
    def get_model_inputs(data_loader: torch.utils.data.DataLoader, data: dict = None):
        """Extract model inputs from data loader or data dictionary.
        
        Args:
            data_loader: PyTorch DataLoader (if provided, fetches next batch).
            data: Dictionary with keys 'states', 'outputs', 'parameters' (alternative to data_loader).
        
        Returns:
            Tuple of (states, outputs, parameters) tensors moved to device.
        """
        if data_loader is None:
            assert data is not None, 'Either data_loader or data must be not None'
        else:
            data = next(iter(data_loader))
        # get data from data loader
        states = data['states'].to(device)
        outputs = data['outputs'].to(device)
        parameters = data['parameters'].to(device)
        return states, outputs, parameters
        

    # define train loop for one epoch
    def train_one_epoch(model, train_loader, optimizer, scaler, epoch):
        """Execute one training epoch with gradient updates.
        
        Iterates through all batches in train_loader, computes losses, performs
        backpropagation with gradient clipping and AMP scaling.
        
        Args:
            model: VAE model instance.
            train_loader: PyTorch DataLoader for training data.
            optimizer: PyTorch optimizer.
            scaler: CUDA AMP gradient scaler.
            epoch: Current epoch number (for logging).
        
        Returns:
            Dictionary with training metrics: loss, mse_loss, kl_loss, regressor_loss, populated_dims.
        """
        # get data from train loader
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # get data from data loader
            states, outputs, parameters = get_model_inputs(train_loader)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                ret_vals_train = model_and_loss_evaluation(model, states, outputs, parameters, n_passes=cfg.nn_model.training.n_passes_train, test_from_regressor=cfg.nn_model.training.test_from_regressor)
            loss = ret_vals_train['loss'] if cfg.nn_model.network.feed_forward_nn is False else ret_vals_train['mse_loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.nn_model.training.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        # call value.item() for each value in return_value
        ret_vals_train = dict({key: value.item() for key, value in ret_vals_train.items()})

        return ret_vals_train
    
    def test_or_validate_one_epoch(model, _data_loader, n_passes: int = 1, all_batches: bool = False,
                                   return_model_outputs: bool = False):
        """Evaluate model on validation or test set without gradient computation.
        
        Args:
            model: VAE model instance.
            _data_loader: PyTorch DataLoader for evaluation data.
            n_passes: Number of stochastic forward passes to average.
            all_batches: If True, evaluate on all batches and average metrics. If False, evaluate only first batch.
            return_model_outputs: If True, return model predictions and latent variables.
        
        Returns:
            Dictionary with evaluation metrics: loss, mse_loss, kl_loss, regressor_loss, populated_dims.
            If return_model_outputs=True, returns tuple (ret_vals, model_outputs) where model_outputs
            contains predictions and latent variables for all evaluated batches.
        """
        model.eval()
        # make sure that the data loader is not shuffled by initializing a new data loader
        if all_batches:
            data_loader = torch.utils.data.DataLoader(_data_loader.dataset, batch_size=_data_loader.batch_size, shuffle=False)
        else:
            data_loader = _data_loader
        _ret_vals = []
        _model_outputs = []
        for step, data in enumerate(data_loader):
            states, outputs, parameters = get_model_inputs(data_loader=None, data=data)
            # forward
            with torch.no_grad():
                ret_vals, model_outputs = model_and_loss_evaluation(model, states, outputs, parameters, train=False, n_passes=n_passes, return_model_outputs=True, test_from_regressor = cfg.nn_model.training.test_from_regressor)
            _ret_vals.append(ret_vals)
            _model_outputs.append(model_outputs)
            if all_batches is False:
                break
        # average over all calls
        if all_batches:
            ret_vals = {}
            for key in _ret_vals[0].keys():
                ret_vals[key] = sum([_ret_val[key] for _ret_val in _ret_vals]) / len(_ret_vals)
        else:
            ret_vals = _ret_vals[0]
        # make one tensor from all model outputs
        if return_model_outputs:
            model_outputs = {key: np.concatenate([_batch_output[key] for _batch_output in _model_outputs], axis=0) for key in _model_outputs[0].keys()}
        return ret_vals if not return_model_outputs else (ret_vals, model_outputs)
    
    def append_context_to_dict_keys(dictionary: dict, context: str):
        """Add context suffix to all dictionary keys for MLflow logging.
        
        Args:
            dictionary: Dictionary with metric names as keys.
            context: Suffix to append (e.g., 'train', 'validation', 'test').
        
        Returns:
            New dictionary with keys formatted as 'original_key_context'.
        """
        return dict({'{}_{}'.format(key, context): value for key, value in dictionary.items()})

    # training loop
    _flag_break_next_epoch = False
    for epoch in range(cfg.nn_model.training.max_epochs):
        # train one epoch
        if not _flag_break_next_epoch:
            ret_vals_train = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        else:
            ret_vals_train = test_or_validate_one_epoch(model, train_loader, n_passes=cfg.nn_model.training.n_passes_test)
        # validate one epoch
        ret_vals_validation = test_or_validate_one_epoch(model, validation_loader, n_passes=cfg.nn_model.training.n_passes_test)
        # test one epoch
        ret_vals_test = test_or_validate_one_epoch(model, test_loader, n_passes=cfg.nn_model.training.n_passes_test)
        # lr scheduler step
        if not _flag_break_next_epoch:
            lr_scheduler.step(ret_vals_validation['loss'] if cfg.nn_model.network.feed_forward_nn is False else ret_vals_validation['mse_loss'])
        # early stopping
            early_stopping(ret_vals_validation['loss'] if cfg.nn_model.network.feed_forward_nn is False else ret_vals_validation['mse_loss'],
                           model, epoch, corresponding_loss = ret_vals_train['loss'])
        # capacity scheduler
        capacity_scheduler.update(ret_vals_validation['mse_loss'])
        # log stats with logging
        string = 'Epoch: {}/{} | train/validate/test: {:.4f}/{:.4f}/{:.4f} | mse: {:.4f}/{:.4f}/{:.4f} | kl_loss: {:.4f}/{:.4f}/{:.4f} | regressor_loss: {:.4f}/{:.4f}/{:.4f} | pop. dim: {}/{}/{} | \
            \t\t\t| batches: {} | lr: {:.6f} |'.format(
            epoch, cfg.nn_model.training.max_epochs,
            ret_vals_train['loss'], ret_vals_validation['loss'], ret_vals_test['loss'],
            ret_vals_train['mse_loss'], ret_vals_validation['mse_loss'], ret_vals_test['mse_loss'],
            ret_vals_train['kl_loss'], ret_vals_validation['kl_loss'], ret_vals_test['kl_loss'],
            ret_vals_train['regressor_loss'], ret_vals_validation['regressor_loss'], ret_vals_test['regressor_loss'],
            ret_vals_train['populated_dims'], ret_vals_validation['populated_dims'], ret_vals_test['populated_dims'],
            len(train_loader),
            optimizer.param_groups[0]['lr'])
        string = string + ' capacity: {:.4f} |'.format(capacity_scheduler.get_capacity()) if cfg.nn_model.training.use_capacity else string
        string = string + ' EarlyStopping: {}/{} |'.format(early_stopping.counter, early_stopping.patience)
        logging.info(string)
        # log stats with mlflow
        mlflow.log_metrics(append_context_to_dict_keys(ret_vals_train, 'train'), step=epoch, )
        mlflow.log_metrics(append_context_to_dict_keys(ret_vals_validation, 'validation'), step=epoch)
        mlflow.log_metrics(append_context_to_dict_keys(ret_vals_test, 'test'), step=epoch)
        mlflow.log_metric('lr', optimizer.param_groups[0]['lr'], step=epoch)
        mlflow.log_metric('EarlyStopping_counter', early_stopping.counter, step=epoch)
        mlflow.log_metric('capacity', capacity_scheduler.get_capacity(), step=epoch) if cfg.nn_model.training.use_capacity else None

        # check early stopping
        if early_stopping.early_stop:
            logging.info("Early stopping")
            mlflow.log_param('ended_by', 'early_stopping')
            # let the evaluation run one more time to record the outputs of the best model
            _flag_break_next_epoch = True
            # load the best model
            model.load(filepaths.filepath_model_current_hydra_output(), device=device)
        if _flag_break_next_epoch:
            break
    
    # Check performance of model on all datasets

    # load best model
    model.load(filepaths.filepath_model_current_hydra_output(), device=device)
    
    # close initial dataset
    dataset.close()
    # copy dataset to hydra output directory
    _path = filepaths.filepath_dataset_current_hydra_output()
    shutil.copy(filepaths.filepath_dataset_from_config(cfg.dataset_name, cfg.dataset_path), _path)
    dataset = h5py.File(_path, 'r+')
    # add model outputs to dataset
    for context, dataloader in zip(['train', 'test', 'validation', 'common_test'], [train_loader, test_loader, validation_loader, common_test_loader]):
        ret_vals, model_outputs = test_or_validate_one_epoch(model, dataloader, n_passes=cfg.nn_model.training.n_passes_test, all_batches=True, return_model_outputs=True)

        # log stats with logging
        string = context
        string = string + ': loss: {:.4f} | mse: {:.4f} | kl_loss: {:.4f} | regressor_loss: {:.4f} | pop. dim: {} |'.format(
            ret_vals['loss'],
            ret_vals['mse_loss'],
            ret_vals['kl_loss'],
            ret_vals['regressor_loss'],
            ret_vals['populated_dims'],
        )
        logging.info(string)

        # save loss function values
        for key, value in ret_vals.items():
            dataset.create_dataset(context+'/'+key, data=value) 
        # save reconstructed timeseries and raw loss function values
        for key, value in model_outputs.items():
            dataset.create_dataset(context+'/'+key, data=value)
        # log to mlflow
        mlflow.log_metrics(append_context_to_dict_keys(ret_vals, context), step=epoch)
    dataset.close()

    # save this file and the vae_architecture.py file to hydra output directory
    shutil.copy(Path(__file__), filepaths.dir_current_hydra_output())
    return ret_vals['mse_loss']

def main():
    """Entry point for VAE training via Hydra CLI.
    
    Initializes Hydra configuration system and launches train with validated
    config. Auto-detects config directory and uses 'train_test_vae' as the
    default config name.
    
    This function can be registered in pyproject.toml, enabling command-line
    execution via a custom script name.
    
    Examples:
        Run from command line::
        
            uv run python -m bnode_core.nn.vae.vae_train_test
        
        With config overrides::
        
            uv run python -m bnode_core.nn.vae.vae_train_test \\
                nn_model.training.lr_start=0.0001 \\
                dataset_name=my_dataset
        
    
    Side Effects:
        - Registers config store with Hydra
        - Auto-detects config directory from filepaths
        - Launches Hydra-decorated train function
    """
    from bnode_core.config import get_config_store
    cs = get_config_store()
    config_dir = filepaths.config_dir_auto_recognize()
    config_name = 'train_test_vae'
    hydra.main(config_path=str(config_dir.absolute()), config_name=config_name, version_base=None)(train)()

if __name__ == '__main__':
    main()