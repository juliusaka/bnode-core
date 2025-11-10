"""Variational Autoencoder (VAE) architecture for timeseries reconstruction.

This module implements a Variational Autoencoder with parameter conditioning for 
timeseries data (states and outputs). The architecture supports multiple modes:

- Standard VAE: Encoder-Decoder with latent space
- PELS-VAE: Parameter-conditioned VAE with Regressor for mu/logvar prediction
- Feed-forward NN: Direct mapping from parameters to timeseries (bypasses latent space)

The model can reconstruct timeseries from either the encoder (during training) or from
the regressor (during testing/prediction), enabling parameter-conditioned generation.

It is intedend to be used for task that model `physical parameters --> complete timeseries`, e.g. the transient
response of a RC circuit with fixed initial condition on different parameter values `R,L,C`.

Attention:
    This documentation is AI generated. Be aware of possible inaccuricies.

Key components:

    - Encoder: Maps timeseries (states + outputs) to latent distribution (mu, logvar)
    - Decoder: Maps latent samples (and optionally parameters) to reconstructed timeseries
    - Regressor: Maps parameters to latent distribution for parameter-conditioned generation
    - Normalization: Time-series and parameter normalization layers

Loss function:

    loss = mse_loss + beta * kl_loss + regressor_loss
    or with capacity scheduling:
    loss = mse_loss + gamma * |kl_loss - capacity| + regressor_loss
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
import time
import os
import sys
import logging

from bnode_core.nn.nn_utils.normalization import NormalizationLayerTimeSeries, NormalizationLayer1D
from bnode_core.nn.nn_utils.kullback_leibler import kullback_leibler
from bnode_core.nn.nn_utils.count_parameters import count_parameters

class Encoder(nn.Module):
    """Encoder network mapping timeseries to latent distribution parameters.
    
    Maps concatenated states and outputs to mean (mu) and log-variance (logvar) 
    of a multivariate Gaussian distribution in latent space. Uses a multi-layer 
    perceptron (MLP) with configurable depth and hidden dimensions.
    
    Architecture:

        Flatten -> Linear(n_channels*seq_len, hidden_dim) -> Activation
        -> [Linear(hidden_dim, hidden_dim) -> Activation] x (n_layers-2)
        -> Linear(hidden_dim, 2*bottleneck_dim) -> Reshape to [mu, logvar]
    
    Attributes:
        bottleneck_dim: Dimensionality of the latent space.
        flatten: Flattens input timeseries to 1D.
        linear: Sequential MLP mapping flattened input to 2*bottleneck_dim outputs.
    """
    
    def __init__(self, n_channels: int, seq_len: int, hidden_dim: int, bottleneck_dim: int,
                 activation: nn.Module = nn.ReLU, n_layers: int = 3):
        """Initialize the Encoder network.
        
        Args:
            n_channels: Number of input channels (states + outputs concatenated).
            seq_len: Length of the timeseries sequence.
            hidden_dim: Number of hidden units in intermediate layers.
            bottleneck_dim: Dimensionality of latent space (output is 2*bottleneck_dim for mu and logvar).
            activation: Activation function class (default: nn.ReLU).
            n_layers: Total number of linear layers (minimum 2, includes input and output layers).
        """
        super().__init__()
        # save dimensions of output
        self.bottleneck_dim = bottleneck_dim

        self.flatten = nn.Flatten()

        # construct MLP
        modules = [
            nn.Linear(n_channels*seq_len, hidden_dim),
            activation(),
        ]
        if n_layers < 2:
            logging.warning('n_layers must be at least 2, setting n_layers to 2')
        for i in range(n_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        modules.append(nn.Linear(hidden_dim, 2*bottleneck_dim))
        self.linear = nn.Sequential(*modules)  

    def forward(self, x):
        """Encode timeseries to latent distribution parameters.
        
        Args:
            x: Input timeseries tensor of shape (batch, n_channels, seq_len).
        
        Returns:
            Tuple of (mu, logvar) where:

                - mu: Mean of latent distribution, shape (batch, bottleneck_dim)
                - logvar: Log-variance of latent distribution, shape (batch, bottleneck_dim)
        """
        x = self.flatten(x)
        latent = self.linear(x)
        latent = torch.reshape(latent, (-1, 2, self.bottleneck_dim))
        mu, logvar = latent[:,0], latent[:,1]
        return mu, logvar

class Decoder(nn.Module):
    """Decoder network for VAE, generating timeseries from latent vectors.
    
    The decoder maps latent vectors (and optionally system parameters) back to timeseries
    data. It supports three modes:

    - Standard VAE: z_latent → timeseries
    - PELS-VAE: (z_latent, parameters) → timeseries (params_to_decoder=True)
    - Feed-forward: parameters → timeseries (bottleneck_dim=0, params_to_decoder=True)
    
    Architecture: Linear (latent+params → hidden) → MLP → Linear (hidden → n_channels*seq_len) → Reshape
    
    Attributes:
        channels: Number of output channels in reconstructed timeseries.
        seq_len: Length of output timeseries sequence.
        params_to_decoder: If True, concatenate normalized parameters to latent vector as decoder input.
        param_normalization: Normalization layer for parameters (if params_to_decoder=True).
        feed_forward_nn: If True, decoder operates in feed-forward mode (no latent vector).
        linear: Sequential MLP mapping latent (+ params) to flattened timeseries.
    """
    
    def __init__(self, n_channels: int, seq_len: int, hidden_dim: int, bottleneck_dim: int,
                 activation: nn.Module = nn.ReLU, n_layers: int = 3, params_to_decoder = False, 
                 param_dim: int = None):
        """Initialize the Decoder network.
        
        Args:
            n_channels: Number of output channels in reconstructed timeseries.
            seq_len: Length of output timeseries sequence.
            hidden_dim: Number of hidden units in intermediate layers.
            bottleneck_dim: Dimensionality of latent space input (0 for feed-forward mode).
            activation: Activation function class (default: nn.ReLU).
            n_layers: Total number of linear layers (minimum 2).
            params_to_decoder: If True, concatenate system parameters to latent input (PELS-VAE mode).
            param_dim: Dimensionality of parameter vector (required if params_to_decoder=True).
        """
        super().__init__()

        # save dimensions of output
        self.channels = n_channels
        self.seq_len = seq_len
        self.params_to_decoder = params_to_decoder
        if params_to_decoder:
            assert param_dim is not None, 'param_dim must be specified if params_to_decoder is True'
            self.param_normalization = NormalizationLayer1D(num_features=param_dim)

        self.feed_forward_nn = True if bottleneck_dim == 0 else False
        if self.feed_forward_nn:
            assert params_to_decoder is True, 'params_to_decoder must be True if bottleneck_dim is 0'
        # construct MLP
        modules = [
            nn.Linear(bottleneck_dim if params_to_decoder is False else bottleneck_dim + param_dim, hidden_dim),
            activation(),
        ]
        if n_layers < 2:
            logging.warning('n_layers must be at least 2, setting n_layers to 2')
        for i in range(n_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        modules.append(nn.Linear(hidden_dim, n_channels*seq_len))
        self.linear = nn.Sequential(*modules)  
              
    def forward(self, z_latent, param = None):
        """Decode latent vector (and optionally parameters) to timeseries.
        
        Args:
            z_latent: Latent vector of shape (batch, bottleneck_dim).
            param: System parameters of shape (batch, param_dim) (required if params_to_decoder=True).
        
        Returns:
            Reconstructed timeseries tensor of shape (batch, n_channels, seq_len).
        """
        if self.params_to_decoder:
            param = self.param_normalization(param)
            x = self.linear(torch.cat((z_latent, param), dim=1)) if not self.feed_forward_nn else self.linear(param)
        else:
            x = self.linear(z_latent)
        x = torch.reshape(x,(-1, self.channels, self.seq_len))
        return x
    
class Regressor(nn.Module):
    """Regressor network mapping system parameters to latent distribution.
    
    Used in PELS-VAE mode to predict latent distribution parameters (mu, logvar) 
    directly from system parameters, without requiring timeseries input. This allows
    the VAE to learn relationships between system parameters and latent representations.
    
    Architecture: 
        
        Normalize params → Linear (params → hidden) → MLP → Linear (hidden → 2*bottleneck_dim) → Reshape to (mu, logvar)
    
    Attributes:
        bottleneck_dim: Dimensionality of the latent space.
        normalization: Normalization layer for input parameters.
        linear: Sequential MLP mapping parameters to 2*bottleneck_dim outputs.
    """
    
    def __init__(self, parameter_dim: int, hidden_dim: int, 
                 bottleneck_dim: int, activation: nn.Module = nn.ReLU, 
                 n_layers: int = 3):
        """Initialize the Regressor network.
        
        Args:
            parameter_dim: Dimensionality of input parameter vector.
            hidden_dim: Number of hidden units in intermediate layers.
            bottleneck_dim: Dimensionality of latent space (output is 2*bottleneck_dim for mu and logvar).
            activation: Activation function class (default: nn.ReLU).
            n_layers: Total number of linear layers (minimum 2).
        """
        super().__init__()
        # save dimensions of output
        self.bottleneck_dim = bottleneck_dim

        self.normalization = NormalizationLayer1D(num_features=parameter_dim)
        
        # construct MLP
        modules = [
            nn.Linear(parameter_dim, hidden_dim),
            activation(),
        ]
        if n_layers < 2:
            logging.warning('n_layers must be at least 2, setting n_layers to 2')
        for i in range(n_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        modules.append(nn.Linear(hidden_dim, 2*bottleneck_dim))
        self.linear = nn.Sequential(*modules)

    def forward(self, param):
        """Predict latent distribution parameters from system parameters.
        
        Args:
            param: System parameters of shape (batch, parameter_dim).
        
        Returns:
            Tuple of (mu, logvar) where:
                - mu: Predicted mean of latent distribution, shape (batch, bottleneck_dim)
                - logvar: Predicted log-variance of latent distribution, shape (batch, bottleneck_dim)
        """
        param = self.normalization(param)
        latent = self.linear(param)
        latent = torch.reshape(latent,(-1, 2, self.bottleneck_dim))
        mu, logvar = latent[:,0], latent[:,1]
        return mu, logvar
    
class VAE(nn.Module):
    """Variational Autoencoder for timeseries modeling with parameter conditioning.
    
    This class implements three operational modes:

    1. **Standard VAE**: Encodes timeseries to latent space, decodes back to timeseries.
       Uses both Encoder and Regressor to predict latent distributions.
    2. **PELS-VAE** (params_to_decoder=True): Decoder receives both latent vector and 
       system parameters, allowing parameter-conditioned reconstruction.
    3. **Feed-forward NN** (feed_forward_nn=True): Bypasses latent space entirely,
       directly mapping parameters to timeseries outputs.
    
    The model jointly trains:

    - Encoder: timeseries → (mu_encoder, logvar_encoder)
    - Regressor: parameters → (mu_regressor, logvar_regressor)
    - Decoder: latent vector (+ params) → timeseries
    
    During training, reconstruction uses Encoder's latent distribution.
    During prediction, reconstruction uses Regressor's latent distribution.
    
    Attributes:
        n_channels: Total number of channels (n_states + n_outputs).
        n_states: Number of state channels.
        n_outputs: Number of output channels.
        timeseries_normalization: Normalization layer for timeseries data.
        feed_forward_nn: If True, operates in feed-forward mode (no latent space).
        Regressor: Parameter-to-latent network (if not feed_forward_nn).
        Encoder: Timeseries-to-latent network (if not feed_forward_nn).
        Decoder: Latent-to-timeseries network.
    """
    
    def __init__(self, n_states: int, n_outputs: int, seq_len: int, parameter_dim: int, 
                 hidden_dim: int, bottleneck_dim: int, activation: nn.Module = nn.ReLU, 
                 n_layers: int = 3, params_to_decoder=False, feed_forward_nn = False):
        """Initialize the VAE model.
        
        Args:
            n_states: Number of state channels in timeseries.
            n_outputs: Number of output channels in timeseries.
            seq_len: Length of timeseries sequence.
            parameter_dim: Dimensionality of system parameters.
            hidden_dim: Number of hidden units in all sub-networks.
            bottleneck_dim: Dimensionality of latent space.
            activation: Activation function class (default: nn.ReLU).
            n_layers: Number of layers in all sub-networks (minimum 2).
            params_to_decoder: If True, decoder receives parameters as additional input (PELS-VAE mode).
            feed_forward_nn: If True, operate in feed-forward mode without latent space.
        """
        if feed_forward_nn is True:
            if params_to_decoder is False:
                Warning('params_to_decoder is set to False, but feed_forward_nn is set to True. Setting params_to_decoder to True')
        super().__init__()
        self.n_channels = n_states + n_outputs
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.timeseries_normalization = NormalizationLayerTimeSeries(n_channels=self.n_channels)
        self.feed_forward_nn = feed_forward_nn
        
        if feed_forward_nn is False:
            self.Regressor = Regressor(parameter_dim, hidden_dim, bottleneck_dim, activation, n_layers)
            self.Encoder = Encoder(self.n_channels, seq_len, hidden_dim,
                                bottleneck_dim, activation, n_layers)
            self.Decoder = Decoder(self.n_channels, seq_len, hidden_dim,
                                bottleneck_dim, activation, n_layers,
                                params_to_decoder, parameter_dim)
        else:
            _bottleneck_dim = 0
            _params_to_decoder = True
            self.Decoder = Decoder(self.n_channels, seq_len, hidden_dim,
                                   _bottleneck_dim, activation, n_layers,
                                      _params_to_decoder, parameter_dim)
        logging.info('VAE with n_channels = {}, seq_len = {}, parameter_dim = {}, \
                     hidden_dim = {}, bottleneck_dim = {}, activation = {}, n_layers = {}, params to decoder: {}'.format(
                         self.n_channels, seq_len, parameter_dim, hidden_dim, bottleneck_dim, activation, n_layers, self.Decoder.params_to_decoder))
        logging.info('VAE initialized with {} parameters'.format(count_parameters(self)))

    def reparametrize(self, mu, logvar):
        """Apply reparametrization trick to sample from latent distribution.
        
        Samples z ~ N(mu, exp(0.5 * logvar)) using z = mu + eps * std, where eps ~ N(0, I).
        This allows backpropagation through the sampling operation.
        
        Args:
            mu: Mean of latent distribution, shape (batch, bottleneck_dim).
            logvar: Log-variance of latent distribution, shape (batch, bottleneck_dim).
        
        Returns:
            Sampled latent vector z of shape (batch, bottleneck_dim).
        """
        # if device.type == 'cuda':
        #     eps = torch.autograd.Variable(torch.cuda.FloatTensor(mu.shape).normal_())
        # else: 
        #     eps = torch.autograd.Variable(torch.FloatTensor(mu.shape).normal_())
        eps = torch.randn_like(mu, device=mu.device)
        std = logvar.mul(0.5).exp()
        z_latent = eps.mul(std).add_(mu)
        return z_latent

    def forward(self, states, outputs, params, train=True, predict = False, n_passes: int = 1, test_with_zero_eps: bool = False, device = None):
        """Perform forward pass through the VAE network.
        
        Three operational modes based on flags:

        1. Training (train=True, predict=False): Encode timeseries, reconstruct using Encoder's latent distribution
        2. Testing (train=False, predict=False): Encode timeseries, reconstruct using Regressor's latent distribution
        3. Prediction (predict=True, train=False): Skip Encoder, reconstruct using Regressor's latent distribution only
        
        Args:
            states: State timeseries of shape (batch, n_states, seq_len).
            outputs: Output timeseries of shape (batch, n_outputs, seq_len).
            params: System parameters of shape (batch, parameter_dim).
            train: If True, use Encoder's latent distribution for reconstruction.
            predict: If True, bypass Encoder and reconstruct from parameters only.
            n_passes: Number of decoder passes to average (for stochastic predictions).
            test_with_zero_eps: If True during testing, use mu directly (zero variance sampling).
            device: Device for tensor operations.
        
        Returns:
            Tuple of (x, x_hat, states_hat, outputs_hat, mu_encoder, logvar_encoder, 
                     mu_regressor, logvar_regressor, retvals_norm) where:

                - x: Concatenated input timeseries (states + outputs)
                - x_hat: Reconstructed timeseries
                - states_hat: Reconstructed states
                - outputs_hat: Reconstructed outputs
                - mu_encoder: Encoder's predicted latent mean
                - logvar_encoder: Encoder's predicted latent log-variance
                - mu_regressor: Regressor's predicted latent mean
                - logvar_regressor: Regressor's predicted latent log-variance
                - retvals_norm: Dictionary of normalized versions of above tensors
        """
        if self.feed_forward_nn is False:
            if predict:
                assert not train, 'predict and train cannot be true at the same time'
            else:
                # concatenate states and outputs
                x = torch.cat((states, outputs), dim=1)
                x_norm = self.timeseries_normalization(x)
                states_norm = x_norm[:,:self.n_states]
                outputs_norm = x_norm[:,self.n_states:]
                mu_encoder, logvar_encoder = self.Encoder(x_norm)
            mu_regressor, logvar_regressor = self.Regressor(params)
            # assign mu, logvar based on if train or not
            if train:
                mu = mu_encoder
                logvar = logvar_encoder
            else:
                mu = mu_regressor
                logvar = logvar_regressor
            # if predict, we need some dummy values for mu_encoder and logvar_encoder
            if predict:
                mu_encoder = torch.ones_like(mu_encoder, device=device) * np.inf
                logvar_encoder = torch.ones_like(logvar_encoder, device=device) * np.inf
            # perform multiple passes through decoder
            x_pass = []
            x_pass_norm = []
            for _ in range(n_passes):
                if train or not test_with_zero_eps:
                    z_latent = self.reparametrize(mu, logvar)
                else:
                    z_latent = mu
                if self.Decoder.params_to_decoder:
                    x_i_hat_norm = self.Decoder(z_latent, params)
                else:
                    x_i_hat_norm = self.Decoder(z_latent)
                x_i_hat = self.timeseries_normalization(x_i_hat_norm, denormalize = True)
                x_pass.append(x_i_hat)
                x_pass_norm.append(x_i_hat_norm)
            # stack along new dimension 1 and take mean along that dimension
            x_hat = torch.stack(x_pass, dim=0).mean(dim=0)
            x_hat_norm = torch.stack(x_pass_norm, dim=0).mean(dim=0)
            # unpack x
            states_hat, outputs_hat = torch.split(x_hat, [self.n_states, self.n_outputs], dim=1)
            # unpack x_norm
            states_hat_norm, outputs_hat_norm = torch.split(x_hat_norm, [self.n_states, self.n_outputs], dim=1)
            retvals_norm = {
                'x': x_norm,
                'x_hat': x_hat_norm,
                'states': states_norm,
                'outputs': outputs_norm,
                'states_hat': states_hat_norm,
                'outputs_hat': outputs_hat_norm,
            }
        else:
            x = torch.cat((states, outputs), dim=1)
            x_norm = self.timeseries_normalization(x)
            states_norm = x_norm[:,:self.n_states]
            outputs_norm = x_norm[:,self.n_states:]
            x_hat_norm = self.Decoder(None, params)
            x_hat = self.timeseries_normalization(x_hat_norm, denormalize = True)
             # unpack x
            states_hat, outputs_hat = torch.split(x_hat, [self.n_states, self.n_outputs], dim=1)
            # unpack x_norm
            states_hat_norm, outputs_hat_norm = torch.split(x_hat_norm, [self.n_states, self.n_outputs], dim=1)
            retvals_norm = {
                'x': x_norm,
                'x_hat': x_hat_norm,
                'states': states_norm,
                'outputs': outputs_norm,
                'states_hat': states_hat_norm,
                'outputs_hat': outputs_hat_norm,
            }
            mu_encoder = torch.ones_like(states_hat_norm, device=device) * np.inf
            logvar_encoder = torch.ones_like(states_hat_norm, device=device) * np.inf
            mu_regressor = torch.ones_like(states_hat_norm, device=device) * np.inf
            logvar_regressor = torch.ones_like(states_hat_norm, device=device) * np.inf

        return x, x_hat, states_hat, outputs_hat, mu_encoder, logvar_encoder, mu_regressor, logvar_regressor, retvals_norm
    
    def predict(self, param):
        """Generate timeseries predictions from system parameters only.
        
        Convenience method for inference mode. Bypasses Encoder and generates
        predictions using only Regressor and Decoder.
        
        Args:
            param: System parameters of shape (batch, parameter_dim).
        
        Returns:
            Same as forward() method with predict=True.
        """
        return self.forward(states=None, outputs=None, params=param, train=False, predict=True)
    
    def save(self, path: Path):
        """Save model state dictionary to disk.
        
        Args:
            path: Path to save the model weights. Parent directories are created if needed.
        """
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.state_dict(), path)
        logging.info('\t \t \tSaved model to {}'.format(path))
    
    def load(self, path: Path, device = None):
        """Load model state dictionary from disk.
        
        Args:
            path: Path to the saved model weights.
            device: Device to map the loaded weights to (e.g., 'cpu', 'cuda').
        """
        self.load_state_dict(torch.load(path, map_location=device))
        logging.info('\tLoaded model from {}'.format(path))

def loss_function(x: torch.tensor, x_hat:torch.tensor, 
                  mu: torch.tensor, mu_hat: torch.tensor, 
                  logvar: torch.tensor, logvar_hat: torch.tensor,
                  beta: float = 1.0, gamma: float = 1000.0, 
                  capacity: float = None,
                  reduce: bool = True,
                  device = None):
    """Compute composite loss function for VAE training.
    
    Implements the PELS-VAE loss combining reconstruction, KL divergence, and regressor losses.
    Supports two modes:

    1. Standard β-VAE: loss = mse_loss + β * kl_loss + regressor_loss
    2. Capacity-constrained: loss = mse_loss + γ * |kl_loss - capacity| + regressor_loss
    
    The regressor loss ensures that the Regressor's predicted latent distribution
    matches the Encoder's latent distribution, enabling parameter-to-latent predictions.
    
    Args:
        x: Original timeseries (normalized), shape (batch, n_channels, seq_len).
        x_hat: Reconstructed timeseries (normalized), shape (batch, n_channels, seq_len).
        mu: Encoder's latent mean, shape (batch, bottleneck_dim).
        mu_hat: Regressor's latent mean, shape (batch, bottleneck_dim).
        logvar: Encoder's latent log-variance, shape (batch, bottleneck_dim).
        logvar_hat: Regressor's latent log-variance, shape (batch, bottleneck_dim).
        beta: Weight for KL divergence term (ignored if capacity is not None).
        gamma: Weight for capacity constraint term (used only if capacity is not None).
        capacity: Target KL divergence capacity. If None, uses standard β-VAE loss.
        reduce: If True, return scalar losses. If False, return per-sample losses.
        device: Device for tensor operations.
    
    Returns:
        Tuple of (loss, mse_loss, kl_loss, regressor_loss) where:

            - loss: Total loss (inf if reduce=False)
            - mse_loss: Mean squared error between x and x_hat
            - kl_loss: KL divergence KL(N(mu, exp(logvar)) || N(0, I))
            - regressor_loss: MSE between (mu, logvar) and (mu_hat, logvar_hat)
    
    Notes:
        The capacity constraint encourages the model to use exactly 'capacity' nats
        of information in the latent space, preventing posterior collapse or over-regularization.
    """
    mse = nn.MSELoss(reduction='mean' if reduce else 'none')
    mse_loss = mse(x_hat, x)
    kl_loss = kullback_leibler(mu, logvar, per_dimension=not reduce, reduce=reduce)
    regressor_loss = mse(mu_hat, mu) + mse(logvar_hat, logvar)
    if reduce:
        if capacity is None:
            loss = mse_loss + beta * kl_loss + regressor_loss
        else:
            if capacity < 0:
                raise ValueError('capacity must be positive')
            # kl_loss is always positive, so subtracting capacity and 
            # taking the absolute value sets a capacity
            loss = mse_loss + gamma * (kl_loss - capacity).abs() + regressor_loss
    else:
        loss = torch.tensor(np.inf, device=device)
    return loss, mse_loss, kl_loss, regressor_loss

