import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
import time
import os
import sys
import logging
from networks.src.normalization import NormalizationLayerTimeSeries, NormalizationLayer1D
from networks.src.kullback_leibler import kullback_leibler
from networks.src.count_parameters import count_parameters

class Encoder(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, hidden_dim: int, bottleneck_dim: int,
                 activation: nn.Module = nn.ReLU, n_layers: int = 3):
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
        x = self.flatten(x)
        latent = self.linear(x)
        latent = torch.reshape(latent, (-1, 2, self.bottleneck_dim))
        mu, logvar = latent[:,0], latent[:,1]
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, hidden_dim: int, bottleneck_dim: int,
                 activation: nn.Module = nn.ReLU, n_layers: int = 3, params_to_decoder = False, 
                 param_dim: int = None):
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
        if self.params_to_decoder:
            param = self.param_normalization(param)
            x = self.linear(torch.cat((z_latent, param), dim=1)) if not self.feed_forward_nn else self.linear(param)
        else:
            x = self.linear(z_latent)
        x = torch.reshape(x,(-1, self.channels, self.seq_len))
        return x
    
class Regressor(nn.Module):
    def __init__(self, parameter_dim: int, hidden_dim: int, 
                 bottleneck_dim: int, activation: nn.Module = nn.ReLU, 
                 n_layers: int = 3):
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
        param = self.normalization(param)
        latent = self.linear(param)
        latent = torch.reshape(latent,(-1, 2, self.bottleneck_dim))
        mu, logvar = latent[:,0], latent[:,1]
        return mu, logvar
    
class VAE(nn.Module):
    def __init__(self, n_states: int, n_outputs: int, seq_len: int, parameter_dim: int, 
                 hidden_dim: int, bottleneck_dim: int, activation: nn.Module = nn.ReLU, 
                 n_layers: int = 3, params_to_decoder=False, feed_forward_nn = False):
        # TODO: describe in documnetation the different networks this can approximate (VAE, PELS_VAE, FFNN)
        # Describe VAE mode for PELS-VAE: reconstruction from latent parameters or regressor
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
        # if device.type == 'cuda':
        #     eps = torch.autograd.Variable(torch.cuda.FloatTensor(mu.shape).normal_())
        # else: 
        #     eps = torch.autograd.Variable(torch.FloatTensor(mu.shape).normal_())
        eps = torch.randn_like(mu, device=mu.device)
        std = logvar.mul(0.5).exp()
        z_latent = eps.mul(std).add_(mu)
        return z_latent

    def forward(self, states, outputs, params, train=True, predict = False, n_passes: int = 1, test_with_zero_eps: bool = False, device = None):
        '''
        performs one forward pass through network.
        x is the stakc of states and outputs
        if train is True, x goes through encoder, decoder, params go through regressor
        if train is False, x goes through encoder to get mu_encoder, logvar_encoder, and params go through regressor
        if predict is True, x is ignored and mu_encoder, logvar_encoder are set to inf, and x is 
            reconstructed from mu_regressor, logvar_regressor (from params)
        '''
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
        return self.forward(states=None, outputs=None, params=param, train=False, predict=True)
    
    def save(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.state_dict(), path)
        logging.info('\t \t \tSaved model to {}'.format(path))
    
    def load(self, path: Path, device = None):
        self.load_state_dict(torch.load(path, map_location=device))
        logging.info('\tLoaded model from {}'.format(path))

def loss_function(x: torch.tensor, x_hat:torch.tensor, 
                  mu: torch.tensor, mu_hat: torch.tensor, 
                  logvar: torch.tensor, logvar_hat: torch.tensor,
                  beta: float = 1.0, gamma: float = 1000.0, 
                  capacity: float = None,
                  reduce: bool = True,
                  device = None):
    '''
    Implements loss function for VAE.
    if capactiy is None, loss = mse_loss + beta * kl_loss + regressor_loss
    if capacity is not None, loss = mse_loss + gamma * (kl_loss - capacity).abs() + regressor_loss,
    beta ist then ignored
    if reduce is True, loss is a scalar, otherwise it is an inf and the other losses are returned
    '''
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

