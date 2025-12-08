"""Normalization layers for neural network inputs with time series and 1D data support.

This module provides PyTorch normalization layers that compute and store mean/std statistics
from data, then normalize (or denormalize) inputs during forward passes. Supports both time
series data (batch, channels, time) and 1D feature vectors (batch, features).
"""

import torch
import torch.nn as nn
import numpy as np
import logging

class NormalizationLayerTimeSeries(nn.Module):
    """Normalization layer for time series data with shape (batch, channels, time).
    
    Computes and stores per-channel mean and standard deviation from input data, then
    normalizes future inputs to zero mean and unit variance. Can also denormalize outputs
    back to original scale. Statistics are computed once during first forward pass or via
    explicit initialization.
    
    Expected input shape: (batch_size, n_channels, sequence_length)
    
    Attributes:
        _initialized (bool): Whether mean/std have been computed from data.
        std (torch.Tensor): Per-channel standard deviations, shape (n_channels,).
        mu (torch.Tensor): Per-channel means, shape (n_channels,).
    """
    def __init__(self, n_channels):
        """Initialize normalization layer buffers.
        
        Args:
            n_channels (int): Number of channels in time series data.
        """
        super().__init__()
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer('std', torch.zeros(n_channels))
        self.register_buffer('mu', torch.zeros(n_channels))
    
    def initialize_normalization(self,x):
        """Compute and store mean and std from input data.
        
        Calculates per-channel statistics across batch and time dimensions. Adds small
        epsilon (1e-3) to variance for numerical stability. Only runs if not already
        initialized.
        
        Args:
            x (torch.Tensor): Input data with shape (batch_size, n_channels, sequence_length).
        
        Side Effects:
            Sets self.mu and self.std buffers if not already initialized.
        """
        if not self._initialized:
            variance = torch.var(x, dim=(0,2)).detach()
            self.std.set_(torch.sqrt(variance + torch.ones(variance.size()).to(variance.device) * 1e-3))
            self.mu.set_(torch.mean(x, dim=(0,2)).detach())
            self._initialized = torch.tensor(True)
            assert self.std.requires_grad == False
            assert self.mu.requires_grad == False

    def forward(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        """Normalize or denormalize input time series.
        
        If not initialized and normalizing, automatically initializes from input data.
        Normalizes via (x - mu) / std or denormalizes via x * std + mu.
        
        Args:
            x (torch.Tensor): Input with shape (batch_size, n_channels, sequence_length).
            denormalize (bool, optional): If False, normalize input. If True, denormalize
                (reverse transformation). Defaults to False.
        
        Returns:
            torch.Tensor: Normalized or denormalized data with same shape as input.
        """
        if denormalize is False:
            if not self._initialized:
                self.initialize_normalization(x)
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        # add dimensions at position 0 (for number of batches) and at position 2 (for sequence length)
        # expand these dimensions
        std = self.std.unsqueeze(0).unsqueeze(2).expand(batch_size,-1,seq_len)
        mu = self.mu.unsqueeze(0).unsqueeze(2).expand(batch_size,-1,seq_len)
        if denormalize is False:
            x = torch.subtract(x,mu)
            x = torch.divide(x, std)
        else:
            x = torch.multiply(x, std)
            x = torch.add(x, mu)
        return x
    
class NormalizationLayer1D(nn.Module):
    """Normalization layer for 1D feature vectors with shape (batch, features).
    
    Computes and stores per-feature mean and standard deviation, then normalizes inputs
    to zero mean and unit variance. Can also denormalize outputs. Supports both 2D
    (batch, features) and 3D (batch, features, time) inputs. Accepts both torch.Tensor
    and numpy.ndarray for initialization.
    
    Expected input shape: (batch_size, num_features) or (batch_size, num_features, sequence_length)
    
    Attributes:
        _initialized (bool): Whether mean/std have been computed.
        std (torch.Tensor): Per-feature standard deviations, shape (num_features,).
        mu (torch.Tensor): Per-feature means, shape (num_features,).
    """
    def __init__(self, num_features):
        """Initialize normalization layer buffers.
        
        Args:
            num_features (int): Number of features/channels to normalize.
        """
        super().__init__()
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer('std', torch.zeros((num_features)))
        self.register_buffer('mu', torch.zeros(num_features))
    
    def initialize_normalization(self, x, eps = 1e-5, verbose = False, name = None):
        """Compute and store mean and std from input data.
        
        Calculates per-feature statistics across batch dimension. Adds epsilon to variance
        for numerical stability. Supports both torch.Tensor and numpy.ndarray inputs.
        
        Args:
            x (torch.Tensor or np.ndarray): Input data with shape (batch_size, num_features).
            eps (float, optional): Small constant added to variance for stability. Defaults to 1e-5.
            verbose (bool, optional): If True, logs initialization info. Defaults to False.
            name (str, optional): Name for logging output. Defaults to None.
        
        Raises:
            ValueError: If x is neither torch.Tensor nor np.ndarray.
            RuntimeError: If normalization layer has already been initialized.
        
        Side Effects:
            Sets self.mu and self.std buffers, logs initialization if verbose=True.
        """
        if not self._initialized:
            if isinstance(x, torch.Tensor):
                variance = torch.var(x, dim=(0)).detach()
                self.std.set_(torch.sqrt(variance + torch.ones(variance.size()).to(variance.device) * eps))
                self.mu.set_(torch.mean(x, dim=(0)).detach())
            elif isinstance(x, np.ndarray):
                variance = np.var(x, axis=0)
                self.std.set_(torch.sqrt(torch.tensor(variance + np.ones(variance.shape) * eps, dtype=torch.float32)))
                self.mu.set_(torch.tensor(np.mean(x, axis=0), dtype=torch.float32))
            else:
                raise ValueError('Unknown type of input: {}'.format(type(x)))
            self._initialized = torch.tensor(True)
            assert self.std.requires_grad == False
            assert self.mu.requires_grad == False

            logging.info("Initialized normalization layer {} with mean {} and std {}".format(name, self.mu, self.std))
        else:
            raise RuntimeError("normalization layer has already been initialized")

    def forward(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        """Normalize or denormalize input features.
        
        If not initialized and normalizing, automatically initializes from input. Handles
        both 2D (batch, features) and 3D (batch, features, time) inputs by broadcasting.
        Normalizes via (x - mu) / std or denormalizes via x * std + mu.
        
        Args:
            x (torch.Tensor): Input with shape (batch_size, num_features) or 
                (batch_size, num_features, sequence_length).
            denormalize (bool, optional): If False, normalize input. If True, denormalize.
                Defaults to False.
        
        Returns:
            torch.Tensor: Normalized or denormalized data with same shape as input.
        """
        if not denormalize:
            if not self._initialized:
                self.initialize_normalization(x)
        batch_size = x.shape[0]
        # add dimension at position 0 and expand to batch_size
        std = self.std.unsqueeze(0).expand(batch_size,-1)
        mu = self.mu.unsqueeze(0).expand(batch_size,-1)
        if len(x.shape) == 3:
            # if x is a 3D tensor, we assume it has shape (batch_size, num_features, sequence_length)
            seq_len = x.shape[2]
            std = std.unsqueeze(2).expand(batch_size,-1,seq_len)
            mu = mu.unsqueeze(2).expand(batch_size,-1,seq_len)
        if not denormalize:
            x = torch.subtract(x, mu)
            x = torch.divide(x, std)
        else:
            x = torch.multiply(x, std)
            x = torch.add(x, mu)
        return x
    
    def __repr__(self) -> str:
        """Return string representation of the layer.
        
        Returns:
            str: String showing layer type and number of features.
        """
        return 'NormalizationLayer1D(num_features={})'.format(self.std.shape[0])