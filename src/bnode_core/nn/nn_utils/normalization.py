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
            self.std.set_(x.var(dim=(0, 2)).add(1e-3).sqrt().detach())
            self.mu.set_(x.mean(dim=(0, 2)).detach())
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
        # (n_channels,) -> (1, n_channels, 1) for broadcasting over batch and time
        mu = self.mu.unsqueeze(0).unsqueeze(-1)
        std = self.std.unsqueeze(0).unsqueeze(-1)
        if denormalize is False:
            return (x - mu) / std
        else:
            return x * std + mu
    
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
    
    def initialize_normalization(self, x, eps = 1e-7, verbose = False, name = None):
        """Compute and store mean and std from input data.
        
        Calculates per-feature statistics across batch dimension. Adds epsilon to variance
        for numerical stability. Supports both torch.Tensor and numpy.ndarray inputs.
        
        Args:
            x (torch.Tensor or np.ndarray): Input data with shape (batch_size, num_features).
            eps (float, optional): Small constant added to variance for stability. Defaults to 1e-7.
            verbose (bool, optional): If True, logs initialization info. Defaults to False.
            name (str, optional): Name for logging output. Defaults to None.
        
        Raises:
            ValueError: If x is neither torch.Tensor nor np.ndarray.
            RuntimeError: If normalization layer has already been initialized.
        
        Side Effects:
            Sets self.mu and self.std buffers, logs initialization if verbose=True.
        """
        if not self._initialized:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            elif not isinstance(x, torch.Tensor):
                raise ValueError('Unknown type of input: {}'.format(type(x)))
            self.std.set_(x.var(dim=0).add(eps).sqrt().detach())
            self.mu.set_(x.mean(dim=0).detach())
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
        mu = self.mu
        std = self.std
        if len(x.shape) == 3:
            # (num_features,) -> (num_features, 1) so broadcasting aligns with dim 1
            mu = mu.unsqueeze(-1)
            std = std.unsqueeze(-1)
        if not denormalize:
            return (x - mu) / std
        else:
            return x * std + mu
    
    def __repr__(self) -> str:
        """Return string representation of the layer.
        
        Returns:
            str: String showing layer type and number of features.
        """
        return 'NormalizationLayer1D(num_features={})'.format(self.std.shape[0])