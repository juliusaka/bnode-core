import torch
import torch.nn as nn
import numpy as np
import logging

class NormalizationLayerTimeSeries(nn.Module):
    '''
    normalize batches of inputs to zero mean and variance 1
    expects data of dimensions (batch_size, number_of_channels, sequence_length)
    # TODO: add dimension description
    '''
    def __init__(self, n_channels):
        '''
        normalize batches of inputs to zero mean and variance 1
        expects data of dimensions (batch_size, number_of_channels, sequence_length)

        Args:
            n_channels (int): number of channels
        '''
        super().__init__()
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer('std', torch.zeros(n_channels))
        self.register_buffer('mu', torch.zeros(n_channels))
    
    def initialize_normalization(self,x):
        '''
        can call this to initialize the normalization layer with the mean and standard deviation of the input data
        Args:
            x (torch.tensor): input data of dimensions (batch_size, number_of_channels, sequence_length)
        '''
        if not self._initialized:
            variance = torch.var(x, dim=(0,2)).detach()
            self.std.set_(torch.sqrt(variance + torch.ones(variance.size()).to(variance.device) * 1e-3))
            self.mu.set_(torch.mean(x, dim=(0,2)).detach())
            self._initialized = torch.tensor(True)
            assert self.std.requires_grad == False
            assert self.mu.requires_grad == False

    def forward(self,x, denormalize = False):
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
    '''
    normalize batches of inputs to zero mean and variance 1
    expects data of dimensions (batch_size, number_of_channels)
    '''
    def __init__(self, num_features):
        '''
        normalize batches of inputs to zero mean and variance 1
        expects data of dimensions (batch_size, number_of_channels)

        Args:
            num_features (int): number of features / channels
        '''
        super().__init__()
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer('std', torch.zeros((num_features)))
        self.register_buffer('mu', torch.zeros(num_features))
    
    def initialize_normalization(self, x, eps = 1e-5, verbose = False, name = None):
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

    def forward(self,x, denormalize = False):
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
    
    # add print method
    def __repr__(self):
        return 'NormalizationLayer1D(num_features={})'.format(self.std.shape[0])