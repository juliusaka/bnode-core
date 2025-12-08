"""Utility for counting trainable parameters in PyTorch models."""

import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model (nn.Module).
    
    Returns:
        Total number of trainable parameters (int).
    
    Example:
        >>> model = VAE(...)
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)