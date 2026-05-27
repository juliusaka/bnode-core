"""Uniform random sampling of model parameters."""

import numpy as np
from bnode_core.config import data_gen_config


def random_sampling_parameters(cfg: data_gen_config) -> np.ndarray:
    """Sample parameter values uniformly within configured bounds.
    
    Generates a 2D array of parameter values by sampling uniformly from the bounds 
    specified in cfg.pModel.RawData.parameters for each parameter.
    
    Args:
        cfg: Data generation configuration containing parameter bounds and n_samples.
            cfg.pModel.RawData.parameters is a dict where each key maps to [lower_bound, upper_bound].
            cfg.pModel.RawData.n_samples specifies the number of parameter sets to generate.
    
    Returns:
        np.ndarray: Parameter values with shape (n_samples, n_parameters). Each row is one 
            sampled parameter set.
    """
    bounds = [[cfg.pModel.RawData.parameters[key][0], cfg.pModel.RawData.parameters[key][1]] for key in cfg.pModel.RawData.parameters.keys()]
    param_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.parameters.keys())))
    for i in range(len(cfg.pModel.RawData.parameters.keys())):
        param_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return param_values
