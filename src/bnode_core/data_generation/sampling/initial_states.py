"""Uniform random sampling of initial state values."""

import numpy as np
from bnode_core.config import data_gen_config


def random_sampling_initial_states(cfg: data_gen_config) -> np.ndarray:
    """Sample initial state values uniformly within configured bounds.
    
    Generates a 2D array of initial state values by sampling uniformly from the bounds 
    specified in cfg.pModel.RawData.states for each state variable.
    
    Args:
        cfg: Data generation configuration containing state bounds and n_samples.
            cfg.pModel.RawData.states is a dict where each key maps to [lower_bound, upper_bound].
            cfg.pModel.RawData.n_samples specifies the number of initial state sets to generate.
    
    Returns:
        np.ndarray: Initial state values with shape (n_samples, n_states). Each row is one 
            sampled initial state vector.
    """
    bounds = [[cfg.pModel.RawData.states[key][0], cfg.pModel.RawData.states[key][1]] for key in cfg.pModel.RawData.states.keys()]
    initial_state_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states.keys())))
    for i in range(len(cfg.pModel.RawData.states.keys())):
        initial_state_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return initial_state_values
