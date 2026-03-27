"""Uniform random sampling of control inputs (strategy ``R``)."""

import numpy as np
from bnode_core.config import data_gen_config


def random_sampling_controls(cfg: data_gen_config) -> np.ndarray:
    """Sample control input values uniformly within configured bounds.
    
    Generates a 3D array of control trajectories by sampling uniformly from the bounds 
    specified in cfg.pModel.RawData.controls for each control variable at each timestep.
    Each control trajectory is independently sampled (no temporal correlation).
    
    Args:
        cfg: Data generation configuration containing control bounds, n_samples, and sequence_length.
            cfg.pModel.RawData.controls is a dict where each key maps to [lower_bound, upper_bound].
            cfg.pModel.RawData.n_samples specifies the number of control trajectories to generate.
            cfg.pModel.RawData.Solver.sequence_length specifies the number of timesteps.
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, sequence_length). 
            Each element is independently sampled from uniform distributions.
    """
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    for i in range(len(cfg.pModel.RawData.controls.keys())):
        ctrl_values[:, i, :] = np.random.uniform(bounds[i][0], bounds[i][1], (cfg.pModel.RawData.n_samples, cfg.pModel.RawData.Solver.sequence_length))
    # last control input is not used.
    return ctrl_values
