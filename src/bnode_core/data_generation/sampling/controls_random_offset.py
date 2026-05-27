"""Random control sampling with bounded offset (strategy ``RO``)."""

import numpy as np
from typing import Optional
from bnode_core.config import data_gen_config


def random_sampling_controls_w_offset(cfg: data_gen_config, seq_len: Optional[int] = None, n_samples: Optional[int] = None) -> np.ndarray:
    """Sample control trajectories with random offset and bounded amplitude.
    
    For each control trajectory, first samples a random offset within the control bounds, 
    then samples an amplitude that ensures the trajectory stays within bounds. Each timestep 
    is sampled uniformly within [offset - amplitude_lower, offset + amplitude_upper].
    
    This produces control trajectories that vary around a central offset value rather than 
    exploring the full control space independently at each timestep.
    
    Args:
        cfg: Data generation configuration containing control bounds.
            cfg.pModel.RawData.controls is a dict where each key maps to [lower_bound, upper_bound].
            cfg.pModel.RawData.n_samples and cfg.pModel.RawData.Solver.sequence_length are used 
            as defaults if n_samples or seq_len are not provided.
        seq_len: Optional sequence length override. If None, uses cfg.pModel.RawData.Solver.sequence_length.
        n_samples: Optional sample count override. If None, uses cfg.pModel.RawData.n_samples.
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, seq_len). Each trajectory 
            varies around a sampled offset with bounded amplitude.
    """
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples if n_samples is None else n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length if seq_len is None else seq_len))
    for j in range(ctrl_values.shape[0]):
        for i in range(len(cfg.pModel.RawData.controls.keys())):
            # get offset
            offset = np.random.uniform(bounds[i][0], bounds[i][1])
            # get amplitude
            amplitude = np.random.uniform(0, bounds[i][1] - bounds[i][0])
            # reduce amplitude if offset is close to bounds
            amplitude_upper = amplitude if bounds[i][1] - amplitude > offset else bounds[i][1] - offset
            amplitude_lower = amplitude if bounds[i][0] + amplitude < offset else offset - bounds[i][0]
            ctrl_values[j, i, :] = np.random.uniform(offset - amplitude_lower, offset + amplitude_upper, ctrl_values.shape[2])
    # last control input is not used.
    return ctrl_values
