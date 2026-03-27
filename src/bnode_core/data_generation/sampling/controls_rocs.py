"""Cubic-spline control sampling with manual clipping (strategy ``ROCS``)."""

import numpy as np
from scipy.interpolate import CubicSpline
from bnode_core.config import data_gen_config
from bnode_core.data_generation.sampling.controls_random_offset import random_sampling_controls_w_offset


def random_sampling_controls_w_offset_cubic_splines_old_clip_manual(cfg: data_gen_config) -> np.ndarray:
    """Sample control trajectories using cubic spline interpolation with manual clipping (ROCS).
    
    Also known as ROCS (Random Offset Cubic Splines). Generates smooth control trajectories by:

    1. Sampling control values at random intervals
    2. Interpolating with cubic splines
    3. Normalizing to fit within bounds via manual clipping
    
    ROCS fills the control space more than RROCS because values exceeding bounds are clipped 
    to the bounds rather than rescaled.
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls_frequency_min_in_timesteps: minimum interval between samples.
            cfg.pModel.RawData.controls_frequency_max_in_timesteps: maximum interval between samples.
            cfg.pModel.RawData.controls: dict of control bounds [lower, upper].
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, sequence_length).
            Smooth trajectories that fill the control space via clipping.
    """
    freq_sequence = np.random.choice(np.arange(cfg.pModel.RawData.controls_frequency_min_in_timesteps, cfg.pModel.RawData.controls_frequency_max_in_timesteps + 1), cfg.pModel.RawData.n_samples)
    # find out at which entry we reached the sequence length
    seq_len_sampling = np.where(np.cumsum(freq_sequence) > cfg.pModel.RawData.Solver.sequence_length)[0][0] + 1
    # sample data
    ctrl_values_sampled = random_sampling_controls_w_offset(cfg, seq_len_sampling+1) 
    # create cubic splines
    x = np.concatenate((np.array([0]),
                       np.cumsum(freq_sequence[:seq_len_sampling]))
                       )
    xnew = np.arange(cfg.pModel.RawData.Solver.sequence_length)
    ctrl_values = CubicSpline(x, ctrl_values_sampled, axis=2)(xnew)
    # normalize values to bounds
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    for i in range(ctrl_values.shape[0]):
        for j in range(ctrl_values.shape[1]):
            min_val = np.min(ctrl_values[i, j, :])
            max_val = np.max(ctrl_values[i, j, :])

            exceeds_bounds = max_val - min_val > bounds[j][1] - bounds[j][0]
            delta = max_val - min_val if  exceeds_bounds else bounds[j][1] - bounds[j][0]

            # calculate base:
            if exceeds_bounds:
                base = bounds[j][0]
            elif min_val < bounds[j][0]:
                base = bounds[j][0]
            elif max_val > bounds[j][1]:
                base = bounds[j][1] - delta
            else:
                base = min_val
            ctrl_values[i, j, :] = (ctrl_values[i, j, :] - min_val) / delta * (bounds[j][1] - bounds[j][0]) + base
            if ctrl_values[i, j, :].min() < bounds[j][0] or ctrl_values[i, j, :].max() > bounds[j][1]:
                print('error in random_sampling_controls_w_offset_cubic_splines')
    return ctrl_values
