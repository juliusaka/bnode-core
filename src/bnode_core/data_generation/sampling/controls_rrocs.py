"""Cubic-spline control sampling with random rescaling (strategy ``RROCS``)."""

import logging
import numpy as np
from scipy.interpolate import CubicSpline
from bnode_core.config import data_gen_config
from bnode_core.data_generation.sampling.controls_random_offset import random_sampling_controls_w_offset


def random_sampling_controls_w_offset_cubic_splines_clip_random(cfg: data_gen_config) -> np.ndarray:
    """Sample control trajectories using cubic spline interpolation with random rescaling (RROCS).
    
    Also known as RROCS (Randomly Rescaled Offset Cubic Splines). Generates smooth control 
    trajectories by:

    1. For each control and sample, sampling values at random intervals (e.g. different frequencies), 
    with sampled amplitudes and offsets
    2. Interpolating with cubic splines
    3. Normalizing to [0, 1] and rescaling with randomly sampled base and delta
    4. Optionally clipping to tighter bounds if specified
    
    RROCS fills the control space less uniformly than ROCS because values are rescaled to fit
    within bounds rather than clipped. This means, that typically at the sampling bounds, less
    samples are present.
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls_frequency_min_in_timesteps: minimum interval between samples.
            cfg.pModel.RawData.controls_frequency_max_in_timesteps: maximum interval between samples.
            cfg.pModel.RawData.controls: dict where each key maps to [lower, upper] or 
                [lower, upper, clip_lower, clip_upper] for optional tighter clipping bounds.
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, sequence_length).
            Smooth trajectories with diverse amplitude and offset characteristics.
    """
    # normalize values to bounds
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    # get clip values, if available
    clip_bounds = [cfg.pModel.RawData.controls[key][2:] if len(cfg.pModel.RawData.controls[key]) == 4 else None for key in cfg.pModel.RawData.controls.keys()]
    for j, clip in enumerate(clip_bounds):
        if clip is not None:
            logging.info('control {}: clip values provided: {}'.format(list(cfg.pModel.RawData.controls.keys())[j], clip))
    
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    # loop over samples
    for i in range(ctrl_values.shape[0]):
        # loop over controls
        for j in range(ctrl_values.shape[1]):
            freq_sequence = np.random.choice(np.arange(cfg.pModel.RawData.controls_frequency_min_in_timesteps, cfg.pModel.RawData.controls_frequency_max_in_timesteps + 1), cfg.pModel.RawData.Solver.sequence_length)
            # find out at which entry we reached the sequence length
            seq_len_sampling = np.where(np.cumsum(freq_sequence) > cfg.pModel.RawData.Solver.sequence_length)[0][0] + 1
            # sample data
            ctrl_values_sampled = random_sampling_controls_w_offset(cfg, seq_len_sampling+1, n_samples=1)
            # create cubic splines
            x = np.concatenate((np.array([0]),
                            np.cumsum(freq_sequence[:seq_len_sampling]))
                            )
            xnew = np.arange(cfg.pModel.RawData.Solver.sequence_length)
            ctrl_values[i, j, :] = CubicSpline(x, ctrl_values_sampled[0, j])(xnew)

            # normalize values to bounds
            min_val = np.min(ctrl_values[i, j, :])
            max_val = np.max(ctrl_values[i, j, :])
            # normalize data to min 0 and max 1
            _values = (ctrl_values[i, j, :] - min_val) / (max_val - min_val)
            # randomly samply base and delta
            base = np.random.uniform(bounds[j][0], bounds[j][1])
            delta = np.random.uniform(0, bounds[j][1]-bounds[j][0])
            # calculate new base if delta is too large
            if base + delta > bounds[j][1]:
                base = bounds[j][1] - delta
            elif base - delta < bounds[j][0]:
                base = bounds[j][0]
            # calculate new values
            ctrl_values[i, j, :] = _values * delta + base
            # clip to clip bounds if available
            if clip_bounds[j] is not None:
                ctrl_values[i, j, :] = np.clip(ctrl_values[i, j, :], clip_bounds[j][0], clip_bounds[j][1])
            # if ctrl_values[i, j, :].min() < bounds[j][0] or ctrl_values[i, j, :].max() > bounds[j][1]:
            #     print('error in random_sampling_controls_w_offset_cubic_splines')
    return ctrl_values
