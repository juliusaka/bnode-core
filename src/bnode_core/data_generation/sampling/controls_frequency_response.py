"""Chirp / frequency-sweep control sampling for system identification (strategy ``RF``)."""

import numpy as np
from bnode_core.config import data_gen_config


def random_frequency_response_sampling_controls(cfg: data_gen_config) -> np.ndarray:
    """Sample frequency-sweep control trajectories for system identification.
    
    Generates control trajectories with a chirp (frequency sweep) starting at the midpoint. 
    The first half is constant, and the second half contains a sine wave with linearly 
    increasing frequency from min to max. Useful for system identification and frequency 
    response analysis.
    
    The frequency sweep goes from _min_frequency (low) to _max_frequency (high), calculated 
    based on the configured control frequency bounds (multiplied by 4 since these represent 
    half-periods).
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls: dict of control bounds [lower, upper].
            cfg.pModel.RawData.controls_frequency_min_in_timesteps: base for max sweep frequency.
            cfg.pModel.RawData.controls_frequency_max_in_timesteps: base for min sweep frequency.
            cfg.pModel.RawData.n_samples: number of trajectories to generate.
            cfg.pModel.RawData.Solver.sequence_length: total trajectory length.
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, sequence_length).
            First half constant, second half contains frequency sweep.
    """
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    
    _max_frequency = cfg.pModel.RawData.controls_frequency_min_in_timesteps * 4 # because this is only half the frequency
    _min_frequency = cfg.pModel.RawData.controls_frequency_max_in_timesteps * 4 

    i_step = cfg.pModel.RawData.Solver.sequence_length // 2
    len_frequency = cfg.pModel.RawData.Solver.sequence_length - i_step

    freq_fun = lambda x: _min_frequency + (_max_frequency - _min_frequency) * (x/len_frequency)
    turns = np.zeros(len_frequency)
    for i in range(1,len_frequency):
        turns[i] = turns[i-1] + (1/freq_fun(i))
    phi = turns * (2 * np.pi) 
    sine = np.sin(phi)
    for i in range(cfg.pModel.RawData.n_samples):
        for j in range(len(cfg.pModel.RawData.controls.keys())):
            _signal_value_start = np.random.uniform(bounds[j][0], bounds[j][1],1)
            ctrl_values[i, j, :i_step] = _signal_value_start[:, None]
            _amplitude = np.random.uniform(0, bounds[j][1] - bounds[j][0])
            if bounds[j][1]  < _signal_value_start + _amplitude:
                _amplitude = bounds[j][1] - _signal_value_start
            if bounds[j][0] > _signal_value_start - _amplitude:
                _amplitude = _signal_value_start - bounds[j][0]
            assert bounds[j][0] + _amplitude <= _signal_value_start <= bounds[j][1] - _amplitude
            _signal_value_end = _signal_value_start + sine * _amplitude
            ctrl_values[i, j, i_step:] = _signal_value_end[:]
    return ctrl_values
