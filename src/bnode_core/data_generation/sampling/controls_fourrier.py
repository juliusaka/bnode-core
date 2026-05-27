"""Fourier-based control sampling for system identification (strategy ``RFS``)."""

import numpy as np
from bnode_core.config import data_gen_config


def random_fourrier_sampling_controls(cfg: data_gen_config) -> np.ndarray:
    """Sample frequency-sweep control trajectories for system identification.
    
    Generates control trajectories based on a Fourier series with random frequencies and amplitudes.
    
    We sample a random number of Fourier components (between 1 and 100).
    Then we generate a sine wave with an amplitude that is randomly sampled within 
    the halth of the control range (to ensure we stay within bounds) and a frequency 
    that is randomly sampled between the min and max frequency. The sine wave is then shifted to 
    the middle of the control range.
    
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
    
    # the understanding in the config is reverse to the actual.
    _max_frequency = cfg.pModel.RawData.controls_frequency_min_in_timesteps / 4
    _min_frequency = cfg.pModel.RawData.controls_frequency_max_in_timesteps / 4


    for i in range(cfg.pModel.RawData.n_samples):
        for j in range(len(cfg.pModel.RawData.controls.keys())):
            _max_range = (bounds[j][1] - bounds[j][0])
            n_fourrier_components = np.random.randint(1, 50)
            fourier_signal = np.zeros(cfg.pModel.RawData.Solver.sequence_length)
            _offset = np.random.uniform(0, 2 * np.pi)
            for k in range(n_fourrier_components-1):
                _amplitude = np.random.uniform(0, 1)
                _frequency = np.random.uniform(_min_frequency, _max_frequency)
                _frequency = np.random.choice(np.array([-1, 1])) * _frequency # also allow negative frequencies for more variety
                fourier_signal += _amplitude * np.sin(2 * np.pi * _frequency * np.arange(cfg.pModel.RawData.Solver.sequence_length) * cfg.pModel.RawData.Solver.timestep + _offset)
            _range_sampled = np.random.uniform(0, _max_range)
            _range_fourrier = np.max(fourier_signal) - np.min(fourier_signal)
            fourier_signal = fourier_signal / _range_fourrier * _range_sampled
            # previously, we did not use the zeroth fourrier component, so we add now a midpoint
            # determine space for shifting the signal up and down without going out of bounds
            if n_fourrier_components > 1:
                fourier_signal = fourier_signal - np.min(fourier_signal) # shift to zero
                assert np.all(fourier_signal >= 0), f"Fourrier signal has negative values after shifting, which should not happen. Min value: {np.min(fourier_signal)}"
            _midpoint_min = bounds[j][0]
            _midpoint_max = bounds[j][1] - np.max(fourier_signal)
            _midpoint = np.random.uniform(_midpoint_min, _midpoint_max)
            fourier_signal += _midpoint
            ctrl_values[i, j, :] = fourier_signal

            assert np.all(ctrl_values[i, j, :] >= bounds[j][0]) and np.all(ctrl_values[i, j, :] <= bounds[j][1]), f"Control values out of bounds for control {j} in sample {i}"
    return ctrl_values
