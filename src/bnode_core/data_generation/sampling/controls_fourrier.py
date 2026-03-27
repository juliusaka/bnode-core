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
    _max_frequency = cfg.pModel.RawData.controls_frequency_min_in_timesteps * 4
    _min_frequency = cfg.pModel.RawData.controls_frequency_max_in_timesteps 

    for i in range(cfg.pModel.RawData.n_samples):
        for j in range(len(cfg.pModel.RawData.controls.keys())):
            n_fourrier_components = np.random.randint(1, 50)
            fourier_signal = np.zeros(cfg.pModel.RawData.Solver.sequence_length)
            for k in range(n_fourrier_components):
                _amplitude = np.random.uniform(0, (bounds[j][1] - bounds[j][0]) / 2)
                _frequency = np.random.uniform(_min_frequency, _max_frequency)
                fourier_signal += _amplitude * np.sin(2 * np.pi * _frequency * np.arange(cfg.pModel.RawData.Solver.sequence_length) * cfg.pModel.RawData.Solver.timestep)
            ctrl_values[i, j, :] = (bounds[j][0] + bounds[j][1]) / 2 + fourier_signal

    return ctrl_values
