"""Step-change control sampling for step-response tests (strategy ``RS``)."""

import numpy as np
from bnode_core.config import data_gen_config


def random_steps_sampling_controls(cfg: data_gen_config) -> np.ndarray:
    """Sample step-change control trajectories for system response testing.
    
    Generates control trajectories with a single step change at the midpoint. Each control 
    starts at a randomly sampled value and steps to another randomly sampled value halfway 
    through the sequence. Useful for testing system step response characteristics.
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls: dict of control bounds [lower, upper].
            cfg.pModel.RawData.n_samples: number of step trajectories to generate.
            cfg.pModel.RawData.Solver.sequence_length: total trajectory length.
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, sequence_length).
            Each trajectory has a step change at sequence_length // 2.
    """
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    
    i_step = cfg.pModel.RawData.Solver.sequence_length // 2
    for i in range(len(cfg.pModel.RawData.controls.keys())):
        _signal_value_before_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        _signal_value_after_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        ctrl_values[:, i, :i_step] = _signal_value_before_step[:, None]
        ctrl_values[:, i, i_step:] = _signal_value_after_step[:, None]
    
    # last control input is not used.
    return ctrl_values
