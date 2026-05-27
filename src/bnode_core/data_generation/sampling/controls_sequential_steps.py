"""Step-change control sampling for step-response tests (strategy ``RSS``)."""

import numpy as np
from bnode_core.config import data_gen_config


def random_sequential_steps_sampling_controls(cfg: data_gen_config) -> np.ndarray:
    """Sample step-change control trajectories for system response testing.
    
    Generates control trajectories with one step at a time for all control signals.
    The sequence of steps on which control input is randomly sampled. The distribution over
    time is evenly distributed. 
    Each control starts at a randomly sampled value and steps to another randomly sampled value at the 
    the sampled time point.
    Useful for testing system step response characteristics and mabye also training.
    
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
    
    i_between_steps = cfg.pModel.RawData.Solver.sequence_length // (len(cfg.pModel.RawData.controls.keys()) + 1)
    order = np.random.shuffle(np.arange(len(cfg.pModel.RawData.controls.keys())))
    
    for i in range(len(cfg.pModel.RawData.controls.keys())):
        i_step = (i + 1) * i_between_steps
        _signal_value_before_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        _signal_value_after_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        ctrl_values[:, i, :i_step] = _signal_value_before_step[:, None]
        ctrl_values[:, i, i_step:] = _signal_value_after_step[:, None]
    
    # last control input is not used.
    return ctrl_values