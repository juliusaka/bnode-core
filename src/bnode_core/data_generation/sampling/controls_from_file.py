"""Load control trajectories from a CSV file (strategy ``file``)."""

import numpy as np
import pandas as pd
from bnode_core.config import data_gen_config


def load_controls_from_file(cfg: data_gen_config) -> np.ndarray:
    """Load control trajectories from a CSV file and resample to simulation time vector.
    
    Reads control values from a CSV file where columns match control variable names from the 
    config. The CSV must include a 'time' column. Control values are resampled via linear 
    interpolation to match the simulation timestep, then replicated for all samples.

    TODO: could be extended to load multiple trajectories for different samples.
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls_file_path: path to CSV file with time and control columns.
            cfg.pModel.RawData.controls: dict of control names (used as column names).
            cfg.pModel.RawData.Solver: simulation time parameters (start, end, timestep).
            cfg.pModel.RawData.n_samples: number of times to replicate the loaded trajectory.
    
    Returns:
        np.ndarray: Control values with shape (n_samples, n_controls, sequence_length).
            Same trajectory replicated across all samples.
    """
    # load controls from file by control variable name
    _df = pd.read_csv(cfg.pModel.RawData.controls_file_path)
    _list = []
    for key in cfg.pModel.RawData.controls.keys():
        # append to list column that matches the key
        _list.append(_df[key].values)
    time_ctrls = _df['time'].values
    # resample to time vector TODO: better make time vector only once
    time = np.arange(cfg.pModel.RawData.Solver.simulationStartTime, cfg.pModel.RawData.Solver.simulationEndTime + cfg.pModel.RawData.Solver.timestep, cfg.pModel.RawData.Solver.timestep)
    ctrl_values = [np.interp(time, time_ctrls, ctrl) for ctrl in _list]
    ctrl_values = np.array(ctrl_values)
    ctrl_values = np.expand_dims(ctrl_values, axis=0)
    ctrl_values = np.repeat(ctrl_values, cfg.pModel.RawData.n_samples, axis=0)
    return ctrl_values
