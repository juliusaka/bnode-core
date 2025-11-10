"""Raw data generation module for parallel FMU simulation.

## Module Description

This module generates raw simulation data by running FMU (Functional Mock-up Unit) models 
in parallel with sampled inputs (initial states, parameters, controls). It uses Dask for 
distributed computing and writes results to HDF5 files with comprehensive logging.

### Command-line Usage

    With uv (recommended):
        uv run raw_data_generation [overrides]
    
    In activated virtual environment:
        raw_data_generation [overrides]
    
    Direct Python execution:
        python -m bnode_core.data_generation.raw_data_generation [overrides]

### Example Commands

    # Generate 1000 samples with default config
    uv run raw_data_generation pModel.RawData.n_samples=1000
    
    # Use specific pModel config and allow overwriting
    uv run raw_data_generation pModel=SHF overwrite=true
    
    # Change control sampling strategy to RROCS
    uv run raw_data_generation pModel.RawData.controls_sampling_strategy=RROCS
    
    # Adjust parallel workers and timeout
    uv run raw_data_generation multiprocessing_processes=8 pModel.RawData.Solver.timeout=120

    # Adjust config path and name
    uv run raw_data_generation --config-path=resources/config --config-name=data_generation_custom

### What This Module Does

1. Loads and validates configuration (FMU path, sampling strategies, solver settings)
2. Sets reproducibility seed (np.random.seed(42))
3. Creates HDF5 raw data file with pre-allocated datasets
4. Samples input values (initial states, parameters, controls) using configured strategies
5. Writes sampled inputs and metadata to HDF5 file
6. Sets up Dask distributed cluster for parallel FMU simulation
7. Submits simulation tasks in batches with timeout monitoring
8. Incrementally writes simulation results (states, outputs, derivatives) to HDF5
9. Logs completion status, failures, timeouts, and processing times per sample
10. Saves configuration YAML file alongside raw data

See main() function for entry point and run_data_generation() for the complete pipeline.

### Key Features

- Parallel execution using Dask LocalCluster with configurable workers
- Per-simulation timeout enforcement via ThreadPoolExecutor
- Automatic worker restart on repeated timeouts
- Incremental result writing (partial data available if interrupted)
- Comprehensive logging: completed, failed, timed-out simulations
- Multiple control sampling strategies (R, RO, ROCS, RROCS, RS, RF, file, Excel)
- Reproducible sampling (fixed seed since 2024-11-23)
- Dask dashboard for monitoring: http://localhost:8787

### Sampling Strategies

    Parameters: 'R' (random uniform)
    Initial states: 'R' (random uniform)
    Controls: 'R' (random uniform), 'RO' (random with offset), 'ROCS' (cubic splines with 
              clipping), 'RROCS' (cubic splines with random rescaling), 'RS' (random steps), 
              'RF' (frequency sweep), 'file' (from CSV), 'constantInput' (from Excel)

### Configuration

    Uses Hydra for configuration management. Config loaded from 'data_generation.yaml'.
    Key config sections: pModel.RawData (all generation parameters including FMU path, bounds, 
    solver settings, sampling strategies), multiprocessing_processes (worker count), 
    memory_limit_per_worker (per-worker memory limit).

### Output Files

    - Raw data HDF5 file: Contains time, states, controls, outputs, parameters, logs
    - Config YAML file: Snapshot of pModel.RawData configuration used for generation
    Both file paths determined by bnode_core.filepaths functions.
"""
import dask.config
import dask.config
import dask.distributed
import hydra
import os
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import logging
from omegaconf import OmegaConf
from datetime import datetime
from time import time, sleep
import dask
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from scipy.interpolate import CubicSpline, Akima1DInterpolator

from bnode_core.config import data_gen_config, get_config_store, convert_cfg_to_dataclass
from bnode_core.filepaths import filepath_raw_data, log_overwriting_file, filepath_raw_data_config, config_dir_auto_recognize
from typing import Tuple, Optional, List

def random_sampling_parameters(cfg: data_gen_config) -> np.ndarray:
    """Sample parameter values uniformly within configured bounds.
    
    Generates a 2D array of parameter values by sampling uniformly from the bounds 
    specified in cfg.pModel.RawData.parameters for each parameter.
    
    Args:
        cfg: Data generation configuration containing parameter bounds and n_samples.
            cfg.pModel.RawData.parameters is a dict where each key maps to [lower_bound, upper_bound].
            cfg.pModel.RawData.n_samples specifies the number of parameter sets to generate.
    
    Returns:
        np.ndarray: Parameter values with shape (n_samples, n_parameters). Each row is one 
            sampled parameter set.
    """
    bounds = [[cfg.pModel.RawData.parameters[key][0], cfg.pModel.RawData.parameters[key][1]] for key in cfg.pModel.RawData.parameters.keys()]
    param_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.parameters.keys())))
    for i in range(len(cfg.pModel.RawData.parameters.keys())):
        param_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return param_values

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
    # freq_sequence = np.random.choice(np.arange(cfg.pModel.RawData.controls_frequency_min_in_timesteps, cfg.pModel.RawData.controls_frequency_max_in_timesteps + 1), cfg.pModel.RawData.n_samples)
    # # find out at which entry we reached the sequence length
    # seq_len_sampling = np.where(np.cumsum(freq_sequence) > cfg.pModel.RawData.Solver.sequence_length)[0][0] + 1
    # # sample data
    # ctrl_values_sampled = random_sampling_controls_w_offset(cfg, seq_len_sampling+1)
    # # create cubic splines
    # x = np.concatenate((np.array([0]),
    #                    np.cumsum(freq_sequence[:seq_len_sampling]))
    #                    )
    # xnew = np.arange(cfg.pModel.RawData.Solver.sequence_length)
    # ctrl_values = CubicSpline(x, ctrl_values_sampled, axis=2)(xnew)
    
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
        #ctrl_values[:, i, :] = np.random.uniform(bounds[i][0], bounds[i][1], (cfg.pModel.RawData.n_samples, cfg.pModel.RawData.Solver.sequence_length))
        _signal_value_before_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        _signal_value_after_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        ctrl_values[:, i, :i_step] = _signal_value_before_step[:, None]
        ctrl_values[:, i, i_step:] = _signal_value_after_step[:, None]
    
    # last control input is not used.
    return ctrl_values

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

def constant_input_simulation_from_excel(cfg: data_gen_config) -> np.ndarray:
    """Load constant control values from an Excel file for steady-state simulations.
    
    Reads an Excel file with a sheet named 'Tabelle1' where each row defines one simulation 
    with constant control values. Control columns must be named to match config control names. 
    Each row's values are held constant for the entire sequence length.
    
    Useful for steady-state simulations or parameter sweeps with constant inputs.
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls_file_path: path to Excel file.
            cfg.pModel.RawData.controls: dict of control names (must match column names in Excel).
            cfg.pModel.RawData.Solver.sequence_length: length to replicate constant values.
    
    Returns:
        np.ndarray: Control values with shape (n_rows, n_controls, sequence_length).
            Each row from Excel becomes one sample with constant control values.
    
    Notes:
        Excel file structure:
        - Sheet name: 'Tabelle1'
        - First row: column headers matching control variable names
        - Each subsequent row: one set of constant control values for one simulation
    """
    file = pd.ExcelFile(cfg.pModel.RawData.controls_file_path)
    _df = file.parse(sheet_name='Tabelle1')
    _list = []
    for key in cfg.pModel.RawData.controls.keys():
        _list.append(_df[key].values)
    ctrl_values = np.array(_list).transpose()
    ctrl_values = np.expand_dims(ctrl_values, axis=2)
    ctrl_values = np.repeat(ctrl_values, (cfg.pModel.RawData.Solver.sequence_length), axis=2)
    return ctrl_values


def random_sampling_initial_states(cfg: data_gen_config) -> np.ndarray:
    """Sample initial state values uniformly within configured bounds.
    
    Generates a 2D array of initial state values by sampling uniformly from the bounds 
    specified in cfg.pModel.RawData.states for each state variable.
    
    Args:
        cfg: Data generation configuration containing state bounds and n_samples.
            cfg.pModel.RawData.states is a dict where each key maps to [lower_bound, upper_bound].
            cfg.pModel.RawData.n_samples specifies the number of initial state sets to generate.
    
    Returns:
        np.ndarray: Initial state values with shape (n_samples, n_states). Each row is one 
            sampled initial state vector.
    """
    bounds = [[cfg.pModel.RawData.states[key][0], cfg.pModel.RawData.states[key][1]] for key in cfg.pModel.RawData.states.keys()]
    initial_state_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states.keys())))
    for i in range(len(cfg.pModel.RawData.states.keys())):
        initial_state_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return initial_state_values

def progress_string(progress: float, length: int = 10) -> str:
    """Generate a visual progress bar string for logging.
    
    Returns a visual progress string of the form '|||||.....' for a given progress value in [0, 1].
    
    Args:
        progress: Progress value between 0 and 1.
        length: Total length of the progress string.
        
    Returns:
        Progress bar string with '|' for completed portion and '.' for remaining.
    """
    progress = max(0, min(1, progress))  # Clamp to [0, 1]
    n_complete = int(round(progress * length))
    n_remaining = length - n_complete
    return '|' * n_complete + '.' * n_remaining

def data_generation(cfg: data_gen_config,
                    initial_state_values: np.ndarray = None,
                    param_values: np.ndarray = None,
                    ctrl_values: np.ndarray = None):
    """Execute parallel FMU simulations and write results to raw data HDF5 file.
    
    Core data generation function that:

    1. Sets up a Dask distributed cluster for parallel FMU simulation
    2. Submits simulation tasks for each sample in batches
    3. Monitors task completion and handles timeouts/failures
    4. Incrementally writes results to the raw data HDF5 file
    5. Logs completion status, failures, and timing information
    
    The function uses ThreadPoolExecutor to enforce per-simulation timeouts and Dask's 
    LocalCluster for parallel execution across multiple workers. Results are written 
    incrementally so partial data is available even if generation is interrupted.
    
    Args:
        cfg: Data generation configuration containing:
            - FMU path and simulation parameters
            - Solver settings (timestep, tolerance, timeout)
            - Multiprocessing and memory settings
            - Output file paths
        initial_state_values: Optional array of shape (n_samples, n_states) with initial states.
        param_values: Optional array of shape (n_samples, n_parameters) with parameter values.
        ctrl_values: Optional array of shape (n_samples, n_controls, sequence_length) with controls.
    
    Notes:
        - The raw data HDF5 file must already exist with pre-allocated datasets.
        - Dask worker memory limits and allowed failures are configured from cfg settings.
        - Progress is logged via the Dask diagnostic dashboard at http://localhost:8787.
        - Per-sample logs (completed, sim_failed, timedout, processing_time) are written 
          incrementally to the HDF5 file.
        - If a worker's tasks timeout repeatedly, the worker is restarted automatically.
        - For large numbers of samples, tasks are submitted in "submission rounds" (batches of 10,000 simulations) 
          to avoid overwhelming the scheduler.
    
    Raises:
        BaseException: Any exception during generation is caught to ensure partial results 
            are saved before re-raising.
    """
    from bnode_core.data_generation.utils.fmu_simulate import fmu_simulate # import here to avoid circular import
    
    # wrap fmu_simulate to include idx and catch exceptions. Time out simulations by using ThreadPoolExecutor.
    def fmu_simulate_wrapped(idx, *args, **kwargs): 
        t0 = time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fmu_simulate, *args, **kwargs)
            try:
                res = future.result(timeout=cfg.pModel.RawData.Solver.timeout)
                res['timeout'] = False
            except TimeoutError:
                res = {'success': False, 'error_messages': ['fmu_simulate timed out limit of {}s'.format(cfg.pModel.RawData.Solver.timeout)], 'timeout': True}
        res ['idx'], res['time'] = idx, time() - t0
        return res
    
    # create dask client
    from dask.distributed import Client, as_completed, LocalCluster, wait
    _n_workers = os.cpu_count()-2 if cfg.multiprocessing_processes is None else cfg.multiprocessing_processes
    logging.info('Setting up dask client with {} workers'.format(_n_workers))
    # increasing the allowed failures helps dealing with fmus that do not clean up memory usage
    # if this does not help, set it manually with "export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=35" in the terminal
    dask.config.set({'distributed.scheduler.allowed-failures': _n_workers + 4})
    logging.info('set distributed.scheduler.allowed-failures to {}'.format(_n_workers + 4))
    # trim memory usage
    dask.config.set({'distributed.worker.memory.target': 0.95})
    dask.config.set({'distributed.worker.memory.spill': 0.95})
    dask.config.set({'distributed.worker.memory.pause': 0.95}) # this stops assigning new tasks to the worker
    dask.config.set({'distributed.worker.memory.terminate': 0.90})
    # set logging level to info
    logging.getLogger('distributed.nanny').setLevel(logging.INFO)
    logging.info('set distributed.worker.memory.target, distributed.worker.memory.spill, distributed.worker.memory.pause, distributed.worker.memory.terminate to 0.95')
    cluster = LocalCluster(n_workers = _n_workers,
                           threads_per_worker = 1, 
                           processes = True, 
                            memory_limit = cfg.memory_limit_per_worker,
                            )
    client = Client(cluster)
    # set logging level to warning
    logging.getLogger('distributed.worker').setLevel(logging.CRITICAL)
    logging.info(client)
    futures = []
    t0 = time()
    logging.info('view diagnostic dashboard at: http://localhost:8787')
    logging.info('view per worker diagnostics at: http://127.0.0.1:8787/info/main/workers.html')
    logging.info('\t logs on this page show fmu simulation progress')
    client.forward_logging(level=logging.WARNING)

    # open raw data file
    raw_data = h5py.File(filepath_raw_data(cfg), 'a')

    # counters for logging
    _n_completed = 0
    _n_failed = 0
    _n_timedout = 0
    _n_finished = 0

    # categories for results: started, completed, failed, timemout, processing time
    raw_data.create_group('logs')
    raw_data.create_dataset('logs/completed', data=np.zeros((cfg.pModel.RawData.n_samples,), dtype=bool))
    raw_data.create_dataset('logs/sim_failed', data=np.zeros((cfg.pModel.RawData.n_samples,), dtype=bool))
    raw_data.create_dataset('logs/timedout', data=np.zeros((cfg.pModel.RawData.n_samples,), dtype=bool))
    raw_data.create_dataset('logs/processing_time', (cfg.pModel.RawData.n_samples,))

    step_tasks_i = min(10000, cfg.pModel.RawData.n_samples)
    max_submission_rounds = cfg.pModel.RawData.n_samples // step_tasks_i 
    for submission_round, max_submission_i in enumerate(range(0, cfg.pModel.RawData.n_samples, step_tasks_i)):
        # submit simulation as futures to dask client (the computation does not block the main thread)
        min_tasks_i = max_submission_i
        max_tasks_i = max_submission_i + step_tasks_i
        logging.info('submission round {}/{}: submitting and computing tasks {}-{} of {}'.format(submission_round +1, max_submission_rounds, min_tasks_i, max_tasks_i, cfg.pModel.RawData.n_samples))
        for i in range(min_tasks_i, max_tasks_i):
            futures.append(client.submit(fmu_simulate_wrapped, i,
                    fmu_path = str(Path(cfg.pModel.RawData.fmuPath).resolve()),
                    state_names = cfg.pModel.RawData.states.keys(),
                    get_state_derivatives = cfg.pModel.RawData.states_der_include,
                    initial_state_values = initial_state_values[i] if initial_state_values is not None else None,
                    parameter_names = cfg.pModel.RawData.parameters.keys() if cfg.pModel.RawData.parameters is not None else None,
                    parameter_values = param_values[i] if param_values is not None else None,
                    control_names = cfg.pModel.RawData.controls.keys(),
                    control_values = ctrl_values[i] if ctrl_values is not None else None,
                    control_from_model_names = cfg.pModel.RawData.controls_from_model if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model else None,
                    output_names = cfg.pModel.RawData.outputs,
                    start_time = cfg.pModel.RawData.Solver.simulationStartTime, 
                    stop_time = cfg.pModel.RawData.Solver.simulationEndTime, 
                    fmu_simulate_step_size = cfg.pModel.RawData.Solver.timestep,
                    fmu_simulate_tolerance = cfg.pModel.RawData.Solver.tolerance,
                    key = i,
                )
            )

        # time logging variables, create new for every submission round (to avoid too large dict)
        start_time_futures = {}
        if cfg.pModel.RawData.Solver.timeout is not None:
            _timeout_worker_restart = min(1.2 * cfg.pModel.RawData.Solver.timeout, cfg.pModel.RawData.Solver.timeout + 30)
        _runtime_per_future = [0.01] * _n_workers # to avoid too many requests to the scheduler, we will sleep for the average runtime of a future divided by the number of workers
        
        # progressively process the incoming results, catch exception and save if necessary
        try: # for catching all exceptions and saving the data that was generated so far
            while not len(futures) == 0:
                # determine which futures run too long and restart their workers
                worker_states = client.run(lambda dask_worker: dask_worker.state.tasks)
                _workers_to_restart = []
                for worker, tasks in worker_states.items():
                    _restart_worker = False
                    for key, task_state in tasks.items():
                        if task_state.state == 'executing':
                            if key not in start_time_futures:
                                start_time_futures[key] = time()
                            else:
                                if cfg.pModel.RawData.Solver.timeout is not None:
                                    if time() - start_time_futures[key] > _timeout_worker_restart:
                                        logging.warning('fmu {} is running for more than {}s, we will restart its worker {}'.format(key, _timeout_worker_restart, worker))
                                        _restart_worker = True
                                        # also remove the future from the list of futures
                                        for future in futures:
                                            if future.key == key:
                                                future.cancel() # cancel the future to avoid further processing
                                                future.release()
                                                futures.remove(future)
                                                _n_timedout += 1
                                                raw_data['logs/timedout'][key] = True
                    if _restart_worker:
                        _workers_to_restart.append(worker)
                client.restart_workers(workers=_workers_to_restart)
                # loop over futures and check if they are done
                for future in futures:
                    if future.done():
                        if future.cancelled():
                            logging.error('fmu {} was cancelled. This should not happen!'.format(future.key))
                            # print reason
                            logging.error('Reason: ')
                            logging.error(future.exception())
                            logging.error('Traceback: ')
                            logging.error(future.traceback())
                            raise Exception('fmu {} was cancelled. This should not happen!'.format(future.key))
                        # get id of result
                        res = future.result()
                        idx = res['idx']

                        # handle counters and save logs
                        raw_data['logs/processing_time'][idx] = res['time']

                        if res['success'] is False:
                            if not res['timeout']:
                                logging.error('fmu {} simulation failed, due to the following errors'.format(res['idx']))
                                for error in res['error_messages']:
                                    logging.error(error)
                                raw_data['logs/sim_failed'][idx] = True
                                _n_failed += 1
                            else:
                                logging.error('fmu {} timed out after {}s'.format(res['idx'], cfg.pModel.RawData.Solver.timeout))
                                raw_data['logs/timedout'][idx] = True
                                _n_timedout += 1
                        else: # if completed
                            raw_data['logs/completed'][idx] = True
                            _n_completed += 1
                        
                        # unpack results
                        if res['timeout'] is False:
                            outputs, states, states_der, controls_from_model = res['outputs'], res['states'], res['states_der'], res['controls_from_model']
                            if cfg.pModel.RawData.outputs is not None:
                                raw_data['outputs'][idx] = outputs
                            if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model is True:
                                raw_data['controls'][idx] = controls_from_model
                            raw_data['states'][idx] = states
                            if cfg.pModel.RawData.states_der_include:
                                raw_data['states_der'][idx] = states_der

                        # mark future as done
                        future.release() # especially necessary when simulating ClaRa
                        futures.remove(future) # remove future from list of futures
                        _n_finished += 1
                        
                        _str0 = 'Progress: '
                        _str1 = progress_string(_n_finished / cfg.pModel.RawData.n_samples)
                        _str2 = ' \t - \tfinished {}/{} ({}%)\t {} ({}%) successful, {} ({}%) failed, {} ({}%) timed out \t fmu {} took {} sec'.format(
                            _n_finished, cfg.pModel.RawData.n_samples, round(_n_finished / cfg.pModel.RawData.n_samples * 100, 1),
                            _n_completed, round(_n_completed / _n_finished * 100, 2),
                            _n_failed, round(_n_failed / _n_finished * 100, 2),
                            _n_timedout, round(_n_timedout / _n_finished * 100, 2),
                            idx, round(res['time'], 3),
                            )
                        logging.info(_str0 + _str1 + _str2)
                        _runtime_per_future.append(res['time'])
                
                sleep(np.mean(_runtime_per_future)/_n_workers) # sleep for the average runtime of a future to avoid too many requests to the scheduler

        except BaseException as e:
            logging.error('Error in data generation: {}'.format(e))
            logging.error(e)
            logging.error('catching exception to save the data that was generated so far')
            raise e
    
    logging.info('multiprocessing time: {}'.format(time() - t0))

    # close raw data file
    raw_data.close()
    logging.info('closed raw data file, all data saved. Proceeding errors have no influence on the data.')
    for future in futures:
        future.release()
    client.shutdown()
    cluster.close()

def sample_all_values(cfg: data_gen_config) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Sample all input values (initial states, parameters, controls) according to config.
    
    Orchestrates sampling for all simulation inputs based on the configured sampling strategies.
    Returns None for any input category not included in the config. For parameters, if sampling 
    is disabled, returns default parameter values for all samples.
    
    Supported sampling strategies:
        - Initial states: 'R' (random uniform)
        - Controls: 'R', 'RO' (random with offset), 'ROCS', 'RROCS', 'RS' (random steps), 
          'RF' (frequency response), 'file' (from CSV), 'constantInput' (from Excel)
        - Parameters: 'R' (random uniform)
    
    Args:
        cfg: Data generation configuration containing:

            - cfg.pModel.RawData.initial_states_include: whether to sample initial states
            - cfg.pModel.RawData.initial_states_sampling_strategy: 'R' for random uniform
            - cfg.pModel.RawData.controls_include: whether to sample controls
            - cfg.pModel.RawData.controls_sampling_strategy: strategy name (see above)
            - cfg.pModel.RawData.parameters_include: whether to sample parameters
            - cfg.pModel.RawData.parameters_sampling_strategy: 'R' for random uniform
            - cfg.pModel.RawData.parameters: dict with parameter bounds and defaults
            - cfg.pModel.RawData.n_samples: number of samples to generate
    
    Returns:
        Tuple of (initial_state_values, param_values, ctrl_values) where:

            - initial_state_values: np.ndarray (n_samples, n_states) or None
            - param_values: np.ndarray (n_samples, n_parameters) or None
            - ctrl_values: np.ndarray (n_samples, n_controls, sequence_length) or None
    """
    if cfg.pModel.RawData.initial_states_include:
        if cfg.pModel.RawData.initial_states_sampling_strategy == 'R':
            initial_state_values = random_sampling_initial_states(cfg)
        logging.info('initial_state_values.shape: {}'.format(initial_state_values.shape))
    else:
        initial_state_values = None
        logging.info('No initial state sampling included in raw data generation')
    
    if cfg.pModel.RawData.controls_include:
        if cfg.pModel.RawData.controls_sampling_strategy == 'R':
            ctrl_values = random_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RO':
            ctrl_values = random_sampling_controls_w_offset(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'ROCS':
            ctrl_values = random_sampling_controls_w_offset_cubic_splines_old_clip_manual(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RROCS':
            ctrl_values = random_sampling_controls_w_offset_cubic_splines_clip_random(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RS':
            ctrl_values = random_steps_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RF':
            ctrl_values = random_frequency_response_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'file':
            ctrl_values = load_controls_from_file(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'constantInput':
            ctrl_values = constant_input_simulation_from_excel(cfg)
        logging.info('ctrl_values.shape: {}'.format(ctrl_values.shape))
    else:
        ctrl_values = None
        logging.info('No control sampling included in raw data generation')

    if cfg.pModel.RawData.parameters_include:
        if cfg.pModel.RawData.parameters_sampling_strategy == 'R':
            param_values = random_sampling_parameters(cfg)
    else:
        # save default parameter values
        if cfg.pModel.RawData.parameters is not None:
            _param_default = [cfg.pModel.RawData.parameters[key][2] for key in cfg.pModel.RawData.parameters.keys()]
            param_values = [_param_default for _ in range(cfg.pModel.RawData.n_samples)]
            param_values = np.array(param_values)
        else:
            param_values = None
        logging.info('No parameter sampling included in raw data generation')
    if param_values is not None:
        logging.info('param_values.shape: {}'.format(param_values.shape))
        
    return initial_state_values, param_values, ctrl_values
    
def run_data_generation(cfg: data_gen_config) -> None:
    """Main orchestration function for raw data generation pipeline.
    
    Complete raw data generation workflow:

    1. Convert and validate configuration
    2. Set reproducibility seed (np.random.seed(42))
    3. Create raw data HDF5 file with pre-allocated datasets
    4. Sample all input values (initial states, parameters, controls)
    5. Write sampled inputs and metadata to HDF5 file
    6. Execute parallel FMU simulations via data_generation()
    7. Save configuration as YAML file
    
    The function prompts for confirmation before overwriting existing raw data files 
    (unless cfg.overwrite is True). It creates the complete HDF5 structure including:

    - Time vector and sampled inputs (initial_states, parameters, controls)
    - Pre-allocated arrays for simulation outputs (states, states_der, outputs)
    - Metadata attributes (creation_date, config YAML)
    - Log datasets for tracking simulation status
    
    This is the Hydra-decorated entry point called by main().
    
    Args:
        cfg: Data generation configuration (automatically populated by Hydra from YAML + CLI args).
            Key settings include:

            - pModel.RawData: all generation parameters (FMU path, bounds, solver, sampling strategies)
            - overwrite: if True, skip confirmation prompt for existing files
            - multiprocessing_processes: number of parallel workers
            - memory_limit_per_worker: memory limit per Dask worker
    
    Notes:
        - Sets np.random.seed(42) for reproducibility (added 2024-11-23).
        - Raw data HDF5 file path determined by filepath_raw_data(cfg).
        - Config YAML path determined by filepath_raw_data_config(cfg).
        - The HDF5 file config attribute stores OmegaConf.to_yaml(cfg.pModel.RawData).
        - Creation date is recorded both in HDF5 attrs and in the config YAML.
    """
    cfg = convert_cfg_to_dataclass(cfg)

    # added np.seed for reproducibility on 23.11.2024 (databases generated before this date are not exactly reproducible)
    np.random.seed(42)
    
    # create hdf5 file for raw data
    if os.path.exists(filepath_raw_data(cfg)) and cfg.overwrite is False:
        response = input(f"File {filepath_raw_data(cfg)} already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborting data generation.")
            sys.exit(0)
    log_overwriting_file(filepath_raw_data(cfg))
    raw_data = h5py.File(filepath_raw_data(cfg), 'w')

    # sample initial states, parameters and controls with given sampling strategy
    initial_state_values, param_values, ctrl_values = sample_all_values(cfg)

    if initial_state_values is not None:
        raw_data.create_dataset('initial_states', data=initial_state_values)

    if param_values is not None:
        raw_data.create_dataset('parameters', data=param_values)
        raw_data.create_dataset('parameters_names', data=np.array(list(cfg.pModel.RawData.parameters.keys()), dtype='S'))

    if ctrl_values is not None and cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model is False:
        raw_data.create_dataset('controls', data=ctrl_values)
        raw_data.create_dataset('controls_names', data=np.array(list(cfg.pModel.RawData.controls.keys()), dtype='S'))

    # generate time vector
    time = np.arange(cfg.pModel.RawData.Solver.simulationStartTime, cfg.pModel.RawData.Solver.simulationEndTime + cfg.pModel.RawData.Solver.timestep, cfg.pModel.RawData.Solver.timestep)

    # allocate memory in hdf5 file for raw data
    raw_data.create_dataset('time', data=time)
    if cfg.pModel.RawData.outputs is not None:
        raw_data.create_dataset('outputs', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.outputs), len(time)))
        raw_data.create_dataset('outputs_names', data=np.array(list(cfg.pModel.RawData.outputs), dtype='S'))
    if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model is True:
        raw_data.create_dataset('controls', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls_from_model), len(time)))
        raw_data.create_dataset('controls_names', data=np.array(list(cfg.pModel.RawData.controls_from_model), dtype='S'))
    raw_data.create_dataset('states', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states), len(time)))
    raw_data.create_dataset('states_names', data=np.array(list(cfg.pModel.RawData.states.keys()), dtype='S'))
    if cfg.pModel.RawData.states_der_include:
        raw_data.create_dataset('states_der', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states), len(time)))
        raw_data.create_dataset('states_der_names', data=np.array(list('der({})'.format(key) for key in cfg.pModel.RawData.states.keys()), dtype='S'))

    # add creation date (YYYY-MM-DD HH:MM:SS)
    creation_date = datetime.now()
    raw_data.attrs['creation_date'] = str(creation_date)
    cfg.pModel.RawData.creation_date = str(creation_date)
    logging.info('added creation date: {} to hdf5-file and config.yaml'.format(creation_date))

    # add config fields to hdf5 file
    raw_data.attrs['config'] = OmegaConf.to_yaml(cfg.pModel.RawData)
    # close hdf5 file
    raw_data.close()

    # generate raw data and save it to hdf5 file
    data_generation(cfg, initial_state_values, param_values, ctrl_values)

    # save pModel config as yaml
    log_overwriting_file(filepath_raw_data_config(cfg))
    OmegaConf.save(cfg.pModel.RawData, filepath_raw_data_config(cfg))

def main():
    """CLI entry point for raw data generation.
    
    Sets up Hydra configuration management and launches run_data_generation(). 
    
    Hydra automatically:

    - Loads the data_generation.yaml config from the auto-detected config directory
    - Parses command-line overrides
    - Creates a working directory for outputs
    - Injects the composed config into run_data_generation()
    
    Usage:
        python raw_data_generation.py [overrides]
        
    Examples:

        python raw_data_generation.py pModel.RawData.n_samples=1000
        python raw_data_generation.py pModel=SHF overwrite=true
    """
    cs = get_config_store()
    config_dir = config_dir_auto_recognize()
    config_name = 'data_generation'
    hydra.main(config_path=str(config_dir.absolute()), config_name=config_name, version_base=None)(run_data_generation)()

if __name__ == '__main__':
    main()