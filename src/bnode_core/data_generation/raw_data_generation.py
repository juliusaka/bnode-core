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

def random_sampling_parameters(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.parameters[key][0], cfg.pModel.RawData.parameters[key][1]] for key in cfg.pModel.RawData.parameters.keys()]
    param_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.parameters.keys())))
    for i in range(len(cfg.pModel.RawData.parameters.keys())):
        param_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return param_values

def random_sampling_controls(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    for i in range(len(cfg.pModel.RawData.controls.keys())):
        ctrl_values[:, i, :] = np.random.uniform(bounds[i][0], bounds[i][1], (cfg.pModel.RawData.n_samples, cfg.pModel.RawData.Solver.sequence_length))
    # last control input is not used.
    return ctrl_values

def random_sampling_controls_w_offset(cfg: data_gen_config, seq_len: int = None, n_samples: int = None):
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

def random_sampling_controls_w_offset_cubic_splines_old_clip_manual(cfg: data_gen_config):
    '''
    also known as ROCS
    ROCS fills out the control space more than RROCS, because after the cubic spline interpolation, which tend to exceeds the bounds, 
    the values are clipped to the bounds.
    '''
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

def random_sampling_controls_w_offset_cubic_splines_clip_random(cfg: data_gen_config):
    '''
    also known as RROCS
    RROCS fills out the control space less than ROCS, because after the cubic spline interpolation, which tend to exceeds the bounds,
    the values are not just clipped to the bounds, but the base and delta are again randomly sampled.
    This ensures that on differen levels with differnt degree of variation, the control space is filled out more evenly.
    '''
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

def random_steps_sampling_controls(cfg: data_gen_config):
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

def random_frequency_response_sampling_controls(cfg: data_gen_config):
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

def load_controls_from_file(cfg: data_gen_config):
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

def constant_input_simulation_from_excel(cfg: data_gen_config):
    """
    This function is used to simulate the constant input simulation from the excel file.
    The Excel file should have the following structure:
    - First row: column names as in config-files
    - Columns: each row defines one combination of control values for one simulation

    The data sheet should have the name 'Tabelle1' and the columns should be named (in row 1) as the control variables.

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


def random_sampling_initial_states(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.states[key][0], cfg.pModel.RawData.states[key][1]] for key in cfg.pModel.RawData.states.keys()]
    initial_state_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states.keys())))
    for i in range(len(cfg.pModel.RawData.states.keys())):
        initial_state_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return initial_state_values

def progress_string(progress: float, length: int = 10) -> str:
    """
    Returns a visual progress string of the form '|||||.....' for a given progress value in [0, 1].
    Args:
        progress (float): Progress value between 0 and 1.
        length (int): Total length of the progress string.
    Returns:
        str: Progress bar string.
    """
    progress = max(0, min(1, progress))  # Clamp to [0, 1]
    n_complete = int(round(progress * length))
    n_remaining = length - n_complete
    return '|' * n_complete + '.' * n_remaining

def data_generation(cfg: data_gen_config,
                    initial_state_values: np.ndarray = None,
                    param_values: np.ndarray = None,
                    ctrl_values: np.ndarray = None):
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

    # add failed idx to raw data file (to ensure backward compatibility). Here, failed idx are simulations that were not completed, but not necessarily failed.
    idx_failed = np.where(raw_data['logs/completed'][:] == False)[0].tolist() 
    raw_data.create_dataset('failed_idx', data=np.array(idx_failed), dtype=int)
    logging.info('Simulation that did not complete: {}'.format(idx_failed))
    logging.info('Added not completed simulation as group failed_idx to raw data file.')

    # close raw data file
    raw_data.close()
    logging.info('closed raw data file, all data saved. Proceeding errors have no influence on the data.')
    for future in futures:
        future.release()
    client.shutdown()
    cluster.close()

def sample_all_values(cfg):
    # sample initial states, parameters and controls with given sampling strategy
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
    
def run_data_generation(cfg: data_gen_config):
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
    cs = get_config_store()
    config_dir = config_dir_auto_recognize()
    config_name = 'data_generation' 
    hydra.main(config_path=str(config_dir.absolute()), config_name=config_name, version_base=None)(run_data_generation)()

if __name__ == '__main__':
    main()