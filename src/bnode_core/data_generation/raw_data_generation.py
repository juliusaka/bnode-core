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
import h5py
import shutil
from pathlib import Path
import logging
from omegaconf import OmegaConf
from datetime import datetime
from time import time, sleep
import dask
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from bnode_core.config import data_gen_config, get_config_store, convert_cfg_to_dataclass
from bnode_core.filepaths import filepath_raw_data, log_overwriting_file, filepath_raw_data_config, config_dir_auto_recognize
import bnode_core.data_generation.sampling as sampling

from typing import Tuple, Optional, List


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
                                                _n_finished += 1
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
           'RSS' (random sequential steps),
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
            initial_state_values = sampling.random_sampling_initial_states(cfg)
        logging.info('initial_state_values.shape: {}'.format(initial_state_values.shape))
    else:
        initial_state_values = None
        logging.info('No initial state sampling included in raw data generation')
    
    if cfg.pModel.RawData.controls_include:
        if cfg.pModel.RawData.controls_sampling_strategy == 'R':
            ctrl_values = sampling.random_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RO':
            ctrl_values = sampling.random_sampling_controls_w_offset(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'ROCS':
            ctrl_values = sampling.random_sampling_controls_w_offset_cubic_splines_old_clip_manual(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RROCS':
            ctrl_values = sampling.random_sampling_controls_w_offset_cubic_splines_clip_random(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RS':
            ctrl_values = sampling.random_steps_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RSS':
            ctrl_values = sampling.random_sequential_steps_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RF':
            ctrl_values = sampling.random_frequency_response_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RFS':
            ctrl_values = sampling.random_fourrier_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'file':
            ctrl_values = sampling.load_controls_from_file(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'constantInput':
            ctrl_values = sampling.constant_input_simulation_from_excel(cfg)
        logging.info('ctrl_values.shape: {}'.format(ctrl_values.shape))
    else:
        ctrl_values = None
        logging.info('No control sampling included in raw data generation')

    if cfg.pModel.RawData.parameters_include:
        if cfg.pModel.RawData.parameters_sampling_strategy == 'R':
            param_values = sampling.random_sampling_parameters(cfg)
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

    # copy hydra folder to output folder
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    dest_dir = Path(filepath_raw_data(cfg)).parent / 'hydra'
    logging.info('Copying hydra output folder from {} to {}'.format(hydra_output_dir, dest_dir))
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(hydra_output_dir, dest_dir)

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