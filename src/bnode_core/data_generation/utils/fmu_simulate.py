import uuid
import logging
import time

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import fmpy
import numpy as np
import shutil
import sys
import os
import io
from bnode_core.config import get_config_store

class logger_max_rate:
    def __init__(self, logger: logging.Logger, max_rate: float = 30):
        self.last_log_time = time.time() - 1.1*max_rate
        self.logger = logger
        self.max_freq = max_rate
    
    def info(self, msg):
        if time.time() - self.last_log_time > self.max_freq:
            self.logger.info(msg)
            self.last_log_time = time.time()

def fmu_simulate(fmu_path,
                state_names = None,
                get_state_derivatives = False,
                initial_state_values = None,
                parameter_names = None,
                parameter_values = None,
                control_names = None,
                control_values = None,
                control_from_model_names = None,
                output_names = None,
                start_time=0.0, 
                stop_time=1800.0, 
                fmu_simulate_step_size=1,
                fmu_simulate_tolerance=1e-4,
                load_result_from_file = False, 
                filepath_multiprocessing = os.path.join(os.getcwd(), '_wrk'),
                copy_fmu = False,
                change_log_level = True):
    """
    Simulate an FMU with the given parameters and return the results.
    It is possible to read the results from a file instead of communicating with the FMU.
    This can be useful for multiprocessing, because communication overhead might be reduced.
    However, saving the mat-file and opening was slower than communicating with the FMU on Windows. 
    Maybe the mat-file generating code is not able to be parallelized on Windows.
    On Linux, it works. However, I did not test if writing and reading the mat-file is faster than 
    communicating with the FMU on Linux.
    After simulation, the results are scaled by multiplying and substracting as in the config file.
    The copy_fmu option was turned off, because it wasnt seen that unpacking the FMU from the same directory
    multiple times caused problems. 

    Parameters
    ----------
    modelName : str
        name of the model
    input_names : list
        list of parameter names
    output_names : list
        list of output names
    parameter_values : list
        list of parameter values
    fmu_path : str
        path to the fmu
    start_time : float
        start time of the simulation
    stop_time : float
        stop time of the simulation
    fmu_simulate_step_size : float
        step size of the simulation
    fmu_simulate_tolerance : float 
        tolerance of the simulation
    verbose : bool
        if True, print additional information (default: False)
    load_result_from_file : bool
        if True, the simulation communicates with the fmu at every simulation timestep. 
        if False, the simulation reads the results from a mat-file (default: False)
        not properly implemented anymore, so not recommended do use
    filepath_multiprocessing : str
        path to the multiprocessing directory (default: os.path.join(working_directory, '_wrk'))
    """
    if load_result_from_file is True:
        raise ValueError('load_result_from_file is removed!')

    t_start = time.time()

    # determine if there is a logger 'distributed.worker' to see if we are in a dask worker
    if __name__ != '__main__' and 'distributed.worker' in logging.root.manager.loggerDict:
        logger = logging.getLogger('distributed.worker')
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture
        logger.setLevel(logging.CRITICAL)
    else:
        stdout_capture = None
        logger = logging.getLogger(__name__)
    # logger = logging.getLogger('distributed.worker')
    # with redirect_stdout(logger):
    logger_delayed = logger_max_rate(logger, 10)
    logger.info('fmu_simulate started')
    # create uniqueId 
    uniqueId = str(uuid.uuid4())


    if copy_fmu:
        os.makedirs(os.path.join(filepath_multiprocessing), exist_ok=True)
        os.makedirs(os.path.join(filepath_multiprocessing, uniqueId), exist_ok=True)
        new_fmu_path = os.path.join(filepath_multiprocessing, uniqueId, os.path.split(fmu_path)[1])
        shutil.copy(fmu_path, new_fmu_path)
    else:
        new_fmu_path = fmu_path

    '''simulation'''
    # read the model description
    model_description = read_model_description(new_fmu_path)
    model_variables = model_description.modelVariables
    # collect the value references and indices
    vrs = {}
    indices = {}
    variability = {}
    i = 0
    for variable in model_variables:
        vrs[variable.name] = variable.valueReference
        indices[variable.name] = i
        variability[variable.name] = variable.variability
        i += 1

    # get the value references for the states, parameters, inputs and outputs
    state_refs = [vrs[name] for name in state_names] if state_names is not None else None
    state_der_refs = [vrs['der({})'.format(name) ] for name in state_names] if state_names is not None and get_state_derivatives is True else None
    parameter_refs = [vrs[name] for name in parameter_names] if parameter_names is not None else None
    control_refs = [vrs[name] for name in control_names] if control_names is not None else None
    output_refs = [vrs[name] for name in output_names] if output_names is not None else None
    control_from_model_refs = [vrs[name] for name in control_from_model_names] if control_from_model_names is not None else None

    # prepare vectors to save the results
    steps = int(np.ceil((stop_time - start_time) / fmu_simulate_step_size))
    outputs = np.nan * np.ones((len(output_refs), steps + 1)) if output_refs is not None else None
    states = np.nan * np.ones((len(state_refs), steps + 1)) if state_refs is not None else None
    states_der = np.nan * np.ones((len(state_der_refs), steps + 1)) if state_der_refs is not None else None
    controls_from_model = np.nan * np.ones((len(control_from_model_refs), steps + 1)) if control_from_model_refs is not None else None

    # prepare error handling
    sim_succesful = True
    error_messages = []

    # extract the FMU
    unzipdir = extract(new_fmu_path)
    fmu = FMU2Slave(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.coSimulation.modelIdentifier,
                    instanceName=uniqueId)
    logger.info('fmu extracted')

    # initialize
    try:
        fmu.instantiate()
        fmu.setupExperiment(startTime=start_time, tolerance=fmu_simulate_tolerance)
        # set the initial state
        if initial_state_values is not None:
            fmu.setReal(vr=state_refs, value=initial_state_values)
        # set the parameters
        if parameter_values is not None:
            # fmu.setReal(vr=parameter_refs, value=parameter_values)
            for i in range(len(parameter_values)):
                # integer or real method
                if np.dtype(parameter_values[i]) == np.int64 or np.dtype(parameter_values[i]) == np.int32:
                    fmu.setInteger(vr=[parameter_refs[i]], value=[parameter_values[i]])
                else:
                    fmu.setReal(vr=[parameter_refs[i]], value=[parameter_values[i]])

        # HERE CONTROL
        if control_values is not None:
            fmu.setReal(vr=control_refs, value=control_values[:,0])
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()
        logger.info('fmu initialized')
    except Exception as e:
        error_messages += ['Initialization failed, aborting simulation']
        error_messages += [str(e)]
        sim_succesful = False
        # final length of outputs is 0

    def record_data(i):
        if output_names is not None:
            outputs[:,i] = np.array(fmu.getReal(output_refs))
        if control_from_model_names is not None: #TODO: first order input interpolation better for training?
            controls_from_model[:,i] = np.array(fmu.getReal(control_from_model_refs))
        if state_names is not None:
            states[:,i] = np.array(fmu.getReal(state_refs))
            if get_state_derivatives is True:
                states_der[:,i] = np.array(fmu.getReal(state_der_refs))

    if sim_succesful is True: # only start if no failure in initialization
        try:
            t = start_time
            # simulation loop
            logger.info('start simulation loop')
            for i in range(steps):
                # set the controls
                if control_values is not None:
                    fmu.setReal(vr=control_refs, value=control_values[:,i])
                # record the data
                record_data(i) 
                # advance the time
                fmu.doStep(currentCommunicationPoint=t, communicationStepSize=fmu_simulate_step_size)
                t += fmu_simulate_step_size
                logger_delayed.info('step ' + str(i) + '/' + str(steps) + ' done.')
                #print('step ' + str(i) + ' of ' + str(int(np.ceil((stop_time - start_time) / fmu_simulate_step_size))) + ' done.')
        except Exception as e:
            error_messages += ['Simulation failed at t = ' + str(t - fmu_simulate_step_size) + ', aborting simulation']
            error_messages += [str(e)]
            sim_succesful = False
    
    # final length of outputs is i+1
    # record the final output
    if sim_succesful:
        record_data(i+1)
    
    try:
        # terminate
        fmu.terminate()
        #fmu.freeInstance()
        fmu.fmi2FreeInstance(fmu.component)
        fmpy.freeLibrary(fmu.dll._handle)

        # delete fmu directory
        shutil.rmtree(unzipdir, ignore_errors=True)
        if copy_fmu:
            shutil.rmtree(os.path.split(new_fmu_path)[0], ignore_errors=True)
    except Exception as e:
        error_messages += ['clearnup failed, aborting simulation']
        error_messages += [str(e)]

    processing_time = time.time() - t_start

    if sim_succesful:
        logger.info('fmu_simulate took {:.4f} seconds for {} steps and was successful'.format(processing_time, i))
    else:
        logger.info('fmu_simulate took {:.4f} seconds for {} steps but failed'.format(processing_time, i))

    # if sim_succesful is False, print the error messages
    if sim_succesful is False:
        if stdout_capture is not None:
            stdout_output = stdout_capture.getvalue()
            sys.stdout = sys.__stdout__ # reset stdout to its original value
            _split = '... Error message'
            if _split in stdout_output:
                split_index = stdout_output.find(_split)
                start_index = max(0, split_index - 1500)
                stdout_output = stdout_output[start_index:]
                error_messages += ['...Last 1500 characters before error message and error message:\n' + stdout_output]
    for error_message in error_messages:
        logger.error(error_message)
    # make result dictionary
    res = {'success': sim_succesful, 'outputs': outputs, 'states': states, 'states_der': states_der, 'controls_from_model': controls_from_model, 'error_messages': error_messages, 'time': processing_time}
    return res

from bnode_core.config import data_gen_config, get_config_store, convert_cfg_to_dataclass
import bnode_core.filepaths as filepaths
from pathlib import Path
import numpy as np
import hydra
import os
from pathlib import Path

def main(cfg: data_gen_config, plot: bool = False, return_res: bool = False) -> None:
    """
    test the fmu_simulate function
    """
    cfg = convert_cfg_to_dataclass(cfg)

    # sample control values
    from bnode_core.data_generation.raw_data_generation import sample_all_values
    cfg.pModel.RawData.n_samples = 1
    initial_state_values, param_values, ctrl_values = sample_all_values(cfg)
    initial_state_values = initial_state_values[0,:] if initial_state_values is not None else None
    param_values = param_values[0,:] if param_values is not None else None
    ctrl_values = ctrl_values[0,:,:] if ctrl_values is not None else None
    # print all values
    if initial_state_values is not None:
        print('initial_state_values')
        for key, val in zip(cfg.pModel.RawData.states.keys(), initial_state_values):
            print('{}: {}'.format(key, val))
    else:
        print('initial_state_values: None')
    if cfg.pModel.RawData.parameters is not None:
        print('param_values')
        for key, val in zip(cfg.pModel.RawData.parameters.keys(), param_values if param_values is not None else [None]):
            print('{}: {}'.format(key, val))
    else:
        print('param_values: None')
    print('ctrl_values')
    if ctrl_values is not None:
        for key, val in zip(cfg.pModel.RawData.controls.keys(), ctrl_values if ctrl_values is not None else [None]):
            print('{}: {} ...'.format(key, val[0:3]))
    else:
        print('ctrl_values: None')

    t0 = time.time()
    res = fmu_simulate(
        fmu_path = str(Path(cfg.pModel.RawData.fmuPath).resolve()),
        state_names = cfg.pModel.RawData.states.keys(),
        get_state_derivatives=cfg.pModel.RawData.states_der_include,
        initial_state_values = initial_state_values,
        parameter_names = cfg.pModel.RawData.parameters.keys() if cfg.pModel.RawData.parameters is not None else None,
        parameter_values = param_values,
        control_names = cfg.pModel.RawData.controls.keys(),
        control_values = ctrl_values if cfg.pModel.RawData.controls_include else None,
        control_from_model_names = cfg.pModel.RawData.controls_from_model if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model else None,
        output_names = cfg.pModel.RawData.outputs,
        start_time = cfg.pModel.RawData.Solver.simulationStartTime, 
        stop_time = cfg.pModel.RawData.Solver.simulationEndTime, 
        fmu_simulate_step_size = cfg.pModel.RawData.Solver.timestep,
        fmu_simulate_tolerance = cfg.pModel.RawData.Solver.tolerance,
    )
    logging.info('Simulation took ' + str(time.time() - t0) + ' seconds')
    outputs, states, states_der, controls_from_model = res['outputs'], res['states'], res['states_der'], res['controls_from_model']
    logging.info('simulation successful: ' + str(res['success']))

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(5,1, sharex=True)
        # turn grid on
        for i in range(4):
            ax[i].grid()
        x = np.arange(cfg.pModel.RawData.Solver.simulationStartTime, cfg.pModel.RawData.Solver.simulationEndTime + cfg.pModel.RawData.Solver.timestep, cfg.pModel.RawData.Solver.timestep)
        # x, states, states_der, ctrl_values = x[1:], states[:,1:], states_der[:,1:], ctrl_values[:,1:]
        if cfg.pModel.RawData.outputs is not None:
            for i in range(len(cfg.pModel.RawData.outputs)):
                ax[0].plot(x, outputs[i,:], label=cfg.pModel.RawData.outputs[i])
        ax[0].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
        ax[0].set_title('outputs')
        for i in range(len(cfg.pModel.RawData.states.keys())):
            ax[1].plot(x, states[i,:], label=list(cfg.pModel.RawData.states.keys())[i])
        ax[1].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
        ax[1].set_title('states')
        if cfg.pModel.RawData.states_der_include:
            for i in range(len(cfg.pModel.RawData.states.keys())):
                ax[2].plot(x, states_der[i,:], label=list(cfg.pModel.RawData.states.keys())[i])
            ax[2].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
            ax[2].set_title('state derivatives')
        if cfg.pModel.RawData.controls_include:
            for i in range(len(cfg.pModel.RawData.controls.keys())):
                ax[3].plot(x, ctrl_values[i,:], label=list(cfg.pModel.RawData.controls.keys())[i])
            ax[3].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
            ax[3].set_title('controls')
        if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model:
            for i in range(len(cfg.pModel.RawData.controls_from_model)):
                ax[4].plot(x, controls_from_model[i,:], label=cfg.pModel.RawData.controls_from_model[i])
            ax[4].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
            ax[4].set_title('controls from model')
        plt.show()
    if return_res:
        return res

if __name__ == '__main__':
    """
    you can run this script with e.g. to set parameters
    python data_generation/src/fmu_simulate.py pModel.RawData.parameters.u_wall=1.8
    or for multirun
    python data_generation/src/fmu_simulate.py pModel.RawData.parameters.u_wall=1.8,2.0,2.2 --multirun
    """
    # sys.argv += ['pModel.RawData.Solver.simulationEndTime=10']
    # sys.argv += ['pModel=PowerPlant']
    cs = get_config_store()
    cfg_dir, cfg_name = filepaths.get_cfg_from_cli()
    cfg_name = 'data_generation' if cfg_name is None else cfg_name
    hydra.main(config_path=str(Path(cfg_dir).absolute()), config_name=cfg_name, version_base=None)(lambda cfg: main(cfg, plot=True))()