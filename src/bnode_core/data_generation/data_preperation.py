import hydra
import os
import sys
import numpy as np
import h5py
from pathlib import Path
import logging
from datetime import datetime
import shutil
from omegaconf import OmegaConf
import scipy
import tqdm

from config import data_gen_config, cs, convert_cfg_to_dataclass, RawDataClass
from filepaths import filepath_raw_data, log_overwriting_file, filepath_raw_data_config, filepath_dataset, filepath_dataset_config

def load_and_validate_raw_data(cfg):
    """
    loads the config file, validates it and compares it to the current config.
    loads the raw data file
    """
    path_raw_data = filepath_raw_data(cfg)
    path_raw_data_config = filepath_raw_data_config(cfg)
    
    if not path_raw_data.exists():
        raise FileNotFoundError(f'Raw data file does not exist: {path_raw_data}')
    if not path_raw_data_config.exists() and not cfg.pModel.RawData.raw_data_from_external_source:
        raise FileNotFoundError(f'Raw data config file does not exist: {path_raw_data_config}')
    
    # load raw data config
    if not cfg.pModel.RawData.raw_data_from_external_source:
        logging.info('Loading raw data config from {}'.format(path_raw_data_config))
        _raw_data_config_dict = OmegaConf.load(path_raw_data_config)
        _raw_data_config_dict = OmegaConf.to_object(_raw_data_config_dict) # make dict
        raw_data_config = RawDataClass(**_raw_data_config_dict) # validate and convert to dataclass
        logging.info('Validated raw data config from {}'.format(path_raw_data_config))

        # compare raw data config to actual config and raise errors / warnings
        logging.info('Comparing raw data config to current config. Creating copy of raw data config without creation date for comparison.')
        _raw_data_config_wo_creation_date = RawDataClass(**_raw_data_config_dict)
        _raw_data_config_wo_creation_date.creation_date = None
        _flag = False
        if cfg.pModel.RawData != _raw_data_config_wo_creation_date:
            for key in cfg.pModel.RawData.__dict__.keys():
                if cfg.pModel.RawData.__dict__[key] != _raw_data_config_wo_creation_date.__dict__[key]:
                    logging.warning(f'Raw data config does not match current config. Specifically key {key} does not match.')
                    _flag = True
        if _flag:
            logging.info('Overwriting raw data config with raw data config loaded from {}'.format(path_raw_data_config))
            cfg.pModel.RawData = raw_data_config
        else: 
            logging.info('Current config matches loaded raw data config. No overwriting of raw data config.')
    else:
        raw_data_config = None
        logging.info('No raw data given as data from external source. Skipping loading raw data config.')

    # load raw data
    raw_data = h5py.File(path_raw_data, 'r')
    logging.info('Loaded raw data from {}'.format(path_raw_data))

    return raw_data, raw_data_config

def get_position_in_raw_data_file(variable: str, temp_raw_data: h5py.File):
    """
    returns the position of the variable in the raw data file.
    raises ValueError if variable is not found or if it is found in multiple datasets.
    returns [dataset_name, idx] where dataset_name is the name of the dataset and idx is the index of the variable in the dataset.
    """
    # returns dataset name and position in dataset
    search_datasets = [key for key in temp_raw_data.keys() if key.endswith('names')]
    temp = []
    for dataset in search_datasets:
        _temp = np.array(temp_raw_data[dataset], dtype=str)
        if variable in _temp:
            temp.append([dataset, np.where(_temp == variable)[0][0]])
    if len(temp) == 0:
        raise ValueError(f'Variable {variable} not found in raw data file.')
    elif len(temp) > 1:
        raise ValueError(f'Variable {variable} found in multiple datasets in raw data file.')
    else:
        temp[0][0] = temp[0][0].replace('_names', '')
        return temp[0]

def transform_raw_data(cfg: data_gen_config, temp_raw_data: h5py.File, raw_data_config: RawDataClass):
    """
    performs transformations on raw data according to the config
    """

    for variable in cfg.pModel.dataset_prep.transforms.keys():
        dataset_name, idx = get_position_in_raw_data_file(variable, temp_raw_data)
        
        if cfg.pModel.dataset_prep.transforms[variable] == 'temperature_k_to_degC':
            temp_raw_data[dataset_name][:,idx] = temp_raw_data[dataset_name][:,idx] - 273.15
        
        elif cfg.pModel.dataset_prep.transforms[variable] == 'power_w_to_kw':
            temp_raw_data[dataset_name][:,idx] = temp_raw_data[dataset_name][:,idx] / 1000
        
        elif cfg.pModel.dataset_prep.transforms[variable] == 'differentiate':
            # derivative present in raw data
            _states_der_in_dataset = False
            if 'states_der' in temp_raw_data.keys():
                state_der_names = np.array(temp_raw_data['states_der_names'], dtype=str).tolist()
                if 'der({})'.format(variable) in state_der_names:
                    _states_der_in_dataset = True
            # generate interpolating lagrange polynomial
            x = temp_raw_data['time'][:]
            y = temp_raw_data[dataset_name][:,idx]
            y_std = np.std(y)
            y_der_std = np.std(temp_raw_data['states_der'][:, idx]) if _states_der_in_dataset else None
            # allocate dicts for statistics
            error_mae = {'0th': [], '1st': []}
            error_max = {'0th': [], '1st': []}
            logging.info(f'Calculating derivative for variable {variable} in dataset {dataset_name}')
            # loop over all samples, calculate derivative and replace in dataset
            for i in tqdm.tqdm(range(y.shape[0]), desc=f'Calculating derivative for {variable}'):
                interpolator = scipy.interpolate.Akima1DInterpolator(x, y[i, :], method='makima')
                der = [interpolator(x, 0), interpolator(x, 1)]
                if _states_der_in_dataset:
                    der.append(interpolator(x, 2))
                # calculate error statistics
                error_mae['0th'].append(np.mean(np.abs(y[i] - der[0])/ y_std))
                error_max['0th'].append(np.max(np.abs(y[i] - der[0]))/ y_std)
                if _states_der_in_dataset:
                    error_mae['1st'].append(np.mean(np.abs(temp_raw_data['states_der'][i, idx] - der[1])/ y_der_std))
                    error_max['1st'].append(np.max(np.abs(temp_raw_data['states_der'][i, idx] - der[1])/ y_der_std))
                # replace in dataset
                temp_raw_data[dataset_name][i, idx, :] = der[1]
                if _states_der_in_dataset:
                    temp_raw_data['states_der'][i, idx, :] = der[2]
            # print error statistics
            logging.info(f'  Error statistics for differentiating variable {variable} in dataset {dataset_name}, normalized by std:')
            logging.info(f'    Mean Absolute Error (0th): {np.mean(error_mae["0th"])}, std: {np.std(error_mae["0th"])}')
            logging.info(f'    Max Error (0th): {np.max(error_max["0th"])}')
            if _states_der_in_dataset:
                logging.info(f'    Mean Absolute Error (1st): {np.mean(error_mae["1st"])}, std: {np.std(error_mae["1st"])}')
                logging.info(f'    Max Error (1st): {np.max(error_max["1st"])}')
        elif cfg.pModel.dataset_prep.transforms[variable].startswith('evaluate_python_'):
            """
            Transform is a python command that evaluates the variable temp_raw_data[dataset_name][:,idx]
            It should be in the format 'evaluate_python_<command>'
            where <command> is a python command that takes the variable as input.
            denote with # a placeholder for temp_raw_data[dataset_name][:,idx].
            """
            command = cfg.pModel.dataset_prep.transforms[variable].replace('evaluate_python_', '')
            commands = command.split('#')
            command = commands[0] + 'temp_raw_data[dataset_name][:,idx]' + commands[1]
            logging.info(f'Transforming variable {variable} in dataset {dataset_name} with python command: {command}')
            temp_raw_data[dataset_name][:,idx] = eval(command)
        else:
            raise NotImplementedError(f'Transform {cfg.pModel.dataset_prep.transforms[variable]} not implemented.')
        logging.info(f'Transformed variable {variable} in dataset {dataset_name} with transform {cfg.pModel.dataset_prep.transforms[variable]}')
    pass

def replace_hdf5_dataset(dataset_name: str, raw_data: h5py.File, data: np.ndarray, remove: bool = False):
    """
    replaces dataset in raw data file with new data
    """
    if dataset_name not in raw_data.keys():
        raise ValueError(f'Dataset {dataset_name} not found in raw data file.')
    if remove:
        del raw_data[dataset_name]
    else:
        if data.shape != raw_data[dataset_name].shape:
            del raw_data[dataset_name]
            raw_data.create_dataset(dataset_name, data=data)           
        else:
            raw_data[dataset_name][...] = data

@hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen', version_base=None)
def main(cfg: data_gen_config):
    cfg = convert_cfg_to_dataclass(cfg)
    
    # load and validate raw data, copy data to temp file
    raw_data, raw_data_cfg = load_and_validate_raw_data(cfg)
    temp_raw_data_path = Path('temp_raw_data.hdf5')
    temp_raw_data = h5py.File(temp_raw_data_path, 'w')
    for key in raw_data.keys():
        raw_data.copy(key, temp_raw_data)
    logging.info('Copied raw data to temporary file {}'.format(temp_raw_data_path))
    
    # remove failed runs from raw data
    if 'failed_idx' in raw_data.keys():
        remove_runs = raw_data['failed_idx']
        remove_runs = np.array(remove_runs, dtype=int)
        logging.info('Removing failed runs from raw data: {}'.format(remove_runs))
    else:
        logging.info('No failed runs in raw data. Skipping removal of failed runs.')
    
    _remove_runs = []
    if len(cfg.pModel.dataset_prep.filter_trajectories_limits) > 0:
        logging.info('Filtering trajectories in raw data according to filter_trajectories config.')
        for key, value in cfg.pModel.dataset_prep.filter_trajectories_limits.items():
            dataset_name, idx = get_position_in_raw_data_file(key, temp_raw_data)
            if type(value) is list: # apply min/max filter
                if len(value) != 2:
                    raise ValueError(f'Filter for {key} must be a list of length 2, got {value}.')
                logging.info(f'Filtering {dataset_name} for {key} with min {value[0]} and max {value[1]}.')
                idx_lower = (temp_raw_data[dataset_name][:, idx] < value[0])
                idx_upper = (temp_raw_data[dataset_name][:, idx] > value[1])
                idx = np.logical_or(idx_lower, idx_upper)
                if np.sum(idx) > 0:
                    logging.info(f'Found {np.sum(idx)} runs that do not match the filter for {key}. Removing them.')
                    _remove_runs.append(np.nonzero(idx)[0])
    if len(cfg.pModel.dataset_prep.filter_trajectories_expression) > 0:
        raise NotImplementedError('Filtering by expression is implementd, but not yet tested. Please use with caution.')
        logging.info('Filtering trajectories in raw data according to filter_trajectories_by_expression config.')
        for key, args in cfg.pModel.dataset_prep.filter_trajectories_by_expression.items():
            dataset_name, idx = get_position_in_raw_data_file(key, temp_raw_data)
            if type(args) is not list:
                raise ValueError(f'Filter for {key} must be a list of expressions, got {args}.')
            logging.info(f'Filtering {dataset_name} for {key} with expressions {args}.')
            _conditions = []
            for arg in args:
                # evaluate the expression
                arg = arg.replace('#', 'temp_raw_data[dataset_name][:, idx]')
                _conditions.append(eval(arg))
            idx = np.logical_or.reduce(_conditions)
            _remove_runs.append(np.nonzero(idx)[0])
    
    # remove runs from raw data
    if len(_remove_runs) > 0:
        _remove_runs = np.concatenate(_remove_runs)
        remove_runs = np.unique(np.concatenate([remove_runs, _remove_runs]))
        logging.info('Found {} runs to remove from raw data: {}'.format(len(remove_runs), remove_runs))
        remove_runs = np.sort(remove_runs)
        for key in ['states', 'states_der', 'controls', 'outputs', 'parameters']:
            if key in temp_raw_data.keys():
                _temp= np.delete(temp_raw_data[key][:], remove_runs, axis=0)
                replace_hdf5_dataset(key, temp_raw_data, data = _temp)
                logging.info('\tRemoved runs from {}'.format(key))
            else:
                logging.info('\tNo {} in raw data. Skipping removal of failed runs.'.format(key))
        raw_data_cfg.n_samples = raw_data_cfg.n_samples - len(remove_runs)
        logging.info('Updated n_samples in raw_data_config to {}'.format(raw_data_cfg.n_samples))
    else:
        logging.info('No runs to remove from raw data.')

    # perform transforms on raw data
    if not cfg.pModel.RawData.raw_data_from_external_source:
        transform_raw_data(cfg, temp_raw_data, raw_data_cfg)

    # only select variables of interest / states, controls, outputs, parameters

    # helper functions
    def get_idx(names_list: h5py.Dataset, chosen_variables: list):
        if chosen_variables == ['all'] or chosen_variables == ['der(all)'] or chosen_variables == None:
            return np.arange(len(names_list))
        else:
            names_list = np.array(names_list, dtype=str).tolist()
            return [names_list.index(variable) for variable in chosen_variables]
    
    def select_variables_of_interest(type: str, variables: list, remove: bool = False):
        # type is states, states_der, controls, outputs, parameters
        if variables == None:
            remove = True
        if type in temp_raw_data.keys():
            _type_with_names = f'{type}_names' # type_names_str = 'states_names' or 'states_der_names' or 'controls_names' or 'outputs_names' or 'parameters_names'
            idx = get_idx(temp_raw_data[_type_with_names], variables) # get idx
            replace_hdf5_dataset(type, temp_raw_data, data = temp_raw_data[type][:,idx], remove=remove) # replace dataset
            replace_hdf5_dataset(_type_with_names, temp_raw_data, data = temp_raw_data[_type_with_names][idx], remove=remove) # replace dataset
            if remove:
                logging.info(f'Removed dataset {type} from raw data.')
            else:
                logging.info(f'Selected {type} {variables} from raw data.') 
        else:
            logging.info(f'No {type} in raw data. Skipping selection of {type}.')
    
    logging.info('... Selecting variables of interest in raw data.')
    select_variables_of_interest('states', cfg.pModel.dataset_prep.states)
    select_variables_of_interest('states_der', ['der({})'.format(state) for state in cfg.pModel.dataset_prep.states])
    select_variables_of_interest('controls', cfg.pModel.dataset_prep.controls)
    select_variables_of_interest('outputs', cfg.pModel.dataset_prep.outputs)
    select_variables_of_interest('parameters', cfg.pModel.dataset_prep.parameters, remove = cfg.pModel.dataset_prep.parameters_remove)

    # only select certain timeframe
    def idx_timeframe(time: np.ndarray, start_time: float, end_time: float):
        idx = np.where((time >= start_time) & (time <= end_time))[0]
        logging.info(f'... Selecting timeframe from {start_time} to {end_time} in raw data.')
        return idx
    
    def replace_timeseries_if_exist(idx, dataset_name: str, raw_data: h5py.File):
        if dataset_name in raw_data.keys():
            replace_hdf5_dataset(dataset_name, raw_data, data = raw_data[dataset_name][:,:,idx])
            logging.info(f'Selected timeframe from {dataset_name} in raw data.')
        else:
            logging.info(f'No {dataset_name} in raw data. Skipping selection of {dataset_name}.')
    
    idx = idx_timeframe(temp_raw_data['time'][:], cfg.pModel.dataset_prep.start_time, cfg.pModel.dataset_prep.end_time)
    cfg.pModel.dataset_prep.sequence_length = len(idx)
    replace_hdf5_dataset('time', temp_raw_data, data = temp_raw_data['time'][idx])
    replace_timeseries_if_exist(idx, 'states', temp_raw_data)
    replace_timeseries_if_exist(idx, 'states_der', temp_raw_data)
    replace_timeseries_if_exist(idx, 'controls', temp_raw_data)
    replace_timeseries_if_exist(idx, 'outputs', temp_raw_data)

    #############################################################
    # special routines, e.g. chunking together from 0 to N for each time-series
    
    # could be added here

    #############################################################

    # save common test and validation sets to temporary raw data file
    logging.info('opening common test and validation sets')
    temp_raw_data.create_group('common_test')
    temp_raw_data.create_group('common_validation')
    raw_data_n_samples = raw_data_cfg.n_samples if not cfg.pModel.RawData.raw_data_from_external_source else temp_raw_data['states'].shape[0]

    # determine idx in raw data set of test and validation sets
    validation_idx_start_total = int(np.floor(raw_data_n_samples * (1 - cfg.pModel.dataset_prep.validation_fraction - cfg.pModel.dataset_prep.test_fraction)))
    test_idx_start_total = int(np.floor(raw_data_n_samples * (1 - cfg.pModel.dataset_prep.test_fraction)))
    
    # to accomendate cases where validation fraction is 0, just ensure to add one element to validation set
    validation_idx_end_total = test_idx_start_total if test_idx_start_total > validation_idx_start_total else validation_idx_start_total + 1
    
    # save idx to cfg
    cfg.pModel.dataset_prep.validation_idx_start = validation_idx_start_total
    cfg.pModel.dataset_prep.test_idx_start = test_idx_start_total
    logging.info('set validation_idx_start to {}, test_idx_start to {} in cfg.'.format(validation_idx_start_total, test_idx_start_total))

    # save common validation and test sets
    for key in ['states', 'states_der', 'controls', 'outputs', 'parameters']:
        if key in temp_raw_data.keys():
            temp_raw_data.create_dataset('common_validation/' + key, data=temp_raw_data[key][validation_idx_start_total:validation_idx_end_total])
            temp_raw_data.create_dataset('common_test/' + key, data=temp_raw_data[key][test_idx_start_total:])
            logging.info('Saved common test and validation sets for {} to temporary raw data file.'.format(key))
        else:
            logging.info('No {} in raw data. Skipping saving common test and validation sets for {}.'.format(key, key))

    # add generation date to datasets
    creation_date = datetime.now()
    temp_raw_data.attrs['creation_date'] = str(creation_date)

    _reached_max_samples = False
    # sample dataset sizes and save datasets
    for n_samples_dataset in cfg.pModel.dataset_prep.n_samples:
        if _reached_max_samples:
            logging.warning('Reached maximum number of samples in raw data. Skipping further dataset creation.')
            break
        if n_samples_dataset > raw_data_n_samples:
            logging.warning('n_samples_dataset must be smaller than n_samples in raw data. Setting n_samples_dataset={} to n_samples={}'.format(n_samples_dataset, raw_data_n_samples))
            n_samples_dataset = raw_data_n_samples
            _reached_max_samples = True
        path_dataset = filepath_dataset(cfg, n_samples_dataset)
        log_overwriting_file(path_dataset)
        dataset_file = h5py.File(path_dataset, 'w')
        dataset_file.create_dataset('time', data=temp_raw_data['time'])
        for key in ['states', 'states_der', 'controls', 'outputs', 'parameters']:
            if key in temp_raw_data.keys():
                if n_samples_dataset > raw_data_n_samples:
                    raise ValueError('n_samples_dataset must be smaller than n_samples in raw data. Reaching this line should not happen.')
                dataset_file.create_dataset(key + '_names', data=temp_raw_data[key + '_names'])
                # get idx
                train_idx_stop = int((n_samples_dataset/raw_data_n_samples) * validation_idx_start_total)
                if not train_idx_stop > 0:
                    train_idx_stop = 1
                common_validation_idx_stop = int((n_samples_dataset/raw_data_n_samples) * len(temp_raw_data['common_validation/' + key]))
                common_test_idx_stop = int((n_samples_dataset/raw_data_n_samples) * len(temp_raw_data['common_test/' + key]))

                # train, validate, test
                dataset_file.create_dataset('train/' + key, data = temp_raw_data[key][:train_idx_stop])
                dataset_file.create_dataset('validation/' + key, data=temp_raw_data['common_validation/' + key][:common_validation_idx_stop])
                dataset_file.create_dataset('test/' + key, data=temp_raw_data['common_test/' + key][:common_test_idx_stop])
                
                logging.info('Saved {} data with {} samples to {}'.format(key, n_samples_dataset, path_dataset))
                # add common datasets
                dataset_file.create_dataset('common_validation/' + key, data=temp_raw_data['common_validation/' + key])
                dataset_file.create_dataset('common_test/' + key, data=temp_raw_data['common_test/' + key])
                logging.info('Added common test and validation sets for {} to {} dataset.'.format(key, path_dataset))
            else:
                logging.info('No {} in raw data. Skipping saving {}'.format(key, key))
        # save config: create new config object, set raw data config, set n_samples to n_samples_dataset, add preparation info
        _conf = OmegaConf.create(cfg)
        if not cfg.pModel.RawData.raw_data_from_external_source:
            _conf.pModel.RawData = raw_data_cfg
        _conf.pModel.dataset_prep.n_samples = [n_samples_dataset]
        # add preparation info
        _conf.pModel.dataset_prep = cfg.pModel.dataset_prep
        path_dataset_config = filepath_dataset_config(cfg, n_samples_dataset)
        log_overwriting_file(path_dataset_config)
        OmegaConf.save(_conf.pModel, path_dataset_config)
        logging.info('Saved pModel config to {}'.format(path_dataset_config))
        # close dataset
        dataset_file.attrs['creation_date'] = str(creation_date)
        dataset_file.close()
        logging.info('Closed dataset {}'.format(path_dataset))

    # delete temporary file
    temp_raw_data.close()
    os.remove(temp_raw_data_path)
    pass

if __name__ == '__main__':
    main()