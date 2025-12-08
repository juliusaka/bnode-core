import sys
import os 

from bnode_core.config import data_gen_config, convert_cfg_to_dataclass, get_config_store
import bnode_core.filepaths as filepaths
from pathlib import Path
import hydra
from pathlib import Path
from fmpy import read_model_description, extract
from typing import List

def get_state_names(fmu_path: Path):
    model_description = read_model_description(Path(fmu_path))
    state_names = []
    for derivative in model_description.derivatives:
        state_names.append(derivative.variable.derivative.name)
    return state_names

def get_input_names(fmu_path: Path):
    model_description = read_model_description(Path(fmu_path))
    input_names = []
    for mv in model_description.modelVariables:
        if mv.causality == 'input':
            input_names.append(mv.name)
    return input_names

def get_output_names(fmu_path: Path):
    model_description = read_model_description(Path(fmu_path))
    output_names = []
    for mv in model_description.modelVariables:
        if mv.causality == 'output':
            output_names.append(mv.name)
    return output_names

def get_variable_names(fmu_path: Path):
    state_names = get_state_names(fmu_path)
    input_names = get_input_names(fmu_path)
    model_description = read_model_description(Path(fmu_path))
    variable_names = []
    for variable in model_description.modelVariables:
        if variable.variability == 'continuous':
            if variable.name not in state_names and variable.name not in input_names:
                variable_names.append(variable.name)
    return variable_names

def get_parameter_names(fmu_path: Path):
    model_description = read_model_description(Path(fmu_path))
    parameter_names = []
    for parameter in model_description.modelVariables:
        if parameter.causality == 'parameter':
            parameter_names.append(parameter.name)
    return parameter_names


def print_variable_names(fmu_path: Path):
    # print to file and console
    path = 'fmu_variable_names.txt'
    original_stdout = sys.stdout
    try:
        with open(path, 'w') as f:
            sys.stdout = f
            print('fmu_path: {}'.format(fmu_path))
            print('\nThe states are:')
            for name in get_state_names(fmu_path):
                print(name + ':')
            print('\nThe inputs are:')
            for name in get_input_names(fmu_path):
                print(name + ':')
            print('\nThe outputs are:')
            for name in get_output_names(fmu_path):
                print(name + ':')
            print('\nThe variables are:')
            for name in get_variable_names(fmu_path):
                print('- ' + name)
            print('\nThe parameters are:')
            for name in get_parameter_names(fmu_path):
                print(name + ':')
    finally:
        sys.stdout = original_stdout

    print('fmu_path: {}'.format(Path(fmu_path)))
    print('\nThe states are:')
    for name in get_state_names(fmu_path):
        print(name + ':')
    print('\nThe inputs are:')
    for name in get_input_names(fmu_path):
        print(name + ':')
    print('\nThe outputs are:')
    for name in get_output_names(fmu_path):
        print('-' + name)
    print('\nThe variables are:')
    for name in get_variable_names(fmu_path):
        print('- ' + name)
    print('\nThe parameters are:')
    for name in get_parameter_names(fmu_path):
        print(name + ':')

def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Usage: python print_fmu_variable_names.py [--config-path | --fmu_path=path_to_fmu]')
        print('If both is not provided, the script will try to auto-recognize the config file location.')
        print('Hint: Filepaths need to be provided using forward slashes (/) on Windows.')
        sys.exit(0)
    
    cfg_path = None
    fmu_path = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--cfg_path') or arg.startswith('--cfg-path'):
            cfg_path = sys.argv.pop(i).split('=')[1]
        if arg.startswith('--fmu_path') or arg.startswith('--fmu-path'):
            fmu_path = sys.argv.pop(i).split('=')[1]
    if cfg_path is not None and fmu_path is not None:
        raise ValueError('Please provide either --cfg_path or --fmu_path, not both.')

    if fmu_path is not None:
        fmu_path = Path(fmu_path)
    else:
        cs = get_config_store()
        config_dir = filepaths.config_dir_auto_recognize()
        config_name = 'data_generation' 
        # programmatically load the config and get the fmu path
        with hydra.initialize_config_dir(config_dir=str(Path(config_dir).absolute()), version_base=None):
            print("setting overrides:", sys.argv[1:])
            cfg = hydra.compose(config_name=config_name, overrides=sys.argv[1:])
        fmu_path = Path(cfg.pModel.RawData.fmuPath)

    print_variable_names(fmu_path)

if __name__ == '__main__':
    main()