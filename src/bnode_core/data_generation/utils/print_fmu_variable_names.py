import sys
import os 

from config import data_gen_config, cs, convert_cfg_to_dataclass
from pathlib import Path
import hydra
from pathlib import Path
from fmpy import read_model_description, extract

def get_state_names(cfg: data_gen_config):
    model_description = read_model_description(Path(cfg.pModel.RawData.fmuPath))
    state_names = []
    for derivative in model_description.derivatives:
        state_names.append(derivative.variable.derivative.name)
    return state_names

def get_input_names(cfg: data_gen_config):
    model_description = read_model_description(Path(cfg.pModel.RawData.fmuPath))
    input_names = []
    for input in model_description.modelVariables:
        if input.causality == 'input':
            input_names.append(input.name)
    return input_names

def get_output_names(cfg: data_gen_config):
    model_description = read_model_description(Path(cfg.pModel.RawData.fmuPath))
    output_names = []
    for output in model_description.modelVariables:
        if output.causality == 'output':
            output_names.append(output.name)
    return output_names

def get_variable_names(cfg: data_gen_config):
    state_names = get_state_names(cfg)
    input_names = get_input_names(cfg)
    model_description = read_model_description(Path(cfg.pModel.RawData.fmuPath))
    variable_names = []
    for variable in model_description.modelVariables:
        if variable.variability == 'continuous':
            if not variable.name in state_names and not variable.name in input_names:
                variable_names.append(variable.name)
    return variable_names

def get_parameter_names(cfg: data_gen_config):
    model_description = read_model_description(Path(cfg.pModel.RawData.fmuPath))
    parameter_names = []
    for parameter in model_description.modelVariables:
        if parameter.causality == 'parameter':
            parameter_names.append(parameter.name)
    return parameter_names
    

# @hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen', version_base=None)
def main(cfg: data_gen_config) -> None:
    cfg = convert_cfg_to_dataclass(cfg)
    # print to file and console
    path = 'fmu_variable_names.txt'
    with open(path, 'w') as f:
        sys.stdout = f
        print('cfg.pModel.RawData.fmuPath: {}'.format(Path(cfg.pModel.RawData.fmuPath)))
        print('\nThe states are:')
        for name in get_state_names(cfg):
            print(name  + ':')
        print('\nThe inputs are:')
        for name in get_input_names(cfg):
            print(name + ':')
        print('\nThe outputs are:')
        for name in get_output_names(cfg):
            print(name + ':')
        print('\nThe variables are:')
        for name in get_variable_names(cfg):
            print('- ' + name)
        print('\nThe parameters are:')
        for name in get_parameter_names(cfg):
            print(name + ':')
    sys.stdout = sys.__stdout__
    print('cfg.pModel.RawData.fmuPath: {}'.format(Path(cfg.pModel.RawData.fmuPath)))
    print('\nThe states are:')
    for name in get_state_names(cfg):
        print(name  + ':')
    print('\nThe inputs are:')
    for name in get_input_names(cfg):
        print(name + ':')
    print('\nThe outputs are:')
    for name in get_output_names(cfg):
        print(name + ':')
    print('\nThe variables are:')
    for name in get_variable_names(cfg):
        print('- ' + name)
    print('\nThe parameters are:')
    for name in get_parameter_names(cfg):
        print(name + ':')

if __name__ == '__main__':
    
    main()