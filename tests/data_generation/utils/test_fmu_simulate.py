from bnode_core.data_generation.utils.fmu_simulate import fmu_simulate, simulate_and_plot
from bnode_core.config import get_config_store
import hydra
from pathlib import Path


def test_fmu_simulate_basic(overrides=['pModel=SHF']):
    cs = get_config_store()
    dir = str(Path('resources/config').absolute())
    print(f'Using config dir: {dir}')
    with hydra.initialize_config_dir(config_dir=dir, version_base=None, job_name='test_fmu_simulate_basic'):
        cfg = hydra.compose(config_name='data_generation', overrides=overrides)
        res = simulate_and_plot(cfg, return_res=True)
        print(res)
        assert res['success'] is True

def test_only_states():
    overrides = ['pModel=SHF', 
                 'pModel.RawData.initial_states_include=true', 
                 'pModel.RawData.parameters_include=false',
                 'pModel.RawData.controls_include=false',
                 ]
    test_fmu_simulate_basic(overrides=overrides)

def test_states_and_parameters():
    overrides = ['pModel=SHF', 
                 'pModel.RawData.initial_states_include=true', 
                 'pModel.RawData.parameters_include=true',
                 'pModel.RawData.controls_include=false',
                 ]
    test_fmu_simulate_basic(overrides=overrides)

def test_only_controls():
    overrides = ['pModel=SHF', 
                 'pModel.RawData.initial_states_include=false', 
                 'pModel.RawData.parameters_include=false',
                 'pModel.RawData.controls_include=true',
                 ]
    test_fmu_simulate_basic(overrides=overrides)

def test_only_controls_RS():
    overrides = ['pModel=SHF', 
                 'pModel.RawData.initial_states_include=false', 
                 'pModel.RawData.parameters_include=false',
                 'pModel.RawData.controls_include=true',
                 'pModel.RawData.controls_sampling_strategy=RS',
                 ]
    test_fmu_simulate_basic(overrides=overrides)

def test_all_inputs():
    overrides = ['pModel=SHF', 
                 'pModel.RawData.initial_states_include=true', 
                 'pModel.RawData.parameters_include=true',
                 'pModel.RawData.controls_include=true',
                 ]
    test_fmu_simulate_basic(overrides=overrides)