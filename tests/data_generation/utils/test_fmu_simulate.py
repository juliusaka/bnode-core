from bnode_core.data_generation.utils.fmu_simulate import fmu_simulate, simulate_and_plot
from bnode_core.config import get_config_store
import hydra
from pathlib import Path


def test_fmu_simulate_basic():
    cs = get_config_store()
    dir = str(Path('resources/config').absolute())
    print(f'Using config dir: {dir}')
    with hydra.initialize_config_dir(config_dir=dir, version_base=None, job_name='test_fmu_simulate_basic'):
        cfg = hydra.compose(config_name='data_generation', overrides=[
            'pModel=SHF',
        ])
        res = simulate_and_plot(cfg, return_res=True)
        print(res)
        assert res['success'] is True

if __name__ == '__main__':
    test_fmu_simulate_basic()