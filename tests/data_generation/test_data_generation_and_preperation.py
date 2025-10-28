import sys
import hydra
from bnode_core import filepaths 
from pathlib import Path
from bnode_core.config import get_config_store
import shutil
from hydra.core.global_hydra import GlobalHydra

def test_01_raw_data_generation():
    cs = get_config_store()
    # avoid passing pytest's CLI args into the called main()
    orig_argv = sys.argv[:]
    sys.argv = [orig_argv[0], '--config-dir=resources/config',
                '--config-name=data_generation',
                'pModel.RawData.n_samples=12',
                'pModel.RawData.versionName=test',
                ]
    hydra.initialize_config_dir(config_dir=str(Path('resources/config').absolute()), version_base=None)
    cfg = hydra.compose(config_name='data_generation', overrides=sys.argv[3:])
    if filepaths.dir_raw_data(cfg).exists():
        shutil.rmtree(filepaths.dir_raw_data(cfg))
    # correctly clear Hydra global state
    GlobalHydra.instance().clear()
    from bnode_core.data_generation.raw_data_generation import  main
    main()
    assert filepaths.dir_raw_data(cfg).exists()
    sys.argv = orig_argv

def test_02_data_preperation():
    from bnode_core.data_generation.data_preperation import main
    import sys
    # avoid passing pytest's CLI args into the called main()
    orig_argv = sys.argv[:]
    sys.argv = [orig_argv[0], '--config-dir=resources/config',
                    '--config-name=data_generation',
                    'pModel.RawData.versionName=test',
                    ]
    main()
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(Path('resources/config').absolute()), version_base=None)
    cfg = hydra.compose(config_name='data_generation', overrides=sys.argv[3:])
    assert filepaths.filepath_dataset(cfg, 12).exists()
    assert filepaths.filepath_dataset_config(cfg, 12).exists()
    sys.argv = orig_argv