from pathlib import Path
import sys
import os
import shutil
from hydra.core.global_hydra import GlobalHydra

from test_bnode import ode_training
from bnode_core.ode.bnode.bnode_export import main as bnode_export_main

dataset_path = r"resources\data\surrogate-test-data\data\datasets\StratifiedHeatFlowModel_v3_c-RROCS__n-100_pytest\StratifiedHeatFlowModel_v3_c-RROCS__n-100_pytest_dataset.hdf5"
parameter_dataset_path = r"resources\data\surrogate-test-data\data\datasets\StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest\StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest_dataset.hdf5"

# Perform tests for bnode expor
# train a simple model using ode_training from the config class, with max_epochs_override=10
# reset the hydra instance after training
# test onnx export using the trained model directory


# Perform tests for
# simple ode training
#       + parameter dataset

# ode training linear_mode = mpc_mod
#   + controls to initial states encoder
# ode training with linear_mode = mpc_mode_for_controls and parameter dataset
#   + no_parameter_encoder
#   

def ode_export_test(test_name: str, training_overrides: list[str] = [], export_overrides: list[str] = [],
                    dataset_path: str = ""):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    training_overrides += ['nn_model.training.max_epochs_override=10']
    training_overrides += [f'dataset_path={str(Path(dataset_path).absolute())}']
    test_dir = ode_training(test_name, overrides=training_overrides)
    test_dir = Path('./tests/_results/ode') / ('test_' + test_name)
    # reset hydra
    GlobalHydra.instance().clear()
    # export model
    orig_argv = sys.argv[:]
    sys.argv = [orig_argv[0], 
                '--config-dir=resources/config',
                '--config-name=onnx_export_pytest',
                'model_directory=' + str(test_dir.absolute()),
                'output_dir=' + str(test_dir.absolute() / 'test_export' / 'onnx'),
                f"hydra.run.dir={str(test_dir.absolute() / 'test_export')}",
                'dataset_path=' + dataset_path,
                'config_path=' + str(test_dir.absolute() / '.hydra' / 'config_validated.yaml'),
                ]
    sys.argv += export_overrides
    bnode_export_main()
    sys.argv = orig_argv

def test_bnode_export():
    """Test basic BNODE export with controls."""
    ode_export_test('bnode_export_test', dataset_path=dataset_path)


def test_bnode_export_with_parameters():
    """Test BNODE export with parameter dataset."""
    ode_export_test('bnode_export_params', 
                    dataset_path=parameter_dataset_path)


def test_bnode_export_linear_mpc():
    """Test BNODE export with linear_mode=mpc_mode."""
    ode_export_test('bnode_export_linear_mpc',
                    training_overrides=['nn_model.network.linear_mode=mpc_mode'],
                    dataset_path=dataset_path)


def test_bnode_export_linear_mpc_for_controls():
    """Test BNODE export with linear_mode=mpc_mode_for_controls and parameter dataset."""
    ode_export_test('bnode_export_linear_mpc_controls',
                    training_overrides=[
                        'nn_model.network.linear_mode=mpc_mode_for_controls'],
                    dataset_path=parameter_dataset_path)

def test_bnode_export_controls_to_state_encoder():
    """Test BNODE export with controls to initial states encoder."""
    ode_export_test('bnode_export_controls_to_state',
                    training_overrides=['nn_model.network.linear_mode=mpc_mode_for_controls',
                                        'nn_model.network.controls_to_state_encoder=true'],
                    dataset_path=dataset_path)


def test_bnode_export_no_parameter_encoder():
    """Test BNODE export with linear_mode=mpc_mode_for_controls and no parameter encoder."""
    ode_export_test('bnode_export_no_param_encoder',
                    training_overrides=[
                        'nn_model.network.linear_mode=mpc_mode_for_controls',
                        'nn_model.network.include_params_encoder=false'
                    ],
                    dataset_path=parameter_dataset_path)
