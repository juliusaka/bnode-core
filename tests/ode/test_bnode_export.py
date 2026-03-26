from pathlib import Path
import sys
import os
import shutil
from hydra.core.global_hydra import GlobalHydra

from test_bnode import ode_training
from bnode_core.ode.bnode.bnode_export import main as bnode_export_main

dataset_path = r"resources/data/surrogate-test-data/data/datasets/StratifiedHeatFlowModel_v3_c-RROCS__n-100_pytest/StratifiedHeatFlowModel_v3_c-RROCS__n-100_pytest_dataset.hdf5"
parameter_dataset_path = r"resources/data/surrogate-test-data/data/datasets/StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest/StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest_dataset.hdf5"

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
                    dataset_path: str = "") -> Path:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    training_overrides += ['nn_model.training.max_epochs_override=10']
    training_overrides += [f'dataset_path={str(Path(dataset_path).absolute())}']
    test_dir = ode_training(test_name, overrides=training_overrides)
    test_dir = Path('./tests/_results/ode') / ('test_' + test_name)
    # reset hydra
    GlobalHydra.instance().clear()
    # export model
    export_dir = test_dir / 'test_export_onnx'
    orig_argv = sys.argv[:]
    sys.argv = [orig_argv[0], 
                '--config-dir=resources/config',
                '--config-name=onnx_export_pytest',
                'model_directory=' + str(test_dir.absolute()),
                'output_dir=' + str(export_dir.absolute()),
                f"hydra.run.dir={str(test_dir.absolute() / 'test_export_hydra')}",
                'dataset_path=' + dataset_path,
                ]
    sys.argv += export_overrides
    bnode_export_main()
    sys.argv = orig_argv
    return export_dir

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


def test_bnode_export_deterministic_mode():
    """Test BNODE export with deterministic mode activated after phase."""
    ode_export_test('bnode_export_det_mode',
                    training_overrides=[
                        'nn_model=bnode_pytest_det',
                    ],
                    dataset_path=dataset_path)


def test_bnode_export_deterministic_mode_from_state0():
    """Test BNODE export with deterministic_mode_from_state0=true."""
    ode_export_test('bnode_export_det_from_state0',
                    training_overrides=[
                        'nn_model=bnode_pytest_det',
                        'nn_model.training.main_training.1.deterministic_mode_from_state0=true',
                    ],
                    dataset_path=dataset_path)


def test_bnode_export_deterministic_linear_mpc():
    """Test BNODE export with deterministic mode and linear_mode=mpc_mode_for_controls."""
    ode_export_test('bnode_export_det_linear_mpc',
                    training_overrides=[
                        'nn_model=bnode_pytest_det',
                        'nn_model.network.linear_mode=mpc_mode_for_controls',
                        'nn_model.training.main_training.1.threshold_count_populated_dimensions=0.1',
                    ],
                    dataset_path=dataset_path)


def _assert_siso_export(export_dir: Path) -> None:
    import json
    assert (export_dir / 'encoder_states_siso.onnx').exists()
    assert (export_dir / 'latent_ode_siso.onnx').exists()
    assert (export_dir / 'decoder_siso.onnx').exists()
    dims = json.loads((export_dir / 'siso_dimensions.json').read_text())
    assert 'encoder_states' in dims
    assert 'latent_ode' in dims
    assert 'decoder' in dims


def test_bnode_export_siso():
    """Test BNODE SISO export: *_siso.onnx files and siso_dimensions.json must be written."""
    export_dir = ode_export_test(
        'bnode_export_siso',
        dataset_path=dataset_path,
        export_overrides=['siso=true'],
    )
    _assert_siso_export(export_dir)


def test_bnode_export_siso_deterministic():
    """Test BNODE SISO export with a deterministic training phase (encoder outputs only mu)."""
    export_dir = ode_export_test(
        'bnode_export_siso_det',
        training_overrides=['nn_model=bnode_pytest_det'],
        dataset_path=dataset_path,
        export_overrides=['siso=true'],
    )
    _assert_siso_export(export_dir)
