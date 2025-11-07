import sys
import os
import hydra
import shutil
from pathlib import Path
from bnode_core.ode import trainer
from bnode_core.config import get_config_store

def ode_training(test_case: str, overrides: list[str] = [],):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    cs = get_config_store()
    # avoid passing pytest's CLI args into the called main()
    orig_argv = sys.argv[:]
    test_dir = Path('./_tests/ode') / ('test_' + test_case)
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True) 
    sys.argv = [orig_argv[0], 
                '--config-dir=resources/config',
                '--config-name=train_test_ode_pytest',
                f"hydra.run.dir={str(test_dir.absolute())}"
                ]
    sys.argv += overrides
    trainer.main()
    sys.argv = orig_argv

def ode_training_params(test_case: str, overrides: list[str] = []):
    overrides += [
        'dataset_path=resources\data\surrogate-test-data\data\datasets\StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest\StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest_dataset.hdf5',
    ]
    ode_training(test_case, overrides=overrides)

def ode_training_initial_states(test_case: str, overrides: list[str] = []):
    overrides += [
        'dataset_path=resources\data\surrogate-test-data\data\datasets\SimpleSeriesResonance_v4_s-R__n-100_pytest\SimpleSeriesResonance_v4_s-R__n-100_pytest_dataset.hdf5',
    ]
    ode_training(test_case, overrides=overrides)


def test_bnode_training():
    ode_training('bnode_training')


def test_use_cuda_false():
    """Test with CUDA disabled (default true)."""
    ode_training('use_cuda_false', overrides=['use_cuda=false'])


# Structural modes tests
def test_controls_to_decoder_false():
    """Test with controls_to_decoder=false (default true)."""
    ode_training('controls_to_decoder_false', overrides=['nn_model.network.controls_to_decoder=false'])


def test_controls_to_state_encoder_true():
    """Test with controls_to_state_encoder=true (default false)."""
    ode_training('controls_to_state_encoder_true', overrides=['nn_model.network.controls_to_state_encoder=true'])


# Linear mode tests
def test_linear_mode_mpc_mode():
    """Test linear_mode=mpc_mode."""
    ode_training('linear_mode_mpc_mode', overrides=['nn_model.network.linear_mode=mpc_mode'])


def test_linear_mode_mpc_mode_for_controls():
    """Test linear_mode=mpc_mode_for_controls."""
    ode_training('linear_mode_mpc_mode_for_controls', overrides=['nn_model.network.linear_mode=mpc_mode_for_controls'])


def test_linear_mode_deep_koopman():
    """Test linear_mode=deep_koopman."""
    ode_training('linear_mode_deep_koopman', overrides=['nn_model.network.linear_mode=deep_koopman'])

def test_linear_mpc_for_controls_controls_to_state_encoder():
    """Test linear_mode=mpc_mode_for_controls with controls_to_state_encoder=true."""
    ode_training('linear_mpc_for_controls_controls_to_state', overrides=[
        'nn_model.network.linear_mode=mpc_mode_for_controls',
        'nn_model.network.controls_to_state_encoder=true'
    ])

# Variance modes tests
def test_variance_constant():
    """Test variance_constant mode."""
    ode_training('variance_constant', overrides=['nn_model.network.lat_ode_type=variance_constant'])


def test_variance_dynamic():
    """Test variance_dynamic mode."""
    ode_training('variance_dynamic', overrides=['nn_model.network.lat_ode_type=variance_dynamic'])


# Reconstruction loss tests
def test_include_reconstruction_loss_state0():
    """Test include_reconstruction_loss_state0=true."""
    ode_training('recon_loss_state0', overrides=[
        'nn_model.training.include_reconstruction_loss_state0_override=true'
    ])


def test_include_reconstruction_loss_outputs0():
    """Test include_reconstruction_loss_outputs0=true."""
    ode_training('recon_loss_outputs0', overrides=[
        'nn_model.training.include_reconstruction_loss_outputs0_override=true'
    ])


# Gradient loss tests
def test_include_states_grad_loss():
    """Test include_states_grad_loss=true."""
    ode_training('states_grad_loss', overrides=[
        'nn_model.training.include_states_grad_loss_override=true'
    ])


def test_include_outputs_grad_loss():
    """Test include_outputs_grad_loss=true."""
    ode_training('outputs_grad_loss', overrides=[
        'nn_model.training.include_outputs_grad_loss_override=true'
    ])


# Multi-shooting condition test
def test_multi_shooting_condition_multiplier():
    """Test multi_shooting_condition_multiplier=10.0."""
    ode_training('multi_shooting_10', overrides=[
        'nn_model.training.multi_shooting_condition_multiplier_override=10.0'
    ])


# Test adaptive step size solver test
def test_solver_dopri5():
    """Test with dopri5 solver."""
    ode_training('solver_dopri5', overrides=[
        'nn_model.training.main_training.1.solver=dopri5',
        'nn_model.training.main_training.1.evaluate_at_control_times=false'
    ])


# Parameter encoder tests
def test_params_training():
    """Test with parameter encoder in training mode."""
    ode_training('params_training')


def test_include_params_encoder_false():
    """Test with include_params_encoder=false (default true)."""
    ode_training_params('include_params_encoder_false', overrides=[
        'nn_model.network.include_params_encoder=false'
        ])


def test_linear_mpc_for_controls_include_param_encoder_false():
    """Test linear_mode=mpc_mode_for_controls with include_param_encoder=false."""
    ode_training_params('linear_mpc_for_controls_no_param_encoder', overrides=[
        'nn_model.network.linear_mode=mpc_mode_for_controls',
        'nn_model.network.include_params_encoder=false'
    ])

# Only state initial states tests
def test_only_initial_states():
    """Test with only initial states as parameters."""
    ode_training_initial_states('only_initial_states')

def test_only_initial_states_linear_mpc():
    """Test with only initial states as parameters and linear_mode=mpc_mode."""
    ode_training_initial_states('only_initial_states_linear_mpc', overrides=[
        'nn_model.network.linear_mode=mpc_mode'
    ])


# Deterministic mode tests (simplified - complex nested list override skipped as requested)

def test_determistic_mode():
    """Test activate_deterministic_mode_after_this_phase=true """
    ode_training('deterministic_mode_after_phase1', overrides=[
        'nn_model=bnode_pytest_det',
    ])

def test_deterministic_mode_from_state0():
    """Test deterministic_mode_from_state0=true (default false)."""
    ode_training('deterministic_mode_from_state0', overrides=[
        'nn_model=bnode_pytest_det',
        'nn_model.training.main_training.1.deterministic_mode_from_state0=true'
    ])


def test_linear_mpc_threshold_populated_dimensions():
    """Test linear_mode=mpc_mode_for_controls with threshold_count_populated_dimensions=0.1."""
    ode_training('linear_mpc_threshold_dims', overrides=[
        'nn_model=bnode_pytest_det',
        'nn_model.network.linear_mode=mpc_mode_for_controls',
        'nn_model.training.main_training.1.threshold_count_populated_dimensions=0.1'
    ])
