import sys
import os
import random
import shutil
from pathlib import Path
import mlflow
import numpy as np
import pytest
import torch

from bnode_core.ode import trainer
from bnode_core.config import get_config_store
from bnode_core.ode.trainer_utils.restart_state import (
    CheckpointRequestedExit,
    INNER_RESTART_STATE_FILENAME,
    OUTER_RESTART_STATE_FILENAME,
    load_train_all_phases_state,
    load_train_one_phase_state,
)


def ode_training(
    test_case: str,
    overrides: list[str] = [],
    clear_output_before_start: bool = True,
) -> Path:
    """Run trainer.main() for *test_case* and return the Hydra output directory.

    Args:
        test_case: Sub-directory name under ``tests/_results/ode/`` (prefixed
            with ``test_``).  The same name is reused across the interrupted and
            resumed legs of a restart test so both runs share the same directory
            and therefore the same restart-checkpoint files.
        overrides: Hydra CLI overrides forwarded to ``trainer.main()``.
        clear_output_before_start: When ``True`` (default) the output directory
            is deleted before the run so each test starts from a clean state.
            Set to ``False`` for the *resumed* leg of a restart test: the
            restart-checkpoint files written by the interrupted run must remain
            on disk so that ``load_restart_state_pair()`` inside
            ``train_all_phases`` can detect them and restore the full training
            state (job index, epoch counter, LR scheduler, GradScaler, RNG
            state) automatically at the start of the next ``trainer.main()``
            call.
    """
    os.environ['HYDRA_FULL_ERROR'] = '1'
    get_config_store()
    # avoid passing pytest's CLI args into the called main()
    orig_argv = sys.argv[:]
    test_dir = Path('./tests/_results/ode') / ('test_' + test_case)
    if clear_output_before_start and test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    sys.argv = [orig_argv[0], 
                '--config-dir=resources/config',
                '--config-name=train_test_ode_pytest',
                f"hydra.run.dir={str(test_dir.absolute())}"
                ]
    sys.argv += overrides
    trainer.main()
    sys.argv = orig_argv
    return test_dir

def ode_training_params(test_case: str, overrides: list[str] = []):
    overrides += [
        'dataset_path=resources/data/surrogate-test-data/data/datasets/StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest/StratifiedHeatFlowModel_v3_p-R_c-RROCS__n-100_pytest_dataset.hdf5',
    ]
    ode_training(test_case, overrides=overrides)

def ode_training_initial_states(test_case: str, overrides: list[str] = []):
    overrides += [
        'dataset_path=resources/data/surrogate-test-data/data/datasets/SimpleSeriesResonance_v4_s-R__n-100_pytest/SimpleSeriesResonance_v4_s-R__n-100_pytest_dataset.hdf5',
    ]
    ode_training(test_case, overrides=overrides)


def _resume_training_overrides(
    *,
    nn_model: str,
    max_epochs: tuple[int, ...],
    batches_per_epoch: int = 1,
    scheduler_phase: int | None = None,
) -> list[str]:
    overrides = [
        f'nn_model={nn_model}',
        'nn_model.training.test=false',
        'use_cuda=false',
        'n_workers_train_loader=0',
        'n_workers_other_loaders=0',
        'prefetch_factor=null',
    ]
    for phase_idx, phase_max_epochs in enumerate(max_epochs):
        overrides.extend(
            [
                f'nn_model.training.main_training.{phase_idx}.max_epochs={phase_max_epochs}',
                f'nn_model.training.main_training.{phase_idx}.batches_per_epoch={batches_per_epoch}',
            ]
        )
    if scheduler_phase is not None:
        overrides.extend(
            [
                f'nn_model.training.main_training.{scheduler_phase}.use_lr_scheduler=true',
                f'nn_model.training.main_training.{scheduler_phase}.lr_scheduler_type=cosine',
                f'nn_model.training.main_training.{scheduler_phase}.cosine_T_max=2',
                f'nn_model.training.main_training.{scheduler_phase}.cosine_eta_min=1e-6',
            ]
        )
    return overrides


def _resume_mlflow_tracking_uri(scope: str) -> str:
    return f"file://{(Path('./tests/_results/ode/mlruns') / scope).absolute()}"


def _resume_mlflow_overrides(scope: str) -> list[str]:
    return [
        f'mlflow_tracking_uri={_resume_mlflow_tracking_uri(scope)}',
        f'mlflow_experiment_name={scope}',
    ]


def _fixed_seq_len_phase_overrides(phase_idx: int, seq_len_train: int = 3) -> list[str]:
    return [
        f'nn_model.training.main_training.{phase_idx}.seq_len_train={seq_len_train}',
        f'nn_model.training.main_training.{phase_idx}.seq_len_increase_in_batches=0',
    ]


def _get_mlflow_run(tracking_uri: str, run_id: str):
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)


def _load_model_state(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def _load_runtime_state(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def _set_training_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_model_states_equal(
    path_a: Path,
    path_b: Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    state_a = _load_model_state(path_a)
    state_b = _load_model_state(path_b)
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert_close_kwargs = {}
        if rtol is not None:
            assert_close_kwargs['rtol'] = rtol
        if atol is not None:
            assert_close_kwargs['atol'] = atol
        torch.testing.assert_close(state_a[key], state_b[key], **assert_close_kwargs)


def _assert_restart_state(
    outer_restart_state,
    inner_restart_state,
    *,
    expected_job_idx: int,
    deterministic_mode_active: bool | None = None,
) -> None:
    assert outer_restart_state.job_idx == expected_job_idx
    assert outer_restart_state.next_epoch_anchor >= inner_restart_state.phase_epoch
    if deterministic_mode_active is not None:
        assert inner_restart_state.deterministic_mode_active is deterministic_mode_active


def _assert_resumed_mlflow_run(
    output_dir: Path,
    *,
    mlflow_scope: str,
    outer_restart_state,
    expected_final_job_idx: int,
):
    resumed_run = _get_mlflow_run(
        _resume_mlflow_tracking_uri(mlflow_scope),
        outer_restart_state.mlflow_run_id,
    )
    assert resumed_run.data.params['mlflow_run_id'] == outer_restart_state.mlflow_run_id
    assert f'job_{expected_final_job_idx}_final_epoch' in resumed_run.data.metrics
    return resumed_run


def _interrupt_after_n_epoch_saves(monkeypatch: pytest.MonkeyPatch, n_saves: int) -> None:
    """Interrupt training after *n_saves* epoch checkpoints have been written.

    Mechanism:
    1. ``monkeypatch.setattr`` replaces ``RestartCheckpointStore.save_epoch_checkpoint``
       with a counting wrapper, scoped to the enclosing ``monkeypatch.context()`` block
       so the real method is restored automatically when that block exits.
    2. The wrapper calls the *real* ``save_epoch_checkpoint`` first, so the restart-
       checkpoint files (outer state, inner state, LR schedulers, GradScaler) are fully
       written to disk before the exception is raised.
    3. On the Nth call the wrapper raises ``CheckpointRequestedExit``, a custom
       ``RuntimeError`` subclass.
    4. ``train_one_phase`` catches ``CheckpointRequestedExit`` explicitly, tags the
       MLflow run as ``"ended by checkpoint request"``, and returns cleanly — no
       exception escapes to the test.

    The checkpoint files being on disk before the exception is the key property that
    the resumed leg of the test relies on.
    """
    original_save = trainer.RestartCheckpointStore.save_epoch_checkpoint
    call_count = [0]

    def _patched_save(self, *args, **kwargs):
        # Real save runs first — files are on disk before we raise.
        result = original_save(self, *args, **kwargs)
        call_count[0] += 1
        if call_count[0] >= n_saves:
            raise CheckpointRequestedExit(
                f"Test-injected interrupt after {call_count[0]} epoch saves."
            )
        return result

    monkeypatch.setattr(
        trainer.RestartCheckpointStore,
        'save_epoch_checkpoint',
        _patched_save,
    )


@pytest.fixture(scope='module')
def resume_main_reference_dir():
    _set_training_seeds()
    return ode_training(
        'resume_main_reference',
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest',
            max_epochs=(3, 3),
            scheduler_phase=1,
        )
        + _fixed_seq_len_phase_overrides(1)
        + _resume_mlflow_overrides('resume_main_reference'),
    )


@pytest.fixture(scope='module')
def resume_deterministic_reference_dir():
    _set_training_seeds()
    return ode_training(
        'resume_deterministic_reference',
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest_det',
            max_epochs=(3, 3, 3),
        )
        + _fixed_seq_len_phase_overrides(1, seq_len_train=3)
        + _fixed_seq_len_phase_overrides(2, seq_len_train=3)
        + _resume_mlflow_overrides('resume_deterministic_reference'),
    )


def test_bnode_training():
    ode_training('bnode_training',
                #  overrides=[
                #             'nn_model.training.max_epochs_override=100',
                #             ]
                            )


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

def test_deterministic_mode():
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


def test_resume_from_same_hydra_output_dir_during_main_training(resume_main_reference_dir, monkeypatch):
    interrupted_case = 'resume_main_same_dir_interrupted'
    mlflow_scope = 'resume_main_same_dir'
    mlflow_overrides = _resume_mlflow_overrides(mlflow_scope)
    with monkeypatch.context() as ctx:
        _set_training_seeds()
        # Interrupt after two epoch-end checkpoint saves in the scheduler phase.
        _interrupt_after_n_epoch_saves(ctx, n_saves=4)
        interrupted_dir = ode_training(
            interrupted_case,
            overrides=_resume_training_overrides(
                nn_model='bnode_pytest',
                max_epochs=(3, 3),
                scheduler_phase=1,
            )
            + _fixed_seq_len_phase_overrides(1)
            + mlflow_overrides,
        )

    outer_restart_path = interrupted_dir / OUTER_RESTART_STATE_FILENAME
    inner_restart_path = interrupted_dir / INNER_RESTART_STATE_FILENAME
    scheduler_restart_path = interrupted_dir / 'lr_schedulers.pt'
    scaler_restart_path = interrupted_dir / 'grad_scaler.pt'
    outer_restart_state = load_train_all_phases_state(outer_restart_path)
    inner_restart_state = load_train_one_phase_state(inner_restart_path)
    _assert_restart_state(
        outer_restart_state,
        inner_restart_state,
        expected_job_idx=2,
        deterministic_mode_active=False,
    )
    assert scheduler_restart_path.exists()
    assert scaler_restart_path.exists()
    scheduler_state = _load_runtime_state(scheduler_restart_path)
    assert 'cosine' in scheduler_state
    assert scheduler_state['cosine']['last_epoch'] > 0
    assert isinstance(_load_runtime_state(scaler_restart_path), dict)

    # Resume: keep the output directory so the restart-checkpoint files written
    # by the interrupted run remain on disk.  load_restart_state_pair() inside
    # train_all_phases detects them automatically at startup and restores the
    # full training state without any explicit reload call in the test.
    _set_training_seeds()
    resumed_dir = ode_training(
        interrupted_case,
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest',
            max_epochs=(3, 3),
            scheduler_phase=1,
        )
        + _fixed_seq_len_phase_overrides(1)
        + mlflow_overrides,
        clear_output_before_start=False,
    )
    assert resumed_dir == interrupted_dir
    # Restart-checkpoint files are removed after a successful run.
    assert not outer_restart_path.exists()
    assert not inner_restart_path.exists()
    assert not scheduler_restart_path.exists()
    assert not scaler_restart_path.exists()
    _assert_resumed_mlflow_run(
        resumed_dir,
        mlflow_scope=mlflow_scope,
        outer_restart_state=outer_restart_state,
        expected_final_job_idx=2,
    )
    _assert_model_states_equal(
        resumed_dir / 'model.pt',
        resume_main_reference_dir / 'model.pt',
    )


def test_resume_from_same_hydra_output_dir_across_deterministic_activation(
    monkeypatch,
):
    interrupted_case = 'resume_deterministic_same_dir_interrupted'
    mlflow_scope = 'resume_deterministic_same_dir'
    mlflow_overrides = _resume_mlflow_overrides(mlflow_scope)
    with monkeypatch.context() as ctx:
        _set_training_seeds()
        # Interrupt after two epoch-end checkpoint saves in the first train phase, before
        # deterministic mode is activated in later phases.
        # We cannot interrupt after det-mode is active because the model
        # state-dict then contains zero-sized masked tensors that cannot be
        # loaded into a freshly-initialised model.
        _interrupt_after_n_epoch_saves(ctx, n_saves=2)
        interrupted_dir = ode_training(
            interrupted_case,
            overrides=_resume_training_overrides(
                nn_model='bnode_pytest_det',
                max_epochs=(3, 3, 3),
            )
            + _fixed_seq_len_phase_overrides(1, seq_len_train=3)
            + _fixed_seq_len_phase_overrides(2, seq_len_train=3)
            + mlflow_overrides,
        )

    outer_restart_path = interrupted_dir / OUTER_RESTART_STATE_FILENAME
    inner_restart_path = interrupted_dir / INNER_RESTART_STATE_FILENAME
    outer_restart_state = load_train_all_phases_state(outer_restart_path)
    inner_restart_state = load_train_one_phase_state(inner_restart_path)
    # Interrupted during phase 2 training before det-mode was applied.
    _assert_restart_state(
        outer_restart_state,
        inner_restart_state,
        expected_job_idx=1,
        deterministic_mode_active=False,
    )

    _set_training_seeds()
    resumed_dir = ode_training(
        interrupted_case,
        overrides=_resume_training_overrides(
            nn_model='bnode_pytest_det',
            max_epochs=(3, 3, 3),
        )
        + _fixed_seq_len_phase_overrides(1, seq_len_train=3)
        + _fixed_seq_len_phase_overrides(2, seq_len_train=3)
        + mlflow_overrides,
        clear_output_before_start=False,
    )

    assert resumed_dir == interrupted_dir
    # Restart state is removed after successful completion.
    assert not outer_restart_path.exists()
    assert not inner_restart_path.exists()
    # MLflow run ID is preserved and phase-3 metrics were logged.
    _assert_resumed_mlflow_run(
        resumed_dir,
        mlflow_scope=mlflow_scope,
        outer_restart_state=outer_restart_state,
        expected_final_job_idx=3,
    )
    # Verify deterministic mode was applied: the final model.pt has the
    # masked tensor shapes (zero-sized latent dims) expected after det-mode.
    final_model_state = _load_model_state(resumed_dir / 'model.pt')
    assert final_model_state['latent_ode_func.net.0.weight'].shape[1] == 0, (
        "Expected det-mode masked weights in final model.pt"
    )
