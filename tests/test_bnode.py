import sys
import hydra
import shutil
from pathlib import Path
from bnode_core.ode import trainer
from bnode_core.config import get_config_store

def ode_training(test_case: str, overrides: list[str] = [],):
    cs = get_config_store()
    # avoid passing pytest's CLI args into the called main()
    orig_argv = sys.argv[:]
    test_dir = Path('./_tests/ode') / ('test_' + test_case)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    sys.argv = [orig_argv[0], 
                '--config-dir=resources/config',
                '--config-name=train_test_ode_pytest',
                f"hydra.run.dir={str(test_dir.absolute())}"
                ]
    sys.argv += overrides
    trainer.main()
    sys.argv = orig_argv

def test_bnode_training():
    ode_training('bnode_training')

"""
Necessary tests:
    - use_cuda false (default true)
    # structural modes:
        - controls to decoder false (default true)
        - controls to state encoder true (default false)
    - linear mode:
        - mpc_mode
        - mpc_mode_for_controls
        - deep_koopman
    - variance modes:
        - variance constant
        - variance dynamic
    - with parameters
        - include_params_encoder false (default true)
        - linear mode: mpc_for_controls
            - include_param_encoder false (default true)
            - controls_to_state_encoder true (default false)
    - nn_model.training.main_training.1.activate_deterministic_mode_after_this_phase true (default false)
        # need to append override +nn_model.training.main_training.2=[batch_size: 400
                                                                    solver: rk4
                                                                    lr_start: 1e-3
                                                                    max_epochs: 100
                                                                    early_stopping_patience: 10
                                                                    seq_len_train: null
                                                                    ]
        - deterministic_mode_from_state0 true (default false)
        - linear mode: mpc_for_controls 
         threshold_count_populated_dimensions: 0.1
    - include_reconstruction_loss_state0: true
    - include_reconstruction_loss_outputs0: true
    - include_reconstruction_loss_state_der: true
    - include_states_grad_loss: true
    - include_outputs_grad_loss: true
    - multi_shooting_condition_multiplier: 10.0
    - nn_model.training.main_training.1.solver=dopri5
"""