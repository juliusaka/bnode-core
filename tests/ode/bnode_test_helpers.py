"""Shared helpers for bnode ODE training tests.

Imported by both ``test_bnode.py`` (feature/variant tests) and
``test_bnode_restart.py`` (restart/resume tests).
"""

import sys
import os
import shutil
from pathlib import Path

from bnode_core.ode import trainer
from bnode_core.config import get_config_store


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
            on disk so that ``load_restart_checkpoint()`` inside
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
