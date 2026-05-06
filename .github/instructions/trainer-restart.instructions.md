---
name: bnode-core trainer restart states
description: Restart-state architecture guidance for trainer resume logic, docs, and tests
applyTo: "src/bnode_core/ode/trainer.py,src/bnode_core/ode/trainer_utils/restart_state.py,src/bnode_core/ode/trainer_utils/restart_utils.py,tests/ode/test_restart_state.py,docs/bnode_core/ode/restart_*.md"
---
# bnode-core trainer restart states

Apply these instructions when editing trainer restart/resume state logic or its documentation.

## State model contract

- Keep the responsibilities distinct:
  - `OuterTrainingState` owns orchestration in `train_all_phases()`
  - `TrainingPhaseState` owns mutable counters and flags for one phase
  - `LiveTrainingState` owns the active runtime objects for one phase
  - `OuterTrainingStateCheckpoint` persists only outer resume metadata
  - `InnerTrainingStateCheckpoint` persists only phase runtime/control state
- Current trainer runs use **two restart files**:
  - `training_outer_restart.pt`
  - `training_inner_restart.pt`
- Do not re-introduce a single mixed restart payload as the main workflow contract.
- `TrainingRestartState` is legacy compatibility only. It may remain for older saved checkpoints or unit coverage, but new trainer control flow should not depend on it.

## Construction and restore order

- `LiveTrainingState` uses a **two-phase construction pattern**:
  1. `create_uninitialized()` in `train_all_phases()` after dataloaders and checkpoint paths are known
  2. `bind_runtime_objects()` inside `train_one_phase()` after optimizer, schedulers, scaler, and early-stopping have been created
- Restore must happen only in `bind_runtime_objects()`, never as a side effect of outer-state creation.
- `trainer.py` may inspect outer checkpoint metadata early to choose the resumed job, but it must not restore runtime state until runtime objects exist.

## Checkpoint boundary

- Outer checkpoint owns:
  - Hydra output directory
  - MLflow run metadata
  - resumed job index
  - `next_epoch_anchor`
- Inner checkpoint owns:
  - model / optimizer / scheduler / scaler / early-stopping state dicts
  - `phase_epoch`
  - `first_epoch_is_evaluation`
  - `nan_counter`
  - `grad_norm_last_reduced_counter`
  - `stable_epochs`
  - `flag_out_of_seq_len_increase`
  - `epoch_stop`
  - `deterministic_mode_active`
  - RNG state
  - current/best checkpoint paths for the active phase
- Do **not** persist:
  - `cfg`
  - dataloaders or dataset handles
  - retry batch-size locals
  - copied `train_cfg` state

## Documentation contract

- Keep `docs/bnode_core/ode/restart_training.md` aligned with:
  - the actual two-file resume workflow
  - the current ownership split between outer state, phase state, live phase state, and the two checkpoint dataclasses
- When code comments or docstrings explain restart ownership, point readers to `docs/bnode_core/ode/restart_training.md`.

## Test expectations

- `tests/ode/test_restart_state.py` should cover:
  - roundtrips for `OuterTrainingStateCheckpoint`
  - roundtrips for `InnerTrainingStateCheckpoint`
  - `create_uninitialized()` with runtime object fields still `None`
  - `bind_runtime_objects()` restoring runtime state from an inner checkpoint
  - `create()` as the backward-compatible convenience wrapper
  - legacy `TrainingRestartState` coverage only as compatibility coverage
- `tests/ode/test_bnode.py` resume tests should assert the two-file layout explicitly:
  - interrupted runs leave both restart files behind
  - successful resumed runs remove both restart files
  - MLflow resume metadata comes from the outer checkpoint
- If checkpoint fields, filenames, or restore ordering change, update docs and all relevant restart tests in the same task.
