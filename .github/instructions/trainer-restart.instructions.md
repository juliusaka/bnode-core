---
name: bnode-core trainer restart states
description: Restart-state architecture guidance for trainer resume logic, docs, and tests
applyTo: "src/bnode_core/ode/trainer.py,src/bnode_core/ode/trainer_utils/restart_state.py,src/bnode_core/ode/trainer_utils/restart_utils.py,tests/ode/test_restart_state.py,docs/bnode_core/ode/restart_*.md"
---
# bnode-core trainer restart states

Apply these instructions when editing trainer restart/resume state logic or its documentation.

## State model contract

- Keep exactly two persisted restart-state classes:
  - `TrainAllPhasesState`
  - `TrainOnePhaseState`
- Current trainer runs use **two restart files**:
  - `training_outer_restart.pt`
  - `training_inner_restart.pt`
- Do not re-introduce wrapper-state layers such as `OuterTrainingState`, `TrainingPhaseState`, or `LiveTrainingState`.
- Do not add legacy compatibility readers, old single-file restart schemas, or obsolete restart filenames unless the user explicitly asks for compatibility.

## Ownership boundary

- `TrainAllPhasesState` owns only the outer resume anchor:
  - `job_idx`
  - `next_epoch_anchor`
  - `mlflow_run_id`
- `TrainOnePhaseState` owns only the persisted inner runtime/control state:
  - `phase_epoch`
  - optimizer state
  - scheduler states
  - scaler state
  - early-stopping state
  - `nan_counter`
  - `grad_norm_last_reduced_counter`
  - `stable_epochs`
  - RNG state
  - `deterministic_mode_active`
  - effective `seq_len_increase_in_batches`
- Keep these as explicit locals in `trainer.py`, not persisted restart fields:
  - `job_list`
  - dataset / dataloader objects
  - retry batch-size locals
  - `first_epoch_is_evaluation`
  - `flag_out_of_seq_len_increase`
  - `epoch_stop`
  - checkpoint-path locals
  - `_seq_len_now`
  - copied `train_cfg` state

## Construction and restore order

- `train_all_phases()` may load the two restart-state files early only to choose the resumed job and epoch anchor.
- `train_one_phase()` must construct:
  - optimizer
  - schedulers
  - scaler
  - early-stopping
  before restoring `TrainOnePhaseState`.
- The model checkpoint stays separate from `TrainOnePhaseState` and must be loaded explicitly in `train_one_phase()`.
- Restore optimizer / scheduler / scaler state whenever `training_inner_restart.pt` exists in the current Hydra output directory.

## Documentation contract

- Keep `docs/bnode_core/ode/restart_training.md` aligned with:
  - the actual two-file resume workflow
  - the explicit-local-variable flow in `trainer.py`
  - the current persisted-field lists for `TrainAllPhasesState` and `TrainOnePhaseState`
- When code comments or docstrings explain restart ownership, point readers to `docs/bnode_core/ode/restart_training.md`.

## Test expectations

- `tests/ode/test_restart_state.py` should cover:
  - roundtrips for `TrainAllPhasesState`
  - roundtrips for `TrainOnePhaseState`
  - restoring optimizer / scheduler / scaler / early-stopping / RNG state from `TrainOnePhaseState`
  - invalid payload rejection
- `tests/ode/test_bnode.py` resume tests should assert the two-file layout explicitly:
  - interrupted runs leave both restart files behind
  - successful resumed runs remove both restart files
  - MLflow resume metadata comes from the outer restart state
- If persisted fields, restore ordering, or restart filenames change, update docs and all relevant restart tests in the same task.
