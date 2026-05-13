---
name: bnode-core trainer restart states
description: Restart-state architecture guidance for trainer resume logic, checkpoint store, docs, and tests
applyTo: "src/bnode_core/ode/trainer.py,src/bnode_core/ode/trainer_utils/restart_state.py,src/bnode_core/ode/trainer_utils/restart_utils.py,src/bnode_core/ode/trainer_utils/restart_checkpoint_store.py,tests/ode/test_restart_state.py,docs/bnode_core/ode/restart_*.md"
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
  - `checkpoint_uuid`
  - `state_version`
- `TrainOnePhaseState` owns only the persisted inner runtime/control state:
  - `phase_epoch`
  - `nan_counter` — accumulates across restarts; reflects lifetime NaN-loss events for this phase, not just the current execution segment
  - `grad_norm_last_reduced_counter`
  - `stable_epochs` — accumulates across restarts; reflects lifetime stable-gradient epochs
  - RNG state
  - `deterministic_mode_active`
  - effective `seq_len_increase_in_batches`
  - `checkpoint_uuid`
  - `state_version`
  - attached `EarlyStopping` module state when the trainer attaches it before `load()`
- Keep these as explicit locals in `trainer.py`, not persisted restart fields:
  - `job_list`
  - dataset / dataloader objects
  - optimizer / scheduler / scaler runtime objects
  - retry batch-size locals
  - `first_epoch_is_evaluation`
  - `flag_out_of_seq_len_increase`
  - `epoch_stop`
  - checkpoint-path locals
  - `_seq_len_now`
  - copied `train_cfg` state

## Construction and restore order

- `train_all_phases()` may load the two restart-state files early only to choose the resumed job and epoch anchor.
- `RestartCheckpointStore` owns:
  - atomic writes for restart/runtime artifacts
  - outer/inner checkpoint UUID pairing
  - restart-runtime cleanup (`training_outer_restart.pt`, `training_inner_restart.pt`, `lr_schedulers.pt`, `grad_scaler.pt`)
- `train_one_phase()` must construct:
  - optimizer
  - schedulers
  - scaler
  - early-stopping
  before calling `TrainOnePhaseState.load(...)`.
- Keep `EarlyStopping` module-backed so it can be attached directly to `TrainOnePhaseState` before save/load instead of being converted into a separate restart dict.
- The model checkpoint stays separate from `TrainOnePhaseState` and must be loaded explicitly in `train_one_phase()`.
- Restore the runtime optimizer, schedulers, and scaler explicitly from:
  - `optimizer.pt`
  - `lr_schedulers.pt`
  - `grad_scaler.pt`
  whenever `training_inner_restart.pt` exists in the current Hydra output directory.
- Keep state-class special serialization explicit via class-level serializer mapping for non-trivial fields and raise clear errors when a declared serializer method is missing.

## Documentation contract

- Keep `docs/bnode_core/ode/restart_training.md` aligned with:
  - the actual two-file resume workflow
  - the explicit-local-variable flow in `trainer.py`
  - the current persisted-field lists for `TrainAllPhasesState` and `TrainOnePhaseState`
- The "Accumulating counters across restarts" section in `restart_training.md` must document that `nan_counter` and `stable_epochs` accumulate across restarts (they are lifetime counts for the phase, not per-segment). Keep this section when editing the doc.
- When code comments or docstrings explain restart ownership, point readers to `docs/bnode_core/ode/restart_training.md`.

## Test expectations

- `tests/ode/test_restart_state.py` should cover:
  - roundtrips for `TrainAllPhasesState`
  - roundtrips for `TrainOnePhaseState`
  - syncing effective `seq_len_increase_in_batches` at epoch-end checkpoint save boundary
  - restoring early-stopping and RNG state from `TrainOnePhaseState`
  - UUID validation/synchronization and atomic save behavior in `RestartCheckpointStore`
  - explicit serializer-missing failure behavior for declared special fields
- `tests/ode/test_bnode.py` resume tests should assert the two-file layout explicitly:
  - interrupted runs leave both restart files behind
  - interrupted runs also leave `lr_schedulers.pt` and `grad_scaler.pt` behind
  - successful resumed runs remove all four restart/runtime checkpoint files
  - MLflow resume metadata comes from the outer restart state
- If persisted fields, restore ordering, or restart filenames change, update docs and all relevant restart tests in the same task.
