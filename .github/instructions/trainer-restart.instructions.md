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
- Current trainer runs use **one restart checkpoint bundle** and **one completion marker**:
  - `training_restart_checkpoint.pt` — active during training; removed on clean completion
  - `training_complete.marker` — written by `clear_restart_artifacts()` at the end of successful training; checked at `train_all_phases()` startup to guard against spurious Slurm requeues after completion
- Do not re-introduce wrapper-state layers such as `OuterTrainingState`, `TrainingPhaseState`, or `LiveTrainingState`.
- Do not add legacy compatibility readers, old multi-file restart schemas, or obsolete restart filenames unless the user explicitly asks for compatibility.

## Ownership boundary

- `TrainAllPhasesState` owns only the outer resume anchor:
  - `job_idx`
  - `next_epoch_anchor`
  - `mlflow_run_id`
  - `state_version`
- `TrainOnePhaseState` owns only the persisted inner runtime/control state:
  - `phase_epoch`
  - `nan_counter` — accumulates across restarts; reflects lifetime NaN-loss events for this phase, not just the current execution segment
  - `grad_norm_last_reduced_counter`
  - `stable_epochs` — accumulates across restarts; reflects lifetime stable-gradient epochs
  - RNG state
  - `deterministic_mode_active`
  - effective `seq_len_increase_in_batches`
  - `state_version`
  - attached `EarlyStopping` module state when the trainer attaches it before `load()`
- Keep these as explicit locals in `trainer.py`, not persisted restart fields:
  - `job_list`
  - dataset / dataloader objects
  - optimizer / scheduler / scaler runtime objects (restored from `bundle["optimizer"]`, `bundle["scheduler"]`, `bundle["scaler"]` respectively)
  - retry batch-size locals
  - `first_epoch_is_evaluation`
  - `flag_out_of_seq_len_increase`
  - `flag_out_of_warmup` — re-derived from `phase_state.phase_epoch * batches_per_epoch >= warmup_batches` on resume; no extra persisted field needed
  - `epoch_stop`
  - checkpoint-path locals
  - `_seq_len_now`
  - copied `train_cfg` state

## Construction and restore order

- `train_all_phases()` loads the bundle early only to choose the resumed job and epoch anchor.
- `RestartCheckpointStore` owns:
  - atomic writes for the restart bundle (`training_restart_checkpoint.pt`)
  - restart cleanup (single bundle file) — `clear_restart_artifacts()` removes the bundle and writes `training_complete.marker`
  - the completion guard — `is_training_complete()` returns `True` when `training_complete.marker` exists
- `train_one_phase()` must construct:
  - optimizer
  - schedulers
  - scaler
  - early-stopping
  before restoring state from the bundle.
- Keep `EarlyStopping` module-backed so it can be attached directly to `TrainOnePhaseState` before save/load instead of being converted into a separate restart dict.
- The model state dict and optimizer state dict are stored inside the bundle (`bundle["model"]` and `bundle["optimizer"]`). They are **not** written as separate files. Load them explicitly in `train_one_phase()` from the state dicts passed in via `restart_model_state` and `restart_optimizer_state`.
- Restore schedulers and scaler from the bundle (passed as `restart_scheduler_states` and `restart_scaler_state` dicts to `train_one_phase()`).
- Keep state-class special serialization explicit via class-level serializer mapping for non-trivial fields and raise clear errors when a declared serializer method is missing.

## Documentation contract

- Keep `docs/bnode_core/ode/restart_training.md` aligned with:
  - the actual single-file bundle resume workflow
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
  - `training_complete.marker` and `is_training_complete()` behavior in `RestartCheckpointStore`
  - early-exit guard in `train_all_phases()` when `is_training_complete()` is True
  - explicit serializer-missing failure behavior for declared special fields
  - saving/loading model and optimizer state dicts in the bundle
- `tests/ode/test_bnode.py` resume tests should assert the single-file bundle layout explicitly:
  - interrupted runs leave `training_restart_checkpoint.pt` behind (no separate `model.pt` or `optimizer.pt`)
  - successful completed runs remove the restart checkpoint file and write `training_complete.marker`
  - a second trainer run on the same output directory (simulating Slurm requeue after completion) exits immediately without retraining
  - MLflow resume metadata comes from the outer state in the bundle
  - `model_phase_{idx}.pt` / `optimizer_phase_{idx}.pt` (EarlyStopping best) remain as separate files
- If persisted fields, restore ordering, or restart filenames change, update docs and all relevant restart tests in the same task.
