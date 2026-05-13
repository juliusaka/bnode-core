# Restart-enabled training workflow

`bnode_core.ode.trainer` resumes interrupted **main-training phases and the test job** with exactly two persisted restart-state objects:

- `train_all_phases_state` in `training_outer_restart.pt`
- `train_one_phase_state` in `training_inner_restart.pt`

## What gets written

At the end of every completed training epoch, `RestartCheckpointStore` updates the current Hydra output directory with atomic writes:

- `training_outer_restart.pt`: minimal outer resume state with the resumed job index, the next global epoch anchor, MLflow run id, and checkpoint UUID metadata
- `training_inner_restart.pt`: minimal inner resume state with phase counters, deterministic-mode status, seq-len state, RNG state, attached `EarlyStopping` state, and matching checkpoint UUID metadata
- `model.pt`: latest in-progress model checkpoint
- `optimizer.pt`: latest in-progress optimizer checkpoint
- `lr_schedulers.pt`: latest in-progress LR scheduler checkpoint dict (saved atomically)
- `grad_scaler.pt`: latest in-progress AMP scaler checkpoint dict (saved atomically)
- `model_phase_<job_idx>.pt` / `optimizer_phase_<job_idx>.pt`: best checkpoint pair for the active phase when early stopping has saved one

Finished runs remove both restart-state files and both runtime checkpoint files.

## State model

The current restart contract deliberately keeps only two persisted state objects and pushes everything else back into explicit locals in the trainer loops.

| State | Lives where | Main purpose |
|-------|-------------|--------------|
| `TrainAllPhasesState` | on disk in `training_outer_restart.pt` | outer orchestration resume anchor |
| `TrainOnePhaseState` | on disk in `training_inner_restart.pt` | phase-local runtime/control resume state |

Important runtime values are **not** wrapped in a long-lived live-state object anymore. `trainer.py` recreates them as locals:

- `job_list`
- restart file paths
- dataset / dataloader objects
- `model`
- `first_epoch_is_evaluation`
- `flag_out_of_seq_len_increase`
- `epoch_stop`
- per-phase checkpoint paths
- `batches_per_epoch`
- `max_epochs`
- `_seq_len_now`

## Checkpoint boundary

### `train_all_phases_state`

`training_outer_restart.pt` stores only:

- `job_idx`
- `next_epoch_anchor`
- `mlflow_run_id`
- `checkpoint_uuid`
- `state_version`

It does **not** store:

- Hydra output dir
- restart file path
- tracking URI / experiment name
- datasets or dataloaders
- model state
- retry locals
- `cfg`

### `train_one_phase_state`

`training_inner_restart.pt` stores only what the phase must restore through the restart state's own `state_dict()`:

- `phase_epoch`
- `nan_counter`
- `grad_norm_last_reduced_counter`
- `stable_epochs`
- RNG state
- `deterministic_mode_active`
- effective `seq_len_increase_in_batches`
- `checkpoint_uuid`
- `state_version`
- attached `EarlyStopping` module state when the trainer has attached that runtime object before saving

It does **not** store:

- model state inside the restart-state file
- optimizer state inside the restart-state file
- scheduler state inside the restart-state file
- scaler state inside the restart-state file
- checkpoint paths
- `first_epoch_is_evaluation`
- `flag_out_of_seq_len_increase`
- `epoch_stop`
- `seq_len_now`
- `cfg`
- datasets or dataloaders

The model remains a separate explicit checkpoint file.

## Resume flow

Resume now happens in this order:

1. `train_all_phases()` loads `training_outer_restart.pt` and `training_inner_restart.pt`.
2. It recreates outer-loop locals and uses `train_all_phases_state.job_idx` plus `next_epoch_anchor` to choose the resumed job (main-training phase or test job).
3. **When resuming a main-training phase:** `train_one_phase()` recreates local path and epoch-bound variables, loads `model.pt`, `optimizer.pt`, `lr_schedulers.pt` and `grad_scaler.pt`, then restores the inner-state counters and RNG.
4. **When resuming the test job:** the trainer skips all training phases, loads `model.pt` into a freshly-initialised model, and runs `_run_test_job` directly.

That keeps the restore boundary visible:

- model checkpoint load is explicit
- scheduler and scaler checkpoint loads are explicit
- only `EarlyStopping` attachment happens before `train_one_phase_state.load(...)`
- restart-state pair UUIDs are validated on load and synchronized on save
- restart files and runtime scheduler/scaler payloads are written atomically
- non-persisted loop control remains local in the trainer

## Accumulating counters across restarts

`nan_counter` and `stable_epochs` are stored in `TrainOnePhaseState` and therefore **accumulate across restarts**. They are not reset to zero when a run is resumed.

- **`nan_counter`**: counts how many consecutive NaN-loss epochs have occurred. The trainer aborts after a config-defined threshold. If a job is interrupted and resumed, the counter picks up from its last saved value. An interrupted run that was already close to the NaN limit will exhaust the remaining NaN budget on the resumed run. When reasoning about NaN budget, always check the value that will be restored, not just the per-epoch behavior.
- **`stable_epochs`**: counts epochs in which gradient-norm reduction was active. It likewise continues accumulating from the checkpoint value on resume.

This is intentional — the counters reflect the *lifetime* behavior of the training phase, not just the current segment of execution. If you need to reset a counter after an interrupted run (for example, after an infrastructure failure that caused spurious NaNs), edit the saved checkpoint directly or start a fresh phase.

## Manual resume entry point

### Resume in the same Hydra output directory

Re-run the trainer with the original `hydra.run.dir`. The trainer auto-detects `training_outer_restart.pt` and `training_inner_restart.pt` there.

```bash
source .venv/bin/activate
trainer hydra.run.dir=outputs/2026-01-15/12-00-00/abc123 mlflow_tracking_uri=http://127.0.0.1:5001 nn_model=bnode_heatpump_test
```

Use the same MLflow tracking URI and experiment as the original run. The outer restart state reopens the stored MLflow run id; conflicting run ids are rejected.

## Operational expectations

- A resumable run must have both restart-state files.  Main-training resume also requires the runtime checkpoints: `model.pt`, `optimizer.pt`, `lr_schedulers.pt`, and `grad_scaler.pt`.  Test-job resume requires only `model.pt`.
- The trainer does not keep legacy restart schemas or legacy single-file restart bundles.

## Test-job resume

Before calling `_run_test_job`, the trainer calls `save_outer_restart_state_for_test_job` to advance `job_idx` to the test-job index and persist the outer restart state.  This means a kill during the test job leaves the outer state pointing at the test job, so the next resume re-runs only the test job — no training epochs are repeated.

MLflow double-logging is safe: if some test metrics were logged before the kill, the resumed run logs all test metrics again at the same step value.  MLflow records all data points without error.
