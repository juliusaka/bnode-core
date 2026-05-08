# Restart-enabled training workflow

`bnode_core.ode.trainer` now resumes interrupted **main-training** phases with exactly two persisted restart-state objects:

- `train_all_phases_state` in `training_outer_restart.pt`
- `train_one_phase_state` in `training_inner_restart.pt`

## What gets written

At the end of every completed training epoch, the trainer updates the current Hydra output directory with:

- `training_outer_restart.pt`: minimal outer resume state with the resumed job index, the next global epoch anchor, and the MLflow run id
- `training_inner_restart.pt`: minimal inner resume state with phase counters, deterministic-mode status, seq-len state, RNG state, and attached `EarlyStopping` state
- `model.pt`: latest in-progress model checkpoint
- `optimizer.pt`: latest in-progress optimizer checkpoint
- `lr_schedulers.pt`: latest in-progress LR scheduler checkpoint dict
- `grad_scaler.pt`: latest in-progress AMP scaler checkpoint dict
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
2. It recreates outer-loop locals and uses `train_all_phases_state.job_idx` plus `next_epoch_anchor` to choose the resumed main-training job.
3. `train_one_phase()` recreates local path and epoch-bound variables.
4. `train_one_phase()` constructs the optimizer, schedulers, scaler, and early-stopping helper, then attaches the live objects to `train_one_phase_state`.
5. It explicitly loads `model.pt`.
6. It explicitly loads `optimizer.pt`.
7. It explicitly loads `lr_schedulers.pt` and `grad_scaler.pt`.
8. It loads `train_one_phase_state` into the live state object, restoring its counters, RNG state, attached `EarlyStopping` module state, and the effective `seq_len_increase_in_batches`.
9. Local-only flags such as `first_epoch_is_evaluation`, `flag_out_of_seq_len_increase`, and `epoch_stop` are recomputed from config plus the persisted minimal state.

That keeps the restore boundary visible:

- model checkpoint load is explicit
- scheduler and scaler checkpoint loads are explicit
- runtime-object attachment happens before `train_one_phase_state.load(...)`
- non-persisted loop control remains local in the trainer

## Manual resume entry point

### Resume in the same Hydra output directory

Re-run the trainer with the original `hydra.run.dir`. The trainer auto-detects `training_outer_restart.pt` and `training_inner_restart.pt` there.

```bash
source .venv/bin/activate
trainer hydra.run.dir=outputs/2026-01-15/12-00-00/abc123 mlflow_tracking_uri=http://127.0.0.1:5001 nn_model=bnode_heatpump_test
```

Use the same MLflow tracking URI and experiment as the original run. The outer restart state reopens the stored MLflow run id; conflicting run ids are rejected.

## Operational expectations

- Restart support currently targets interrupted **main-training phases**.
- A resumable run must have both restart-state files plus the current runtime checkpoints: `model.pt`, `optimizer.pt`, `lr_schedulers.pt`, and `grad_scaler.pt`.
- The trainer does not keep legacy restart schemas or legacy single-file restart bundles.
