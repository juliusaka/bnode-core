# Restart-enabled training workflow

`bnode_core.ode.trainer` resumes interrupted **main-training phases and the test job** with exactly two persisted restart-state objects, bundled in a single file:

- `train_all_phases_state` (outer) — in `training_restart_checkpoint.pt`
- `train_one_phase_state` (inner) — in `training_restart_checkpoint.pt`

## What gets written

At the end of every completed training epoch, `RestartCheckpointStore` updates the current Hydra output directory with a single atomic write:

- `training_restart_checkpoint.pt`: a versioned bundle containing the outer state, inner state, LR scheduler state dict, GradScaler state dict, model state dict, and optimizer state dict — written as one `torch.save` call and installed via `os.replace`, so the file is always either the previous complete bundle or the new complete bundle (never a partial mix)
- `model_phase_<job_idx>.pt` / `optimizer_phase_<job_idx>.pt`: best checkpoint pair for the active phase when early stopping has saved one

Finished runs remove the restart checkpoint bundle file and write `training_complete.marker` in its place.

## Already-complete guard

When Slurm requeues a job after training has already finished (e.g. due to a node failure immediately after the final epoch), the trainer detects `training_complete.marker` at startup and returns immediately without rerunning any training phases.  This prevents the common failure mode where a spurious requeue causes training to restart from scratch.

## State model

The current restart contract deliberately keeps only two persisted state objects and pushes everything else back into explicit locals in the trainer loops.

| State | Lives where | Main purpose |
|-------|-------------|--------------|
| `TrainAllPhasesState` | `bundle["outer"]` in `training_restart_checkpoint.pt` | outer orchestration resume anchor |
| `TrainOnePhaseState` | `bundle["inner"]` in `training_restart_checkpoint.pt` | phase-local runtime/control resume state |

Important runtime values are **not** wrapped in a long-lived live-state object anymore. `trainer.py` recreates them as locals:

- `job_list`
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

The outer state stored in `bundle["outer"]` contains only:

- `job_idx`
- `next_epoch_anchor`
- `mlflow_run_id`
- `state_version`

It does **not** store:

- Hydra output dir
- tracking URI / experiment name
- datasets or dataloaders
- model state
- retry locals
- `cfg`

### `train_one_phase_state`

The inner state stored in `bundle["inner"]` contains only what the phase must restore:

- `phase_epoch`
- `nan_counter`
- `grad_norm_last_reduced_counter`
- `stable_epochs`
- RNG state
- `deterministic_mode_active`
- effective `seq_len_increase_in_batches`
- `state_version`
- attached `EarlyStopping` module state when the trainer has attached that runtime object before saving

It does **not** store:

- checkpoint paths
- `first_epoch_is_evaluation`
- `flag_out_of_seq_len_increase`
- `epoch_stop`
- `seq_len_now`
- `cfg`
- datasets or dataloaders

The model state dict and optimizer state dict are stored directly in the bundle at `bundle["model"]` and `bundle["optimizer"]`. Scheduler and scaler states are stored in `bundle["scheduler"]` and `bundle["scaler"]` respectively.

## Resume flow

Resume happens in this order:

1. `train_all_phases()` loads `training_restart_checkpoint.pt` via `RestartCheckpointStore.load_checkpoint_if_available()`, which returns `(outer_state, inner_state, scheduler_states, scaler_state, model_state_dict, optimizer_state_dict)`.
2. It recreates outer-loop locals and uses `train_all_phases_state.job_idx` plus `next_epoch_anchor` to choose the resumed job (main-training phase or test job).
3. **When resuming a main-training phase:** `train_one_phase()` receives the model and optimizer state dicts from the bundle as `restart_model_state` and `restart_optimizer_state`, plus scheduler and scaler state dicts as `restart_scheduler_states` and `restart_scaler_state`. It loads the model and optimizer from those state dicts, then restores the scheduler/scaler from the bundle dicts, then restores the inner-state counters and RNG from the bundle.
4. **When resuming the test job:** the trainer skips all training phases, loads the model state from the bundle (`restart_model_state`) into a freshly-initialised model, and runs `_run_test_job` directly.

That keeps the restore boundary visible:

- model and optimizer checkpoint loads come directly from bundle state dicts
- scheduler and scaler checkpoint loads come directly from the bundle dicts
- only `EarlyStopping` attachment happens before `train_one_phase_state` reload
- restart bundle is written atomically — one `os.replace`, no partial-save window
- non-persisted loop control remains local in the trainer

## Accumulating counters across restarts

`nan_counter` and `stable_epochs` are stored in `TrainOnePhaseState` and therefore **accumulate across restarts**. They are not reset to zero when a run is resumed.

- **`nan_counter`**: counts how many consecutive NaN-loss epochs have occurred. The trainer aborts after a config-defined threshold. If a job is interrupted and resumed, the counter picks up from its last saved value. An interrupted run that was already close to the NaN limit will exhaust the remaining NaN budget on the resumed run. When reasoning about NaN budget, always check the value that will be restored, not just the per-epoch behavior.
- **`stable_epochs`**: counts epochs in which gradient-norm reduction was active. It likewise continues accumulating from the checkpoint value on resume.

This is intentional — the counters reflect the *lifetime* behavior of the training phase, not just the current segment of execution. If you need to reset a counter after an interrupted run (for example, after an infrastructure failure that caused spurious NaNs), edit the saved checkpoint directly or start a fresh phase.

## Manual resume entry point

### Resume in the same Hydra output directory

Re-run the trainer with the original `hydra.run.dir`. The trainer auto-detects `training_restart_checkpoint.pt` there.

```bash
source .venv/bin/activate
trainer hydra.run.dir=outputs/2026-01-15/12-00-00/abc123 mlflow_tracking_uri=http://127.0.0.1:5001 nn_model=bnode_heatpump_test
```

Use the same MLflow tracking URI and experiment as the original run. The outer restart state reopens the stored MLflow run id; conflicting run ids are rejected.

## Slurm-managed restarts

### How it works

Slurm sends `SIGTERM` to the job script before killing it at the time limit. The template should register an `on_term` handler that calls `scontrol requeue` and exits cleanly:

```bash
#SBATCH --open-mode=append   # keep log file across requeues
#SBATCH --requeue             # also requeue on node failure

on_term() {
    echo "Time limit reached. Requeuing job ${SLURM_JOB_ID}..."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap on_term TERM
```

When `on_term` calls `exit 0`, the `EXIT` trap registered by the generated block fires and kills the background trainer processes. The trainer has already written `training_restart_checkpoint.pt` at the end of the last completed epoch, so the requeued job resumes from there.

| Mechanism | Detail |
|-----------|--------|
| `#SBATCH --requeue` | Slurm requeues automatically on node failure (SIGKILL) |
| `trap on_term TERM` | Requeues at time limit (SIGTERM); lets EXIT trap clean up trainers |
| `#SBATCH --open-mode=append` | Log file is appended, not overwritten, on each requeue |
| `RUN_DIR` (UUID-pinned) | Fixed at generation time; every requeue lands in the same Hydra output directory |
| `training_restart_checkpoint.pt` | Written atomically at the end of each epoch; auto-detected on next start |
| `bnode_slurm_script_completed=1` | Disarms the EXIT trap on successful completion so Slurm records a clean exit |

## Operational expectations

- A resumable run must have `training_restart_checkpoint.pt`. The bundle contains model, optimizer, scheduler, and scaler state — no separate `model.pt` or `optimizer.pt` restart files are required.
- The trainer does not keep legacy restart schemas or multi-file restart layouts.

## Test-job resume

Before calling `_run_test_job`, the trainer calls `save_outer_for_test_job` to advance `job_idx` to the test-job index and re-save the bundle with the updated outer state. This means a kill during the test job leaves the bundle pointing at the test job, so the next resume re-runs only the test job — no training epochs are repeated.

MLflow double-logging is safe: if some test metrics were logged before the kill, the resumed run logs all test metrics again at the same step value. MLflow records all data points without error.
