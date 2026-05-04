# Restart-enabled training workflow

`bnode_core.ode.trainer` can resume interrupted main-training phases from a validated restart bundle instead of replaying finished work.

## What gets written

At the end of every completed training epoch, the trainer updates the current Hydra output directory with:

- `training_restart.pt`: validated restart bundle with phase index, next epoch, optimizer/scheduler/scaler state, early-stopping state, RNG state, MLflow metadata, and checkpoint reason
- `model.pt`: latest in-progress model checkpoint
- `optimizer.pt`: latest in-progress optimizer checkpoint
- `model_phase_<job_idx>.pt` / `optimizer_phase_<job_idx>.pt`: best checkpoint pair for the active phase when early stopping has saved one

If a restart artifact is incomplete or its schema/paths do not match the current Hydra run directory, the trainer fails loudly instead of silently starting over.

## State model

The trainer uses three related state objects during restart-enabled training. They have different jobs and different lifetimes.

| State | Lives where | Main purpose |
|-------|-------------|--------------|
| `TrainingPhaseState` | in memory | Mutable counters and flags for one training phase |
| `LiveTrainingState` | in memory | Runtime bundle for the active phase, including model, optimizer, schedulers, scaler, and paths |
| `TrainingRestartState` | on disk in `training_restart.pt` | Serialized checkpoint schema used to resume a phase later |

### `TrainingPhaseState`

`TrainingPhaseState` contains the small mutable values that change while a phase runs:

- `phase_epoch_0`, `epoch_start`, `epoch_stop`
- `first_epoch_is_evaluation`
- `nan_counter`
- `grad_norm_last_reduced_counter`
- `stable_epochs`
- `flag_out_of_seq_len_increase`
- `deterministic_mode_active`

This object is intentionally narrow. It tracks phase-local progress and training-control flags, but it does not own the model, optimizer, or checkpoint files.

### `LiveTrainingState`

`LiveTrainingState` is the runtime object used inside training code such as `train_one_epoch(...)`.

It owns the live training objects for the current phase, including:

- `model`
- `optimizer`
- `lr_schedulers`
- `scaler`
- `early_stopping`
- `train_cfg`
- checkpoint paths and Hydra output paths
- the current `TrainingPhaseState`

It is the main in-memory state carrier once a phase has created all runtime objects.

### Two-phase construction

`LiveTrainingState` uses a two-phase pattern so that `TrainingPhaseState` counters
(epoch bounds, NaN counter, sequence-length flags) are available as soon as paths
and epoch bounds are known — before the optimizer and schedulers are created.

**Phase 1 — `create_uninitialized()`**

Called in `train_all_phases()` just before `train_one_phase()` is invoked (after dataloaders
are created, since epoch bounds require `len(dataloaders['train'])`).

- Builds `TrainingPhaseState` from `TrainingRestartState` metadata (or fresh if no
  restart state is present).
- Populates all config and path fields.
- Leaves runtime object fields (`model`, `optimizer`, `lr_schedulers`, `scaler`,
  `early_stopping`) as `None`.

`train_one_phase()` accepts an optional `live_state` parameter. If none is passed,
`create_uninitialized()` is called inside the function instead, so `train_one_phase()`
remains usable as a standalone entry point.

**Phase 2 — `bind_runtime_objects()`**

Called after the optimizer, schedulers, scaler, and early-stopping helper have been
constructed.

- Sets the runtime object fields on the existing `LiveTrainingState` instance.
- If a `restart_state` is provided, calls `load_checkpoint()` to restore all
  state_dicts (model weights, optimizer state, scheduler state, scaler state,
  early-stopping state) and the RNG state.

`LiveTrainingState.create()` is a convenience wrapper that combines both phases and
is kept for backward compatibility.

### Why `bind_runtime_objects()` is deferred to `train_one_phase()`

`create_uninitialized()` is called in `train_all_phases()` just before `train_one_phase()` is
invoked, once dataloaders exist. At that point the trainer already knows:

- the phase file paths
- epoch bounds (derived from `len(dataloaders['train'])` and `train_cfg`)
- restart metadata (from `TrainingRestartState` if resuming)

But it cannot call `bind_runtime_objects()` yet, because the selected phase still
needs to create runtime objects that are not available at that point:

- the phase-specific `optimizer`
- the active schedulers
- the AMP scaler
- the early-stopping helper

For that reason, resume is split into two steps:

1. `LiveTrainingState.create_uninitialized(restart_state=restart_state)` — called in
   `train_all_phases()`, builds `phase_state` from the checkpoint; all config/path fields
   populated; runtime objects are `None`. The `live_state` object is passed into
   `train_one_phase()` as a parameter.
2. After the optimizer, schedulers, scaler, and early-stopping helper are created inside
   `train_one_phase()`: `live_state.bind_runtime_objects(..., restart_state=restart_state)`
   sets runtime objects and restores all state_dicts.

### `TrainingRestartState`

`TrainingRestartState` is the validated serialized checkpoint payload written to `training_restart.pt`.

It stores the data needed to restore a phase later, including:

- MLflow and Hydra metadata
- `job_idx`, `epoch_0`, `next_epoch`
- serialized model / optimizer / scheduler / scaler / early-stopping state
- persisted copies of phase counters
- RNG state

`trainer.py` should treat this as a persistence contract, not as the main runtime object. The current design keeps the serialized schema internal and restores it into `LiveTrainingState` when the runtime objects are ready.

### Recommended mental model

- Use `TrainingPhaseState` for **phase-local mutable counters**
- Use `LiveTrainingState` for **active runtime state**
- Use `TrainingRestartState` for **serialized resume data**

If you are reading the code, the important transition is:

`TrainingRestartState` (checkpoint file) -> `LiveTrainingState` (active phase runtime)

## Safe checkpoint behavior

- Checkpoints are written only at epoch boundaries in normal trainer control flow — never in a signal handler.
- Every checkpoint is written with `checkpoint_reason="epoch_end"`.
- Slurm time-limit handling is done by the surrounding job script (requeue), not inside the trainer process.

## Manual resume entry points

### Resume in the same Hydra output directory

Re-run the trainer with the original `hydra.run.dir`. The trainer auto-detects `training_restart.pt` there.

```bash
source .venv/bin/activate
trainer hydra.run.dir=outputs/2026-01-15/12-00-00/abc123 mlflow_tracking_uri=http://127.0.0.1:5001 nn_model=bnode_heatpump_test
```

Use the same MLflow tracking URI / experiment as the original run. The restart bundle reopens the stored MLflow run ID; conflicting `mlflow_run_id`, experiment, or tracking URI settings are rejected.

Resume only works from that original Hydra output directory. Pointing the trainer at a restart bundle from some other directory is rejected; there is no separate restart-state override path.

## Operational expectations

- Restart support currently targets interrupted **main-training phases**.
- Finished runs clear `training_restart.pt`; its presence indicates resumable state still exists.
- External wrappers should request checkpoints, wait for the trainer to write `training_restart.pt`, and only then decide whether to relaunch or requeue.
