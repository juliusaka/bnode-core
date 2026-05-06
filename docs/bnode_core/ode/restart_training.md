# Restart-enabled training workflow

`bnode_core.ode.trainer` can resume interrupted **main-training** phases from two validated restart checkpoints instead of replaying finished work.

## What gets written

At the end of every completed training epoch, the trainer updates the current Hydra output directory with:

- `training_outer_restart.pt`: outer orchestration checkpoint with the resumed job index, the next global epoch anchor, and MLflow/Hydra resume metadata
- `training_inner_restart.pt`: inner phase checkpoint with model, optimizer, scheduler, scaler, early-stopping, RNG, and phase-control state
- `model.pt`: latest in-progress model checkpoint
- `optimizer.pt`: latest in-progress optimizer checkpoint
- `model_phase_<job_idx>.pt` / `optimizer_phase_<job_idx>.pt`: best checkpoint pair for the active phase when early stopping has saved one

If either restart checkpoint is missing, incomplete, or points at a different Hydra output directory, the trainer fails loudly instead of silently starting over.

## State model

The current restart design separates outer orchestration state from inner phase runtime state.

| State | Lives where | Main purpose |
|-------|-------------|--------------|
| `OuterTrainingState` | in memory | Owns the outer loop in `train_all_phases()` |
| `TrainingPhaseState` | in memory | Owns mutable counters and flags for one training phase |
| `LiveTrainingState` | in memory | Owns the active phase runtime objects |
| `OuterTrainingStateCheckpoint` | on disk in `training_outer_restart.pt` | Stores outer resume metadata only |
| `InnerTrainingStateCheckpoint` | on disk in `training_inner_restart.pt` | Stores phase runtime state and phase-control values |

`TrainingRestartState` still exists in the codebase only as a legacy compatibility schema for older restart bundles and targeted unit coverage. Current trainer runs no longer write `training_restart.pt`.

## Checkpoint boundary

### Outer checkpoint

`training_outer_restart.pt` stores only orchestration-level resume data:

- Hydra output directory
- MLflow run metadata
- current job index
- `next_epoch_anchor`

It does **not** store:

- the model state dict
- optimizer or scheduler state
- datasets or dataloaders
- `cfg`

### Inner checkpoint

`training_inner_restart.pt` stores only the active phase state:

- model / optimizer / scheduler / scaler / early-stopping state dicts
- phase-local progress via `phase_epoch`
- `first_epoch_is_evaluation`
- `nan_counter`
- `grad_norm_last_reduced_counter`
- `stable_epochs`
- `flag_out_of_seq_len_increase`
- `epoch_stop`
- `deterministic_mode_active`
- RNG state
- current and best checkpoint paths for the active phase

It does **not** store:

- `cfg`
- dataloaders or dataset handles
- recreated setup values such as retry batch size or worker counts
- persisted copies of `train_cfg`

## Resume flow

Resume is explicit and happens in two layers.

1. `train_all_phases()` loads `training_outer_restart.pt` and `training_inner_restart.pt`.
2. `OuterTrainingState` selects the resumed job and exposes the next global epoch anchor.
3. Once dataloaders and checkpoint paths exist, `LiveTrainingState.create_uninitialized()` creates the phase-local state with runtime object fields still set to `None`.
4. `train_one_phase()` creates the optimizer, schedulers, scaler, and early-stopping helper.
5. `LiveTrainingState.bind_runtime_objects()` attaches those runtime objects and restores their state dicts from `training_inner_restart.pt`.

That split keeps restore order visible:

- outer checkpoint first, to decide **where** training resumes
- inner checkpoint second, after the runtime objects that own the state dicts already exist

## Safe checkpoint behavior

- Checkpoints are written only at epoch boundaries in normal trainer control flow.
- Every checkpoint is written with `checkpoint_reason="epoch_end"`.
- Both restart files are removed after a successful run finishes.
- Slurm time-limit handling is done by the surrounding job script (requeue), not inside the trainer process.

## Manual resume entry point

### Resume in the same Hydra output directory

Re-run the trainer with the original `hydra.run.dir`. The trainer auto-detects `training_outer_restart.pt` and `training_inner_restart.pt` there.

```bash
source .venv/bin/activate
trainer hydra.run.dir=outputs/2026-01-15/12-00-00/abc123 mlflow_tracking_uri=http://127.0.0.1:5001 nn_model=bnode_heatpump_test
```

Use the same MLflow tracking URI and experiment as the original run. The outer restart checkpoint reopens the stored MLflow run ID; conflicting run, experiment, or tracking URI settings are rejected.

Resume only works from that original Hydra output directory. Pointing the trainer at restart checkpoints from some other directory is rejected; there is no separate restart-state override path.

## Operational expectations

- Restart support currently targets interrupted **main-training phases**.
- A resumable run must have both `training_outer_restart.pt` and `training_inner_restart.pt`.
- Finished runs remove both restart files; their presence indicates resumable state still exists.
