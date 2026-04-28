# Restart-enabled training workflow

`bnode_core.ode.trainer` can resume interrupted main-training phases from a validated restart bundle instead of replaying finished work.

## What gets written

At the end of every completed training epoch, the trainer updates the current Hydra output directory with:

- `training_restart.pt`: validated restart bundle with phase index, next epoch, optimizer/scheduler/scaler state, early-stopping state, RNG state, MLflow metadata, and checkpoint reason
- `model.pt`: latest in-progress model checkpoint
- `optimizer.pt`: latest in-progress optimizer checkpoint
- `model_phase_<job_idx>.pt` / `optimizer_phase_<job_idx>.pt`: best checkpoint pair for the active phase when early stopping has saved one

If a restart artifact is incomplete or its schema/paths do not match the requested resume mode, the trainer fails loudly instead of silently starting over.

## Safe checkpoint behavior

- The trainer installs minimal `SIGUSR1` and `SIGTERM` handlers that only record that a checkpoint was requested.
- Serialization still happens in normal trainer control flow after a safe epoch boundary.
- A checkpoint requested by Slurm is written with `checkpoint_reason="signal_request"` and records the received signal name.

## Manual resume entry points

### Resume in the same Hydra output directory

Re-run the trainer with the original `hydra.run.dir`. The trainer auto-detects `training_restart.pt` there.

```bash
source .venv/bin/activate
trainer hydra.run.dir=outputs/2026-01-15/12-00-00/abc123 mlflow_tracking_uri=http://127.0.0.1:5001 nn_model=bnode_heatpump_test
```

Use the same MLflow tracking URI / experiment as the original run. The restart bundle reopens the stored MLflow run ID; conflicting `mlflow_run_id`, experiment, or tracking URI settings are rejected.

### Resume from an explicit restart artifact

Point `restart_state_path` at an existing restart bundle. This is the manual entry point for relaunching into a new Hydra output directory while reusing the original MLflow run.

```bash
source .venv/bin/activate
trainer \
  restart_state_path=/absolute/path/to/old-run/training_restart.pt \
  hydra.run.dir=outputs/manual-resume/run-01 \
  mlflow_tracking_uri=http://127.0.0.1:5001 \
  nn_model=bnode_heatpump_test
```

When `restart_state_path` targets another Hydra output directory, the trainer copies the referenced current/best checkpoints into the new output directory and tags the resumed MLflow run with both source and target Hydra output paths.

## Operational expectations

- Restart support currently targets interrupted **main-training phases**.
- Finished runs clear `training_restart.pt`; its presence indicates resumable state still exists.
- External wrappers should request checkpoints, wait for the trainer to write `training_restart.pt`, and only then decide whether to relaunch or requeue.
