# Restart-enabled training workflow

`bnode_core.ode.trainer` can resume interrupted main-training phases from a validated restart bundle instead of replaying finished work.

## What gets written

At the end of every completed training epoch, the trainer updates the current Hydra output directory with:

- `training_restart.pt`: validated restart bundle with phase index, next epoch, optimizer/scheduler/scaler state, early-stopping state, RNG state, MLflow metadata, and checkpoint reason
- `model.pt`: latest in-progress model checkpoint
- `optimizer.pt`: latest in-progress optimizer checkpoint
- `model_phase_<job_idx>.pt` / `optimizer_phase_<job_idx>.pt`: best checkpoint pair for the active phase when early stopping has saved one

If a restart artifact is incomplete or its schema/paths do not match the current Hydra run directory, the trainer fails loudly instead of silently starting over.

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
