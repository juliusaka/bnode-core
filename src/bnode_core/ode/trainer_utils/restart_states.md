# Current restart state inventory

This file summarizes the **current** persisted restart-state contract after the
wrapper-state removal.

## 1. `train_all_phases()` side

### `TrainAllPhasesState`

Persisted outer state written to `training_outer_restart.pt`.

| Field | Purpose |
| --- | --- |
| `job_idx` | which main-training job resumes |
| `next_epoch_anchor` | next global epoch passed back into `train_one_phase()` |
| `mlflow_run_id` | strict resume into the same MLflow run |

Everything else is local again in `train_all_phases()`:

- `job_list`
- outer / inner restart paths
- datasets / dataloaders
- `model_created`
- `model`
- retry batch-size locals

## 2. `train_one_phase()` side

### `TrainOnePhaseState`

Persisted inner state written to `training_inner_restart.pt`.

| Field | Purpose |
| --- | --- |
| `phase_epoch` | progress inside the current phase |
| `optimizer_state` | restore optimizer state after recreation |
| `scheduler_states` | restore scheduler state after recreation |
| `scaler_state` | restore AMP scaler state after recreation |
| `early_stopping_state` | restore early-stopping progress |
| `nan_counter` | continue NaN-recovery behavior |
| `grad_norm_last_reduced_counter` | continue clip-grad reduction behavior |
| `stable_epochs` | preserve seq-len-stability progress |
| `rng_state` | preserve RNG continuity |
| `deterministic_mode_active` | preserve deterministic-mode status |
| `seq_len_increase_in_batches` | preserve the exact resumed seq-len / epoch schedule |

The model is **not** embedded in this state object. It remains an explicit checkpoint file.

## 3. Variables that are local again

These are intentionally not persisted in restart-state files anymore:

- `first_epoch_is_evaluation`
- `flag_out_of_seq_len_increase`
- `epoch_stop`
- `_seq_len_now`
- current / best checkpoint paths
- `batches_per_epoch`
- `max_epochs`
- datasets / dataloaders

Those values are reconstructed from config, filepaths, and the two minimal restart states.
