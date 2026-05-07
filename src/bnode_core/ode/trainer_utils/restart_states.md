# Current restart state inventory

This file summarizes the **current** state-like objects and their fields in
`restart_state.py` so the structure can be redesigned more easily.

## 1. `train_all_phases()` side

### `OuterTrainingStateCheckpoint`

Persisted outer checkpoint written to `training_outer_restart.pt`.

| Field | Type |
| --- | --- |
| `schema_version` | `int` | USER: we do not need version numbers
| `hydra_output_dir` | `str` | USER: given by hydra
| `restart_state_path` | `str` | USER: is always in the output dir, can be removed
| `checkpoint_reason` | `str` | USER: not needed
| `mlflow_run_id` | `str | None` | 
| `mlflow_tracking_uri` | `str | None` | USER: not needed
| `mlflow_experiment_name` | `str | None` | USER: not needed
| `job_idx` | `int` | USER: needed 
| `next_epoch_anchor` | `int` | 
| `slurm_job_id` | `str | None` | USER: not needed

USER: Rename to train_all_phases_state, use a torch.nn.module superclass. no need for custom checkpointing

### `OuterTrainingState`

Live outer state used by `train_all_phases()`.

| Attribute | Type / source |
| --- | --- |
| `cfg` | constructor arg |
| `job_list` | constructor arg |
| `outer_restart_state_path` | constructor arg |
| `inner_restart_state_path` | constructor arg |
| `outer_restart_state` | `OuterTrainingStateCheckpoint | None` |
| `inner_restart_state` | `InnerTrainingStateCheckpoint | None` |
| `job_start_idx` | derived from `outer_restart_state.job_idx` or `0` |
| `next_epoch_anchor` | derived from `outer_restart_state.next_epoch_anchor` or `0` |

USER: completely remove this. Re-add the variables as before the adding the centralized restart states (see git commits)

## 2. `train_one_phase()` side

### `InnerTrainingStateCheckpoint`

Persisted inner checkpoint written to `training_inner_restart.pt`.

| Field | Type |
| --- | --- |
| `schema_version` | `int` | USER: not needed
| `hydra_output_dir` | `str` | USER: not needed
| `restart_state_path` | `str` | USER: not needed
| `checkpoint_reason` | `str` | USER: not needed
| `job_idx` | `int` | USER: not needed
| `phase_epoch` | `int` |
| `first_epoch_is_evaluation` | `bool` |
| `current_model_path` | `str` |
| `current_optimizer_path` | `str` |
| `best_model_path` | `str` |
| `best_optimizer_path` | `str` |
| `model_state` | `dict[str, Any]` |
| `optimizer_state` | `dict[str, Any]` |
| `scheduler_states` | `dict[str, dict[str, Any]]` |
| `scaler_state` | `dict[str, Any]` |
| `early_stopping_state` | `dict[str, Any]` |
| `nan_counter` | `int` |
| `grad_norm_last_reduced_counter` | `int` |
| `stable_epochs` | `int` |
| `flag_out_of_seq_len_increase` | `bool` |
| `epoch_stop` | `int | None` |
| `rng_state` | `dict[str, Any]` |
| `deterministic_mode_active` | `bool` |
| `slurm_job_id` | `str | None` |

### `TrainingPhaseState`

Phase-local mutable counters and flags.

| Field | Type |
| --- | --- |
| `phase_epoch_0` | `int` |
| `epoch_start` | `int` |
| `epoch_stop` | `int` |
| `first_epoch_is_evaluation` | `bool` |
| `nan_counter` | `int` |
| `grad_norm_last_reduced_counter` | `int` |
| `stable_epochs` | `int` |
| `flag_out_of_seq_len_increase` | `bool` |
| `deterministic_mode_active` | `bool` |

### `LiveTrainingState`

Live inner runtime state used by `train_one_phase()`.

| Attribute | Type / source |
| --- | --- |
| `cfg` | constructor arg |
| `model` | `torch.nn.Module | None` |
| `optimizer` | `torch.optim.Optimizer | None` |
| `lr_schedulers` | `dict | None` |
| `scaler` | `GradScaler | None` |
| `early_stopping` | `Any | None` |
| `train_cfg` | constructor arg |
| `job_idx` | `int` |
| `pre_train` | `bool` |
| `device` | `torch.device` |
| `phase_state` | `TrainingPhaseState` |
| `path_best_model` | `Path` |
| `path_optimizer_best_model` | `Path` |
| `path_current_model` | `Path` |
| `path_current_optimizer` | `Path` |
| `hydra_output_dir` | `Path` |
| `restart_manager_path` | `Path | None` |
| `max_epochs` | `int` |
| `batches_per_epoch` | `int | None` |

## 3. Current file shape

The current file effectively contains:

1. one outer checkpoint object
2. one outer live object
3. one inner checkpoint object
4. one phase-local helper state
5. one inner live object

## 4. If reduced to only two states

If the target is really only:

1. `train_all_phases_state`
2. `train_one_phase_state`

then the main current split to collapse is:

- merge or remove the separate checkpoint-vs-live naming layers
- decide whether `TrainingPhaseState` remains separate or is absorbed into the
  `train_one_phase_state`
- decide whether the phase-local counters stay as a separate helper state or move fully into `train_one_phase_state`
