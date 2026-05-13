# Training Flow Reference: `trainer.py` (HEAD of `modelica_export_copilot`)

> **Source file:** `/tmp/trainer_new.py` — 1685 lines.  
> This document is a complete, stand-alone reference for anyone who has not read the code.  
> It covers all ten requested aspects in order.

---

## Table of Contents

1. [Top-Level Entry Point](#1-top-level-entry-point)
2. [Initialization: Dataset, Model, Optimizer, Scheduler](#2-initialization)
3. [Outer Training Loop: Phases and Curriculum](#3-outer-training-loop)
4. [Inner Training Loop: Batch Iteration and Backprop](#4-inner-training-loop)
5. [Validation / Evaluation Logic](#5-validation--evaluation-logic)
6. [Checkpointing / State Persistence](#6-checkpointing--state-persistence)
7. [Restart / Resume Logic](#7-restart--resume-logic)
8. [Sequence-Length Curriculum](#8-sequence-length-curriculum)
9. [Early Stopping / Convergence Criteria](#9-early-stopping--convergence-criteria)
10. [Notable Helper Classes, Dataclasses, and Utility Modules](#10-notable-helper-classes-dataclasses-and-utility-modules)

---

## 1. Top-Level Entry Point

### `main()` (line ~1637)

```
trainer [config_overrides]          # via uv run / installed console-script
python -m bnode_core.ode.trainer    # direct module invocation
```

`main()` performs three setup steps then delegates entirely to Hydra:

1. **Config store registration** — calls `get_config_store()` (from `bnode_core.config`) to register all Hydra-structured-config dataclasses.
2. **Config directory auto-detection** — calls `filepaths.config_dir_auto_recognize()`, which chooses either the superproject `config/` tree or the package's `resources/config/`, depending on the current working directory.
3. **Hydra launch** — wraps `train_all_phases` with `hydra.main(config_path=..., config_name='train_test_ode', version_base=None)` and immediately calls the resulting wrapper.

### `@log_hydra_to_mlflow` decorator on `train_all_phases`

Before `train_all_phases` runs, the decorator starts an MLflow run, logs the resolved Hydra config as parameters and tags, and ensures the run is properly closed (including on exception).

### Config type

The Hydra config is validated into `train_test_config_class` (from `bnode_core.config`).  
Key top-level fields:

| Field | Purpose |
|---|---|
| `dataset_name` / `dataset_path` | Primary HDF5 dataset identifier |
| `dataset_norm_name` / `dataset_norm_path` | Optional separate normalization dataset |
| `dataset_ref_name` / `dataset_ref_path` | Optional reference dataset for periodic comparison |
| `nn_model.model_type` | `'node'` or `'bnode'` |
| `nn_model.training.pre_train` | Enable pre-training (NODE only) |
| `nn_model.training.main_training` | List of `base_training_settings_class` objects, one per phase |
| `nn_model.training.test` | Whether to run a final test job |
| `use_cuda` | Prefer GPU if available |
| `use_amp` | Enable automatic mixed precision |
| `n_workers_train_loader` / `n_workers_other_loaders` | DataLoader worker counts |
| `batch_print_interval` | How often to log per-batch stats |

---

## 2. Initialization

### 2.1 Dataset Loading — inside `train_all_phases()`

Three HDF5 datasets are optionally opened:

```python
hdf5_dataset, _         = load_dataset_and_config(cfg.dataset_name, cfg.dataset_path)
hdf5_dataset_norm, _    = load_dataset_and_config(cfg.dataset_norm_name, cfg.dataset_norm_path)   # optional
hdf5_dataset_ref, _     = load_dataset_and_config(cfg.dataset_ref_name, cfg.dataset_ref_path)     # optional
```

`load_dataset_and_config` is from `bnode_core.nn.nn_utils.load_data`. It resolves the HDF5 file path via `filepaths.filepath_dataset_from_config` and returns an open `h5py.File` handle.

For each training/test job, `_create_datasets_and_dataloaders_for_job()` is called. It creates `TimeSeriesDataset` objects (via `make_stacked_dataset`) for six contexts:

| Context key | Description |
|---|---|
| `'train'` | Training split, stride=1 |
| `'validation'` | Validation split, strided |
| `'test'` | Test split, strided |
| `'common_test'` | Cross-scenario test split |
| `'testnorm'` | Test split from `hdf5_dataset_norm` (or `None`) |
| `'ref'` | Full test split from `hdf5_dataset_ref` (or `None`) |

Batch sizes:
- Training: `train_cfg.batch_size`  
- Validation/test: `4 × train_cfg.batch_size` (because no backprop)
- Final test job: `cfg.nn_model.training.batch_size_test` for all contexts

`torch.utils.data.DataLoader` is created for each context with:
- `shuffle=True`, `drop_last=True` for training jobs
- `shuffle=False`, `drop_last=False` for test jobs
- `pin_memory=True`, `persistent_workers=True` (when workers > 0)
- `collate_fn=timeseries_collate_fn` — custom collation for variable-length time-series batches

**Sequence length** is read back from the constructed dataset and stored in `job['train_cfg'].seq_len_train`.

### 2.2 Model Construction — `initialize_model()`

Reads `cfg.nn_model.model_type` and branches:

**NODE path** (`model_type == 'node'`):
```python
model = NeuralODE(
    states_dim, controls_dim, parameters_dim, outputs_dim,
    controls_to_output_nn, hidden_dim, n_layers,
    hidden_dim_output_nn, n_layers_output_nn,
    activation=eval(cfg.nn_model.network.activation),
    intialization, initialization_ode,
    use_input_smoother,
)
model.normalization_init(hdf5_dataset)
```

**BNODE path** (`model_type == 'bnode'`):
```python
feedthrough_mask = build_feedthrough_mask(control_names, cfg.nn_model.network.feedthrough_controls, controls_dim)
model = BalancedNeuralODE(
    states_dim, lat_states_mu_dim, parameters_dim, lat_parameters_dim,
    controls_dim, lat_controls_dim, outputs_dim,
    hidden_dim, n_layers,
    controls_to_decoder, predict_states,
    activation, initialization_type, initialization_type_ode, initialization_type_ode_matrix,
    lat_ode_type, include_params_encoder, params_to_state_encoder,
    params_to_control_encoder, params_to_decoder,
    controls_to_state_encoder,
    state_encoder_linear, control_encoder_linear, parameter_encoder_linear,
    ode_linear, decoder_linear,
    lat_state_mu_independent,
    use_input_smoother,
    feedthrough_controls_mask,
)
model.normalization_init(hdf5_dataset)
```

`normalization_init` computes running mean/std from HDF5 dataset statistics and stores them as non-trainable buffers inside the model. When `dataset_norm` is available, normalization is initialized from it instead of the primary dataset.

The model is placed on `device` with `model.to(device)`. The number of trainable parameters is logged.

**Special loading modes** (handled in `_initialize_or_reload_model_for_job()`):

| Config flag | Effect |
|---|---|
| `load_trained_model_for_test=True` | Load from `path_trained_model`, skip all training |
| `load_pretrained_model=True` | Load from `path_pretrained_model`, adjust `seq_len_epoch_start` |

### 2.3 Optimizer — `_create_phase_optimizer()`

Determined by `train_cfg.optimizer` (case-insensitive):

| Value | Optimizer | Key parameters |
|---|---|---|
| `'adam'` | `torch.optim.Adam` | `lr_start`, `weight_decay`, `beta1_adam`, `beta2_adam` |
| `'lbfgs'` | `torch.optim.LBFGS` | `lr_start`, `lbfgs_max_iter`, `lbfgs_history_size`, `lbfgs_tolerance_grad`, `lbfgs_tolerance_change`, `lbfgs_line_search_fn` |

If `pre_train=False` and `train_cfg.reload_optimizer=True`, the optimizer state from the previous phase's checkpoint (`filepath_optimizer_current_hydra_output(job_idx-1)`) is loaded and the learning rate is reset to `train_cfg.lr_start`.

### 2.4 Learning Rate Schedulers — `_create_phase_lr_schedulers()`

Schedulers are only created for main-training phases (`pre_train=False`, `test=False`) when `train_cfg.use_lr_scheduler=True`.

| `lr_scheduler_type` | Scheduler | Step trigger |
|---|---|---|
| `'cosine'` | `CosineAnnealingLR` | Per batch (in `train_one_epoch`) |
| `'plateau'` | `ReduceLROnPlateau` | Per epoch, using validation loss |

**Cosine scheduler**: `T_max` is either `train_cfg.cosine_T_max × batches_per_epoch` or `(max_epochs // 10) × batches_per_epoch`.  
**Plateau scheduler**: patience is either `train_cfg.plateau_patience` (if set) or auto-computed from `lr_on_plateau_iterations_to_min_lr()` capped at `early_stopping_patience/5`.

### 2.5 AMP Scaler

```python
scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_cuda and cfg.use_amp)
```

Used during Adam/standard-optimizer backward passes. LBFGS disables AMP entirely.

---

## 3. Outer Training Loop

### 3.1 Job List — `_build_job_list()`

Returns an ordered list of dicts, each describing one "job":

```python
[
  {'skip': <bool>, 'test': False, 'train_cfg': pre_training_cfg, 'pre_train': True},   # pre-train (optional)
  {'skip': <bool>, 'test': False, 'train_cfg': main_training[0], 'pre_train': False},  # phase 0
  {'skip': <bool>, 'test': False, 'train_cfg': main_training[1], 'pre_train': False},  # phase 1, …
  {'skip': False,  'test': True,  'train_cfg': main_training[-1],'pre_train': False},  # final test (optional)
]
```

Skip conditions:
- Pre-train job is skipped if `cfg.nn_model.training.pre_train=False`, or `load_pretrained_model=True`, or `load_trained_model_for_test=True`.
- Main-training jobs are skipped if `load_trained_model_for_test=True`.

### 3.2 `train_all_phases()` Main Loop

```
for idx, job in enumerate(job_list[job_start_idx:], start=job_start_idx):
    while True:   # inner retry loop for CUDA OOM errors
        _create_datasets_and_dataloaders_for_job(...)
        _initialize_or_reload_model_for_job(...)
        if job['skip']: continue
        if job['test'] is False:
            next_epoch_anchor = train_one_phase(...)
        else:
            _run_test_job(...)
        break  # exit retry loop on success
    except CheckpointRequestedExit:  return
    except RuntimeError (CUDA OOM):  reduce batch_size by 30%, retry
```

**CUDA OOM recovery**: on `RuntimeError` containing `'CUDA out of memory'` or `'CUDA memory is almost full'`, the batch size is reduced to 70% of its current value, caches are cleared, and the job is retried after 10 seconds.

**`CheckpointRequestedExit`**: raised by `RestartCheckpointStore` to gracefully stop after writing a checkpoint. Caught here; training terminates cleanly.

**Phase sequencing**: at the end of each training job, `seq_len_epoch_start` for the next job is set to the current job's `seq_len_train` (so the next phase starts with the same sequence length). For pre-training, this is set to `1`.

**Cleanup**: after all jobs complete, `_clear_restart_state()` removes the outer and inner restart checkpoint files.

### 3.3 Per-Phase Epoch Budget — `_compute_phase_epoch_settings()`

```
batches_per_epoch = len(train_loader)  if train_cfg.batches_per_epoch is None  else train_cfg.batches_per_epoch
epochs_for_seq_len_increase = ceil(seq_len_increase_in_batches / batches_per_epoch)  [if seq_len_epoch_start < seq_len_train]
max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase
```

The extra `epochs_for_seq_len_increase` epochs are added on top of `max_epochs` to accommodate the curriculum warm-up period. Early stopping is suppressed during this period (see §9).

---

## 4. Inner Training Loop

### 4.1 `train_one_phase()` Epoch Loop

```python
for epoch in range(epoch_0, phase_epoch_0 + max_epochs):
    # --- termination checks (break conditions) ---
    if flag_max_epoch or flag_early_stopping or flag_break_after_loss or flag_nan_counter:
        model.load(path=path_best_model)
        flag_break_after_epoch = True

    # --- first epoch of phase or end epoch: eval-only ---
    if first_epoch_is_evaluation or flag_break_after_epoch:
        ret_vals_train, ... = test_or_validate_one_epoch(...)   # no grad, single batch
    else:
        ret_vals_train, train_iter = train_one_epoch(...)

    # --- validation (all batches) ---
    ret_vals_validation = test_or_validate_one_epoch(..., all_batches=True)

    # --- single-batch test ---
    ret_vals_test, ... = test_or_validate_one_epoch(..., all_batches=False)

    # --- optional ref / testnorm (periodic) ---
    ...

    # --- LR scheduling, early stopping, MLflow logging ---
    ...

    # --- per-epoch restart checkpoint ---
    checkpoint_store.save_epoch_checkpoint(...)

    if flag_break_after_epoch: break
```

The very first epoch of a phase (`first_epoch_is_evaluation=True`) is always an evaluation pass (no training) to establish a baseline. This flag is cleared after that first epoch.

### 4.2 `train_one_epoch()` — Batch Loop

```python
for batch_idx in range(batches_per_epoch):
    data_batch, train_iter = _next_batch(train_loader, train_iter)
    # apply curriculum truncation (see §8)
    ...
    if not is_lbfgs:
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp and use_cuda):
            ret_vals_train = model.model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, ...)
        loss = ret_vals_train['loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:  # LBFGS
        def _closure():
            optimizer.zero_grad()
            out = model.model_and_loss_evaluation(...)
            out['loss'].backward()
            clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            return out['loss']
        loss = optimizer.step(_closure)
    # cosine LR step per batch
    if lr_schedulers and 'cosine' in lr_schedulers:
        lr_schedulers['cosine'].step()
```

### 4.3 Forward Pass / Loss — `model.model_and_loss_evaluation()`

This is a method on `NeuralODE` / `BalancedNeuralODE`. From `trainer.py`'s perspective it is a black box that:
- Receives `data_batch` (dict of tensors), `train_cfg`, `pre_train`, `device`, and flags `return_model_outputs`, `test`, `last_batch`, `activate_deterministic_mode`.
- Returns a dict of scalar metrics including at minimum `'loss'`. May also include `'rmse_states'`, `'rmse_outputs'`, `'rmse_states_outputs'`, `'ode_calls_forward'`, etc.
- For BNODE: reparameterization (variational sampling) is active during training (`test=False`) and disabled during evaluation (`test=True`) and deterministic mode.

**First epoch special case** (`epoch_this_phase in [0, 1]` and not pre-train): `train_cfg.evaluate_at_control_times = True` is forced on a deep copy of `train_cfg` to get a reliable memory-usage estimate.

### 4.4 Gradient Clipping

For Adam/standard optimizers: `clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)` after `scaler.unscale_()`.  
For LBFGS: clipping is applied inside `_closure()`.  
The actual gradient norm `_norm` is returned and stored in `ret_vals_train['grad_norm']`.

### 4.5 CUDA Memory Guard

After every batch, if `use_cuda`:
- Logs `torch.cuda.memory_reserved()` as `'CUDA_memory_reserved_GB'` to MLflow.
- If reserved memory exceeds 60% of total GPU memory (or projected memory at full sequence length would exceed 60%), raises `RuntimeError('CUDA memory is almost full')` which is caught in `train_all_phases()` to trigger batch-size reduction.
- Hard limit: if reserved > 98% of total, immediately raises.

### 4.6 Return Values from `train_one_epoch()`

In addition to the model's metric dict, the following are added:

| Key | Description |
|---|---|
| `'grad_norm'` | Gradient norm before clipping |
| `'clip_grad_norm'` | Current clip norm setting |
| `'seq_len_now'` | Effective sequence length used in this epoch |
| `'time_forward'` / `'time_backward'` / `'time_optimizer_step'` / `'time_loader'` / `'time_total'` | Timing breakdown |
| `'time_per_batch'` / `'time_per_batch_forward'` etc. | Per-batch timing |
| `'ode_calls_backward'` | ODE function evaluations during backward (from `model.ode_fun_count`) |

### 4.7 DataLoader Iterator — `_next_batch()`

A persistent iterator is maintained per context in `dataloader_iters` dict. `_next_batch()` advances the iterator, recreating it on `StopIteration`. This allows `batches_per_epoch` to be much larger than `len(train_loader)` (sampling with replacement across epochs).

---

## 5. Validation / Evaluation Logic

### `test_or_validate_one_epoch()`

Two modes controlled by `all_batches`:

**`all_batches=True`** (used for validation):
```python
model.eval()
for batch in data_loader:
    with torch.no_grad():
        ret_vals.append(model.model_and_loss_evaluation(..., test=True))
# metrics are averaged across all batches
ret_vals = {key: np.mean([x[key] for x in ret_vals]) for key in ret_vals[0]}
```

**`all_batches=False`** (used for test / testnorm / ref during training):
```python
model.eval()
data_batch, data_iter = _next_batch(data_loader, data_iter)
with torch.no_grad():
    ret_vals = model.model_and_loss_evaluation(..., test=True)
```
The persistent `data_iter` is returned so the caller can maintain position across epochs.

### When each context is evaluated per epoch

| Context | Frequency |
|---|---|
| `'train'` | Every epoch (eval pass on one batch) |
| `'validation'` | Every epoch, all batches |
| `'test'` | Every epoch, one batch |
| `'ref'` / `'testnorm'` | Every `cfg.nn_model.training.ref_and_testnorm_every_n_epochs` epochs, plus at first epoch and on break |

### Handling non-finite validation values

`AssertionError` with `'non-finite values in'` is caught for validation, test, ref, and testnorm passes. The metrics dict is filled with `float('nan')` and training continues. Other `AssertionError` subtypes are re-raised.

### `activate_deterministic_mode`

On the final break epoch, if `train_cfg.activate_deterministic_mode_after_this_phase=True`, the model's deterministic mode is activated before the evaluation pass. The resulting model is saved to `path_best_model`. This is relevant for BNODE models (disables latent sampling permanently).

### Early stopping metric selection — `_get_early_stopping_corresponding_metric()`

Tries these keys in order and uses the first present: `'rmse_states_outputs'`, `'rmse_states'`, `'rmse_outputs'`. The corresponding RMSE value is tracked alongside the primary validation loss as `corresponding_score` in `EarlyStopping`.

---

## 6. Checkpointing / State Persistence

### 6.1 Model Checkpoints

Saved by `EarlyStopping.save_checkpoint()` calling `model.save(path)` when validation loss improves.

| File | `filepaths` function | When written |
|---|---|---|
| `model_phase_{i}.pt` | `filepath_model_current_hydra_output(job_idx)` | Every time validation loss improves during phase `i` |
| `model_pretrained.pt` | `filepath_pretrained_model_current_hydra_output()` | Every time validation loss improves during pre-training |
| `model_current.pt` | `filepath_model_current_hydra_output()` (no index) | After every epoch without NaN (used for NaN recovery) |

### 6.2 Optimizer Checkpoints

Saved alongside model by `EarlyStopping.save_checkpoint()` via `torch.save(optimizer.state_dict(), optimizer_path)`.

| File | `filepaths` function | When written |
|---|---|---|
| `optimizer_phase_{i}.pt` | `filepath_optimizer_current_hydra_output(job_idx)` | On validation loss improvement |
| `optimizer_pretrained.pt` | `filepath_optimizer_current_hydra_output()` (pre-train) | On validation loss improvement |
| `optimizer_current.pt` | `filepath_optimizer_current_hydra_output()` (no index) | After every epoch without NaN |

### 6.3 Restart Checkpoint Files (per-epoch, atomic)

Written by `RestartCheckpointStore.save_epoch_checkpoint()` at the end of every training epoch (not test, not pre-train):

| File | `filepaths` function | Content |
|---|---|---|
| `training_outer_restart.pt` | `filepath_training_outer_restart_state_current_hydra_output()` | `TrainAllPhasesState` state dict |
| `training_inner_restart.pt` | `filepath_training_inner_restart_state_current_hydra_output()` | `TrainOnePhaseState` state dict |
| `lr_schedulers.pt` | `filepath_lr_schedulers_current_hydra_output()` | Dict of scheduler `state_dict`s, keyed by name |
| `grad_scaler.pt` | `filepath_grad_scaler_current_hydra_output()` | `GradScaler.state_dict()` |

All four writes are atomic (write to a UUID-named temp file, `os.fsync`, `os.replace`), with directory fsynced after each replacement. A shared UUID (`checkpoint_uuid`) links the outer and inner state files to guard against loading a mismatched pair.

After all jobs complete successfully, `_clear_restart_state()` deletes all four files.

### 6.4 What is stored in each file

**`TrainAllPhasesState`** (outer):
- `job_idx` — which job to resume from
- `next_epoch_anchor` — global epoch counter at phase start
- `mlflow_run_id` — used to validate MLflow run continuity on resume
- `checkpoint_uuid` — integrity cross-check with inner file

**`TrainOnePhaseState`** (inner):
- `phase_epoch` — epoch within the current phase
- `nan_counter` — consecutive NaN loss count
- `grad_norm_last_reduced_counter` — NaN recovery counter for clip_grad_norm reduction
- `stable_epochs` — consecutive stable epochs for curriculum abort
- `deterministic_mode_active` — flag for BNODE
- `seq_len_increase_in_batches` — possibly-updated curriculum budget
- `rng_state` — complete RNG state snapshot (PyTorch CPU, PyTorch CUDA, NumPy, Python `random`)
- `checkpoint_uuid`
- `early_stopping` — full `EarlyStopping` module state (via `get_extra_state` / `set_extra_state`)

**Serialization**: `TrainAllPhasesState` and `TrainOnePhaseState` are `torch.nn.Module` subclasses. Fields are registered as buffers so `state_dict()` / `load_state_dict()` handles them. Optional strings (MLflow run ID, UUID) are UTF-8 encoded to `torch.uint8` tensors. The RNG state is encoded field-by-field into tensors.

### 6.5 Final Test Outputs

When `save_predictions_in_dataset=True`, predictions are written to an HDF5 file at `filepath_dataset_current_hydra_output()`. The file is created once and contexts are appended. Per-context datasets are created at the first batch, then filled with a write-offset loop.

---

## 7. Restart / Resume Logic

### 7.1 Detection at Startup

At the beginning of `train_all_phases()`:

```python
train_all_phases_state, train_one_phase_state, outer_path, inner_path = load_restart_state_pair(job_list=job_list)
```

Inside `load_restart_state_pair()` (from `restart_utils.py`):
1. `RestartCheckpointStore.from_current_hydra_output()` instantiates paths.
2. `load_state_pair_if_available()` checks if both `training_outer_restart.pt` and `training_inner_restart.pt` exist.  
   - If neither exists → fresh start, returns `(None, None)`.
   - If only one exists → raises `ValueError` (corrupt state).
   - If both exist → loads both and validates UUID match.
3. `_validate_restart_run_id()` checks that the active MLflow run matches the run ID stored in the outer state.
4. `_validate_restart_target()` checks that:
   - `job_idx` is within the job list bounds.
   - The target job is a main-training job (not pre-train, not test).

### 7.2 Outer Loop Resume

If `train_all_phases_state is not None`:
```python
job_start_idx = train_all_phases_state.job_idx
next_epoch_anchor = train_all_phases_state.next_epoch_anchor
```
The outer `enumerate` starts from `job_start_idx`, skipping completed phases entirely.

`train_one_phase_state` is passed to `train_one_phase()` only for the first resumed job (subsequently set to `None`).

### 7.3 Inner Loop Resume — inside `train_one_phase()`

If `train_one_phase_state is not None`:

1. The `seq_len_increase_in_batches` is restored from the phase state (overrides the config value, which may have been modified during the prior run).
2. Model weights loaded from `path_current_model` (the per-epoch "current" checkpoint).
3. Optimizer state loaded from `path_current_optimizer`.
4. LR scheduler states loaded from `lr_schedulers.pt`; scheduler keys are validated to match.
5. GradScaler state loaded from `grad_scaler.pt`.
6. `phase_state.load(inner_restart_state_path)` restores `TrainOnePhaseState` fields (phase_epoch, counters, RNG state, early-stopping state).
7. `restore_rng_state(phase_state.rng_state, use_cuda=cfg.use_cuda)` restores all RNG states.

Epoch loop resumes at:
```python
epoch_0 = train_all_phases_state.next_epoch_anchor     # global epoch counter
phase_epoch_0 = epoch_0 - phase_state.phase_epoch      # phase-local epoch 0
epoch range = [epoch_0, phase_epoch_0 + max_epochs)
```

### 7.4 Validation that Restart Files Are Complete

```python
if not path_current_model.exists(): raise FileNotFoundError(...)
if not path_current_optimizer.exists(): raise FileNotFoundError(...)
if not path_current_lr_schedulers.exists(): raise FileNotFoundError(...)
if not path_current_grad_scaler.exists(): raise FileNotFoundError(...)
```

Explicit errors are raised if any required file is missing rather than silently reinitializing.

### 7.5 Limitations / Design Notes

- **TODO comment** (line ~599, `_initialize_or_reload_model_for_job`): `"Can't we reuse/differently split this function to use it also when resuming training?"` — the model initialization on resume is handled separately inside `train_one_phase()` rather than being unified with `_initialize_or_reload_model_for_job()`.
- Restart only works for **main-training phases**. Pre-training and test jobs cannot be restarted mid-run.
- The `CheckpointRequestedExit` exception is a hook for external job-scheduler integration: raise it to cleanly stop training after persisting state.

---

## 8. Sequence-Length Curriculum

### 8.1 Purpose

When transitioning to a longer sequence length (new phase or curriculum warm-up), the model is exposed to gradually increasing sequence lengths over `seq_len_increase_in_batches` batches. Validation always uses the full `seq_len_train`.

### 8.2 Configuration Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `train_cfg.seq_len_epoch_start` | `int` or `None` | Sequence length at start of this phase |
| `train_cfg.seq_len_train` | `int` or `None` | Target (full) sequence length |
| `train_cfg.seq_len_increase_in_batches` | `int` | Total batches over which to ramp up |
| `train_cfg.seq_len_increase_abort_after_n_stable_epochs` | `int` | Abort curriculum early after N stable epochs |

### 8.3 Phase Transition

After each training phase, `train_all_phases()` sets:
```python
job_list[idx + 1]['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train
```
So the next phase starts at the ending length of the previous phase.

### 8.4 Batch-Level Ramp — inside `train_one_epoch()`

```python
_batches_this_phase = epoch_this_phase * batches_per_epoch + batch_idx
if _batches_this_phase < train_cfg.seq_len_increase_in_batches:
    _seq_len_now = seq_len_epoch_start + int(
        _batches_this_phase / seq_len_increase_in_batches
        * (seq_len_train - seq_len_epoch_start)
    )
    _seq_len_now = min(_seq_len_now, seq_len_train)
    # truncate all 3D tensors in data_batch along the time axis:
    for key in data_batch:
        if len(data_batch[key].shape) == 3:
            data_batch[key] = data_batch[key][:, :, :_seq_len_now]
else:
    _seq_len_now = seq_len_train
```

Linear interpolation from `seq_len_epoch_start` to `seq_len_train` over `seq_len_increase_in_batches` batches. All 3-D tensors in the batch dict are truncated in-place along the time axis (axis 2).

### 8.5 Epoch Budget Inflation — `_compute_phase_epoch_settings()`

```python
epochs_for_seq_len_increase = ceil(seq_len_increase_in_batches / batches_per_epoch)
max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase
```

The extra epochs ensure that early stopping patience is not consumed during the warm-up window.

### 8.6 Early Abort of Curriculum — `flag_out_of_seq_len_increase`

```python
if (phase_state.stable_epochs > train_cfg.seq_len_increase_abort_after_n_stable_epochs
        and flag_out_of_seq_len_increase is False):
    # Collapse the remaining curriculum window to the current batch count
    train_cfg.seq_len_increase_in_batches = batches_per_epoch * (epoch - phase_epoch_0)
    # Extend epoch_stop by the same amount so max_epochs budget is preserved
    epoch_stop = phase_epoch_0 + train_cfg.max_epochs + (epoch - phase_epoch_0)
```

`phase_state.stable_epochs` is incremented when `loss_validation < 2 * loss_train` (model generalises well) and reset to 0 otherwise.

### 8.7 Post-Curriculum Flag

Once `batches_this_phase > seq_len_increase_in_batches`, `flag_out_of_seq_len_increase = True`. Only then does early stopping actually trigger phase termination (early stopping is active but ignored while the flag is False).

---

## 9. Early Stopping / Convergence Criteria

### `EarlyStopping` Class (`bnode_core.nn.nn_utils.early_stopping`)

Instantiated at the start of each phase:
```python
early_stopping = EarlyStopping(
    patience=train_cfg.early_stopping_patience,
    threshold=train_cfg.early_stopping_threshold,
    threshold_mode=train_cfg.early_stopping_threshold_mode,   # 'abs' or 'rel'
    path=path_best_model,
    optimizer_path=path_optimizer_best_model,
)
```

**Improvement criterion:**
- `'abs'` mode: `loss < best_score - threshold`
- `'rel'` mode: `loss < best_score * (1 - threshold)`

When improved: saves model + optimizer checkpoint, resets counter.  
When not improved: increments counter.  
When `counter >= patience`: sets `early_stop = True`.

**Counter reset after curriculum**: `early_stopping.reset_counter()` is called when `flag_out_of_seq_len_increase` transitions to True, so patience starts fresh at full sequence length.

### Phase Termination Conditions (checked at top of each epoch)

| Flag | Condition | MLflow tag |
|---|---|---|
| `flag_max_epoch` | `epoch == phase_epoch_0 + max_epochs - 1` | `'max epochs'` |
| `flag_early_stopping` | `early_stopping.early_stop and flag_out_of_seq_len_increase` | `'early stopping'` |
| `flag_break_after_loss` | `early_stopping.best_score < train_cfg.break_after_loss_of` | `'break after loss'` |
| `flag_nan_counter` | `phase_state.nan_counter > 50` | `'4 NaNs in loss'` (sic — actually 50)` |

When any flag is set, `model.load(path_best_model)` is called before the final evaluation pass, then the loop breaks.

### NaN/Inf Loss Recovery

If `ret_vals_train['loss']` is NaN or Inf (or `AssertionError` was caught):
1. If `train_cfg.reload_model_if_loss_nan=True` and `nan_counter <= 49`:
   - Reload `path_current_model` and `path_current_optimizer`.
   - Increment `grad_norm_last_reduced_counter`. If > 2, multiply `clip_grad_norm` by 0.7 and reset counter.
   - Log `loss_nan_reload=1` to MLflow.
2. If `nan_counter > 49`:
   - Reload the **best** model and optimizer (`path_best_model`).
3. If `nan_counter > 55`: raise `ValueError` and abort.
4. If `reload_model_if_loss_nan=False`: log warning and continue with current model.

On a clean (non-NaN) epoch: reset `nan_counter` and `grad_norm_last_reduced_counter` to 0, save model and optimizer to `path_current_model` / `path_current_optimizer`.

### `ReduceLROnPlateau` Integration

Stepped per epoch using validation loss:
```python
if lr_schedulers and 'plateau' in lr_schedulers:
    val_loss = ret_vals_validation.get('loss', None)
    if val_loss is not None and not (np.isnan(val_loss) or np.isinf(val_loss)):
        lr_schedulers['plateau'].step(val_loss)
```

NaN/Inf validation loss skips the scheduler step.

---

## 10. Notable Helper Classes, Dataclasses, and Utility Modules

### `TrainAllPhasesState` (`restart_state.py`)

`torch.nn.Module` subclass. Persists the coarse state of `train_all_phases()`:  
`job_idx`, `next_epoch_anchor`, `mlflow_run_id`, `checkpoint_uuid`.  
All fields are serialized as typed PyTorch buffers. Optional strings use a UTF-8 → `uint8` tensor encoding. Version field `_state_version = 1` guards against loading stale files.

### `TrainOnePhaseState` (`restart_state.py`)

`torch.nn.Module` subclass. Persists the fine state of `train_one_phase()`:  
`phase_epoch`, `nan_counter`, `grad_norm_last_reduced_counter`, `stable_epochs`, `deterministic_mode_active`, `seq_len_increase_in_batches`, `rng_state`, `checkpoint_uuid`, `early_stopping`.  
The complete RNG state (PyTorch CPU + CUDA, NumPy, Python `random`) is encoded/decoded to/from tensors via `capture_rng_state()` / `restore_rng_state()`.

### `RestartCheckpointStore` (`restart_checkpoint_store.py`)

Manages atomic persistence of all four restart artifacts. Key methods:

| Method | Action |
|---|---|
| `from_current_hydra_output()` | Factory using `filepaths` to locate Hydra output dir |
| `load_state_pair_if_available()` | Load + UUID-validate outer+inner states |
| `save_epoch_checkpoint(...)` | Atomic write of all four files in dependency order |
| `clear_restart_artifacts()` | Delete all four files on clean completion |

Atomicity: write to `.{name}.{uuid}.tmp`, `fsync`, `os.replace`, then `fsync` the directory. The inner state is written before the outer state so a crash mid-write cannot produce a valid but stale outer state.

### `EarlyStopping` (`bnode_core.nn.nn_utils.early_stopping`)

`torch.nn.Module` subclass. Monitors validation loss with configurable patience and threshold. Key attributes: `counter`, `best_score`, `corresponding_score`, `early_stop`, `score_last_save`. Serializable via `get_extra_state()` / `set_extra_state()` for inclusion in `TrainOnePhaseState`. `reset()` resets all state; `reset_counter()` resets only the patience counter (preserves `best_score`).

### `CheckpointRequestedExit` (`restart_state.py`)

Simple `RuntimeError` subclass. Raised (typically by external tooling interacting with `RestartCheckpointStore`) to request a graceful training stop after the epoch checkpoint is persisted. Caught in `train_all_phases()` which returns immediately.

### `mlflow_proxy` (`bnode_core.utils.mlflow_proxy`)

A thin wrapper around the MLflow client that silently handles the case where no MLflow run is active. Used throughout trainer.py as a drop-in for `mlflow.*` calls. Exposes `log_param`, `log_metric`, `log_metrics`, `set_tag_if_active`.

### `@log_hydra_to_mlflow` (`bnode_core.utils.hydra_mlflow_decorator`)

Function decorator applied to `train_all_phases`. Starts/resumes an MLflow experiment run and logs the full Hydra config (flat key-value pairs) as MLflow parameters. Ensures the run is ended on normal return or exception.

### `filepaths` module (`bnode_core.filepaths`)

Central path-resolution module. Relevant functions used in trainer:

| Function | Returns |
|---|---|
| `dir_current_hydra_output()` | Hydra's working output directory for the current run |
| `filepath_model_current_hydra_output(phase)` | `model_phase_{phase}.pt` or `model_current.pt` |
| `filepath_pretrained_model_current_hydra_output()` | `model_pretrained.pt` |
| `filepath_optimizer_current_hydra_output(phase)` | `optimizer_phase_{phase}.pt` or `optimizer_current.pt` |
| `filepath_lr_schedulers_current_hydra_output()` | `lr_schedulers.pt` |
| `filepath_grad_scaler_current_hydra_output()` | `grad_scaler.pt` |
| `filepath_training_outer_restart_state_current_hydra_output()` | `training_outer_restart.pt` |
| `filepath_training_inner_restart_state_current_hydra_output()` | `training_inner_restart.pt` |
| `filepath_dataset_current_hydra_output()` | HDF5 prediction output file |
| `filepath_from_local_or_ml_artifacts(path)` | Resolves `mlflow://` URIs or local paths |

### `load_dataset_and_config`, `make_stacked_dataset`, `TimeSeriesDataset`, `timeseries_collate_fn` (`bnode_core.nn.nn_utils.load_data`)

- `load_dataset_and_config(name, path)` → opens and returns an `h5py.File` handle.
- `make_stacked_dataset(hdf5, context, load_seq_len, seq_len_batches, stride, max_samples)` → constructs a `TimeSeriesDataset` by slicing and stacking time-series trajectories.
- `TimeSeriesDataset` → standard PyTorch `Dataset`; each item is a dict of tensors (`states`, `controls`, `parameters`, `outputs`, optionally `state_der`).
- `timeseries_collate_fn` → collation function handling variable-length padding.

### `NeuralODE` / `BalancedNeuralODE` (model classes)

Both expose the same training interface contract:
- `model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, *, return_model_outputs, test, last_batch, activate_deterministic_mode)` → returns metrics dict (+ optional model-output dict).
- `normalization_init(hdf5_dataset)` → initializes normalization buffers.
- `model.save(path)` / `model.load(path, device)` → save/load model state.
- `model.get_progress_string(ret_vals_train, ret_vals_validation, ret_vals_test, pre_train)` → compact string for per-epoch logging.
- `model.ode_fun_count` (optional) → ODE evaluations during last backward pass.

`build_feedthrough_mask(control_names, feedthrough_controls, controls_dim)` constructs a boolean mask for `BalancedNeuralODE` from human-readable control variable names.

### `lr_on_plateau_iterations_to_min_lr` (`bnode_core.nn.nn_utils.lr_scheduler`)

Helper that analytically computes how many plateau-patience steps are needed for the LR to decay from `lr_start` to `lr_min` given a multiplicative `factor`. Used to auto-compute plateau patience when `train_cfg.plateau_patience` is `None`.

---

## Notable TODOs and Design Gaps

1. **TODO in `_create_datasets_and_dataloaders_for_job`** (line ~490):  
   `# TODO: I believe this is never reached` — the guard for an empty dataset during test-only mode may be dead code.

2. **TODO on model reload and resume** (line ~599):  
   `"Can't we reuse/differently split this function to use it also when resuming training?"` — model initialization on resume is done inside `train_one_phase()` rather than through the shared `_initialize_or_reload_model_for_job()`.

3. **MLflow tag typo** (line ~1430):  
   `mlflow_proxy.set_tag_if_active('job {} ended by'.format(job_idx), '4 NaNs in loss')` — the message says "4 NaNs" but the threshold is 50.

4. **Pre-train restart not supported** — `_validate_restart_target()` explicitly rejects restart if `target_job["pre_train"]` is True.

5. **Test job restart not supported** — same restriction for test jobs.

6. **LR scheduler type mismatch on resume is fatal** — if the config is changed between runs (different `lr_scheduler_type`), the scheduler key validation will raise `ValueError`, preventing resume. There is no migration path.

7. **`batch_size_test` is a global field** — during CUDA OOM on test jobs, `cfg.nn_model.training.batch_size_test` is mutated in-place. This is a side effect on the global config.

8. **`seq_len_increase_abort_after_n_stable_epochs` epoch_stop extension** — when the curriculum is aborted early, `epoch_stop` is extended by `(epoch - phase_epoch_0)` batches/epochs to compensate. This expansion is not reset if `stable_epochs` drops back to zero.

9. **Cosine scheduler `T_max` defaults to `max_epochs // 10`** — this means a very short cosine period relative to total training. The comment in the code notes this but does not document the rationale.

10. **`evaluate_at_control_times` deep copy** — the `train_cfg` deep copy in the first two epochs is intentional (to avoid permanently modifying the config), but adds overhead.
