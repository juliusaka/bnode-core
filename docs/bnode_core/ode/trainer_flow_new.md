# Training Flow Reference: `trainer.py` (HEAD of `modelica_export_copilot`, commit 2ab3859)

> **Source file:** `bnode/bnode-core/src/bnode_core/ode/trainer.py` — 1734 lines.  
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

### `main()` (line ~1708)

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
- Training: `train_cfg.batch_size` (or `batch_size_test` for test jobs)
- Validation/test: `4 × train_cfg.batch_size` (no backprop)
- Final test job: `cfg.nn_model.training.batch_size_test` for all contexts

After a CUDA OOM, `batch_size_reduction_factor` (a compounding factor starting at 1.0, multiplied by 0.7 each retry) is passed to `_create_datasets_and_dataloaders_for_job`, which reduces both training and validation batch sizes by that factor (minimum 1). The config is **not** mutated.

`torch.utils.data.DataLoader` is created for each context with:
- `shuffle=True`, `drop_last=True` for training jobs
- `shuffle=False`, `drop_last=False` for test jobs
- `pin_memory=True`, `persistent_workers=True` (when workers > 0)
- `collate_fn=timeseries_collate_fn` — custom collation for variable-length time-series batches

**Sequence length** is read back from the constructed dataset and stored in `job['train_cfg'].seq_len_train` (uses `datasets['train'].seq_len` if present, otherwise `datasets['train'].datasets['time'].shape[-1]`).

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

When resuming at a test job (i.e., a restart bundle exists and `job_idx` points to a test job), and no explicit load path is configured, the model weights are loaded directly from the `restart_model_state` dict in the bundle.

### 2.3 Optimizer — `_create_phase_optimizer()`

Determined by `train_cfg.optimizer` (case-insensitive):

| Value | Optimizer | Key parameters |
|---|---|---|
| `'adam'` | `torch.optim.Adam` | `lr_start`, `weight_decay`, `beta1_adam`, `beta2_adam` |
| `'lbfgs'` | `torch.optim.LBFGS` | `lr_start`, `lbfgs_max_iter`, `lbfgs_history_size`, `lbfgs_tolerance_grad`, `lbfgs_tolerance_change`, `lbfgs_line_search_fn` |

If `pre_train=False` and `train_cfg.reload_optimizer=True`, the optimizer state from the previous phase's checkpoint (`filepath_optimizer_current_hydra_output(job_idx-1)`) is loaded and the learning rate is reset to `train_cfg.lr_start`. Failure is caught and logged as a warning (does not abort).

### 2.4 Learning Rate Schedulers — `_create_phase_lr_schedulers()`

Schedulers are only created for main-training phases (`pre_train=False`, `test=False`) when `train_cfg.use_lr_scheduler=True`.

| `lr_scheduler_type` | Scheduler | Step trigger |
|---|---|---|
| `'cosine'` | `CosineAnnealingLR` | Per batch (in `train_one_epoch`) |
| `'plateau'` | `ReduceLROnPlateau` | Per epoch, using validation loss |

**Cosine scheduler**: `T_max` is either `train_cfg.cosine_T_max × batches_per_epoch` or `(max_epochs // 10) × batches_per_epoch`.  
**Plateau scheduler**: patience is either `train_cfg.plateau_patience` (if set) or auto-computed from `lr_on_plateau_iterations_to_min_lr()` capped at `early_stopping_patience/5`.

If no schedulers are configured, `lr_schedulers` is set to `None`.

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

**BNODE + pre-train guard**: if `cfg.nn_model.training.pre_train=True` and `cfg.nn_model.model_type='bnode'`, a `ValueError` is raised immediately at job-list construction — no silent failure.

Skip conditions:
- Pre-train job is skipped if `cfg.nn_model.training.pre_train=False`, or `load_pretrained_model=True`, or `load_trained_model_for_test=True`.
- Main-training jobs are skipped if `load_trained_model_for_test=True`.

### 3.2 `train_all_phases()` Main Loop

```
load_restart_checkpoint(job_list)  →  (outer_state, inner_state, scheduler_states,
                                        scaler_state, model_state, optimizer_state,
                                        checkpoint_store)
train_all_phases_state = outer_state or TrainAllPhasesState()   # always non-None

for idx, job in enumerate(job_list[job_start_idx:], start=job_start_idx):
    batch_size_reduction_factor = None
    while True:   # retry loop for CUDA OOM
        _create_datasets_and_dataloaders_for_job(... batch_size_reduction_factor=...)
        _initialize_or_reload_model_for_job(... restart_model_state=...)
        if job['skip']: log; continue
        if job['test'] is False:
            checkpoint_store.save_outer_for_test_job(...)   # NOT called for training jobs
            next_epoch_anchor = train_one_phase(...)
        else:
            train_all_phases_state.job_idx = idx
            checkpoint_store.save_outer_for_test_job(train_all_phases_state)
            _run_test_job(...)
        break
    except CheckpointRequestedExit: mlflow_proxy.set_tag; return
    except RuntimeError (CUDA OOM):
        oom_reduction_count += 1
        batch_size_reduction_factor = (batch_size_reduction_factor or 1.0) * 0.7
        warn if oom_reduction_count >= 30; sleep 10 s; retry

checkpoint_store.clear_restart_artifacts()
```

**CUDA OOM recovery**: on `RuntimeError` containing `'CUDA out of memory'` or `'CUDA memory is almost full'`, `batch_size_reduction_factor` is compounded by ×0.7. After 30+ retries, an additional warning is emitted. This factor is reset to `None` at the start of each new phase.

**`CheckpointRequestedExit`**: a `RuntimeError` subclass raised to trigger a graceful stop after checkpoint write. Caught here; an MLflow tag `'ended by' = 'checkpoint request'` is set and training returns.

**Phase sequencing**: after each training job completes, `seq_len_epoch_start` for the next job is set to the current job's `seq_len_train` (or `1` for the pre-train→first-phase transition).

**Cleanup**: after all jobs complete, `checkpoint_store.clear_restart_artifacts()` deletes the single bundle file.

### 3.3 Per-Phase Epoch Budget — `_compute_phase_epoch_settings()`

```
batches_per_epoch = len(train_loader)  if train_cfg.batches_per_epoch is None  else train_cfg.batches_per_epoch
epochs_for_seq_len_increase = int(seq_len_increase_in_batches / batches_per_epoch)   # floor
max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase
```

If `seq_len_epoch_start >= seq_len_train` (no ramp needed), `epochs_for_seq_len_increase = 0` and `train_cfg.seq_len_increase_in_batches` is zeroed.

---

## 4. Inner Training Loop

### 4.1 `train_one_phase()` Epoch Loop

```python
for epoch in range(epoch_0, phase_epoch_0 + max_epochs):
    if epoch == epoch_stop:
        break
    # --- termination checks ---
    flag_max_epoch = epoch == epoch_stop - 1
    flag_early_stopping = early_stopping.early_stop and flag_out_of_seq_len_increase
    flag_break_after_loss = early_stopping.best_score < train_cfg.break_after_loss_of  # if configured
    flag_nan_counter = phase_state.nan_counter > 50
    flag_break_after_epoch = False

    if any termination flag:
        model.load(path=path_best_model)
        flag_break_after_epoch = True

    # --- curriculum check ---
    if stable_epochs > abort_threshold and not flag_out_of_seq_len_increase:
        # collapse curriculum window and extend epoch_stop

    # --- training or eval-only pass ---
    if not flag_break_after_epoch and not first_epoch_is_evaluation:
        ret_vals_train = train_one_epoch(...)
        # NaN/Inf/AssertionError handling (see §9)
    else:
        ret_vals_train = test_or_validate_one_epoch(...)  # eval-only
        first_epoch_is_evaluation = False

    # --- validation, test, ref, testnorm ---
    ret_vals_validation = test_or_validate_one_epoch(..., all_batches=True)
    # EarlyStopping update, plateau scheduler step
    ret_vals_test = test_or_validate_one_epoch(..., all_batches=False)
    # ref/testnorm periodic

    # --- MLflow logging ---

    if flag_break_after_epoch:
        # log final metrics; break

    # --- per-epoch restart checkpoint (not written on terminal epoch) ---
    phase_state.phase_epoch = epoch + 1 - phase_epoch_0
    phase_state.rng_state = capture_rng_state(...)
    checkpoint_store.save_epoch_checkpoint(
        train_all_phases_state, phase_state, lr_schedulers, scaler, model, optimizer
    )
```

The very first epoch of a phase (`first_epoch_is_evaluation=True`) is always an evaluation pass (no training) to establish a baseline. On resume (`train_one_phase_state is not None`), this flag is `False` — the resumed run does NOT redo the baseline epoch.

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
For LBFGS: clipping is applied inside `_closure()`, and then again after the step purely to measure `_norm` for logging (no actual clipping effect on the already-stepped weights).  
The actual gradient norm `_norm` is returned and stored in `ret_vals_train['grad_norm']`.

### 4.5 CUDA Memory Guard

After every batch, if `use_cuda`:
- Logs `torch.cuda.memory_reserved() / (1024**3)` as `'CUDA_memory_reserved_GB'` to MLflow (correct GiB conversion).
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

Tries these keys in order and uses the first present: `'rmse_states_outputs'`, `'rmse_states'`, `'rmse_outputs'`. Returns `(None, None)` if none are present.  
The corresponding RMSE value is tracked alongside the primary validation loss as `corresponding_score` in `EarlyStopping`. The MLflow key `best_{metric_name}` is only logged when `early_stopping_metric_name is not None`.

---

## 6. Checkpointing / State Persistence

### 6.1 Model Checkpoints

Saved by `EarlyStopping.save_checkpoint()` calling `model.save(path)` when validation loss improves.

| File | `filepaths` function | When written |
|---|---|---|
| `model_phase_{i}.pt` | `filepath_model_current_hydra_output(job_idx)` | Every time validation loss improves during phase `i` |
| `model_pretrained.pt` | `filepath_pretrained_model_current_hydra_output()` | Every time validation loss improves during pre-training |
| `model_current.pt` | `filepath_model_current_hydra_output()` (no index) | After every non-NaN epoch (rolling; also inside restart bundle) |

### 6.2 Optimizer Checkpoints

Saved alongside model by `EarlyStopping.save_checkpoint()`.

| File | `filepaths` function | When written |
|---|---|---|
| `optimizer_phase_{i}.pt` | `filepath_optimizer_current_hydra_output(job_idx)` | On validation loss improvement |
| `optimizer_pretrained.pt` | `filepath_optimizer_current_hydra_output()` (pre-train) | On validation loss improvement |
| `optimizer_current.pt` | `filepath_optimizer_current_hydra_output()` (no index) | After every non-NaN epoch (also inside restart bundle) |

### 6.3 Restart Bundle — single atomic file

Written by `RestartCheckpointStore.save_epoch_checkpoint()` at the end of every non-terminal training epoch:

| File | `filepaths` function | Content |
|---|---|---|
| `training_restart_checkpoint.pt` | `filepath_restart_checkpoint_current_hydra_output()` | Single bundle dict (version 2) |

**Bundle structure** (`bundle_version=2`):

```python
{
    "bundle_version": 2,
    "outer": train_all_phases_state.to_state_dict(),   # TrainAllPhasesState
    "inner": train_one_phase_state.to_state_dict(),    # TrainOnePhaseState
    "scheduler": {name: scheduler.state_dict(), ...},  # or {}
    "scaler": scaler.state_dict(),
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
```

Atomicity: write to `.{name}.{uuid}.tmp`, `fsync`, `os.replace`, then `fsync` the directory.

On successful completion, `checkpoint_store.clear_restart_artifacts()` deletes the bundle file.

**`save_outer_for_test_job`**: before running a test job, the existing bundle is reloaded and the outer state is updated with the current `job_idx`. This ensures resume works even if interrupted during the test phase.

### 6.4 What is stored in `TrainAllPhasesState`

`torch.nn.Module` subclass (via `to_state_dict` / `load_from_state_dict`):
- `job_idx` — which job to resume from
- `next_epoch_anchor` — global epoch counter at phase start
- `mlflow_run_id` — used to validate MLflow run continuity on resume (via `mlflow.active_run()`)

### 6.5 What is stored in `TrainOnePhaseState`

`torch.nn.Module` subclass:
- `phase_epoch` — epoch within the current phase
- `nan_counter` — consecutive NaN loss count
- `grad_norm_last_reduced_counter` — NaN recovery counter for clip_grad_norm reduction
- `stable_epochs` — consecutive stable epochs for curriculum abort
- `deterministic_mode_active` — flag for BNODE
- `seq_len_increase_in_batches` — possibly-updated curriculum budget (captured after early abort)
- `rng_state` — complete RNG state snapshot (PyTorch CPU, PyTorch CUDA, NumPy, Python `random`), stored as a pickled uint8 tensor
- `early_stopping` — full `EarlyStopping` instance (via `phase_state.early_stopping` assignment)

### 6.6 Final Test Outputs

When `save_predictions_in_dataset=True`, predictions are written to an HDF5 file at `filepath_dataset_current_hydra_output()`. The file is created once and contexts are appended. Per-context datasets are created at the first batch, then filled with a write-offset loop.

---

## 7. Restart / Resume Logic

### 7.1 Detection at Startup

At the beginning of `train_all_phases()`:

```python
(train_all_phases_state, train_one_phase_state,
 restart_scheduler_states, restart_scaler_state,
 restart_model_state, restart_optimizer_state,
 checkpoint_store) = load_restart_checkpoint(job_list=job_list)
train_all_phases_state = train_all_phases_state or TrainAllPhasesState()
```

`load_restart_checkpoint` (from `restart_utils.py`):
1. `RestartCheckpointStore.from_current_hydra_output()` creates the store pointing to `training_restart_checkpoint.pt`.
2. `checkpoint_store.load_checkpoint_if_available()` checks if the bundle exists.
   - If not → returns `(None, None, None, None, None, None)` → fresh start.
   - If yes → loads bundle, validates version (must be 2), reconstructs `TrainAllPhasesState` and `TrainOnePhaseState`.
3. `_validate_restart_run_id()` checks that the active MLflow run matches the run ID stored in the outer state.
4. `_validate_restart_target()` checks that `job_idx` is within the job list bounds and the target job is not a pre-train phase.

### 7.2 Outer Loop Resume

If `train_all_phases_state.job_idx > 0` (restored from bundle):
```python
job_start_idx = train_all_phases_state.job_idx
next_epoch_anchor = train_all_phases_state.next_epoch_anchor
```
The outer `enumerate` starts from `job_start_idx`, skipping completed phases entirely.

`train_one_phase_state` (from bundle) is passed to `train_one_phase()` only for the first resumed job (`idx == job_start_idx`). After that, it is set to `None`.

Similarly, `restart_scheduler_states`, `restart_scaler_state`, `restart_model_state`, `restart_optimizer_state` are passed only for the first resumed job and then cleared.

### 7.3 Inner Loop Resume — inside `train_one_phase()`

If `train_one_phase_state is not None`:

1. Validates that `restart_model_state` and `restart_optimizer_state` are not `None` (raises `ValueError` otherwise).
2. `model.load_state_dict({k: v.to(device) for k, v in restart_model_state.items()})` — loads model weights from bundle.
3. `optimizer.load_state_dict(restart_optimizer_state)` — loads optimizer state from bundle.
4. Scheduler states validated (key-set must match) and loaded from `restart_scheduler_states`.
5. `scaler.load_state_dict(restart_scaler_state)` if not `None`.
6. `_bundle = torch.load(checkpoint_store.checkpoint_path, ..., weights_only=False)` reloads the full bundle; `phase_state.load_from_state_dict(_bundle["inner"])` restores `TrainOnePhaseState` fields.
7. `restore_rng_state(phase_state.rng_state, use_cuda=cfg.use_cuda)` restores all RNG states.

Epoch loop resumes at:
```python
epoch_0 = train_all_phases_state.next_epoch_anchor   # global epoch counter
phase_epoch_0 = epoch_0 - phase_state.phase_epoch    # phase-local start
epoch range = [epoch_0, phase_epoch_0 + max_epochs)
first_epoch_is_evaluation = False  # no baseline re-run on resume
```

### 7.4 Test Job Resume

When the bundle's `job_idx` points to a test job:
- `_validate_restart_target()` allows this (only pre-train jobs are rejected).
- `_initialize_or_reload_model_for_job()` receives `restart_model_state` and loads weights from it when `job.test=True` and no explicit `load_trained_model_for_test` path is configured.

### 7.5 Limitations / Design Notes

- Restart only works for **main-training and test phases**. Pre-training jobs cannot be restarted mid-run.
- The `CheckpointRequestedExit` exception is a hook for external job-scheduler integration: raise it to cleanly stop training after persisting state.
- LR scheduler type mismatch on resume is fatal (`ValueError`).
- `nan_counter` and `stable_epochs` persist across restarts (intentional, not documented in inline comments).

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

Linear interpolation from `seq_len_epoch_start` to `seq_len_train` over `seq_len_increase_in_batches` batches.

### 8.5 Epoch Budget — `_compute_phase_epoch_settings()`

```python
epochs_for_seq_len_increase = int(seq_len_increase_in_batches / batches_per_epoch)  # floor
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

Once `batches_this_phase > seq_len_increase_in_batches`, `flag_out_of_seq_len_increase = True` and early stopping becomes active.

---

## 9. Early Stopping / Convergence Criteria

### `EarlyStopping` Class (`bnode_core.nn.nn_utils.early_stopping`)

Instantiated at the start of each phase:
```python
early_stopping = EarlyStopping(
    patience=train_cfg.early_stopping_patience,
    verbose=True,
    threshold=train_cfg.early_stopping_threshold,
    threshold_mode=train_cfg.early_stopping_threshold_mode,   # 'abs' or 'rel'
    path=path_best_model,
    optimizer_path=path_optimizer_best_model,
    trace_func=logging.info,
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
| `flag_max_epoch` | `epoch == epoch_stop - 1` | `'max epochs'` |
| `flag_early_stopping` | `early_stopping.early_stop and flag_out_of_seq_len_increase` | `'early stopping'` |
| `flag_break_after_loss` | `early_stopping.best_score < train_cfg.break_after_loss_of` | `'break after loss'` |
| `flag_nan_counter` | `phase_state.nan_counter > 50` | `'50 NaNs in loss'` |

When any flag is set, `model.load(path_best_model)` is called before the final evaluation pass, then the loop breaks.

### NaN/Inf Loss Recovery

If `ret_vals_train['loss']` is NaN or Inf (or `AssertionError` was caught):
1. If `train_cfg.reload_model_if_loss_nan=True` and `nan_counter <= 49`:
   - Reload model and optimizer from `checkpoint_store.checkpoint_path` bundle (`_bundle["model"]`, `_bundle["optimizer"]`).
   - Increment `grad_norm_last_reduced_counter`. If > 2, multiply `clip_grad_norm` by 0.7 and reset counter.
   - Log `loss_nan_reload=1` to MLflow.
   - If bundle load itself fails (no checkpoint exists yet): raise `ValueError` (first epoch unstable).
2. If `nan_counter > 49`:
   - Reload the **best** model (`model.load(path_best_model)`) and optimizer (`torch.load(path_optimizer_best_model)`).
3. If `nan_counter > 55`: raise `ValueError` and abort.
4. If `reload_model_if_loss_nan=False`: log warning and continue with current model.

On a clean (non-NaN) epoch: reset `nan_counter` and `grad_norm_last_reduced_counter` to 0.  
**Note**: `model_current.pt` and `optimizer_current.pt` are no longer written separately on clean epochs — the rolling state is persisted only via the end-of-epoch restart bundle.

### `ReduceLROnPlateau` Integration

Stepped per epoch using validation loss:
```python
if lr_schedulers and 'plateau' in lr_schedulers:
    val_loss = ret_vals_validation.get('loss', None)
    if val_loss is not None and not (np.isnan(val_loss) or np.isinf(val_loss)):
        lr_schedulers['plateau'].step(val_loss)
```

NaN/Inf validation loss skips the scheduler step.

### `KeyboardInterrupt` Handling

The entire epoch loop is wrapped in:
```python
try:
    for epoch in range(...):
        ...
except KeyboardInterrupt:
    mlflow_proxy.set_tag_if_active('ended by', 'keyboard interrupt')
    model.load(path=path_best_model, device=device)  # fallback loop if file missing
```

---

## 10. Notable Helper Classes, Dataclasses, and Utility Modules

### `TrainAllPhasesState` (`restart_state.py`)

Serializes the coarse state of `train_all_phases()` via `to_state_dict` / `load_from_state_dict`:
- `job_idx`, `next_epoch_anchor`, `mlflow_run_id`.

### `TrainOnePhaseState` (`restart_state.py`)

Serializes the fine state of `train_one_phase()`:
- `phase_epoch`, `nan_counter`, `grad_norm_last_reduced_counter`, `stable_epochs`, `deterministic_mode_active`, `seq_len_increase_in_batches`, `rng_state`.
- `early_stopping` is assigned as a Python attribute (not serialized inside the state dict; held directly on the instance).

### `RestartCheckpointStore` (`restart_checkpoint_store.py`)

Manages atomic persistence of the single restart bundle. Key methods:

| Method | Action |
|---|---|
| `from_current_hydra_output()` | Factory using `filepaths` to locate Hydra output dir |
| `load_checkpoint_if_available()` | Load bundle if present; validate version 2; return all 6 components |
| `save_epoch_checkpoint(...)` | Atomic write of complete bundle (outer + inner + scheduler + scaler + model + optimizer) |
| `save_outer_for_test_job(...)` | Reload existing bundle, update outer state, re-save atomically |
| `clear_restart_artifacts()` | Delete the bundle file on clean completion |

Atomicity: write to `.{name}.{uuid}.tmp`, `fsync`, `os.replace`, then `fsync` the directory.

### `EarlyStopping` (`bnode_core.nn.nn_utils.early_stopping`)

`torch.nn.Module` subclass. Monitors validation loss with configurable patience and threshold. Key attributes: `counter`, `best_score`, `corresponding_score`, `early_stop`. Serializable via `get_extra_state()` / `set_extra_state()`. `reset()` resets all state; `reset_counter()` resets only the patience counter (preserves `best_score`).

### `CheckpointRequestedExit` (`restart_state.py`)

Simple `RuntimeError` subclass. Raised to request a graceful training stop after the epoch checkpoint is persisted. Caught in `train_all_phases()`.

### `mlflow_proxy` (`bnode_core.utils.mlflow_proxy`)

A thin wrapper around the MLflow client that silently handles the case where no MLflow run is active. Exposes `log_param`, `log_metric`, `log_metrics`, `set_tag_if_active`. Used throughout trainer.py as a drop-in for raw `mlflow.*` calls.

One bare `mlflow.active_run()` call remains in `train_one_phase` (line ~1679) to read the run ID for `TrainAllPhasesState.mlflow_run_id`. It is guarded: `_active_run = mlflow.active_run(); train_all_state.mlflow_run_id = _active_run.info.run_id if _active_run is not None else None`.

### `@log_hydra_to_mlflow` (`bnode_core.utils.hydra_mlflow_decorator`)

Decorator applied to `train_all_phases`. Starts/resumes an MLflow experiment run and logs the full Hydra config as MLflow parameters. Ensures the run is ended on normal return or exception.

### `filepaths` module (`bnode_core.filepaths`)

Central path-resolution module. Relevant functions:

| Function | Returns |
|---|---|
| `dir_current_hydra_output()` | Hydra's working output directory for the current run |
| `filepath_model_current_hydra_output(phase)` | `model_phase_{phase}.pt` or `model_current.pt` |
| `filepath_pretrained_model_current_hydra_output()` | `model_pretrained.pt` |
| `filepath_optimizer_current_hydra_output(phase)` | `optimizer_phase_{phase}.pt` or `optimizer_current.pt` |
| `filepath_restart_checkpoint_current_hydra_output()` | `training_restart_checkpoint.pt` |
| `filepath_dataset_current_hydra_output()` | HDF5 prediction output file |
| `filepath_from_local_or_ml_artifacts(path)` | Resolves `mlflow://` URIs or local paths |

### `load_dataset_and_config`, `make_stacked_dataset`, `TimeSeriesDataset`, `timeseries_collate_fn` (`bnode_core.nn.nn_utils.load_data`)

- `load_dataset_and_config(name, path)` → opens and returns an `h5py.File` handle.
- `make_stacked_dataset(hdf5, context, load_seq_len, seq_len_batches, stride, max_samples)` → constructs a `TimeSeriesDataset`.
- `TimeSeriesDataset` → standard PyTorch `Dataset`; each item is a dict of tensors.
- `timeseries_collate_fn` → collation function for variable-length padding.

### `NeuralODE` / `BalancedNeuralODE` (model classes)

Both expose the same training interface:
- `model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, *, return_model_outputs, test, last_batch, activate_deterministic_mode)` → returns metrics dict.
- `normalization_init(hdf5_dataset)` → initializes normalization buffers.
- `model.save(path)` / `model.load(path, device)` → save/load model state.
- `model.get_progress_string(ret_vals_train, ret_vals_validation, ret_vals_test, pre_train)` → compact string for per-epoch logging.
- `model.ode_fun_count` (optional) → ODE evaluations during last backward pass.

---

## Notable TODOs and Design Gaps

1. **`batch_size_test` mutation removed** — OOM retry now uses `batch_size_reduction_factor` local variable; global config is no longer mutated.
2. **TODO in `_create_datasets_and_dataloaders_for_job`** (line ~468): `# TODO: I believe this is never reached` — the guard for an empty dataset during test-only mode may be dead code.
3. **LR scheduler type mismatch on resume is fatal** — if config is changed between an interrupted run and a resume attempt, the scheduler key validation raises `ValueError` with no migration path.
4. **`seq_len_increase_abort_after_n_stable_epochs` epoch_stop extension** — when the curriculum is aborted early, `epoch_stop` is extended permanently. This extension is not reset if `stable_epochs` subsequently drops back to zero.
5. **`nan_counter` and `stable_epochs` persist across restarts** — intentional but not documented inline. Resuming a run that had 30 NaN epochs will start with `nan_counter=30`.
6. **`model_current.pt` / `optimizer_current.pt` no longer written separately** — rolling state is only in the restart bundle. If the bundle is deleted manually, NaN recovery cannot load a rolling checkpoint.
