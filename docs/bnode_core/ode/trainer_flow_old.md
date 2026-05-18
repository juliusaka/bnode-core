# Training Flow Reference: `trainer.py` (git commit d6ffb64)

> **Purpose.** This document is a complete, self-contained reference for the training
> pipeline as implemented in `bnode_core/ode/trainer.py` at git commit `d6ffb64`.
> It is written for a reader who has never seen the code. Every function name, class
> name, variable, and file-path helper is quoted exactly as it appears in the source.

---

## Table of Contents

1. [Top-level entry point](#1-top-level-entry-point)
2. [Initialization](#2-initialization)
3. [Outer training loop structure](#3-outer-training-loop-structure)
4. [Inner training loop](#4-inner-training-loop)
5. [Validation / evaluation logic](#5-validation--evaluation-logic)
6. [Checkpointing / state persistence](#6-checkpointing--state-persistence)
7. [Restart / resume logic](#7-restart--resume-logic)
8. [Sequence-length curriculum](#8-sequence-length-curriculum)
9. [Early stopping / convergence criteria](#9-early-stopping--convergence-criteria)
10. [Helper classes, dataclasses, and utility modules](#10-helper-classes-dataclasses-and-utility-modules)

---

## 1. Top-level Entry Point

### CLI registration

The module is registered in `pyproject.toml` as the console-script `trainer`. It is
invoked as:

```bash
uv run trainer [config_overrides]
# or, after venv activation:
python -m bnode_core.ode.trainer
```

Hydra dot-notation overrides work directly on the CLI:

```bash
uv run trainer nn_model=bnode_heatpump_test use_cuda=false
uv run trainer nn_model=latent_ode_base -m   # multi-run sweep
```

### `main()`

```python
def main():
```

`main()` is the script entry point (line 1307). It performs three steps:

1. Calls `get_config_store()` (from `bnode_core.config`) to register structured-config
   classes with Hydra's config store.
2. Calls `filepaths.config_dir_auto_recognize()` to locate the Hydra config directory
   (`config/` at superproject root, or `resources/config/` inside the package,
   depending on CWD).
3. Wraps `train_all_phases` with `hydra.main(config_path=..., config_name='train_test_ode', version_base=None)` and calls the result immediately.

The config name `'train_test_ode'` maps to `train_test_ode.yaml` in the resolved
config directory.

### `@log_hydra_to_mlflow` decorator

`train_all_phases` is decorated with `@log_hydra_to_mlflow` (from
`bnode_core.utils.hydra_mlflow_decorator`). This decorator starts an MLflow run,
logs the resolved Hydra config as MLflow parameters, and copies all Hydra output
artifacts to the MLflow artifact store at the end of the function.

### Top-level config type

`train_all_phases` receives a single argument `cfg: train_test_config_class`, a
Hydra-managed dataclass from `bnode_core.config`. It contains every knob the trainer
needs: dataset paths, model architecture, training phases, device preferences, etc.

---

## 2. Initialization

All initialization happens inside `train_all_phases` (line 324) and the helper
`initialize_model` (line 198).

### 2.1 Dataset loading

```python
hdf5_dataset, _ = load_dataset_and_config(cfg.dataset_name, cfg.dataset_path)
```

`load_dataset_and_config` (from `bnode_core.nn.nn_utils.load_data`) opens the primary
HDF5 dataset file and returns the `h5py.File` handle plus its stored config.

Two optional secondary datasets may also be opened:

| Config key | Variable name | Purpose |
|---|---|---|
| `cfg.dataset_norm_name / _path` | `hdf5_dataset_norm` | Normalisation dataset (used in place of primary for `normalization_init` and the `testnorm` split) |
| `cfg.dataset_ref_name / _path` | `hdf5_dataset_ref` | Reference dataset evaluated as context `'ref'` during training and final test |

Both default to `None` if the corresponding config keys are absent.

### 2.2 Model construction – `initialize_model()`

```python
model = initialize_model(cfg, datasets['train'], hdf5_dataset_norm or hdf5_dataset)
```

`initialize_model` (line 198) carries out:

1. **CUDA detection.** Sets `cfg.use_cuda = True` only if both `cfg.use_cuda` was
   requested *and* `torch.cuda.is_available()` returns `True`. Otherwise forces CPU.
2. **Model type dispatch.** Reads `cfg.nn_model.model_type`:
   - `'node'` → constructs `NeuralODE` (from `bnode_core.ode.node.node_architecture`).
   - `'bnode'` → constructs `BalancedNeuralODE` (from
     `bnode_core.ode.bnode.bnode_architecture`). For BNODE, may also call
     `build_feedthrough_mask()` if `cfg.nn_model.network.feedthrough_controls` is set,
     to create a boolean mask for direct control-to-output connections.
   - Any other value raises `ValueError`.
3. **Normalization.** If `initialize_normalization=True` (the default), calls
   `model.normalization_init(hdf5_dataset)` to set per-channel mean/std from the
   supplied HDF5 handle (uses `hdf5_dataset_norm` when available, otherwise primary
   dataset).
4. **Device placement.** Calls `model.to(device)`.
5. **Logging.** Logs the model string representation and total trainable parameter
   count via `logging.info`.

#### NeuralODE constructor arguments (key ones)

`states_dim`, `controls_dim`, `parameters_dim`, `outputs_dim`, `hidden_dim`
(`cfg.nn_model.network.linear_hidden_dim`), `n_layers`
(`cfg.nn_model.network.n_linear_layers`), `activation` (evaled from string config),
`intialization` (note: typo in source), `use_input_smoother`.

#### BalancedNeuralODE additional arguments (key ones)

`lat_states_mu_dim` (`cfg.nn_model.network.lat_states_dim`), `lat_parameters_dim`,
`lat_controls_dim`, `lat_ode_type`, `include_params_encoder`,
`params_to_state_encoder`, `params_to_control_encoder`, `controls_to_state_encoder`,
`state_encoder_linear`, `control_encoder_linear`, `parameter_encoder_linear`,
`ode_linear`, `decoder_linear`, `lat_state_mu_independent`, `feedthrough_controls_mask`.

### 2.3 Pretrained / pre-trained model loading (optional)

After the model is first constructed (flag `_created_model_this_job is True`):

- If `cfg.nn_model.training.load_pretrained_model is True`:  
  `model.load(path=_path, device=device)` where `_path` is resolved via
  `filepaths.filepath_from_local_or_ml_artifacts(cfg.nn_model.training.path_pretrained_model)`.
  Also sets `seq_len_epoch_start` for the current job.
- If `cfg.nn_model.training.load_trained_model_for_test is True`:  
  Same loading pattern from `cfg.nn_model.training.path_trained_model`; all training
  jobs are skipped and only the final test runs.

### 2.4 Dataloader construction (per phase)

For each non-skipped job, `make_stacked_dataset` (from
`bnode_core.nn.nn_utils.load_data`) is called once for each context:
`'train'`, `'test'`, `'validation'`, `'common_test'`, plus optional `'testnorm'` and
`'ref'`.

Key parameters passed to `make_stacked_dataset`:

| Parameter | Training | Validation/Test |
|---|---|---|
| `_load_seq_len` | `job['train_cfg'].load_seq_len` | `None` (full) |
| `_seq_len_batches` | `job['train_cfg'].seq_len_train` | `None` (full) |
| `stride` | `1` | `_stride_valid_test` (= `seq_len_train` for training phases, `1` for final test) |
| `max_samples` | `None` | `batches_per_epoch × batch_size` (limits validation cost) |

`torch.utils.data.DataLoader` is then created for each context:

- Training: `shuffle=True`, `drop_last=True`, batch size = `job['train_cfg'].batch_size`
- Validation / test / testnorm: `shuffle=False`, `drop_last=False`, batch size = **4 ×**
  training batch size (no backward pass, so more fits in memory).
- Common test and ref: full dataset as one batch (for `'ref'`) or same 4× rule.
- All loaders use `collate_fn=timeseries_collate_fn`, `pin_memory=True`,
  `persistent_workers=True` (if `n_workers > 0`).

After construction, `job['train_cfg'].seq_len_train` is updated to reflect the actual
sequence length stored in the dataset object (because `make_stacked_dataset` may round
or adjust it).

### 2.5 Optimizer and scheduler setup (`train_one_phase`, line 924)

#### Optimizer

Controlled by `train_cfg.optimizer` (string):

| Value | Class | Key config fields |
|---|---|---|
| `'adam'` | `torch.optim.Adam` | `lr_start`, `weight_decay`, `beta1_adam`, `beta2_adam` |
| `'lbfgs'` | `torch.optim.LBFGS` | `lr_start`, `lbfgs_max_iter`, `lbfgs_history_size`, `lbfgs_tolerance_grad`, `lbfgs_tolerance_change`, `lbfgs_line_search_fn` |

Any other string raises `ValueError`.

Optional optimizer reload across phases: if `train_cfg.reload_optimizer is True`,
the state dict from the previous phase's optimizer file is loaded via
`torch.load(filepaths.filepath_optimizer_current_hydra_output(job_idx-1))`,
and then the learning rate in every param group is reset to `train_cfg.lr_start`.
Failure to load is caught and logged as a warning (does not abort).

#### Learning-rate schedulers

Only active if `train_cfg.use_lr_scheduler is True` and `pre_train is False`.
Controlled by `train_cfg.lr_scheduler_type`:

| Value | Class | Stepping cadence | Key config fields |
|---|---|---|---|
| `'cosine'` | `CosineAnnealingLR` | **Per batch** (inside `train_one_epoch`) | `cosine_T_max` (epochs, default `max_epochs // 10`), `cosine_eta_min` |
| `'plateau'` | `ReduceLROnPlateau` | **Per epoch**, on validation loss | `plateau_patience`, `plateau_factor`, `plateau_min_lr`, `plateau_mode`, `plateau_threshold`, `plateau_cooldown`, `plateau_eps` |

Plateau patience auto-computation: if `train_cfg.plateau_patience is None`, the helper
`lr_on_plateau_iterations_to_min_lr()` (from `bnode_core.nn.nn_utils.lr_scheduler`) is
called to estimate how many steps are needed to decay from `lr_start` to `plateau_min_lr`
at the given `factor`, and patience is set to
`min(early_stopping_patience // 5, (max_epochs / 3) // _iters)`.

`lr_schedulers` is a dict (`{'cosine': ..., 'plateau': ...}`). If no scheduler is
configured, it is set to `None`.

#### GradScaler (AMP)

```python
scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_cuda and cfg.use_amp)
```

Automatic mixed precision is used for standard (non-LBFGS) optimizers when both
`cfg.use_cuda` and `cfg.use_amp` are `True`. For LBFGS, AMP is explicitly disabled
(comment: "disable AMP for simplicity").

---

## 3. Outer Training Loop Structure

### 3.1 Job list construction

`train_all_phases` assembles a flat `job_list` (list of dicts):

```python
job_list = []
# slot 0: optional pre-training
job_list.append({'skip': ..., 'test': False, 'train_cfg': cfg.nn_model.training.pre_training, 'pre_train': True})
# slots 1..N: main training phases
for idx, main_train_cfg in enumerate(cfg.nn_model.training.main_training):
    job_list.append({'skip': ..., 'test': False, 'train_cfg': main_train_cfg, 'pre_train': False})
# final slot: optional test
if cfg.nn_model.training.test is True:
    job_list.append({'skip': False, 'test': True, 'train_cfg': cfg.nn_model.training.main_training[-1], 'pre_train': False})
```

Skip conditions:

| Job type | Skipped when |
|---|---|
| Pre-training | `not cfg.nn_model.training.pre_train` OR `load_pretrained_model` OR `load_trained_model_for_test` |
| Main training | `load_trained_model_for_test` |
| Test | Never skipped |

The outer loop iterates over `job_list`. Each job is wrapped in a `while True:` loop
whose sole purpose is to catch `RuntimeError: 'CUDA out of memory'` / `'CUDA memory is
almost full'`, sleep 10 s, reduce batch size to 70 %, and retry.

### 3.2 Phases (main training)

The number of main training phases equals `len(cfg.nn_model.training.main_training)`.
Each element is a `base_training_settings_class` instance. Phases can have different:

- ODE solver type and tolerances.
- Learning rate, batch size, sequence length.
- Early-stopping patience and threshold.
- Whether to activate deterministic mode at phase end
  (`activate_deterministic_mode_after_this_phase`).

**Phase transition**: after each training job completes, `train_all_phases` writes the
current job's `seq_len_train` into `job_list[idx+1]['train_cfg'].seq_len_epoch_start`.
This is how the sequence-length curriculum is threaded across phases.

### 3.3 Pre-training (NODE only)

When `job['pre_train'] is True`:

- `_load_seq_len` = `job['train_cfg'].load_seq_len`, `_seq_len_batches = 1`
  (single time-step sequences).
- Uses collocation / state-derivative data (`state_der` key in dataset).
- `_batches_per_epoch = len(dataloaders['train'])` (full pass, ignoring
  `batches_per_epoch`).
- `epochs_for_seq_len_increase = 0` (no curriculum during pre-training).
- Not supported for BNODE (module docstring explicitly notes this).

### 3.4 Epoch counter

`epoch_0` is an `int` that accumulates across phases. `train_one_phase` receives
`epoch_0` as `epoch_0` and returns `epoch + 1` at the end (i.e., the index of the
first epoch of the next phase). This means MLflow step numbers are monotonically
increasing across all phases of a single run.

---

## 4. Inner Training Loop

### 4.1 `train_one_phase` – epoch loop

```python
for epoch in range(epoch_0, epoch_0 + max_epochs):
    if epoch == epoch_stop:
        break
    ...
    if not _flag_break_after_epoch and not _flag_first_epoch_this_phase:
        ret_vals_train, dataloader_iters['train'] = train_one_epoch(...)
    else:
        ret_vals_train, dataloader_iters['train'] = test_or_validate_one_epoch(...)
```

`epoch_stop` is initially `epoch_0 + max_epochs` where:

```python
max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase
```

`epochs_for_seq_len_increase` accounts for the extra epochs needed during curriculum
warm-up:

```python
epochs_for_seq_len_increase = int(train_cfg.seq_len_increase_in_batches / _batches_per_epoch)
```

**First epoch of each phase** (`_flag_first_epoch_this_phase = True`): instead of
training, runs a single-batch evaluation (`test_or_validate_one_epoch`) on the training
set. This gives a baseline loss before any weight updates and is also used to get a
memory estimate (`evaluate_at_control_times=True` on epochs 0 and 1).

**Last epoch / termination**: when any termination flag is set (see §9), the best
model is reloaded (`model.load(path=_path_best_model, device=device)`), one final
evaluation pass is run, final metrics are logged to MLflow with the suffix
`_job_{job_idx-1}_final`, and the loop `break`s.

### 4.2 `train_one_epoch` (line 732)

```python
def train_one_epoch(model, optimizer, train_loader, train_iter, scaler,
                    train_cfg, pre_train, device, epoch, use_amp, use_cuda,
                    batch_print_interval, epoch_this_phase, lr_schedulers=None):
```

Sets `model.train()` and iterates for `batches_per_epoch` steps.

#### Persistent iterator

A persistent `train_iter` object is passed in and returned. `_next_batch(train_loader,
train_iter)` advances it, automatically restarting the DataLoader when exhausted.
This lets `batches_per_epoch` exceed `len(train_loader)` (arbitrary epoch length).

#### Per-batch sequence-length cropping (curriculum)

If `pre_train is False` and the cumulative batch count
`_batches_this_phase < train_cfg.seq_len_increase_in_batches`:

```python
_seq_len_now = seq_len_epoch_start + int(
    _batches_this_phase / seq_len_increase_in_batches
    * (seq_len_train - seq_len_epoch_start)
)
_seq_len_now = min(_seq_len_now, seq_len_train)
# Slice all 3-D tensors in data_batch to [:, :, :_seq_len_now]
for keys in data_batch.keys():
    if len(data_batch[keys].shape) == 3:
        data_batch[keys] = data_batch[keys][:, :, :_seq_len_now]
```

Once `_batches_this_phase >= seq_len_increase_in_batches`, `_seq_len_now =
seq_len_train` (full length).

#### Forward pass + loss

**Standard optimizers (Adam etc.):**

```python
optimizer.zero_grad()
with torch.amp.autocast('cuda', enabled=use_amp and use_cuda):
    ret_vals_train = model.model_and_loss_evaluation(
        data_batch, train_cfg, pre_train, device,
        return_model_outputs=False, test=False,
        last_batch=(batch_idx == batches_per_epoch - 1),
    )
loss = ret_vals_train['loss']
scaler.scale(loss).backward()
```

**LBFGS optimizer:**

A closure `_closure()` is defined that calls `zero_grad()`, forward, backward, and
`clip_grad_norm_` internally, then stores the return dict in `ret_vals_train` via
`nonlocal`. `optimizer.step(_closure)` is called once per batch; internal iterations
are controlled by `train_cfg.lbfgs_max_iter`.

#### CUDA memory guard

After the backward pass, the trainer checks CUDA memory usage:

- On epoch 0 of a phase: if current reserved memory > 60 % of total, or
  projected memory at full `seq_len_train` > 60 %, raises
  `RuntimeError('CUDA memory is almost full')`.
- Any time: if reserved > 98 % of total, raises the same error.

This is caught in `train_all_phases` and triggers a batch-size reduction + retry.

#### Gradient clipping

For Adam-family optimizers (after `scaler.unscale_`):

```python
_norm = clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
scaler.step(optimizer)
scaler.update()
```

For LBFGS: clipping is done inside the closure; after the step,
`clip_grad_norm_` is called again to *measure* `_norm` for logging (no actual clipping
effect here).

`_norm` and `train_cfg.clip_grad_norm` are included in the returned `ret_vals_train`
dict.

#### Per-batch LR scheduler step

Only `'cosine'` schedulers are stepped here:

```python
if lr_schedulers and 'cosine' in lr_schedulers:
    lr_schedulers['cosine'].step()
```

`'plateau'` is stepped at epoch level in `train_one_phase` (see §5).

#### Return values

`train_one_epoch` returns `(ret_vals_train, train_iter)`.

`ret_vals_train` is the last batch's `model.model_and_loss_evaluation` return dict,
augmented with timing stats (`time_forward`, `time_backward`, `time_optimizer_step`,
`time_loader`, derived `time_per_batch_*`) and `grad_norm`, `clip_grad_norm`,
`seq_len_now`, `ode_calls_backward`.

---

## 5. Validation / Evaluation Logic

### 5.1 `test_or_validate_one_epoch` (line 889)

```python
def test_or_validate_one_epoch(model, data_loader, train_cfg, pre_train, device,
                               all_batches=False, return_model_outputs=False,
                               activate_deterministic_mode=False, data_iter=None):
```

Sets `model.eval()`, then:

- **`all_batches=True`** (used for validation split): iterates over the entire
  DataLoader under `torch.no_grad()`. Returns the **mean** of each metric across
  batches.
- **`all_batches=False`** (used for training-set and test-set monitoring): draws a
  **single batch** from a persistent `data_iter`, advances and returns it. This is
  intentionally cheap (one batch) for epoch-frequency logging.

Both modes call `model.model_and_loss_evaluation(..., test=True)` with
`return_model_outputs=False` by default.

`activate_deterministic_mode` is passed through to `model_and_loss_evaluation`; it
is only `True` for BNODE models at phase end when
`train_cfg.activate_deterministic_mode_after_this_phase` is set.

### 5.2 What is evaluated each epoch

In `train_one_phase`, after each training epoch (or skipped-training epoch):

| Context | Frequency | `all_batches` | Notes |
|---|---|---|---|
| `'train'` | Every epoch | `False` (1 batch) | Single-batch; cheap monitor |
| `'validation'` | Every epoch | `True` (full) | Used by EarlyStopping and ReduceLROnPlateau |
| `'test'` | Every epoch | `False` (1 batch) | Single-batch monitor |
| `'ref'` | Every `ref_and_testnorm_every_n_epochs` epochs, plus first/last | `False` (1 batch) | Only if `dataloaders['ref'] is not None` |
| `'testnorm'` | Every `ref_and_testnorm_every_n_epochs` epochs, plus first/last | `False` (1 batch) | Only if `dataloaders['testnorm'] is not None` |

All metrics are logged to MLflow with `step=epoch` and contextual key suffix via
`append_context_to_dict_keys`.

### 5.3 `ReduceLROnPlateau` epoch-level step

```python
if lr_schedulers and 'plateau' in lr_schedulers:
    val_loss = ret_vals_validation.get('loss', None)
    if val_loss is not None and not (np.isnan(val_loss) or np.isinf(val_loss)):
        lr_schedulers['plateau'].step(val_loss)
```

This happens immediately after the full-validation epoch result is available.

### 5.4 Final test (test job)

When `job['test'] is True`, `train_all_phases` iterates over all six contexts
(`'train'`, `'test'`, `'validation'`, `'common_test'`, `'testnorm'`, `'ref'`) using
full-dataset evaluation (`all_batches=True`) or the streaming batch-by-batch loop for
contexts where `_save_predictions is True`.

**Prediction saving** (streaming loop, used when
`cfg.nn_model.training.save_predictions_in_dataset is True` and context is in
`cfg.nn_model.training.save_predictions_for`):

1. Creates or appends to an HDF5 file at
   `filepaths.filepath_dataset_current_hydra_output()`.
2. Copies the raw dataset group for the context into the new file.
3. Iterates batch by batch; on the first batch, creates HDF5 datasets of shape
   `(total_len, ...)` for each output key to save (`states_hat`, `states_der_hat`,
   `outputs_hat`, and optionally all internal variables if
   `test_save_internal_variables is True` and context is in
   `test_save_internal_variables_for`).
4. Writes each batch's outputs with `write_offset` tracking.
5. Accumulates metrics as a running sum; computes mean at the end.

MLflow receives final metrics as both `{metric}_{context}` and `{metric}_{context}_final`
at step `_epoch_0 + 1`.

---

## 6. Checkpointing / State Persistence

### 6.1 File paths

All paths are resolved through helpers in `bnode_core.filepaths`:

| Helper | Resolved path | Contents |
|---|---|---|
| `filepath_model_current_hydra_output()` | `<hydra_output_dir>/model.pt` | **Rolling** best model — updated every epoch that does not produce NaN |
| `filepath_optimizer_current_hydra_output()` | `<hydra_output_dir>/optimizer.pt` | Optimizer state dict matching the rolling model |
| `filepath_model_current_hydra_output(job_idx)` | `<hydra_output_dir>/model_phase_{job_idx}.pt` | **Phase-best** model (lowest validation loss) — saved by `EarlyStopping` |
| `filepath_optimizer_current_hydra_output(job_idx)` | `<hydra_output_dir>/optimizer_phase_{job_idx}.pt` | Optimizer state dict matching the phase-best model |
| `filepath_pretrained_model_current_hydra_output()` | `<hydra_output_dir>/model_pretrained.pt` | Phase-best model for the pre-training job |
| `filepath_dataset_current_hydra_output()` | `<hydra_output_dir>/dataset_predictions.h5` | HDF5 file containing model predictions (only when `save_predictions_in_dataset`) |

> **Note.** Variable names in the code use `_path_best_model` for the phase-best path
> (written by `EarlyStopping`) and `_path_current_model` for the rolling path.

### 6.2 When files are written

| Event | Files written |
|---|---|
| After every non-NaN training epoch | `model.pt`, `optimizer.pt` (rolling) |
| When validation loss improves (EarlyStopping) | `model_phase_{i}.pt`, `optimizer_phase_{i}.pt` (or `model_pretrained.pt`) |
| When `activate_deterministic_mode_after_this_phase` fires | Overwrites `model_phase_{i}.pt` with deterministic-mode weights |
| Final test | `dataset_predictions.h5`, copy of `trainer.py` to hydra output dir |
| End of `train_all_phases` (via decorator) | All Hydra output dir contents → MLflow artifact store; failures tracked in `could_not_log_artifacts.txt` |

### 6.3 Formats

- Model checkpoints: `model.save(path=...)` — the exact format depends on
  `model.save()` / `model.load()` methods (likely `torch.save` of `state_dict` inside
  the model class).
- Optimizer state: `torch.save(optimizer.state_dict(), path)`.
- Predictions: HDF5 (via `h5py`).
- Artifact failures: plain text file `could_not_log_artifacts.txt`.

---

## 7. Restart / Resume Logic

> **Important.** There is **no explicit cross-run restart mechanism** in this version of
> the trainer. The concepts of "restart" and "resume" are implemented as *within-epoch
> recovery* (NaN reloading) and *across-phases loading* (pretrained/trained model),
> not as a full checkpoint-based resume from an interrupted run.

### 7.1 NaN / AssertionError recovery (within-phase)

When the loss for an epoch is `NaN` or `Inf`, or an `AssertionError` occurs in the ODE
integrator:

```
if nan_counter <= 49:
    model.load(path=_path_current_model, device=device)   # rolling checkpoint
    optimizer.load_state_dict(torch.load(_path_current_optimizer))
    nan_counter += 1
    grad_norm_last_reduced_counter += 1
    if grad_norm_last_reduced_counter > 2:
        train_cfg.clip_grad_norm *= 0.7   # progressively reduce clipping norm
        grad_norm_last_reduced_counter = 0
elif 49 < nan_counter <= 55:
    model.load(path=_path_best_model, device=device)      # phase-best checkpoint
    optimizer.load_state_dict(torch.load(_path_optimizer_best_model))
    nan_counter += 1
else:  # nan_counter > 55
    raise ValueError('Loss is NaN for more than 55 epochs ...')
```

After a non-NaN epoch, `nan_counter` is reset to 0.

This mechanism is only active if `train_cfg.reload_model_if_loss_nan is True`.

### 7.2 Loading a pre-trained model (across runs)

`cfg.nn_model.training.load_pretrained_model = True` + `path_pretrained_model`:
loads weights from a previous run's pre-training output into the freshly constructed
model before the first main training phase. Does not restore optimizer state.

### 7.3 Loading a trained model for test only

`cfg.nn_model.training.load_trained_model_for_test = True` + `path_trained_model`:
loads a fully trained model, skips all training jobs, and runs only the final test job.

### 7.4 Optimizer reload across phases

`train_cfg.reload_optimizer = True`: loads the optimizer state dict saved at the
*previous* phase (by `job_idx - 1`) and resets the learning rate to `lr_start`.

### 7.5 Design gap / TODO

There is no mechanism to resume an interrupted run mid-phase. If a run is killed between
epochs, the rolling checkpoint (`model.pt`) exists but the trainer always starts a new
Hydra output directory, discarding it. A full restart-from-checkpoint capability
would require detecting an existing output directory, reading epoch counter state, and
re-entering the epoch loop at the correct position.

> The module docstring documents **NaN Recovery** but does not document any
> cross-run resume. A comment inside the docstring says "LR scheduling might be
> a better long-term solution" for NaN recovery, suggesting this recovery mechanism
> is considered temporary.

---

## 8. Sequence-Length Curriculum

### 8.1 Overview

The curriculum gradually increases the number of time-steps fed to the ODE solver
during training — starting short (easier, cheaper) and ending at the full configured
length. Validation and test always use the **full** sequence length, regardless of
where training is in the curriculum.

### 8.2 Parameters (per phase, in `base_training_settings_class`)

| Field | Meaning |
|---|---|
| `seq_len_train` | Final (target) sequence length for this phase |
| `seq_len_epoch_start` | Sequence length at the beginning of this phase (set automatically from the previous phase's `seq_len_train`) |
| `seq_len_increase_in_batches` | Number of **cumulative batches** (from phase start) over which to linearly ramp from `seq_len_epoch_start` to `seq_len_train` |
| `seq_len_increase_abort_after_n_stable_epochs` | Number of consecutive "stable" epochs after which ramping is declared complete early |

`seq_len_epoch_start` is *not* set by the user in most cases; it is propagated by
`train_all_phases` at phase boundaries:

```python
job_list[idx+1]['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train
```

For the pre-training → first main training transition:

```python
job_list[idx+1]['train_cfg'].seq_len_epoch_start = 1  # pre-train uses seq_len = 1
```

If a pretrained model is loaded with a known sequence length,
`cfg.nn_model.training.pre_trained_model_seq_len` can override this.

If `seq_len_epoch_start >= seq_len_train` (no ramp needed):

```python
epochs_for_seq_len_increase = 0
train_cfg.seq_len_increase_in_batches = 0
```

and the maximum epoch count is not extended.

### 8.3 Per-batch linear interpolation

Inside `train_one_epoch`, at each batch:

```python
_batches_this_phase = epoch_this_phase * batches_per_epoch + batch_idx
if _batches_this_phase < seq_len_increase_in_batches:
    _seq_len_now = seq_len_epoch_start + int(
        _batches_this_phase / seq_len_increase_in_batches
        * (seq_len_train - seq_len_epoch_start)
    )
    _seq_len_now = min(_seq_len_now, seq_len_train)
    # slice 3-D tensors: [:, :, :_seq_len_now]
```

This is applied *after* the batch is fetched from the DataLoader (data is pre-loaded at
full length; slicing happens in CPU memory before the forward pass).

### 8.4 Early abort of curriculum

In `train_one_phase`, each epoch after validation:

```python
if ret_vals_validation['loss'] < 2 * ret_vals_train['loss']:
    _stable_epochs += 1
else:
    _stable_epochs = 0
```

When `_stable_epochs > seq_len_increase_abort_after_n_stable_epochs`:

```python
train_cfg.seq_len_increase_in_batches = _batches_per_epoch * (epoch - epoch_0)
epoch_stop = epoch_0 + train_cfg.max_epochs + (epoch - epoch_0)
```

This has two effects:
1. The ramp is declared complete immediately (all subsequent batches see `seq_len_train`).
2. `epoch_stop` is extended by `(epoch - epoch_0)` extra epochs to compensate for the
   epochs spent in curriculum warm-up.

### 8.5 `_flag_out_of_seq_len_increase`

This boolean flag gates early stopping: early stopping is **disabled** while the
curriculum ramp is active (`_flag_out_of_seq_len_increase is False`). Once
`_batches_this_phase > seq_len_increase_in_batches` (detected epoch-level):

```python
_flag_out_of_seq_len_increase = True
early_stopping.reset_counter()   # clear patience counter accumulated during ramp
```

### 8.6 First-epoch probe

On epoch 0 and 1 of a phase (`epoch_this_phase in [0, 1]`, pre-training excluded),
`train_cfg.evaluate_at_control_times` is forced to `True` (on a `copy.deepcopy` of
`train_cfg` so the original is not mutated). The model docstring notes this is to "get
a good estimate for memory usage" before committing to a full-length sequence.

---

## 9. Early Stopping / Convergence Criteria

### 9.1 `EarlyStopping` (from `bnode_core.nn.nn_utils.early_stopping`)

```python
early_stopping = EarlyStopping(
    patience=train_cfg.early_stopping_patience,
    verbose=True,
    threshold=train_cfg.early_stopping_threshold,
    threshold_mode=train_cfg.early_stopping_threshold_mode,
    path=_path_best_model,
    optimizer_path=_path_optimizer_best_model,
    trace_func=logging.info,
)
```

Called every epoch after full validation:

```python
early_stopping(
    ret_vals_validation['loss'],
    model,
    epoch,
    optimizer,
    corresponding_loss=ret_vals_validation['rmse_states_outputs'],
)
```

`EarlyStopping` internally saves the model (and optimizer) whenever validation loss
improves beyond `threshold`, and increments a patience counter otherwise. When
`patience` epochs elapse without improvement, `early_stopping.early_stop` becomes
`True`.

`corresponding_loss` (`rmse_states_outputs`) is stored as a secondary metric tracked
alongside the primary loss — it is logged to MLflow as `best_rmse_states_outputs` when
the EarlyStopping counter resets to 0.

`reset_counter()` is called when transitioning out of sequence-length curriculum, so
patience is counted fresh from the moment the full sequence length is active.

### 9.2 Termination flags (checked at epoch start)

The epoch loop checks **four** independent termination flags:

| Flag | Condition | MLflow label |
|---|---|---|
| `_flag_max_epoch` | `epoch == epoch_stop - 1` | `'max epochs'` |
| `_flag_early_stopping` | `early_stopping.early_stop and _flag_out_of_seq_len_increase` | `'early stopping'` |
| `_flag_break_after_loss_of` | `early_stopping.best_score < train_cfg.break_after_loss_of` (if configured) | `'break after loss'` |
| `_flag_nan_counter` | `nan_counter > 50` | `'4 NaNs in loss'` *(note: comment says 50, label says 4 — apparent inconsistency in source)* |

When any flag fires, `_flag_break_after_epoch = True`. The loop still performs one
final evaluation pass before breaking.

### 9.3 `break_after_loss_of`

An optional lower-bound threshold: if `early_stopping.best_score` (best validation
loss so far) drops below `train_cfg.break_after_loss_of`, training terminates. This
allows a phase to end as soon as a target loss level is achieved.

### 9.4 KeyboardInterrupt handling

The entire epoch loop is wrapped in:

```python
try:
    for epoch in range(...):
        ...
except KeyboardInterrupt:
    mlflow.log_param('ended by', 'keyboard interrupt')
    model.load(path=_path_best_model, device=device)
    # fallback loop over job_idx..0 if best model file does not exist
```

The model is restored to the best checkpoint before `train_one_phase` returns.

---

## 10. Helper Classes, Dataclasses, and Utility Modules

### 10.1 `train_test_config_class` and `base_training_settings_class`

Defined in `bnode_core.config`. These are Hydra-structured dataclasses that describe
the entire config tree:

- `train_test_config_class`: top-level; contains `nn_model`, `dataset_name`,
  `dataset_path`, `dataset_norm_name/path`, `dataset_ref_name/path`, `use_cuda`,
  `use_amp`, `n_workers_train_loader`, `n_workers_other_loaders`, `prefetch_factor`,
  `batch_print_interval`, `mlflow_experiment_name`, `dataset_norm_name`,
  `dataset_ref_name`.
- `base_training_settings_class`: per-phase config; contains `lr_start`, `batch_size`,
  `seq_len_train`, `seq_len_epoch_start`, `seq_len_increase_in_batches`,
  `seq_len_increase_abort_after_n_stable_epochs`, `batches_per_epoch`, `max_epochs`,
  `early_stopping_patience`, `early_stopping_threshold`,
  `early_stopping_threshold_mode`, `break_after_loss_of`, `clip_grad_norm`,
  `weight_decay`, `beta1_adam`, `beta2_adam`, `optimizer`, `lbfgs_*`,
  `use_lr_scheduler`, `lr_scheduler_type`, `cosine_*`, `plateau_*`,
  `reload_optimizer`, `reload_model_if_loss_nan`, `evaluate_at_control_times`,
  `activate_deterministic_mode_after_this_phase`, `load_seq_len`, `pre_train` (flag),
  and pre-training sub-config `pre_training`.

`get_config_store()` registers these with Hydra so they appear in `--help` output.

### 10.2 `NeuralODE` (`bnode_core.ode.node.node_architecture`)

Direct neural differential equation model. Key interface methods used by the trainer:

- `normalization_init(hdf5_dataset)` — sets normalization statistics.
- `model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, ...)` — forward
  pass + loss; returns dict with `'loss'`, `'rmse_states_outputs'`, `'ode_calls_forward'`.
- `get_progress_string(ret_train, ret_val, ret_test, pre_train)` — formats a one-line
  summary string for epoch logging.
- `save(path)` / `load(path, device)` — checkpoint serialization.
- `ode_fun_count` attribute — ODE function evaluation counter (used for logging).

### 10.3 `BalancedNeuralODE` (`bnode_core.ode.bnode.bnode_architecture`)

Latent-space ODE with encoder-decoder. Provides the same interface as `NeuralODE`.
Additional behaviour:

- Uses reparameterization (variational inference) during training (`test=False`);
  disables it during evaluation (`test=True`) or deterministic mode.
- `build_feedthrough_mask(control_names, feedthrough_controls, controls_dim)`:
  utility function in the same module; builds a boolean mask for direct
  control-to-output connections.

### 10.4 `EarlyStopping` (`bnode_core.nn.nn_utils.early_stopping`)

Standard patience-based early stopping. Attributes read by the trainer:

- `early_stop: bool` — triggers termination flag.
- `counter: int` — epochs without improvement (logged to MLflow each epoch).
- `best_score: float` — best validation loss seen (logged when counter resets).
- `corresponding_score: float` — secondary metric (RMSE) at the best epoch.

Methods called by trainer: `__call__(val_loss, model, epoch, optimizer, corresponding_loss)`,
`reset_counter()`.

### 10.5 `load_dataset_and_config` / `make_stacked_dataset` / `TimeSeriesDataset` / `timeseries_collate_fn` (`bnode_core.nn.nn_utils.load_data`)

- `load_dataset_and_config(name, path)` — opens HDF5 file and returns handle + config.
- `make_stacked_dataset(hdf5, context, load_seq_len, seq_len_batches, stride, max_samples)` —
  returns a `TimeSeriesDataset` that slices and stacks time-series windows from the
  HDF5 context group.
- `TimeSeriesDataset` — `torch.utils.data.Dataset` subclass. Key attributes:
  `seq_len` (if using a map-style dataset), and `datasets` dict containing `'time'`
  and other tensors.
- `timeseries_collate_fn` — custom collate function passed to every DataLoader.

### 10.6 `lr_on_plateau_iterations_to_min_lr` (`bnode_core.nn.nn_utils.lr_scheduler`)

Utility function that computes the number of `ReduceLROnPlateau` patience steps needed
to decay `lr_start` to `lr_min` given `factor` and `eps`. Used only to auto-compute
`plateau_patience` when it is `None` in config.

### 10.7 `log_hydra_to_mlflow` decorator (`bnode_core.utils.hydra_mlflow_decorator`)

Wraps `train_all_phases`. Responsibilities:
- Starts (or reuses) an MLflow run with `mlflow_experiment_name` from config.
- Logs all Hydra-resolved config parameters to MLflow.
- After `train_all_phases` returns, copies the entire Hydra output directory to
  MLflow artifacts. Failures are written to `could_not_log_artifacts.txt` in the
  output dir.

### 10.8 `bnode_core.filepaths`

Central module for all path resolution. Key functions used by the trainer:

| Function | Returns |
|---|---|
| `config_dir_auto_recognize()` | `Path` to the Hydra config directory |
| `filepath_model_current_hydra_output(idx=None)` | `Path` to `model.pt` or `model_phase_{idx}.pt` |
| `filepath_optimizer_current_hydra_output(idx=None)` | `Path` to `optimizer.pt` or `optimizer_phase_{idx}.pt` |
| `filepath_pretrained_model_current_hydra_output()` | `Path` to `model_pretrained.pt` |
| `filepath_dataset_current_hydra_output()` | `Path` to the predictions HDF5 file |
| `dir_current_hydra_output()` | `Path` to the current Hydra run output directory |
| `filepath_from_local_or_ml_artifacts(path_str)` | Resolves a path that may be a local path or MLflow artifact URI |

### 10.9 `_next_batch` (module-level helper, line 712)

```python
def _next_batch(data_loader, iterator):
```

A thin wrapper that advances a DataLoader iterator, auto-recreating it on
`StopIteration`. Used for both training (persistent iterator across epochs) and single-
batch validation passes.

### 10.10 `append_context_to_dict_keys` (line 918)

```python
def append_context_to_dict_keys(dictionary, context, pre_train=False):
```

Prepends `'pre_'` (if `pre_train=True`) and appends `'_{context}'` to every key in a
metrics dict before MLflow logging. Examples:

- `{'loss': 0.01}`, context `'validation'` → `{'loss_validation': 0.01}`
- `{'loss': 0.01}`, context `'validation'`, `pre_train=True` → `{'pre_loss_validation': 0.01}`

---

## Appendix: TODOs and Design Gaps

| Location | Issue |
|---|---|
| Line ~495 | `# TODO: I believe this is never reached` — comment inside the branch that creates a `None` dataloader for empty test datasets during test-only mode. |
| §7.5 (restart) | No cross-run resume. An interrupted run loses its training progress; only manual model loading is supported. |
| Module docstring | NaN recovery note: "LR scheduling might be a better long-term solution". The current clip-norm reduction is a workaround. |
| Line 1073 | MLflow log param says `'4 NaNs in loss'` but `_flag_nan_counter` fires at `nan_counter > 50`. The label is a stale artifact and does not match the code. |
| Line 820 | `torch.cuda.memory_reserved()/(1024^3)` — `^` is bitwise XOR in Python, not exponentiation. The CUDA memory metric is therefore logged in wrong units (bytes divided by ~1027 rather than GiB). This is a latent bug. |
| Pre-training + BNODE | Pre-training is explicitly unsupported for BNODE but is only guarded by a docstring note, not a runtime check. Passing `pre_train=True` with a BNODE model will silently attempt pre-training with `seq_len=1`. |
| `initialize_model` docstring | Documents `model_type` parameter but the function signature does not have it (removed at some point). The docstring is stale. |
