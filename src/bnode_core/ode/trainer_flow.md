# trainer.py current flow

This file is a pseudo-code map of the current `trainer.py` flow before the restart redesign from `.copilot/session-state/08229b4d-5d38-4240-bcef-f55b5845bba0/trainer_restart_plan.md`.

The goal is to make the current control flow, variable ownership, and save-vs-local boundaries easier to inspect before we split the trainer into clearer subfunctions.

## Main entry points

### `main()`

- registers the Hydra config store
- resolves the config root
- runs `train_all_phases(cfg)`

### `initialize_model(cfg, train_dataset, hdf5_dataset, initialize_normalization=True)`

Responsibilities:

- resolves device usage from `cfg.use_cuda`
- chooses NODE vs BNODE
- builds the model
- initializes normalization from dataset statistics
- moves the model to the target device

Important variables:

- `cfg.use_cuda`: mutated to reflect actual device usage
- `model_type`: derived from `cfg.nn_model.model_type`
- `model`: runtime model object reused across jobs

### `train_all_phases(cfg)`

Responsibilities:

- loads datasets and optional norm/ref datasets
- builds the outer `job_list`
- checks restart metadata to decide which job resumes
- creates per-job datasets and dataloaders
- creates or reloads the model
- dispatches to `train_one_phase(...)` or final test logic
- handles CUDA-memory retry loop
- clears the restart marker after all jobs finish

Important variables:

- `job_list`: ordered outer workflow of pre-train, main-train jobs, and optional final test
- `restart_state`: serialized restart checkpoint data used only to choose the resumed job here
- `restart_state_path`: checkpoint file path used again by `train_one_phase`
- `job_start_idx`: first outer job to execute
- `_created_model`: whether the shared model has already been initialized
- `_epoch_0`: global epoch anchor passed between jobs
- `datasets`: per-job dataset map
- `dataloaders`: per-job dataloader map
- `model`: shared runtime model across jobs

Pseudo-code:

```text
def train_all_phases(cfg):
    log start
    device = resolve device from cfg

    hdf5_dataset = load main dataset
    hdf5_dataset_norm = maybe load norm dataset
    hdf5_dataset_ref = maybe load ref dataset

    job_list = []
    add optional pre-train job
    add each main-training job
    add optional final test job

    restart_state, restart_state_path = load restart metadata if it exists
    job_start_idx = restart_state.job_idx or 0
    if restart_state exists:
        validate resumed job
        replace resumed job train_cfg with saved train_cfg values

    _created_model = False
    _epoch_0 = restart_state.next_epoch or 0

    for idx, job in job_list[job_start_idx:]:
        while True:
            try:
                if not job["skip"]:
                    datasets = build datasets for contexts
                    dataloaders = build dataloaders for contexts
                    job["train_cfg"].seq_len_train = actual train dataset seq len

                    if model not created yet:
                        model = initialize_model(...)

                    if configured:
                        maybe load pretrained model
                        maybe load trained model for test

                if job["skip"]:
                    log skip
                elif not job["test"]:
                    _epoch_0 = train_one_phase(
                        cfg, model, dataloaders, job["train_cfg"],
                        test=False, pre_train=job["pre_train"],
                        job_idx=idx, epoch_0=_epoch_0,
                        restart_state=restart_state for this job only,
                        restart_manager=restart_state_path,
                    )
                    restart_state = None
                    maybe seed next job seq_len_epoch_start
                else:
                    run final evaluation for each context
                    optionally write predictions/internal variables to HDF5
                    log metrics/artifacts

                maybe clear CUDA cache
                break

            except RuntimeError as e:
                if CUDA memory issue:
                    reduce batch size
                    retry same job
                else:
                    raise

    clear restart marker
```

### `_next_batch(data_loader, iterator)`

Responsibilities:

- advances a persistent iterator
- recreates it on exhaustion

Important variables:

- `iterator`: caller-owned iterator that survives between function calls

### `train_one_epoch(live_state, train_loader, train_iter, epoch)`

Responsibilities:

- runs the batch loop for one training epoch
- performs sequence-length warmup slicing
- runs optimizer step with Adam or LBFGS
- steps per-batch schedulers
- returns final batch metrics and updated iterator

Important variables:

- `live_state`: current runtime bundle for the phase
- `train_cfg`: phase config, sometimes deep-copied for epoch-local overrides
- `epoch_this_phase`: epoch offset relative to `phase_state.phase_epoch_0`
- `_batches_this_phase`: batch-progress counter used by seq-len warmup
- `_seq_len_now`: active sequence length for the current batch
- `ret_vals_train`: last batch metrics returned by the model
- `_norm`: current gradient norm

Pseudo-code:

```text
def train_one_epoch(live_state, train_loader, train_iter, epoch):
    derive epoch_this_phase
    train_cfg = live_state.train_cfg
    maybe deep-copy train_cfg for first-epoch memory-estimation behavior

    for batch_idx in range(batches_per_epoch):
        data_batch, train_iter = _next_batch(train_loader, train_iter)
        maybe crop batch tensors according to seq-len warmup

        if optimizer is not LBFGS:
            zero grad
            autocast forward
            backward through GradScaler
        else:
            run LBFGS closure

        guard against high CUDA memory usage
        clip gradients
        optimizer/scaler step
        maybe step cosine scheduler
        log batch progress

    convert tensor metrics to plain values
    append timing + grad norm + seq len metadata
    return ret_vals_train, train_iter
```

### `test_or_validate_one_epoch(...)`

Responsibilities:

- runs either full-dataset evaluation or one-batch evaluation
- optionally returns model outputs for prediction export
- optionally reuses a persistent iterator for single-batch evaluation

Important variables:

- `all_batches`: selects full-pass vs one-batch mode
- `return_model_outputs`: selects metrics only vs metrics + predictions
- `data_iter`: persistent iterator for one-batch mode

### `append_context_to_dict_keys(dictionary, context, pre_train=False)`

- prefixes metric names for MLflow logging

### `train_one_phase(...)`

Responsibilities:

- derives per-phase epoch bounds and file paths
- builds optimizer, scheduler, scaler, and early stopping helpers
- creates `LiveTrainingState`
- restores checkpointed runtime state if resuming
- runs the epoch loop
- handles NaN recovery, evaluation, metric logging, early stopping, deterministic mode, and checkpoint saves
- returns the next global epoch anchor for the next outer job

Important variables:

- `_phase_epoch_0`: global epoch anchor for this phase
- `_batches_per_epoch`: effective batches per epoch for this phase
- `epochs_for_seq_len_increase`: warmup extension derived from seq-len settings
- `max_epochs`: total epoch horizon including warmup extension
- `_path_best_model`, `_path_optimizer_best_model`: best-checkpoint paths
- `_path_current_model`, `_path_current_optimizer`: rolling current-checkpoint paths
- `optimizer`, `lr_schedulers`, `scaler`, `early_stopping`: runtime helpers built per phase
- `live_state`: current runtime bundle
- `phase_state`: mutable phase counters/flags inside `live_state`
- `dataloader_iters`: persistent iterators for train/test/validation/ref/testnorm
- `_flag_break_after_epoch`: forces one final evaluation epoch before exit

Pseudo-code:

```text
def train_one_phase(...):
    device = resolve device
    _phase_epoch_0 = restart_state.epoch_0 if resuming else epoch_0

    if not test:
        derive _batches_per_epoch
        derive epochs_for_seq_len_increase
        max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase

        derive best/current checkpoint paths
        optimizer = build optimizer
        maybe reload previous optimizer state
        early_stopping = build helper
        scaler = build AMP scaler
        lr_schedulers = maybe build schedulers

        live_state = LiveTrainingState.create(...)
        phase_state = live_state.phase_state

        try:
            dataloader_iters = one iterator per available dataloader

            for epoch in range(phase_state.epoch_start, phase_state.phase_epoch_0 + max_epochs):
                if epoch == phase_state.epoch_stop:
                    break

                compute stop flags:
                    max epoch
                    early stopping
                    break-after-loss threshold
                    too many NaN reloads

                if stop requested:
                    mark final-eval epoch
                    load best model

                maybe shorten seq-len warmup and move epoch_stop

                if normal training epoch:
                    ret_vals_train = train_one_epoch(...)
                    if train loss is NaN or assertion-reload happened:
                        maybe reload current/best checkpoint
                        maybe reduce clip_grad_norm
                        increment phase_state.nan_counter
                    else:
                        reset NaN counters
                        save rolling current model/optimizer
                else:
                    do eval-only "first/final" epoch
                    maybe activate deterministic mode
                    phase_state.first_epoch_is_evaluation = False

                evaluate validation
                step plateau scheduler
                update early stopping
                update stable_epochs

                evaluate test
                maybe evaluate ref
                maybe evaluate testnorm

                log lr + early stopping + progress

                if final-eval epoch:
                    log final metrics and break

                if seq-len warmup finished:
                    phase_state.flag_out_of_seq_len_increase = True
                    early_stopping.reset_counter()

                if restart_manager exists:
                    live_state.save_checkpoint(epoch + 1)

        except KeyboardInterrupt:
            tag run as interrupted
            try to reload best available model

        log final epoch

    return epoch + 1
```

## Current state and data ownership map

### Outer orchestration data in `train_all_phases()`

These variables drive which job runs and what gets built for that job:

- `job_list`
- `job_start_idx`
- `restart_state` metadata for outer-job selection
- `_epoch_0`
- `datasets`
- `dataloaders`
- `_created_model`
- `model`

Observation:

- outer orchestration, dataset construction, model setup, resume selection, and final test logic all live in one function
- the outer loop mutates both long-lived values (`model`, `_epoch_0`) and per-job locals (`datasets`, `dataloaders`, `_batch_size`)

### Inner phase runtime data in `train_one_phase()`

These are true live runtime objects that exist only after phase setup:

- `optimizer`
- `lr_schedulers`
- `scaler`
- `early_stopping`
- `live_state`
- `phase_state`
- `dataloader_iters`

Observation:

- runtime creation and restore happen in the same function
- the epoch loop also owns recovery, validation/test scheduling, final metric logging, and checkpoint saves

### Proposed split for persisted state in the new two-state design

This section is written with the target restart plan in mind, not as a defense of the current single `TrainingRestartState`.

The current checkpoint still stores one mixed bundle, but the refactor target should be two explicit state classes with two smaller checkpoint data structures.

### Proposed outer state class for `train_all_phases()`

Purpose:

- own outer-loop orchestration
- decide which job runs next
- keep resume metadata for the outer workflow
- recreate per-job datasets/dataloaders instead of serializing them

Likely runtime-owned fields:

- `cfg`
- `device`
- `job_list`
- `job_start_idx`
- `restart_state_path`
- `_epoch_0` or a renamed outer epoch/job handoff field
- `_created_model`
- `model` or a model-handoff reference if the shared model remains outer-owned
- dataset handles: `hdf5_dataset`, `hdf5_dataset_norm`, `hdf5_dataset_ref`

Likely outer checkpoint data:

- current outer job index (`job_idx`)
- outer progress handoff needed to resume the correct phase (`_epoch_0` or renamed equivalent)
- Hydra / MLflow resume metadata if that responsibility stays at the outer level

Should not be in the outer checkpoint:

- `cfg`
- dataset handles
- `datasets`
- `dataloaders`
- per-job dataloader construction settings

### Proposed inner state class for `train_one_phase()`

Purpose:

- own phase-local runtime objects
- own mutable phase-control values
- save and restore only the minimum phase checkpoint data

Likely runtime-owned fields:

- `train_cfg`
- phase-local checkpoint paths
- `optimizer`
- `lr_schedulers`
- `scaler`
- `early_stopping`
- `dataloader_iters`
- `phase_state`
- the runtime model reference if model ownership moves to the inner phase

Likely inner checkpoint data:

- model state dict
- optimizer state dict
- scheduler state dicts
- scaler state dict
- early stopping state dict
- next epoch / epoch progress counters
- `first_epoch_is_evaluation`
- `nan_counter`
- `grad_norm_last_reduced_counter`
- `stable_epochs`
- `flag_out_of_seq_len_increase`
- `epoch_stop`
- `deterministic_mode_active`
- RNG state

Needs an explicit decision during the refactor:

- whether the shared `model` belongs to the outer state or the inner state
- whether the outer checkpoint or inner checkpoint owns the model state dict
user: should be inner state, while each phase still explicitely save its best model
- how much of the current `_epoch_0` / `phase_epoch_0` / `next_epoch` bookkeeping collapses into one outer field plus one inner field
user: reduce this, if possible. I would be fine with one peoch field in the outer field, that keeps track of the las completed epoch of the inner field, and one inner epoch field, that is added with the outer field only for logging and printing.

Fields that are currently saved but do not fit the target checkpoint boundary:

- `training_cfg_state`: the plan says `cfg` should be recreated, not checkpointed
- any restart metadata that exists only because the current design serializes one mixed object instead of two narrower checkpoint data structures

### Data that is recreated locally and should stay clearly local

- dataset handles: `hdf5_dataset`, `hdf5_dataset_norm`, `hdf5_dataset_ref`
- `datasets`
- `dataloaders`
- per-context `_batch_size`, `_num_workers`, `_drop_last`, `_shuffle`
- temporary eval/export values such as `ret_vals_*`, `model_outputs_batch`, `keys_to_save`
- batch-level timing variables in `train_one_epoch`

## Flow pain points visible in the current code

1. `train_all_phases()` mixes too many levels: job planning, restart selection, dataset creation, model loading, training dispatch, test export, and CUDA retry handling.
2. `train_one_phase()` mixes setup, restore, epoch control, NaN recovery, evaluation orchestration, metric logging, and checkpoint persistence.
3. `_epoch_0`, `phase_state.phase_epoch_0`, `phase_state.epoch_start`, `phase_state.epoch_stop`, and `restart_state.next_epoch` are all related, but the ownership boundary is not obvious at a glance.
4. Saved state and transient locals are interleaved; it is easy to lose track of which values are merely helpers for one loop iteration and which must survive a restart.
5. Final test/export logic is embedded inside the outer loop instead of being a clearly separate workflow step.

## Refactor notes before the restart redesign

DONE: Extract job-list construction from `train_all_phases()` into one helper that returns an explicit outer workflow description.

DONE: Extract dataset and dataloader creation into a dedicated per-job setup function so the outer loop only orchestrates jobs instead of building data structures inline.

DONE: Extract model setup / model reload decisions into a helper so the outer loop does not mix "create model once" and "load model for this job" logic.

DONE: Split final test/export logic into a separate function. That will make the outer loop visibly branch into either `run_training_job(...)` or `run_test_job(...)`.

#TODO: Introduce an explicit outer live state for `train_all_phases()` that owns only long-lived orchestration values such as `job_list`, current job index, shared model, and the next epoch anchor.

#TODO: Make the boundary between recreated inputs and checkpoint-worthy state explicit in one place, instead of scattering it across `train_all_phases()`, `train_one_phase()`, and `restart_state.py`.

DONE: In `train_one_phase()`, split setup from execution enough to introduce a dedicated runtime-preparation helper.
DONE: `prepare_phase_runtime(...)` -> build optimizer/schedulers/scaler/early_stopping/paths
#TODO: `run_phase_epochs(...)` -> own the epoch loop only

#TODO: Inside the epoch loop, extract smaller steps for:
#TODO: `compute_phase_stop_flags(...)`
#TODO: `run_training_epoch_or_eval_epoch(...)`
#TODO: `handle_nan_recovery(...)`
#TODO: `evaluate_phase_contexts(...)`
#TODO: `update_phase_control_state(...)`
#TODO: `save_phase_restart_checkpoint(...)`

#TODO: Replace underscore-prefixed multi-purpose locals such as `_epoch_0`, `_phase_epoch_0`, `_flag_break_after_epoch`, and `_batches_this_phase` with named state fields or clearly scoped helper-return values.

#TODO: Define one small section in code that lists "runtime-only fields" versus "checkpoint fields" so future restart work does not have to rediscover the save boundary by reading multiple functions.

#TODO: Avoid mutating `train_cfg` in many places during execution unless the mutation is part of explicit phase state. If a mutation must survive resume, it should be represented as saved state; otherwise it should stay local.

#TODO: Make eval-only epochs an explicit concept with a helper or named phase-step object instead of encoding that behavior via `first_epoch_is_evaluation` plus several flags spread through the loop.

## Feedback focus for the next step

Before implementation, please review the proposed extraction boundaries with special attention to:

1. which outer-loop values should become explicit orchestration state
2. which inner-loop values should remain local helper variables
3. which currently saved values should stop being saved in the restart redesign
4. whether the suggested helper boundaries match how you want to read the trainer
