# trainer.py current flow

This file is a pseudo-code map of the **current** `trainer.py` flow after the restart-state simplification.

## Main entry points

### `main()`

- registers the Hydra config store
- resolves the config root
- runs `train_all_phases(cfg)`

### `train_all_phases(cfg)`

Responsibilities:

- loads datasets and optional norm/ref datasets
- builds the outer `job_list`
- loads `train_all_phases_state` and `train_one_phase_state` if both restart files exist
- recreates outer-loop locals (`job_start_idx`, `next_epoch_anchor`, paths, model state)
- creates per-job datasets and dataloaders
- creates or reloads the model
- dispatches to `train_one_phase(...)` or final test logic
- handles CUDA-memory retry loop
- clears the restart files after all jobs finish

Important persisted state:

- `train_all_phases_state.job_idx`
- `train_all_phases_state.next_epoch_anchor`
- `train_all_phases_state.mlflow_run_id`

Important locals:

- `job_list`
- `outer_restart_state_path`
- `inner_restart_state_path`
- `job_start_idx`
- `next_epoch_anchor`
- `model_created`
- `model`
- `datasets`
- `dataloaders`

Pseudo-code:

```text
def train_all_phases(cfg):
    load datasets
    job_list = build outer workflow

    train_all_phases_state, train_one_phase_state, outer_path, inner_path = load restart pair
    job_start_idx = train_all_phases_state.job_idx or 0
    next_epoch_anchor = train_all_phases_state.next_epoch_anchor or 0

    for idx, job in job_list[job_start_idx:]:
        while True:
            try:
                if not job["skip"]:
                    datasets = build datasets
                    dataloaders = build dataloaders
                    model = initialize or reload model

                if job["skip"]:
                    log skip
                elif not job["test"]:
                    phase_restart_state = train_one_phase_state only for the resumed job
                    next_epoch_anchor = train_one_phase(
                        cfg, model, dataloaders, train_cfg,
                        job_idx=idx,
                        epoch_0=next_epoch_anchor,
                        train_one_phase_state=phase_restart_state,
                        outer_restart_state_path=outer_path,
                        inner_restart_state_path=inner_path,
                    )
                    clear in-memory restart states
                    maybe seed next job seq_len_epoch_start
                else:
                    run final evaluation

                break
            except CUDA-memory error:
                shrink batch size and retry

    delete both restart files
```

### `train_one_epoch(...)`

Responsibilities:

- runs the batch loop for one training epoch
- performs sequence-length warmup slicing
- runs optimizer step with Adam or LBFGS
- steps per-batch schedulers
- returns final batch metrics and updated iterator

Important locals:

- `epoch_this_phase`
- `_batches_this_phase`
- `_seq_len_now`
- `ret_vals_train`
- `_norm`

### `train_one_phase(...)`

Responsibilities:

- recreates local phase-control variables instead of using a live wrapper object
- builds optimizer, scheduler, scaler, and early stopping helpers
- explicitly loads `model.pt` on resume
- restores optimizer / scheduler / scaler / early-stopping / RNG state from `train_one_phase_state`
- runs the epoch loop
- saves both restart-state files at epoch boundaries
- returns the next global epoch anchor for the next outer job

Important persisted state:

- `train_one_phase_state.phase_epoch`
- optimizer / scheduler / scaler / early-stopping state
- `nan_counter`
- `grad_norm_last_reduced_counter`
- `stable_epochs`
- `rng_state`
- `deterministic_mode_active`
- `seq_len_increase_in_batches`

Important locals:

- `phase_epoch_0`
- `epoch_stop`
- `first_epoch_is_evaluation`
- `flag_out_of_seq_len_increase`
- `path_best_model`
- `path_optimizer_best_model`
- `path_current_model`
- `path_current_optimizer`
- `batches_per_epoch`
- `max_epochs`

Pseudo-code:

```text
def train_one_phase(...):
    build local checkpoint paths
    optimizer = create optimizer
    early_stopping = create helper
    scaler = create scaler
    batches_per_epoch, max_epochs = derive phase settings
    lr_schedulers = create schedulers

    if train_one_phase_state exists:
        load model.pt
        restore optimizer/scheduler/scaler/early-stopping/RNG from state

    recreate local flags and counters

    for epoch in range(epoch_0, phase_epoch_0 + max_epochs):
        maybe break on max epoch / early stopping / loss / NaN rules
        maybe extend epoch_stop when seq-len increase ends early

        if not first_epoch_is_evaluation and not breaking:
            train_one_epoch(...)
            handle NaN reload logic
            save current model/optimizer
        else:
            run eval-only train pass
            maybe activate deterministic mode

        run validation / test / ref / testnorm evaluations
        update early stopping and stable epoch counters
        maybe mark seq-len increase as finished
        save train_one_phase_state and train_all_phases_state

    return epoch + 1
```
