# Trainer Alignment Report: old (d6ffb64) vs new (HEAD of `modelica_export_copilot`)

---

## Executive Summary

The new version of `trainer.py` is a substantial, well-motivated refactoring of the
old version. The most significant addition is a complete **cross-run restart/resume
system** (`TrainAllPhasesState`, `TrainOnePhaseState`, `RestartCheckpointStore`,
`CheckpointRequestedExit`) that the old version entirely lacked. This fills the most
important design gap identified in the old code's own docstring. The refactoring also
extracts roughly a dozen module-level helper functions (`_build_job_list`,
`_create_datasets_and_dataloaders_for_job`, `_initialize_or_reload_model_for_job`,
`_run_test_job`, `_compute_phase_epoch_settings`, `_build_phase_checkpoint_paths`,
`_create_phase_optimizer`, `_create_phase_lr_schedulers`) from the monolithic
`train_all_phases` and `train_one_phase`, meaningfully reducing complexity.

The new version is safer in several concrete ways: raw `mlflow.*` calls are replaced by
the null-safe `mlflow_proxy` wrapper; the early-stopping secondary metric selection is
generalised so models without `rmse_states_outputs` degrade gracefully to `rmse_states`
or `rmse_outputs`; and the `break_after_loss_of` None-guard that existed in the old
code is preserved. The `test` parameter has been removed from `train_one_phase`
(test jobs are now handled entirely via `_run_test_job`), which eliminates a dual-mode
function that was confusing to reason about.

The biggest risks going into merge review are: (1) the `test` parameter removal from
`train_one_phase` is an interface break — any external caller that passes `test=True`
will fail; (2) the per-epoch restart checkpoint is correctly skipped when
`flag_break_after_epoch` is True (i.e., at phase end), but the `TrainAllPhasesState`
still writes the *previous* epoch's `job_idx` on every normal epoch, so a kill during
the last normal epoch of a phase would attempt to resume from the last completed epoch
of that phase, which is correct; (3) the long-standing CUDA memory metric bug
(`1024^3` is bitwise XOR in Python, not `1024**3`) is **not fixed** — it is carried
into the new version unchanged; (4) one bare `mlflow.active_run()` call (line ~1632)
remains despite the transition to `mlflow_proxy`, creating a subtle inconsistency where
run-ID capture could fail silently in environments without an active MLflow run.

---

## Diff Statistics

| Metric | Value |
|---|---|
| Old file (`trainer_old.py`) | **1 332 lines** |
| New file (`trainer_new.py`) | **1 685 lines** |
| Net addition | **+353 lines** |
| Lines removed (raw diff `<` count) | **654** |
| Lines added (raw diff `>` count) | **1 006** |
| Diff hunks (`---`) | **37** |

The majority of removed lines were relocated into new helper functions; almost all of
the added lines are either those extracted helpers or the new restart-state machinery.

---

## Section-by-Section Comparison

---

### 1. Entry Point (`main()`)

**What changed**

- `main()` itself is unchanged in logic (3-step pattern: config store, auto-recognize,
  Hydra wrap). It moved from line 1307 to line 1659.
- `import os` and `from typing import TYPE_CHECKING` (lines 165, 190 in old) are
  removed from the top-level imports. `os` is no longer used anywhere in the new file
  (confirmed by grep).
- New top-level imports added:
  - `from bnode_core.utils.mlflow_proxy import mlflow_proxy`
  - `from bnode_core.ode.trainer_utils.restart_state import CheckpointRequestedExit, TrainAllPhasesState, TrainOnePhaseState, capture_rng_state, restore_rng_state`
  - `from bnode_core.ode.trainer_utils.restart_checkpoint_store import RestartCheckpointStore`
  - `from bnode_core.ode.trainer_utils.restart_utils import _clear_restart_state, load_restart_state_pair`
- `import mlflow` is kept (needed for one bare `mlflow.active_run()` call at line ~1632
  of the new file).

**Severity:** 🟢 Low — no behavioral change to `main()` itself.

**Alignment assessment:** Correct. The import cleanup is tidy and the new imports are
all required by the new features. The residual bare `mlflow` import is needed for the
one surviving `mlflow.active_run()` call (see §6 for details).

---

### 2. Initialization

**What changed**

- `initialize_model()` (line 198 old → line 218 new): no logic changes. The function
  signature and body are identical in behavior.
- **Dataset loading** (`train_all_phases`): the inline dataset-creation block (old
  lines ~400–520) is extracted into `_create_datasets_and_dataloaders_for_job()` (new
  line 399) plus helper functions `_job_dataset_loading_settings()` (new line 377) and
  `_log_job_start()` (new line 368).
- **`seq_len_train` read-back bug fix**: old code uses
  `datasets['train'].datasets['time'].shape[2]` (hardcoded axis 2); new code uses
  `datasets['train'].datasets['time'].shape[-1]` (last axis). This is a correctness
  fix for the fallback branch when `TimeSeriesDataset` does not expose a `seq_len`
  attribute.
- **Model initialization condition** in `train_all_phases` outer loop: old checks
  `if model is None`; new checks `if model is None or cfg.nn_model.training.load_trained_model_for_test is True`. This ensures the trained model is reloaded from disk on each CUDA OOM retry when `load_trained_model_for_test=True`.
- **Optimizer construction** extracted to `_create_phase_optimizer()` (new line 1133).
- **LR scheduler construction** extracted to `_create_phase_lr_schedulers()` (new
  line 1173). The `AMP GradScaler` is now initialized inside `train_one_phase()` (new
  line ~1261) rather than inline.

**Severity:** 🟡 Medium — the `seq_len_train` shape-index fix changes behavior on the
fallback path; the model-reload condition change affects CUDA OOM retry semantics.

**Alignment assessment:** The new version correctly and completely implements what the
old version intended, and fixes two latent bugs. No gaps observed.

---

### 3. Outer Loop (`train_all_phases`)

**What changed**

- **Job list construction** extracted to `_build_job_list(cfg)` (new line 340). Logic
  is identical to the old inline version.
- **Restart-resume wiring** (new, no old equivalent):
  - `load_restart_state_pair(job_list=job_list)` is called at the top of
    `train_all_phases()` (new line ~776).
  - `job_start_idx = train_all_phases_state.job_idx` (instead of always 0).
  - The `for idx, job in enumerate(job_list)` becomes
    `enumerate(job_list[job_start_idx:], start=job_start_idx)`, so completed phases
    are skipped entirely on resume.
  - `_clear_restart_state()` is called after all jobs complete (new line ~184).
- **`CheckpointRequestedExit` handling** (new): the outer `while True:` retry block now
  also catches `CheckpointRequestedExit` and returns immediately — enabling clean
  scheduler-driven preemption.
- **`_epoch_0` renamed to `next_epoch_anchor`**: same semantics, clearer name.
- **`test` job execution** moved from inside `train_one_phase` (old) to
  `_run_test_job()` (new, line 554) called directly from `train_all_phases`.
- **Phase sequencing** (seq_len handoff): logic unchanged — new sets
  `job_list[idx+1]['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train if
  job['pre_train'] is False else 1`.

**Severity:** 🟡 Medium — restart-resume path is new behavior but is purely additive;
fresh starts follow the same code path as before (`train_all_phases_state.job_idx`
defaults to 0).

**Alignment assessment:** Correctly implements the old outer loop, plus adds restart.
One subtle difference: `train_one_phase_state` is set to `None` after the first resumed
phase (new line ~144), so only the directly-resumed phase gets state injection; all
subsequent phases start fresh. This is correct and intentional.

---

### 4. Inner Loop (`train_one_epoch`, `train_one_phase` epoch loop)

**What changed**

- **`train_one_epoch()` is unchanged** in logic (moved from line 732 to line 901).
  Signature is identical. Persistent iterator, curriculum truncation, AMP/LBFGS
  branching, gradient clipping, cosine LR stepping, return dict — all preserved.
- **`train_one_phase()` signature change** (line 924 old → line 1228 new):
  - `test: bool` parameter **removed** (test jobs no longer enter `train_one_phase`).
  - Added parameters: `train_one_phase_state`, `train_all_phases_state`,
    `outer_restart_state_path`, `inner_restart_state_path`.
- **Internal flag rename**: `_flag_first_epoch_this_phase` → `first_epoch_is_evaluation`.
  New semantics: on resume, `first_epoch_is_evaluation = (train_one_phase_state is
  None)`, so a resumed phase does NOT re-run the baseline evaluation epoch, correctly
  picking up where it left off.
- **`epoch_stop` variable**: old uses `epoch_0` (accumulating global counter); new uses
  `phase_epoch_0 + max_epochs` (derived from the phase anchor). Behavior equivalent.
- **`_flag_break_after_epoch` → `flag_break_after_epoch`**: renamed (underscores
  dropped), behavior identical.
- **Per-epoch restart checkpoint** (`checkpoint_store.save_epoch_checkpoint(...)`) is
  called at the bottom of the epoch loop **after** the `if flag_break_after_epoch:
  break` guard (new lines ~1634–1642). This means the restart checkpoint is **not**
  written on the terminal epoch of a phase — intentional: a completed phase should
  not be resumable.

**Severity:** 🔴 High — removing `test: bool` from `train_one_phase` is a **breaking
interface change**. Any external code (tests, scripts) calling `train_one_phase(...,
test=True, ...)` will receive a `TypeError`.

**Alignment assessment:** The non-test training path is correctly preserved. The
restart-state injection at epoch-loop startup (lines ~1286–1335) is thorough and
validates all four checkpoint files before proceeding. The "no restart checkpoint on
terminal epoch" design is correct.

---

### 5. Validation / Evaluation Logic

**What changed**

- `test_or_validate_one_epoch()` (line 889 old → line 1066 new): **unchanged** in
  logic and signature.
- `append_context_to_dict_keys()` (line 918 old → line 1095 new): **unchanged**.
- **Early stopping secondary metric** (significant behavioral change):
  - Old (line 1184): hardcoded
    `corresponding_loss=ret_vals_validation['rmse_states_outputs']`.
    Raises `KeyError` if that key is absent from the metrics dict.
  - New: `_get_early_stopping_corresponding_metric(ret_vals_validation)` (new line
    210) tries `'rmse_states_outputs'`, then `'rmse_states'`, then `'rmse_outputs'`
    in order, returning `(None, None)` if none are present. The
    `corresponding_loss=corresponding_metric_value` call is guarded by
    `if early_stopping_metric_name is not None`.
  - The MLflow metric key logged also changes: old always logs
    `best_rmse_states_outputs`; new logs `best_{early_stopping_metric_name}` (e.g.,
    `best_rmse_states` for a NODE-only run without `rmse_states_outputs`).
- **`activate_deterministic_mode` flag** (new line ~1462): unchanged logic — set on
  `flag_break_after_epoch and train_cfg.activate_deterministic_mode_after_this_phase`.

**Severity:** 🟡 Medium — the metric selection change is behavioral (different MLflow
key name, different fallback behavior) but is an improvement, not a regression.

**Alignment assessment:** Correctly implements the old behavior for the common case
(BNODE with `rmse_states_outputs`). The fallback path is a genuine improvement for
NODE-only runs. MLflow metric key name change (`best_rmse_states_outputs` →
`best_{metric_name}`) may break downstream dashboards or automated comparisons.

---

### 6. Checkpointing / State Persistence

**What changed**

- **Rolling model / optimizer checkpoints** (`model.save(path_current_model)`,
  `torch.save(optimizer.state_dict(), path_current_optimizer)`) logic unchanged (new
  lines ~1456–1457 = old lines ~1142–1143). Written after every non-NaN epoch.
- **Phase-best model / optimizer** written by `EarlyStopping`: unchanged.
- **`_build_phase_checkpoint_paths(pre_train, job_idx)`** (new function, line 1125):
  consolidates the four checkpoint paths into one call, replacing repeated inline
  `filepaths.*` calls throughout `train_one_phase`.
- **New restart checkpoint files** (`training_outer_restart.pt`,
  `training_inner_restart.pt`, `lr_schedulers.pt`, `grad_scaler.pt`) written atomically
  by `checkpoint_store.save_epoch_checkpoint(...)` at the end of each non-terminal
  epoch (new lines ~1625–1641). These have no equivalent in the old version.
- **`mlflow.log_param` → `mlflow_proxy.set_tag_if_active`** for termination reason:
  old uses `mlflow.log_param('job {} ended by')`, new uses
  `mlflow_proxy.set_tag_if_active('job {} ended by')`. MLflow tags vs parameters is a
  semantic difference — tags are mutable and appear separately in the MLflow UI.
- **One bare `mlflow.active_run()` call remains** (new line ~1632) inside
  `train_one_phase` to read the run ID for the `TrainAllPhasesState.mlflow_run_id`
  field. This call is **not** guarded by `mlflow_proxy`, so if no MLflow run is active,
  `mlflow.active_run()` returns `None` (safe), but it still imports raw `mlflow`
  bypassing the proxy abstraction.
- **CUDA memory metric bug inherited**: `torch.cuda.memory_reserved()/(1024^3)` — `^`
  is Python bitwise XOR (`1024 XOR 3 = 1027`), not exponentiation. The reported value
  is in bytes divided by 1027, not GiB. **Present in both old and new versions.**

**Severity:** 🟡 Medium — new restart checkpoint files are purely additive. The
`log_param` → `set_tag_if_active` change is a behavioral difference in how MLflow
records termination reasons. The bare `mlflow.active_run()` call is a minor consistency
issue. The CUDA XOR bug is inherited (not introduced).

**Alignment assessment:** Correctly implements all old checkpoint behavior. New restart
checkpoints add correctness guarantees (UUID cross-linking, atomic writes). No gaps.

---

### 7. Restart / Resume Logic

**What changed**

- **Old version**: no cross-run resume. §7 of the old flow document documents this as a
  known design gap. Resume was limited to NaN recovery (reload rolling checkpoint) and
  pretrained/trained-model loading across runs.
- **New version**: full epoch-granular, cross-run resume via:
  - `TrainAllPhasesState` (`restart_state.py`): stores `job_idx`, `next_epoch_anchor`,
    `mlflow_run_id`, `checkpoint_uuid` as typed PyTorch buffers.
  - `TrainOnePhaseState` (`restart_state.py`): stores `phase_epoch`, `nan_counter`,
    `grad_norm_last_reduced_counter`, `stable_epochs`, `deterministic_mode_active`,
    `seq_len_increase_in_batches`, complete RNG state, `checkpoint_uuid`, full
    `EarlyStopping` state.
  - `RestartCheckpointStore.save_epoch_checkpoint()`: atomic write of all four files
    with UUID integrity cross-link.
  - `load_restart_state_pair()`: detects and validates existing checkpoint pair on
    startup.
  - `_clear_restart_state()`: cleans up all four files on clean completion.
- **NaN recovery logic**: unchanged in behavior (reload rolling checkpoint for
  `nan_counter <= 49`, reload best for `49 < nan_counter <= 55`, raise at > 55).
  The counter is now stored in `phase_state.nan_counter` (persisted across restarts)
  instead of a local variable (reset on every cold start).
- **Restart limitations documented** in new flow (§7.5): pre-training and test jobs
  cannot be restarted; LR scheduler type mismatch on resume is fatal (raises
  `ValueError`).

**Severity:** 🟡 Medium — all new behavior, no regression risk from old paths. The
`nan_counter` now persisting across restarts is a subtle behavioral difference: a run
that was killed after 30 NaN epochs will resume with `nan_counter=30`, not 0.

**Alignment assessment:** Fully addresses the major design gap in the old version.
The NaN counter persistence is probably desirable (avoids re-running the NaN budget on
resume) but is not called out in the documentation and should be verified intentional.

---

### 8. Sequence-Length Curriculum

**What changed**

- `train_one_epoch` curriculum truncation logic: **unchanged** (same `_batches_this_phase`
  interpolation, same `shape[:, :, :_seq_len_now]` slicing).
- `_compute_phase_epoch_settings()` (new function, line 1102): extracts the
  `batches_per_epoch` / `epochs_for_seq_len_increase` / `max_epochs` computation from
  the old inline block at lines ~970–990. Logic identical, but now uses `math.ceil`
  explicitly for `epochs_for_seq_len_increase` (old used `int(...)` which is floor).
  This can result in one extra epoch being budgeted when `seq_len_increase_in_batches`
  is not divisible by `batches_per_epoch` — a minor behavioral difference.
- **Stable-epoch counting** now via `phase_state.stable_epochs` (int buffer on
  `TrainOnePhaseState`) instead of local `_stable_epochs`. This means stable-epoch
  count **persists across restarts** — consistent with persisting `nan_counter`.
- **`flag_out_of_seq_len_increase` initialization on resume**: new code initializes
  it from `batches_completed_before_resume > train_cfg.seq_len_increase_in_batches`
  (new lines ~1347–1360), ensuring that a resumed run does not incorrectly re-enter the
  curriculum window.
- **`epoch_stop` extension on curriculum abort**: old uses `epoch_0` as base; new uses
  `phase_epoch_0`. Equivalent when `phase_epoch_0 == epoch_0` (single-phase) but
  correctly handles multi-phase restart where they differ.

**Severity:** 🟡 Medium — `math.ceil` vs `int` for `epochs_for_seq_len_increase` is a
minor behavioral change (one extra epoch in edge cases). Curriculum state persistence
across restart is intentional and correct.

**Alignment assessment:** Correctly implements the old curriculum logic with proper
restart-resume support. The `ceil` change may give marginally different epoch budgets
but does not affect training correctness.

---

### 9. Early Stopping / Convergence Criteria

**What changed**

- `EarlyStopping` instantiation: **unchanged** — same `patience`, `threshold`,
  `threshold_mode`, `path`, `optimizer_path`, `trace_func` arguments.
- **Four termination flags** (max epoch, early stopping, break-after-loss, NaN counter):
  logic **unchanged**. Renamed from `_flag_*` to `flag_*` (underscores dropped).
- **`early_stopping.early_stop` serialization**: `EarlyStopping` is now a
  `torch.nn.Module` with `get_extra_state()` / `set_extra_state()`, and is stored
  inside `TrainOnePhaseState`. On resume, the patience counter and best score are
  restored to exactly where they were — the old version would always restart patience
  from zero on a fresh cold start.
- **`_flag_break_after_loss_of` → `flag_break_after_loss`**: renamed. Guard logic for
  `None` values **identical** between old and new.
- **MLflow termination logging**: old uses `mlflow.log_param(...)`; new uses
  `mlflow_proxy.set_tag_if_active(...)`. Parameters are immutable in MLflow; tags are
  mutable. This changes how termination reason appears in experiment tracking.

**Severity:** 🟢 Low — behavior is identical for normal (non-resume) runs. The
serialization of `EarlyStopping` state is a pure addition.

**Alignment assessment:** Correct. The resume-aware patience counter is the key
improvement. MLflow tag vs param change is a deliberate UX improvement (tags appear
prominently in the MLflow run view; params do not).

---

### 10. Helpers / Utilities

**What changed**

- **New module-level helper functions** (all extracted from inline blocks; logic preserved):
  - `_get_early_stopping_corresponding_metric()` (line 210)
  - `_build_job_list()` (line 340)
  - `_log_job_start()` (line 368)
  - `_job_dataset_loading_settings()` (line 377)
  - `_create_datasets_and_dataloaders_for_job()` (line 399)
  - `_initialize_or_reload_model_for_job()` (line 493)
  - `_run_test_job()` (line 554)
  - `_compute_phase_epoch_settings()` (line 1102)
  - `_build_phase_checkpoint_paths()` (line 1125)
  - `_create_phase_optimizer()` (line 1133)
  - `_create_phase_lr_schedulers()` (line 1173)
- **New utility modules** (imported, not defined here):
  - `mlflow_proxy` — null-safe MLflow wrapper
  - `TrainAllPhasesState`, `TrainOnePhaseState`, `capture_rng_state`,
    `restore_rng_state`, `CheckpointRequestedExit` (from `restart_state.py`)
  - `RestartCheckpointStore` (from `restart_checkpoint_store.py`)
  - `_clear_restart_state`, `load_restart_state_pair` (from `restart_utils.py`)
- `_next_batch()` (line 712 old → line 881 new): **unchanged**.
- `append_context_to_dict_keys()` (line 918 old → line 1095 new): **unchanged**.
- **TODO comment** on `_initialize_or_reload_model_for_job()`: explicitly marked in
  new flow document (§7.5). The function cannot be shared with resume initialization
  because resume loads from `path_current_model`, while fresh start may load from
  `path_pretrained_model` or `path_trained_model`.

**Severity:** 🟢 Low — pure refactoring, no behavioral change.

**Alignment assessment:** Extraction is clean. All new helpers are covered by the new
flow document. The TODO is accurately reflected in both source and documentation.

---

## Summary Table

| Section | Change Summary | Severity | Risk Notes |
|---|---|---|---|
| 1. Entry Point | Import cleanup; new restart imports | 🟢 Low | `mlflow` import kept for one bare call |
| 2. Initialization | Extracted to helpers; `seq_len` fallback bug fixed; `shape[-1]` vs `shape[2]` | 🟡 Medium | Fallback-path behavior change; model reload condition on OOM retry changed |
| 3. Outer Loop | Extracted helpers; restart wiring; `_epoch_0` → `next_epoch_anchor`; test job extracted | 🟡 Medium | `job_start_idx` now from state (fresh = 0, so safe) |
| 4. Inner Loop | `test` param removed from `train_one_phase`; restart injection; `first_epoch_is_evaluation` semantics on resume | 🔴 High | Breaking interface change for `test` param; no restart checkpoint on terminal epoch (intentional) |
| 5. Validation | Early-stopping metric generalised via `_get_early_stopping_corresponding_metric` | 🟡 Medium | MLflow metric key name changes from `best_rmse_states_outputs` to `best_{metric_name}` |
| 6. Checkpointing | New restart checkpoint files; `log_param` → `set_tag_if_active`; one bare `mlflow.active_run()` remains | 🟡 Medium | CUDA XOR metric bug inherited; tag vs param semantic difference |
| 7. Restart / Resume | Entirely new; full epoch-granular cross-run resume | 🟡 Medium | `nan_counter` and `stable_epochs` now persist across restarts (intentional but undocumented) |
| 8. Curriculum | `_compute_phase_epoch_settings` extracted; `ceil` vs `int`; state persisted across restarts | 🟡 Medium | One extra epoch possible in edge case; `flag_out_of_seq_len_increase` correctly restored on resume |
| 9. Early Stopping | Flags renamed; `EarlyStopping` state now serializable and restored on resume; `log_param` → `set_tag_if_active` | 🟢 Low | Pure improvement; no regression |
| 10. Helpers / Utilities | 11 new module-level functions extracted; 3 new utility module imports | 🟢 Low | Pure refactoring; logic preserved |

---

## Notable Bugs / Design Gaps Introduced or Fixed

### Fixed in the new version

1. **`seq_len_train` shape-index bug** (§2): old code used `datasets['train'].datasets['time'].shape[2]` (hardcoded axis), new uses `shape[-1]` (last axis). On a dataset where the time axis is not position 2, old code would silently use the wrong dimension as the sequence length.

2. **`KeyError` on missing `rmse_states_outputs`** (§5): old code unconditionally
   accessed `ret_vals_validation['rmse_states_outputs']` for the early-stopping
   corresponding metric. For NODE models whose metrics dict does not include that key,
   this would raise a `KeyError` each epoch. New code uses the fallback helper.

3. **Zero restart capability** (§7): the entire cross-run resume system is new. The
   old version's docstring explicitly noted this as a design gap.

4. **`EarlyStopping` patience not persisted across interrupts** (§9): old version
   would always restart patience from zero after a kill-and-resume (via manual model
   loading). New version serializes and restores `EarlyStopping` state.

5. **`nan_counter` not persisted across interrupts** (§7): old version would restart
   the NaN budget from 0 after a kill. New version restores it from `TrainOnePhaseState`.

### Not fixed (inherited from old version)

6. **CUDA memory metric XOR bug** (§6): `torch.cuda.memory_reserved()/(1024^3)` — `^`
   is Python bitwise XOR. The value logged as `CUDA_memory_reserved_GB` is actually
   bytes ÷ 1027 (not GiB). Fix: replace with `/(1024**3)` or `/(1024*1024*1024)`.
   Present at old line 820 and new line 989.

7. **MLflow tag typo** (§9): `'4 NaNs in loss'` is logged when `nan_counter > 50`.
   The comment in old code even says "if more than 25 NaNs in loss" — three different
   numbers in the same block. Carried unchanged into new version (new line ~1379).

8. **Pre-training not guarded for BNODE** (§3): Pre-training with a BNODE model is
   unsupported (documented in the module docstring) but there is no runtime guard.
   Unchanged in new version.

9. **`batch_size_test` mutated on global config** (§3): On CUDA OOM during a test job,
   `cfg.nn_model.training.batch_size_test` is modified in-place. Because `cfg` is a
   reference to the Hydra config object, this mutation persists across retries and
   potentially across test invocations in the same process.

### Newly introduced in the new version

10. **One bare `mlflow.active_run()` call bypasses `mlflow_proxy`** (§6): At new line
    ~1632, `mlflow.active_run().info.run_id` is called directly (to populate
    `TrainAllPhasesState.mlflow_run_id`). If `mlflow.active_run()` returns `None`
    (possible when no MLflow run is active), the expression `None.info.run_id` raises
    `AttributeError`. The old pattern guarded this with `if mlflow.active_run() is not
    None else None` already shown in the surrounding code — but the attribute access is
    still before the `None` check: `mlflow.active_run().info.run_id if
    mlflow.active_run() is not None else None`. This calls `active_run()` twice and
    has a race condition where the run could be ended between the two calls. In
    practice this is low risk but is worth fixing.

11. **LR scheduler type mismatch on resume is fatal** (new §7.5): if `lr_scheduler_type`
    is changed in config between an interrupted run and a resume attempt, the scheduler
    key validation in `train_one_phase` raises `ValueError` with no migration path.
    This is documented in the new flow document but may surprise users who tweak configs
    across runs.

12. **`test` parameter removal breaks callers** (§4): `train_one_phase(... test=True
    ...)` was the old API for running the test job. Any caller outside `train_all_phases`
    that used this interface (e.g., tests or notebooks) will now receive `TypeError:
    unexpected keyword argument 'test'`.

---

## Recommendations

Ordered by priority:

1. **[🔴 Immediate] Fix the bare `mlflow.active_run()` AttributeError risk** (new line
   ~1632). Replace:
   ```python
   mlflow.active_run().info.run_id if mlflow.active_run() is not None else None
   ```
   with:
   ```python
   run = mlflow.active_run()
   run.info.run_id if run is not None else None
   ```
   or route through `mlflow_proxy` if a proxy method for this is added.

2. **[🔴 Immediate] Audit all callers of `train_one_phase`** for the removed `test`
   parameter. Search tests, notebooks, and scripts for `train_one_phase(..., test=True,
   ...)` or positional calls with the old 8-argument signature. Update or remove.

3. **[🟡 Before merge] Fix the CUDA memory metric bug** in both files — change
   `/(1024^3)` to `/(1024**3)` (new line 989). This is a 3-character fix that corrects
   a misleading metric that has been wrong since the old version. File separately in
   `bnode-core` so it can be tracked.

4. **[🟡 Before merge] Document `nan_counter` and `stable_epochs` persistence semantics**
   in the restart-state architecture docs (`.github/instructions/trainer-restart.instructions.md`
   and the MkDocs restart page). Engineers need to know that these counters accumulate
   across restarts, not just within a single run, so they can reason about NaN budget
   exhaustion correctly.

5. **[🟡 Before merge] Check downstream MLflow dashboards** for the metric key rename:
   `best_rmse_states_outputs` → `best_{metric_name}`. If any automated comparison
   scripts, Grafana boards, or CI checks query the old key name, they will silently
   get no data after this merge.

6. **[🟡 Before merge] Verify `math.ceil` vs `int` for `epochs_for_seq_len_increase`**
   does not materially affect any existing training configs. For configs where
   `seq_len_increase_in_batches % batches_per_epoch != 0`, the new version budgets one
   additional epoch. This should be harmless but should be confirmed against reference
   training runs.

7. **[🟢 Follow-up] Add a runtime guard for BNODE + pre-training** in
   `train_all_phases` or `_build_job_list`: raise `ValueError` if
   `cfg.nn_model.training.pre_train=True` and `cfg.nn_model.model_type='bnode'`. The
   docstring note is insufficient protection.

8. **[🟢 Follow-up] Fix `batch_size_test` global mutation** on OOM retry. Use a local
   variable for the retry batch size and pass it explicitly, rather than mutating
   `cfg.nn_model.training.batch_size_test` in-place. The outer retry loop already
   tracks `retry_batch_size` as a local variable; it just also writes back to `cfg`.

9. **[🟢 Follow-up] Fix the `'4 NaNs in loss'` MLflow tag** (new line ~1379). Change
   the message to `'50 NaNs in loss'` or `'nan_counter > 50'` to match the actual
   threshold used in `flag_nan_counter = phase_state.nan_counter > 50`.

10. **[🟢 Follow-up] Resolve the TODO in `_initialize_or_reload_model_for_job`** (new
    line ~599). The resume path loads the model inside `train_one_phase` from
    `path_current_model`, while the cold-start path uses `_initialize_or_reload_model_for_job`.
    Consider whether a unified function with a `resume: bool` parameter would reduce
    the duplication and make the two paths easier to keep in sync.
