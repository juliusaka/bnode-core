# Trainer Alignment Report: Old (d6ffb64) vs New (2ab3859)

> **Compares:** `trainer_flow_old.md` (git commit d6ffb64, branch `modelica_export_copilot`)  
> **Against:** `trainer_flow_new.md` (HEAD commit 2ab3859, branch `modelica_export_copilot`)  
> **Source:** `bnode/bnode-core/src/bnode_core/ode/trainer.py`  
> **Sections** follow the same ten headings as the flow documents.

---

## Executive Summary

| Category | Verdict |
|---|---|
| **Overall architecture** | Substantially refactored; same conceptual pipeline |
| **Training output (model weights)** | Potentially affected — see §6, §9 |
| **NaN recovery** | Changed source of rolling checkpoint; logic equivalent |
| **Restart/resume** | Completely redesigned; old four-file protocol replaced by single bundle |
| **Bugs fixed** | CUDA memory unit bug fixed; MLflow NaN tag fixed |
| **New runtime guard** | BNODE + pre-train now raises `ValueError` immediately |
| **OOM handling** | Local factor replaces global config mutation |

---

## 1. Entry Point

**No functional change.** `main()` still calls `config_store`, `config_dir_auto_recognize`, and wraps `train_all_phases` with `hydra.main`. The `@log_hydra_to_mlflow` decorator is identical.

---

## 2. Initialization

### 2.1 Dataset / DataLoader Creation

| Aspect | Old | New |
|---|---|---|
| OOM batch-size reduction | Mutated `cfg.nn_model.training.batch_size_test` globally, then called `_create_datasets_and_dataloaders_for_job` | `batch_size_reduction_factor` (local float, compounding ×0.7) passed as argument; config **not** mutated |
| batch_size_reduction warning | No extra warning | Warning logged after 30 OOM retries |

**Potential output impact:** With the old code, a CUDA OOM permanently reduced `batch_size_test` in the Hydra config object, which could affect subsequent phases if they read back `cfg.nn_model.training.batch_size_test`. With the new code the config is untouched; each phase starts fresh. This difference only matters in multi-phase OOM scenarios, but it was unintentional behaviour in the old version.

### 2.2 Model Construction

**No functional change** in the build paths. `normalization_init` and `.to(device)` are unchanged.

**New addition:** `_initialize_or_reload_model_for_job` accepts `restart_model_state` parameter. If resuming at a test job with no other load path configured, model weights are loaded from `restart_model_state` (from bundle) rather than raising an error or starting from random weights.

### 2.3 Optimizer

**No functional change.** `_create_phase_optimizer` construction is unchanged. Reload behaviour on `reload_optimizer=True` is unchanged.

### 2.4 LR Schedulers

**No functional change** in scheduler construction or stepping.

### 2.5 AMP Scaler

**No functional change.**

---

## 3. Outer Training Loop

### 3.1 `_build_job_list()`

| Aspect | Old | New |
|---|---|---|
| BNODE + pre-train guard | Silently skipped or behaviour undefined | Logs a warning and marks the pre-train job as skipped; training continues with main-training phases |

**New (potential output impact if relied on silently):** Any pipeline that previously used BNODE + `pre_train=True` and observed silent behaviour will now see a warning and the pre-train job will be skipped explicitly. No impact on normal usage.

### 3.2 `train_all_phases()` main loop

| Aspect | Old | New |
|---|---|---|
| `train_all_phases_state` initialization | Implicitly set by `load_restart_state_pair`; not documented what happens when None | `train_all_phases_state = outer_state or TrainAllPhasesState()` — always non-None, explicit |
| `save_outer_for_test_job` | Not present | Called just before running any test job; ensures bundle reflects correct `job_idx` |
| OOM detection string | `'CUDA out of memory'` | Both `'CUDA out of memory'` and `'CUDA memory is almost full'` |
| `checkpoint_store.clear_restart_artifacts()` | `_clear_restart_state()` (4 separate file deletions) | Single bundle file deletion |

### 3.3 Epoch Budget

| Aspect | Old | New |
|---|---|---|
| `epochs_for_seq_len_increase` formula | Documented as `math.ceil(...)` in old flow doc | Actual code uses `int(...)` (floor division) — **old flow doc was wrong** |

**The code in both old and new commits uses `int(...)` (floor).** The old `trainer_flow_new.md` (now replaced) incorrectly documented it as `ceil`. No behaviour change between commits; only the documentation was wrong.

---

## 4. Inner Training Loop

### 4.1 `train_one_phase()` Signature

| Aspect | Old | New |
|---|---|---|
| Signature | `(cfg, job, ..., train_one_phase_state, ...)` — no separate model/optimizer/scheduler state params | Added: `checkpoint_store`, `restart_scheduler_states`, `restart_scaler_state`, `restart_model_state`, `restart_optimizer_state` |
| Resume — model/optimizer loading | Loaded from separate `_restore_files` paths (rolling `.pt` files) | Loaded from `restart_model_state` / `restart_optimizer_state` (bundle dict keys) |
| Resume — inner state loading | Loaded from `training_inner_restart.pt` via `phase_state.load_from_state_dict(...)` separately | Loaded from `_bundle["inner"]` directly |

**Potential output impact:** The resumed model and optimizer state come from the same source (the atomic bundle), so the effective state should be identical. However, if the old code's separate files happened to be from a slightly different checkpoint than the inner state file, there was a potential inconsistency window. The new bundle is atomic — all state is from the same epoch. This means resumes with the new code are strictly correct; the old code had a potential off-by-one-epoch inconsistency.

### 4.2 Batch Loop, Gradient Computation

**No functional change** in the batch iteration, forward pass, backward pass, or gradient clipping logic.

### 4.3 CUDA Memory Metric Bug — **Fixed**

| Aspect | Old | New |
|---|---|---|
| Memory unit calculation | `/(1024^3)` (Python bitwise XOR → divides by 1027) → **wrong units** | `/(1024**3)` (exponentiation → correct GiB) |

**Impact:** The old code logged an incorrect CUDA memory value to MLflow (approximately 0.3% larger than actual GiB). The threshold comparisons (60%, 98%) were also computed on wrong values. In practice the thresholds are wide enough that this did not affect OOM trigger behaviour in any tested scenario, but the logged metric was wrong.

---

## 5. Validation / Evaluation Logic

**No functional change.** `test_or_validate_one_epoch`, `_get_early_stopping_corresponding_metric`, `activate_deterministic_mode`, and the ref/testnorm frequency logic are unchanged.

---

## 6. Checkpointing / State Persistence

This section has the **largest structural change** between old and new.

### 6.1 Restart Bundle: Four Files → Single Atomic Bundle

| Aspect | Old | New |
|---|---|---|
| Restart files | `training_outer_restart.pt`, `training_inner_restart.pt`, `lr_schedulers.pt`, `grad_scaler.pt` | `training_restart_checkpoint.pt` (bundle version 2) |
| Model/optimizer in restart | Separate rolling files `model_current.pt`, `optimizer_current.pt` written after every clean epoch | Embedded in the bundle: `_bundle["model"]`, `_bundle["optimizer"]` |
| Load function | `load_restart_state_pair()` → 4 values | `load_restart_checkpoint()` → 7 values (adds model_state, optimizer_state, checkpoint_store) |
| Atomicity | Per-file atomic write | Single bundle atomic write (all 7 components together) |
| Cleanup | `_clear_restart_state()` deletes 4 files separately | `checkpoint_store.clear_restart_artifacts()` deletes 1 file |

**Potential output impact:** The old code wrote `model_current.pt` and `optimizer_current.pt` independently of the inner/outer restart files. This created a window where the inner state could be from epoch N but model_current could be from epoch N+1 (if the epoch checkpoint completed model saving but not inner state saving before a crash). The new bundle is atomic — all state is from the same epoch. This means resumes with the new code are strictly correct; the old code had a potential off-by-one-epoch inconsistency.

### 6.2 `save_outer_for_test_job` — New Method

Old code had no mechanism to persist the outer state specifically for test job resume. New code calls `checkpoint_store.save_outer_for_test_job(train_all_phases_state)` before running a test job. This is an added capability.

### 6.3 `model_current.pt` / `optimizer_current.pt`

| Aspect | Old | New |
|---|---|---|
| Written | After every clean (non-NaN) training epoch | **Not written separately** — only included in the restart bundle |

**Implication:** If a user manually deletes the restart bundle, NaN recovery cannot fall back to the rolling model. The rolling model is only accessible via the bundle. This is a deliberate design simplification.

---

## 7. Restart / Resume Logic

| Aspect | Old | New |
|---|---|---|
| Bundle version validation | Not versioned | Bundle version checked: must be 2 (raises `ValueError` if not) |
| `_validate_restart_target` — test jobs | Test jobs were rejected (marked as non-restartable) | Test jobs **allowed**: only pre-train jobs are rejected |
| Model/optimizer loading on resume | Loaded from `path_current_model` / `path_current_optimizer` files | Loaded from `restart_model_state` / `restart_optimizer_state` in bundle |
| Scheduler loading on resume | Loaded from `lr_schedulers.pt` | Loaded from `restart_scheduler_states` dict in bundle |
| Scaler loading on resume | Loaded from `grad_scaler.pt` | Loaded from `restart_scaler_state` dict in bundle |

**Potential output impact (resume correctness):** Old code could have an inconsistency between `training_inner_restart.pt` (epoch N) and `model_current.pt` (potentially epoch N+1). New code is guaranteed consistent. On an identical (non-crashed) run, outcomes are the same.

**New capability (test job resume):** Previously a run interrupted during the test phase would have had to rerun from the last training phase. Now it resumes at the test job.

---

## 8. Sequence-Length Curriculum

**No functional change.** All curriculum logic, ramp formula, early abort, and epoch budget calculation are identical between old and new code. (The `int()` vs `ceil()` discrepancy was a documentation error in the old flow doc, not a code change.)

---

## 9. Early Stopping / Convergence Criteria

### MLflow NaN Tag — **Fixed**

| Aspect | Old | New |
|---|---|---|
| MLflow tag on NaN abort | `'4 NaNs in loss'` | `'50 NaNs in loss'` |

**Impact:** The old tag was incorrect (the threshold is `nan_counter > 50`, so the actual tag text was just mislabelled). No functional difference in training behaviour.

### NaN Recovery: Rolling Checkpoint Source

| Aspect | Old | New |
|---|---|---|
| Rolling model (nan_counter ≤ 49) | Load from `path_current_model` (file) | Load from `checkpoint_store.checkpoint_path` bundle, key `"model"` |
| Rolling optimizer (nan_counter ≤ 49) | Load from `path_current_optimizer` (file) | Load from bundle, key `"optimizer"` |
| Best model (nan_counter > 49) | `model.load(path_best_model)` + `torch.load(path_optimizer_best_model)` | Same (unchanged) |
| First-epoch failure | Unclear (no bundle exists yet) | `ValueError`: "cannot reload from bundle: no checkpoint" |

**Potential output impact:** If the `model_current.pt` from the old code was written slightly after the inner restart file (within the same epoch), there was a microsecond window of inconsistency on NaN reload. The new code reloads from the same bundle that was atomically written at end-of-epoch, so model + optimizer are always co-consistent on reload.

### Other Early Stopping Conditions

**No functional change.** `EarlyStopping` patience, threshold, `break_after_loss_of`, and `flag_max_epoch` logic are unchanged.

---

## 10. Helper Classes and Utilities

| Component | Old | New |
|---|---|---|
| `load_restart_state_pair` | Returns 4 values: outer, inner, scheduler_states, scaler_state | **Renamed** `load_restart_checkpoint`, returns 7 values (+model_state, +optimizer_state, +checkpoint_store) |
| `_validate_restart_target` | Rejected pre-train AND test jobs | Rejects only pre-train jobs |
| `_clear_restart_state` | Separate function in `restart_utils.py`, deletes 4 files | **Removed**; replaced by `checkpoint_store.clear_restart_artifacts()` |
| `RestartCheckpointStore` | Managed 4 separate files | Manages single bundle file; `save_outer_for_test_job` added; `checkpoint_path` property |
| `TrainAllPhasesState` | Same schema | Unchanged |
| `TrainOnePhaseState` | Same schema | Unchanged |
| `EarlyStopping` | Same interface | Unchanged |
| `mlflow_proxy` | Used throughout | Same; one bare `mlflow.active_run()` remains at line ~1679 (same as old) |

---

## Critical Differences that Could Affect Training Output

The following differences are **functionally significant** — they could produce different model weights, different training trajectories, or different convergence outcomes even given the same initial config and seed.

### A. NaN Recovery Source (§9)

**Old:** rolling model loaded from `model_current.pt` + inner state from `training_inner_restart.pt` — these could be from different epochs.  
**New:** both come from the atomic bundle — always co-consistent.

**Risk level: Low-to-Medium.** In the common case (no crash between model write and inner state write), they were the same epoch. But in a crash scenario, old code could resume with a model one epoch ahead of the optimizer/inner state, leading to inconsistent training state. New code cannot have this inconsistency.

### B. OOM Batch-Size Reduction (§2.1)

**Old:** `cfg.nn_model.training.batch_size_test` was permanently mutated after OOM.  
**New:** `batch_size_reduction_factor` is local to the current phase retry loop; config is unchanged.

**Risk level: Low, conditional.** Only differs when:
1. A CUDA OOM occurred in a prior phase, **and**
2. The subsequent phase reads `batch_size_test` from config (specifically for test jobs)

In this case the old code would use a smaller batch size for subsequent test jobs (silently), while new code restores original config semantics. This could produce a slightly different validation signal, though model weights are not directly affected.

### C. CUDA Memory Threshold Evaluation (§4.3)

**Old:** threshold comparisons (60%, 98% of total GPU memory) computed on value divided by 1027 instead of 1073741824 (off by ~0.3%).  
**New:** correct `1024**3` division.

**Risk level: Very Low.** The thresholds are wide margins; 0.3% error does not change when OOM preemption fires in any realistic scenario. The logged metric was wrong, but the training behaviour was effectively the same.

### D. Test Job Resume Capability (§7)

**Old:** Cannot resume a run interrupted during the test phase.  
**New:** Can resume at the test phase using the outer bundle.

**Risk level: N/A for training output** (test phase does not update model weights). But operationally significant.

### E. First-Epoch NaN Handling

**Old:** If a NaN occurred on the very first epoch (before any restart checkpoint exists), `model_current.pt` would also not exist; the reload would fail with a file-not-found error.  
**New:** If a NaN occurs on the very first epoch (no bundle yet), a `ValueError` is explicitly raised with a clear message.

**Risk level: Low.** Only affects robustness of error reporting, not normal training.

---

## Non-Critical Differences (Refactoring / Correctness Fixes)

| Aspect | Notes |
|---|---|
| BNODE + pre-train `ValueError` | Now a warning + skip; previously silent/undefined |
| MLflow NaN tag text | Fixed from `'4 NaNs in loss'` to `'50 NaNs in loss'` |
| `int(...)` vs `ceil` in epoch budget | Code was always `int()` (floor); old flow doc was wrong — no code change |
| Atomic bundle consistency | All restart state from same epoch guaranteed |
| Test job resume allowed | `_validate_restart_target` now only rejects pre-train |
| `_clear_restart_state` removed | Replaced by cleaner `checkpoint_store.clear_restart_artifacts()` |
| OOM retry warning after 30 retries | New informational log |
