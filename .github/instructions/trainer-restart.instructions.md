---
name: bnode-core trainer restart states
description: Restart-state architecture guidance for trainer resume logic, docs, and tests
applyTo: "src/bnode_core/ode/trainer.py,src/bnode_core/ode/trainer_utils/restart_state.py,src/bnode_core/ode/trainer_utils/restart_utils.py,tests/ode/test_restart_state.py,docs/bnode_core/ode/restart_*.md"
---
# bnode-core trainer restart states

Apply these instructions when editing trainer restart/resume state logic or its documentation.

## State model contract

- Keep the three responsibilities distinct:
  - `TrainingPhaseState` for phase-local mutable counters
  - `LiveTrainingState` for the active in-memory runtime bundle
  - `TrainingRestartState` for the serialized checkpoint schema
- `LiveTrainingState` uses a **two-phase construction pattern**:
  1. `create_uninitialized()` — called in `train_all_phases()` just before `train_one_phase()`
     is invoked (after dataloaders exist, so epoch bounds can be computed). Builds
     `phase_state` from restart metadata. Runtime object fields (`model`, `optimizer`,
     `lr_schedulers`, `scaler`, `early_stopping`) are `None` at this point.
     Falls back to being called inside `train_one_phase()` when no pre-created
     `live_state` is provided, so the function remains usable as a standalone entry point.
  2. `bind_runtime_objects()` — called inside `train_one_phase()` after all runtime
     objects have been created; sets them and restores their state_dicts from the checkpoint.
   - `create()` is a convenience wrapper combining both steps; prefer the explicit
     two-phase calls: `create_uninitialized()` in `train_all_phases()` and
     `bind_runtime_objects()` inside `train_one_phase()`.
- `trainer.py` may inspect checkpoint metadata early to select the resumed job, but
  `bind_runtime_objects()` (which restores state_dicts) must only be called after all
  runtime objects exist.
- Do not let `trainer.py` spread raw `TrainingRestartState` field access across the
  phase loop when the same behavior can live behind `LiveTrainingState`.

## Documentation contract

- Keep `docs/bnode_core/ode/restart_training.md` aligned with both:
  - the operational restart workflow
  - the responsibilities of `TrainingPhaseState`, `LiveTrainingState`, and `TrainingRestartState`
- When code comments or docstrings explain the state model, point readers to `docs/bnode_core/ode/restart_training.md`.

## Test expectations

- `tests/ode/test_restart_state.py` should cover:
  - restart-state roundtrips for the serialized schema
  - `create_uninitialized()` — verifies runtime object fields are `None` and all config/path fields are set; uses `dataclasses.fields()` to iterate all fields so new fields surface as test failures
  - `bind_runtime_objects()` two-phase roundtrip — verifies state_dicts are fully restored after the second phase
  - `create()` backward-compat — verifies the convenience wrapper still restores runtime state correctly
- If checkpoint fields or restore ordering change, update all relevant test variants in the same task.
