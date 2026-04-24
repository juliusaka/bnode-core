---
name: BNODE export
description: Guidance for the BNODE ONNX exporter and its export integration tests
applyTo: "src/bnode_core/ode/bnode/bnode_export.py,tests/ode/test_bnode_export.py"
---
# BNODE ONNX export instructions

Apply these instructions when working in:

- `src/bnode_core/ode/bnode/bnode_export.py`
- `tests/ode/test_bnode_export.py`

Skip these instructions for unrelated `bnode-core` work.

## Environment and command policy

- Use the existing environment; do not run `uv sync` or change dependency extras unless the user explicitly asks.
- These commands are written for the `bnode-core` repository root.

```bash
source .venv/bin/activate
```

- Prefer `python -m ...` and `python -m pytest ...` over `uv run`.

## Entry point and config discovery

- The exporter entry point is `bnode_core.ode.bnode.bnode_export:main`.
- The installed console-script shortcut is `bnode_export`.
- The module resolves its Hydra config directory with `config_dir_auto_recognize()` and then loads config name `onnx_export`.
- This means the active working tree matters:
  - from the superproject root, Hydra can resolve the top-level heat-pump `config/`
  - from inside `bnode-core`, the normal config root is the package-local config location
- Do not hardcode a different config root unless the test or caller explicitly needs it.

## Correct run commands

### Normal export

```bash
python -m bnode_core.ode.bnode.bnode_export model_directory=<trained-model-dir>
```

Equivalent installed script:

```bash
bnode_export model_directory=<trained-model-dir>
```

### Typical local export with explicit output path

```bash
python -m bnode_core.ode.bnode.bnode_export \
  model_directory=<trained-model-dir> \
  output_dir=<export-dir>
```

### Export from MLflow

```bash
python -m bnode_core.ode.bnode.bnode_export \
  mlflow_run_id=<run-id> \
  mlflow_tracking_uri=<tracking-uri> \
  output_dir=<export-dir>
```

### SISO export

```bash
python -m bnode_core.ode.bnode.bnode_export \
  model_directory=<trained-model-dir> \
  output_dir=<export-dir> \
  siso=true
```

## Required inputs

The exporter expects a trained BNODE artifact set, loaded from either:

- `mlflow_run_id`
- or `model_directory`

Important inputs and defaults:

- `model_directory` points to the trained run directory containing `.hydra/`, checkpoints, and usually `dataset.hdf5`
- if `dataset_path` is omitted, the exporter uses `<artifacts-dir>/dataset.hdf5`
- if `config_path` is omitted, the exporter uses `<artifacts-dir>/.hydra/config_validated.yaml`
- if `model_checkpoint_path` is omitted, the exporter uses the latest `model_phase_*.pt`
- if `output_dir` is omitted, the exporter writes into the current Hydra output directory

## Output artifacts to expect

Normal exports write component-wise ONNX files such as:

- `encoder_states.onnx`
- `encoder_controls.onnx` when controls are used
- `encoder_parameters.onnx` when parameter encoding is used
- `latent_ode.onnx`
- `latent_ode_ssm_from_param.onnx` for linear parameterized latent ODEs
- `decoder.onnx`

They also write:

- `encoder_*_example_io.hdf5`
- `latent_ode_example_io.hdf5`
- `decoder_example_io.hdf5`
- `bnode_config.yaml`
- `hydra/` copy of the current Hydra output directory when available

SISO exports additionally write:

- `*_siso.onnx`
- `siso_dimensions.json`

## Export behavior that matters

- Export always runs on CPU: `cfg.use_cuda = False` is forced during loading.
- The model is switched to `eval()` before export.
- `torch.onnx.export(..., dynamo=False)` is intentional here; do not change to the new exporter casually.
- The exporter saves example I/O HDF5 files for validation. Keep them in sync if you add or rename inputs/outputs.
- Decoder normalization statistics are copied into `siso_dimensions.json` during SISO export and are consumed downstream by other applications.
- In this self-contained exporter workflow, treat `siso_dimensions.json` and the exported artifact layout as contract data for downstream consumers in this workspace. Do **not** add fallback branches for hypothetical alternate metadata layouts unless the user explicitly asks for compatibility support.
- For linear latent ODEs with parameter dependence, the SSM parameter mapping is exported separately as `latent_ode_ssm_from_param.onnx`.
- SISO export is intentionally disabled for linear latent ODE exports (`latent_ode` stays multi-input there) because the matrix-valued inputs cannot be flattened into the same simple concatenated interface.

## Test behavior you need to know

- `tests/ode/test_bnode_export.py` is an integration-heavy suite:
  - it first trains a small BNODE via `ode_training(...)`
  - then clears Hydra state with `GlobalHydra.instance().clear()`
  - then invokes `bnode_export_main()` by patching `sys.argv`
- The export tests use `resources/config/onnx_export_pytest.yaml` with:
  - `--config-dir=resources/config`
  - `--config-name=onnx_export_pytest`
- Test outputs are written under `tests/_results/ode/test_<name>/test_export_onnx/`

Important test cases:

- `test_bnode_export`
- `test_bnode_export_with_parameters`
- `test_bnode_export_linear_mpc`
- `test_bnode_export_linear_mpc_for_controls`
- `test_bnode_export_controls_to_state_encoder`
- `test_bnode_export_no_parameter_encoder`
- `test_bnode_export_deterministic_mode`
- `test_bnode_export_deterministic_mode_from_state0`
- `test_bnode_export_deterministic_linear_mpc`
- `test_bnode_export_siso`
- `test_bnode_export_siso_deterministic`

## Common pitfalls

- Keep the difference clear between:
  - the trained model directory
  - the export output directory
  - the Hydra run directory
- Do not assume `config_path` overrides are supported; the current implementation explicitly raises for custom `config_path`.
- If you change exporter inputs/outputs, update:
  - ONNX export input/output names
  - example I/O HDF5 contents
  - SISO dimension metadata
  - downstream consumers of the ONNX export artifacts
- After training in tests, Hydra global state must be cleared before invoking the exporter again in-process.
- If artifacts come from a remote MLflow server, the exporter downloads them into the current Hydra output directory and later cleans up that temporary directory.

## Good debugging order

```bash
# 1. fast export-focused integration coverage
python -m pytest tests/ode/test_bnode_export.py -x -v --tb=short

# 2. SISO-specific checks
python -m pytest tests/ode/test_bnode_export.py -k "siso" -x -v --tb=short

# 3. deterministic export variants
python -m pytest tests/ode/test_bnode_export.py -k "deterministic" -x -v --tb=short
```

## Done means

- The exporter still keeps the trained-model directory, export output directory, and Hydra run directory distinct.
- Exported ONNX artifact names, example I/O HDF5 files, and `siso_dimensions.json` stay in sync with any changed inputs or outputs.
- Any contract change is reflected in the matching tests, command examples, and instruction text in the same task.
- The most targeted exporter test coverage for the change has been run from the correct repository root.
