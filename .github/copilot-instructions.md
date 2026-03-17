# bnode-core Copilot Instructions

## Testing

### Environment
- Python venv: activate with `source .venv/bin/activate` from the **workspace root** 
- All pytest commands must be run from `bnode/bnode-core/`. A pytest fixture ensures this.
- **Do not open a new terminal for each test run.** Activate the venv once, `cd bnode/bnode-core`, then reuse the same shell.

### pytest Configuration
- `pyproject.toml` sets `addopts = "-n auto --dist loadscope"` (parallel via pytest-xdist)
- Useful flags: `-x` (stop on first failure), `-v` (verbose), `--tb=short` or `--tb=long`

### Long-Running Test Execution Policy
- For slow integration tests, run **one command once** and wait for completion. Do not re-run the same test command unless the previous run clearly failed or was explicitly cancelled.
- Always execute from the correct working directory (`bnode/bnode-core`) in the same shell session.
- Prefer a single explicit command that streams output (`-v --tb=short`) and then wait; avoid issuing extra probe commands that can interrupt/replace the running process.
- If output is truncated by tooling, write output to a log file and read that file after completion rather than restarting the test.
- After each long test run, report: working directory, exact command, final pytest summary, and exit code.

### Test Files

| File | Purpose | Speed |
|------|---------|-------|
| `tests/test_config.py` | Config/dataclass validation | Fast (~2s) |
| `tests/test_filepaths.py` | File path utilities | Fast |
| `tests/ode/test_set_mask.py` | Deterministic mode weight trimming unit tests | Fast (~2s) |
| `tests/ode/test_get_control_input.py` | Control input utility tests | Fast |
| `tests/ode/test_bnode.py` | BNODE training integration tests | Slow (~15-25s each) |
| `tests/ode/test_bnode_export.py` | ONNX export integration tests | Slow (~10-20s each) |
| `tests/ode/test_node.py` | NODE training tests | Slow |

### Key Test Names

**Training** (`tests/ode/test_bnode.py`):
- `test_determistic_mode` — basic deterministic mode (note: typo in name, missing 'n')
- `test_deterministic_mode_from_state0` — deterministic mode using state0 masks

**Export** (`tests/ode/test_bnode_export.py`):
- `test_bnode_export` — basic export
- `test_bnode_export_with_parameters` — with parameter encoder
- `test_bnode_export_linear_mpc` — linear MPC
- `test_bnode_export_linear_mpc_for_controls` — linear MPC for controls
- `test_bnode_export_controls_to_state_encoder` — controls to state encoder
- `test_bnode_export_no_parameter_encoder` — no parameter encoder
- `test_bnode_export_deterministic_mode` — deterministic export
- `test_bnode_export_deterministic_mode_from_state0` — deterministic from state0
- `test_bnode_export_deterministic_linear_mpc` — deterministic linear MPC

### Example Commands
```bash
# Setup (once per session)
source .venv/bin/activate && cd bnode/bnode-core

# Run a specific test
python -m pytest tests/ode/test_bnode.py::test_determistic_mode -x -v --tb=short

# Run tests by keyword
python -m pytest tests/ode/test_bnode_export.py -k "deterministic" -x -v --tb=short

# Run all export tests
python -m pytest tests/ode/test_bnode_export.py -x -v --tb=short

# Run fast unit tests
python -m pytest tests/ode/test_set_mask.py -x -v --tb=short

# Run config tests
python -m pytest tests/test_config.py -x -v --tb=short
```

### Notes
- Tests use the Hydra config system — test configs live in `resources/config/nn_model/`
- Export tests create temp dirs under `tests/_results/ode/`
- `test_determistic_mode` has a typo (missing 'n') — always use the exact name
- Slow tests train a small model end-to-end; don't run the full suite unnecessarily

## Configuration (`src/bnode_core/config.py`)

### Overview
Pydantic `@dataclass` schemas validated at startup via Hydra + OmegaConf. All YAML configs are converted to typed dataclasses by `convert_cfg_to_dataclass()`.

### Dataclass Hierarchy

| Class | Purpose |
|-------|---------|
| `SolverClass` | Simulation timing and solver behavior |
| `RawDataClass` | FMU paths, sampling, variable names for data generation |
| `base_dataset_prep_class` | Slicing, filtering, transforming dataset |
| `base_pModelClass` | Physical model settings |
| `data_gen_config` | Top-level data generation config |
| `base_network_class` | Base NN hyperparameters (layers, hidden dims) |
| `base_training_settings_class` | Base training settings (optimizer, LR, epochs, scheduler) |
| `abstract_nn_model_class` | Marker base with `model_type` discriminator |
| `base_nn_model_class` | Simple feed-forward model config |
| `pels_vae_network_class` | VAE-specific network config |
| `pels_vae_training_settings_class` | VAE-specific training config |
| `neural_ode_network_class` | Neural ODE network config |
| `latent_ode_network_class` | **BNODE network config** — latent dims, linear modes, ODE type |
| `latent_timestepper_training_settings` | Per-phase training settings (solver, adjoint, sequence length) |
| `base_latent_ode_training_settings_class` | **BNODE training config** — multi-phase training, overrides |
| `base_latent_ode_nn_model_class` | Binds `latent_ode_network_class` + `base_latent_ode_training_settings_class` |
| `train_test_config_class` | Top-level runtime config (dataset, MLflow, CUDA, workers) |
| `load_latent_ode_config_class` | Config for loading a trained model (MLflow or local) |
| `onnx_export_config_class` | ONNX export settings (extends `load_latent_ode_config_class`) |

### Key Concepts

**Model type discriminator**: `abstract_nn_model_class.model_type` selects the config variant:
- `None` → `base_nn_model_class` (feed-forward)
- `'node'` → `base_ode_nn_model_class` (neural ODE)
- `'bnode'` → `base_latent_ode_nn_model_class` (latent ODE / BNODE)

**Multi-phase training** (`base_latent_ode_training_settings_class`):
- `main_training` is a `List[latent_timestepper_training_settings]` — each element is one training phase
- Top-level `*_override` fields broadcast settings to all phases (e.g., `lr_start_override`, `solver_override`)
- Only one phase may set `activate_deterministic_mode_after_this_phase=True` (validated)
- If deterministic mode activates, next phase auto-gets `reload_optimizer=False` (stale momentum buffers)

**Linear modes** (`latent_ode_network_class.linear_mode`):
- `'mpc_mode'` → all linear except state encoder
- `'mpc_mode_for_controls'` → linear ODE/decoder + control encoder
- `'deep_koopman'` → only ODE linear
- `None` → use individual `*_linear` flags

**Latent ODE type** (`latent_ode_network_class.lat_ode_type`):
- `'variance_constant'` — BNODE with constant variance
- `'variance_dynamic'` — BNODE with dynamic variance
- `'vanilla'` — standard latent ODE (no BNODE)

### Utility Functions
- `get_config_store()` — registers all dataclasses with Hydra's ConfigStore
- `convert_cfg_to_dataclass(cfg)` — converts OmegaConf DictConfig → validated Pydantic dataclass (handles both structured and flat YAML)
- `save_dataclass_as_yaml(cfg, path)` — persists dataclass to YAML

### YAML Config Locations
- Training configs: `config/nn_model/` (workspace root)
- Test configs: `resources/config/nn_model/` (inside bnode-core)
- ONNX export config: `config/onnx_export.yaml`
