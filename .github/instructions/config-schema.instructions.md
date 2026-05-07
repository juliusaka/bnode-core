---
name: bnode-core config schema
description: Dataclass hierarchy and Hydra config guidance for config.py and related YAML configs
applyTo: "src/bnode_core/config.py,tests/test_config.py,resources/config/**/*.yaml"
---
# bnode-core config schema

Apply these instructions when changing `src/bnode_core/config.py`, `tests/test_config.py`, or package-local Hydra configs under `resources/config/`.

## Overview

- `src/bnode_core/config.py` defines the Pydantic `@dataclass` schemas used by Hydra and OmegaConf.
- YAML configs are converted to typed dataclasses through `convert_cfg_to_dataclass()`.
- Keep schema changes, config fixtures, and config-focused tests aligned in the same task.

## Dataclass hierarchy

| Class | Purpose |
|-------|---------|
| `SolverClass` | Simulation timing and solver behavior |
| `RawDataClass` | FMU paths, sampling, and variable names for data generation |
| `base_dataset_prep_class` | Slicing, filtering, and transform settings for prepared datasets |
| `base_pModelClass` | Physical model settings |
| `data_gen_config` | Top-level data-generation config |
| `base_network_class` | Shared network hyperparameters |
| `base_training_settings_class` | Shared training settings |
| `abstract_nn_model_class` | Marker base with the `model_type` discriminator |
| `base_nn_model_class` | Feed-forward model config |
| `base_ode_nn_model_class` | Neural ODE model config |
| `pels_vae_network_class` | VAE-specific network config |
| `pels_vae_training_settings_class` | VAE-specific training config |
| `neural_ode_network_class` | Neural ODE network config |
| `latent_ode_network_class` | BNODE network config: latent dims, linear modes, and latent ODE type |
| `latent_timestepper_training_settings` | Per-phase latent-ODE training settings |
| `base_latent_ode_training_settings_class` | BNODE training config with multi-phase settings and overrides |
| `base_latent_ode_nn_model_class` | Binds latent-ODE network and training settings |
| `train_test_config_class` | Top-level runtime config for training and testing |
| `load_latent_ode_config_class` | Config for loading a trained latent-ODE model |
| `onnx_export_config_class` | ONNX export settings built on `load_latent_ode_config_class` |

## Key concepts

### Model type discriminator

`abstract_nn_model_class.model_type` selects the concrete config family:

- `None` → `base_nn_model_class`
- `"node"` → `base_ode_nn_model_class`
- `"bnode"` → `base_latent_ode_nn_model_class`

### Multi-phase BNODE training

`base_latent_ode_training_settings_class` supports phased training through `main_training`, a `List[latent_timestepper_training_settings]`.

- Top-level `*_override` fields broadcast settings such as learning rate or solver choice across phases.
- Only one phase may set `activate_deterministic_mode_after_this_phase=True`.
- When deterministic mode activates, the next phase automatically gets `reload_optimizer=False` to avoid stale optimizer state.

### Linear modes

`latent_ode_network_class.linear_mode` controls grouped linearization behavior:

- `"mpc_mode"` → all linear except the state encoder
- `"mpc_mode_for_controls"` → linear latent ODE and decoder plus control encoder
- `"deep_koopman"` → only the latent ODE stays linear
- `None` → use the explicit per-component `*_linear` flags

### Latent ODE types

`latent_ode_network_class.lat_ode_type` selects the latent-dynamics family:

- `"variance_constant"` — BNODE with constant variance
- `"variance_dynamic"` — BNODE with dynamic variance
- `"vanilla"` — latent ODE without BNODE variance modeling

## Utility functions

- `get_config_store()` registers the dataclasses with Hydra's `ConfigStore`.
- `convert_cfg_to_dataclass(cfg)` converts `DictConfig` objects into validated dataclasses.
- `save_dataclass_as_yaml(cfg, path)` persists a dataclass config as YAML.

## Config locations

- Package-local runtime and pytest configs live under `resources/config/`.
- The superproject keeps its own top-level `config/` tree; do not assume both roots are active at once.
- `train_test_config_class` resumes training only from `training_outer_restart.pt` and `training_inner_restart.pt` in the active Hydra output directory; do not add config fields that redirect resume state to an external path unless the user explicitly asks for that behavior.
- Do not add legacy config aliases, compatibility fields, or older restart-path options unless the user explicitly asks for compatibility.
- If you change schema fields or defaults, update the matching YAML fixtures and `tests/test_config.py` expectations together.
