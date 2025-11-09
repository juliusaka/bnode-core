# Introduction

## Overview: Raw Data Generation and Dataset Preparation

This project separates raw data generation (heavy numerical simulation) from dataset preparation (conversions, filtering, slicing and dataset packaging). The separation exists because raw-data generation is often computationally expensive (for example FMU simulations run in parallel). Doing the expensive simulations once and storing the raw outputs lets you repeat cheaper, deterministic preparation steps (unit conversions, derivatives, filtering, variable selection, time-window selection, and creation of train/validation/test splits) many times without re-running the simulators.

A second benefit of the split is reproducible dataset sizing: you can generate a single large raw dataset and then prepare multiple dataset files with different sample counts (n_samples) while keeping the same selection logic for train/validation/test. To support this, `data_preperation.py` creates per-dataset train/validation/test subsets while also storing full "common" validation/test sets (prefixed with `common_`) so that smaller dataset sizes preserve the same validation/test elements (just fewer train samples).

Below is a concise reference of what is produced and saved by the two scripts.

---

## Raw Data File (created by `raw_data_generation.py`)

When `run_data_generation` finishes it writes a single HDF5 raw data file (path from `filepath_raw_data(cfg)` of the `bnode_core.filepaths` module) and a companion YAML config (path from `filepath_raw_data_config(cfg)`). The raw HDF5 stores simulation inputs, outputs and run-level metadata. The script also writes a YAML copy of the `pModel` raw-data config to the filesystem.

### Fields and groups written to the raw HDF5 file

#### Attributes

- `creation_date` — string with the file creation timestamp (YYYY-MM-DD HH:MM:SS). 
- `config` — the raw-data `pModel.RawData` configuration serialized to YAML (OmegaConf.to_yaml(cfg.pModel.RawData)). This is a snapshot of the configuration used for generation. 

#### Datasets (high-level)

- `time` — 1D array containing the simulation time vector used for all time-series (from simulationStartTime to simulationEndTime with the configured timestep).
- `parameters` (optional) — shape (n_samples, n_parameters). Sampled parameter values when `parameters_include` is enabled. Else: the default model parameter values used for all samples.
- `parameters_names` — array of parameter names (stored as bytes/strings).
- `controls` (optional) — shape (n_samples, n_controls, sequence_length) if controls are stored directly. If the sampled controls are only used for sampling (and later extracted from the model) different storage rules apply (see config options).
- `controls_names` — array of control names (bytes/strings).
- `states` — shape (n_samples, n_states, len(time)). Time series of states for each sample. The initial state at time=0 corresponds to the sampled initial states (if any).
- `states_names` — array of state names (bytes/strings).
- `states_der` (optional) — shape (n_samples, n_states, len(time)) if `states_der_include` is True (time derivatives of states).
- `states_der_names` — array of derivative names (e.g. `der(state_name)`) when derivatives are included.
- `outputs` (optional) — shape (n_samples, n_outputs, len(time)) for model outputs if configured.
- `outputs_names` — array of output names.

#### Logs / progress information

- `logs` group — created to track simulation progress and per-sample status:
  - `logs/completed` — boolean array (n_samples,) marking which sample runs completed successfully.
  - `logs/sim_failed` — boolean array (n_samples,) marking runs that failed.
  - `logs/timedout` — boolean array (n_samples,) marking runs that timed out.
  - `logs/processing_time` — per-sample processing times (float) recorded while generating data.
  - This log is written incrementally during data generation to allow usage of the raw data file even if generation is interrupted.
- `failed_idx` — integer array with indices of runs that did not complete (added at the end for backward compatibility). 


---

## Dataset Files (created by `data_preperation.py`)

`data_preperation.py` reads a raw HDF5 file and produces one or more dataset HDF5 files sized to requested sample counts (specified by a list for the config entry `n_samples`). Each dataset file groups data into train/validation/test splits (and includes `common_` groups so small datasets keep the same validation/test elements as larger ones).

### What `data_preperation` writes into each dataset file

#### File-level attribute

- `creation_date` — timestamp when the dataset file was created. TODO: AI, is this correct? I cannot find it in the dataset files.

#### Datasets

- `time` — copied from the raw file (already trimmed to the requested timeframe).
- For each key in `['states', 'states_der', 'controls', 'outputs', 'parameters']` present in the raw data:
  - `<key>_names` — names corresponding to that dataset (e.g. `states_names`).
  - `train/<key>` — train subset samples (shape depends on `n_samples_dataset` and selection logic).
  - `validation/<key>` — validation subset samples.
  - `test/<key>` — test subset samples.
  - `common_validation/<key>` — full common validation set copied from the temporary raw file (used across dataset sizes to ensure the same validation elements).
  - `common_test/<key>` — full common test set copied from the temporary raw file.

#### Side files

- A dataset-specific YAML config is saved for each dataset file path (via `filepath_dataset_config(cfg, n_samples_dataset)`), containing a copy of the `pModel` config with `n_samples` set to that dataset size and any dataset-prep metadata (the script saves `_conf.pModel` using OmegaConf.save).

### Key behaviors of data_preperation

#### Transforms and filtering

- `data_preperation` performs deterministic transforms and filtering:
  - Unit conversions (e.g., `temperature_k_to_degC`, `power_w_to_kw`).
  - Numerical differentiation using an Akima interpolator (if `differentiate` transform is requested) and error-statistics logging.
  - Arbitrary Python-evaluated transforms (prefix `evaluate_python_`) — applied to a variable using an expression with `#` as a placeholder for the timeseries slice.

#### Filtering

- Runs flagged as failed (or listed in `failed_idx`) can be removed.
- Additional filters based on per-variable min/max or expressions can exclude samples.
- Time-window trimming selects only the requested timeframe; the `sequence_length` is adjusted accordingly.

#### Variable selection

- Only the requested variables are kept per dataset (states, controls, outputs, parameters); others are removed to reduce dataset size.
