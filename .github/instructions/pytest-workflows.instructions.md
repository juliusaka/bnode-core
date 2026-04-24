---
name: bnode-core pytest workflows
description: Pytest execution policy, suite map, and high-value test commands for bnode-core
applyTo: "tests/**/*.py,pyproject.toml"
---
# bnode-core pytest workflows

Apply these instructions when editing `tests/**/*.py` or pytest-related settings in `pyproject.toml`.

## Pytest configuration

- `pyproject.toml` sets `addopts = "-n auto --dist loadscope"` for parallel execution via `pytest-xdist`.
- Useful flags:
  - `-x` to stop on the first failure
  - `-v` for verbose output
  - `--tb=short` or `--tb=long` for traceback detail

## Long-running test execution policy

- For slow integration tests, run one explicit command once and wait for completion.
- Do not re-run the same test command unless the previous run clearly failed or was explicitly cancelled.
- Always execute from the `bnode-core` repository root in the same activated shell session.
- Prefer a single command that streams output, usually `-x -v --tb=short`.
- If output is truncated by tooling, write it to a project-local log file and inspect that file after completion rather than restarting the test.
- After each long run, report the working directory, exact command, final pytest summary, and exit code.

## Suite map

| File | Purpose | Speed |
|------|---------|-------|
| `tests/test_config.py` | Config/dataclass validation | Fast (~2s) |
| `tests/test_filepaths.py` | File path utilities | Fast |
| `tests/ode/test_set_mask.py` | Deterministic mode weight trimming unit tests | Fast (~2s) |
| `tests/ode/test_get_control_input.py` | Control input utility tests | Fast |
| `tests/ode/test_bnode.py` | BNODE training integration tests | Slow (~15-25s each) |
| `tests/ode/test_node.py` | NODE training tests | Slow |
| `tests/ode/test_bnode_export.py` | ONNX export integration tests | Slow (~10-20s each); see `bnode-export.instructions.md` for exporter-specific details |

## High-value test names

Training coverage in `tests/ode/test_bnode.py`:

- `test_deterministic_mode`
- `test_deterministic_mode_from_state0`

Use `bnode-export.instructions.md` for the exporter-specific case list in `tests/ode/test_bnode_export.py`.

## Test config and output conventions

- Test Hydra configs live in `resources/config/`, especially `resources/config/nn_model/`.
- Export-heavy tests write outputs under `tests/_results/ode/`.
- Slow ODE tests train small models end to end; avoid running the full slow suite unnecessarily.

## Common commands

```bash
source .venv/bin/activate

# run a specific BNODE training test
python -m pytest tests/ode/test_bnode.py::test_deterministic_mode -x -v --tb=short

# run export tests by keyword
python -m pytest tests/ode/test_bnode_export.py -k "deterministic" -x -v --tb=short

# run all export tests
python -m pytest tests/ode/test_bnode_export.py -x -v --tb=short

# run fast unit tests
python -m pytest tests/ode/test_set_mask.py -x -v --tb=short

# run config tests
python -m pytest tests/test_config.py -x -v --tb=short
```
