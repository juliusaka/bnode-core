---
name: bnode-core Python workflows
description: Environment baseline, command patterns, Hydra config roots, and workflow-contract guidance for Python work in bnode-core
applyTo: "src/**/*.py,tests/**/*.py,resources/config/**/*.yaml,pyproject.toml"
---
# bnode-core Python workflows

Apply these instructions when working in Python sources, tests, package-local Hydra configs, or `pyproject.toml` in `bnode-core`.

## Environment and command policy

- Use the existing environment; do not run `uv sync`, `uv sync --extra ...`, or otherwise change extras unless the user explicitly asks.
- These commands are written for the `bnode-core` repository root.

```bash
source .venv/bin/activate
```

- Prefer `python -m ...`, `python -m pytest ...`, and `python -m ruff ...` over `uv run`.
- Run pytest from the repository root. A pytest fixture normalizes the working directory internally.
- Do not open a new terminal for each test run. Activate the venv once, stay in the same shell, then reuse it.

## Common commands

| Task | Command |
|------|---------|
| Lint | `python -m ruff check src tests` |
| All tests | `python -m pytest tests -x -v --tb=short` |
| Single fast config test | `python -m pytest tests/test_config.py::test_convert_cfg_to_dataclass -x -v --tb=short` |
| Fast mask test | `python -m pytest tests/ode/test_set_mask.py -x -v --tb=short` |
| Data generation | `python -m bnode_core.data_generation.raw_data_generation` |
| Data preparation | `python -m bnode_core.data_generation.data_preperation` |
| Training | `python -m bnode_core.ode.trainer` |
| ONNX export | `python -m bnode_core.ode.bnode.bnode_export` |

## Always update docs and instructions

When behavior, commands, paths, outputs, or terminology change:

- Update the code, the relevant MkDocs pages, and the relevant `.instructions.md` files in the **same task**. Do not treat this as optional cleanup.
- When tests or fixtures change, review the corresponding instruction sections for variant names, dataset names, prerequisite lists, and output locations. Replace stale content instead of layering new bullets on top of obsolete ones.
- If a section is stale, rewrite it in the same task instead of appending a note and leaving the old text in place.
- Do not preserve legacy compatibility branches, old schema support, historical aliases, or obsolete artifact names unless the user explicitly asks for compatibility.

## Instruction precedence

- This file is the baseline for normal `bnode-core` Python work.
- If a more specific targeted file also applies, keep following this file for environment and root-selection policy, but let the more specific file win on workflow details:
  - `config-schema.instructions.md` for config dataclasses and related YAML fixtures
  - `bnode-export.instructions.md` for exporter behavior and exporter tests
  - `pytest-workflows.instructions.md` for pytest execution policy
  - `docs-structure.instructions.md` for MkDocs layout changes

## Hydra config roots

- Package-local runtime and test configs live under `resources/config/`.
- From inside `bnode-core`, module auto-discovery resolves the package-local config root.
- From the superproject root, the active Hydra root changes to the superproject `config/` tree instead.
- Keep that distinction explicit when changing commands, tests, or config-loading behavior.

## Working-root decision ladder

1. Use the `bnode-core` root when the task depends on package-local `resources/config/` or normal package-local tests.
2. Use the superproject root when the task intentionally exercises the heat-pump project `config/` tree.
3. When updating commands, docs, or tests, state the assumed root explicitly if switching roots would change Hydra discovery or file paths.

## Workflow-contract rules

- For self-contained workflows in this repository, treat exporter-produced artifacts and metadata as a strict contract.
- Do **not** add fallback handling for hypothetical alternate schemas, older metadata layouts, or partially missing fields unless the user explicitly asks for compatibility support.
- In these workflows, robustness means validating expected metadata and failing clearly when it is missing or inconsistent, not silently downgrading behavior.
