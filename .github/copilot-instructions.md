# bnode-core Copilot Instructions

## Critical rule

**Always update docs and instruction files in the same task as code changes.** See `python-workflows.instructions.md` → "Always update docs and instructions" for the full rule.

## Communication

- Explain proposed changes in enough detail that the user can clearly understand the intended work before or while making it, and provide detailed feedback about the outcome.

## Scope and baseline

- These instructions apply to the `bnode-core` repository root. If you are working from the superproject root, follow the superproject instruction file instead of translating paths manually.
- Assume the environment is already provisioned by the user. Do **not** run `uv sync`, change dependency extras, or rebuild the environment unless explicitly asked.
- Activate the existing environment, then prefer direct Python entry points:

```bash
source .venv/bin/activate
```

- Prefer `python -m ...`, `python -m pytest ...`, and `python -m ruff ...` over `uv run`.
- Run pytest from the repository root and reuse the same activated shell session.

## Targeted instruction files

- `.github/instructions/python-workflows.instructions.md` — environment policy, common commands, Hydra config roots, and workflow-contract guidance for normal Python work
- `.github/instructions/pytest-workflows.instructions.md` — pytest execution policy, suite map, and high-value test commands
- `.github/instructions/config-schema.instructions.md` — `src/bnode_core/config.py`, `tests/test_config.py`, and `resources/config/**/*.yaml`
- `.github/instructions/bnode-export.instructions.md` — BNODE ONNX exporter changes in `src/bnode_core/ode/bnode/bnode_export.py` and `tests/ode/test_bnode_export.py`
- `.github/instructions/trainer-restart.instructions.md` — restart-state architecture, trainer resume flow, and restart-state docs/tests
- `.github/instructions/docs-structure.instructions.md` — MkDocs page placement and `mkdocs.yml` updates

Read the targeted file for the area you are changing instead of duplicating its detailed guidance here.

## Instruction precedence

- The most specific targeted instruction file wins over this routing summary on overlapping details.
- When multiple targeted files apply, resolve them by specificity:
  - `bnode-export.instructions.md` or `config-schema.instructions.md`
  - then `pytest-workflows.instructions.md` or `docs-structure.instructions.md`
  - then `python-workflows.instructions.md`
- Keep using `python-workflows.instructions.md` for environment policy and Hydra-root expectations even when a more specific file also applies.

## Working-root decision ladder

1. Use the superproject root when the task depends on the top-level heat-pump `config/` tree.
2. Use the `bnode-core` root when the task depends on package-local `resources/config/`, docs, or package-local test behavior.
3. If command behavior depends on the root, say which root the command assumes instead of translating paths implicitly.
