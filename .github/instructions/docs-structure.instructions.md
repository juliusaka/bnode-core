---
name: bnode-core docs structure
description: MkDocs structure and placement rules for bnode-core documentation pages and nav updates
applyTo: "docs/**/*.md,mkdocs.yml"
---
# bnode-core documentation structure

Apply these instructions when editing `docs/**/*.md` or `mkdocs.yml` in `bnode-core`.

## Documentation layout

- Follow the current repository documentation layout; it is already organized around the Python package structure.
- Keep the top-level docs files minimal:
  - `docs/index.md` is the MkDocs home page and includes `README.md`
  - `docs/reference.md` is the documentation landing page
- Put module-specific guides under `docs/bnode_core/...` to mirror the package structure in `src/bnode_core/...`.

## Navigation conventions

- Keep `mkdocs.yml` hierarchical.
- Add new pages under the most relevant existing group in `API Reference` rather than creating many new top-level sections.
- Preserve the current broad grouping style:
  - Data Generation
  - (B)NODE Module
  - Neural Networks
  - configuration-related pages

## Page-writing conventions

- Use a clear H1 title at the top of each page.
- Prefer workflow-oriented prose for concepts and usage notes.
- Use mkdocstrings pages when the primary goal is API exposure of a specific module.
- Use fenced code blocks, tables, and short headed sections for commands and output structure.

## Scope conventions

- Keep MkDocs content user-facing.
- Do not place Copilot-only instructions in `docs/`.
- When a page explains a script or module workflow, mention the real file/module names used in the codebase.
- Keep spelling consistent with the repository, including established names like `data_preperation`.

## When updating navigation

- Keep YAML indentation and nesting consistent with the current file.
- Update `docs/reference.md` when a new page should serve as a recommended entry point.
- Avoid flattening the nav or moving existing docs unless the user specifically asked for a reorganization.
