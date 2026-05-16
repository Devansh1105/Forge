# AGENTS.md

Guidance for AI coding assistants working in this repository.

## Current Phase

Forge is using a Liger-like workflow shape, but the upstream implementation has
been removed. Keep early kernel work close to the current scaffold and avoid
broad package/API refactors until the Forge v1 repo design is agreed.

## Repo Conventions

- Source layout: `src/forge/{ops,transformers,patching}/` for Triton ops,
  wrappers, and future model patching.
- Tests: `test/kernels/` for unit correctness and `test/integration/` for
  model-level checks.
- Benchmarks: `benchmark/scripts/` for shared scripts and `benchmark/personal/`
  for exploratory contributor work.
- Kernel docs: `docs/kernels/`.
- Lint/format: `make checkstyle` uses `ruff`.
- Install dev mode: `pip install -e ".[dev]"`.

See `FORGE.md` and `docs/forge/kernel-deliverable.md` before implementing a new
kernel.
