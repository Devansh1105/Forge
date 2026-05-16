# Forge

Forge is a Triton-first kernel and training optimization project. 

## Current Layout

```text
src/forge/ops/           Triton kernels and low-level functions
src/forge/transformers/  Autograd wrappers and model-facing modules
src/forge/patching/      Future patch/unpatch integration surface
test/kernels/            Kernel correctness tests
test/integration/        Model and patching integration tests
benchmark/scripts/       Shared benchmark scripts
benchmark/personal/      Contributor-owned benchmark experiments
docs/kernels/            Concise kernel implementation notes
docs/forge/              Project workflow and scaffold docs
```


## Install

```bash
pip install -e ".[dev]"
```

## Development

```bash
make checkstyle
make test
```


