# Forge

Forge is a Triton-first kernel and training optimization project. 

## Current Layout

```text
src/forge/ops/                         Triton kernels and low-level functions
src/forge/transformers/                Autograd wrappers and model-facing modules
src/forge/transformers/model/          Architecture-specific patching modules
src/forge/transformers/trainer/        Trainer integrations
src/forge/transformers/experimental/   Unstable transformer integrations
src/forge/patching/                    Future patch/unpatch integration surface
test/kernels/                          Kernel correctness tests
test/integration/                      Model and patching integration tests
benchmark/scripts/                     Shared benchmark scripts
benchmark/personal/                    Contributor-owned benchmark experiments
docs/kernels/                          Concise kernel implementation notes
docs/forge/                            Project workflow and scaffold docs
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

The sample benchmark writes rows to `benchmark/data/all_benchmark_data.csv`, and
`benchmark/benchmarks_visualizer.py` plots those rows.

The SwiGLU reference code is adapted from Liger Kernel. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
