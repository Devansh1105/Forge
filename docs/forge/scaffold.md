# Temporary Scaffold Design

Forge keeps the simple contribution lanes that make Liger easy to work in, but
uses Forge-owned package names and docs.

```text
src/forge/ops/                         Triton kernels and low-level functions
src/forge/transformers/                Autograd/module/HF-facing wrappers
src/forge/transformers/model/          Architecture-specific patching modules
src/forge/transformers/trainer/        Trainer integrations
src/forge/transformers/experimental/   Unstable transformer integrations
src/forge/patching/                    Future patch/unpatch integration surface
test/kernels/                          Kernel correctness tests
test/integration/                      Model integration tests
benchmark/scripts/                     Shared benchmark scripts
benchmark/personal/                    Contributor-owned exploratory benchmarks
docs/kernels/                          Concise kernel implementation notes
```

## Current Sample

The current reference sample is SwiGLU. It intentionally includes the full
upstream SwiGLU surface adapted to Forge:

- elementwise SwiGLU forward/backward autograd function
- Llama/Qwen-style MLP wrapper
- Mixtral block-sparse wrapper
- Mixtral/Qwen MoE experts wrapper
- Phi3 wrapper
- Qwen3 MoE wrapper
- Hunyuan wrapper
- Falcon H1 multiplier support
- Liger-style benchmark script and benchmark helpers

## First Sprint Rules

- Keep each kernel PR focused on one kernel family.
- Write reference tests before optimizing edge cases.
- Keep exploratory benchmark work under `benchmark/personal/` until it is stable.
- Promote stable benchmark scripts into `benchmark/scripts/`.
- Document shape/dtype support and benchmark assumptions in `docs/kernels/`.
