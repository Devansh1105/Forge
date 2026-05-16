# Temporary Scaffold Design

Forge is intentionally starting close to Liger's workflow shape, but without
copied upstream implementation logic:

```text
src/forge/ops/           Triton kernels and low-level functions
src/forge/transformers/  Autograd/module/Hugging Face-facing wrappers
src/forge/patching/      Future patch/unpatch integration surface
test/kernels/            Kernel correctness tests
test/integration/        Model integration tests
benchmark/scripts/       Shared benchmark scripts
benchmark/personal/      Contributor-owned exploratory benchmarks
docs/kernels/            Concise kernel implementation notes
```

## Why Start Clean

The first two weeks are about kernel velocity, not package design. We keep the
simple lanes that make Liger easy to contribute to, but new code should be
Forge-owned. This avoids accidentally treating upstream implementations as the
baseline architecture.

The v1 refactor should decide:

- Public package name and API shape.
- Kernel registry or dispatch pattern.
- Model patch/unpatch interface.
- Benchmark and convergence testing layout.
- Multi-GPU test strategy.
- Separation between internal Triton ops and public user-facing modules.

## First Sprint Rules

- Do not introduce broad structural refactors in kernel PRs.
- Keep each kernel PR focused on one kernel family.
- Write reference tests before optimizing edge cases.
- Keep exploratory benchmark work under `benchmark/personal/` until it is stable.
- Promote stable benchmark scripts into `benchmark/scripts/`.
- Document the implementation and expected shape/dtype coverage in `docs/kernels/`.
