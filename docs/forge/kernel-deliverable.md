# Kernel Deliverable Checklist

Each kernel should land as a focused PR with code, tests, benchmark evidence, and
short documentation.

## Required Files

- `src/forge/ops/<kernel>.py`
  - Triton forward kernel.
  - Triton backward kernel when gradients are needed.
  - A small Python callable around the Triton launch.

- `src/forge/transformers/<kernel>.py`
  - Autograd wrapper, module wrapper, or functional wrapper.
  - Keep the public surface small until the Forge v1 API is designed.

- `test/kernels/test_<kernel>.py`
  - Compare against a PyTorch reference.
  - Cover forward and backward.
  - Cover `bf16` and `fp32` when meaningful.
  - Cover representative Qwen 2.5 0.5B shapes.

- `benchmark/personal/<name>/<kernel>/`
  - Exploratory benchmark script.
  - Notes on shape choices, hardware, findings, and failed ideas.

- `benchmark/scripts/benchmark_<kernel>.py`
  - Shared benchmark script once the benchmark stabilizes.

- `docs/kernels/<kernel>.md`
  - Concise explanation of the operation.
  - Triton implementation notes.
  - Supported shapes, dtypes, and constraints.
  - Correctness tolerance and benchmark summary.

## PR Acceptance Bar

- Correctness tests pass against PyTorch reference.
- Backward gradients match within documented tolerance.
- Benchmark script runs on the claimed hardware.
- PR description includes hardware type, dtype, shapes, baseline, and speed/memory
  results.
- No unrelated refactors.

## Suggested Tolerances

Use tighter tolerances where possible, but start with:

- `fp32`: `rtol=1e-5`, `atol=1e-6`.
- `bf16`: `rtol=1e-2`, `atol=1e-2`.

If a kernel needs looser tolerances, document the reason in the PR and kernel
docs.
