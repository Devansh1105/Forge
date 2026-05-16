# Kernel Notes

Add one concise markdown file per kernel. These docs should explain enough for a
reviewer to understand the operation, Triton mapping, constraints, correctness
checks, and benchmark results without reading every line of the implementation.

Use this filename pattern:

```text
docs/kernels/<kernel>.md
```

Suggested sections:

- Operation
- Triton implementation
- Supported shapes and dtypes
- Correctness tests
- Benchmark summary
- Open issues

Current samples:

- [SwiGLU](swiglu.md)
