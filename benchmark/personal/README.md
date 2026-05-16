# Personal Benchmarks

Use this folder for contributor-owned benchmark experiments before promoting a
script into the shared benchmark suite.

Suggested layout:

```text
benchmark/personal/<name>/<kernel>/
  README.md
  benchmark_<kernel>_scratch.py
  results/
```

Each personal benchmark README should record:

- Hardware type.
- Torch and Triton versions.
- Shapes and dtypes.
- Baseline implementation.
- Timing and memory results.
- Notes on failed approaches or surprising findings.

Do not treat personal benchmark results as final project claims. Promote stable
benchmarks into `benchmark/scripts/` before using them in docs or release notes.
