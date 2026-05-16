## Forge Benchmarks

Use `benchmark/personal/` for exploratory work and `benchmark/scripts/` for
shared benchmark scripts that reviewers can rerun.

Every benchmark result should record:

- Hardware type.
- Torch and Triton versions.
- Kernel dtype.
- Input shapes.
- Baseline implementation.
- Timing method.
- Speed and memory results.

The shared benchmark framework will be designed during the repo v1 phase. Until
then, keep benchmark scripts simple and explicit.
