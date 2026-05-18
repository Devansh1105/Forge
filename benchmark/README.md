## Forge Benchmarks

Use `benchmark/personal/` for exploratory work and `benchmark/scripts/` for
shared benchmark scripts that reviewers can rerun. Shared scripts should write
rows to `benchmark/data/all_benchmark_data.csv`; `benchmark/benchmarks_visualizer.py`
plots that CSV.

Every benchmark result should record:

- Hardware type.
- Torch and Triton versions.
- Kernel dtype.
- Input shapes.
- Baseline implementation.
- Timing method.
- Speed and memory results.

Run the SwiGLU sample:

```bash
PYTHONPATH=src python benchmark/scripts/benchmark_swiglu.py \
  --sweep-mode token_length \
  --model llama_3_8b \
  --overwrite
```

For model-config sweeps:

```bash
PYTHONPATH=src python benchmark/scripts/benchmark_swiglu.py \
  --sweep-mode model_config \
  --bt 2048 \
  --overwrite
```

Plot the latest CSV rows:

```bash
python benchmark/benchmarks_visualizer.py \
  --kernel-name swiglu \
  --metric-name speed \
  --kernel-operation-mode full \
  --overwrite
```

The fuller benchmark framework will be designed during the repo v1 phase. Until
then, keep benchmark scripts simple and explicit.
