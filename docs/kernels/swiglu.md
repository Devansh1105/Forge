# SwiGLU Sample

This sample shows the expected shape of a Forge kernel contribution. It is
adapted from LinkedIn's Liger Kernel SwiGLU implementation under BSD-2-Clause,
but simplified for Forge's starter scaffold.

## Operation

SwiGLU uses an elementwise gated activation:

```python
out = silu(gate) * up
```

In a transformer MLP, `gate` and `up` are usually produced by separate linear
projections, followed by a down projection:

```python
y = down_proj(silu(gate_proj(x)) * up_proj(x))
```

## Files

- `src/forge/ops/swiglu.py`: Triton forward/backward kernels and autograd function.
- `src/forge/transformers/swiglu.py`: minimal MLP wrapper.
- `test/kernels/test_swiglu.py`: correctness checks against PyTorch.
- `benchmark/scripts/benchmark_swiglu.py`: Liger-style benchmark against Hugging Face.
- `benchmark/personal/devansh/swiglu/README.md`: personal benchmark notes template.

## Triton Mapping

The sample treats the last dimension as the row width and flattens all leading
dimensions into rows. Each Triton program handles one row. The forward kernel
loads `gate` and `up`, computes `silu(gate) * up`, and stores the output.

The backward kernel recomputes sigmoid/silu instead of saving extra activation
buffers:

```text
grad_up = grad_out * silu(gate)
grad_gate = grad_out * up * d_silu(gate)
```

## Supported Scope

This is a starter sample, not the final optimized Forge SwiGLU.

- CUDA only.
- `fp32` correctness path is covered by the sample test.
- `bf16` should be benchmarked on A100/H100 before claiming speedups.
- No model patching is included yet.

## Benchmark

Example:

```bash
PYTHONPATH=src python benchmark/scripts/benchmark_swiglu.py \
  --sweep-mode token_length \
  --model llama_3_8b \
  --overwrite
```

Visualize:

```bash
python benchmark/benchmarks_visualizer.py \
  --kernel-name swiglu \
  --metric-name speed \
  --kernel-operation-mode full \
  --overwrite
```
