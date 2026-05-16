## Benchmark Guidelines

For the first sprint, benchmark scripts should be plain Python and easy to run.

Minimum benchmark requirements:

- Compare Forge kernel against a clear PyTorch baseline.
- Measure forward and backward separately when the kernel has gradients.
- Use CUDA events or `triton.testing.do_bench`.
- Warm up before collecting timings.
- Synchronize before reading timings.
- Record peak memory where relevant.
- Include A100/H100 details when reporting results.

Avoid making broad performance claims from one shape. Report the exact shapes
and dtypes used.
