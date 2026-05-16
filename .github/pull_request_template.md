## Summary

Describe the change and the kernel or repo area it touches.

## Kernel Details

- Kernel:
- Owner:
- Hardware:
- Dtypes:
- Shapes covered:
- Baseline:

## Correctness

- [ ] Forward matches PyTorch reference.
- [ ] Backward matches PyTorch reference.
- [ ] Edge cases are covered or documented.
- [ ] Tolerances are documented if they differ from the default checklist.

## Benchmarks

- [ ] Personal benchmark added under `benchmark/personal/<name>/<kernel>/`.
- [ ] Shared benchmark added or updated under `benchmark/scripts/`, if ready.
- [ ] Results include hardware, dtype, shapes, baseline, and speed/memory numbers.

## Docs

- [ ] Kernel note added or updated under `docs/kernels/`.
- [ ] Limitations and open issues are documented.

## Testing Done

Paste commands and key results.

```bash
# examples
make checkstyle
pytest test/path/to/test_kernel.py
python benchmark/scripts/benchmark_kernel.py
```
