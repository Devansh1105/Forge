# Forge Tests

Tests are split by scope:

```text
test/kernels/      Isolated kernel correctness tests
test/integration/  Model patching, wrapper, and training-path integration tests
test/utils.py      Shared test helpers
test/conftest.py   Global pytest fixtures
```

## Expectations

- Add tests with every kernel PR.
- Compare against a simple PyTorch reference implementation.
- Cover forward and backward when the operation has gradients.
- Keep isolated kernel tests in `test/kernels/`.
- Put model-level or patching behavior in `test/integration/`.
- Mark CUDA-only tests with a skip when CUDA is unavailable.
- Document loose tolerances in the test and in the kernel docs.

## Naming

Use this pattern:

```text
test/kernels/test_<kernel>.py
test/integration/test_<feature>.py
```
