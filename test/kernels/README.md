# Kernel Tests

This folder is for isolated correctness tests for Triton kernels and their thin
autograd wrappers.

Good kernel tests should:

- Build small, explicit tensors.
- Compare Forge output against a PyTorch reference.
- Check gradients against the same reference.
- Cover representative model shapes where practical.
- Include odd/non-power-of-two shapes.
- Test `fp32` first, then `bf16` when the hardware supports it.

Avoid model monkey patching here. If a test needs Hugging Face model replacement
or a training loop, put it in `test/integration/`.

Current reference:

- `test_swiglu.py`
