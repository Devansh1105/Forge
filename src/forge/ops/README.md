# Forge Ops

Add low-level Triton kernel implementations here.

Expected pattern for a kernel:

- Private Triton JIT functions for forward and backward.
- A small Python launch wrapper.
- No model-specific monkey patching in this layer.
