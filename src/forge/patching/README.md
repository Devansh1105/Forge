# Patching

This folder is reserved for Forge's public model patching surface.

The intended API shape is:

```python
import forge

forge.patch(model, ...)
forge.unpatch(model, ...)
```

The concrete API is not finalized yet. Until repo v1 design is agreed, keep this
folder minimal and put model-specific experiments under
`src/forge/transformers/model/` or `src/forge/transformers/experimental/`.

## Design Goals

- Easy to apply and undo.
- Explicit about which kernels are patched.
- Debuggable when a model architecture is unsupported.
- Compatible with distributed training assumptions.
- Safe to compose with other libraries where possible.

## Do Not Put Here Yet

- Large architecture-specific implementations.
- Kernel code.
- Benchmark scripts.
- One-off experiments that are not part of the public patch API.
