# Integration Tests

This folder is for tests that cross module boundaries.

Use it for:

- `forge.patch` / `forge.unpatch` behavior.
- Hugging Face model module replacement.
- End-to-end Qwen/Llama/Gemma patch smoke tests.
- Small training-step checks.
- Convergence or loss sanity checks that require model context.

Keep integration tests smaller than benchmarks. They should answer whether the
system is wired correctly, not prove final performance.

Prefer tiny configs and synthetic data unless a test explicitly needs a real
checkpoint.
