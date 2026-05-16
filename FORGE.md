# Forge Scaffold

Forge is an engineering-first effort to build a stronger kernel and training
optimization platform inspired by Liger and Unsloth. 


## Current Decision

- Keep a Liger-like kernel workflow during Week 1.
- Put Triton kernels in `src/forge/ops/`.
- Put PyTorch/autograd or module wrappers in `src/forge/transformers/`.
- Put future patch/unpatch integration code in `src/forge/patching/`.
- Put kernel correctness tests under `test/kernels/`.
- Put model integration tests under `test/integration/`.
- Put shared benchmark scripts under `benchmark/scripts/`.
- Put personal exploration benchmarks under `benchmark/personal/<name>/<kernel>/`.
- Put concise kernel docs under `docs/kernels/`.

This keeps the repo easy to work in now without inheriting upstream Liger logic.

## Week 1 Kernel Contract

Every kernel PR should include:

- Triton forward and backward implementation.
- Autograd wrapper or `nn.Module` wrapper where appropriate.
- Correctness tests against a PyTorch reference.
- A personal benchmark folder with the exploratory benchmark script and notes.
- A shared benchmark script under `benchmark/scripts/` when the kernel is ready.
- A concise docs page in `docs/kernels/`.


## Near-Term Roadmap

- Week 1: isolated kernels for the Qwen 2.5 0.5B path, starting with SwiGLU and
  RoPE.
- Week 2: continue kernels while designing Forge v1 APIs, benchmark structure,
  testing strategy, and patching boundaries.
- Week 3: refactor accepted kernels into the Forge v1 structure and implement
  model patching for the first Qwen target.
- Week 4: integration tests, convergence tests, comprehensive benchmarks, and
  checkpoint review.

## Licensing

Forge is currently using a BSD 2-Clause license. If the team later copies or
adapts third-party code, preserve the original license and attribution in the
same PR.

