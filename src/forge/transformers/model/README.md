# Model Patching Modules

This folder is reserved for architecture-specific integration code, for example
Qwen, Llama, Gemma, Phi, or Mixtral patching.

Keep isolated kernels in `src/forge/ops/` and thin wrappers in
`src/forge/transformers/`. Put model-specific replacement logic here only when
it cannot live in the generic wrapper layer.
