"""Autograd and model-facing wrappers for Forge kernels."""

from forge.transformers.functional import forge_swiglu
from forge.transformers.swiglu import ForgeBlockSparseTop2MLP
from forge.transformers.swiglu import ForgeExperts
from forge.transformers.swiglu import ForgeFalconH1SwiGLUMLP
from forge.transformers.swiglu import ForgeHunyuanV1SwiGLUMLP
from forge.transformers.swiglu import ForgePhi3SwiGLUMLP
from forge.transformers.swiglu import ForgeQwen3MoeSwiGLUMLP
from forge.transformers.swiglu import ForgeSwiGLUMLP

__all__ = [
    "ForgeBlockSparseTop2MLP",
    "ForgeExperts",
    "ForgeFalconH1SwiGLUMLP",
    "ForgeHunyuanV1SwiGLUMLP",
    "ForgePhi3SwiGLUMLP",
    "ForgeQwen3MoeSwiGLUMLP",
    "ForgeSwiGLUMLP",
    "forge_swiglu",
]
