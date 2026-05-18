"""Functional wrappers for Forge transformer kernels."""

from forge.ops import ForgeSiLUMulFunction


def forge_swiglu(a, b):
    return ForgeSiLUMulFunction.apply(a, b)
