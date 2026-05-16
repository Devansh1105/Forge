"""Sample SwiGLU Triton op.

Adapted from LinkedIn's Liger Kernel SwiGLU implementation under BSD-2-Clause.
Original copyright: Copyright 2024 LinkedIn Corporation.
This is intentionally small and kept as a readable Forge reference sample.
"""

import torch
import triton
import triton.language as tl

from forge.ops.utils import calculate_settings
from forge.ops.utils import ensure_contiguous


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols

    gate_ptr += row_id * stride
    up_ptr += row_id * stride
    out_ptr += row_id * stride

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)
    out = _silu(gate).cast(up.dtype) * up
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    stride,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols

    grad_out_ptr += row_id * stride
    gate_ptr += row_id * stride
    up_ptr += row_id * stride

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = grad_out * silu_gate
    grad_gate = grad_out * (silu_gate * (1 - sigmoid_gate) + sigmoid_gate) * up

    # Store into saved activation buffers. Autograd returns these as gradients.
    tl.store(gate_ptr + offsets, grad_gate, mask=mask)
    tl.store(up_ptr + offsets, grad_up, mask=mask)


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `silu(gate) * up` over the last dimension."""

    original_shape = gate.shape
    n_cols = original_shape[-1]
    gate_2d = gate.view(-1, n_cols)
    up_2d = up.view(-1, n_cols)
    out = torch.empty_like(gate_2d)

    block_size, num_warps = calculate_settings(n_cols)
    _swiglu_forward_kernel[(gate_2d.shape[0],)](
        gate_2d,
        up_2d,
        out,
        out.stride(0),
        n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    return gate_2d, up_2d, out.view(*original_shape)


def swiglu_backward(gate: torch.Tensor, up: torch.Tensor, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for `silu(gate) * up`."""

    original_shape = grad_out.shape
    n_cols = original_shape[-1]
    grad_out_2d = grad_out.view(-1, n_cols)

    block_size, num_warps = calculate_settings(n_cols)
    _swiglu_backward_kernel[(grad_out_2d.shape[0],)](
        grad_out_2d,
        gate,
        up,
        grad_out_2d.stride(0),
        n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    return gate.view(*original_shape), up.view(*original_shape)


class ForgeSiLUMulFunction(torch.autograd.Function):
    """Autograd wrapper for the sample SwiGLU elementwise fusion."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate, up, out = swiglu_forward(gate, up)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        return swiglu_backward(gate, up, grad_out)
