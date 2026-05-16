import functools

import triton


def calculate_settings(n_cols: int) -> tuple[int, int]:
    """Choose a simple power-of-two block size and warp count for row kernels."""

    block_size = triton.next_power_of_2(n_cols)
    if block_size < 2048:
        num_warps = 4
    elif block_size < 8192:
        num_warps = 8
    else:
        num_warps = 16
    return block_size, num_warps


def ensure_contiguous(fn):
    """Make tensor arguments contiguous before launching Triton kernels."""

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        contiguous_args = [arg.contiguous() if hasattr(arg, "contiguous") else arg for arg in args]
        return fn(ctx, *contiguous_args, **kwargs)

    return wrapper
