"""Simple SwiGLU benchmark sample.

Run on a CUDA machine:

    PYTHONPATH=src python benchmark/scripts/benchmark_swiglu.py --rows 4096 --cols 11008 --dtype bf16
"""

import argparse

import torch
import triton

from forge.ops import ForgeSiLUMulFunction


def _dtype(name: str):
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _reference(gate, up):
    return torch.nn.functional.silu(gate) * up


def _bench_forward(fn):
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn)


def _bench_full(fn, gate, up):
    def run():
        gate.grad = None
        up.grad = None
        out = fn(gate, up)
        out.backward(torch.ones_like(out))

    torch.cuda.synchronize()
    return triton.testing.do_bench(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=11008)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_swiglu.py requires CUDA")

    dtype = _dtype(args.dtype)
    gate = torch.randn(args.rows, args.cols, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(args.rows, args.cols, device="cuda", dtype=dtype, requires_grad=True)

    forge_forward_ms = _bench_forward(lambda: ForgeSiLUMulFunction.apply(gate, up))
    torch_forward_ms = _bench_forward(lambda: _reference(gate, up))
    forge_full_ms = _bench_full(lambda a, b: ForgeSiLUMulFunction.apply(a, b), gate, up)
    torch_full_ms = _bench_full(_reference, gate, up)

    print(f"shape=({args.rows}, {args.cols}) dtype={args.dtype}")
    print(f"forward_ms forge={forge_forward_ms:.4f} torch={torch_forward_ms:.4f}")
    print(f"full_ms    forge={forge_full_ms:.4f} torch={torch_full_ms:.4f}")
    print(f"speedup_forward={torch_forward_ms / forge_forward_ms:.3f}x")
    print(f"speedup_full={torch_full_ms / forge_full_ms:.3f}x")


if __name__ == "__main__":
    main()
