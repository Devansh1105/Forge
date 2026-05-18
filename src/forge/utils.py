"""Small Forge runtime helpers."""

import torch


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def infer_comm_backend() -> str:
    device = infer_device()
    if device == "cuda":
        return "nccl"
    if device == "xpu":
        return "xccl"
    if device == "npu":
        return "hccl"
    return "gloo"


def get_total_gpu_memory(device: int | None = None) -> int:
    if not torch.cuda.is_available():
        return 0
    device = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_properties(device).total_memory
