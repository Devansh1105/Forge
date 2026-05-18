import torch


def supports_bfloat16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
