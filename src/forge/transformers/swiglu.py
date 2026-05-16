"""Sample model-facing SwiGLU wrapper."""

import torch.nn as nn

from forge.ops import ForgeSiLUMulFunction


class ForgeSwiGLUMLP(nn.Module):
    """Minimal Llama/Qwen-style SwiGLU MLP sample.

    Expected config attributes:
    - `hidden_size`
    - `intermediate_size`
    - `hidden_act` in {"silu", "swish"}
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in {"silu", "swish"}:
            raise ValueError(f"Activation function {config.hidden_act} is not supported for SwiGLU.")

    def forward(self, x):
        return self.down_proj(ForgeSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
