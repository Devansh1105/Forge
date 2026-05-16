from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("SwiGLU sample requires CUDA", allow_module_level=True)

def _reference_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


@pytest.mark.parametrize(
    "shape",
    [
        (2, 128),
        (4, 16, 256),
        (6, 42, 431),
    ],
)
def test_swiglu_function_matches_torch_forward_backward(shape):
    from forge.ops import ForgeSiLUMulFunction

    gate = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    reference_gate = gate.detach().clone().requires_grad_(True)
    reference_up = up.detach().clone().requires_grad_(True)

    actual = ForgeSiLUMulFunction.apply(gate, up)
    expected = _reference_swiglu(reference_gate, reference_up)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(gate.grad, reference_gate.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(up.grad, reference_up.grad, rtol=1e-5, atol=1e-6)


def test_swiglu_mlp_sample_matches_reference_math():
    from forge.transformers import ForgeSwiGLUMLP

    config = SimpleNamespace(hidden_size=128, intermediate_size=256, hidden_act="silu")
    forge_mlp = ForgeSwiGLUMLP(config).cuda()
    reference_mlp = ForgeSwiGLUMLP(config).cuda()
    reference_mlp.load_state_dict(forge_mlp.state_dict())

    x = torch.randn(2, 16, config.hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = forge_mlp(x)
    expected = reference_mlp.down_proj(
        _reference_swiglu(reference_mlp.gate_proj(reference_x), reference_mlp.up_proj(reference_x))
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-5)
