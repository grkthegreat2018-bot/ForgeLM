"""SwiGLU clamp key — GPT-OSS style clamped SwiGLU activation.

GPT-OSS modifies SwiGLU with:
1. Value clamping: gate clamped to [None, limit], up clamped to [-limit, limit]
2. Scaled sigmoid: σ(α·x) with α=1.702 (approximates GELU)
3. Residual: (up + 1) * glu instead of up * glu

The clamping prevents outlier activations from destabilizing quantized
inference. The +1 residual ensures the linear path always contributes.

This is a TRIVIAL key — no weight changes, just a modified activation
function. The limit and alpha are hyperparameters, not learned.

Key class: TRIVIAL — fixed formula, no data or training.

Reference: GPT-OSS, openai/gpt-oss
"""
import torch
import torch.nn.functional as F

from research.keys.base import Key, KeyClass, KeyResult


class SwiGLUClampKey(Key):
    """GPT-OSS SwiGLU clamp key — clamped activation with residual.

    Modifies the SwiGLU activation to include:
    - Value clamping (prevents outliers)
    - Scaled sigmoid (α=1.702, approximates GELU)
    - +1 residual on the linear path

    No weight changes — only the activation function changes.

    Key class: TRIVIAL — fixed formula, no data or training.
    """

    @property
    def name(self) -> str:
        return "swiglu_clamp"

    @property
    def description(self) -> str:
        return "GPT-OSS clamped SwiGLU with scaled sigmoid and residual"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Apply clamped SwiGLU to gate_up activations.

        Args:
            data: {"gate_up": tensor (..., 2*d_ff) — interleaved gate/up,
                   "alpha": float (default 1.702),
                   "limit": float (default 7.0)}

        Returns:
            {"output": tensor (..., d_ff) — SwiGLU output}
        """
        try:
            gate_up = data["gate_up"]
            alpha = data.get("alpha", 1.702)
            limit = data.get("limit", 7.0)

            # Split interleaved: even=gate, odd=up
            gate = gate_up[..., ::2]
            up = gate_up[..., 1::2]

            # Clamp
            gate = gate.clamp(min=None, max=limit)
            up = up.clamp(min=-limit, max=limit)

            # Scaled sigmoid SwiGLU
            glu = gate * torch.sigmoid(alpha * gate)

            # Residual: (up + 1) * glu
            output = (up + 1) * glu

            return KeyResult(
                success=True,
                weights={"output": output},
                metadata={"alpha": alpha, "limit": limit},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Cannot reverse activation function."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"lossy": True, "runtime_only": True},
        )


def swiglu_clamp(gate_up, alpha=1.702, limit=7.0):
    """Functional clamped SwiGLU (GPT-OSS style).

    Args:
        gate_up: (..., 2*d_ff) interleaved gate and up projections
        alpha: sigmoid scale (1.702 approximates GELU)
        limit: clamp value (7.0 prevents outliers)

    Returns:
        (..., d_ff) SwiGLU output
    """
    gate = gate_up[..., ::2].clamp(min=None, max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(alpha * gate))


if __name__ == "__main__":
    key = SwiGLUClampKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    gate_up = torch.randn(4, 128)  # 2*d_ff = 128, d_ff = 64
    r = key.forward({"gate_up": gate_up, "alpha": 1.702, "limit": 7.0})
    print(f"Forward: {r.success}")
    print(f"  Output shape: {r.weights['output'].shape}")
    # Verify clamping works
    large_input = torch.full((4, 128), 100.0)
    r2 = key.forward({"gate_up": large_input, "limit": 7.0})
    # Gate clamped to 7, up clamped to 7
    # glu = 7 * sigmoid(1.702 * 7) ≈ 7 * 1.0 = 7
    # output = (7 + 1) * 7 = 56
    assert r2.weights["output"][0, 0].item() < 60  # bounded
    print(f"  Clamping verified: large input → bounded output ({r2.weights['output'][0,0]:.2f})")
