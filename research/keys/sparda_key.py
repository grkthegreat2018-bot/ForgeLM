"""SparDA Forecast key — init Forecast projection from K projection.

SparDA (2025) adds a fourth per-layer projection (Forecast) alongside
Q, K, V that predicts which KV blocks the next layer will need. This
enables lookahead prefetch for sparse attention.

The Forecast projection can be initialized from the K projection:
- K projects hidden states to keys (what to attend to)
- Forecast projects hidden states to predict future attention needs
- They're correlated: tokens that produce certain keys will likely
  be attended to by the next layer

Key: copy K projection weights as Forecast initialization.
This gives a meaningful starting point — the Forecast predicts that
tokens similar to the current layer's keys will be needed.

Key class: PARTIAL — approximate init, needs light fine-tuning.

Reference: SparDA, arxiv 2606.04511
"""
import torch
import torch.nn as nn
from research.keys.base import Key, KeyClass, KeyResult


class SparDAKey(Key):
    """SparDA Forecast projection key — init from K projection.

    Copies the K projection weight as the Forecast projection weight.
    The Forecast then predicts which KV blocks the next layer needs,
    based on the correlation between current keys and future attention.

    Key class: PARTIAL — approximate init, needs fine-tune.
    """

    @property
    def name(self) -> str:
        return "sparda"

    @property
    def description(self) -> str:
        return "Init Forecast projection from K projection (SparDA sparse attention)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Derive Forecast weights from K projection.

        Args:
            data: {"k_weight": tensor (n_kv_heads * head_dim, d_model),
                   "n_kv_heads": int,
                   "head_dim": int,
                   "d_model": int}

        Returns:
            {"forecast_weight": tensor (same shape as k_weight),
             "forecast_bias": tensor or None}
        """
        try:
            k_weight = data["k_weight"]

            # Forecast = copy of K (they project from the same input space)
            # The Forecast predicts attention needs, K produces keys
            # At init, Forecast ≈ K means "predict that tokens similar to
            # current keys will be needed by the next layer"
            forecast_weight = k_weight.clone()

            # Scale down slightly — Forecast is a predictor, not a direct projection
            # Scaling by 1/sqrt(2) gives a softer prediction
            forecast_weight = forecast_weight / (2 ** 0.5)

            # Bias: copy from K if available, else zero
            k_bias = data.get("k_bias")
            if k_bias is not None:
                forecast_bias = k_bias.clone() / (2 ** 0.5)
            else:
                forecast_bias = None

            return KeyResult(
                success=True,
                weights={"forecast_weight": forecast_weight,
                         "forecast_bias": forecast_bias},
                metadata={
                    "n_kv_heads": data.get("n_kv_heads"),
                    "head_dim": data.get("head_dim"),
                    "init_method": "k_projection_copy",
                    "scale": 1.0 / (2 ** 0.5),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Extract approximate K weights from Forecast (multiply by sqrt(2))."""
        try:
            forecast_weight = weights["forecast_weight"]
            k_weight = forecast_weight * (2 ** 0.5)
            return KeyResult(
                success=True,
                data={"k_weight": k_weight},
                metadata={"approximate": True},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def init_sparda_forecast_in_model(model):
    """Add SparDA Forecast projections to all attention layers (in-place).

    For each attention layer, adds a forecast_proj Linear with weights
    initialized from the K projection.

    Returns:
        Number of layers modified.
    """
    modified = 0
    for block in model.blocks:
        attn = block.attn

        # Find K projection
        k_weight = None
        k_bias = None
        k_proj = None
        for name in ["k_proj", "k", "key", "k_up_proj"]:
            if hasattr(attn, name):
                k_proj = getattr(attn, name)
                k_weight = k_proj.weight.data.clone()
                if hasattr(k_proj, 'bias') and k_proj.bias is not None:
                    k_bias = k_proj.bias.data.clone()
                break

        if k_weight is None:
            continue

        # Infer dimensions
        n_kv_heads = getattr(attn, 'n_kv_heads', getattr(attn, 'n_heads', 12))
        head_dim = k_weight.shape[0] // n_kv_heads
        d_model = k_weight.shape[1]

        key = SparDAKey()
        result = key.forward({
            "k_weight": k_weight,
            "k_bias": k_bias,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "d_model": d_model,
        })

        if result.success:
            # Create Forecast projection
            forecast = nn.Linear(d_model, k_weight.shape[0], bias=k_bias is not None)
            forecast.weight.data = result.weights["forecast_weight"]
            if result.weights["forecast_bias"] is not None:
                forecast.bias.data = result.weights["forecast_bias"]

            attn.forecast_proj = forecast
            modified += 1

    return modified


if __name__ == "__main__":
    key = SparDAKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    n_kv_heads, head_dim, d_model = 2, 64, 256
    k_weight = torch.randn(n_kv_heads * head_dim, d_model)

    r = key.forward({"k_weight": k_weight, "n_kv_heads": n_kv_heads,
                     "head_dim": head_dim, "d_model": d_model})
    print(f"Forward: {r.success}")
    print(f"  Forecast weight: {r.weights['forecast_weight'].shape}")
    print(f"  Scale: {r.metadata['scale']:.4f}")

    # Verify it's a scaled copy of K
    err = (r.weights["forecast_weight"] - k_weight / (2 ** 0.5)).abs().max().item()
    print(f"  Is scaled K copy: {err < 1e-6}")

    # Reverse
    rv = key.reverse(r.weights)
    err = (rv.data["k_weight"] - k_weight).abs().max().item()
    print(f"  Reverse err: {err:.2e}")
