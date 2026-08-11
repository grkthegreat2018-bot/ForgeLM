"""Value Residual key — copy V projection from layer 0 to all layers.

ResFormer (ACL 2025) adds a value residual connection: the value vectors
from the first attention layer are added to all subsequent layers' values.
This preserves token-level information that gets smoothed out in deep layers.

SVFormer variant: all layers SHARE the first layer's V projection weight,
reducing KV cache by ~50%.

The key is trivial: copy V projection weight from layer 0 to all layers.
This is a FULL key — direct weight copy, no training needed.

Reference: ResFormer, arxiv 2410.17897
"""
import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class ValueResidualKey(Key):
    """Value residual connection key — copy V from layer 0.

    Two modes:
    1. ResFormer: add V_0 residual to each layer's V (V_i' = V_i + V_0)
    2. SVFormer: share V_0 across all layers (V_i = V_0, halves KV cache)

    Key class: FULL — direct weight copy, no training.
    """

    @property
    def name(self) -> str:
        return "value_residual"

    @property
    def description(self) -> str:
        return "Copy V projection from layer 0 to all layers (ResFormer/SVFormer)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict) -> KeyResult:
        """Derive value residual weights from layer 0's V projection.

        Args:
            data: {"v_weights": list of tensors (n_layers, d_model, d_model),
                   "mode": "resformer" or "svformer"}

        Returns:
            {"v_weights": list of modified tensors,
             "v_residual_weight": tensor (layer 0's V weight)}
        """
        try:
            v_weights = data["v_weights"]
            mode = data.get("mode", "resformer")
            n_layers = len(v_weights)

            # Layer 0's V projection is the residual source
            v0 = v_weights[0].clone()

            if mode == "svformer":
                # SVFormer: all layers share V_0
                new_v = [v0.clone() for _ in range(n_layers)]
            else:
                # ResFormer: V_i' = V_i + V_0 (additive residual)
                new_v = [v_weights[0].clone()]  # layer 0 unchanged
                for i in range(1, n_layers):
                    new_v.append(v_weights[i] + v0)

            return KeyResult(
                success=True,
                weights={"v_weights": new_v, "v_residual_weight": v0},
                metadata={"mode": mode, "n_layers": n_layers},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Extract original V weights (subtract V_0 from layers 1+)."""
        try:
            v_weights = weights["v_weights"]
            v0 = weights["v_residual_weight"]
            original = [v_weights[0].clone()]
            for i in range(1, len(v_weights)):
                original.append(v_weights[i] - v0)
            return KeyResult(success=True, data={"v_weights": original})
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def init_value_residual_in_model(model, mode="resformer"):
    """Add value residual connections to a model (in-place).

    Args:
        model: ConfigurableResearchLLM with .blocks
        mode: "resformer" (additive) or "svformer" (shared)

    Returns:
        Number of layers modified.
    """
    # Find V projection in layer 0
    v0_weight = None
    for name in ["v_proj", "v", "value", "v_down_proj"]:
        if hasattr(model.blocks[0].attn, name):
            v0_weight = getattr(model.blocks[0].attn, name).weight.data.clone()
            break

    if v0_weight is None:
        raise ValueError("Could not find V projection in layer 0")

    key = ValueResidualKey()

    # Collect all V weights
    v_weights = []
    for block in model.blocks:
        for name in ["v_proj", "v", "value", "v_down_proj"]:
            if hasattr(block.attn, name):
                v_weights.append(getattr(block.attn, name).weight.data)
                break

    result = key.forward({"v_weights": v_weights, "mode": mode})
    if not result.success:
        raise RuntimeError(f"Value residual key failed: {result.error}")

    # Apply modified weights
    modified = 0
    for i, block in enumerate(model.blocks):
        for name in ["v_proj", "v", "value", "v_down_proj"]:
            if hasattr(block.attn, name):
                getattr(block.attn, name).weight.data.copy_(result.weights["v_weights"][i])
                modified += 1
                break

    return modified


if __name__ == "__main__":
    key = ValueResidualKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test with 4 layers
    d = 256
    v_weights = [torch.randn(d, d) for _ in range(4)]

    # ResFormer mode
    r = key.forward({"v_weights": v_weights, "mode": "resformer"})
    print(f"ResFormer: {r.success}, {len(r.weights['v_weights'])} layers")
    # Layer 0 unchanged
    assert (r.weights["v_weights"][0] - v_weights[0]).abs().max() < 1e-6
    # Layer 1 = V_1 + V_0
    assert (r.weights["v_weights"][1] - (v_weights[1] + v_weights[0])).abs().max() < 1e-6
    print("  ResFormer: layer 0 unchanged, layers 1+ = V_i + V_0 ✓")

    # SVFormer mode
    r2 = key.forward({"v_weights": v_weights, "mode": "svformer"})
    print(f"SVFormer: {r2.success}")
    # All layers = V_0
    for i in range(4):
        assert (r2.weights["v_weights"][i] - v_weights[0]).abs().max() < 1e-6
    print("  SVFormer: all layers = V_0 ✓")

    # Reverse
    rv = key.reverse(r.weights)
    for i in range(1, 4):
        assert (rv.data["v_weights"][i] - v_weights[i]).abs().max() < 1e-5
    print("  Reverse: recovered original V weights ✓")
