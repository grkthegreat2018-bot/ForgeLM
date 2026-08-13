"""DenseFormer DWA key — depth-weighted averaging with identity init.

DenseFormer (NeurIPS 2024) adds a Depth-Weighted Average (DWA) module after
each transformer block. DWA at depth i computes a weighted average of all
previous block outputs: X_i = sum(α_{i,j} * X_j) for j=0..i.

Key insight: if α_{i,i} = 1 and all other α_{i,j} = 0, DWA is identity.
This means DenseFormer reduces to standard Transformer at initialization.
Fine-tuning then learns the optimal cross-layer mixing weights.

This is a TRIVIAL key — identity init, no data or training.

Reference: DenseFormer, arxiv 2402.02622
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class DenseFormerKey(Key):
    """DenseFormer DWA key — identity-init depth-weighted averaging.

    Initializes DWA weights so that α_{i,i} = 1, all others = 0.
    This makes DenseFormer behave exactly like standard Transformer.
    Fine-tuning learns optimal cross-layer information flow.

    Key class: TRIVIAL — identity init, no data or training.
    """

    @property
    def name(self) -> str:
        return "denseformer"

    @property
    def description(self) -> str:
        return "Depth-weighted averaging (DWA) with identity init"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize DWA weights for all layers.

        Args:
            data: {"n_layers": int,
                   "dilation": int (default 1, sparsifies DWA connections)}

        Returns:
            {"dwa_weights": list of tensors — dwa_weights[i] is (i+1,) vector}
        """
        try:
            n_layers = data["n_layers"]
            dilation = data.get("dilation", 1)

            dwa_weights = []
            for i in range(n_layers):
                # DWA at layer i has i+1 weights (for layers 0..i)
                w = torch.zeros(i + 1)
                w[i] = 1.0  # identity — only use current layer output

                # Apply dilation: zero out non-dilated connections
                # (keeps connections at multiples of dilation)
                if dilation > 1:
                    for j in range(i + 1):
                        if (i - j) % dilation != 0 and j != i:
                            w[j] = 0.0  # already zero, but explicit

                dwa_weights.append(w)

            return KeyResult(
                success=True,
                weights={"dwa_weights": dwa_weights},
                metadata={
                    "n_layers": n_layers, "dilation": dilation,
                    "init": "identity", "total_params": sum(len(w) for w in dwa_weights),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Identity init — reverse is passthrough."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"reversible": True},
        )


def apply_denseformer_to_model(model, dilation=1, safe=True):
    """Add DWA modules to a model (in-place).

    Uses safety validation. DWA weights are identity-init (α_{i,i}=1, others=0),
    so the model output should be unchanged after attachment.

    Args:
        model: ConfigurableResearchLLM with .blocks
        dilation: DWA dilation factor (1 = dense, >1 = sparse)
        safe: if True, use safe_apply with rollback on corruption.

    Returns:
        Total DWA parameters added.
    """
    def _apply(m):
        n_layers = len(m.blocks)
        key = DenseFormerKey()
        result = key.forward({"n_layers": n_layers, "dilation": dilation})
        if not result.success:
            raise RuntimeError(f"DenseFormer key failed: {result.error}")
        for i, block in enumerate(m.blocks):
            block.dwa_weights = torch.nn.Parameter(result.weights["dwa_weights"][i].clone())
        return m

    if safe:
        from research.keys.safety import safe_apply
        safe_apply(model, _apply, identity_init=True, atol=1e-5, rtol=1e-4)
    else:
        _apply(model)

    n_layers = len(model.blocks)
    key = DenseFormerKey()
    result = key.forward({"n_layers": n_layers, "dilation": dilation})
    return result.metadata["total_params"]


if __name__ == "__main__":
    key = DenseFormerKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"n_layers": 4, "dilation": 1})
    print(f"Forward: {r.success}")
    for i, w in enumerate(r.weights["dwa_weights"]):
        print(f"  Layer {i}: {w.tolist()}")
    # Verify identity init
    for i, w in enumerate(r.weights["dwa_weights"]):
        assert w[i] == 1.0
        assert all(w[j] == 0.0 for j in range(i))
    print("  Identity init verified ✓")
    print(f"  Total DWA params: {r.metadata['total_params']}")
