"""SandwichNorm key — normalization before AND after sublayers.

SandwichNorm places RMSNorm both before and after each sublayer
(attention and FFN). This combines the stability of Pre-Norm with
the representation control of Post-Norm.

The key insight: the post-norm can be initialized as identity
(scale=1, shift=0) so the model behaves like standard Pre-Norm
at initialization. Fine-tuning then learns the post-norm parameters.

Key class: TRIVIAL — identity-init post-norm, no data or training.

Reference: SandwichNorm, various 2024-2025 papers
"""
import torch
import torch.nn as nn

from research.keys.base import Key, KeyClass, KeyResult


class SandwichNormKey(Key):
    """SandwichNorm key — identity-init post-norm.

    Adds a post-sublayer RMSNorm initialized to identity (scale=1).
    The model behaves like standard Pre-Norm at init.

    Key class: TRIVIAL — identity init, no data or training.
    """

    @property
    def name(self) -> str:
        return "sandwich_norm"

    @property
    def description(self) -> str:
        return "Post-sublayer RMSNorm with identity init (SandwichNorm)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize post-norm parameters for all layers.

        Args:
            data: {"d_model": int, "n_layers": int}

        Returns:
            {"post_attn_norms": list of (weight, bias) — identity init,
             "post_ffn_norms": list of (weight, bias) — identity init}
        """
        try:
            d_model = data["d_model"]
            n_layers = data["n_layers"]

            # Identity RMSNorm: scale=1 (no shift for RMSNorm)
            post_attn = [torch.ones(d_model) for _ in range(n_layers)]
            post_ffn = [torch.ones(d_model) for _ in range(n_layers)]

            return KeyResult(
                success=True,
                weights={"post_attn_norms": post_attn, "post_ffn_norms": post_ffn},
                metadata={"d_model": d_model, "n_layers": n_layers, "init": "identity"},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Identity init — reverse is passthrough."""
        return KeyResult(success=True, data=weights, metadata={"reversible": True})


def apply_sandwich_norm_to_model(model):
    """Add post-sublayer RMSNorm to a model (in-place).

    Args:
        model: ConfigurableResearchLLM

    Returns:
        Number of norms added.
    """
    d_model = getattr(model.config, 'd_model', 256)
    n_layers = len(model.blocks)

    key = SandwichNormKey()
    result = key.forward({"d_model": d_model, "n_layers": n_layers})

    if not result.success:
        raise RuntimeError(f"SandwichNorm key failed: {result.error}")

    added = 0
    for i, block in enumerate(model.blocks):
        # Post-attention norm
        post_attn = nn.RMSNorm(d_model)
        post_attn.weight.data = result.weights["post_attn_norms"][i].clone()
        block.post_attn_norm = post_attn
        added += 1

        # Post-FFN norm
        post_ffn = nn.RMSNorm(d_model)
        post_ffn.weight.data = result.weights["post_ffn_norms"][i].clone()
        block.post_ffn_norm = post_ffn
        added += 1

    return added


if __name__ == "__main__":
    key = SandwichNormKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"d_model": 256, "n_layers": 4})
    print(f"Forward: {r.success}")
    print(f"  Post-attn norms: {len(r.weights['post_attn_norms'])}")
    print(f"  Post-FFN norms: {len(r.weights['post_ffn_norms'])}")
    # Verify identity init
    assert all((w == 1.0).all() for w in r.weights["post_attn_norms"])
    assert all((w == 1.0).all() for w in r.weights["post_ffn_norms"])
    print("  Identity init verified ✓")
