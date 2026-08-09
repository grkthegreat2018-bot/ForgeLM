"""ALiBi key — attention linear biases for length extrapolation.

ALiBi (ICLR 2022) replaces positional embeddings with a simple linear bias
on attention scores: score[i,j] -= m_h * |i - j|, where m_h is a per-head
slope (geometric sequence, NOT learned).

No weight changes needed — only the attention bias computation changes.
The slopes m_h are fixed: m_h = 1 / (2^(n_heads_ratio)).

This is a TRIVIAL key — pure formula, no data or training. The model's
weights are completely unchanged; only the attention bias is added at runtime.

ALiBi enables training on short sequences and extrapolating to longer ones.

Reference: ALiBi, arxiv 2108.12409
"""
import math

import torch

from research.keys.base import Key, KeyClass, KeyResult


class ALiBiKey(Key):
    """ALiBi attention linear bias key — distance-based position encoding.

    Replaces RoPE/positional embeddings with a linear distance penalty
    on attention scores. No weight changes — only runtime bias.

    Key class: TRIVIAL — fixed formula, no data or training.
    """

    @property
    def name(self) -> str:
        return "alibi"

    @property
    def description(self) -> str:
        return "Attention linear biases (distance-based position, no weights)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Compute ALiBi bias matrix and per-head slopes.

        Args:
            data: {"n_heads": int,
                   "seq_len": int (default 1024),
                   "max_seq_len": int (optional, for precomputed bias)}

        Returns:
            {"slopes": tensor (n_heads,) — per-head slope values,
             "bias": tensor (1, n_heads, seq_len, seq_len) — bias matrix}
        """
        try:
            n_heads = data["n_heads"]
            seq_len = data.get("seq_len", 1024)

            # Compute slopes: geometric sequence
            # m_h = 1 / 2^(n * (2^(-log2(n_heads))))
            # Simplified: start at 2^(-8/n_heads) and halve each head
            slopes = self._get_alibi_slopes(n_heads)

            # Build bias matrix
            # bias[h, i, j] = -slopes[h] * |i - j|
            positions = torch.arange(seq_len)
            relative_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
            # (seq_len, seq_len)
            bias = -slopes.unsqueeze(1).unsqueeze(2) * relative_dist.unsqueeze(0)
            # (n_heads, seq_len, seq_len) → (1, n_heads, seq_len, seq_len)
            bias = bias.unsqueeze(0)

            return KeyResult(
                success=True,
                weights={"slopes": slopes, "bias": bias},
                metadata={
                    "n_heads": n_heads, "seq_len": seq_len,
                    "method": "geometric_slopes",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def _get_alibi_slopes(self, n_heads):
        """Compute ALiBi slopes for n_heads.

        Uses the geometric sequence: m_h = 2^(-8 * h / n_heads) for h=0..n_heads-1
        (closest power of 2 approach from the original paper).
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-8 / n)  # 2^(-8/n)
            return torch.tensor([start ** i for i in range(n)])

        if (n_heads & (n_heads - 1)) == 0:
            # Power of 2 — simple geometric sequence
            return get_slopes_power_of_2(n_heads)
        else:
            # Not power of 2 — use closest power of 2 and interpolate
            n_near = 2 ** math.floor(math.log2(n_heads))
            slopes_near = get_slopes_power_of_2(n_near)
            # Add extra slopes for remaining heads
            n_extra = n_heads - n_near
            slopes_extra = get_slopes_power_of_2(2 * n_near)[n_near::2][:n_extra]
            return torch.cat([slopes_near, slopes_extra])

    def reverse(self, weights: dict) -> KeyResult:
        """ALiBi is a runtime bias — no weights to reverse."""
        return KeyResult(
            success=True,
            data={"slopes": weights["slopes"]},
            metadata={"reversible": True, "runtime_only": True},
        )


def apply_alibi_to_model(model, seq_len=1024):
    """Replace RoPE with ALiBi in a model (modifies attention bias).

    Args:
        model: ConfigurableResearchLLM
        seq_len: sequence length for precomputed bias

    Returns:
        Per-head slopes tensor.
    """
    n_heads = getattr(model.config, 'n_heads', 12)
    key = ALiBiKey()
    result = key.forward({"n_heads": n_heads, "seq_len": seq_len})

    if not result.success:
        raise RuntimeError(f"ALiBi key failed: {result.error}")

    slopes = result.weights["slopes"]

    # Apply to all attention layers
    for block in model.blocks:
        attn = block.attn
        attn.alibi_slopes = slopes
        # Disable RoPE if present
        if hasattr(attn, 'use_rope'):
            attn.use_rope = False

    return slopes


if __name__ == "__main__":
    key = ALiBiKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"n_heads": 8, "seq_len": 16})
    print(f"Forward: {r.success}")
    print(f"  Slopes: {[f'{s:.6f}' for s in r.weights['slopes'].tolist()]}")
    print(f"  Bias shape: {r.weights['bias'].shape}")
    # Verify slopes are geometric (halving)
    slopes = r.weights["slopes"]
    for i in range(1, len(slopes)):
        ratio = slopes[i] / slopes[i-1]
        assert abs(ratio - slopes[1] / slopes[0]) < 1e-4, f"Slope ratio inconsistent at {i}"
    print("  Geometric slopes verified ✓")
    # Verify bias is negative (penalty)
    assert (r.weights["bias"] <= 0).all()
    print("  All biases ≤ 0 (penalty) ✓")
