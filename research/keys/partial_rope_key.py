"""MHA2MLA partial-RoPE key — remove RoPE from low-contribution dimensions.

MHA2MLA (ACL 2025) converts MHA/GQA models to MLA by:
1. Identifying which RoPE dimensions contribute least to attention scores
2. Removing RoPE from those dimensions (converting them to NoPE)
3. The NoPE dimensions can then be compressed via SVD (low-rank)

This is the partial-RoPE step. The key insight: not all RoPE dimensions matter
equally. High-frequency dimensions (short wavelength) encode local position
and are critical. Low-frequency dimensions (long wavelength) barely change
across positions and can be removed with minimal quality loss.

We measure RoPE dimension contribution by:
- Computing attention with each dimension ablated (set to 0)
- Dimensions where ablation has small effect → safe to remove
- Or: use the RoPE frequency itself (low-freq = low contribution)

Key class: PARTIAL — needs calibration data to measure dimension importance,
or can use the frequency-based heuristic (no data needed).

Reference: MHA2MLA, arxiv 2502.14814
"""
import math

import torch

from research.keys.base import Key, KeyClass, KeyResult


class PartialRoPEKey(Key):
    """Partial-RoPE key — remove RoPE from low-contribution dimensions.

    Two modes:
    1. "frequency" (no data): remove RoPE from lowest-frequency dimensions
       (longest wavelength = least position information)
    2. "calibration" (needs data): measure attention change per dimension,
       remove those with smallest impact

    Key class: PARTIAL (calibration mode) or TRIVIAL (frequency mode).
    """

    @property
    def name(self) -> str:
        return "partial_rope"

    @property
    def description(self) -> str:
        return "Remove RoPE from low-contribution dimensions (MHA2MLA step 1)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Determine which RoPE dimensions to keep vs remove.

        Args:
            data: {"head_dim": int,
                   "base": float (RoPE base, default 10000),
                   "remove_ratio": float (fraction of dims to remove, e.g. 0.5),
                   "mode": "frequency" or "calibration",
                   "attention_scores": tensor (optional, for calibration mode)}

        Returns:
            {"keep_mask": tensor (head_dim//2,) bool — True for kept RoPE dims,
             "n_kept": int,
             "n_removed": int}
        """
        try:
            head_dim = data["head_dim"]
            base = data.get("base", 10000.0)
            remove_ratio = data.get("remove_ratio", 0.5)
            mode = data.get("mode", "frequency")

            n_freqs = head_dim // 2
            n_remove = int(n_freqs * remove_ratio)
            n_keep = n_freqs - n_remove

            if mode == "frequency":
                # Frequency-based: low-frequency dims (long wavelength) are
                # least important for position encoding
                freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
                # Low frequency = small freq value = long wavelength
                # Sort by frequency descending (keep high-freq, remove low-freq)
                sorted_indices = freqs.argsort(descending=True)
                keep_indices = sorted_indices[:n_keep]
                keep_mask = torch.zeros(n_freqs, dtype=torch.bool)
                keep_mask[keep_indices] = True

            elif mode == "calibration":
                # Calibration-based: measure attention change per dimension
                attn = data.get("attention_scores")
                if attn is None:
                    # Fall back to frequency mode
                    return self.forward({**data, "mode": "frequency"})

                # Compute per-dimension importance from attention entropy
                # High-entropy attention = dimension matters more
                # attn: (n_heads, seq, seq) — we measure how much each
                # RoPE dimension contributes to the attention pattern
                # Simplified: use attention score variance per head
                # (higher variance = more selective = more position-dependent)
                # This is a proxy; true calibration would ablate each dim
                attn_var = attn.var(dim=-1).mean(dim=(0, 1))  # (seq,)
                # Map to frequency dims (approximate)
                importance = torch.ones(n_freqs)
                for i in range(n_freqs):
                    importance[i] = attn_var[min(i, len(attn_var) - 1)].item()

                sorted_indices = importance.argsort(descending=True)
                keep_indices = sorted_indices[:n_keep]
                keep_mask = torch.zeros(n_freqs, dtype=torch.bool)
                keep_mask[keep_indices] = True

            else:
                return KeyResult(success=False, error=f"Unknown mode: {mode}")

            return KeyResult(
                success=True,
                weights={"keep_mask": keep_mask},
                metadata={
                    "n_kept": n_keep, "n_removed": n_remove,
                    "mode": mode, "remove_ratio": remove_ratio,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Cannot restore removed RoPE dimensions."""
        return KeyResult(
            success=True,
            data={"keep_mask": weights["keep_mask"]},
            metadata={"lossy": True},
        )


def apply_partial_rope_to_model(model, remove_ratio=0.5, mode="frequency"):
    """Apply partial-RoPE to a model (modifies which dims get RoPE).

    Args:
        model: ConfigurableResearchLLM
        remove_ratio: fraction of RoPE dimensions to remove
        mode: "frequency" (no data) or "calibration" (needs attention scores)

    Returns:
        dict of {layer_idx: keep_mask}
    """
    results = {}
    for i, block in enumerate(model.blocks):
        attn = block.attn
        head_dim = getattr(attn, 'head_dim', None)
        if head_dim is None:
            # Infer from q_proj
            if hasattr(attn, 'q_proj'):
                n_heads = getattr(attn, 'n_heads', 12)
                head_dim = attn.q_proj.out_features // n_heads
            else:
                continue

        base = getattr(attn, 'base', 10000.0)
        key = PartialRoPEKey()
        result = key.forward({
            "head_dim": head_dim,
            "base": base,
            "remove_ratio": remove_ratio,
            "mode": mode,
        })
        if result.success:
            keep_mask = result.weights["keep_mask"]
            # Store the mask on the attention layer
            attn.rope_keep_mask = keep_mask
            results[i] = keep_mask

    return results


if __name__ == "__main__":
    key = PartialRoPEKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Frequency mode (no data needed)
    r = key.forward({"head_dim": 128, "base": 10000, "remove_ratio": 0.5, "mode": "frequency"})
    print(f"Frequency mode: {r.success}")
    print(f"  Kept: {r.metadata['n_kept']}, Removed: {r.metadata['n_removed']}")
    print(f"  Mask: {r.weights['keep_mask'][:10].tolist()}... (high-freq kept)")

    # Calibration mode
    attn = torch.softmax(torch.randn(4, 32, 32), dim=-1)
    r2 = key.forward({"head_dim": 128, "remove_ratio": 0.5, "mode": "calibration",
                      "attention_scores": attn})
    print(f"Calibration mode: {r2.success}")
    print(f"  Kept: {r2.metadata['n_kept']}")
