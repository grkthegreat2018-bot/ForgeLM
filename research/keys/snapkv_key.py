"""SnapKV key — attention-based KV cache eviction without training.

SnapKV (NeurIPS 2024) compresses the KV cache by identifying which prompt
tokens are important based on attention patterns from an "observation window"
(the last few tokens of the prompt). Tokens that receive high attention
from the observation window are kept; the rest are evicted.

This is a PARTIAL key — needs one forward pass to get attention scores,
but no training or weight updates.

Reference: SnapKV, arxiv 2404.14469
"""
import torch
import torch.nn.functional as F

from research.keys.base import Key, KeyClass, KeyResult


class SnapKVKey(Key):
    """SnapKV KV cache eviction key — attention-based token selection.

    Uses attention scores from an observation window (last few tokens)
    to identify important KV positions. No training needed — just one
    forward pass to collect attention scores.

    Key class: PARTIAL — needs attention scores from one forward pass.
    """

    @property
    def name(self) -> str:
        return "snapkv"

    @property
    def description(self) -> str:
        return "Attention-based KV cache eviction (observation window scoring)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Compute KV cache eviction mask from attention scores.

        Args:
            data: {"attention_scores": tensor (n_heads, obs_window, seq_len),
                   "budget": int or float — number of tokens to keep (int)
                             or fraction (float, e.g. 0.3),
                   "n_sinks": int (default 4) — attention sink tokens to always keep,
                   "window_size": int (default 0) — recent window to always keep}

        Returns:
            {"cache_mask": tensor (seq_len,) bool — True for kept tokens,
             "n_kept": int,
             "importance_scores": tensor (seq_len,) — per-token importance}
        """
        try:
            attn = data["attention_scores"]  # (n_heads, obs_window, seq_len)
            budget = data["budget"]
            n_sinks = data.get("n_sinks", 4)
            window_size = data.get("window_size", 0)
            seq_len = attn.shape[-1]

            # Convert budget fraction to count
            if isinstance(budget, float):
                budget = int(seq_len * budget)

            # Compute importance: sum of attention from observation window
            # Average across heads, sum across observation window
            importance = attn.mean(dim=0).sum(dim=0)  # (seq_len,)

            # Build mask
            mask = torch.zeros(seq_len, dtype=torch.bool)

            # Always keep sinks (first n_sinks tokens)
            n_sinks = min(n_sinks, seq_len)
            mask[:n_sinks] = True

            # Always keep recent window
            if window_size > 0:
                window_start = max(n_sinks, seq_len - window_size)
                mask[window_start:] = True

            # Count already-kept tokens
            n_kept = mask.sum().item()
            remaining_budget = max(0, budget - n_kept)

            # Select top-important tokens from the middle (not sinks, not window)
            middle_start = n_sinks
            middle_end = seq_len - window_size if window_size > 0 else seq_len
            if middle_end > middle_start and remaining_budget > 0:
                middle_importance = importance[middle_start:middle_end]
                n_middle = middle_end - middle_start
                n_select = min(remaining_budget, n_middle)
                if n_select > 0:
                    top_indices = middle_importance.topk(n_select).indices + middle_start
                    mask[top_indices] = True

            n_kept = mask.sum().item()

            return KeyResult(
                success=True,
                weights={"cache_mask": mask, "importance_scores": importance},
                metadata={
                    "n_kept": n_kept, "n_evicted": seq_len - n_kept,
                    "budget": budget, "n_sinks": n_sinks,
                    "window_size": window_size, "seq_len": seq_len,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Cannot recover evicted tokens (lossy)."""
        return KeyResult(
            success=True,
            data={"cache_mask": weights["cache_mask"]},
            metadata={"lossy": True},
        )


def compute_snapkv_from_model(model, input_ids, budget=0.3, n_sinks=4, window_size=32):
    """Run model with attention hooks, compute SnapKV eviction mask per layer.

    Args:
        model: the LLM
        input_ids: input token IDs (1D or 2D)
        budget: fraction of tokens to keep (float) or count (int)
        n_sinks: number of attention sink tokens to always keep
        window_size: observation window size (last N tokens)

    Returns:
        dict of {layer_idx: {"cache_mask": tensor, "n_kept": int}}
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    attention_weights = {}
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            # Try to capture attention weights from output
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1]
                if attn is not None and attn.dim() >= 3:
                    attention_weights[idx] = attn.detach()
        return hook

    for i, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    results = {}
    for idx, attn in attention_weights.items():
        # attn: (batch, n_heads, seq, seq) — take last window_size tokens as observation
        seq_len = attn.shape[-1]
        obs_start = max(0, seq_len - window_size)
        obs_attn = attn[0, :, obs_start:, :]  # (n_heads, obs_window, seq_len)

        key = SnapKVKey()
        result = key.forward({
            "attention_scores": obs_attn,
            "budget": budget,
            "n_sinks": n_sinks,
            "window_size": 0,  # SnapKV doesn't force a recent window
        })
        if result.success:
            results[idx] = {
                "cache_mask": result.weights["cache_mask"],
                "n_kept": result.metadata["n_kept"],
            }

    return results


if __name__ == "__main__":
    key = SnapKVKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Synthetic test: 128 tokens, 4 heads, 16-token observation window
    seq_len = 128
    attn = torch.softmax(torch.randn(4, 16, seq_len), dim=-1)

    r = key.forward({"attention_scores": attn, "budget": 0.3, "n_sinks": 4})
    print(f"Forward: {r.success}")
    print(f"  Kept: {r.metadata['n_kept']}/{seq_len} ({r.metadata['n_kept']/seq_len:.0%})")
    print(f"  Sinks kept: {r.weights['cache_mask'][:4].all().item()}")
    print(f"  Importance shape: {r.weights['importance_scores'].shape}")
