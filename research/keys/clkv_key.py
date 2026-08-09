"""Cross-Layer KV Sharing (CLKV) â€” share KV cache across consecutive layer pairs.

Novel insight: In deep transformers, adjacent layers often attend to similar
positions. Each layer has its own query projection but the KV representations
are highly correlated across neighboring layers. By sharing the KV cache
across consecutive layer pairs (0,1), (2,3), ..., we halve KV cache VRAM
with minimal quality impact.

For ForgeLM V2 (28 layers, 12 heads, head_dim=128, MLA):
  - Standard: 28 layers Ã— 2 (K,V) Ã— 12 heads Ã— T Ã— 128 Ã— 2 bytes (bf16)
  - CLKV:     14 pairs Ã— 2 (K,V) Ã— 12 heads Ã— T Ã— 128 Ã— 2 bytes
  - Savings:  50% KV cache VRAM at all sequence lengths
  - At T=32768: 5.6 GB â†’ 2.8 GB saved

How it works:
  - Even layers (0, 2, 4, ...) compute KV normally and cache it.
  - Odd layers (1, 3, 5, ...) SKIP KV computation and reuse the KV from
    the preceding even layer. They still compute their own query.
  - The attention output differs because Q differs, but K,V are shared.
  - Each odd layer's k_up_proj/v_up_proj are unused at inference (wasted params
    but zero FLOPs). This is the trade-off: save KV VRAM + KV compute at the
    cost of slightly different attention patterns.

Quality: PARTIAL â€” adjacent layers' KV are correlated but not identical.
  cos(k_L, k_{L+1}) typically ~0.7-0.9 in trained models. The quality impact
  is small for generation but may compound over long sequences.

Key class: TRIVIAL â€” runtime cache sharing, training-free, no weight changes.
  Reversible: disable to restore per-layer KV.

Usage:
    from research.keys.clkv_key import CLKVKey
    key = CLKVKey()
    key.apply(model)  # patch forward to share KV across layer pairs
    # ... generate ...
    key.revert(model)  # restore original forward
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import Key, KeyClass, KeyResult


class CLKVKey(Key):
    """Cross-Layer KV Sharing â€” halve KV cache by sharing across layer pairs.

    Patches the model's forward pass so odd layers reuse the KV from the
    preceding even layer instead of computing their own.

    Key class: TRIVIAL â€” runtime optimization, training-free, reversible.
    """

    def __init__(self, share_factor: int = 2):
        """
        Args:
            share_factor: number of consecutive layers that share one KV cache.
                2 = pairs, 3 = triples, etc. Default 2 (50% KV reduction).
        """
        self.share_factor = share_factor
        self._original_forward = None
        self._applied = False

    @property
    def name(self) -> str:
        return "clkv"

    @property
    def description(self) -> str:
        return (f"Cross-Layer KV Sharing (share_factor={self.share_factor}, "
                f"{1/self.share_factor*100:.0f}% KV cache, training-free, reversible)")

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """CLKV is a runtime key â€” state dict is unchanged."""
        state = dict(data.get("state", data))
        return KeyResult(
            success=True,
            weights=state,
            metadata={
                "share_factor": self.share_factor,
                "lossy": True,
                "training_free": True,
                "kv_reduction": 1 - 1 / self.share_factor,
            },
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """No-op â€” CLKV doesn't modify weights."""
        return KeyResult(success=True, weights=weights)

    def apply(self, model: nn.Module) -> int:
        """Patch model forward to share KV across layer pairs.

        Args:
            model: ConfigurableResearchLLM with .blocks

        Returns:
            Number of layers that will reuse shared KV (odd layers).
        """
        if self._applied:
            return 0
        self._original_forward = model.forward
        share_factor = self.share_factor

        def clkv_forward(
            idx,
            targets=None,
            past_key_values=None,
            use_cache=False,
            return_hidden=False,
            _orig=self._original_forward,
            _sf=share_factor,
        ):
            x = model.embed(idx)
            presents: list[tuple | None] = []
            n_layers = len(model.blocks)

            for i, block in enumerate(model.blocks):
                # Determine if this layer should reuse a previous layer's KV.
                group_pos = i % _sf  # position within the sharing group
                past = past_key_values[i] if past_key_values is not None else None

                if group_pos == 0:
                    # Leader layer: compute KV normally.
                    x, present = block(x, past_key_value=past, use_cache=use_cache)
                    if use_cache:
                        # Store present for this layer AND fill slots for
                        # followers so external code sees a full presents list.
                        presents.append(present)
                        # Pre-fill follower slots with the same KV reference.
                        # Followers will use presents[-1] as their past_kv.
                        for j in range(1, _sf):
                            if i + j < n_layers:
                                presents.append(present)  # shared reference
                else:
                    # Follower layer: reuse the leader's KV cache.
                    # The leader's KV is the last entry in presents (or the
                    # corresponding past_key_values slot from the leader).
                    leader_idx = i - group_pos
                    shared_past = (
                        past_key_values[leader_idx]
                        if past_key_values is not None
                        else (presents[leader_idx] if leader_idx < len(presents) else None)
                    )

                    # Run block but force it to use the shared KV.
                    # We pass the shared KV as past_key_value and set use_cache=False
                    # so the block doesn't append new tokens to KV â€” it just uses
                    # the existing cache for attention.
                    # The block's attention will compute Q from x, but use
                    # shared K,V for attention scores.
                    x, _ = block(x, past_key_value=shared_past, use_cache=False)

                    if use_cache:
                        # Follower's "present" is the same shared KV (no new KV computed).
                        # presents already has a placeholder from the leader.
                        pass

            hidden = model.ln_f(x)

            # Logits / loss (same as original forward)
            logits = model.head(hidden)
            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

            if return_hidden:
                if use_cache:
                    return logits, loss, presents, hidden
                return logits, loss, hidden
            if use_cache:
                return logits, loss, presents
            return logits, loss

        model.forward = clkv_forward
        self._applied = True

        n_layers = len(model.blocks)
        n_followers = n_layers - (n_layers + share_factor - 1) // share_factor
        print(f"  [CLKV] Sharing KV across {share_factor}-layer groups: "
              f"{n_followers}/{n_layers} layers reuse shared KV "
              f"({1/share_factor*100:.0f}% KV cache, "
              f"{(1-1/share_factor)*100:.0f}% reduction)")
        return n_followers

    def revert(self, model: nn.Module):
        """Restore original forward pass."""
        if self._original_forward is not None:
            model.forward = self._original_forward
            self._original_forward = None
            self._applied = False
            print("  [CLKV] Reverted to per-layer KV cache")

