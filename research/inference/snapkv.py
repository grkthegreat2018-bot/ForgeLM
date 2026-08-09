"""SnapKV: observation-window KV cache eviction.

Inspired by SnapKV (Li et al., 2024): use attention scores from an initial
"observation window" to identify important tokens globally, then evict
low-importance tokens from the cache.

The key insight: attention patterns from recent tokens (observation window)
are good predictors of which older tokens will be important for future
attention. Tokens that receive high attention from the observation window
are "heavy hitters" and should be retained.

This is a pure runtime change — no weight modification, no training.

Usage:
    from research.inference.snapkv import SnapKVCache
    cache = SnapKVCache(observation_window=128, budget=512)
"""
from typing import Dict, List, Optional, Tuple

import torch


class SnapKVCache:
    """KV cache with SnapKV observation-window eviction.

    Structure:
    - Observation window: last `observation_window` tokens (always retained)
    - Budget region: top-`budget` tokens by accumulated attention score
    - When cache exceeds budget + observation_window, evict lowest-score tokens

    The attention scores are accumulated from the observation window's
    attention patterns, giving a data-driven importance ranking.
    """

    def __init__(self, observation_window: int = 128, budget: int = 512,
                 n_kv_heads: int = 2, head_dim: int = 128,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.obs_window = observation_window
        self.budget = budget
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None  # [n_kv, max_capacity] accumulated scores
        self.seq_len = 0
        self.max_capacity = budget + observation_window

    def bf16_or_dtype(self):
        """Return bf16 if using bf16 dtype (saves 2x VRAM for attention scores),
        otherwise return the configured dtype."""
        return torch.bfloat16 if self.dtype == torch.bfloat16 else self.dtype

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int,
               attention_weights: torch.Tensor | None = None):
        """Append K/V and optionally update importance scores.

        Args:
            k, v: [B, n_kv, T, head_dim]
            position: logical position
            attention_weights: [B, n_kv, T, current_cache_size] from last
                              attention computation (for importance scoring)
        """
        B, _, T, _ = k.shape

        if self.k_cache is None:
            self.k_cache = k.clone()
            self.v_cache = v.clone()
            self.seq_len = T
            self.attention_scores = torch.zeros(B, self.n_kv, T,
                                                device=self.device, dtype=self.bf16_or_dtype())
            return

        # Update attention scores from observation window
        if attention_weights is not None:
            # attention_weights: [B, n_kv, T, cache_size]
            # Accumulate: each cached token's importance = sum of attention it receives
            scores = attention_weights.sum(dim=2)  # [B, n_kv, cache_size]
            if self.attention_scores is not None:
                # Pad scores to match current cache
                cs = scores.shape[-1]
                as_ = self.attention_scores.shape[-1]
                if cs == as_:
                    self.attention_scores = self.attention_scores + scores
                elif cs > as_:
                    self.attention_scores = torch.cat([
                        self.attention_scores + scores[:, :, :as_],
                        scores[:, :, as_:]
                    ], dim=-1)
                else:
                    self.attention_scores = self.attention_scores[:, :, :cs] + scores

        # Append new tokens
        self.k_cache = torch.cat([self.k_cache, k], dim=2)
        self.v_cache = torch.cat([self.v_cache, v], dim=2)
        new_scores = torch.zeros(B, self.n_kv, T,
                                 device=self.device, dtype=self.bf16_or_dtype())
        self.attention_scores = torch.cat([self.attention_scores, new_scores], dim=-1)
        self.seq_len = self.k_cache.shape[2]

        # Evict if over capacity
        if self.seq_len > self.max_capacity:
            self._evict()

    def _evict(self):
        """Evict lowest-importance tokens, keeping observation window."""
        total = self.k_cache.shape[2]
        n_to_evict = total - self.max_capacity

        # Observation window = last obs_window tokens (protected)
        obs_start = total - self.obs_window

        # Score the non-observation tokens
        candidate_scores = self.attention_scores[:, :, :obs_start].mean(dim=1)  # [B, obs_start]
        # Average across batch
        candidate_scores = candidate_scores.mean(dim=0)  # [obs_start]

        # Find lowest-scoring tokens to evict
        _, indices = torch.sort(candidate_scores)
        evict_indices = indices[:n_to_evict].sort()[0]  # sorted for indexing

        # Keep mask: True for retained tokens
        keep = torch.ones(total, dtype=torch.bool, device=self.device)
        keep[evict_indices] = False

        # Apply eviction
        self.k_cache = self.k_cache[:, :, keep]
        self.v_cache = self.v_cache[:, :, keep]
        self.attention_scores = self.attention_scores[:, :, keep]
        self.seq_len = self.k_cache.shape[2]

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current KV cache."""
        return self.k_cache, self.v_cache

    def get_past_kv(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.k_cache is None or self.seq_len == 0:
            return None
        return (self.k_cache, self.v_cache)

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None
        self.seq_len = 0

    def info(self) -> dict:
        current_size = self.k_cache.shape[2] if self.k_cache is not None else 0
        return {
            "type": "snapkv",
            "observation_window": self.obs_window,
            "budget": self.budget,
            "max_capacity": self.max_capacity,
            "current_size": current_size,
            "seq_len": self.seq_len,
            "compression": max(1.0, self.seq_len / max(1, current_size)),
        }
