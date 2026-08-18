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

    def _ensure_buffer(self, B: int, T: int, dtype: torch.dtype):
        """Ensure pre-allocated buffer is large enough for B batch, T new tokens."""
        needed = max(self.max_capacity + T, self.seq_len + T)
        if self.k_cache is None:
            self.k_cache = torch.zeros(B, self.n_kv, needed, self.head_dim,
                                       device=self.device, dtype=dtype)
            self.v_cache = torch.zeros_like(self.k_cache)
            self.attention_scores = torch.zeros(B, self.n_kv, needed,
                                                device=self.device, dtype=self.bf16_or_dtype())
        elif self.k_cache.shape[2] < needed:
            new_size = max(needed, self.k_cache.shape[2] * 2)
            new_k = torch.zeros(B, self.n_kv, new_size, self.head_dim,
                                device=self.device, dtype=self.k_cache.dtype)
            new_v = torch.zeros_like(new_k)
            new_scores = torch.zeros(B, self.n_kv, new_size,
                                     device=self.device, dtype=self.attention_scores.dtype)
            new_k[:, :, :self.seq_len] = self.k_cache[:, :, :self.seq_len]
            new_v[:, :, :self.seq_len] = self.v_cache[:, :, :self.seq_len]
            new_scores[:, :, :self.seq_len] = self.attention_scores[:, :, :self.seq_len]
            self.k_cache = new_k
            self.v_cache = new_v
            self.attention_scores = new_scores

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
        self._ensure_buffer(B, T, k.dtype)

        # Update attention scores from observation window
        if attention_weights is not None and self.seq_len > 0:
            # attention_weights: [B, n_kv, T, cache_size]
            # Accumulate: each cached token's importance = sum of attention it receives
            scores = attention_weights.sum(dim=2)  # [B, n_kv, cache_size]
            cs = scores.shape[-1]
            if cs <= self.seq_len:
                self.attention_scores[:, :, :cs] += scores
            else:
                self.attention_scores[:, :, :self.seq_len] += scores[:, :, :self.seq_len]

        # Write new tokens to buffer at current position (no torch.cat)
        end = self.seq_len + T
        self.k_cache[:, :, self.seq_len:end].copy_(k)
        self.v_cache[:, :, self.seq_len:end].copy_(v)
        self.attention_scores[:, :, self.seq_len:end].zero_()
        self.seq_len = end

        # Evict if over capacity
        if self.seq_len > self.max_capacity:
            self._evict()

    def _evict(self):
        """Evict lowest-importance tokens, keeping observation window."""
        total = self.seq_len
        n_to_evict = total - self.max_capacity

        # Observation window = last obs_window tokens (protected)
        obs_start = total - self.obs_window

        # Score the non-observation tokens
        candidate_scores = self.attention_scores[:, :, :obs_start].mean(dim=1)  # [B, obs_start]
        # Average across batch
        candidate_scores = candidate_scores.mean(dim=0)  # [obs_start]

        # Use topk to find highest-scoring tokens to keep (O(n log k) vs O(n log n) sort)
        n_keep = obs_start - n_to_evict
        _, keep_indices = torch.topk(candidate_scores, n_keep)
        keep_indices = keep_indices.sort()[0]

        # Keep mask: True for retained tokens
        keep = torch.zeros(total, dtype=torch.bool, device=self.device)
        keep[keep_indices] = True
        keep[obs_start:] = True  # always keep observation window

        # Compact buffer: copy kept entries to front (no reallocation)
        new_seq_len = keep.sum().item()
        self.k_cache[:, :, :new_seq_len] = self.k_cache[:, :, keep]
        self.v_cache[:, :, :new_seq_len] = self.v_cache[:, :, keep]
        self.attention_scores[:, :, :new_seq_len] = self.attention_scores[:, :, keep]
        self.seq_len = new_seq_len

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current KV cache."""
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]

    def get_past_kv(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.k_cache is None or self.seq_len == 0:
            return None
        return (self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len])

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None
        self.seq_len = 0

    def info(self) -> dict:
        current_size = self.seq_len
        return {
            "type": "snapkv",
            "observation_window": self.obs_window,
            "budget": self.budget,
            "max_capacity": self.max_capacity,
            "current_size": current_size,
            "seq_len": self.seq_len,
            "compression": max(1.0, self.seq_len / max(1, current_size)),
        }
