"""Streaming attention sinks + sliding window.

Inspired by StreamingLLM (Xiao et al., 2024): retain initial "sink" tokens
(first few positions) + a sliding window of recent tokens. This enables
infinite-length generation without recompute, with minimal quality loss.

The key insight: attention scores concentrate on initial tokens (attention
sinks) and recent tokens. Middle tokens get little attention and can be
safely evicted.

This is a pure runtime change — no weight modification, no training.

Usage:
    from research.inference.kv.streaming_llm import StreamingKVCache
    cache = StreamingKVCache(n_sinks=4, window_size=512)
"""
from typing import Dict, Optional, Tuple

import torch


class StreamingKVCache:
    """KV cache with attention sinks + sliding window.

    Structure:
    - Sink tokens: positions 0..n_sinks-1 (always retained)
    - Window tokens: most recent `window_size` tokens
    - Middle tokens: evicted

    Position tracking: we maintain a position map that maps logical
    positions to physical cache indices, so RoPE still works correctly.
    """

    def __init__(self, n_sinks: int = 4, window_size: int = 512,
                 n_kv_heads: int = 2, head_dim: int = 128,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.n_sinks = n_sinks
        self.window_size = window_size
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Storage: [B, n_kv, max_capacity, head_dim]
        self.max_capacity = n_sinks + window_size
        self.k_cache = None
        self.v_cache = None

        # Pre-allocated positions tensor (updated in-place, no per-get() alloc).
        # Maps physical cache index → logical position (for RoPE).
        self._pos_tensor = torch.zeros(self.max_capacity, dtype=torch.long,
                                       device=self.device)
        self.seq_len = 0  # total logical positions seen

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """Append K/V for one or more positions.

        Args:
            k, v: [B, n_kv, T, head_dim]
            position: starting logical position
        """
        B, _, T, _ = k.shape

        if self.k_cache is None:
            self.k_cache = torch.zeros(B, self.n_kv, self.max_capacity,
                                       self.head_dim, device=self.device,
                                       dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

        # Vectorized index computation (replaces per-token Python loop).
        offsets = torch.arange(T, device=self.device)
        positions = position + offsets  # [T] logical positions
        self.seq_len = max(self.seq_len, position + T)

        # Physical cache indices: sinks stay at their position, window tokens
        # cycle through the window region.
        sink_mask = positions < self.n_sinks
        window_idx = (positions - self.n_sinks) % self.window_size
        indices = torch.where(sink_mask, positions,
                              self.n_sinks + window_idx)  # [T]

        # Write K/V at computed indices (vectorized, no Python loop).
        self.k_cache[:, :, indices] = k
        self.v_cache[:, :, indices] = v

        # Update positions tensor in-place (no per-get() allocation).
        self._pos_tensor[indices] = positions

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current KV cache and position indices.

        Returns:
            k, v: [B, n_kv, current_size, head_dim]
            positions: [current_size] logical positions for RoPE
        """
        # Determine current size
        current_size = min(self.seq_len, self.max_capacity)
        k = self.k_cache[:, :, :current_size]
        v = self.v_cache[:, :, :current_size]
        # Return a view of the pre-allocated positions tensor (no allocation).
        pos = self._pos_tensor[:current_size]
        return k, v, pos

    def get_past_kv(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get KV as (k, v) tuple for model's past_key_value format."""
        if self.k_cache is None or self.seq_len == 0:
            return None
        current_size = min(self.seq_len, self.max_capacity)
        return (self.k_cache[:, :, :current_size],
                self.v_cache[:, :, :current_size])

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self._pos_tensor.zero_()
        self.seq_len = 0

    def info(self) -> dict:
        current_size = min(self.seq_len, self.max_capacity)
        return {
            "type": "streaming_llm",
            "n_sinks": self.n_sinks,
            "window_size": self.window_size,
            "max_capacity": self.max_capacity,
            "current_size": current_size,
            "seq_len": self.seq_len,
            "compression": max(1.0, self.seq_len / max(1, current_size)),
        }
