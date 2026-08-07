"""Streaming attention sinks + sliding window.

Inspired by StreamingLLM (Xiao et al., 2024): retain initial "sink" tokens
(first few positions) + a sliding window of recent tokens. This enables
infinite-length generation without recompute, with minimal quality loss.

The key insight: attention scores concentrate on initial tokens (attention
sinks) and recent tokens. Middle tokens get little attention and can be
safely evicted.

This is a pure runtime change — no weight modification, no training.

Usage:
    from research.inference.streaming_llm import StreamingKVCache
    cache = StreamingKVCache(n_sinks=4, window_size=512)
"""
import torch
from typing import Optional, Tuple, Dict


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

        # Position tracking: which logical positions are in the cache
        self.positions = []  # list of logical positions
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

        for i in range(T):
            pos = position + i
            self.seq_len = max(self.seq_len, pos + 1)

            if pos < self.n_sinks:
                # Sink token — store at its position index
                idx = pos
                self.k_cache[:, :, idx] = k[:, :, i]
                self.v_cache[:, :, idx] = v[:, :, i]
                if pos not in self.positions:
                    self.positions.append(pos)
            else:
                # Non-sink token — store in window region
                # Window starts at n_sinks, cycles through
                window_idx = (pos - self.n_sinks) % self.window_size
                idx = self.n_sinks + window_idx
                self.k_cache[:, :, idx] = k[:, :, i]
                self.v_cache[:, :, idx] = v[:, :, i]

                # Update position tracking
                # Remove old position at this slot, add new
                old_pos = self.n_sinks + window_idx
                if len(self.positions) > old_pos:
                    self.positions[old_pos] = pos
                else:
                    self.positions.append(pos)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current KV cache and position indices.

        Returns:
            k, v: [B, n_kv, current_size, head_dim]
            positions: [current_size] logical positions for RoPE
        """
        # Determine current size
        current_size = min(self.seq_len, self.max_capacity)
        k = self.k_cache[:, :, :current_size]
        v = self.v_cache[:, :, :current_size]
        pos = torch.tensor(self.positions[:current_size], device=self.device)
        return k, v, pos

    def get_past_kv(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get KV as (k, v) tuple for model's past_key_value format."""
        if self.k_cache is None or self.seq_len == 0:
            return None
        current_size = min(self.seq_len, self.max_capacity)
        return (self.k_cache[:, :, :current_size],
                self.v_cache[:, :, :current_size])

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.positions = []
        self.seq_len = 0

    def info(self) -> Dict:
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
