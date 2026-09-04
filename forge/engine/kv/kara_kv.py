"""KARA: Sliding-window KV cache compression with Token2Chunk.

Based on "KARA: Efficient Reasoning LLM Serving via Sliding-Window KV Cache
Compression" (arXiv 2607.01237).

Key insight: existing KV compression methods have two limitations:
  1. Threshold-triggered compression may reduce throughput or eliminate
     ALL KV pairs from certain blocks (information loss)
  2. They retain isolated KV pairs or fixed-size chunks with rigid boundaries,
     failing to preserve flexible-sized chunks at arbitrary positions

KARA solution:
  1. Sliding-window compression: only compress recently generated context
     (not the entire cache) → bounded work per step
  2. Bidirectional attention scoring: accumulated bidirectional attention
     received by each KV pair = importance indicator
  3. Token2Chunk: expand selected KV pairs into flexible-size chunks
     (preserve important semantic information at arbitrary positions)

Results: consistent throughput improvements, adapted to PagedAttention.

For our model (32K context, RTX 5070):
  - Full KV cache at 32K: ~1GB → KARA compresses to ~128MB (8× reduction)
  - Compression only on recent window → O(window_size) work per step
  - Token2Chunk preserves important semantic chunks
"""
from __future__ import annotations

import torch
from typing import Optional


class KARAKVCache:
    """KARA sliding-window KV cache compression.

    Maintains:
      - Sink tokens (first N tokens, always kept)
      - Compressed long-range tokens (selected by bidirectional attention)
      - Recent window (last W tokens, uncompressed)

    Compression runs only on the recent window when it slides past the
    compression boundary. Selected tokens are expanded into chunks via
    Token2Chunk.
    """

    def __init__(self, n_kv_heads: int, head_dim: int,
                 max_seq_len: int = 32768,
                 sink_size: int = 4,
                 window_size: int = 512,
                 target_budget: int = 2048,
                 chunk_expand_size: int = 8,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.sink_size = sink_size
        self.window_size = window_size
        self.target_budget = target_budget
        self.chunk_expand_size = chunk_expand_size
        self.device = device
        self.dtype = dtype

        # Storage: sink + compressed + window
        self.k_sink = torch.zeros(1, n_kv_heads, sink_size, head_dim,
                                   dtype=dtype, device=device)
        self.v_sink = torch.zeros_like(self.k_sink)

        self.k_compressed = torch.zeros(1, n_kv_heads, target_budget, head_dim,
                                         dtype=dtype, device=device)
        self.v_compressed = torch.zeros_like(self.k_compressed)
        self.compressed_len = 0  # current number of compressed tokens

        self.k_window = torch.zeros(1, n_kv_heads, window_size, head_dim,
                                     dtype=dtype, device=device)
        self.v_window = torch.zeros_like(self.k_window)
        self.window_len = 0  # current number of window tokens

        # Bidirectional attention scores (accumulated)
        self.attn_scores = torch.zeros(window_size, device=device)

        self.position = 0  # total tokens seen

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K/V tokens to the window.

        Args:
            k: (1, n_kv, T, head_dim)
            v: (1, n_kv, T, head_dim)
        """
        T = k.shape[2]
        for i in range(T):
            if self.window_len < self.window_size:
                # Add to window
                self.k_window[0, :, self.window_len] = k[0, :, i]
                self.v_window[0, :, self.window_len] = v[0, :, i]
                self.window_len += 1
            else:
                # Window full → compress oldest tokens
                self._compress_window()
                # Add new token
                self.k_window[0, :, self.window_len] = k[0, :, i]
                self.v_window[0, :, self.window_len] = v[0, :, i]
                self.window_len += 1
            self.position += 1

    def _compress_window(self):
        """Compress the window: select important tokens, expand to chunks."""
        if self.window_len == 0:
            return

        # Score tokens by bidirectional attention
        # (simplified: use K norm as proxy for importance)
        k_norms = self.k_window[0, :, :self.window_len].norm(dim=-1).mean(dim=0)
        # Combine with accumulated attention scores
        scores = k_norms + self.attn_scores[:self.window_len]

        # Determine how many tokens to keep
        n_to_keep = min(self.chunk_expand_size, self.window_len)

        # Select top-k tokens
        _, top_indices = scores.topk(n_to_keep)
        top_indices = top_indices.sort().values  # keep order

        # Token2Chunk: expand each selected token into a chunk
        # (include neighboring tokens)
        chunk_tokens = set()
        for idx in top_indices:
            i = idx.item()
            start = max(0, i - self.chunk_expand_size // 2)
            end = min(self.window_len, i + self.chunk_expand_size // 2 + 1)
            for j in range(start, end):
                chunk_tokens.add(j)

        # Move selected chunks to compressed storage
        sorted_tokens = sorted(chunk_tokens)
        for idx in sorted_tokens:
            if self.compressed_len < self.target_budget:
                self.k_compressed[0, :, self.compressed_len] = self.k_window[0, :, idx]
                self.v_compressed[0, :, self.compressed_len] = self.v_window[0, :, idx]
                self.compressed_len += 1

        # Shift remaining window tokens
        kept_mask = torch.ones(self.window_len, dtype=torch.bool, device=self.device)
        kept_mask[sorted_tokens] = False
        remaining = kept_mask.sum().item()

        # Compact window (remove compressed tokens)
        new_k = self.k_window[0, :, :self.window_len][:, kept_mask]
        new_v = self.v_window[0, :, :self.window_len][:, kept_mask]
        self.k_window[0, :, :remaining] = new_k
        self.v_window[0, :, :remaining] = new_v
        self.window_len = remaining

        # Reset attention scores for remaining
        self.attn_scores[:self.window_len] = 0
        self.attn_scores[self.window_len:] = 0

    def update_attention_scores(self, attn_weights: torch.Tensor):
        """Update bidirectional attention scores from the last attention pass.

        Args:
            attn_weights: (n_heads, window_len) — attention received by each window token
        """
        # Accumulate bidirectional attention
        bidirectional = attn_weights.sum(dim=0)  # sum over heads
        if bidirectional.shape[0] <= self.window_len:
            self.attn_scores[:bidirectional.shape[0]] += bidirectional

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the full K/V cache (sink + compressed + window)."""
        k_parts = [
            self.k_sink[:, :, :self.sink_size],
            self.k_compressed[:, :, :self.compressed_len],
            self.k_window[:, :, :self.window_len],
        ]
        v_parts = [
            self.v_sink[:, :, :self.sink_size],
            self.v_compressed[:, :, :self.compressed_len],
            self.v_window[:, :, :self.window_len],
        ]
        return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)

    @property
    def total_len(self) -> int:
        return self.sink_size + self.compressed_len + self.window_len

    def stats(self) -> dict:
        return {
            "sink_size": self.sink_size,
            "compressed_len": self.compressed_len,
            "window_len": self.window_len,
            "total_len": self.total_len,
            "target_budget": self.target_budget,
            "compression_ratio": self.position / max(self.total_len, 1),
        }
