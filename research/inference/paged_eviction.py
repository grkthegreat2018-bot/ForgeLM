"""PagedEviction: structured block-wise KV cache pruning for PagedAttention.

Based on "PagedEviction: Structured Block-wise KV Cache Pruning" (arXiv 2509.04377).

Key insight: existing token-level eviction methods (H2O, SnapKV) evict individual
tokens, which causes memory fragmentation in PagedAttention (which allocates
fixed-size blocks). PagedEviction evicts ENTIRE blocks, maintaining the paged
structure and avoiding fragmentation.

Benefits:
  - 3020 tok/s on LLaMA-1B (37% over full cache, 39% over Inverse Key L2-Norm)
  - 10-12% latency reduction across 1B/3B/8B models
  - Block-level evictions avoid frequent cache updates → scalable under batching
  - Compatible with FlashAttention (doesn't need attention scores)

Eviction policy: block importance = sum of L2 norms of K vectors in the block.
Low-norm blocks contribute less to attention → evicted first. This is
attention-score-free (works with FlashAttention which never returns scores).

Also supports observation-window-based eviction (SnapKV-style) when attention
scores ARE available: blocks outside the observation window with low cumulative
attention are evicted.
"""
from __future__ import annotations

import torch

from research.inference.kv_backend import KVCacheStrategy


class PagedEvictionKVCache(KVCacheStrategy):
    """Block-wise KV cache eviction compatible with paged attention.

    Divides the KV cache into fixed-size blocks (default 16 tokens, matching
    PagedAttention). When the cache exceeds the budget, evicts entire blocks
    with the lowest importance score.

    Importance score (no attention scores needed):
      block_score = mean(||K_block||_2)  (L2 norm of keys in the block)
    Higher-norm keys participate more in attention → keep them.

    With attention scores (optional, for SnapKV-style scoring):
      block_score = mean(attention_weights[block_tokens])
    """
    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Block structure
        self.block_size = 16  # standard PagedAttention block size
        self.budget = min(max_seq_len, 2048)  # max tokens to keep
        self.obs_window = 128  # observation window for scoring

        # Pre-allocate full KV cache
        self.k_cache = torch.zeros(
            1, n_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device)
        self.v_cache = torch.zeros(
            1, n_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device)

        # Block metadata
        self.n_blocks = max_seq_len // self.block_size
        self.block_valid = torch.zeros(self.n_blocks, dtype=torch.bool, device=device)
        self.block_scores = torch.zeros(self.n_blocks, dtype=torch.float32, device=device)

        self.seq_len = 0  # total tokens written
        self.active_len = 0  # tokens after eviction

    def append(self, k, v, position, attention_weights=None):
        """Append K/V tokens and potentially evict blocks.

        Args:
            k: (1, n_kv, T, head_dim) new keys
            v: (1, n_kv, T, head_dim) new values
            position: current position in the cache
            attention_weights: optional (1, n_heads, T, S) for SnapKV-style scoring
        """
        T = k.shape[2]
        pos = position
        self.k_cache[:, :, pos:pos + T] = k
        self.v_cache[:, :, pos:pos + T] = v
        self.seq_len = pos + T

        # Mark blocks as valid
        start_block = pos // self.block_size
        end_block = (pos + T - 1) // self.block_size
        self.block_valid[start_block:end_block + 1] = True

        # Update block scores
        self._update_block_scores(start_block, end_block, attention_weights)

        # Evict if over budget
        if self.seq_len > self.budget:
            self._evict_blocks()

        self.active_len = self.seq_len - self._evicted_token_count()

    def _update_block_scores(self, start_blk, end_blk, attention_weights=None):
        """Compute importance scores for blocks in [start_blk, end_blk]."""
        for blk in range(start_blk, end_blk + 1):
            s = blk * self.block_size
            e = min(s + self.block_size, self.seq_len)
            if e <= s:
                continue
            k_blk = self.k_cache[:, :, s:e]  # (1, n_kv, blk_size, hd)
            if attention_weights is not None:
                # SnapKV-style: use attention weights from observation window
                # Sum attention received by tokens in this block
                aw = attention_weights[:, :, -self.obs_window:, s:e]
                self.block_scores[blk] = aw.mean().item()
            else:
                # L2-norm-based scoring (no attention scores needed)
                self.block_scores[blk] = k_blk.float().norm(dim=-1).mean().item()

    def _evict_blocks(self):
        """Evict lowest-scoring blocks to stay within budget."""
        n_valid = self.block_valid.sum().item()
        n_to_evict = (self.seq_len - self.budget) // self.block_size + 1
        if n_to_evict <= 0:
            return

        # Don't evict the observation window (last obs_window tokens)
        obs_start_blk = max(0, (self.seq_len - self.obs_window) // self.block_size)

        # Get valid blocks outside observation window
        candidates = self.block_valid.clone()
        candidates[obs_start_blk:] = False

        if candidates.sum() == 0:
            return

        # Get scores of candidate blocks
        candidate_indices = candidates.nonzero(as_tuple=True)[0]
        candidate_scores = self.block_scores[candidate_indices]

        # Evict lowest-scoring blocks
        n_evict = min(n_to_evict, len(candidate_indices))
        _, lowest_idx = candidate_scores.topk(n_evict, largest=False)
        evict_blocks = candidate_indices[lowest_idx]

        for blk in evict_blocks:
            self.block_valid[blk] = False
            s = blk * self.block_size
            e = s + self.block_size
            # Zero out evicted block
            self.k_cache[:, :, s:e] = 0
            self.v_cache[:, :, s:e] = 0

    def _evicted_token_count(self):
        return (~self.block_valid).sum().item() * self.block_size

    def get(self, positions=None):
        """Return active K/V (compacted to remove evicted blocks).

        For simplicity, returns the full cache with evicted blocks zeroed.
        The attention mask handles the zeroed positions.
        """
        if positions is not None:
            return (self.k_cache[:, :, positions], self.v_cache[:, :, positions])

        # Build a compacted view: gather only valid blocks
        valid_blocks = self.block_valid.nonzero(as_tuple=True)[0]
        if len(valid_blocks) == 0:
            return (self.k_cache[:, :, :1], self.v_cache[:, :, :1])

        # Gather valid blocks into contiguous tensor
        k_parts = []
        v_parts = []
        for blk in valid_blocks:
            s = blk * self.block_size
            e = min(s + self.block_size, self.seq_len)
            k_parts.append(self.k_cache[:, :, s:e])
            v_parts.append(self.v_cache[:, :, s:e])

        return (torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2))

    def get_block_mask(self) -> torch.Tensor:
        """Return a boolean mask of valid token positions (for attention)."""
        mask = torch.zeros(self.seq_len, dtype=torch.bool, device=self.device)
        for blk in range(self.n_blocks):
            if self.block_valid[blk]:
                s = blk * self.block_size
                e = min(s + self.block_size, self.seq_len)
                mask[s:e] = True
        return mask

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.block_valid.zero_()
        self.block_scores.zero_()
        self.seq_len = 0
        self.active_len = 0

    def info(self):
        evicted = self._evicted_token_count()
        return {
            "type": "paged_eviction",
            "seq_len": self.seq_len,
            "active_len": self.active_len,
            "evicted_tokens": evicted,
            "budget": self.budget,
            "block_size": self.block_size,
            "compression": self.seq_len / max(self.active_len, 1),
        }
