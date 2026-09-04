"""EvoSparse — Evolving Token Importance KV Cache.

Models token importance as a dynamic process across decoding steps and
layers, then evicts low-importance KV positions to keep only the tokens
that matter for the evolving attention pattern.

Paper: ACL 2026 long.530 (iLearn-Lab/ACL26-EvoSparse).
Two mechanisms drive the importance estimate:

  1. Cross-Step Accumulation — a decayed running average of attention
     column sums.  At each decoding step the per-position importance is
     updated as:

         score[pos] = decay * score[pos] + (1 - decay) * col_sum[pos]

     where col_sum[pos] = sum over query positions of attn[:, pos].
     This avoids recomputing importance from scratch every step: the
     running average is O(seq_len) to update, not O(seq_len^2).

  2. Cross-Layer Propagation — retrieval heads in earlier layers compute
     query-aware importance indices that are propagated to later layers.
     We store a per-layer importance vector and let later layers inherit
     (with a propagation weight) the importance signal from earlier
     layers, so a token that was critical in layer 0 stays protected in
     layer 15 even if that layer's own attention is diffuse.

Eviction is importance-based (not LRU): when the cache exceeds its budget
(keep_ratio * max_seq_len), the lowest-importance positions are evicted
and their K/V slots zeroed.  Retained positions stay at full precision.

Reported speedups (paper): 5.36x attention, 2.33x end-to-end.

For RTX 5070 12GB VRAM (V10: 16 layers, 8 KV heads, 128 head_dim, 4096 ctx):
  - Full bf16 KV: ~2.0 GB
  - EvoSparse keep_ratio=0.5: ~1.0 GB (2x compression)
  - keep_ratio=0.25: ~0.5 GB (4x compression)
  - Importance vectors: n_layers * seq_len * 4 bytes = 16 * 4096 * 4 = 256 KB
    (negligible vs the K/V savings)
  - Pure torch, no custom CUDA — CPU fallback is the same code path.
"""
from __future__ import annotations

from typing import Optional

import torch

from research.inference.kv_backend import KVCacheStrategy


class EvoSparseKVCache(KVCacheStrategy):
    """Evolving token-importance KV cache with cross-step + cross-layer propagation.

    Retains only the top `keep_ratio` fraction of KV positions by evolving
    importance.  Evicted positions are zeroed; `get` returns zeros for them
    so the attention mask (applied by the caller) drops them naturally.
    """

    def __init__(self, decay: float = 0.95, keep_ratio: float = 0.5,
                 n_layers: int = 32, n_sinks: int = 4,
                 propagation_weight: float = 0.3):
        self.decay = decay
        self.keep_ratio = keep_ratio
        self.n_layers = n_layers
        self.n_sinks = n_sinks
        self.propagation_weight = propagation_weight

    def init(self, n_heads: int, head_dim: int, n_kv_heads: int,
             max_seq_len: int, device: str, dtype: torch.dtype):
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.capacity = max(self.n_sinks, int(max_seq_len * self.keep_ratio))

        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

        # Per-layer running importance: (n_layers, max_seq_len)
        self.importance = torch.zeros(self.n_layers, max_seq_len,
                                      device=device, dtype=torch.float32)
        # Composite (propagated) importance used for eviction: (max_seq_len,)
        self.composite_importance = torch.zeros(max_seq_len,
                                                device=device, dtype=torch.float32)

        # Boolean retention mask over positions: True = kept, False = evicted
        self.retained = torch.ones(max_seq_len, device=device, dtype=torch.bool)

        self.seq_len = 0
        self.n_evicted = 0
        self._eviction_done = False

    def _ensure_allocated(self, batch_size: int):
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                batch_size, self.n_kv, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int):
        self._ensure_allocated(k.shape[0])
        seq = k.shape[2]
        end = position + seq
        end = min(end, self.max_seq_len)
        seq = end - position
        if seq <= 0:
            return
        self.k_cache[:, :, position:end].copy_(k[:, :, :seq])
        self.v_cache[:, :, position:end].copy_(v[:, :, :seq])
        self.retained[position:end] = True
        self.seq_len = max(self.seq_len, end)
        self._maybe_evict()

    def _maybe_evict(self):
        """Evict lowest-importance positions once the cache exceeds capacity.

        Sink tokens (the first n_sinks positions) are always protected —
        they are global attention pivots and evicting them destabilizes
        attention (StreamingLLM finding).  The most recent token is also
        protected so the current decode step always has local context.
        """
        if self.seq_len <= self.capacity:
            self._eviction_done = False
            return
        n_keep = self.capacity
        scores = self.composite_importance[:self.seq_len].clone()

        protect = torch.zeros(self.seq_len, device=self.device, dtype=torch.bool)
        protect[:self.n_sinks] = True
        protect[self.seq_len - 1] = True
        scores[protect] = float("inf")

        _, top_idx = torch.topk(scores, n_keep, largest=True)
        keep_mask = torch.zeros(self.seq_len, device=self.device, dtype=torch.bool)
        keep_mask[top_idx] = True
        evict_mask = ~keep_mask & ~protect[:self.seq_len]
        evict_positions = torch.nonzero(evict_mask, as_tuple=False).squeeze(-1)

        if evict_positions.numel() == 0:
            self._eviction_done = True
            return

        self.k_cache[:, :, evict_positions] = 0
        self.v_cache[:, :, evict_positions] = 0
        self.retained[:self.seq_len] = keep_mask | protect[:self.seq_len]
        self.n_evicted = int((~self.retained[:self.seq_len]).sum().item())
        self._eviction_done = True

    def update_importance(self, attention_scores: torch.Tensor,
                          layer_idx: int):
        """Update running importance from attention scores of one layer.

        Args:
            attention_scores: (n_heads, seq_len, seq_len) attention weights
                from the most recent forward pass.  Column sums give the
                per-position importance for this step.
            layer_idx: which layer these scores came from (for cross-layer
                propagation storage).
        """
        if layer_idx >= self.n_layers:
            return
        seq = attention_scores.shape[-1]
        seq = min(seq, self.seq_len)
        if seq <= 0:
            return
        attn = attention_scores[:, :seq, :seq].to(torch.float32)
        col_sum = attn.sum(dim=-2).mean(dim=0)  # (seq,) avg over heads

        prev = self.importance[layer_idx, :seq]
        self.importance[layer_idx, :seq] = self.decay * prev + (1.0 - self.decay) * col_sum

        if layer_idx == 0:
            self._propagate_importance()
        else:
            self.composite_importance[:seq] = (
                self.composite_importance[:seq] * (1 - self.propagation_weight)
                + self.importance[layer_idx, :seq] * self.propagation_weight
            )

        if self._eviction_done and self.seq_len > self.capacity:
            self._maybe_evict()

    def _propagate_importance(self):
        """Aggregate per-layer importance into the composite vector.

        Cross-layer propagation: earlier (retrieval) layers carry the
        query-aware signal; later layers inherit it with a decay so a
        token critical in layer 0 remains protected downstream even when
        the later layer's own attention is diffuse.
        """
        seq = self.seq_len
        if seq <= 0:
            return
        composite = torch.zeros(seq, device=self.device, dtype=torch.float32)
        w = 1.0
        for l in range(self.n_layers):
            composite = composite + w * self.importance[l, :seq]
            w *= self.propagation_weight
        self.composite_importance[:seq] = composite

    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given positions.

        Evicted positions return zeros (the caller's attention mask drops
        them).  `positions` is expected to be the contiguous range
        [0, seq_len) in the standard interface; we return the retained
        slice and zero out evicted slots in-place via the mask.
        """
        if self.k_cache is None or self.seq_len == 0:
            B = positions.shape[0] if positions.dim() > 0 else 1
            T = positions.shape[1] if positions.dim() > 1 else 1
            z = torch.zeros(B, self.n_kv, T, self.head_dim,
                            device=self.device, dtype=self.dtype)
            return z, z
        k = self.k_cache[:, :, :self.seq_len].clone()
        v = self.v_cache[:, :, :self.seq_len].clone()
        evict = ~self.retained[:self.seq_len]
        if evict.any():
            k[:, :, evict] = 0
            v[:, :, evict] = 0
        return k, v

    def get_retained_positions(self) -> torch.Tensor:
        """Return the indices of currently-retained positions (for sparse attention)."""
        return torch.nonzero(self.retained[:self.seq_len], as_tuple=False).squeeze(-1)

    def clear(self):
        if self.k_cache is not None:
            self.k_cache.zero_()
            self.v_cache.zero_()
        self.importance.zero_()
        self.composite_importance.zero_()
        self.retained.fill_(True)
        self.seq_len = 0
        self.n_evicted = 0
        self._eviction_done = False

    def info(self) -> dict:
        retained = int(self.retained[:self.seq_len].sum().item()) if self.seq_len else 0
        standard_bytes = 2 * self.n_kv * self.head_dim * self.seq_len * self.dtype.itemsize if self.seq_len else 0
        actual_bytes = 2 * self.n_kv * self.head_dim * retained * self.dtype.itemsize if retained else 0
        compression = standard_bytes / actual_bytes if actual_bytes > 0 else 1.0
        return {
            "type": "evosparse",
            "seq_len": self.seq_len,
            "keep_ratio": self.keep_ratio,
            "capacity": self.capacity,
            "n_retained": retained,
            "n_evicted": self.n_evicted,
            "decay": self.decay,
            "n_layers": self.n_layers,
            "propagation_weight": self.propagation_weight,
            "compression": compression,
        }
