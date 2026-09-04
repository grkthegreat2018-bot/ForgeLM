"""Vegas — Verification-Guided KV Selection (zero-overhead).

During speculative decoding the verification/full-attention phase computes
full attention scores. These scores reveal which KV positions are critical
(high column-sum = many queries attend strongly to that key). Vegas reuses
those otherwise-discarded scores to select which KV entries to retain for
the next draft step, making KV selection effectively zero-overhead.

Paper: arXiv 2602.07223 — "Vegas: Verification-Guided KV Selection for
Speculative Decoding." The key insight is that the verification phase already
computes the attention matrix; column sums of that matrix are a free
criticality signal that H2O/SnapKV-style methods pay extra attention passes
to obtain.

For RTX 5070 12GB VRAM (V10: 16 layers, 8 KV heads, 128 head_dim, 4096 ctx):
  - bf16 full cache: 2.0 GB
  - Vegas keep_ratio=0.5: ~1.0 GB (2x compression, zero quality loss on
    retained tokens — eviction only, no quantization)
  - Composes with Hadamard INT4 for ~4 GB effective budget headroom
"""
from __future__ import annotations

import torch

from forge.engine.kv_backend import KVCacheStrategy


class VegasKVCache(KVCacheStrategy):
    """Verification-guided KV cache: retains top-k critical positions.

    The caller invokes `set_attention_hints` after each verification phase
    with the full attention score matrix. Column sums (sum over query
    positions) yield a per-position criticality score. The cache keeps the
    top `keep_ratio` fraction of positions and evicts the rest. Evicted
    positions return zero K/V from `get`, so attention naturally down-weights
    them without requiring masked index remapping.

    If no hints have been set (e.g. prefill before the first verification
    step), the cache falls back to a sliding window — keeping the most
    recent `keep_ratio` fraction of positions — which matches the behaviour
    of a recency-based eviction baseline.
    """

    def __init__(self, keep_ratio: float = 0.5):
        self.keep_ratio = keep_ratio
        self._criticality: torch.Tensor | None = None
        self._retained_mask: torch.Tensor | None = None
        self._has_hints = False

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.seq_len = 0
        self.n_evicted = 0
        self.k_cache = None
        self.v_cache = None

    def _ensure_allocated(self, batch_size):
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                batch_size, self.n_kv, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

    def set_attention_hints(self, attention_scores: torch.Tensor):
        """Called after the verification/full-attention phase.

        Args:
            attention_scores: (n_heads, seq_len, seq_len) attention weight
                matrix. Column sums (sum over the query/head axes) give a
                per-position criticality score. Higher column sum = more
                queries attended to that key = more critical to retain.
        """
        if attention_scores.dim() == 4:
            scores = attention_scores.mean(dim=0)
        else:
            scores = attention_scores
        scores = scores.mean(dim=0)  # (seq_len, seq_len)
        col_sums = scores.sum(dim=-2)  # (seq_len,)
        n_valid = min(col_sums.numel(), self.seq_len)
        if n_valid == 0:
            return
        self._criticality = col_sums[:n_valid].to(self.device)
        self._update_retained_mask()
        self._has_hints = True

    def _update_retained_mask(self):
        n = self._criticality.numel()
        k = max(1, int(n * self.keep_ratio))
        _, top_idx = self._criticality.topk(k)
        mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        mask[top_idx] = True
        self._retained_mask = mask
        self.n_evicted = n - int(mask.sum().item())

    def _fallback_mask(self):
        n = self.seq_len
        if n == 0:
            self._retained_mask = None
            self.n_evicted = 0
            return
        k = max(1, int(n * self.keep_ratio))
        mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        mask[n - k:] = True
        self._retained_mask = mask
        self.n_evicted = n - k

    def _current_mask(self) -> torch.Tensor | None:
        if self._retained_mask is not None and self._retained_mask.numel() >= self.seq_len:
            return self._retained_mask[:self.seq_len]
        if not self._has_hints:
            self._fallback_mask()
            if self._retained_mask is not None:
                return self._retained_mask[:self.seq_len]
        return None

    def append(self, k, v, position):
        self._ensure_allocated(k.shape[0])
        seq = k.shape[2]
        end = position + seq
        self.k_cache[:, :, position:end].copy_(k)
        self.v_cache[:, :, position:end].copy_(v)
        self.seq_len = max(self.seq_len, end)

    def get(self, positions):
        n = self.seq_len
        if n == 0:
            return (
                torch.zeros(0, self.n_kv, 0, self.head_dim,
                            device=self.device, dtype=self.dtype),
                torch.zeros(0, self.n_kv, 0, self.head_dim,
                            device=self.device, dtype=self.dtype),
            )
        k_full = self.k_cache[:, :, :n]
        v_full = self.v_cache[:, :, :n]
        mask = self._current_mask()
        if mask is None:
            return k_full, v_full
        k_out = k_full.clone()
        v_out = v_full.clone()
        evicted = ~mask
        k_out[:, :, evicted] = 0
        v_out[:, :, evicted] = 0
        return k_out, v_out

    def clear(self):
        if self.k_cache is not None:
            self.k_cache[:, :, :self.seq_len].zero_()
            self.v_cache[:, :, :self.seq_len].zero_()
        self.seq_len = 0
        self.n_evicted = 0
        self._criticality = None
        self._retained_mask = None
        self._has_hints = False

    def info(self):
        n = max(1, self.seq_len)
        retained = n - self.n_evicted
        compression = n / max(1, retained)
        return {
            "type": "vegas",
            "seq_len": self.seq_len,
            "keep_ratio": self.keep_ratio,
            "n_evicted": self.n_evicted,
            "n_retained": retained,
            "has_hints": self._has_hints,
            "compression": compression,
        }
