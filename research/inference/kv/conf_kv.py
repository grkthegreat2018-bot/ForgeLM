"""CONF-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage.

Based on "CONF-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision
Storage for Long-Horizon LLM" (arXiv 2605.24786).

Key insight: the model's current uncertainty (next-token distribution entropy)
is a signal computed on EVERY decoding step but unused by existing KV eviction
methods. CONF-KV converts this into a scalar confidence score and uses it to
choose the per-step cache budget:
  - High confidence → prune aggressively (model is sure, needs less context)
  - Low confidence → retain more context (model is uncertain, needs all info)

Within each budget, tokens are ranked by composite of:
  - Accumulated attention mass
  - Recency
  - Protected recent window (local coherence)

Combined with:
  - Blockwise online-softmax attention
  - Mixed FP16/INT8 storage (important tokens FP16, rest INT8)
  - Pyramidal per-layer budget (deeper layers get more budget)

Results: near footprint of 512-token sliding window, within 1.5-2.1 perplexity
of full KV. 91.4% retrieval on 32K Needle-in-a-Haystack (vs 53.8% sliding window).

For our model:
  - Dynamic budget: 256-4096 tokens depending on confidence
  - Mixed precision: important tokens bf16, rest INT8
  - Pyramidal: layers 0-5 get 512, 6-10 get 1024, 11-15 get 2048
"""
from __future__ import annotations

import math
import torch
from typing import Optional


class ConfidenceScorer:
    """Computes model confidence from next-token distribution.

    Confidence = 1 - normalized_entropy(logits)
    High confidence → model is sure → can prune aggressively
    Low confidence → model is uncertain → retain more context
    """

    def __init__(self, min_budget: int = 256, max_budget: int = 4096,
                 temperature: float = 1.0):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.temperature = temperature
        self._confidence_history: list[float] = []

    def compute_confidence(self, logits: torch.Tensor) -> float:
        """Compute confidence from logits.

        Args:
            logits: (V,) or (B, V) next-token logits

        Returns:
            confidence: 0 (uncertain) to 1 (confident)
        """
        if logits.dim() > 1:
            logits = logits[-1]  # last token's logits

        probs = torch.softmax(logits / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = math.log(probs.shape[0])

        confidence = 1.0 - (entropy / max_entropy).item()
        self._confidence_history.append(confidence)
        return confidence

    def get_budget(self, confidence: float, layer_idx: int = 0,
                   n_layers: int = 16) -> int:
        """Get cache budget based on confidence and layer (pyramidal).

        Pyramidal: deeper layers get more budget (they need more context
        for higher-level reasoning).
        """
        # Base budget from confidence
        base = self.min_budget + (1 - confidence) * (self.max_budget - self.min_budget)

        # Pyramidal scaling: layer 0 gets 0.5×, layer N-1 gets 1.5×
        layer_factor = 0.5 + 1.0 * layer_idx / max(n_layers - 1, 1)
        budget = int(base * layer_factor)

        return max(self.min_budget, min(budget, self.max_budget * 2))


class MixedPrecisionKV:
    """Mixed FP16/INT8 KV storage.

    Important tokens stored in FP16 (full precision).
    Less important tokens stored in INT8 (half memory).
    """

    def __init__(self, n_kv_heads: int, head_dim: int,
                 max_tokens: int = 4096,
                 fp16_ratio: float = 0.3,
                 device: str = "cuda"):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.max_tokens = max_tokens
        self.fp16_ratio = fp16_ratio
        self.device = device

        # FP16 storage (for important tokens)
        n_fp16 = int(max_tokens * fp16_ratio)
        self.k_fp16 = torch.zeros(1, n_kv_heads, n_fp16, head_dim,
                                   dtype=torch.bfloat16, device=device)
        self.v_fp16 = torch.zeros_like(self.k_fp16)
        self.fp16_len = 0

        # INT8 storage (for less important tokens)
        n_int8 = max_tokens - n_fp16
        self.k_int8 = torch.zeros(1, n_kv_heads, n_int8, head_dim,
                                   dtype=torch.int8, device=device)
        self.v_int8 = torch.zeros_like(self.k_int8)
        self.k_scales = torch.ones(n_int8, device=device)
        self.v_scales = torch.ones(n_int8, device=device)
        self.int8_len = 0

        # Importance scores
        self.importance = torch.zeros(max_tokens, device=device)
        self.positions = torch.zeros(max_tokens, dtype=torch.long, device=device)

    def add(self, k: torch.Tensor, v: torch.Tensor,
            importance: float, position: int):
        """Add a token with its importance score."""
        # Decide precision: top fp16_ratio → FP16, rest → INT8
        if self.fp16_len < int(self.max_tokens * self.fp16_ratio):
            # Store in FP16
            self.k_fp16[0, :, self.fp16_len] = k[0, :, 0]
            self.v_fp16[0, :, self.fp16_len] = v[0, :, 0]
            self.importance[self.fp16_len] = importance
            self.positions[self.fp16_len] = position
            self.fp16_len += 1
        else:
            # Check if this token is more important than the least important FP16 token
            min_fp16_idx = self.importance[:self.fp16_len].argmin()
            if importance > self.importance[min_fp16_idx].item():
                # Swap: move min FP16 to INT8, add new to FP16
                old_k = self.k_fp16[0, :, min_fp16_idx].clone()
                old_v = self.v_fp16[0, :, min_fp16_idx].clone()
                old_imp = self.importance[min_fp16_idx].item()
                old_pos = self.positions[min_fp16_idx].item()

                # Add new token to FP16
                self.k_fp16[0, :, min_fp16_idx] = k[0, :, 0]
                self.v_fp16[0, :, min_fp16_idx] = v[0, :, 0]
                self.importance[min_fp16_idx] = importance
                self.positions[min_fp16_idx] = position

                # Move old to INT8
                self._add_int8(old_k, old_v, old_imp, old_pos)
            else:
                # Store in INT8
                self._add_int8(k[0, :, 0], v[0, :, 0], importance, position)

    def _add_int8(self, k: torch.Tensor, v: torch.Tensor,
                  importance: float, position: int):
        """Add token to INT8 storage."""
        if self.int8_len >= self.k_int8.shape[2]:
            # Evict least important INT8 token
            int8_start = int(self.max_tokens * self.fp16_ratio)
            min_idx = self.importance[int8_start:int8_start + self.int8_len].argmin()
            global_min_idx = int8_start + min_idx.item()

            # Replace
            k_scale = k.abs().max() / 127.0
            v_scale = v.abs().max() / 127.0
            self.k_int8[0, :, min_idx] = (k / k_scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
            self.v_int8[0, :, min_idx] = (v / v_scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
            self.k_scales[min_idx] = k_scale
            self.v_scales[min_idx] = v_scale
            self.importance[global_min_idx] = importance
            self.positions[global_min_idx] = position
        else:
            k_scale = k.abs().max() / 127.0
            v_scale = v.abs().max() / 127.0
            self.k_int8[0, :, self.int8_len] = (k / k_scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
            self.v_int8[0, :, self.int8_len] = (v / v_scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
            self.k_scales[self.int8_len] = k_scale
            self.v_scales[self.int8_len] = v_scale
            idx = int(self.max_tokens * self.fp16_ratio) + self.int8_len
            self.importance[idx] = importance
            self.positions[idx] = position
            self.int8_len += 1

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full K/V (FP16 + dequantized INT8)."""
        # Dequantize INT8
        k_int8_deq = self.k_int8[:, :, :self.int8_len].float() * self.k_scales[:self.int8_len]
        v_int8_deq = self.v_int8[:, :, :self.int8_len].float() * self.v_scales[:self.int8_len]

        k = torch.cat([
            self.k_fp16[:, :, :self.fp16_len],
            k_int8_deq.bfloat16(),
        ], dim=2)
        v = torch.cat([
            self.v_fp16[:, :, :self.fp16_len],
            v_int8_deq.bfloat16(),
        ], dim=2)
        return k, v

    @property
    def total_len(self) -> int:
        return self.fp16_len + self.int8_len

    def memory_bytes(self) -> int:
        """Total memory usage."""
        fp16_bytes = self.fp16_len * self.n_kv * self.head_dim * 2  # bf16
        int8_bytes = self.int8_len * self.n_kv * self.head_dim * 1  # int8
        scales_bytes = self.int8_len * 2 * 4  # fp32 scales
        return fp16_bytes + int8_bytes + scales_bytes

    def stats(self) -> dict:
        full_fp16_bytes = self.total_len * self.n_kv * self.head_dim * 2
        return {
            "fp16_tokens": self.fp16_len,
            "int8_tokens": self.int8_len,
            "total_tokens": self.total_len,
            "memory_bytes": self.memory_bytes(),
            "full_fp16_bytes": full_fp16_bytes,
            "memory_savings_pct": (1 - self.memory_bytes() / max(full_fp16_bytes, 1)) * 100,
        }


class ConfKVCache:
    """CONF-KV: confidence-aware KV cache with mixed-precision storage.

    Combines:
      - Confidence-driven dynamic budget (from next-token entropy)
      - Mixed FP16/INT8 storage (important → FP16, rest → INT8)
      - Pyramidal per-layer budget (deeper layers get more)
      - Composite token ranking (attention + recency)
    """

    def __init__(self, n_kv_heads: int, head_dim: int,
                 min_budget: int = 256, max_budget: int = 4096,
                 sink_size: int = 4, window_size: int = 64,
                 fp16_ratio: float = 0.3,
                 device: str = "cuda"):
        self.confidence_scorer = ConfidenceScorer(min_budget, max_budget)
        self.mixed_kv = MixedPrecisionKV(
            n_kv_heads, head_dim, max_tokens=max_budget,
            fp16_ratio=fp16_ratio, device=device)
        self.sink_size = sink_size
        self.window_size = window_size
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device

        # Sink + protected window
        self.k_sink = torch.zeros(1, n_kv_heads, sink_size, head_dim,
                                   dtype=torch.bfloat16, device=device)
        self.v_sink = torch.zeros_like(self.k_sink)
        self.sink_len = 0

        self.k_window = torch.zeros(1, n_kv_heads, window_size, head_dim,
                                     dtype=torch.bfloat16, device=device)
        self.v_window = torch.zeros_like(self.k_window)
        self.window_len = 0

        self.attn_accum = torch.zeros(max_budget, device=device)
        self.position = 0
        self._current_budget = max_budget

    def update_confidence(self, logits: torch.Tensor, layer_idx: int = 0):
        """Update cache budget based on model confidence."""
        confidence = self.confidence_scorer.compute_confidence(logits)
        self._current_budget = self.confidence_scorer.get_budget(
            confidence, layer_idx)

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K/V tokens."""
        T = k.shape[2]

        # First tokens → sink
        if self.sink_len < self.sink_size:
            n_to_sink = min(T, self.sink_size - self.sink_len)
            self.k_sink[0, :, self.sink_len:self.sink_len + n_to_sink] = k[0, :, :n_to_sink]
            self.v_sink[0, :, self.sink_len:self.sink_len + n_to_sink] = v[0, :, :n_to_sink]
            self.sink_len += n_to_sink
            k = k[:, :, n_to_sink:]
            v = v[:, :, n_to_sink:]
            T = k.shape[2]
            self.position += n_to_sink

        for i in range(T):
            if self.window_len >= self.window_size:
                self._promote_from_window()
            self.k_window[0, :, self.window_len] = k[0, :, i]
            self.v_window[0, :, self.window_len] = v[0, :, i]
            self.window_len += 1
            self.position += 1

    def _promote_from_window(self):
        """Move oldest window token to mixed-precision storage."""
        k = self.k_window[0, :, 0:1]
        v = self.v_window[0, :, 0:1]

        # Composite importance: attention mass + recency
        if self.mixed_kv.total_len > 0:
            max_pos = self.mixed_kv.positions[:self.mixed_kv.total_len].max().item()
        else:
            max_pos = 0
        recency = 1.0 / (1 + self.position - max_pos)
        importance = self.attn_accum[0].item() + recency

        self.mixed_kv.add(k.unsqueeze(0), v.unsqueeze(0), importance, self.position)

        # Shift window
        self.k_window[0, :, :self.window_len - 1] = self.k_window[0, :, 1:self.window_len]
        self.v_window[0, :, :self.window_len - 1] = self.v_window[0, :, 1:self.window_len]
        self.window_len -= 1

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full K/V (sink + mixed-precision + window)."""
        k_mixed, v_mixed = self.mixed_kv.get_kv()
        k = torch.cat([
            self.k_sink[:, :, :self.sink_len],
            k_mixed,
            self.k_window[:, :, :self.window_len],
        ], dim=2)
        v = torch.cat([
            self.v_sink[:, :, :self.sink_len],
            v_mixed,
            self.v_window[:, :, :self.window_len],
        ], dim=2)
        return k, v

    @property
    def total_len(self) -> int:
        return self.sink_len + self.mixed_kv.total_len + self.window_len

    def stats(self) -> dict:
        return {
            "current_budget": self._current_budget,
            "confidence": self.confidence_scorer._confidence_history[-1] if self.confidence_scorer._confidence_history else 0,
            "mixed_kv": self.mixed_kv.stats(),
            "total_len": self.total_len,
        }
