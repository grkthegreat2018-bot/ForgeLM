"""KV cache strategy backends.

Pluggable KV cache implementations selectable at runtime:
  - StandardKVCache: basic tensor cache (baseline)
  - PagedKVCacheStrategy: vLLM-style paged memory (wraps paged_kv.py)
  - RotorQuantKVCache: Givens rotation + Lloyd-Max quantization (wraps rotorquant.py)
  - HadamardKVCache: block-diagonal Hadamard + INT4 quantization (SAW-INT4 inspired)
  - CompressedKVCacheStrategy: H2O eviction + KV quant (wraps kv_compress.py)

All implement the KVCacheStrategy interface:
  init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
  append(k, v, position) -> None
  get(positions) -> (k, v)
  clear()
  info() -> dict
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class KVCacheStrategy(ABC):
    """Base interface for KV cache backends."""

    @abstractmethod
    def init(self, n_heads: int, head_dim: int, n_kv_heads: int,
             max_seq_len: int, device: str, dtype: torch.dtype):
        pass

    @abstractmethod
    def append(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """Append K/V for one position. k/v shape: [B, n_kv, 1, head_dim]."""
        pass

    @abstractmethod
    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given positions. Returns [B, n_kv, T, head_dim]."""
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def info(self) -> dict:
        """Return cache stats (size, compression ratio, etc.)."""
        pass


class StandardKVCache(KVCacheStrategy):
    """Basic KV cache — pre-allocated tensor cache (no torch.cat per step).

    Unlike the old torch.cat approach, this pre-allocates the full cache
    upfront and fills by position — inspired by llama.cpp's KV cell model.
    This avoids per-token allocation + copy (10-20% decode speedup).
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        # Pre-allocate K/V cache — filled by position, no reallocation
        self.k_cache = None  # lazily allocated on first append (need batch size)
        self.v_cache = None
        self.seq_len = 0

    def _ensure_allocated(self, batch_size):
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                batch_size, self.n_kv, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

    def append(self, k, v, position):
        self._ensure_allocated(k.shape[0])
        seq = k.shape[2]
        end = position + seq
        # Fill by position — no allocation, just in-place copy
        self.k_cache[:, :, position:end].copy_(k)
        self.v_cache[:, :, position:end].copy_(v)
        self.seq_len = end

    def get(self, positions):
        # Return only the filled portion (slices are views, no copy)
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]

    def clear(self):
        # Zero out the filled portion (keep allocation for reuse)
        if self.k_cache is not None:
            self.k_cache[:, :, :self.seq_len].zero_()
            self.v_cache[:, :, :self.seq_len].zero_()
        self.seq_len = 0

    def info(self):
        size_mb = 0
        if self.k_cache is not None:
            size_mb = (self.k_cache.numel() + self.v_cache.numel()) * 2 / 1e6
        return {"type": "standard_prealloc", "seq_len": self.seq_len,
                "max_seq_len": self.max_seq_len, "size_mb": size_mb,
                "compression": 1.0}


class PagedKVCacheStrategy(KVCacheStrategy):
    """vLLM-style paged KV cache — wraps existing paged_kv.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from research.quantization.paged_kv import PagedKVCache
        self.cache = PagedKVCache(
            n_blocks=max(256, max_seq_len // 16),
            block_size=16, n_heads=n_kv_heads,
            head_dim=head_dim, device=device,
        )
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = position + 1

    def get(self, positions):
        return self.cache.get(positions)

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return {"type": "paged", "seq_len": self.seq_len,
                "n_blocks": self.cache.n_blocks, "compression": 1.0}


class RotorQuantKVCache(KVCacheStrategy):
    """RotorQuant KV cache — Givens rotation + Lloyd-Max quantization.

    Wraps existing rotorquant.py. 3-4 bit KV cache with 0.94% error.
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from research.quantization.rotorquant import RotorQuantCache
        self.cache = RotorQuantCache(
            d_model=n_kv_heads * head_dim, n_heads=n_kv_heads,
            head_dim=head_dim, bits=4, max_seq_len=max_seq_len,
            n_kv_heads=n_kv_heads, device=device, dtype=dtype,
        )
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = position + 1

    def get(self, positions):
        return self.cache.get(positions)

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return {"type": "rotorquant", "seq_len": self.seq_len,
                "bits": 4, "compression": 16 / 4}


class HadamardKVCache(KVCacheStrategy):
    """Block-diagonal Hadamard rotation + INT4 quantization.

    Inspired by SAW-INT4 (arxiv 2604.19157): "a simple design—token-wise INT4
    quantization with block-diagonal Hadamard rotation—consistently achieves
    the best accuracy-efficiency trade-off."

    The Hadamard rotation Gaussianizes the K/V distribution, making uniform
    INT4 quantization near-lossless. Block-diagonal structure ensures
    compatibility with paged memory layouts.

    If the model already has QuaRot (Hadamard on V/O), the V cache benefits
    from double rotation — but we still rotate K (which QuaRot doesn't cover).
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.bits = 4
        self.block_size = 64  # Hadamard block size

        # Block-diagonal Hadamard via per-block FWHT (O(n log n), no matrix
        # materialization). Equivalent to matmul with block-diagonal H/sqrt(bs)
        # but avoids the O(n^2) construction and the O(n^2) per-append matmul.
        bs = self.block_size
        assert bs & (bs - 1) == 0, "block_size must be power of 2"
        self.n_blocks = (head_dim + bs - 1) // bs
        self.padded_dim = self.n_blocks * bs
        self._norm = 1.0 / (bs ** 0.5)
        # Precompute FWHT bit-reversal stages as strided index pairs for speed.
        self._stages = self._fwht_stage_indices(bs, device)

        # Storage: quantized K/V
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self.seq_len = 0
        self.qmax = (1 << (self.bits - 1)) - 1  # 7 for 4-bit

    @staticmethod
    def _fwht_stage_indices(n: int, device):
        """Precompute (i, j) index pairs for each of log2(n) FWHT butterfly stages."""
        stages = []
        h = 1
        while h < n:
            idx_i = []
            idx_j = []
            for i in range(0, n, h * 2):
                for k in range(i, i + h):
                    idx_i.append(k)
                    idx_j.append(k + h)
            stages.append((
                torch.tensor(idx_i, device=device, dtype=torch.long),
                torch.tensor(idx_j, device=device, dtype=torch.long),
            ))
            h *= 2
        return stages

    def _fwht(self, t):
        """In-place Fast Walsh-Hadamard Transform along the last dim.

        t: [..., block_size] (block_size must be power of 2). Returns the
        unnormalized Hadamard transform of the last dim.
        """
        for idx_i, idx_j in self._stages:
            a = t[..., idx_i]
            b = t[..., idx_j]
            t[..., idx_i] = a + b
            t[..., idx_j] = a - b
        return t

    def _rotate(self, t):
        """Apply block-diagonal Hadamard rotation via per-block FWHT.

        Equivalent to t @ H where H is block-diagonal(H_bs/sqrt(bs)), but
        O(n log n) per block instead of O(n^2) matmul and no matrix tensor.
        t shape: [B, n_kv, T, head_dim].
        """
        *lead, hd = t.shape
        if hd != self.padded_dim:
            t = F.pad(t, (0, self.padded_dim - hd), value=0)
        t = t.reshape(*lead, self.n_blocks, self.block_size)
        t = self._fwht(t.contiguous().clone())
        t = t * self._norm
        t = t.reshape(*lead, self.padded_dim)
        if hd != self.padded_dim:
            t = t[..., :hd]
        return t

    def _inverse_rotate(self, t):
        """Inverse Hadamard rotation. H/sqrt(n) is orthogonal and symmetric,
        so the inverse equals the forward transform."""
        return self._rotate(t)

    def _quantize(self, t):
        """Per-token INT4 quantization on rotated values."""
        # t shape: [B, n_kv, T, head_dim]
        scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        q = torch.clamp(torch.round(t / scale), -self.qmax, self.qmax)
        return q, scale

    def append(self, k, v, position):
        # Rotate then quantize
        k_rot = self._rotate(k)
        v_rot = self._rotate(v)
        k_q, k_s = self._quantize(k_rot)
        v_q, v_s = self._quantize(v_rot)

        B, n_kv, T, hd = k_q.shape
        end = position + T
        if self.k_quant is None:
            # Pre-allocate the static max buffer once — writing by position
            # avoids the per-token torch.cat (which reallocated + copied the
            # whole cache on every decode step).
            self.k_quant = torch.zeros(B, n_kv, self.max_seq_len, hd,
                                       device=k_q.device, dtype=k_q.dtype)
            self.v_quant = torch.zeros_like(self.k_quant)
            self.k_scale = torch.zeros(B, n_kv, self.max_seq_len, 1,
                                       device=k_s.device, dtype=k_s.dtype)
            self.v_scale = torch.zeros_like(self.k_scale)
        self.k_quant[:, :, position:end] = k_q
        self.v_quant[:, :, position:end] = v_q
        self.k_scale[:, :, position:end] = k_s
        self.v_scale[:, :, position:end] = v_s
        self.seq_len = max(self.seq_len, end)

    def get(self, positions):
        # Dequantize: q * scale, then inverse rotate
        k = self.k_quant[:, :, :self.seq_len] * self.k_scale[:, :, :self.seq_len]
        v = self.v_quant[:, :, :self.seq_len] * self.v_scale[:, :, :self.seq_len]
        k = self._inverse_rotate(k)
        v = self._inverse_rotate(v)
        return k, v

    def clear(self):
        # Keep the allocation for reuse (static buffer pattern) — just reset.
        self.seq_len = 0

    def info(self):
        size_mb = 0
        if self.k_quant is not None:
            # INT4 = 0.5 bytes per element (stored as float for simplicity)
            n = self.seq_len
            actual_bytes = n * self.n_kv * self.padded_dim * 0.5
            actual_bytes += n * self.n_kv * 2  # per-token scales
            size_mb = 2 * actual_bytes / 1e6  # K + V
        return {"type": "hadamard_int4", "seq_len": self.seq_len,
                "bits": self.bits, "block_size": self.block_size,
                "size_mb": size_mb, "compression": 16 / self.bits}


class CompressedKVCacheStrategy(KVCacheStrategy):
    """H2O heavy-hitter eviction + KV quantization — wraps kv_compress.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from research.quantization.kv_compress import CompressedKVCache
        self.cache = CompressedKVCache(
            n_heads=n_kv_heads, head_dim=head_dim,
            max_tokens=max_seq_len, bits=8,
            keep_ratio=0.5, device=device,
        )
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = position + 1

    def get(self, positions):
        return self.cache.get(positions)

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return {"type": "compressed_h2o", "seq_len": self.seq_len,
                "bits": 8, "keep_ratio": 0.5, "compression": 16 / 8 * 2}


def build_kv_cache(strategy: str = "standard", **kwargs) -> KVCacheStrategy:
    """Factory: build KV cache by strategy name."""
    strategies = {
        "standard": StandardKVCache,
        "paged": PagedKVCacheStrategy,
        "rotorquant": RotorQuantKVCache,
        "hadamard_int4": HadamardKVCache,
        "compressed": CompressedKVCacheStrategy,
        "streaming": StreamingKVCacheStrategy,
        "snapkv": SnapKVCacheStrategy,
        "2bit": KV2BitCacheStrategy,
    }
    cls = strategies.get(strategy, StandardKVCache)
    return cls()


class StreamingKVCacheStrategy(KVCacheStrategy):
    """StreamingLLM: attention sinks + sliding window. Wraps streaming_llm.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from research.inference.streaming_llm import StreamingKVCache
        n_sinks = 4
        window = min(512, max_seq_len)
        self.cache = StreamingKVCache(
            n_sinks=n_sinks, window_size=window,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            device=device, dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        k, v, _ = self.cache.get()
        return k, v

    def get_past_kv(self):
        return self.cache.get_past_kv()

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return self.cache.info()


class SnapKVCacheStrategy(KVCacheStrategy):
    """SnapKV: observation-window eviction. Wraps snapkv.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from research.inference.snapkv import SnapKVCache
        obs_window = 128
        budget = min(512, max_seq_len)
        self.cache = SnapKVCache(
            observation_window=obs_window, budget=budget,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            device=device, dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k, v, position, attention_weights=None):
        self.cache.append(k, v, position, attention_weights=attention_weights)
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        return self.cache.get()

    def get_past_kv(self):
        return self.cache.get_past_kv()

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return self.cache.info()


class KV2BitCacheStrategy(KVCacheStrategy):
    """2-bit KV cache quantization (NSNQuant). 7x compression, <0.3 PPL loss."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from research.quantization.kv_2bit import KV2BitCache
        n_layers = 16
        self.cache = KV2BitCache(
            n_kv_heads=n_kv_heads, head_dim=head_dim, n_layers=n_layers,
            max_seq_len=max_seq_len, device=device, dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k, v, position, attention_weights=None):
        if not hasattr(self, '_layer_counter'):
            self._layer_counter = 0
        self.cache.append(self._layer_counter % self.cache.n_layers, k, v)
        self._layer_counter += 1
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        return self.cache.get(self._layer_counter % self.cache.n_layers)

    def get_past_kv(self):
        return self.cache.get(self._layer_counter % self.cache.n_layers)

    def clear(self):
        self.cache.seq_len = 0
        self._layer_counter = 0
        self.seq_len = 0

    def info(self):
        return self.cache.info()

