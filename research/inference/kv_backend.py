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
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod


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
    def get(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given positions. Returns [B, n_kv, T, head_dim]."""
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def info(self) -> Dict:
        """Return cache stats (size, compression ratio, etc.)."""
        pass


class StandardKVCache(KVCacheStrategy):
    """Basic KV cache — stores full-precision K/V tensors."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    def append(self, k, v, position):
        if self.k_cache is None:
            B = k.shape[0]
            self.k_cache = torch.zeros(B, self.n_kv, 0, self.head_dim,
                                       device=self.device, dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)
        self.k_cache = torch.cat([self.k_cache, k], dim=2)
        self.v_cache = torch.cat([self.v_cache, v], dim=2)
        self.seq_len = self.k_cache.shape[2]

    def get(self, positions):
        return self.k_cache, self.v_cache

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    def info(self):
        size_mb = 0
        if self.k_cache is not None:
            size_mb = (self.k_cache.numel() + self.v_cache.numel()) * 2 / 1e6
        return {"type": "standard", "seq_len": self.seq_len, "size_mb": size_mb,
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
        self.bits = 4
        self.block_size = 64  # Hadamard block size

        # Generate block-diagonal Hadamard matrix (pure torch, no scipy)
        bs = self.block_size
        H = self._hadamard_matrix(bs) / (bs ** 0.5)
        # Block-diagonal: repeat H for each block
        n_blocks = head_dim // bs
        if n_blocks * bs != head_dim:
            # Pad to next multiple of block_size
            n_blocks = (head_dim + bs - 1) // bs
            padded_dim = n_blocks * bs
            H_full = torch.zeros(padded_dim, padded_dim)
            for i in range(n_blocks):
                H_full[i*bs:(i+1)*bs, i*bs:(i+1)*bs] = H
            self.H = H_full[:head_dim, :head_dim].to(device, dtype)
        else:
            self.H = torch.block_diag(*[H] * n_blocks).to(device, dtype)
        self.H_inv = self.H.T  # Hadamard is orthogonal: inverse = transpose

        # Storage: quantized K/V
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self.seq_len = 0
        self.qmax = (1 << (self.bits - 1)) - 1  # 7 for 4-bit

    @staticmethod
    def _hadamard_matrix(n: int) -> torch.Tensor:
        """Generate Hadamard matrix of order n (must be power of 2). Pure torch."""
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        return H

    def _rotate(self, t):
        """Apply Hadamard rotation: t @ H (rotate columns)."""
        # t shape: [B, n_kv, T, head_dim]
        return torch.matmul(t, self.H)

    def _inverse_rotate(self, t):
        """Apply inverse rotation: t @ H^T."""
        return torch.matmul(t, self.H_inv)

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

        if self.k_quant is None:
            self.k_quant = k_q
            self.v_quant = v_q
            self.k_scale = k_s
            self.v_scale = v_s
        else:
            self.k_quant = torch.cat([self.k_quant, k_q], dim=2)
            self.v_quant = torch.cat([self.v_quant, v_q], dim=2)
            self.k_scale = torch.cat([self.k_scale, k_s], dim=2)
            self.v_scale = torch.cat([self.v_scale, v_s], dim=2)
        self.seq_len = self.k_quant.shape[2]

    def get(self, positions):
        # Dequantize: q * scale, then inverse rotate
        k = self.k_quant * self.k_scale
        v = self.v_quant * self.v_scale
        k = self._inverse_rotate(k)
        v = self._inverse_rotate(v)
        return k, v

    def clear(self):
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self.seq_len = 0

    def info(self):
        size_mb = 0
        if self.k_quant is not None:
            # INT4 = 0.5 bytes per element, but stored as int8 for simplicity
            actual_bytes = self.k_quant.numel() * 0.5 + self.k_scale.numel() * 2
            actual_bytes += self.v_quant.numel() * 0.5 + self.v_scale.numel() * 2
            size_mb = actual_bytes / 1e6
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

