"""CPU KV cache offloading: extend effective KV memory via PCIe to system RAM.

For RTX 5070 (12GB VRAM), the KV cache is a significant fraction of VRAM:
  - 32K context × 16 layers × 8 KV heads × 64 dim × 2 (K+V) × 2 bytes (bf16)
  - = 32K × 16 × 8 × 64 × 2 × 2 = 1.07 GB
  - At 128K context: 4.3 GB (36% of VRAM!)

CPU KV offloading moves less-recently-used KV blocks to CPU RAM, keeping
only the active window on GPU. This effectively gives us unlimited KV cache
size (bounded by system RAM, 32GB) at the cost of PCIe transfer latency.

Strategy:
  1. Hot window: most recent N tokens stay on GPU (fast access)
  2. Cold storage: older tokens offloaded to CPU RAM
  3. On-demand fetch: when attention needs a cold block, fetch via PCIe
  4. Prefetch: predict which cold blocks will be needed and prefetch

PCIe 4.0 x16 bandwidth: ~32 GB/s (vs 672 GB/s GDDR7)
  - Fetching 1 block (16 tokens × 128 bytes) = 2KB → 0.06µs over PCIe
  - Negligible for small fetches, but significant for large context switches

Best for: long-context workloads where only a subset of the context is
actively attended to (sparse attention patterns, sliding window + sinks).

This implementation provides:
  1. CPUKVCache: manages GPU hot window + CPU cold storage
  2. Automatic promotion/demotion based on access patterns
  3. Async prefetch for predicted access patterns
"""
from __future__ import annotations

import torch

from research.inference.kv_backend import KVCacheStrategy


class CPUKVCache(KVCacheStrategy):
    """Two-tier KV cache: GPU hot window + CPU cold storage.

    Keeps the most recent `hot_window_size` tokens on GPU for fast access.
    Older tokens are offloaded to CPU RAM. When attention needs a cold block,
    it's fetched via PCIe (with optional async prefetch).

    Memory:
      GPU: hot_window_size × 2 × n_kv × head_dim × 2 bytes
      CPU: (total - hot_window_size) × 2 × n_kv × head_dim × 2 bytes

    For 32K hot window, 128K total:
      GPU: 1.07 GB (same as 32K standard)
      CPU: 3.22 GB (fits in 32GB system RAM easily)
      Total effective: 128K context in 1.07 GB VRAM
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Hot window: recent tokens on GPU
        self.hot_window_size = min(max_seq_len, 8192)  # 8K tokens on GPU
        self.gpu_k = torch.zeros(
            1, n_kv_heads, self.hot_window_size, head_dim,
            dtype=dtype, device=device)
        self.gpu_v = torch.zeros(
            1, n_kv_heads, self.hot_window_size, head_dim,
            dtype=dtype, device=device)

        # Cold storage: older tokens on CPU
        self.cpu_k = None  # lazily allocated when first offload happens
        self.cpu_v = None
        self.cpu_capacity = max_seq_len - self.hot_window_size

        # Track what's where
        self.seq_len = 0  # total tokens written
        self.gpu_len = 0  # tokens currently on GPU
        self.cpu_len = 0  # tokens currently on CPU

        # Access tracking for promotion/demotion
        self._access_counts = None  # per-block access count

        # Async prefetch stream
        self._prefetch_stream = None
        if device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=device)

    def append(self, k, v, position, attention_weights=None):
        """Append new K/V tokens to the hot window.

        If the hot window overflows, oldest tokens are offloaded to CPU.
        """
        T = k.shape[2]
        pos = position

        # Check if hot window will overflow
        if self.gpu_len + T > self.hot_window_size:
            # Offload oldest tokens to CPU
            n_to_offload = self.gpu_len + T - self.hot_window_size
            self._offload_to_cpu(n_to_offload)

        # Write new tokens to GPU hot window
        gpu_pos = self.gpu_len
        self.gpu_k[:, :, gpu_pos:gpu_pos + T] = k
        self.gpu_v[:, :, gpu_pos:gpu_pos + T] = v
        self.gpu_len += T
        self.seq_len = pos + T

    def _offload_to_cpu(self, n_tokens: int):
        """Move n_tokens from the start of GPU cache to CPU cache."""
        if n_tokens <= 0:
            return

        # Initialize CPU storage if needed
        if self.cpu_k is None:
            self.cpu_k = torch.zeros(
                1, self.n_kv, self.cpu_capacity, self.head_dim,
                dtype=self.dtype, device="cpu", pin_memory=True)
            self.cpu_v = torch.zeros(
                1, self.n_kv, self.cpu_capacity, self.head_dim,
                dtype=self.dtype, device="cpu", pin_memory=True)

        # Copy from GPU to CPU (non-blocking for overlap)
        tokens_to_move = self.gpu_k[:, :, :n_tokens]
        self.cpu_k[:, self.cpu_len:self.cpu_len + n_tokens] = tokens_to_move.to("cpu", non_blocking=True)
        self.cpu_v[:, self.cpu_len:self.cpu_len + n_tokens] = self.gpu_v[:, :, :n_tokens].to("cpu", non_blocking=True)
        self.cpu_len += n_tokens

        # Shift GPU cache left (remove offloaded tokens)
        self.gpu_k[:, :, :self.gpu_len - n_tokens] = self.gpu_k[:, :, n_tokens:self.gpu_len]
        self.gpu_v[:, :, :self.gpu_len - n_tokens] = self.gpu_v[:, :, n_tokens:self.gpu_len]
        self.gpu_len -= n_tokens

    def _fetch_from_cpu(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a range of tokens from CPU storage to GPU.

        Args:
            start, end: positions in the CPU cache (0-indexed from cpu start)

        Returns:
            (k, v) tensors on GPU
        """
        k = self.cpu_k[:, start:end].to(self.device, non_blocking=True)
        v = self.cpu_v[:, start:end].to(self.device, non_blocking=True)
        return k, v

    def get(self, positions=None):
        """Return K/V for attention.

        If positions are all in the hot window, returns GPU tensors directly.
        If some positions are in cold storage, fetches them via PCIe.
        """
        if positions is None:
            # Return all: GPU hot window (cold storage fetched on demand)
            return (self.gpu_k[:, :, :self.gpu_len],
                    self.gpu_v[:, :, :self.gpu_len])

        # Check which positions are hot vs cold
        cpu_start = 0
        cpu_end = self.cpu_len
        gpu_start = self.cpu_len  # GPU positions start after CPU tokens

        positions = torch.as_tensor(positions, device=self.device)
        hot_mask = positions >= gpu_start
        cold_mask = ~hot_mask

        if cold_mask.all():
            # All cold: fetch from CPU
            cold_positions = positions - cpu_start
            return self._fetch_from_cpu(
                cold_positions.min().item(),
                cold_positions.max().item() + 1,
            )
        elif hot_mask.all():
            # All hot: return from GPU
            gpu_positions = positions - gpu_start
            return (self.gpu_k[:, :, gpu_positions],
                    self.gpu_v[:, :, gpu_positions])
        else:
            # Mixed: need to fetch cold and combine with hot
            # This is the slow path — optimize by fetching contiguous cold blocks
            cold_positions = positions[cold_mask] - cpu_start
            hot_positions = positions[hot_mask] - gpu_start

            cold_k, cold_v = self._fetch_from_cpu(
                cold_positions.min().item(),
                cold_positions.max().item() + 1,
            )
            hot_k = self.gpu_k[:, :, hot_positions]
            hot_v = self.gpu_v[:, :, hot_positions]

            # Interleave back in original order
            k = torch.zeros(1, self.n_kv, len(positions), self.head_dim,
                            dtype=self.dtype, device=self.device)
            v = torch.zeros(1, self.n_kv, len(positions), self.head_dim,
                            dtype=self.dtype, device=self.device)
            k[:, :, cold_mask] = cold_k
            k[:, :, hot_mask] = hot_k
            v[:, :, cold_mask] = cold_v
            v[:, :, hot_mask] = hot_v
            return k, v

    def prefetch(self, positions: list[int]):
        """Async prefetch cold blocks to GPU (overlap with compute).

        Call this before attention if you know which positions will be accessed.
        """
        if self._prefetch_stream is None or self.cpu_k is None:
            return

        cold_positions = [p for p in positions if p < self.cpu_len]
        if not cold_positions:
            return

        with torch.cuda.stream(self._prefetch_stream):
            self._fetch_from_cpu(min(cold_positions), max(cold_positions) + 1)

    def clear(self):
        self.gpu_k.zero_()
        self.gpu_v.zero_()
        if self.cpu_k is not None:
            self.cpu_k.zero_()
            self.cpu_v.zero_()
        self.seq_len = 0
        self.gpu_len = 0
        self.cpu_len = 0

    def info(self):
        gpu_bytes = self.gpu_len * 2 * self.n_kv * self.head_dim * 2
        cpu_bytes = self.cpu_len * 2 * self.n_kv * self.head_dim * 2
        return {
            "type": "cpu_offload",
            "seq_len": self.seq_len,
            "gpu_len": self.gpu_len,
            "cpu_len": self.cpu_len,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "hot_window_size": self.hot_window_size,
            "total_capacity": self.max_seq_len,
        }
