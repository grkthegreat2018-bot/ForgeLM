"""HiSparse: Hierarchical HBM-DRAM KV cache management.

Reference: arXiv 2608.07009 — "HiSparse: Hierarchical KV Cache Management for
Efficient Long-Context LLM Inference"

HiSparse maintains the **full KV history** in host memory (DRAM) while keeping
a **fixed-size GPU cache** (HBM) for the working set.  The key insight is that
attention during decode touches only a sparse subset of positions, so a modest
GPU cache with LRU eviction achieves high hit rates while DRAM provides exact
recall for any position.

Architecture (two-tier):
  - GPU HBM cache: fixed ``gpu_cache_size`` slots, LRU eviction.  Stores the
    most-recently-accessed KV positions.  This is the hot tier — attention
    reads from here with zero PCIe overhead.
  - CPU DRAM store: full KV history (all positions ever appended), backed by
    pinned memory for fast H2D transfers.  This is the cold tier — a miss
    triggers a synchronous CPU→GPU copy.

The paper describes a **fused CUDA kernel** that performs hit detection, LRU
replacement, and H2D fetches *inside* the decode CUDA graph, hiding ~50% of
miss overhead via layer-wise prefetching.  This Python reference implementation
does the hit/miss logic and LRU management in Python with synchronous H2D
copies; the fused kernel + CUDA-graph integration is documented as future work.

Exactness: outputs are unchanged — every position is recoverable from DRAM, so
no information is lost (unlike eviction-based methods like SnapKV/H2O).

VRAM budget (RTX 5070, 12GB):
  GPU: gpu_cache_size × 2 × n_kv × head_dim × 2 bytes
    e.g. 4096 × 2 × 8 × 64 × 2 = 8 MB per layer (negligible vs 12GB)
  CPU: max_seq_len × 2 × n_kv × head_dim × 2 bytes (pinned)
    e.g. 128K × 2 × 8 × 64 × 2 = 256 MB per layer (fits in 32GB RAM)

Merged into SGLang in the paper; this is the standalone Python reference.
"""
from __future__ import annotations

import torch

from forge.engine.kv_backend import KVCacheStrategy


class HiSparseKVCache(KVCacheStrategy):
    """Two-tier KV cache: fixed-size GPU HBM cache (LRU) + CPU DRAM full history.

    Every appended position is written to the CPU DRAM store (full history,
    pinned memory).  A fixed-size GPU cache holds the working set with LRU
    eviction.  On ``get``, GPU hits return directly; misses trigger a
    synchronous CPU→GPU copy (the fused CUDA kernel is future work).

    Memory:
      GPU: gpu_cache_size × 2 × n_kv × head_dim × dtype_bytes
      CPU: max_seq_len × 2 × n_kv × head_dim × dtype_bytes (pinned)
    """

    def __init__(self, gpu_cache_size: int = 4096):
        """Configure the GPU HBM cache size.

        Args:
            gpu_cache_size: max number of KV positions resident in GPU memory.
                Larger values improve hit rate but consume more VRAM.
        """
        self._gpu_cache_size_cfg = gpu_cache_size

    def init(self, n_heads: int, head_dim: int, n_kv_heads: int,
             max_seq_len: int, device, dtype: torch.dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = (device if isinstance(device, torch.device)
                       else torch.device(device))
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.gpu_cache_size = min(self._gpu_cache_size_cfg, max_seq_len)

        # GPU HBM cache: fixed-size pool of slots
        self.gpu_k = torch.zeros(
            1, n_kv_heads, self.gpu_cache_size, head_dim,
            dtype=dtype, device=self.device)
        self.gpu_v = torch.zeros_like(self.gpu_k)

        # CPU DRAM store: full history, pinned for fast H2D
        self.cpu_k = torch.zeros(
            1, n_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device="cpu", pin_memory=True)
        self.cpu_v = torch.zeros_like(self.cpu_k)

        # Position → GPU slot mapping (and reverse)
        self._pos_to_slot: dict[int, int] = {}
        self._slot_to_pos: dict[int, int] = {}

        # LRU: monotonic access counter; each slot has a last-access timestamp
        self._access_counter = 0
        self._slot_access: dict[int, int] = {}

        # Free slots (not currently holding any position)
        self._free_slots: list[int] = list(range(self.gpu_cache_size))

        # Hit/miss statistics
        self._hits = 0
        self._misses = 0

        self.seq_len = 0
        self.gpu_cached_count = 0
        self.cpu_cached_count = 0

        # Prefetch stream for layer-wise overlap (best-effort)
        self._prefetch_stream = None
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int,
               attention_weights=None):
        """Append new K/V tokens at ``position``.

        Always writes to the CPU DRAM store (full history).  Then attempts to
        place the new positions into the GPU cache, evicting LRU slots if the
        cache is full.
        """
        T = k.shape[2]
        end = position + T

        # Write to CPU DRAM store (full history — always)
        self.cpu_k[:, :, position:end] = k.to("cpu", non_blocking=True)
        self.cpu_v[:, :, position:end] = v.to("cpu", non_blocking=True)

        # Try to place each new position into the GPU cache
        for i in range(T):
            pos = position + i
            self._insert_gpu(pos, k[:, :, i:i+1], v[:, :, i:i+1])

        self.seq_len = max(self.seq_len, end)
        self.cpu_cached_count = self.seq_len

    def _insert_gpu(self, pos: int, k_tok: torch.Tensor, v_tok: torch.Tensor):
        """Insert a single position into the GPU cache, evicting LRU if full."""
        if pos in self._pos_to_slot:
            slot = self._pos_to_slot[pos]
            self.gpu_k[:, :, slot:slot+1] = k_tok
            self.gpu_v[:, :, slot:slot+1] = v_tok
            self._touch_slot(slot)
            return

        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._evict_lru()

        self.gpu_k[:, :, slot:slot+1] = k_tok
        self.gpu_v[:, :, slot:slot+1] = v_tok
        self._pos_to_slot[pos] = slot
        self._slot_to_pos[slot] = pos
        self._touch_slot(slot)
        self.gpu_cached_count = len(self._pos_to_slot)

    def _evict_lru(self) -> int:
        """Evict the least-recently-used slot and return it for reuse."""
        slot = min(self._slot_access, key=lambda s: self._slot_access[s])
        evicted_pos = self._slot_to_pos.pop(slot)
        self._pos_to_slot.pop(evicted_pos)
        self._slot_access.pop(slot)
        self.gpu_cached_count = len(self._pos_to_slot)
        return slot

    def _touch_slot(self, slot: int):
        """Update the LRU access timestamp for a slot."""
        self._access_counter += 1
        self._slot_access[slot] = self._access_counter

    def get(self, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given positions.

        GPU cache hits return directly.  Misses trigger a synchronous CPU→GPU
        copy from the DRAM store.  Hit/miss counts are updated for statistics.

        Args:
            positions: 1-D tensor or list of absolute positions.  If None,
                returns all GPU-cached positions (the working set).

        Returns:
            (k, v) tensors of shape [1, n_kv, len(positions), head_dim] on GPU.
        """
        if positions is None:
            if not self._pos_to_slot:
                empty = torch.empty(1, self.n_kv, 0, self.head_dim,
                                    dtype=self.dtype, device=self.device)
                return empty, empty.clone()
            slots = sorted(self._pos_to_slot.values(),
                           key=lambda s: self._slot_to_pos[s])
            return self.gpu_k[:, :, slots], self.gpu_v[:, :, slots]

        if isinstance(positions, (list, tuple)):
            positions = torch.as_tensor(positions, dtype=torch.long)

        # Flatten to 1D list of positions
        if positions.dim() > 1:
            positions = positions.reshape(-1)
        pos_list = positions.tolist()
        n = len(pos_list)

        k_out = torch.empty(1, self.n_kv, n, self.head_dim,
                            dtype=self.dtype, device=self.device)
        v_out = torch.empty_like(k_out)

        hit_slots = []
        hit_indices = []
        miss_positions = []
        miss_indices = []

        for i, pos in enumerate(pos_list):
            if pos in self._pos_to_slot:
                slot = self._pos_to_slot[pos]
                self._touch_slot(slot)
                hit_slots.append(slot)
                hit_indices.append(i)
                self._hits += 1
            else:
                miss_positions.append(pos)
                miss_indices.append(i)
                self._misses += 1

        if hit_slots:
            hit_idx = torch.tensor(hit_indices, device=self.device, dtype=torch.long)
            slot_idx = torch.tensor(hit_slots, device=self.device, dtype=torch.long)
            k_out[:, :, hit_idx] = self.gpu_k[:, :, slot_idx]
            v_out[:, :, hit_idx] = self.gpu_v[:, :, slot_idx]

        if miss_positions:
            miss_idx = torch.tensor(miss_indices, device=self.device, dtype=torch.long)
            pos_tensor = torch.tensor(miss_positions, device=self.device, dtype=torch.long)
            # Synchronous fetch from CPU DRAM (fused CUDA kernel is future work)
            fetched_k = self.cpu_k[:, :, pos_tensor].to(self.device, non_blocking=True)
            fetched_v = self.cpu_v[:, :, pos_tensor].to(self.device, non_blocking=True)
            k_out[:, :, miss_idx] = fetched_k
            v_out[:, :, miss_idx] = fetched_v
            # Promote fetched positions into the GPU cache
            for j, pos in enumerate(miss_positions):
                self._insert_gpu(pos, fetched_k[:, :, j:j+1], fetched_v[:, :, j:j+1])

        return k_out, v_out

    def prefetch_layer(self, layer_idx: int, positions):
        """Prefetch KV for the next layer while the current layer computes.

        In the paper, this is fused into the decode CUDA graph so the H2D
        transfer overlaps with the current layer's GEMM.  Here we issue an
        async copy on a side stream as a best-effort overlap.

        Args:
            layer_idx: index of the layer whose KV will be needed next.
            positions: positions to prefetch into the GPU cache.
        """
        if self._prefetch_stream is None:
            return
        if isinstance(positions, (list, tuple)):
            positions = torch.as_tensor(positions, dtype=torch.long)
        positions = positions.tolist() if hasattr(positions, 'tolist') else list(positions)
        miss_positions = [p for p in positions if p not in self._pos_to_slot]
        if not miss_positions:
            return
        with torch.cuda.stream(self._prefetch_stream):
            pos_tensor = torch.tensor(miss_positions, device=self.device, dtype=torch.long)
            fetched_k = self.cpu_k[:, :, pos_tensor].to(self.device, non_blocking=True)
            fetched_v = self.cpu_v[:, :, pos_tensor].to(self.device, non_blocking=True)
            for j, pos in enumerate(miss_positions):
                self._insert_gpu(pos, fetched_k[:, :, j:j+1], fetched_v[:, :, j:j+1])

    def hit_rate(self) -> float:
        """Return the cache hit rate (hits / total accesses)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear(self):
        self.gpu_k.zero_()
        self.gpu_v.zero_()
        if self.cpu_k is not None:
            self.cpu_k.zero_()
            self.cpu_v.zero_()
        self._pos_to_slot.clear()
        self._slot_to_pos.clear()
        self._slot_access.clear()
        self._free_slots = list(range(self.gpu_cache_size))
        self._access_counter = 0
        self._hits = 0
        self._misses = 0
        self.seq_len = 0
        self.gpu_cached_count = 0
        self.cpu_cached_count = 0

    def info(self) -> dict:
        per_tok = 2 * self.n_kv * self.head_dim * self.dtype.itemsize
        return {
            "type": "hisparse_hbm_dram",
            "seq_len": self.seq_len,
            "gpu_cache_size": self.gpu_cache_size,
            "gpu_cached_count": self.gpu_cached_count,
            "cpu_cached_count": self.cpu_cached_count,
            "hit_rate": self.hit_rate(),
            "hits": self._hits,
            "misses": self._misses,
            "gpu_bytes": self.gpu_cached_count * per_tok,
            "cpu_bytes": self.cpu_cached_count * per_tok,
            "compression": 1.0,
        }
