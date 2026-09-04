"""VToken: Token-Level Virtualization for Reclaimable KV Cache.

Based on "VToken: Decoupling Logical Token Liveness from Physical Block
Placement for Reclaimable KV Cache" (arXiv 2608.13263).

Key insight: paged KV caches (vLLM, PagedAttention) allocate fixed-size blocks.
When tokens are evicted at block granularity (PagedEviction), entire blocks
are freed -- but partially-dead blocks (e.g., 3 of 16 tokens dead) waste
13/16 of their space. Token-level eviction (H2O, SnapKV) reclaims individual
tokens but fragments the paged structure, breaking PagedAttention and CUDA
Graph capture.

VToken decouples logical token liveness from physical block placement via
token-table indirection:
  - Each logical position has a "live" or "dead" status (liveness bitmap)
  - logical_to_physical[pos] = (block_idx, offset) maps logical -> physical
  - mark_dead(positions): mark tokens as dead (no physical change yet)
  - repack(): move live tokens into contiguous blocks, free dead blocks

This achieves token-level reclamation (like H2O) while preserving the
fixed-size block structure (like PagedEviction). Repacking is async
(side CUDA stream) so it doesn't stall decode.

Benefits (from paper):
  - 27-72% retained KV block reduction vs paged baseline
  - Preserves PagedAttention + CUDA Graph (blocks remain fixed-size)
  - Token-level granularity (no 13/16 waste from partial-block eviction)

VRAM budget (RTX 5070, 12GB, bf16):
  Per block: 2 * block_size * n_kv * head_dim * 2 bytes
  Default (block_size=16, n_kv=4, head_dim=64): 16 KB/block
  max_blocks=256 (4096 tokens): 4 MB -- negligible vs model weights
  max_blocks=512 (8192 tokens): 8 MB
  Token table (l2p_block + l2p_offset + liveness) adds ~24 bytes/token
  (3 tensors), ~96 KB for 4096 tokens -- also negligible.

CPU fallback: pure torch, no CUDA-specific ops. Runs on CPU with identical
semantics (repack is synchronous on CPU since there's no async stream).
"""
from __future__ import annotations

import torch

from research.inference.kv_backend import KVCacheStrategy


class VTokenKVCache(KVCacheStrategy):
    """Token-level virtualized KV cache with reclaimable physical blocks.

    Stores KV in fixed-size physical blocks (PagedAttention-compatible).
    A token table (logical_to_physical) indirection decouples logical token
    liveness from physical placement, enabling token-level reclamation via
    async repacking of live tokens into fewer blocks.

    Args:
        block_size: physical block size in tokens (default 16, matches
            PagedAttention). Larger blocks = less table overhead but more
            waste per partial block.
        max_blocks: maximum physical blocks (GPU memory budget). If None,
            defaults to ceil(max_seq_len / block_size) -- enough for the
            full sequence without repacking. Set lower to force repacking.
    """

    def __init__(self, block_size: int = 16, max_blocks: int | None = None):
        self.block_size = block_size
        self._max_blocks_param = max_blocks

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        if self._max_blocks_param is not None:
            self.max_blocks = self._max_blocks_param
        else:
            self.max_blocks = (max_seq_len + self.block_size - 1) // self.block_size

        self.k_blocks = torch.zeros(
            self.max_blocks, n_kv_heads, self.block_size, head_dim,
            dtype=dtype, device=device)
        self.v_blocks = torch.zeros_like(self.k_blocks)

        self.l2p_block = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        self.l2p_offset = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        self.token_live = torch.ones(max_seq_len, dtype=torch.bool, device=device)

        self.block_used = torch.zeros(self.max_blocks, dtype=torch.bool, device=device)
        self.alloc_cursor = 0

        self.seq_len = 0

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """Append K/V tokens at logical position.

        Args:
            k: (1, n_kv, T, head_dim) new keys
            v: (1, n_kv, T, head_dim) new values
            position: starting logical position
        """
        T = k.shape[2]
        end = position + T
        capacity = self.max_blocks * self.block_size

        if self.alloc_cursor + T > capacity:
            self.repack()
            if self.alloc_cursor + T > capacity:
                raise RuntimeError(
                    f"VToken KV cache exhausted: need {self.alloc_cursor + T} "
                    f"slots, capacity {capacity} "
                    f"(max_blocks={self.max_blocks}, block_size={self.block_size}). "
                    f"Mark more tokens dead or increase max_blocks.")

        start_slot = self.alloc_cursor
        end_slot = start_slot + T

        slots = torch.arange(start_slot, end_slot, device=self.device)
        blks = slots // self.block_size
        offs = slots % self.block_size
        logical_positions = torch.arange(position, end, device=self.device)

        self.l2p_block[logical_positions] = blks
        self.l2p_offset[logical_positions] = offs
        self.token_live[logical_positions] = True

        start_blk = start_slot // self.block_size
        end_blk = (end_slot - 1) // self.block_size
        self.block_used[start_blk:end_blk + 1] = True

        self.k_blocks[blks, :, offs] = k[0].permute(1, 0, 2)
        self.v_blocks[blks, :, offs] = v[0].permute(1, 0, 2)

        self.alloc_cursor = end_slot
        self.seq_len = max(self.seq_len, end)

    def mark_dead(self, positions):
        """Mark logical positions as dead (reclaimable by repack).

        Does not immediately free physical space -- call repack() to reclaim.
        Typical use: after a conversation turn ends, mark old system prompt
        tokens as dead so their physical blocks can be repacked away.

        Args:
            positions: int, list of ints, or (T,) tensor of logical positions
        """
        if isinstance(positions, int):
            positions = [positions]
        positions = torch.as_tensor(positions, device=self.device, dtype=torch.long)
        self.token_live[positions] = False

    def repack(self) -> int:
        """Repack live tokens into contiguous blocks, free dead blocks.

        Moves all live tokens to the front of physical storage (contiguous
        blocks), then frees any blocks beyond the live region. Updates the
        token table to reflect new physical locations.

        In the paper, this runs on a side CUDA stream (async) so it doesn't
        stall decode. This implementation is synchronous; for async, wrap
        in a torch.cuda.Stream and fence with a stream wait before the next
        attention pass.

        Returns:
            Number of physical blocks freed.
        """
        if self.seq_len == 0:
            return 0

        live_positions = self.token_live[:self.seq_len].nonzero(as_tuple=True)[0]
        n_live = live_positions.shape[0]
        old_blocks_used = self.block_used.sum().item()

        if n_live == 0:
            self._zero_used_blocks()
            self.block_used.zero_()
            self.alloc_cursor = 0
            return old_blocks_used

        old_blks = self.l2p_block[live_positions]
        old_offs = self.l2p_offset[live_positions]
        k_live = self.k_blocks[old_blks, :, old_offs]
        v_live = self.v_blocks[old_blks, :, old_offs]

        self._zero_used_blocks()
        self.block_used.zero_()

        new_slots = torch.arange(n_live, device=self.device)
        new_blks = new_slots // self.block_size
        new_offs = new_slots % self.block_size

        self.k_blocks[new_blks, :, new_offs] = k_live
        self.v_blocks[new_blks, :, new_offs] = v_live

        n_new_blocks = (n_live + self.block_size - 1) // self.block_size
        self.block_used[:n_new_blocks] = True

        self.l2p_block[live_positions] = new_blks
        self.l2p_offset[live_positions] = new_offs

        self.alloc_cursor = n_live
        freed = old_blocks_used - n_new_blocks
        # Clear dead flags — dead tokens have been reclaimed
        self.token_live[:self.seq_len] = True
        return freed

    def _zero_used_blocks(self):
        used = self.block_used.nonzero(as_tuple=True)[0]
        if used.shape[0] > 0:
            self.k_blocks[used] = 0
            self.v_blocks[used] = 0

    def get(self, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given logical positions via token table indirection.

        Args:
            positions: (T,) tensor of logical positions, or None for all
                positions [0, seq_len). Dead tokens are included; use
                get_live_mask() to exclude them from attention.

        Returns:
            (k, v) each of shape (1, n_kv, T, head_dim)
        """
        if self.seq_len == 0:
            dummy = torch.zeros(1, self.n_kv, 1, self.head_dim,
                                dtype=self.dtype, device=self.device)
            return dummy, dummy.clone()

        if positions is None:
            positions = torch.arange(self.seq_len, device=self.device)
        elif isinstance(positions, int):
            positions = torch.tensor([positions], device=self.device, dtype=torch.long)
        else:
            positions = torch.as_tensor(positions, device=self.device, dtype=torch.long)

        blks = self.l2p_block[positions]
        offs = self.l2p_offset[positions]
        k = self.k_blocks[blks, :, offs]
        v = self.v_blocks[blks, :, offs]
        # k/v shape: (*positions.shape, n_kv, head_dim) → flatten to (N, n_kv, head_dim)
        k = k.reshape(-1, k.shape[-2], k.shape[-1])
        v = v.reshape(-1, v.shape[-2], v.shape[-1])
        return k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0)

    def get_live_mask(self) -> torch.Tensor:
        """Return boolean mask of live token positions (for attention)."""
        return self.token_live[:self.seq_len].clone()

    def get_live_positions(self) -> torch.Tensor:
        """Return logical positions of live tokens."""
        return self.token_live[:self.seq_len].nonzero(as_tuple=True)[0]

    def clear(self):
        self._zero_used_blocks()
        self.block_used.zero_()
        self.l2p_block.zero_()
        self.l2p_offset.zero_()
        self.token_live.fill_(True)
        self.alloc_cursor = 0
        self.seq_len = 0

    def info(self) -> dict:
        n_blocks_used = self.block_used.sum().item()
        n_blocks_free = self.max_blocks - n_blocks_used
        n_dead = (~self.token_live[:self.seq_len]).sum().item()
        n_live = self.seq_len - n_dead
        total_slots = n_blocks_used * self.block_size
        fragmentation_ratio = 1.0 - (n_live / max(total_slots, 1))
        blocks_without_vtoken = (self.seq_len + self.block_size - 1) // self.block_size
        blocks_with_vtoken = (n_live + self.block_size - 1) // self.block_size
        reclaim_efficiency = (
            (blocks_without_vtoken - blocks_with_vtoken) / max(blocks_without_vtoken, 1)
            if blocks_without_vtoken > 0 else 0.0
        )
        return {
            "type": "vtoken",
            "seq_len": self.seq_len,
            "block_size": self.block_size,
            "max_blocks": self.max_blocks,
            "n_blocks_used": n_blocks_used,
            "n_blocks_free": n_blocks_free,
            "n_dead_tokens": n_dead,
            "n_live_tokens": n_live,
            "fragmentation_ratio": fragmentation_ratio,
            "reclaim_efficiency": reclaim_efficiency,
        }
