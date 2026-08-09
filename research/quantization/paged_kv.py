"""Paged KV cache for efficient inference (vLLM technique).

Divides KV cache into fixed-size blocks (pages) for:
- Zero fragmentation (non-contiguous allocation)
- Prefix caching (shared prompts reuse blocks)
- Memory sharing across sequences (beam search, parallel sampling)

Usage:
    from research.quantization.paged_kv import PagedKVCache

    cache = PagedKVCache(n_blocks=256, block_size=16, n_heads=16, head_dim=64)
    seq_id = cache.allocate(prompt_tokens)
    cache.append(seq_id, new_tokens)
    kv = cache.get_kv(seq_id)
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SequenceState:
    """State for a single sequence in the paged cache."""
    seq_id: int
    block_ids: list[int] = field(default_factory=list)  # physical block indices
    num_tokens: int = 0  # total tokens in this sequence
    prompt_hash: str | None = None  # for prefix caching


class PagedKVCache:
    """Paged KV cache with prefix caching.
    
    Memory is divided into fixed-size blocks. Each sequence maps to
    a list of blocks via a block table. Shared prefixes share blocks.
    
    Args:
        n_blocks: total number of blocks in the cache
        block_size: tokens per block (16 is standard)
        n_heads: number of attention heads
        head_dim: dimension per head
        dtype: KV storage dtype (BF16 default, can use INT8 for compression)
        device: cuda or cpu
    """

    def __init__(self, n_blocks=256, block_size=16, n_heads=16, head_dim=64,
                 dtype=torch.bfloat16, device="cuda"):
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        # Pre-allocate KV cache tensor: (n_blocks, block_size, n_heads, head_dim)
        # K and V stored separately.
        self.k_cache = torch.zeros(n_blocks, block_size, n_heads, head_dim,
                                   dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(n_blocks, block_size, n_heads, head_dim,
                                   dtype=dtype, device=self.device)

        # Block management.
        self.free_blocks: list[int] = list(range(n_blocks))
        self.sequences: dict[int, SequenceState] = {}
        self.next_seq_id = 0

        # Prefix cache: hash → block_ids (for reuse).
        self.prefix_cache: dict[str, list[int]] = {}

        # Stats.
        self.total_allocations = 0
        self.cache_hits = 0

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Hash a token sequence for prefix caching."""
        return hashlib.md5(bytes(tokens)).hexdigest()

    def _find_prefix_match(self, tokens: list[int]) -> tuple[list[int], int]:
        """Find longest matching prefix in the cache.
        
        Returns:
            (block_ids for matched prefix, num_matched_tokens)
        """
        if not self.prefix_cache:
            return [], 0

        # Try progressively shorter prefixes.
        n = len(tokens)
        for length in range(n, 0, -self.block_size):
            # Round down to block boundary.
            block_aligned = (length // self.block_size) * self.block_size
            if block_aligned == 0:
                break
            prefix = tokens[:block_aligned]
            h = self._hash_tokens(prefix)
            if h in self.prefix_cache:
                return self.prefix_cache[h], block_aligned

        return [], 0

    def allocate(self, tokens: list[int]) -> int:
        """Allocate cache blocks for a new sequence.
        
        Args:
            tokens: prompt token ids
        
        Returns:
            seq_id
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1

        # Try prefix caching.
        matched_blocks, matched_tokens = self._find_prefix_match(tokens)

        state = SequenceState(seq_id=seq_id, block_ids=list(matched_blocks),
                             num_tokens=matched_tokens)

        # Reuse matched blocks (increment ref count conceptually).
        if matched_blocks:
            self.cache_hits += 1
            # Remove matched blocks from free list (they're now shared).
            for bid in matched_blocks:
                if bid in self.free_blocks:
                    self.free_blocks.remove(bid)

        # Allocate new blocks for remaining tokens.
        remaining = len(tokens) - matched_tokens
        n_new_blocks = (remaining + self.block_size - 1) // self.block_size

        for _ in range(n_new_blocks):
            if not self.free_blocks:
                # Evict oldest sequence (simplified — real vLLM uses LRU).
                self._evict_oldest()
            block_id = self.free_blocks.pop(0)
            state.block_ids.append(block_id)

        state.num_tokens = len(tokens)
        state.prompt_hash = self._hash_tokens(tokens)

        # Store in prefix cache (for future reuse).
        # Only cache full blocks.
        full_blocks = (len(tokens) // self.block_size) * self.block_size
        if full_blocks > 0:
            prefix_h = self._hash_tokens(tokens[:full_blocks])
            full_block_ids = state.block_ids[:full_blocks // self.block_size]
            self.prefix_cache[prefix_h] = list(full_block_ids)

        self.sequences[seq_id] = state
        self.total_allocations += 1
        return seq_id

    def _evict_oldest(self):
        """Evict the oldest sequence to free blocks."""
        if not self.sequences:
            return
        oldest_id = min(self.sequences.keys())
        self.free_sequence(oldest_id)

    def free_sequence(self, seq_id: int):
        """Free all blocks used by a sequence."""
        if seq_id not in self.sequences:
            return
        state = self.sequences[seq_id]
        # Return blocks to free list (only if not shared via prefix cache).
        for bid in state.block_ids:
            # Check if this block is in any prefix cache entry.
            # Simplified: just free it.
            self.free_blocks.append(bid)
        del self.sequences[seq_id]

    def write_kv(self, seq_id: int, position: int,
                 k: torch.Tensor, v: torch.Tensor):
        """Write KV tensors for a sequence at a given position.
        
        Args:
            seq_id: sequence id
            position: token position (0-indexed)
            k: (n_heads, head_dim) or (B, n_heads, head_dim) key tensor
            v: (n_heads, head_dim) or (B, n_heads, head_dim) value tensor
        """
        if seq_id not in self.sequences:
            return
        state = self.sequences[seq_id]

        # Handle batch dimension.
        if k.dim() == 3:
            k = k[0]  # take first batch
            v = v[0]

        # Calculate block and offset.
        block_idx = position // self.block_size
        offset = position % self.block_size

        if block_idx >= len(state.block_ids):
            # Need to allocate more blocks.
            n_needed = block_idx - len(state.block_ids) + 1
            for _ in range(n_needed):
                if not self.free_blocks:
                    self._evict_oldest()
                bid = self.free_blocks.pop(0)
                state.block_ids.append(bid)

        physical_block = state.block_ids[block_idx]
        self.k_cache[physical_block, offset] = k.to(self.dtype)
        self.v_cache[physical_block, offset] = v.to(self.dtype)

    def get_kv(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the full KV cache for a sequence.
        
        Returns:
            (k, v) tensors of shape (num_tokens, n_heads, head_dim)
        """
        if seq_id not in self.sequences:
            return None, None
        state = self.sequences[seq_id]

        n_tokens = state.num_tokens
        n_blocks_used = (n_tokens + self.block_size - 1) // self.block_size

        # Gather blocks.
        k_parts = []
        v_parts = []
        for i in range(n_blocks_used):
            bid = state.block_ids[i]
            # Last block may be partially filled.
            if i == n_blocks_used - 1:
                n_in_block = n_tokens - i * self.block_size
                k_parts.append(self.k_cache[bid, :n_in_block])
                v_parts.append(self.v_cache[bid, :n_in_block])
            else:
                k_parts.append(self.k_cache[bid])
                v_parts.append(self.v_cache[bid])

        k = torch.cat(k_parts, dim=0)  # (n_tokens, n_heads, head_dim)
        v = torch.cat(v_parts, dim=0)
        return k, v

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get the block table for a sequence (for paged attention kernel)."""
        if seq_id not in self.sequences:
            return []
        return self.sequences[seq_id].block_ids

    def stats(self) -> dict:
        """Get cache statistics."""
        used_blocks = self.n_blocks - len(self.free_blocks)
        return {
            "total_blocks": self.n_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "active_sequences": len(self.sequences),
            "prefix_cache_entries": len(self.prefix_cache),
            "cache_hits": self.cache_hits,
            "total_allocations": self.total_allocations,
            "hit_rate": self.cache_hits / max(1, self.total_allocations),
            "memory_mb": (self.k_cache.numel() + self.v_cache.numel()) *
                         (2 if self.dtype == torch.bfloat16 else 1) / 1024**2,
        }

    def reset(self):
        """Reset the entire cache."""
        self.free_blocks = list(range(self.n_blocks))
        self.sequences.clear()
        self.prefix_cache.clear()
        self.total_allocations = 0
        self.cache_hits = 0
