"""Forward Cache — cache forward pass results for repeated inputs (C2).

In self-play and curriculum learning, the same prompts are seen multiple
times (especially prompts the model fails on repeatedly). Recomputing the
forward pass for an identical input is pure waste.

This module provides a simple LRU cache keyed on input token IDs.
On cache hit: return cached logits/hidden states (zero compute).
On cache miss: compute forward pass, store result.

Research basis: SYSTEMS_IDEATION.md C2 — "Forward Caching"
  - 20-40% of prompts repeat in self-play → 20-40% fewer forward passes
  - Composes with self_play, infinite_curriculum, replay buffers

Usage:
    from research.runtime.forward_cache import ForwardCache
    cache = ForwardCache(max_entries=1000)
    logits, hidden = cache.forward(model, input_ids)
"""
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch


class ForwardCache:
    """LRU cache for forward pass results.

    Caches (logits, hidden_states) keyed on input token ID hash.
    On hit: returns cached result (zero compute).
    On miss: runs forward pass, stores result.

    Memory: each entry is ~2 * seq_len * vocab_size * 2 bytes (bf16 logits).
    For seq_len=128, vocab=152K: ~78KB per entry. 1000 entries = ~78MB.
    """

    def __init__(self, max_entries: int = 1000, device: str = "cuda"):
        self.max_entries = max_entries
        self.device = device
        self._cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, input_ids: torch.Tensor) -> int:
        """Hash the input token IDs for cache key."""
        tokens = input_ids[0]
        L = tokens.shape[0]
        powers = torch.arange(L, device=tokens.device, dtype=torch.int64)
        powers = torch.pow(torch.tensor(31, device=tokens.device, dtype=torch.int64), powers)
        h = torch.sum(tokens.to(torch.int64) * powers).item()
        return h

    def forward(self, model, input_ids: torch.Tensor,
                use_cache: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with caching.

        Args:
            model: the LLM
            input_ids: (1, seq_len) token IDs
            use_cache: whether to also return KV cache (not cached — only logits/hidden)

        Returns:
            (logits, hidden_or_none)
        """
        key = self._key(input_ids)

        if key in self._cache:
            self._hits += 1
            cached_logits, cached_hidden = self._cache[key]
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            # Return clones to avoid mutation
            return cached_logits.clone(), (
                cached_hidden.clone() if cached_hidden is not None else None)

        # Cache miss — compute forward pass
        self._misses += 1
        with torch.no_grad():
            if use_cache:
                result = model(input_ids, use_cache=True)
                logits = result[0]
                hidden = result[1] if len(result) > 1 else None
                # Don't cache KV cache (it's stateful and grows)
            else:
                result = model(input_ids)
                logits = result[0]
                hidden = result[1] if len(result) > 1 else None

        # Store in cache
        self._cache[key] = (logits.detach().clone(),
                            hidden.detach().clone() if hidden is not None else None)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        return logits, hidden

    def invalidate(self):
        """Clear the entire cache."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cache_size": self.size,
            "max_entries": self.max_entries,
        }

    def print_stats(self):
        s = self.stats()
        print(f"  [ForwardCache] hits={s['hits']} misses={s['misses']} "
              f"hit_rate={s['hit_rate']:.1%} size={s['cache_size']}/{s['max_entries']}")
