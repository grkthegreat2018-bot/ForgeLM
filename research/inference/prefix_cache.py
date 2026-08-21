"""Prefix cache manager for ForgeEngine.

Encapsulates the prefix-caching logic that was inline in
``ForgeEngine``.  The cache itself remains a plain attribute on the
engine (``engine._prefix_cache``) for backwards compatibility — tests
and external code set it directly to either a ``dict``, an
``LRUPrefixCache``, or a ``LearnedPrefixCache``.  This module provides
the *operations* on that cache as standalone functions.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from research.model_loader import unpack_output_with_kv

_PREFIX_CACHE_KEY_LENGTH = 32
_DEFAULT_MAX_ENTRIES = 64


class LRUPrefixCache:
    """Bounded LRU cache for prefix KV states.

    Replaces the unbounded ``dict`` that was used as the default prefix
    cache.  Prevents OOM in long-running serving scenarios by evicting
    the least-recently-used entries when the cache is full.

    Drop-in compatible with the old dict interface (``.get()``, ``.put()``,
    ``in``), plus ``.stats()`` for diagnostics.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES):
        self._cache: OrderedDict = OrderedDict()
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, key):
        if key not in self._cache:
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def put(self, key, value, length: int = 0):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        # Evict oldest entries if over capacity
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


def cache_key(ids: torch.Tensor) -> tuple[int, ...]:
    """Compute a hashable cache key from the first ``_PREFIX_CACHE_KEY_LENGTH``
    tokens of ``ids``."""
    key_len = min(_PREFIX_CACHE_KEY_LENGTH, ids.shape[1])
    return tuple(ids[0, :key_len].cpu().tolist())


def get_cached_prefix(prefix_cache, ids: torch.Tensor):
    """Look up a cached prefix KV for ``ids``.

    Returns ``(cached_len, past_kv)`` or ``None`` on miss / when the
    cache is disabled or the prompt is too short.
    """
    if prefix_cache is None or ids.shape[1] <= 16:
        return None
    cached = prefix_cache.get(cache_key(ids))
    if cached is None:
        return None
    if hasattr(cached, "kv_cache"):
        return cached.length, cached.kv_cache
    cached_prefix, cached_past_kv = cached
    cached_len = (
        cached_prefix if isinstance(cached_prefix, int)
        else cached_prefix.shape[1]
    )
    return cached_len, cached_past_kv


def cache_prompt_prefix(engine, ids: torch.Tensor) -> None:
    """Capture and store the KV cache for the prefix of ``ids``."""
    prefix_cache = engine._prefix_cache
    if prefix_cache is None or ids.shape[1] <= 16:
        return
    key = cache_key(ids)
    if not hasattr(prefix_cache, "put") and key in prefix_cache:
        return
    key_len = min(_PREFIX_CACHE_KEY_LENGTH, ids.shape[1])
    with torch.inference_mode():
        prefix_out = engine.model(ids[:, :key_len], use_cache=True)
        _, prefix_kv = unpack_output_with_kv(prefix_out)
    if hasattr(prefix_cache, "put"):
        prefix_cache.put(key, prefix_kv, key_len)
    else:
        prefix_cache[key] = (key_len, prefix_kv)


def generate_from_prefix_cache(
    engine, ids, max_new_tokens, temperature, top_p, top_k,
    repetition_penalty,
):
    """Fast path: if the prefix is cached, reuse its KV and skip prefill.

    Returns the full ``output_ids`` tensor on hit, or ``None`` on miss
    (caller falls back to normal decoding).
    """
    cached = get_cached_prefix(engine._prefix_cache, ids)
    if cached is None:
        return None
    cached_len, cached_past_kv = cached
    suffix_ids = ids[:, cached_len:]
    if suffix_ids.shape[1] > 0 and cached_past_kv is not None:
        with torch.inference_mode():
            out = engine.model(
                suffix_ids, past_key_values=cached_past_kv, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)
        output_ids = engine._decode_with_kv(
            ids, logits, past_kv, max_new_tokens, temperature, top_p,
            top_k=top_k, repetition_penalty=repetition_penalty)
        print(f"  [PrefixCache] HIT + REUSE (prefix len={cached_len}, "
              f"saved prefill)")
        return output_ids
    print(f"  [PrefixCache] HIT (prefix len={cached_len})")
    return None
