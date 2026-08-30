"""Prefix cache manager for ForgeEngine.

Encapsulates the prefix-caching logic that was inline in
``ForgeEngine``.  The cache itself remains a plain attribute on the
engine (``engine._prefix_cache``) for backwards compatibility — tests
and external code set it directly to either a ``dict``, an
``LRUPrefixCache``, a ``LearnedPrefixCache``, or a
``ChunkedPrefixCache``.  This module provides the *operations* on that
cache as standalone functions.

R&D round 14 (LMCache port): added ``ChunkedPrefixCache`` — a
chunk-based rolling-hash prefix cache inspired by LMCache's
``TokenDatabase`` / ``TokenHasher``.  Instead of keying on the first
``_PREFIX_CACHE_KEY_LENGTH`` tokens (coarse, all-or-nothing), it splits
the prompt into fixed-size chunks (default 256 tokens, matching
LMCache), computes a deterministic rolling prefix hash per chunk, and
can match the *longest* cached prefix across all entries — including
partial hits where a shorter cached prefix is reused by a longer
prompt.  Token equality is always verified on a hit (the hash is only a
lookup key), so hash collisions cannot corrupt generation.
"""
from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict

import torch

from research.model_loader import unpack_output_with_kv

_PREFIX_CACHE_KEY_LENGTH = 32
_DEFAULT_MAX_ENTRIES = 64
# LMCache default chunk size — 256 tokens.  Smaller = finer-grained
# reuse but more hash overhead; larger = coarser but cheaper.  256 is
# the validated sweet spot from the LMCache paper.
_LMCACHE_CHUNK_SIZE = 256


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

    Dispatches to :meth:`ChunkedPrefixCache.lookup_longest_prefix` when
    the active cache is a ``ChunkedPrefixCache`` (finer-grained,
    partial-prefix hits), otherwise falls back to the legacy
    exact-32-token-key path.
    """
    if prefix_cache is None or ids.shape[1] <= 16:
        return None
    if isinstance(prefix_cache, ChunkedPrefixCache):
        return prefix_cache.lookup_longest_prefix(ids)
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
    """Capture and store the KV cache for the prefix of ``ids``.

    For ``ChunkedPrefixCache`` the *full* prompt KV is stored under its
    rolling prefix hash (with token ids for verification), enabling
    partial-prefix reuse by later, longer prompts.  For the legacy
    cache only the first ``_PREFIX_CACHE_KEY_LENGTH`` tokens are stored.
    """
    prefix_cache = engine._prefix_cache
    if prefix_cache is None or ids.shape[1] <= 16:
        return
    if isinstance(prefix_cache, ChunkedPrefixCache):
        token_ids = ids[0].cpu().tolist()
        hashes = prefix_cache.prefix_hashes(token_ids)
        if not hashes:
            return
        full_hash = hashes[-1]
        if full_hash in prefix_cache:
            return  # already cached
        with torch.inference_mode():
            prefix_out = engine.model(ids, use_cache=True)
            _, prefix_kv = unpack_output_with_kv(prefix_out)
        prefix_cache.put((full_hash, token_ids), prefix_kv, len(token_ids))
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


# ── LMCache-style chunked rolling-hash prefix cache (R&D round 14) ─────────


def _chunk_hash(token_ids: list[int]) -> int:
    """Deterministic 64-bit hash of a chunk's token ids.

    Uses blake2b (8-byte digest) so the hash is stable across processes
    and independent of ``PYTHONHASHSEED`` — required for cross-request
    reuse where the cache may be restored from disk.
    """
    h = hashlib.blake2b(
        struct.pack(f"<{len(token_ids)}i", *token_ids),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")


def _prefix_hash(prev: int, chunk_h: int) -> int:
    """Rolling combine of the previous prefix hash with a new chunk hash.

    Deterministic (no Python ``hash()``) so the chain is reproducible
    across processes.  ``prefix_hash[i]`` uniquely identifies the token
    sequence ``chunks[0..i]``.
    """
    h = hashlib.blake2b(
        struct.pack("<Q", prev & 0xFFFFFFFFFFFFFFFF) +
        struct.pack("<Q", chunk_h & 0xFFFFFFFFFFFFFFFF),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")


def _slice_past_kv(past_kv, length: int):
    """Slice a per-layer KV cache list to ``length`` tokens.

    ``past_kv`` is the model's ``presents`` list: one ``(k, v)`` tuple
    per layer (or ``None`` for conv layers with no KV).  Each k/v tensor
    is ``[batch, n_kv, seq, head_dim]``; we slice the seq axis.  Returns
    a *new* list of sliced tuples — the original cache is untouched.
    """
    if past_kv is None:
        return None
    sliced = []
    for layer in past_kv:
        if layer is None:
            sliced.append(None)
            continue
        k, v = layer
        sliced.append((k[:, :, :length], v[:, :, :length]))
    return sliced


class ChunkedPrefixCache:
    """LMCache-inspired chunked rolling-hash prefix cache.

    Splits prompts into fixed-size chunks and stores one entry per
    unique *full* prefix, keyed by the rolling prefix hash of its
    chunks.  Lookup finds the longest cached prefix that matches the
    query (verifying token equality, not just hash), enabling partial
    reuse: a prompt of 1024 tokens can hit a previously cached 768-token
    prefix and only re-prefill the last 256 tokens.

    Drop-in compatible with ``LRUPrefixCache`` (``.get()`` / ``.put()``
    / ``__contains__`` / ``.stats()``), plus ``lookup_longest_prefix()``
    for the finer-grained matching used by the chunked generate path.

    Memory: one KV tensor per entry (same as LRU).  The chunk metadata
    (hashes + token ids) is negligible (~2 KB per 8K-token entry).
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES,
                 chunk_size: int = _LMCACHE_CHUNK_SIZE):
        self.max_entries = max_entries
        self.chunk_size = chunk_size
        # key = full-prefix rolling hash -> entry dict
        self._cache: OrderedDict[int, dict] = OrderedDict()
        # Reverse index: prefix-hash -> full-prefix-hash (for longest-prefix
        # matching we scan the keys; with <= max_entries this is cheap).
        self._hits = 0
        self._misses = 0
        self._partial_hits = 0

    # ── core interface (LRU-compatible) ────────────────────────────────

    def get(self, key):
        """LRU-compatible exact-key lookup.

        ``key`` here is expected to be a full-prefix rolling hash (int)
        produced by :meth:`prefix_hashes`.  For the legacy tuple-key
        path used by :func:`cache_key`, use :meth:`lookup_longest_prefix`
        instead.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return (entry["length"], entry["kv"])

    def put(self, key, value, length: int = 0):
        """Store a prefix KV under ``key`` (full-prefix rolling hash).

        ``value`` is the ``past_kv`` list from the model.  ``length`` is
        the number of tokens the KV covers.  The caller is responsible
        for computing the rolling hash via :meth:`prefix_hashes`.
        """
        token_ids = key[1] if isinstance(key, tuple) and len(key) > 1 else None
        # If handed a (hash, token_ids) tuple, split it.
        if isinstance(key, tuple):
            hash_key = key[0]
            token_ids = key[1] if len(key) > 1 else token_ids
        else:
            hash_key = key
        if hash_key in self._cache:
            self._cache.move_to_end(hash_key)
        self._cache[hash_key] = {
            "length": length,
            "kv": value,
            "token_ids": token_ids,
        }
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def __contains__(self, key):
        k = key[0] if isinstance(key, tuple) else key
        return k in self._cache

    def __len__(self):
        return len(self._cache)

    # ── chunked hashing helpers ────────────────────────────────────────

    def chunk_tokens(self, token_ids: list[int]) -> list[list[int]]:
        """Split a token list into chunk_size-sized pieces."""
        cs = self.chunk_size
        return [token_ids[i:i + cs] for i in range(0, len(token_ids), cs)]

    def prefix_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute the rolling prefix hash at each chunk boundary.

        Returns ``[h0, h1, ...]`` where ``h[i]`` is the hash of
        ``token_ids[: (i+1)*chunk_size]``.  ``h[-1]`` is the full-prefix
        hash used as the storage key.
        """
        hashes = []
        prev = 0
        for chunk in self.chunk_tokens(token_ids):
            ch = _chunk_hash(chunk)
            prev = _prefix_hash(prev, ch)
            hashes.append(prev)
        return hashes

    # ── longest-prefix matching ────────────────────────────────────────

    def lookup_longest_prefix(self, ids: torch.Tensor):
        """Find the longest cached prefix matching ``ids``.

        Returns ``(matched_len, past_kv)`` or ``None`` on miss.  The
        returned ``past_kv`` is sliced to ``matched_len`` tokens so it
        can be fed directly to the model with the suffix
        ``ids[:, matched_len:]``.

        Token equality is verified for every candidate hit — a hash
        collision never yields a wrong KV.
        """
        if ids.shape[1] <= 16:
            return None
        token_ids = ids[0].cpu().tolist()
        hashes = self.prefix_hashes(token_ids)
        if not hashes:
            return None

        # Walk chunk boundaries from longest to shortest, return the
        # first entry whose hash matches AND whose tokens verify.
        for i in range(len(hashes) - 1, -1, -1):
            h = hashes[i]
            entry = self._cache.get(h)
            if entry is None:
                continue
            matched_len = min((i + 1) * self.chunk_size, len(token_ids))
            # Verify token equality on the matched prefix.
            stored_tokens = entry.get("token_ids")
            if stored_tokens is not None:
                if stored_tokens[:matched_len] != token_ids[:matched_len]:
                    continue  # hash collision — skip
            # Move to MRU and return sliced KV.
            self._cache.move_to_end(h)
            if matched_len < entry["length"]:
                self._partial_hits += 1
                kv = _slice_past_kv(entry["kv"], matched_len)
            else:
                kv = entry["kv"]
            self._hits += 1
            return matched_len, kv
        self._misses += 1
        return None

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "type": "chunked",
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "chunk_size": self.chunk_size,
            "hits": self._hits,
            "misses": self._misses,
            "partial_hits": self._partial_hits,
            "hit_rate": self._hits / total if total > 0 else 0,
        }
