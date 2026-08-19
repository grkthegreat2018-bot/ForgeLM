"""Suffix decoding: training-free speculative decoding for repetitive outputs.

Based on "Suffix Decoding: A Model-Free Approach to Speculative Decoding"
and vLLM's suffix decoding implementation.

Key insight: many LLM workloads produce output that contains suffixes from
the prompt or from earlier in the generation. Examples:
  - Code completion: output repeats function signatures from the prompt
  - RAG: output quotes passages from retrieved context
  - Summarization: output reuses phrases from the input document
  - Multi-turn: output references earlier conversation turns

Suffix decoding maintains a suffix tree of all previously seen token sequences
(prompt + generated). When the current generation position matches a suffix,
the continuation is used as a speculative draft.

Compared to n-gram:
  - Suffix decoding matches LONGER sequences (not just fixed n-grams)
  - Better for long repetitive patterns (function bodies, boilerplate)
  - Composes with n-gram: use suffix for long matches, n-gram for short

vLLM reports suffix decoding achieves 1.5-2× speedup on code/RAG workloads
with zero training cost and zero extra memory.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch


class SuffixTreeNode:
    """Node in a suffix tree for suffix decoding."""

    __slots__ = ("children", "token", "depth", "is_end")

    def __init__(self, token: int = -1, depth: int = 0):
        self.children: dict[int, SuffixTreeNode] = {}
        self.token = token
        self.depth = depth
        self.is_end = False


class SuffixTree:
    """Compact suffix tree for token sequence matching.

    Stores all suffixes of the token sequence seen so far. When looking up
    a continuation, traverses the tree from the root following the current
    token sequence and returns the longest matching continuation.

    Memory: O(n²) worst case, but pruned to last max_tokens tokens.
    """

    def __init__(self, max_tokens: int = 4096, max_depth: int = 32):
        self.root = SuffixTreeNode()
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self._tokens: list[int] = []
        self._total_inserted = 0

    def insert(self, tokens: list[int]):
        """Insert all suffixes of the token sequence into the tree."""
        self._tokens.extend(tokens)
        if len(self._tokens) > self.max_tokens:
            # Rebuild from recent tokens only
            recent = self._tokens[-self.max_tokens:]
            self._tokens = recent
            self.root = SuffixTreeNode()
            self._rebuild(recent)

        # Insert all suffixes of the new tokens
        start_idx = max(0, len(self._tokens) - len(tokens) - self.max_depth)
        for i in range(start_idx, len(self._tokens)):
            self._insert_suffix(self._tokens[i:], self.max_depth)

    def _insert_suffix(self, suffix: list[int], max_depth: int):
        node = self.root
        for i, tok in enumerate(suffix):
            if i >= max_depth:
                break
            if tok not in node.children:
                node.children[tok] = SuffixTreeNode(token=tok, depth=i + 1)
            node = node.children[tok]
        node.is_end = True

    def _rebuild(self, tokens: list[int]):
        """Rebuild the tree from a token list."""
        for i in range(len(tokens)):
            self._insert_suffix(tokens[i:], self.max_depth)

    def lookup(self, tokens: list[int], max_draft_len: int = 8) -> Optional[list[int]]:
        """Find the longest matching continuation.

        Args:
            tokens: current token sequence (recent context)
            max_draft_len: maximum draft tokens to return

        Returns:
            continuation: list of draft tokens, or None if no match
        """
        if not tokens:
            return None

        # Try matching from the last token backwards
        node = self.root
        matched = 0
        for tok in tokens[-self.max_depth:]:
            if tok in node.children:
                node = node.children[tok]
                matched += 1
            else:
                break

        if matched == 0:
            return None

        # Extract continuation from the deepest matched node
        continuation = []
        current = node
        while len(continuation) < max_draft_len:
            if not current.children:
                break
            # Pick the most recently inserted child (highest recency)
            # For simplicity, pick any child (first one)
            best_child = next(iter(current.children.values()))
            continuation.append(best_child.token)
            current = best_child

        return continuation if continuation else None

    def clear(self):
        self.root = SuffixTreeNode()
        self._tokens.clear()
        self._total_inserted = 0


class SuffixDecoder:
    """Suffix decoding speculative decoder.

    Maintains a suffix tree of prompt + generated tokens. At each decode step,
    looks up the current context in the suffix tree and uses the longest
    matching continuation as a speculative draft.

    Composes with n-gram and EAGLE-3: suffix decoding is tried first (longest
    match), then n-gram (shorter matches), then EAGLE-3 (novel generation).
    """

    def __init__(self, max_tree_tokens: int = 4096, max_depth: int = 32,
                 max_draft_len: int = 8):
        self.suffix_tree = SuffixTree(max_tokens=max_tree_tokens,
                                       max_depth=max_depth)
        self.max_draft_len = max_draft_len
        self._attempts = 0
        self._hits = 0

    def update(self, tokens: list[int]):
        """Update the suffix tree with new tokens."""
        self.suffix_tree.insert(tokens)

    def draft(self, tokens: list[int]) -> tuple[list[int], str]:
        """Generate a speculative draft using suffix matching.

        Args:
            tokens: recently generated token IDs

        Returns:
            (draft_tokens, drafter_name)
        """
        self._attempts += 1
        draft = self.suffix_tree.lookup(tokens, max_draft_len=self.max_draft_len)
        if draft and len(draft) > 0:
            self._hits += 1
            return draft, "suffix"
        return [], "none"

    def record_result(self, drafter: str, n_accepted: int, n_draft: int):
        """Record the result of a speculative step."""
        if drafter == "suffix" and n_accepted > 0:
            self._hits += 1

    def hit_rate(self) -> float:
        return self._hits / max(self._attempts, 1)

    def stats(self) -> dict:
        return {
            "drafter": "suffix",
            "attempts": self._attempts,
            "hits": self._hits,
            "hit_rate": self.hit_rate(),
            "tree_tokens": len(self.suffix_tree._tokens),
        }


class ComboSpeculativeDecoder:
    """Combines suffix decoding + n-gram + EAGLE-3 for maximum coverage.

    Tries each drafter in order of cost (cheapest first):
      1. Suffix decoding (free, longest matches, best for code/RAG)
      2. N-gram lookup (free, shorter matches, best for repeated phrases)
      3. EAGLE-3/MTP (model-based, novel generation)

    Picks the longest draft from any drafter.
    """

    def __init__(self, eagle_head=None, mtp_module=None,
                 n_gram_size: int = 3, max_draft_len: int = 8):
        from research.decoding.adaptive_speculative import (
            AdaptiveSpeculativeDecoder, NGramCache)
        self.suffix_decoder = SuffixDecoder(max_draft_len=max_draft_len)
        self.ngram_cache = NGramCache(n=n_gram_size, max_draft_len=max_draft_len)
        self.eagle_head = eagle_head
        self.mtp_module = mtp_module
        self.max_draft_len = max_draft_len

    def update(self, tokens: list[int]):
        """Update all drafter caches."""
        self.suffix_decoder.update(tokens)
        self.ngram_cache.update(tokens)

    def draft(self, tokens: list[int],
              hidden_state: torch.Tensor | None = None) -> tuple[list[int], str]:
        """Generate the best speculative draft from all drafters."""
        best_draft = []
        best_drafter = "none"

        # 1. Suffix decoding (longest matches, free)
        suffix_draft = self.suffix_decoder.suffix_tree.lookup(
            tokens, max_draft_len=self.max_draft_len)
        if suffix_draft and len(suffix_draft) > len(best_draft):
            best_draft = suffix_draft
            best_drafter = "suffix"

        # 2. N-gram lookup (shorter matches, free)
        ngram_draft = self.ngram_cache.lookup(tokens)
        if ngram_draft and len(ngram_draft) > len(best_draft):
            best_draft = ngram_draft
            best_drafter = "ngram"

        # 3. EAGLE-3 / MTP (model-based, novel generation)
        if hidden_state is not None:
            if self.eagle_head is not None:
                try:
                    with torch.inference_mode():
                        eagle_draft = self.eagle_head.predict(hidden_state)
                    eagle_draft = eagle_draft.tolist() if hasattr(eagle_draft, 'tolist') else list(eagle_draft)
                    if eagle_draft and len(eagle_draft) > len(best_draft):
                        best_draft = eagle_draft
                        best_drafter = "eagle"
                except Exception:
                    pass
            elif self.mtp_module is not None:
                try:
                    with torch.inference_mode():
                        mtp_draft = self.mtp_module.predict(hidden_state)
                    mtp_draft = mtp_draft.tolist() if hasattr(mtp_draft, 'tolist') else list(mtp_draft)
                    if mtp_draft and len(mtp_draft) > len(best_draft):
                        best_draft = mtp_draft
                        best_drafter = "mtp"
                except Exception:
                    pass

        return best_draft, best_drafter

    def stats(self) -> dict:
        return {
            "suffix": self.suffix_decoder.stats(),
            "ngram_hit_rate": self.ngram_cache.hit_rate(),
        }
