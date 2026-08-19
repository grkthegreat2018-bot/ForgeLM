"""ToaST: Tokenization with Split Trees.

Based on "Tokenization with Split Trees" (arXiv 2605.22705).

Key insight: BPE greedily merges tokens, which is suboptimal. ToaST:
  1. Greedily splits each pre-token into a full binary tree using
     precomputed byte n-gram counts (independent of vocabulary)
  2. Given a vocabulary, inference recursively descends each split tree
     and emits the first in-vocabulary node reached on each path
  3. Vocabulary selection formulated as Integer Program (IP) that
     minimizes total token count under this inference procedure
  4. LP relaxation is near-integral → provably near-optimal vocabularies

Results: 11% fewer tokens than BPE/WordPiece/UnigramLM at vocab ≥ 40K.
1.5B models with ToaST achieve highest CORE score (+2.6-7.6% over baselines).
Uses common single-byte tokens less frequently → better Rényi efficiency.

For our model (vocab=65536):
  - ToaST: ~11% fewer tokens → ~11% faster inference, longer effective context
  - Or: same token count with 11% more content per sequence
  - Requires retraining (different tokenization → different input distribution)
"""
from __future__ import annotations

from collections import Counter
from typing import Optional


class SplitTree:
    """A binary split tree for a single pre-token.

    Greedily splits the pre-token into a binary tree where each node
    represents a substring. The tree is built top-down: at each node,
    split at the position that maximizes the product of n-gram counts
    of the two children.
    """

    def __init__(self, text: str, ngram_counts: Counter,
                 max_depth: int = 20):
        self.text = text
        self.ngram_counts = ngram_counts
        self.max_depth = max_depth
        self.root = self._build_tree(text, 0)

    def _build_tree(self, text: str, depth: int) -> dict:
        """Recursively build the split tree."""
        node = {'text': text, 'children': None}

        if depth >= self.max_depth or len(text) <= 1:
            return node

        # Find the best split position
        best_split = self._find_best_split(text)
        if best_split is None:
            return node

        left_text = text[:best_split]
        right_text = text[best_split:]

        node['children'] = {
            'left': self._build_tree(left_text, depth + 1),
            'right': self._build_tree(right_text, depth + 1),
        }
        return node

    def _find_best_split(self, text: str) -> Optional[int]:
        """Find the split position that maximizes count product."""
        if len(text) <= 1:
            return None

        best_score = -1
        best_pos = None

        for i in range(1, len(text)):
            left = text[:i]
            right = text[i:]
            left_count = self.ngram_counts.get(left, 0)
            right_count = self.ngram_counts.get(right, 0)
            score = left_count * right_count

            if score > best_score:
                best_score = score
                best_pos = i

        # Only split if it improves (both children have some count)
        if best_score <= 0:
            return None
        return best_pos

    def tokenize(self, vocab: set[str]) -> list[str]:
        """Tokenize using the split tree and a vocabulary.

        Recursively descends the tree and emits the first in-vocabulary
        node reached on each path.
        """
        return self._tokenize_node(self.root, vocab)

    def _tokenize_node(self, node: dict, vocab: set[str]) -> list[str]:
        """Recursively tokenize a tree node."""
        text = node['text']

        # If this node's text is in the vocabulary, emit it
        if text in vocab:
            return [text]

        # If no children, emit as bytes (fallback)
        if node['children'] is None:
            # Fallback: emit individual bytes
            return [f'<0x{ord(c):02X}>' for c in text]

        # Descend into children
        children = node['children']
        left_tokens = self._tokenize_node(children['left'], vocab)
        right_tokens = self._tokenize_node(children['right'], vocab)
        return left_tokens + right_tokens


class ToaSTTokenizer:
    """ToaST: Tokenization with Split Trees.

    1. Precompute byte n-gram counts from corpus
    2. Build split trees for each pre-token
    3. Select vocabulary via Integer Program (LP relaxation)
    4. Tokenize by descending split trees with the selected vocabulary
    """

    def __init__(self, vocab_size: int = 65536,
                 max_ngram: int = 8, max_depth: int = 20):
        self.vocab_size = vocab_size
        self.max_ngram = max_ngram
        self.max_depth = max_depth
        self.ngram_counts: Counter = Counter()
        self.vocab: set[str] = set()
        self.vocab_ids: dict[str, int] = {}
        self._split_trees: dict[str, SplitTree] = {}

    def train(self, corpus: list[str]):
        """Train the ToaST tokenizer.

        Args:
            corpus: list of training texts
        """
        # Step 1: Compute byte n-gram counts
        self._compute_ngram_counts(corpus)

        # Step 2: Build split trees for all pre-tokens
        pretokens = set()
        for text in corpus:
            # Simple pre-tokenization: split on whitespace
            for word in text.split():
                pretokens.add(word)

        for word in pretokens:
            self._split_trees[word] = SplitTree(
                word, self.ngram_counts, self.max_depth)

        # Step 3: Select vocabulary via LP relaxation
        self._select_vocabulary(corpus)

        print(f"  [ToaST] Trained: vocab={len(self.vocab)}, "
              f"trees={len(self._split_trees)}, "
              f"ngrams={len(self.ngram_counts)}")

    def _compute_ngram_counts(self, corpus: list[str]):
        """Compute byte n-gram counts from corpus."""
        for text in corpus:
            for n in range(1, self.max_ngram + 1):
                for i in range(len(text) - n + 1):
                    ngram = text[i:i + n]
                    self.ngram_counts[ngram] += 1

    def _select_vocabulary(self, corpus: list[str]):
        """Select vocabulary via Integer Program (LP relaxation).

        The IP minimizes total token count over all split trees.
        LP relaxation is near-integral in practice.

        Simplified: greedy selection based on coverage.
        """
        # Collect all candidate tokens from split trees
        candidates = Counter()
        for tree in self._split_trees.values():
            self._collect_candidates(tree.root, candidates)

        # Greedy: select tokens that reduce total token count the most
        # (In practice, this would be an LP solver)
        sorted_candidates = candidates.most_common(self.vocab_size)

        self.vocab = {token for token, _ in sorted_candidates}
        self.vocab_ids = {token: idx for idx, (token, _) in enumerate(sorted_candidates)}

        # Add byte-level fallback tokens
        for i in range(256):
            byte_token = f'<0x{i:02X}>'
            if byte_token not in self.vocab_ids:
                self.vocab_ids[byte_token] = len(self.vocab_ids)
                self.vocab.add(byte_token)

    def _collect_candidates(self, node: dict, candidates: Counter):
        """Collect all candidate tokens from a split tree."""
        candidates[node['text']] += 1
        if node['children'] is not None:
            self._collect_candidates(node['children']['left'], candidates)
            self._collect_candidates(node['children']['right'], candidates)

    def encode(self, text: str) -> list[int]:
        """Encode text using ToaST."""
        ids = []
        for word in text.split():
            if word in self._split_trees:
                tokens = self._split_trees[word].tokenize(self.vocab)
            else:
                # Build a new tree on the fly
                tree = SplitTree(word, self.ngram_counts, self.max_depth)
                tokens = tree.tokenize(self.vocab)

            for token in tokens:
                if token in self.vocab_ids:
                    ids.append(self.vocab_ids[token])
                else:
                    # Byte fallback
                    for c in token:
                        byte_token = f'<0x{ord(c):02X}>'
                        ids.append(self.vocab_ids.get(byte_token, 0))
        return ids

    def stats(self) -> dict:
        return {
            "vocab_size": len(self.vocab_ids),
            "n_split_trees": len(self._split_trees),
            "n_ngrams": len(self.ngram_counts),
            "max_ngram": self.max_ngram,
        }
