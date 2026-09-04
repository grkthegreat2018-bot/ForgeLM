"""Multi-level data quality pipeline for self-play training data.

Research basis:
- ResoFilter (NAACL 2025): data-parameter resonance; 50% less data yields comparable results.
- Multi-level deduplication: exact (n-gram overlap >90%), semantic (embedding cosine >0.85),
  AST structural (Dice coefficient >0.85).
- QDC Framework: score = 0.5*Quality + 0.3*Diversity + 0.2*Complexity.
- Difficulty filter: keep tasks with success rate in [0.3, 0.7] (Goldilocks zone).
- Diversity score: target >0.7 via embedding pairwise distance.
- Contamination detection: hierarchical (token-level Min-K% Prob, semantic clustering).
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import Counter
from typing import Optional

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: list[str], n: int = 5) -> set[tuple]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else {tuple(tokens)}


def _tfidf_vectors(samples: list[dict]) -> np.ndarray:
    """Build a simple TF-IDF bag-of-words matrix (numpy only)."""
    docs = [_tokenize(s["prompt"] + " " + s.get("solution", "")) for s in samples]
    vocab = sorted({w for d in docs for w in d})
    if not vocab:
        return np.zeros((len(samples), 0))
    idx = {w: i for i, w in enumerate(vocab)}
    n_docs = len(docs)
    df = np.zeros(len(vocab))
    for d in docs:
        for w in set(d):
            df[idx[w]] += 1
    idf = np.log((1 + n_docs) / (1 + df)) + 1
    mat = np.zeros((n_docs, len(vocab)))
    for r, d in enumerate(docs):
        c = Counter(d)
        for w, cnt in c.items():
            mat[r, idx[w]] = cnt
    mat = mat * idf[np.newaxis, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


def _ast_signature(code: str) -> Counter:
    """Return a Counter of AST node type names as a structural fingerprint."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()
    return Counter(type(n).__name__ for n in ast.walk(tree))


def _dice(a: Counter, b: Counter) -> float:
    """Sørensen–Dice coefficient over two Counters."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    return 2.0 * inter / (sum(a.values()) + sum(b.values()))


class DataQualityPipeline:
    """Orchestrates multi-level dedup, difficulty filtering, and QDC scoring."""

    # --- Deduplication stages ---

    def dedup_exact(self, samples: list[dict]) -> list[dict]:
        """Remove near-duplicates with n-gram overlap >90% (exact/lexical dedup)."""
        seen_sigs: list[set[tuple]] = []
        out: list[dict] = []
        for s in samples:
            toks = _tokenize(s["prompt"] + " " + s.get("solution", ""))
            ng = _ngrams(toks)
            dup = False
            for sig in seen_sigs:
                if not ng or not sig:
                    continue
                inter = len(ng & sig)
                union = len(ng | sig)
                if union and inter / union > 0.90:
                    dup = True
                    break
            if not dup:
                seen_sigs.append(ng)
                out.append(s)
        return out

    def dedup_semantic(self, samples: list[dict], threshold: float = 0.85) -> list[dict]:
        """Remove semantic duplicates via TF-IDF cosine similarity > threshold."""
        if len(samples) <= 1:
            return list(samples)
        vecs = _tfidf_vectors(samples)
        keep = [0]
        for i in range(1, len(samples)):
            dup = False
            for j in keep:
                cos = float(np.dot(vecs[i], vecs[j]))
                if cos > threshold:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        return [samples[i] for i in keep]

    def dedup_ast(self, samples: list[dict], threshold: float = 0.85) -> list[dict]:
        """Remove structural duplicates via AST node-type Dice coefficient > threshold."""
        sigs = [_ast_signature(s.get("solution", "")) for s in samples]
        keep = [0]
        for i in range(1, len(samples)):
            dup = False
            for j in keep:
                if _dice(sigs[i], sigs[j]) > threshold:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        return [samples[i] for i in keep]

    # --- Filtering & scoring ---

    def filter_difficulty(
        self, samples: list[dict], success_rates: list[float], low: float = 0.3, high: float = 0.7
    ) -> list[dict]:
        """Keep samples in the Goldilocks zone: success rate in [low, high]."""
        if len(success_rates) != len(samples):
            return list(samples)
        return [s for s, sr in zip(samples, success_rates) if low <= sr <= high]

    def compute_diversity_score(self, samples: list[dict]) -> float:
        """Mean pairwise cosine distance over TF-IDF vectors (target >0.7)."""
        if len(samples) <= 1:
            return 0.0
        vecs = _tfidf_vectors(samples)
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, 0.0)
        n_pairs = len(samples) * (len(samples) - 1)
        return float(1.0 - sim.sum() / n_pairs) if n_pairs else 0.0

    def compute_qdc_score(self, sample: dict) -> float:
        """QDC Framework: 0.5*Quality + 0.3*Diversity + 0.2*Complexity."""
        quality = float(sample.get("quality", 0.0))
        sol = sample.get("solution", "")
        complexity = min(1.0, math.log1p(len(_tokenize(sol))) / 6.0) if sol else 0.0
        diversity = 1.0 if sample.get("embedding") is not None else 0.5
        return 0.5 * quality + 0.3 * diversity + 0.2 * complexity

    # --- Full pipeline ---

    def run_pipeline(
        self, samples: list[dict], success_rates: Optional[list[float]] = None
    ) -> tuple[list[dict], dict]:
        """Run all stages; return filtered samples and a stats dict."""
        stats: dict = {"n_input": len(samples)}
        s = self.dedup_exact(samples)
        stats["n_after_exact"] = len(s)
        s = self.dedup_semantic(s)
        stats["n_after_semantic"] = len(s)
        s = self.dedup_ast(s)
        stats["n_after_ast"] = len(s)
        if success_rates is not None:
            s = self.filter_difficulty(s, success_rates)
        stats["n_after_difficulty"] = len(s)
        stats["diversity_score"] = self.compute_diversity_score(s)
        qdc = [self.compute_qdc_score(x) for x in s]
        stats["mean_qdc"] = float(np.mean(qdc)) if qdc else 0.0
        return s, stats
