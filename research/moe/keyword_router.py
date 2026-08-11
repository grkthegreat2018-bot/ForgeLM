"""Shared keyword-matching router base class.

Both AirMoE routers (ExpertRouter in airmoe_infinite.py and TopicRouter in
airmoe_hotswap.py) implement the same keyword classify() logic. This module
consolidates that logic into a single base class so the scoring/threshold
behavior lives in exactly one place.

Scoring: a query is lowercased, then each topic scores 1 point per keyword
that appears as a substring of the query. Topics scoring at or above the
confidence threshold (min_score) are candidates; the highest score wins.
"""
from __future__ import annotations

from collections.abc import Iterable


class KeywordRouter:
    """Base class for keyword-matching topic routers.

    Args:
        keywords: mapping of topic name -> list of keyword strings.
        fallback: topic name returned when nothing scores above threshold.
        min_score: confidence threshold — minimum keyword hits required for
            a topic to be considered a match (default 1, i.e. any hit).
    """

    def __init__(self, keywords: dict[str, list[str]] | None = None,
                 fallback: str = "general", min_score: int = 1):
        self.keywords: dict[str, list[str]] = dict(keywords) if keywords else {}
        self.fallback = fallback
        self.min_score = max(1, min_score)

    def _iter_keywords(self) -> Iterable[tuple[str, list[str]]]:
        """Yield (topic, keywords) pairs. Subclasses may override to source
        keywords dynamically (e.g. from a mutable manifest or index)."""
        return self.keywords.items()

    def _topic_names(self) -> Iterable[str]:
        """Topic names for list_topics(). Subclasses may override."""
        return self.keywords.keys()

    def _score(self, query_lower: str) -> dict[str, int]:
        """Score all topics against an already-lowercased query.

        Only topics meeting the confidence threshold are included.
        """
        scores = {}
        for topic, kws in self._iter_keywords():
            score = sum(1 for kw in kws if kw in query_lower)
            if score >= self.min_score:
                scores[topic] = score
        return scores

    def classify(self, query: str) -> str:
        """Classify a query into the best-matching topic.

        Returns the topic with the highest keyword match score, or the
        fallback topic if nothing meets the confidence threshold.
        """
        scores = self._score(query.lower())
        if scores:
            return max(scores, key=scores.get)
        return self.fallback

    def classify_multi(self, query: str, top_n: int = 2) -> list[str]:
        """Classify a query into up to top_n topics, sorted by match score."""
        scores = self._score(query.lower())
        if not scores:
            return [self.fallback]
        sorted_topics = sorted(scores, key=scores.get, reverse=True)
        return sorted_topics[:top_n]

    def list_topics(self) -> list[str]:
        """List all known topic names (sorted)."""
        return sorted(self._topic_names())
