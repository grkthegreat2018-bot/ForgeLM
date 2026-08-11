"""Tests for research.moe.keyword_router and its AirMoE subclasses.

Covers the shared KeywordRouter base (hits, misses, threshold) and verifies
the ExpertRouter/TopicRouter subclass APIs stayed compatible after the
consolidation.
"""

import json

import pytest

from research.moe.airmoe_hotswap import TopicRouter
from research.moe.airmoe_infinite import ExpertRouter
from research.moe.keyword_router import KeywordRouter

# ── KeywordRouter base ───────────────────────────────────────────────────────

KEYWORDS = {
    "python_algorithms": ["sort", "search", "recursive", "tree", "graph"],
    "math_geometry": ["triangle", "circle", "angle", "area"],
    "science_physics": ["physics", "force", "energy", "quantum"],
}


class TestKeywordRouter:
    def test_classify_hit(self):
        router = KeywordRouter(KEYWORDS)
        assert router.classify("Sort this list with a graph traversal") == "python_algorithms"

    def test_classify_miss_returns_fallback(self):
        router = KeywordRouter(KEYWORDS)
        assert router.classify("tell me a bedtime story") == "general"

    def test_classify_custom_fallback(self):
        router = KeywordRouter(KEYWORDS, fallback="misc")
        assert router.classify("no keywords here") == "misc"

    def test_classify_case_insensitive(self):
        router = KeywordRouter(KEYWORDS)
        assert router.classify("QUANTUM physics") == "science_physics"

    def test_classify_picks_highest_score(self):
        router = KeywordRouter(KEYWORDS)
        # "sort" + "search" (2 hits) beats "angle" (1 hit)
        assert router.classify("sort and search at an angle") == "python_algorithms"

    def test_threshold_blocks_low_score(self):
        router = KeywordRouter(KEYWORDS, min_score=2)
        # Only one keyword hit → below threshold → fallback
        assert router.classify("calculate the area") == "general"
        # Two hits → accepted
        assert router.classify("triangle area") == "math_geometry"

    def test_threshold_clamped_to_at_least_one(self):
        router = KeywordRouter(KEYWORDS, min_score=0)
        assert router.min_score == 1

    def test_classify_multi(self):
        router = KeywordRouter(KEYWORDS)
        topics = router.classify_multi("sort a graph near a circle", top_n=2)
        assert topics[0] == "python_algorithms"
        assert len(topics) == 2

    def test_classify_multi_miss(self):
        router = KeywordRouter(KEYWORDS)
        assert router.classify_multi("nothing matches") == ["general"]

    def test_list_topics(self):
        router = KeywordRouter(KEYWORDS)
        assert router.list_topics() == sorted(KEYWORDS.keys())

    def test_empty_keywords(self):
        router = KeywordRouter()
        assert router.classify("anything") == "general"
        assert router.list_topics() == []


# ── ExpertRouter (airmoe_infinite) API compat ────────────────────────────────


@pytest.fixture()
def manifest_file(tmp_path):
    manifest = {
        "n_layers": 4,
        "topics": {
            "python_algorithms": {
                "label": "Python Algorithms",
                "keywords": ["sort", "search", "recursive"],
                "subtopics": [],
            },
            "math_geometry": {
                "label": "Geometry",
                "keywords": ["triangle", "circle", "area"],
                "subtopics": [],
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


class TestExpertRouterCompat:
    def test_loads_manifest(self, manifest_file):
        router = ExpertRouter(manifest_file)
        assert router.n_layers == 4
        assert router.list_topics() == ["math_geometry", "python_algorithms"]

    def test_classify_hit(self, manifest_file):
        router = ExpertRouter(manifest_file)
        assert router.classify("sort an array recursively") == "python_algorithms"

    def test_classify_miss_falls_back_to_general(self, manifest_file):
        router = ExpertRouter(manifest_file)
        assert router.classify("unrelated query") == "general"

    def test_classify_multi(self, manifest_file):
        router = ExpertRouter(manifest_file)
        topics = router.classify_multi("sort the triangle vertices", top_n=2)
        assert set(topics) == {"python_algorithms", "math_geometry"}

    def test_classify_multi_miss(self, manifest_file):
        router = ExpertRouter(manifest_file)
        assert router.classify_multi("no match at all") == ["general"]

    def test_register_topic(self, manifest_file):
        router = ExpertRouter(manifest_file)
        router.register_topic("cooking", ["recipe", "bake"], label="Cooking")
        assert router.classify("bake a recipe") == "cooking"
        assert "cooking" in router.list_topics()

    def test_is_keyword_router_subclass(self, manifest_file):
        assert isinstance(ExpertRouter(manifest_file), KeywordRouter)


# ── TopicRouter (airmoe_hotswap) API compat ──────────────────────────────────


@pytest.fixture()
def index_file(tmp_path):
    index = {"topics": {"python_math": {}, "math_algebra": {}}}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return str(path)


class TestTopicRouterCompat:
    def test_classify_hit_available_topic(self, index_file):
        router = TopicRouter(index_file)
        # "prime" is a python_math keyword and python_math has a module
        assert router.classify("Check if 17 is prime") == "python_math"

    def test_classify_skips_unavailable_topics(self, index_file):
        router = TopicRouter(index_file)
        # "circle" belongs to math_geometry, which has no module in the index
        assert router.classify("area of a circle") != "math_geometry"

    def test_classify_fallback_first_available(self, index_file):
        router = TopicRouter(index_file)
        # No keyword match → first available topic (not "general")
        topic = router.classify("completely unrelated query zzz")
        assert topic in {"python_math", "math_algebra"}

    def test_classify_empty_index_returns_general(self, tmp_path):
        path = tmp_path / "index.json"
        path.write_text(json.dumps({"topics": {}}), encoding="utf-8")
        router = TopicRouter(str(path))
        assert router.classify("prime numbers") == "general"

    def test_list_topics(self, index_file):
        router = TopicRouter(index_file)
        assert router.list_topics() == ["math_algebra", "python_math"]

    def test_is_keyword_router_subclass(self, index_file):
        assert isinstance(TopicRouter(index_file), KeywordRouter)
