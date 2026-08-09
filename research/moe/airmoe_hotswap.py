"""AirMoE Hotswap Loader — loads topic modules on demand from disk.

The router classifies the user query into a topic, then the loader:
  1. Checks if the topic module is already in VRAM (LRU cache)
  2. If not, loads it from disk
  3. Injects the knowledge pack (KV cache or context patch)
  4. Evicts least-recently-used modules if VRAM is full

Each topic module is a self-contained directory:
  airmoe_modules/<topic>/
    module.json     — metadata
    knowledge.json  — knowledge chunks
    knowledge.txt   — concatenated knowledge for injection
    samples.jsonl   — samples for self-play

Usage:
    from research.moe.airmoe_hotswap import AirMoEHotswapLoader, TopicRouter
    router = TopicRouter("research/checkpoints/airmoe_modules/index.json")
    loader = AirMoEHotswapLoader(model, tokenizer, cache_size=3)

    topic = router.classify("Check if 17 is prime")
    loader.load_topic(topic)  # hotswap — loads from disk, evicts LRU if needed
    # Now model has the knowledge for this topic injected
    result = thinker.generate_with_thinking("Check if 17 is prime")
"""
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class TopicRouter:
    """Classifies user queries into topics for AirMoE hotswap.

    Uses keyword matching (fast, no model needed) to route queries
    to the correct topic module.
    """

    # Topic keywords for routing
    TOPIC_KEYWORDS = {
        "python_math": ["fibonacci", "factorial", "prime", "gcd", "lcm",
                         "sqrt", "matrix", "sum", "average", "quadratic"],
        "python_strings": ["string", "reverse", "palindrome", "regex",
                            "substring", "concat", "split", "join",
                            "replace", "uppercase", "lowercase"],
        "python_algorithms": ["sort", "search", "binary search", "recursive",
                               "dynamic programming", "tree", "graph",
                               "bfs", "dfs", "traverse"],
        "python_oop": ["class", "object", "inheritance", "method",
                        "constructor", "self.", "__init__"],
        "python_file_io": ["file", "open", "read", "write", "csv", "json",
                            "path", "directory"],
        "python_general": ["python", "def", "print", "list", "dict",
                            "tuple", "set", "loop", "for", "while"],
        "math_arithmetic": ["add", "subtract", "multiply", "divide",
                             "sum", "product", "average", "remainder"],
        "math_algebra": ["solve", "equation", "algebra", "variable",
                          "linear", "polynomial"],
        "math_geometry": ["triangle", "circle", "angle", "area",
                          "perimeter", "volume", "geometry"],
        "math_probability": ["probability", "combinatoric", "permutation",
                              "combination", "dice", "coin"],
        "math_theory": ["proof", "theorem", "lemma", "corollary",
                         "axiom", "mathematical proof"],
        "logic": ["logic", "syllogism", "deductive", "premise",
                   "conclusion", "inference", "valid", "sound"],
        "reasoning_general": ["reason", "explain", "why", "how",
                               "analyze", "deduce", "infer"],
        "science_method": ["hypothesis", "experiment", "variable",
                            "control", "observation", "scientific method"],
        "science_biology": ["biology", "cell", "organism", "evolution",
                             "dna", "protein", "gene"],
        "science_chemistry": ["chemistry", "molecule", "reaction",
                               "bond", "atom", "element", "compound"],
        "science_physics": ["physics", "force", "energy", "quantum",
                             "velocity", "acceleration", "momentum"],
        "science_general": ["science", "nature", "physical", "chemical",
                             "biological"],
    }

    def __init__(self, index_path: str):
        """Initialize router with the module index.

        Args:
            index_path: path to airmoe_modules/index.json
        """
        self.index_path = index_path
        with open(index_path, encoding="utf-8") as f:
            self.index = json.load(f)
        self.available_topics = set(self.index.get("topics", {}).keys())

    def classify(self, query: str) -> str:
        """Classify a query into a topic.

        Returns the best-matching topic that has an available module.
        Falls back to the first available topic if no match.
        """
        query_lower = query.lower()
        scores = {}

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if topic not in self.available_topics:
                continue
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[topic] = score

        if scores:
            return max(scores, key=scores.get)

        # Fallback: pick first available topic
        if self.available_topics:
            return next(iter(self.available_topics))
        return "general"

    def list_topics(self) -> list[str]:
        """List all available topic modules."""
        return sorted(self.available_topics)


class AirMoEHotswapLoader:
    """Loads AirMoE topic modules on demand with LRU eviction.

    Only the active topic module's knowledge is in VRAM.
    When switching topics, the LRU module is evicted and the new one loaded.

    Knowledge injection methods:
      1. "context" — prepend knowledge text to the prompt (simple, always works)
      2. "kv_cache" — pre-compute KV cache and inject (zero-token, needs KnowledgePack key)
    """

    def __init__(self, model, tokenizer, device: str = "cuda",
                 cache_size: int = 3,
                 injection_method: str = "context",
                 max_knowledge_chars: int = 1500):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache_size = cache_size
        self.injection_method = injection_method
        self.max_knowledge_chars = max_knowledge_chars

        # LRU cache: topic → knowledge text
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.current_topic: str | None = None
        self.current_knowledge: str = ""

        # Stats
        self.load_count = 0
        self.evict_count = 0
        self.cache_hits = 0

    def load_topic(self, topic: str, modules_base: str = "") -> str:
        """Load a topic module from disk (hotswap).

        If the topic is already cached, just move it to MRU position.
        Otherwise, load from disk and evict LRU if cache is full.

        Returns the knowledge text for the topic.
        """
        # Cache hit
        if topic in self.cache:
            self.cache.move_to_end(topic)
            self.current_topic = topic
            self.current_knowledge = self.cache[topic]
            self.cache_hits += 1
            return self.current_knowledge

        # Cache miss — load from disk
        knowledge = self._load_from_disk(topic, modules_base)

        # Evict LRU if cache is full
        while len(self.cache) >= self.cache_size:
            evicted_topic, _ = self.cache.popitem(last=False)
            self.evict_count += 1
            print(f"    [AirMoE] Evicted topic: {evicted_topic}")

        # Add to cache
        self.cache[topic] = knowledge
        self.current_topic = topic
        self.current_knowledge = knowledge
        self.load_count += 1

        print(f"    [AirMoE] Loaded topic: {topic} "
              f"({len(knowledge)} chars, cache: {len(self.cache)}/{self.cache_size})")

        return knowledge

    def _load_from_disk(self, topic: str, modules_base: str = "") -> str:
        """Load knowledge text for a topic from disk."""
        if not modules_base:
            from research.paths import AIRMOE_MODULES_DIR, as_str
            modules_base = as_str(AIRMOE_MODULES_DIR)

        # Try knowledge.txt first (pre-concatenated)
        knowledge_path = Path(modules_base) / topic / "knowledge.txt"
        if knowledge_path.exists():
            text = knowledge_path.read_text(encoding="utf-8")
            return text[:self.max_knowledge_chars]

        # Fallback: build from knowledge.json
        json_path = Path(modules_base) / topic / "knowledge.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                chunks = json.load(f)
            text = "\n\n---\n\n".join(c["text"] for c in chunks[:20])
            return text[:self.max_knowledge_chars]

        print(f"    [AirMoE] WARNING: No knowledge file for topic '{topic}'")
        return ""

    def get_context_prefix(self) -> str:
        """Get the knowledge text to prepend to the next prompt.

        This is the 'context' injection method — the knowledge is prepended
        to the user's prompt, giving the model relevant context.
        """
        if not self.current_knowledge:
            return ""
        return f"# Reference knowledge:\n{self.current_knowledge}\n\n"

    def inject_kv_cache(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Inject knowledge via KV cache (zero-token injection).

        This uses the KnowledgePack key to pre-compute KV caches from
        the knowledge text and inject them into the model's KV cache.

        Returns the extended input_ids (with KV cache prepended) or None
        if KV cache injection is not available.
        """
        if not self.current_knowledge:
            return None

        try:
            from research.keys.knowledge_pack_key import KnowledgePack
            # Pre-compute KV cache from knowledge text
            pack = KnowledgePack.from_text(
                self.model, self.tokenizer, self.current_knowledge,
                device=self.device)
            # Inject into model's KV cache
            pack.inject(self.model)
            return input_ids  # input_ids unchanged (KV cache is separate)
        except Exception as e:
            print(f"    [AirMoE] KV cache injection failed: {e}, using context")
            return None

    def get_stats(self) -> dict:
        """Get hotswap statistics."""
        return {
            "loads": self.load_count,
            "evictions": self.evict_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache),
            "max_cache": self.cache_size,
            "current_topic": self.current_topic,
        }

    def print_stats(self):
        s = self.get_stats()
        print(f"\n{'='*70}")
        print("AirMoE Hotswap Statistics")
        print(f"{'='*70}")
        print(f"  Topic loads:     {s['loads']}")
        print(f"  Cache hits:      {s['cache_hits']}")
        print(f"  Evictions:       {s['evictions']}")
        print(f"  Cache:           {s['cache_size']}/{s['max_cache']}")
        print(f"  Current topic:   {s['current_topic']}")
        print(f"{'='*70}")


def main():
    """Test the AirMoE hotswap loader."""
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.tokenizer_cache import get_tokenizer

    print("=" * 70)
    print("AirMoE Hotswap Loader Test")
    print("=" * 70)

    # Load model
    print("\n[1] Loading ForgeLM V2...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
    model.to("cuda").eval()
    tokenizer = get_tokenizer("research/checkpoints/qwen_hf")

    # Create router and loader
    from research.paths import AIRMOE_MODULES_DIR, as_str
    index_path = as_str(AIRMOE_MODULES_DIR / "index.json")

    if not os.path.exists(index_path):
        print(f"\n  ERROR: Index not found at {index_path}")
        print("  Run: python -m research.training_packs_airmoe")
        return

    router = TopicRouter(index_path)
    loader = AirMoEHotswapLoader(model, tokenizer, device="cuda",
                                  cache_size=3, injection_method="context")

    print(f"\n[2] Available topics: {router.list_topics()}")

    # Test routing + hotswap
    test_queries = [
        "Check if 17 is prime",
        "Reverse the string 'hello'",
        "Solve: x^2 + 5x + 6 = 0",
        "What is the scientific method?",
        "Prove that the sum of two even numbers is even",
    ]

    print("\n[3] Testing routing + hotswap...")
    for query in test_queries:
        topic = router.classify(query)
        print(f"\n  Query: {query[:50]}")
        print(f"  Topic: {topic}")

        knowledge = loader.load_topic(topic)
        context = loader.get_context_prefix()
        print(f"  Context: {len(context)} chars")

    loader.print_stats()


if __name__ == "__main__":
    main()
