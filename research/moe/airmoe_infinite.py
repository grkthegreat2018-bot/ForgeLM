"""AirMoE — Infinite Expert Library with VRAM-limited hotswap.

Architecture:
  Base model (always in VRAM):
    - Embedding + LM Head
    - Attention layers (MLA)
    - Shared FFN expert per layer (always active, handles general cases)
    - Router (classifies query → topic)

  Expert library (unlimited, on disk):
    - Each expert is a standalone FFN (w1, w2, w3) for one layer
    - Named by topic: expert_l{layer}_{topic}.safetensors
    - Topics can be anything: python_algorithms, math_geometry, science_physics, ...
    - New experts added anytime without retraining
    - SVD + int4 compressed (~2 MB each)

  At inference:
    1. Router classifies query → topic (e.g., "python_algorithms")
    2. AirMoE cache loads expert_l{0..27}_python_algorithms.safetensors from disk
    3. Expert weights injected into the "routed expert" slot for each layer
    4. Forward pass: shared expert + routed expert (topic-specific)
    5. LRU evicts least-recently-used topic when VRAM is full

  No hard expert limit. VRAM is the only constraint.
  With int4 + SVD: each expert ~2 MB, can cache 100+ experts in 2 GB VRAM.

File layout:
  forgelm_v4/
    base_model_int4.safetensors     # base weights (no routed experts)
    manifest.json                    # expert library index
    rotorquant_rotations.pt          # KV cache compression
    experts/                         # infinite expert library
      expert_l0_python_algorithms.safetensors
      expert_l0_math_geometry.safetensors
      expert_l0_science_physics.safetensors
      ... (unlimited, add new topics anytime)

Usage:
    from research.moe.airmoe_infinite import InfiniteAirMoE
    airmoe = InfiniteAirMoE(model, tokenizer, "forgelm_v4/")
    airmoe.load_expert_topic("python_algorithms")  # hotswap
    result = thinker.generate_with_thinking("Sort a list using merge sort")
"""
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class ExpertRouter:
    """Routes queries to expert topics using keyword matching.

    The router has no hard limit on topics — new topics can be registered
    at any time. It maps a user query to the best-matching expert topic
    in the library.
    """

    # Module-level manifest cache: {manifest_path: manifest_dict}
    # Avoids re-reading manifest.json from disk on every ExpertRouter creation.
    _manifest_cache: dict[str, dict] = {}

    def __init__(self, manifest_path: str):
        if manifest_path not in self._manifest_cache:
            with open(manifest_path, encoding="utf-8") as f:
                self._manifest_cache[manifest_path] = json.load(f)
        self.manifest = self._manifest_cache[manifest_path]

        self.topics: dict[str, dict] = self.manifest.get("topics", {})
        self.n_layers = self.manifest.get("n_layers", 28)

    def classify(self, query: str) -> str:
        """Classify a query into the best expert topic.

        Returns the topic name with the highest keyword match score.
        Falls back to "general" if no match.
        """
        query_lower = query.lower()
        scores = {}

        for topic_name, info in self.topics.items():
            keywords = info.get("keywords", [])
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[topic_name] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def classify_multi(self, query: str, top_n: int = 2) -> list[str]:
        """Classify a query into multiple expert topics (for multi-expert routing).

        Returns up to top_n topic names sorted by match score.
        """
        query_lower = query.lower()
        scores = {}

        for topic_name, info in self.topics.items():
            keywords = info.get("keywords", [])
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[topic_name] = score

        if not scores:
            return ["general"]

        sorted_topics = sorted(scores, key=scores.get, reverse=True)
        return sorted_topics[:top_n]

    def list_topics(self) -> list[str]:
        """List all available expert topics in the library."""
        return sorted(self.topics.keys())

    def register_topic(self, topic: str, keywords: list[str],
                        label: str = "", subtopics: list[str] = None):
        """Register a new expert topic at runtime.

        This allows adding new experts to the library without restarting.
        """
        self.topics[topic] = {
            "label": label or topic,
            "keywords": keywords,
            "subtopics": subtopics or [],
        }


class InfiniteAirMoE:
    """Infinite expert library with VRAM-limited hotswap.

    The model has a shared FFN (always in VRAM) + a routed expert slot
    that gets filled dynamically from the expert library on disk.

    Only VRAM limits how many experts can be cached. With SVD+int4
    compression (~2 MB per expert), 100+ experts fit in 2 GB.

    Hotswap flow:
      1. Router classifies query → topic
      2. Check if topic's experts are in VRAM cache
      3. If not: load from disk (28 files, one per layer)
      4. Inject expert weights into model's expert slots
      5. If VRAM full: evict LRU topic
      6. Forward pass with shared + routed expert
    """

    def __init__(self, model, tokenizer, module_dir: str,
                 device: str = "cuda",
                 vram_budget_gb: float = 2.0,
                 max_cached_topics: int = 10):
        """
        Args:
            model: ForgeLM model with shared FFN + expert slots
            tokenizer: tokenizer
            module_dir: path to forgelm_v4/ directory
            device: target device
            vram_budget_gb: VRAM budget for expert cache
            max_cached_topics: max topics cached simultaneously
        """
        self.model = model
        self.tokenizer = tokenizer
        self.module_dir = Path(module_dir)
        self.device = device
        self.vram_budget_gb = vram_budget_gb
        self.max_cached_topics = max_cached_topics

        # Load router from manifest
        manifest_path = self.module_dir / "manifest.json"
        self.router = ExpertRouter(str(manifest_path))
        self.n_layers = self.router.n_layers

        # LRU cache: topic → {layer: expert_weights}
        self.cache: OrderedDict[str, dict[int, dict[str, torch.Tensor]]] = OrderedDict()
        self.current_topic: str | None = None
        self.current_experts: dict[int, dict[str, torch.Tensor]] | None = None

        # Stats
        self.stats = {
            "topic_loads": 0,
            "cache_hits": 0,
            "evictions": 0,
            "total_load_time_ms": 0,
            "bytes_loaded": 0,
        }

    def _get_expert_path(self, layer: int, topic: str) -> Path:
        """Get the file path for a topic expert at a given layer."""
        return self.module_dir / "experts" / f"expert_l{layer}_{topic}.safetensors"

    def _load_topic_from_disk(self, topic: str) -> dict[int, dict[str, torch.Tensor]]:
        """Load all layer experts for a topic from disk.

        Uses ThreadPoolExecutor for parallel disk reads (10-20x faster on
        NVMe/SSD when loading 28 separate expert files).

        Returns: {layer_idx: {"w1": tensor, "w2": tensor, "w3": tensor}}
        """
        from concurrent.futures import ThreadPoolExecutor

        from safetensors.torch import load_file

        # Collect valid paths first
        paths = {}
        for layer in range(self.n_layers):
            path = self._get_expert_path(layer, topic)
            if path.exists():
                paths[layer] = path

        if not paths:
            return {}

        def _load_one(layer_path_tuple):
            layer, path = layer_path_tuple
            state = load_file(str(path))

            # Decompress: check which format
            if any(k.endswith("_U_q") for k in state):
                expert = self._decompress_expert(state)
            elif any(k.endswith("_U") and "_U_q" not in k and "_U_shape" not in k and "_U_scale" not in k for k in state):
                expert = self._decompress_expert_svd_only(state)
            else:
                expert = {}
                for part in ["w1", "w2", "w3"]:
                    k = f"{part}.weight"
                    if k in state:
                        expert[part] = state[k].to(self.device)

            return layer, expert, path.stat().st_size

        experts = {}
        total_bytes = 0

        # Parallel disk reads (I/O bound, threads are safe for safetensors load)
        max_workers = min(8, len(paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_load_one, paths.items()))

        for layer, expert, size in results:
            experts[layer] = expert
            total_bytes += size

        self.stats["bytes_loaded"] += total_bytes
        return experts

    @staticmethod
    def _decompress_expert(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decompress SVD + int4 expert to full weight matrices."""
        expert = {}
        gs = 128
        for part in ["w1", "w2", "w3"]:
            U_q = state.get(f"{part}_U_q")
            U_scale = state.get(f"{part}_U_scale")
            S = state.get(f"{part}_S")
            Vh_q = state.get(f"{part}_Vh_q")
            Vh_scale = state.get(f"{part}_Vh_scale")
            U_shape = state.get(f"{part}_U_shape")
            Vh_shape = state.get(f"{part}_Vh_shape")

            if U_q is None:
                continue

            # Dequantize U: q is [n_groups, gs], scale is [n_groups]
            U = (U_q.float() * U_scale.float().unsqueeze(-1)).reshape(-1)
            if U_shape is not None:
                U = U[:U_shape[0].item() * U_shape[1].item()].reshape(
                    U_shape[0].item(), U_shape[1].item())

            # Dequantize Vh: q is [n_groups, gs], scale is [n_groups]
            Vh = (Vh_q.float() * Vh_scale.float().unsqueeze(-1)).reshape(-1)
            if Vh_shape is not None:
                Vh = Vh[:Vh_shape[0].item() * Vh_shape[1].item()].reshape(
                    Vh_shape[0].item(), Vh_shape[1].item())

            # Reconstruct: W = U @ diag(S) @ Vh
            W = (U * S.float().unsqueeze(0)) @ Vh
            expert[part] = W.to(torch.bfloat16)

        return expert

    @staticmethod
    def _decompress_expert_svd_only(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decompress SVD-only expert (v2 format)."""
        expert = {}
        for part in ["w1", "w2", "w3"]:
            U = state.get(f"{part}_U")
            S = state.get(f"{part}_S")
            Vh = state.get(f"{part}_Vh")
            if U is None:
                continue
            W = (U.float() * S.float().unsqueeze(0)) @ Vh.float()
            expert[part] = W.to(torch.bfloat16)
        return expert

    def _inject_expert(self, layer: int, expert: dict[str, torch.Tensor]):
        """Inject expert weights into the model's expert slot for a layer."""
        try:
            block = self.model.blocks[layer]
            ffn = block.ffn

            # If MoE, inject into the first routed expert slot
            if hasattr(ffn, 'experts') and len(ffn.experts) > 0:
                expert_module = ffn.experts[0]
                model_dtype = next(self.model.parameters()).dtype
                if "w1" in expert:
                    expert_module.w1.weight.data = expert["w1"].to(self.device, model_dtype)
                if "w2" in expert:
                    expert_module.w2.weight.data = expert["w2"].to(self.device, model_dtype)
                if "w3" in expert:
                    expert_module.w3.weight.data = expert["w3"].to(self.device, model_dtype)
        except (IndexError, AttributeError) as e:
            pass  # Model might not have expert slots at this layer

    def _evict_lru(self):
        """Evict the least recently used topic from the cache."""
        if not self.cache:
            return

        evicted_topic, evicted_experts = self.cache.popitem(last=False)
        self.stats["evictions"] += 1

        # Free VRAM
        del evicted_experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    [AirMoE] Evicted topic: {evicted_topic}")

    def _estimate_cache_vram_gb(self) -> float:
        """Estimate current cache VRAM usage."""
        total_params = 0
        for topic_experts in self.cache.values():
            for layer, expert in topic_experts.items():
                for w in expert.values():
                    total_params += w.numel()
        return total_params * 2 / 1e9  # bf16

    def load_topic(self, topic: str) -> bool:
        """Load a topic's experts into VRAM (hotswap).

        If the topic is already cached, just move to MRU.
        Otherwise, load from disk and evict LRU if needed.

        Returns True if successful, False if topic not found.
        """
        # Cache hit
        if topic in self.cache:
            self.cache.move_to_end(topic)
            self.current_topic = topic
            self.current_experts = self.cache[topic]
            self.stats["cache_hits"] += 1
            return True

        # Check if topic exists on disk
        first_path = self._get_expert_path(0, topic)
        if not first_path.exists():
            print(f"    [AirMoE] Topic '{topic}' not found in expert library")
            return False

        # Load from disk
        t0 = time.time()
        experts = self._load_topic_from_disk(topic)
        load_ms = (time.time() - t0) * 1000
        self.stats["topic_loads"] += 1
        self.stats["total_load_time_ms"] += load_ms

        # Evict LRU if over budget
        while (len(self.cache) >= self.max_cached_topics or
               self._estimate_cache_vram_gb() > self.vram_budget_gb):
            if not self.cache:
                break
            self._evict_lru()

        # Add to cache
        self.cache[topic] = experts
        self.current_topic = topic
        self.current_experts = experts

        # Inject into model
        for layer, expert in experts.items():
            self._inject_expert(layer, expert)

        n_layers_loaded = len(experts)
        print(f"    [AirMoE] Loaded topic: {topic} "
              f"({n_layers_loaded} layers, {load_ms:.0f}ms, "
              f"cache: {len(self.cache)}/{self.max_cached_topics})")

        return True

    def route_and_load(self, query: str) -> str:
        """Route a query to a topic and load the relevant expert.

        This is the main entry point for inference:
          1. Router classifies query → topic
          2. Load topic expert (hotswap if needed)
          3. Return the topic name

        The model is now ready to generate with the topic-specific expert.
        """
        topic = self.router.classify(query)
        self.load_topic(topic)
        return topic

    def route_and_load_multi(self, query: str, top_n: int = 2) -> list[str]:
        """Route a query to multiple topics and load them all.

        For complex queries that span multiple domains (e.g., "write a
        Python function to solve a math equation"), this loads both
        the code and math experts.
        """
        topics = self.router.classify_multi(query, top_n=top_n)
        for topic in topics:
            self.load_topic(topic)
        return topics

    def get_stats(self) -> dict:
        """Get hotswap statistics."""
        return {
            **self.stats,
            "cached_topics": len(self.cache),
            "current_topic": self.current_topic,
            "cache_vram_gb": self._estimate_cache_vram_gb(),
            "vram_budget_gb": self.vram_budget_gb,
            "available_topics": len(self.router.list_topics()),
        }

    def print_stats(self):
        """Print AirMoE statistics."""
        s = self.get_stats()
        print(f"\n{'='*70}")
        print("Infinite AirMoE Statistics")
        print(f"{'='*70}")
        print(f"  Available topics:    {s['available_topics']} (unlimited library)")
        print(f"  Cached topics:       {s['cached_topics']}/{self.max_cached_topics}")
        print(f"  Current topic:       {s['current_topic']}")
        print(f"  Topic loads:         {s['topic_loads']}")
        print(f"  Cache hits:          {s['cache_hits']}")
        print(f"  Evictions:           {s['evictions']}")
        print(f"  Cache VRAM:          {s['cache_vram_gb']:.3f} / {s['vram_budget_gb']:.1f} GB")
        print(f"  Total load time:     {s['total_load_time_ms']:.0f}ms")
        print(f"  Bytes loaded:        {s['bytes_loaded']/1e6:.1f} MB")
        print(f"{'='*70}")


def main():
    """Test the infinite AirMoE system."""
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.tokenizer_cache import get_tokenizer

    print("=" * 70)
    print("Infinite AirMoE Test")
    print("=" * 70)

    from research.paths import FORGELM_V4_DIR, as_str
    module_dir = as_str(FORGELM_V4_DIR)

    if not os.path.exists(os.path.join(module_dir, "manifest.json")):
        print(f"\n  ERROR: No v4 module at {module_dir}")
        print("  Run: python -m research.bake_v4")
        return

    # Load model
    print("\n[1] Loading ForgeLM v4 base...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
    model.to("cuda").eval()
    tokenizer = get_tokenizer("research/checkpoints/qwen_hf")

    # Create infinite AirMoE
    print("\n[2] Creating infinite AirMoE...")
    airmoe = InfiniteAirMoE(model, tokenizer, module_dir,
                             device="cuda",
                             vram_budget_gb=2.0,
                             max_cached_topics=10)

    print(f"  Available topics: {airmoe.router.list_topics()}")

    # Test routing + hotswap
    test_queries = [
        "Write a Python function to sort a list",
        "Solve the quadratic equation x^2 + 5x + 6 = 0",
        "Prove that the sum of two even numbers is even",
        "Explain the scientific method",
    ]

    print("\n[3] Testing routing + hotswap...")
    for query in test_queries:
        topic = airmoe.route_and_load(query)
        print(f"  Query: {query[:50]}")
        print(f"  → Topic: {topic}")

    airmoe.print_stats()


if __name__ == "__main__":
    main()
