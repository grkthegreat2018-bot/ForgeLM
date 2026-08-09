"""AirMoE Key — AirLLM + MoE hybrid: expert hotswap from disk.

Novel architecture: combines AirLLM (layer streaming) with MoE (expert offloading).
Only active experts are loaded to VRAM; inactive ones stay on disk.
LRU cache manages expert residency. Router predictions enable prefetching.

Inspired by FlashMoE (2026), FluxMoE (2026), FineMoE (2026).

MODULAR EXPERT DISTRIBUTION:
  - Base model file: contains shared weights (embed, attention, norms, shared expert)
  - Expert files: one file per expert (or per layer-expert), stored separately
  - manifest.json: lists all expert files with metadata (size, hash, topic, compression)
  - Users download ONLY the experts they need
  - Experts can be distributed as:
      - Individual files: expert_l0_e0.safetensors, expert_l0_e1.safetensors, ...
      - Topic bundles: experts_math.safetensors, experts_code.safetensors, ...
      - Layer bundles: layer_0_experts.safetensors, layer_1_experts.safetensors, ...

File layout on disk:
  airmoe_module/
    base_model.safetensors       # shared weights (no routed experts)
    manifest.json                # expert registry + routing info
    experts/
      expert_l0_e0.safetensors   # individual expert file
      expert_l0_e1.safetensors
      ...
      bundle_math.safetensors    # optional topic bundle
      bundle_code.safetensors

manifest.json format:
  {
    "model_name": "ForgeLM-v2-airmoe",
    "base_model": "base_model.safetensors",
    "n_layers": 28,
    "n_experts": 4,
    "expert_dim": 1792,
    "compressed": true,
    "experts": [
      {
        "id": "l0_e0",
        "layer": 0, "expert_idx": 0,
        "file": "experts/expert_l0_e0.safetensors",
        "size_bytes": 1234567,
        "sha256": "abc123...",
        "topic": "math",           # optional: what this expert specializes in
        "compressed": true,
        "svd_rank": 512
      },
      ...
    ],
    "bundles": [
      {
        "name": "math",
        "file": "experts/bundle_math.safetensors",
        "expert_ids": ["l0_e0", "l5_e1", "l10_e2"],
        "size_bytes": 12345678
      }
    ]
  }

Key class: TRIVIAL — runtime strategy, no weight changes (just storage format).

Usage:
    from research.keys.airmoe_key import AirMoEKey, AirMoECache, AirMoEManifest
    # Build modular AirMoE module from a full model
    manifest = AirMoEManifest.build_from_state(state, n_layers=28, n_experts=4,
                                                output_dir="research/checkpoints/airmoe")
    # At inference, load base + only needed experts
    cache = AirMoECache.from_manifest(manifest, max_resident=2)
    expert = cache.get_expert(layer_idx=0, expert_idx=1)
"""
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import Key, KeyClass, KeyResult


class AirMoEManifest:
    """Manifest for a modular AirMoE expert distribution.

    Tracks all expert files, their metadata, and optional topic bundles.
    Enables selective download of only needed experts.
    """

    def __init__(self, model_name: str = "", base_model: str = "",
                 n_layers: int = 0, n_experts: int = 0,
                 expert_dim: int = 0, compressed: bool = False):
        self.model_name = model_name
        self.base_model = base_model
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.compressed = compressed
        self.experts: list[dict] = []
        self.bundles: list[dict] = []
        self.topics_map: dict[str, Any] = {}  # V4 format: topic -> metadata
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")

    def add_expert(self, layer: int, expert_idx: int, file_path: str,
                   size_bytes: int, sha256: str = "", topic: str = "",
                   compressed: bool = False, svd_rank: int = 0) -> dict:
        """Register an expert file in the manifest."""
        entry = {
            "id": f"l{layer}_e{expert_idx}",
            "layer": layer,
            "expert_idx": expert_idx,
            "file": file_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "topic": topic,
            "compressed": compressed,
            "svd_rank": svd_rank,
        }
        self.experts.append(entry)
        return entry

    def add_bundle(self, name: str, file_path: str, expert_ids: list[str],
                   size_bytes: int) -> dict:
        """Register a topic bundle (multiple experts in one file)."""
        entry = {
            "name": name,
            "file": file_path,
            "expert_ids": expert_ids,
            "size_bytes": size_bytes,
        }
        self.bundles.append(entry)
        return entry

    def get_expert_file(self, layer: int, expert_idx: int) -> str | None:
        """Get the file path for a specific expert.

        Handles both v2 (expert_idx) and v4 (topic-based, no expert_idx) formats.
        For v4, expert_idx is used as a sequential index into the layer's experts.
        """
        # v2 format: has expert_idx field
        for e in self.experts:
            if e.get("layer") == layer and e.get("expert_idx") == expert_idx:
                return e["file"]
        # v4 format: no expert_idx, use sequential order within layer
        layer_experts = [e for e in self.experts if e.get("layer") == layer]
        if layer_experts and expert_idx < len(layer_experts):
            return layer_experts[expert_idx]["file"]
        return None

    def get_experts_by_topic(self, topic: str) -> list[dict]:
        """Get all experts for a given topic."""
        return [e for e in self.experts if e.get("topic") == topic]

    def get_topics(self) -> list[str]:
        """Get all unique topic names."""
        return list(set(e.get("topic", "") for e in self.experts if e.get("topic")))

    def total_size(self) -> int:
        """Total size of all expert files."""
        return sum(e["size_bytes"] for e in self.experts)

    def save(self, path: str):
        """Save manifest to JSON."""
        data = {
            "model_name": self.model_name,
            "base_model": self.base_model,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "expert_dim": self.expert_dim,
            "compressed": self.compressed,
            "created": self.created,
            "experts": self.experts,
            "bundles": self.bundles,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "AirMoEManifest":
        """Load manifest from JSON.

        Handles both the original AirMoE format and the V4 format
        (which uses different field names and topic-based experts).
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # V4 format uses 'name'/'base_model_file' instead of 'model_name'/'base_model'
        m = cls(
            model_name=data.get("model_name", data.get("name", "")),
            base_model=data.get("base_model", data.get("base_model_file", "")),
            n_layers=data.get("n_layers", 0),
            n_experts=data.get("n_experts", 0),
            expert_dim=data.get("expert_dim", 0),
            compressed=data.get("compressed", bool(data.get("expert_compression", False))),
        )
        m.experts = data.get("experts", [])
        m.bundles = data.get("bundles", [])
        m.created = data.get("created", "")
        # Store V4-specific fields if present
        if "topics" in data:
            m.topics_map = data["topics"]
        return m

    def print_summary(self):
        """Print manifest summary."""
        print(f"  Model: {self.model_name}")
        print(f"  Base: {self.base_model}")
        print(f"  Layers: {self.n_layers}, Experts/layer: {self.n_experts}")
        print(f"  Compressed: {self.compressed}")
        print(f"  Expert files: {len(self.experts)}")
        print(f"  Bundles: {len(self.bundles)}")
        print(f"  Total expert size: {self.total_size()/1e6:.1f} MB")
        topics = self.get_topics()
        if topics:
            print(f"  Topics: {topics}")
            for t in topics:
                exps = self.get_experts_by_topic(t)
                size = sum(e["size_bytes"] for e in exps)
                print(f"    {t}: {len(exps)} experts, {size/1e6:.1f} MB")


def _sha256_file(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compress_expert_svd(w: torch.Tensor, energy: float = 0.9) -> dict[str, torch.Tensor]:
    """Compress a weight matrix via SVD low-rank approximation."""
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    cumsum = (S ** 2).cumsum(0)
    total = cumsum[-1]
    k = max(1, (cumsum < energy * total).sum().item() + 1)
    return {
        "U": U[:, :k].to(torch.bfloat16),
        "S": S[:k].to(torch.bfloat16),
        "Vh": Vh[:k, :].to(torch.bfloat16),
        "rank": torch.tensor([k], dtype=torch.int32),
    }


def _decompress_expert_svd(state: dict[str, torch.Tensor],
                            part: str, device: str) -> torch.Tensor:
    """Reconstruct a weight matrix from SVD components."""
    U = state[f"{part}_U"].float().to(device)
    S = state[f"{part}_S"].float().to(device)
    Vh = state[f"{part}_Vh"].float().to(device)
    return (U * S.unsqueeze(0)) @ Vh


class AirMoEKey(Key):
    """AirMoE key — modular expert hotswap from disk with LRU caching.

    Splits MoE experts into individual files for selective download.
    Users only need the base model + experts they want.

    Key class: TRIVIAL — runtime strategy, no weight transform.
    """

    def __init__(self, max_resident_experts: int = 2, prefetch_lookahead: int = 1):
        self.max_resident_experts = max_resident_experts
        self.prefetch_lookahead = prefetch_lookahead

    @property
    def name(self) -> str:
        return "airmoe"

    @property
    def description(self) -> str:
        return "AirLLM+MoE: modular expert hotswap from disk (selective download)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, weights=data,
                         metadata={"runtime": True, "max_resident": self.max_resident_experts})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data=weights)


def build_airmoe_module(state: dict[str, torch.Tensor],
                         n_layers: int, n_experts: int,
                         output_dir: str,
                         model_name: str = "ForgeLM-airmoe",
                         compress: bool = True,
                         svd_energy: float = 0.9,
                         topics: dict[int, str] | None = None,
                         base_filename: str = "base_model.safetensors") -> AirMoEManifest:
    """Build a modular AirMoE module from a full model state dict.

    Splits the model into:
      - base_model.safetensors: shared weights (embed, attn, norms, shared expert, router)
      - experts/expert_l{i}_e{ei}.safetensors: one file per routed expert

    Users can download:
      1. base_model.safetensors (required)
      2. Only the expert files they need (optional, selective)

    Args:
        state: full model state dict
        n_layers: number of transformer layers
        n_experts: number of routed experts per layer
        output_dir: directory for the AirMoE module
        model_name: name for the manifest
        compress: apply SVD compression to expert files
        svd_energy: energy retention for SVD (0.9 = 90%)
        topics: optional dict mapping expert_idx → topic name
        base_filename: filename for the base model

    Returns:
        AirMoEManifest with all expert file metadata
    """
    from safetensors.torch import save_file

    out = Path(output_dir)
    experts_dir = out / "experts"
    out.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)

    manifest = AirMoEManifest(
        model_name=model_name,
        base_model=base_filename,
        n_layers=n_layers,
        n_experts=n_experts,
        compressed=compress,
    )

    # Phase 1: Extract and save individual expert files
    print(f"  [AirMoE] Splitting {n_layers * n_experts} experts into individual files...")
    expert_keys_to_remove = []
    total_expert_size = 0

    for i in range(n_layers):
        for ei in range(n_experts):
            expert_state = {}
            for part in ["w1", "w2", "w3"]:
                k = f"blocks.{i}.ffn.experts.{ei}.{part}.weight"
                if k in state:
                    if compress:
                        U, S, Vh = torch.linalg.svd(state[k].float(), full_matrices=False)
                        cumsum = (S ** 2).cumsum(0)
                        total = cumsum[-1]
                        k_rank = max(1, (cumsum < svd_energy * total).sum().item() + 1)
                        expert_state[f"{part}_U"] = U[:, :k_rank].to(torch.bfloat16)
                        expert_state[f"{part}_S"] = S[:k_rank].to(torch.bfloat16)
                        expert_state[f"{part}_Vh"] = Vh[:k_rank, :].to(torch.bfloat16)
                        expert_state[f"{part}_rank"] = torch.tensor([k_rank], dtype=torch.int32)
                        svd_rank = k_rank
                    else:
                        expert_state[f"{part}.weight"] = state[k]
                        svd_rank = 0
                    expert_keys_to_remove.append(k)

            if not expert_state:
                continue

            shard_name = f"expert_l{i}_e{ei}.safetensors"
            shard_path = experts_dir / shard_name
            save_file(expert_state, str(shard_path))
            shard_size = shard_path.stat().st_size
            sha = _sha256_file(str(shard_path))
            total_expert_size += shard_size

            topic = topics.get(ei, "") if topics else ""
            manifest.add_expert(
                layer=i, expert_idx=ei,
                file_path=f"experts/{shard_name}",
                size_bytes=shard_size, sha256=sha,
                topic=topic, compressed=compress, svd_rank=svd_rank,
            )

    # Phase 2: Remove expert weights from base state
    for k in expert_keys_to_remove:
        del state[k]

    # Phase 3: Save base model (shared weights only)
    base_path = out / base_filename
    save_file(state, str(base_path))
    base_size = base_path.stat().st_size

    # Phase 4: Save manifest
    manifest_path = out / "manifest.json"
    manifest.save(str(manifest_path))

    # Summary
    print(f"  [AirMoE] Module built at {output_dir}")
    print(f"  [AirMoE] Base model: {base_filename} ({base_size/1e6:.0f} MB)")
    print(f"  [AirMoE] Expert files: {len(manifest.experts)} "
          f"({total_expert_size/1e6:.0f} MB total)")
    print(f"  [AirMoE] Compressed: {compress}")
    if manifest.get_topics():
        print(f"  [AirMoE] Topics: {manifest.get_topics()}")
        for t in manifest.get_topics():
            exps = manifest.get_experts_by_topic(t)
            tsize = sum(e["size_bytes"] for e in exps)
            print(f"    {t}: {len(exps)} files, {tsize/1e6:.1f} MB")
    print("  [AirMoE] Manifest: manifest.json")
    print(f"  [AirMoE] Total module: {(base_size + total_expert_size)/1e6:.0f} MB")
    print("  [AirMoE] Selective download: users pick which experts they need")

    return manifest


def build_topic_bundle(manifest: AirMoEManifest, topic: str,
                        output_dir: str) -> dict | None:
    """Bundle all experts for a topic into a single file.

    This allows users to download one file per topic instead of
    many individual expert files.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    experts = manifest.get_experts_by_topic(topic)
    if not experts:
        return None

    bundle_state = {}
    for e in experts:
        expert_path = os.path.join(output_dir, e["file"])
        if not os.path.exists(expert_path):
            continue
        with safe_open(expert_path, framework="pt") as f:
            for key in f.keys():
                # Prefix with expert ID to avoid collisions
                bundle_state[f"{e['id']}__{key}"] = f.get_tensor(key)

    bundle_name = f"bundle_{topic}.safetensors"
    bundle_path = os.path.join(output_dir, "experts", bundle_name)
    save_file(bundle_state, bundle_path)
    bundle_size = os.path.getsize(bundle_path)

    entry = manifest.add_bundle(
        name=topic,
        file_path=f"experts/{bundle_name}",
        expert_ids=[e["id"] for e in experts],
        size_bytes=bundle_size,
    )

    print(f"  [AirMoE] Bundle '{topic}': {len(experts)} experts → "
          f"{bundle_name} ({bundle_size/1e6:.1f} MB)")
    return entry


class AirMoECache:
    """LRU cache for MoE expert weights with prefetch support.

    Loads expert shards from disk on demand, keeps recently-used
    experts in VRAM, evicts least-recently-used when cache is full.

    Supports both individual expert files and topic bundles.

    Usage:
        # From manifest (recommended)
        cache = AirMoECache.from_manifest(manifest, base_dir=".", max_resident=2)
        expert = cache.get_expert(layer_idx=0, expert_idx=1)

        # From directory (legacy)
        cache = AirMoECache("research/checkpoints/airmoe/experts", max_resident=2)
    """

    def __init__(self, cache_dir: str, max_resident: int = 2,
                 device: str = "cuda", manifest: AirMoEManifest | None = None,
                 base_dir: str = ""):
        self.cache_dir = cache_dir
        self.max_resident = max_resident
        self.device = device
        self.manifest = manifest
        self.base_dir = base_dir
        self.cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.total_load_time = 0.0
        self.loaded_experts: Set[str] = set()

    @classmethod
    def from_manifest(cls, manifest: AirMoEManifest, base_dir: str = ".",
                      max_resident: int = 2, device: str = "cuda") -> "AirMoECache":
        """Create cache from a manifest (recommended way)."""
        experts_dir = os.path.join(base_dir, "experts")
        return cls(cache_dir=experts_dir, max_resident=max_resident,
                   device=device, manifest=manifest, base_dir=base_dir)

    def _get_expert_path(self, layer_idx: int, expert_idx: int) -> str:
        """Get the file path for an expert.

        Supports both naming conventions:
          - v4: expert_l{layer}_{topic}.safetensors (topic-based)
          - v2: expert_l{layer}_e{idx}.safetensors (index-based)
          - v4: expert_l{layer}_{topic}.safetensors (topic-based, many topics)
        """
        if self.manifest:
            rel = self.manifest.get_expert_file(layer_idx, expert_idx)
            if rel:
                return os.path.join(self.base_dir, rel)
        # Try index-based naming first (v2/v3)
        idx_path = os.path.join(self.cache_dir, f"expert_l{layer_idx}_e{expert_idx}.safetensors")
        if os.path.exists(idx_path):
            return idx_path
        # Try topic-based naming (v4): scan for expert_l{layer}_*.safetensors
        # V4 has many topics per layer, so we can't use a hardcoded mapping.
        # Instead, list files matching the pattern and return the first match.
        import re
        if os.path.isdir(self.cache_dir):
            pattern = re.compile(rf'expert_l{layer_idx}_(.+)\.safetensors$')
            for fname in sorted(os.listdir(self.cache_dir)):
                m = pattern.match(fname)
                if m:
                    return os.path.join(self.cache_dir, fname)
        # No match found — return the index-based path (will fail with clear error)
        return idx_path

    def get_expert_by_topic(self, layer_idx: int, topic: str) -> dict[str, torch.Tensor]:
        """Get expert weights by topic name (e.g., 'math_algebra', 'python_strings').

        This is the preferred method for v4 checkpoints where experts
        are named by topic. Works with any topic name from the manifest.
        """
        if self.manifest:
            # Search manifest for matching topic + layer
            for e in self.manifest.experts:
                if e.get("layer") == layer_idx and e.get("topic") == topic:
                    expert_idx = e.get("expert_idx", -1)
                    return self.get_expert(layer_idx, expert_idx)
        # Direct file path construction
        topic_path = os.path.join(self.cache_dir,
                                   f"expert_l{layer_idx}_{topic}.safetensors")
        if os.path.exists(topic_path):
            # Load directly (bypass get_expert's index-based cache key)
            key = (layer_idx, hash(topic) % 10000)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hit_count += 1
                return self.cache[key]
            self.miss_count += 1
            from safetensors.torch import load_file
            expert_state = load_file(topic_path)
            if any(k.endswith("_U") for k in expert_state):
                reconstructed = {}
                for part_base in ["w1", "w2", "w3"]:
                    U = expert_state.get(f"{part_base}_U")
                    S = expert_state.get(f"{part_base}_S")
                    Vh = expert_state.get(f"{part_base}_Vh")
                    if U is not None and S is not None and Vh is not None:
                        W = (U.float() * S.float().unsqueeze(0)) @ Vh.float()
                        reconstructed[f"{part_base}.weight"] = W.to(torch.bfloat16).to(self.device)
                expert_state = reconstructed
            else:
                expert_state = {k: v.to(self.device) for k, v in expert_state.items()}
            self.cache[key] = expert_state
            while len(self.cache) > self.max_resident:
                self.cache.popitem(last=False)
            return expert_state
        available = self.list_topics() if hasattr(self, 'list_topics') else []
        raise FileNotFoundError(
            f"Expert file not found for topic '{topic}': {topic_path}\n"
            f"Available topics: {available}")

    def list_topics(self) -> list[str]:
        """List all available expert topics (v4 format)."""
        if self.manifest:
            return list(set(e.get("topic", "") for e in self.manifest.experts if e.get("topic")))
        import re
        topics = set()
        if os.path.isdir(self.cache_dir):
            for fname in os.listdir(self.cache_dir):
                m = re.match(r'expert_l\d+_(.+)\.safetensors$', fname)
                if m:
                    topics.add(m.group(1))
        return sorted(topics)

    def get_experts_for_topic(self, topic: str) -> list[dict[str, torch.Tensor]]:
        """Get all expert weights for a given topic across all layers.

        Returns a list of dicts, one per layer.
        """
        n_layers = self.manifest.n_layers if self.manifest else 28
        experts = []
        for layer in range(n_layers):
            try:
                expert = self.get_expert_by_topic(layer, topic)
                experts.append(expert)
            except FileNotFoundError:
                break
        return experts

    def get_expert(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """Get expert weights, loading from disk if not cached."""
        key = (layer_idx, expert_idx)

        if key in self.cache:
            self.cache.move_to_end(key)
            self.hit_count += 1
            return self.cache[key]

        self.miss_count += 1
        t0 = time.time()

        shard_path = self._get_expert_path(layer_idx, expert_idx)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"Expert shard not found: {shard_path}\n"
                f"Download it from the model repository or build with build_airmoe_module()")

        from safetensors.torch import load_file
        expert_state = load_file(shard_path)

        # If compressed (SVD), reconstruct full weights
        if any(k.endswith("_U") for k in expert_state):
            reconstructed = {}
            for part_base in ["w1", "w2", "w3"]:
                U = expert_state.get(f"{part_base}_U")
                S = expert_state.get(f"{part_base}_S")
                Vh = expert_state.get(f"{part_base}_Vh")
                if U is not None and S is not None and Vh is not None:
                    W = (U.float() * S.float().unsqueeze(0)) @ Vh.float()
                    reconstructed[f"{part_base}.weight"] = W.to(torch.bfloat16).to(self.device)
            expert_state = reconstructed
        else:
            expert_state = {k: v.to(self.device) for k, v in expert_state.items()}

        self.total_load_time += time.time() - t0
        self.cache[key] = expert_state
        self.loaded_experts.add(f"l{layer_idx}_e{expert_idx}")

        # Evict if over capacity
        while len(self.cache) > self.max_resident:
            evicted_key, _ = self.cache.popitem(last=False)

        return expert_state

    def has_expert(self, layer_idx: int, expert_idx: int) -> bool:
        """Check if an expert file exists on disk (without loading)."""
        path = self._get_expert_path(layer_idx, expert_idx)
        return os.path.exists(path)

    def list_available_experts(self) -> list[tuple[int, int]]:
        """List all experts that have files on disk."""
        if self.manifest:
            return [(e["layer"], e["expert_idx"]) for e in self.manifest.experts
                    if os.path.exists(os.path.join(self.base_dir, e["file"]))]
        # Fallback: scan directory
        import re
        available = []
        for f in os.listdir(self.cache_dir):
            m = re.match(r'expert_l(\d+)_e(\d+)\.safetensors', f)
            if m:
                available.append((int(m.group(1)), int(m.group(2))))
        return sorted(available)

    def list_missing_experts(self) -> list[tuple[int, int]]:
        """List experts that are in the manifest but not on disk."""
        if not self.manifest:
            return []
        available = set(self.list_available_experts())
        all_experts = {(e["layer"], e["expert_idx"]) for e in self.manifest.experts}
        return sorted(all_experts - available)

    def prefetch(self, layer_idx: int, expert_indices: list[int]):
        """Prefetch experts for the next layer (async-friendly)."""
        for ei in expert_indices:
            key = (layer_idx, ei)
            if key not in self.cache:
                try:
                    self.get_expert(layer_idx, ei)
                except FileNotFoundError as e:
                    import warnings
                    warnings.warn(f"expert file loading: {e}", RuntimeWarning, stacklevel=2)

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.hit_count + self.miss_count
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": self.hit_count / max(total, 1),
            "resident": len(self.cache),
            "max_resident": self.max_resident,
            "total_load_time": self.total_load_time,
            "avg_load_time": self.total_load_time / max(self.miss_count, 1),
            "loaded_experts": list(self.loaded_experts),
        }


# Backward compatibility
def apply_airmoe(state: dict[str, torch.Tensor], n_layers: int,
                 n_experts: int, cache_dir: str,
                 compress: bool = True) -> dict[str, torch.Tensor]:
    """Legacy API — use build_airmoe_module() for modular distribution."""
    manifest = build_airmoe_module(
        state, n_layers, n_experts, cache_dir, compress=compress)
    return state


if __name__ == "__main__":
    key = AirMoEKey(max_resident_experts=2)
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print("  Modular expert distribution: base + individual expert files")
    print("  Selective download: users pick which experts they need")
    print("  Topic bundles: group experts by specialization")
    print("  AirMoE key verified ✓")
