"""Memory/offload evolution domains.

Five compact search domains modeling memory tradeoffs on RTX 5070 (12GB VRAM):
  1. HybridOffload      — CPU/GPU weight split, VRAM saved vs latency added
  2. CpuKvOffload       — KV cache offload to CPU, memory freed vs decode latency
  3. ExpertHotload      — MoE expert hotloading from disk, VRAM vs miss rate
  4. MemoryBudget       — VRAM allocation across components, utilization vs OOM risk
  5. CheckpointRecompute — activation checkpointing, memory saved vs recompute overhead

All evaluations use small synthetic tensor ops (torch) to model tradeoffs.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Any
from . import BaseDomain


# ---------------------------------------------------------------------------
# 1. HybridOffload
# ---------------------------------------------------------------------------
class HybridOffload(BaseDomain):
    """CPU/GPU weight offload: offload_ratio, prefetch_depth, pin_memory, overlap."""

    def name(self) -> str: return "hybrid_offload"
    def output_dim(self) -> int: return 4

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        return {
            "offload_ratio": float(np.clip(p[0], 0, 1)),
            "prefetch_depth": int(1 + round(p[1] * 7)),          # 1-8
            "pin_memory": bool(p[2] > 0.5),
            "overlap_compute": bool(p[3] > 0.5),
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([
            config.get("offload_ratio", 0.5),
            (config.get("prefetch_depth", 4) - 1) / 7,
            1.0 if config.get("pin_memory", True) else 0.0,
            1.0 if config.get("overlap_compute", True) else 0.0,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        r = config["offload_ratio"]
        depth = config["prefetch_depth"]
        pin = config["pin_memory"]
        overlap = config["overlap_compute"]
        # VRAM saved (fraction of 12GB weights moved to CPU)
        vram_saved = torch.tensor(r * 2.34)  # bf16 weight bytes
        # Latency: PCIe transfer cost decreases with prefetch depth + overlap
        base_lat = torch.tensor(r * 8.0)  # ms baseline for full offload
        # Prefetch depth: logarithmic (diminishing returns) + memory cost.
        # Each extra prefetch layer uses ~150MB VRAM for staging buffer.
        speedup = torch.tensor(1.0 + np.log2(depth + 1) * 0.3 + (0.3 if overlap else 0.0))
        pin_bonus = torch.tensor(0.2 if pin else 0.0)
        latency = base_lat / speedup - pin_bonus
        # Prefetch memory cost: deeper prefetch = more staging VRAM
        prefetch_mem_cost = depth * 0.05  # 150MB per layer / 3GB scale
        score = (vram_saved * 3.0 - latency * 0.8 - prefetch_mem_cost).item()
        return {
            "score": float(score),
            "behavioral": (float(vram_saved), float(latency.clamp(min=0))),
            "metadata": {"vram_saved_gb": float(vram_saved), "latency_ms": float(latency.clamp(min=0)),
                         "prefetch_mem_cost": prefetch_mem_cost},
        }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("vram_saved", 10, 0.0, 2.5), ("latency_ms", 10, 0.0, 10.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "prefetch_depth": [1, 2, 3, 4, 5, 6, 7, 8],
            "pin_memory": [False, True],
            "overlap_compute": [False, True],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        return [
            {"offload_ratio": 0.0, "prefetch_depth": 1, "pin_memory": False, "overlap_compute": False},
            {"offload_ratio": 0.5, "prefetch_depth": 4, "pin_memory": True, "overlap_compute": True},
            {"offload_ratio": 1.0, "prefetch_depth": 8, "pin_memory": True, "overlap_compute": True},
        ]


# ---------------------------------------------------------------------------
# 2. CpuKvOffload
# ---------------------------------------------------------------------------
class CpuKvOffload(BaseDomain):
    """KV cache offload to CPU: offload_layers, threshold, prefetch_size, async_copy."""

    def name(self) -> str: return "cpu_kv_offload"
    def output_dim(self) -> int: return 4

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        return {
            "offload_layers": int(round(p[0] * 16)),              # 0-16
            "offload_threshold": float(np.clip(p[1], 0, 1)),
            "prefetch_size": int(64 * (2 ** round(p[2] * 5))),   # 64-2048
            "async_copy": bool(p[3] > 0.5),
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        ps = config.get("prefetch_size", 512)
        ps_idx = int(np.log2(max(ps, 64) / 64))
        return torch.tensor([
            config.get("offload_layers", 8) / 16,
            config.get("offload_threshold", 0.5),
            ps_idx / 5,
            1.0 if config.get("async_copy", True) else 0.0,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        layers = config["offload_layers"]
        thresh = config["offload_threshold"]
        ps = config["prefetch_size"]
        async_c = config["async_copy"]
        # KV cache: 16 layers * 8 KV heads * 64 head_dim * seq_len(4096) * 2 (K+V) * bf16
        kv_total = torch.tensor(16 * 8 * 64 * 4096 * 2 * 2 / 1e9)  # GB
        kv_offloaded = kv_total * (layers / 16) * thresh
        # Decode latency: transfer cost grows with layers, shrinks with prefetch + async
        base_lat = torch.tensor(layers * 0.5)
        pf_speedup = torch.tensor(1.0 + np.log2(max(ps, 64) / 64) * 0.15)
        async_bonus = torch.tensor(0.3 if async_c else 0.0)
        latency = (base_lat / pf_speedup - async_bonus).clamp(min=0)
        score = (kv_offloaded * 5.0 - latency * 1.2).item()
        return {
            "score": float(score),
            "behavioral": (float(kv_offloaded), float(latency)),
            "metadata": {"kv_freed_gb": float(kv_offloaded), "decode_lat_ms": float(latency)},
        }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("kv_freed_gb", 10, 0.0, 1.5), ("decode_lat_ms", 10, 0.0, 8.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "offload_layers": [0, 4, 8, 12, 16],
            "prefetch_size": [64, 128, 256, 512, 1024, 2048],
            "async_copy": [False, True],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        return [
            {"offload_layers": 0, "offload_threshold": 0.0, "prefetch_size": 64, "async_copy": False},
            {"offload_layers": 8, "offload_threshold": 0.7, "prefetch_size": 512, "async_copy": True},
            {"offload_layers": 16, "offload_threshold": 1.0, "prefetch_size": 2048, "async_copy": True},
        ]


# ---------------------------------------------------------------------------
# 3. ExpertHotload
# ---------------------------------------------------------------------------
class ExpertHotload(BaseDomain):
    """MoE expert hotloading from disk: n_hot, prefetch_ahead, cache_strategy, disk_cache."""

    CACHE_STRATEGIES = ["lru", "lfu", "priority"]

    def name(self) -> str: return "expert_hotload"
    def output_dim(self) -> int: return 4

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        strat_idx = int(p[2] * len(self.CACHE_STRATEGIES)) % len(self.CACHE_STRATEGIES)
        return {
            "n_hot_experts": int(1 + round(p[0] * 7)),            # 1-8
            "prefetch_ahead": int(1 + round(p[1] * 3)),           # 1-4
            "cache_strategy": self.CACHE_STRATEGIES[strat_idx],
            "disk_cache_size": int(256 * (2 ** round(p[3] * 4))), # 256MB-4GB
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        strat = config.get("cache_strategy", "lru")
        strat_idx = self.CACHE_STRATEGIES.index(strat) if strat in self.CACHE_STRATEGIES else 0
        dcs = config.get("disk_cache_size", 1024)
        dcs_idx = int(np.log2(max(dcs, 256) / 256))
        return torch.tensor([
            (config.get("n_hot_experts", 4) - 1) / 7,
            (config.get("prefetch_ahead", 2) - 1) / 3,
            strat_idx / len(self.CACHE_STRATEGIES),
            dcs_idx / 4,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        n_hot = config["n_hot_experts"]
        ahead = config["prefetch_ahead"]
        strat = config["cache_strategy"]
        dcs = config["disk_cache_size"]
        # Model: 8 experts total, each ~150MB. Hot experts in VRAM, rest on disk.
        expert_mb = 150
        vram_used = torch.tensor(n_hot * expert_mb / 1024)  # GB
        # Miss rate: fewer hot experts = more misses; prefetch + good cache reduces misses
        miss_base = torch.tensor((8 - n_hot) / 8.0)
        pf_reduction = torch.tensor(ahead * 0.08)
        strat_bonus = torch.tensor({"lru": 0.05, "lfu": 0.08, "priority": 0.12}[strat])
        cache_bonus = torch.tensor(min(dcs / 4096, 1.0) * 0.1)
        miss_rate = (miss_base - pf_reduction - strat_bonus - cache_bonus).clamp(min=0, max=1)
        score = (vram_used * (-2.0) + (1 - miss_rate) * 10.0).item()  # low VRAM + low miss
        return {
            "score": float(score),
            "behavioral": (float(vram_used), float(miss_rate)),
            "metadata": {"vram_used_gb": float(vram_used), "miss_rate": float(miss_rate)},
        }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("vram_used_gb", 10, 0.0, 1.2), ("miss_rate", 10, 0.0, 1.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "n_hot_experts": [1, 2, 3, 4, 5, 6, 7, 8],
            "prefetch_ahead": [1, 2, 3, 4],
            "cache_strategy": self.CACHE_STRATEGIES,
            "disk_cache_size": [256, 512, 1024, 2048, 4096],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        return [
            {"n_hot_experts": 8, "prefetch_ahead": 1, "cache_strategy": "lru", "disk_cache_size": 256},
            {"n_hot_experts": 4, "prefetch_ahead": 2, "cache_strategy": "priority", "disk_cache_size": 1024},
            {"n_hot_experts": 1, "prefetch_ahead": 4, "cache_strategy": "lfu", "disk_cache_size": 4096},
        ]


# ---------------------------------------------------------------------------
# 4. MemoryBudget
# ---------------------------------------------------------------------------
class MemoryBudget(BaseDomain):
    """VRAM budget allocation: kv, weight, activation, reserve fractions."""

    def name(self) -> str: return "memory_budget"
    def output_dim(self) -> int: return 4

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        return {
            "kv_budget": float(0.2 + np.clip(p[0], 0, 1) * 0.6),       # 0.2-0.8
            "weight_budget": float(0.1 + np.clip(p[1], 0, 1) * 0.4),   # 0.1-0.5
            "activation_budget": float(0.1 + np.clip(p[2], 0, 1) * 0.2), # 0.1-0.3
            "reserve": float(0.05 + np.clip(p[3], 0, 1) * 0.15),       # 0.05-0.2
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([
            (config.get("kv_budget", 0.5) - 0.2) / 0.6,
            (config.get("weight_budget", 0.3) - 0.1) / 0.4,
            (config.get("activation_budget", 0.2) - 0.1) / 0.2,
            (config.get("reserve", 0.1) - 0.05) / 0.15,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        kv = config["kv_budget"]
        wt = config["weight_budget"]
        act = config["activation_budget"]
        res = config["reserve"]
        total = torch.tensor(kv + wt + act + res)
        vram = 12.0  # GB
        # Utilization: how much of VRAM is usefully allocated (not wasted in reserve)
        utilization = torch.tensor((kv + wt + act) / vram)
        # OOM risk: if total > 1.0, we exceed VRAM
        oom_risk = torch.relu(total - 1.0) * 10.0  # penalty
        # Ideal: weights ~2.34GB/12=0.195, kv ~0.3, act ~0.15, reserve ~0.1
        ideal_wt = torch.tensor(2.34 / 12.0)
        wt_penalty = torch.abs(torch.tensor(wt) - ideal_wt) * 5.0
        # Score: high utilization, low OOM risk, close to ideal weight budget
        score = (utilization * 8.0 - oom_risk - wt_penalty).item()
        return {
            "score": float(score),
            "behavioral": (float(utilization), float(oom_risk)),
            "metadata": {"utilization": float(utilization), "oom_risk": float(oom_risk), "total_frac": float(total)},
        }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("utilization", 10, 0.0, 1.0), ("oom_risk", 10, 0.0, 5.0)]

    def seed_configs(self) -> list[dict[str, Any]]:
        return [
            {"kv_budget": 0.3, "weight_budget": 0.195, "activation_budget": 0.15, "reserve": 0.1},
            {"kv_budget": 0.5, "weight_budget": 0.3, "activation_budget": 0.2, "reserve": 0.1},
            {"kv_budget": 0.8, "weight_budget": 0.4, "activation_budget": 0.25, "reserve": 0.15},
        ]


# ---------------------------------------------------------------------------
# 5. CheckpointRecompute
# ---------------------------------------------------------------------------
class CheckpointRecompute(BaseDomain):
    """Activation checkpointing: n_checkpoint_layers, strategy, block_size."""

    RECOMPUTE_STRATEGIES = ["selective", "full", "block"]

    def name(self) -> str: return "checkpoint_recompute"
    def output_dim(self) -> int: return 3

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        strat_idx = int(p[1] * len(self.RECOMPUTE_STRATEGIES)) % len(self.RECOMPUTE_STRATEGIES)
        return {
            "n_checkpoint_layers": int(round(p[0] * 16)),          # 0-16
            "recompute_strategy": self.RECOMPUTE_STRATEGIES[strat_idx],
            "block_size": int(64 * (2 ** round(p[2] * 3))),       # 64-512
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        strat = config.get("recompute_strategy", "selective")
        strat_idx = self.RECOMPUTE_STRATEGIES.index(strat) if strat in self.RECOMPUTE_STRATEGIES else 0
        bs = config.get("block_size", 256)
        bs_idx = int(np.log2(max(bs, 64) / 64))
        return torch.tensor([
            config.get("n_checkpoint_layers", 8) / 16,
            strat_idx / len(self.RECOMPUTE_STRATEGIES),
            bs_idx / 3,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        n_ck = config["n_checkpoint_layers"]
        strat = config["recompute_strategy"]
        bs = config["block_size"]
        # Activation memory per layer (approx): batch*seq*d_model*bf16
        act_per_layer = torch.tensor(4 * 2048 * 2048 * 2 / 1e9)  # ~0.033 GB
        # Memory saved: checkpointed layers don't store activations
        mem_saved = act_per_layer * n_ck
        # Recompute overhead: full > selective > block; larger blocks = less overhead
        strat_factor = torch.tensor({"selective": 0.4, "full": 1.0, "block": 0.7}[strat])
        bs_factor = torch.tensor(1.0 / (1.0 + np.log2(max(bs, 64) / 64) * 0.2))
        overhead = (strat_factor * n_ck * 0.3 * bs_factor).clamp(min=0)
        score = (mem_saved * 20.0 - overhead * 1.5).item()
        return {
            "score": float(score),
            "behavioral": (float(mem_saved), float(overhead)),
            "metadata": {"mem_saved_gb": float(mem_saved), "recompute_overhead_ms": float(overhead)},
        }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("mem_saved_gb", 10, 0.0, 0.6), ("recompute_overhead", 10, 0.0, 5.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "n_checkpoint_layers": [0, 4, 8, 12, 16],
            "recompute_strategy": self.RECOMPUTE_STRATEGIES,
            "block_size": [64, 128, 256, 512],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        return [
            {"n_checkpoint_layers": 0, "recompute_strategy": "selective", "block_size": 256},
            {"n_checkpoint_layers": 8, "recompute_strategy": "selective", "block_size": 256},
            {"n_checkpoint_layers": 16, "recompute_strategy": "full", "block_size": 512},
        ]
