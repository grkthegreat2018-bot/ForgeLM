"""Memory domain simulators — pure metric computation, no scoring logic."""
from __future__ import annotations
import torch
import numpy as np
from . import register


@register("hybrid_offload_simulate")
def hybrid_offload_simulate(config: dict, domain=None) -> dict:
    r = float(config["offload_ratio"])
    depth = int(config["prefetch_depth"])
    pin = bool(config["pin_memory"])
    overlap = bool(config["overlap_compute"])
    vram_saved = float(torch.tensor(r * 2.34))
    base_lat = float(torch.tensor(r * 8.0))
    speedup = float(torch.tensor(1.0 + np.log2(depth + 1) * 0.3 + (0.3 if overlap else 0.0)))
    pin_bonus = float(torch.tensor(0.2 if pin else 0.0))
    latency = base_lat / speedup - pin_bonus
    prefetch_mem_cost = depth * 0.05
    latency_clamped = float(max(latency, 0.0))
    return {"vram_saved": vram_saved, "latency": latency_clamped,
            "prefetch_mem_cost": prefetch_mem_cost,
            "behavioral_0": vram_saved, "behavioral_1": latency_clamped}


@register("cpu_kv_offload_simulate")
def cpu_kv_offload_simulate(config: dict, domain=None) -> dict:
    layers = int(config["offload_layers"])
    thresh = float(config["offload_threshold"])
    ps = int(config["prefetch_size"])
    async_c = bool(config["async_copy"])
    kv_total = float(torch.tensor(16 * 8 * 64 * 4096 * 2 * 2 / 1e9))
    kv_offloaded = kv_total * (layers / 16) * thresh
    base_lat = float(torch.tensor(layers * 0.5))
    pf_speedup = float(torch.tensor(1.0 + np.log2(max(ps, 64) / 64) * 0.15))
    async_bonus = float(torch.tensor(0.3 if async_c else 0.0))
    latency = max(base_lat / pf_speedup - async_bonus, 0.0)
    return {"kv_freed": kv_offloaded, "latency": latency,
            "behavioral_0": kv_offloaded, "behavioral_1": latency}


@register("expert_hotload_simulate")
def expert_hotload_simulate(config: dict, domain=None) -> dict:
    n_hot = int(config["n_hot_experts"])
    ahead = int(config["prefetch_ahead"])
    strat = config["cache_strategy"]
    dcs = int(config["disk_cache_size"])
    expert_mb = 150
    vram_used = float(torch.tensor(n_hot * expert_mb / 1024))
    miss_base = float(torch.tensor((8 - n_hot) / 8.0))
    pf_reduction = float(torch.tensor(ahead * 0.08))
    strat_bonus = float(torch.tensor({"lru": 0.05, "lfu": 0.08, "priority": 0.12}[strat]))
    cache_bonus = float(torch.tensor(min(dcs / 4096, 1.0) * 0.1))
    miss_rate = max(0.0, min(1.0, miss_base - pf_reduction - strat_bonus - cache_bonus))
    return {"vram_used": vram_used, "miss_rate": miss_rate,
            "behavioral_0": vram_used, "behavioral_1": miss_rate}


@register("memory_budget_simulate")
def memory_budget_simulate(config: dict, domain=None) -> dict:
    kv = float(config["kv_budget"])
    wt = float(config["weight_budget"])
    act = float(config["activation_budget"])
    res = float(config["reserve"])
    total = float(torch.tensor(kv + wt + act + res))
    vram = 12.0
    utilization = float(torch.tensor((kv + wt + act) / vram))
    oom_risk = float(torch.relu(torch.tensor(total - 1.0)) * 10.0)
    ideal_wt = float(torch.tensor(2.34 / 12.0))
    wt_penalty = float(torch.abs(torch.tensor(wt) - ideal_wt) * 5.0)
    return {"utilization": utilization, "oom_risk": oom_risk,
            "wt_penalty": wt_penalty,
            "behavioral_0": utilization, "behavioral_1": oom_risk}


@register("checkpoint_recompute_simulate")
def checkpoint_recompute_simulate(config: dict, domain=None) -> dict:
    n_ck = int(config["n_checkpoint_layers"])
    strat = config["recompute_strategy"]
    bs = int(config["block_size"])
    act_per_layer = float(torch.tensor(4 * 2048 * 2048 * 2 / 1e9))
    mem_saved = act_per_layer * n_ck
    strat_factor = float(torch.tensor({"selective": 0.4, "full": 1.0, "block": 0.7}[strat]))
    bs_factor = float(torch.tensor(1.0 / (1.0 + np.log2(max(bs, 64) / 64) * 0.2)))
    overhead = max(0.0, strat_factor * n_ck * 0.3 * bs_factor)
    return {"mem_saved": mem_saved, "overhead": overhead,
            "behavioral_0": mem_saved, "behavioral_1": overhead}
