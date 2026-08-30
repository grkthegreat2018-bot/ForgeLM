"""KV domain simulators — pure metric computation, no scoring logic.

Each simulator is a port-exact replica of the original evaluate() metric
computation. Scoring formulas live in the JSON specs.
"""
from __future__ import annotations
import torch
import numpy as np
from . import register


def _hadamard(n: int, device=None) -> torch.Tensor:
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / np.sqrt(n)


def _quant_error(x: torch.Tensor, bits: int) -> float:
    levels = 2 ** bits
    xmax = x.abs().amax().clamp(min=1e-8)
    x_q = torch.round(x / xmax * (levels / 2 - 1)) / (levels / 2 - 1) * xmax
    return float((x - x_q).norm() / x.norm().clamp(min=1e-8))


@register("rotor_quant_kv_simulate")
def rotor_quant_kv_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    d = 64
    g = torch.Generator(device=device).manual_seed(42)
    K = (torch.randn(512, 8, d, device=device, generator=g) * 0.1).clone()
    for _ in range(int(config["n_rotations"])):
        if config["rot_type"] == "hadamard":
            H = _hadamard(d, device=device); K = torch.einsum("shd,df->shf", K, H)
        elif config["rot_type"] == "dct":
            n = torch.arange(d, dtype=torch.float32, device=device)
            D = torch.cos(np.pi * (2 * n.unsqueeze(0) + 1) * n.unsqueeze(1) / (2 * d)) / np.sqrt(d)
            K = torch.einsum("shd,df->shf", K, D)
        else:
            R = torch.randn(d, d, device=device) / np.sqrt(d); K = torch.einsum("shd,df->shf", K, R)
    err = _quant_error(K, int(config["quant_bits"]))
    comp = 16.0 / int(config["quant_bits"])
    compute = int(config["n_rotations"]) * (1.0 if config["rot_type"] == "hadamard" else 2.5)
    return {"compression": comp, "quant_error": err, "compute": compute,
            "behavioral_0": comp, "behavioral_1": err}


@register("hadamard_kv_simulate")
def hadamard_kv_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    d = 64
    g = torch.Generator(device=device).manual_seed(7)
    K = (torch.randn(512, 8, d, device=device, generator=g) * 0.1).clone()
    hd = min(int(config["hadamard_dim"]), d)
    err_orig = _quant_error(K, int(config["quant_bits"]))
    for _ in range(int(config["n_apply"])):
        H = _hadamard(hd, device=device)
        K[..., :hd] = torch.einsum("shd,df->shf", K[..., :hd], H)
    err_rot = _quant_error(K, int(config["quant_bits"]))
    err_reduction = max(0.0, err_orig - err_rot)
    compute = int(config["n_apply"]) * hd / 64.0
    comp = 16.0 / int(config["quant_bits"])
    return {"err_reduction": err_reduction, "compute_cost": compute,
            "compression": comp, "err_orig": err_orig, "err_rot": err_rot,
            "behavioral_0": err_reduction, "behavioral_1": compute}


@register("streaming_kv_simulate")
def streaming_kv_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    seq_len = 4096
    g = torch.Generator(device=device).manual_seed(11)
    K = torch.randn(seq_len, 8, 64, device=device, generator=g) * 0.1
    chunk, n_sink, overlap = int(config["chunk_size"]), int(config["n_sink"]), float(config["overlap"])
    step = max(1, int(chunk * (1 - overlap)))
    n_chunks = max(1, (seq_len - n_sink) // step)
    covered = n_sink + n_chunks * chunk
    coverage = min(1.0, covered / seq_len)
    mem_ratio = (n_sink + chunk * (1 + overlap)) / seq_len
    sink_idx = torch.linspace(0, seq_len - 1, n_sink).long()
    chunk_start = seq_len - chunk
    kept_idx = torch.cat([sink_idx, torch.arange(chunk_start, seq_len)])
    kept = K[kept_idx]
    coverage_err = 1.0 - float(kept.std() / K.std().clamp(min=1e-8))
    return {"coverage": coverage, "mem_ratio": mem_ratio, "coverage_err": coverage_err,
            "behavioral_0": coverage, "behavioral_1": mem_ratio}


@register("kv_zip_kv_simulate")
def kv_zip_kv_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    d = 64
    g = torch.Generator(device=device).manual_seed(23)
    K = (torch.randn(512 * 8, d, device=device, generator=g) * 0.1)[:256]
    n_sub = K.shape[0]
    cb_size = min(int(config["codebook_size"]), n_sub)
    idx = torch.randperm(n_sub, device=K.device)[:cb_size]
    codebook = K[idx].clone()
    K_sub = K
    for _ in range(min(int(config["n_iter"]), 5)):
        dist = torch.cdist(K_sub, codebook)
        assign = dist.argmin(dim=1)
        one_hot = torch.zeros(K_sub.shape[0], cb_size, device=K.device)
        one_hot.scatter_(1, assign.unsqueeze(1), 1.0)
        counts = one_hot.sum(dim=0).clamp(min=1)
        codebook = (one_hot.T @ K_sub) / counts.unsqueeze(1)
    dist_full = torch.cdist(K_sub, codebook)
    recon = codebook[dist_full.argmin(dim=1)]
    recon_err = float((K_sub - recon).norm() / K_sub.norm().clamp(min=1e-8))
    mem_bytes = cb_size * d * 2 + 1024 * 4
    orig_bytes = 1024 * d * 2
    actual_comp = orig_bytes / max(mem_bytes, 1)
    return {"actual_comp": actual_comp, "recon_err": recon_err,
            "compression_ratio": int(config["compression_ratio"]),
            "codebook_size": cb_size, "n_iter": int(config["n_iter"]),
            "behavioral_0": actual_comp, "behavioral_1": recon_err}


@register("xquant_kv_simulate")
def xquant_kv_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    n_layers, seq_len = 16, 2048
    ratio = float(config["recomputation_ratio"])
    bits = int(config["quant_bits"])
    interval = int(config["checkpoint_interval"])
    n_recompute = int(n_layers * ratio)
    n_stored = n_layers - n_recompute
    mem_full = n_layers * seq_len * 64 * 2
    mem_quant = n_stored * seq_len * 64 * (bits / 16)
    mem_ratio = mem_quant / mem_full
    recompute_cost = n_recompute / interval
    if ratio >= 1.0:
        inference_penalty = 50.0
    elif ratio > 0.5:
        inference_penalty = (ratio - 0.5) * 60
    else:
        inference_penalty = ratio * 10
    if n_stored > 0:
        g = torch.Generator(device=device).manual_seed(31)
        errs = []
        for i in range(min(n_stored, 4)):
            K_i = torch.randn(seq_len, 8, 64, device=device, generator=g) * 0.1
            errs.append(_quant_error(K_i[:512], bits))
        err = float(np.mean(errs)) if errs else 0.0
    else:
        err = 0.0
    memory_saved = 1.0 - mem_ratio
    return {"memory_saved": memory_saved, "recompute_cost": recompute_cost,
            "mem_ratio": mem_ratio, "quant_err": err,
            "inference_penalty": inference_penalty,
            "behavioral_0": memory_saved, "behavioral_1": recompute_cost}


@register("kv_recompute_simulate")
def kv_recompute_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    n_layers = 16
    n_recomp = int(config["recompute_layers"])
    strat = config["recompute_strategy"]
    threshold = float(config["threshold"])
    g = torch.Generator(device=device).manual_seed(41)
    K_list = [torch.randn(2048, 8, 64, device=device, generator=g) * 0.1 for _ in range(n_layers)]
    layer_entropy = [float(k.var(dim=0).mean()) for k in K_list]
    if strat == "selective":
        entropies = torch.tensor(layer_entropy)
        norm_ent = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
        recomp_mask = norm_ent < threshold
        n_actual = min(n_recomp, int(recomp_mask.sum()))
    else:
        n_actual = min(n_recomp, n_layers)
    n_stored = n_layers - n_actual
    mem_saved_ratio = n_actual / n_layers
    cost_per_layer = 2.0 if strat == "full" else 1.3
    compute_cost = n_actual * cost_per_layer
    quality = 1.0 - n_stored * 0.01
    recomp_ratio = n_actual / n_layers
    if recomp_ratio >= 1.0:
        inference_penalty = 40.0
    elif recomp_ratio > 0.5:
        inference_penalty = (recomp_ratio - 0.5) * 50
    else:
        inference_penalty = recomp_ratio * 8
    return {"memory_saved": mem_saved_ratio, "compute_cost": compute_cost,
            "quality": quality, "inference_penalty": inference_penalty,
            "behavioral_0": mem_saved_ratio, "behavioral_1": compute_cost}


@register("cross_layer_kv_simulate")
def cross_layer_kv_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    n_layers, seq_len, d, n_heads = 16, 512, 64, 8
    g = torch.Generator(device=device).manual_seed(53)
    K = (torch.randn(n_layers, seq_len, n_heads, d, device=device, generator=g) * 0.1).clone()
    ratio = float(config["share_ratio"])
    n_groups = int(config["n_share_groups"])
    mode = config["share_mode"]
    group_size = max(1, n_layers // n_groups)
    n_shared = 0
    recon_err = 0.0
    for gi in range(n_groups):
        start = gi * group_size
        end = min(start + group_size, n_layers)
        group = K[start:end]
        if mode == "avg":
            shared = group.mean(dim=0, keepdim=True)
        elif mode == "max":
            shared_idx = group.abs().argmax(dim=0)
            shared = group.gather(0, shared_idx.unsqueeze(0))
        else:
            group_flat = group.reshape(group.shape[0], -1)
            U, S, Vh = torch.linalg.svd(group_flat, full_matrices=False)
            weights = torch.softmax(S, dim=0)
            shared_flat = (weights.unsqueeze(0) @ U.T @ group_flat)
            shared = shared_flat.reshape(1, *group.shape[1:])
        n_share_in_group = max(1, int((end - start) * ratio))
        target = group[-n_share_in_group:]
        recon = shared.expand(n_share_in_group, *group.shape[1:])
        recon_err += float((target - recon).norm().item() / (target.norm().item() + 1e-8))
        n_shared += n_share_in_group
    param_reduction = n_shared / n_layers
    recon_err /= max(1, n_groups)
    overhead = 1.5 if mode == "learned" else (1.2 if mode == "max" else 1.0)
    return {"param_reduction": param_reduction, "recon_error": recon_err,
            "overhead": overhead,
            "behavioral_0": param_reduction, "behavioral_1": recon_err}


@register("paged_evict_kv_simulate")
def paged_evict_kv_simulate(config: dict, domain=None) -> dict:
    seq_len = 4096
    page_size = int(config["page_size"])
    n_pages = int(config["n_pages"])
    policy = config["eviction_policy"]
    rng = np.random.RandomState(67)
    access_pattern = rng.zipf(1.5, size=seq_len).clip(1, 100)
    total_tokens = seq_len
    capacity = page_size * n_pages
    mem_eff = min(1.0, total_tokens / capacity) if capacity > 0 else 0.0
    n_total_pages = (total_tokens + page_size - 1) // page_size
    if n_total_pages <= n_pages:
        hit_rate = 1.0
    else:
        page_importance = []
        for i in range(n_total_pages):
            start = i * page_size
            end = min(start + page_size, total_tokens)
            page_importance.append(float(access_pattern[start:end].mean()))
        page_importance = np.array(page_importance)
        if policy == "lru":
            hit_rate = n_pages / n_total_pages
        elif policy == "lfu":
            top = np.argsort(page_importance)[-n_pages:]
            hit_rate = float(page_importance[top].sum() / max(page_importance.sum(), 1))
        else:
            top = np.argsort(page_importance)[-n_pages:]
            hit_rate = float(page_importance[top].sum() / max(page_importance.sum(), 1))
    overhead = n_pages * 0.001
    return {"hit_rate": hit_rate, "mem_eff": mem_eff, "overhead": overhead,
            "page_size": page_size,
            "behavioral_0": hit_rate, "behavioral_1": mem_eff}
