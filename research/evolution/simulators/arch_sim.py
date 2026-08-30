"""Architecture simulator functions — raw metric computation only.

Scoring is handled by RewardGuard + domain JSON specs, NOT here.
Each simulator returns a dict of raw metrics (load_balance, expert_util,
param_reduction, recon_err, etc.).

These are PORT-EXACT replicas of the original domain evaluate() methods,
separated into pure metric computation (here) + declarative scoring (JSON).
"""
from __future__ import annotations

import torch

from . import register


@register("moe_routing_simulate")
def moe_routing_simulate(config: dict, domain=None) -> dict:
    """MoE: n_experts, top_k, router_mode, load_balance_weight, shared_expert.

    Metrics: load_balance, expert_util, load_balance_weight, shared_bonus.
    """
    device = domain.device if domain is not None else torch.device("cpu")
    ne = config["n_experts"]
    tk = config["top_k"]
    lbw = config["load_balance_weight"]
    se = config["shared_expert"]
    try:
        from research.inference.scheduler.moe_optim import ElbowRouter
        router_logits = torch.randn(128, ne, device=device)
        router = ElbowRouter(min_experts=1, max_experts=tk)
        expert_indices, expert_weights = router.route(router_logits)
        expert_counts = torch.zeros(ne, device=device)
        for i in range(ne):
            expert_counts[i] = (expert_indices == i).float().sum()
        balance = 1.0 - float(expert_counts.std().item() / (expert_counts.mean().item() + 1e-8))
        util = float((expert_counts > 0).float().mean().item())
    except (ImportError, Exception):
        router = torch.randn(128, ne, device=device)
        topk_vals, topk_idx = router.topk(tk, dim=-1)
        expert_counts = torch.zeros(ne, device=device)
        for i in range(ne):
            expert_counts[i] = (topk_idx == i).float().sum()
        balance = 1.0 - float(expert_counts.std().item() / (expert_counts.mean().item() + 1e-8))
        util = float((expert_counts > 0).float().mean().item())
    shared_bonus = 2.0 if se else 0.0
    return {"load_balance": balance, "expert_util": util,
            "load_balance_weight": lbw, "shared_bonus": shared_bonus,
            "behavioral_0": balance, "behavioral_1": util}


@register("factorized_embed_simulate")
def factorized_embed_simulate(config: dict, domain=None) -> dict:
    """Factorized embedding: rank, init_mode, tie_factor, vocab_size.

    Metrics: param_reduction, recon_err, total_err.
    """
    device = domain.device if domain is not None else torch.device("cpu")
    r = config["rank"]
    vs = config["vocab_size"]
    tf = config["tie_factor"]
    d = 2048
    full_params = vs * d
    tied_params = vs * r + r * d * (1 - tf * 0.5)
    reduction = 1.0 - tied_params / full_params
    d_sample = min(d, 512)
    n_sample = min(vs, 64)
    emb = torch.randn(n_sample, d_sample, device=device)
    if config["init_mode"] == "svd":
        U, S, Vh = torch.linalg.svd(emb, full_matrices=False)
        r_clip = min(r, S.shape[0])
        emb_recon = U[:, :r_clip] @ torch.diag(S[:r_clip]) @ Vh[:r_clip]
    else:
        proj = torch.randn(d_sample, r, device=device) / (d_sample ** 0.5)
        back = torch.randn(r, d_sample, device=device) / (r ** 0.5)
        emb_recon = emb @ proj @ back
    err = float((emb - emb_recon).norm().item() / (emb.norm().item() + 1e-8))
    tied_err_bonus = tf * 0.1
    total_err = err + tied_err_bonus
    # Flag adjustment: cancel out tie_factor_eval flag handler which returns
    # -0.1 * recon_err when tie_factor is truthy (nonzero).
    flag_adjustment = 0.1 * err if tf else 0.0
    return {"param_reduction": reduction, "recon_err": err,
            "total_err": total_err, "flag_adjustment": flag_adjustment,
            "behavioral_0": reduction, "behavioral_1": total_err}


@register("titan_memory_simulate")
def titan_memory_simulate(config: dict, domain=None) -> dict:
    """TITAN: memory_rank, gate_init, n_memory_slots, update_freq.

    Metrics: memory_capacity, param_ratio, gate, freshness, update_cost,
    gate_interference.
    """
    device = domain.device if domain is not None else torch.device("cpu")
    r = config["memory_rank"]
    gi = config["gate_init"]
    ns = config["n_memory_slots"]
    uf = config["update_freq"]
    d = 512
    h = torch.randn(32, d, device=device)
    memory = torch.randn(ns, r, d, device=device) * 0.01
    gate = torch.sigmoid(torch.tensor(gi))
    retrieved = (h @ memory.view(-1, d).T).mean(dim=-1) * gate
    capacity = float(retrieved.std().item() / (h.std().item() + 1e-8))
    param_ratio = (ns * r * d) / (d * d * 16)
    if uf <= 7:
        freshness = 1.0 - (uf - 1) * 0.05
        update_cost = uf * 0.02
    else:
        freshness = 0.7 - (uf - 7) * 0.08
        update_cost = 0.14
    gate_interference = 0.0
    if float(gate) > 0.5:
        gate_interference = (float(gate) - 0.5) * 10
    # Flag adjustments: cancel out update_freq_score (+min(uf,8)*0.5) and
    # freshness_model (+(freshness-0.5)*2.0) flag handlers. gate_interference
    # flag reads metrics["gate_norm"]/["main_norm"] which don't exist → 0.
    flag_adjustment = -(min(uf, 8) * 0.5 + (freshness - 0.5) * 2.0)
    return {"memory_capacity": capacity, "param_ratio": param_ratio,
            "gate": float(gate), "freshness": freshness,
            "update_cost": update_cost, "gate_interference": gate_interference,
            "flag_adjustment": flag_adjustment,
            "behavioral_0": capacity, "behavioral_1": param_ratio}


@register("ffn_skip_simulate")
def ffn_skip_simulate(config: dict, domain=None) -> dict:
    """FFN skip: threshold, n_eval_layers, strategy, min_keep.

    Metrics: compute_saved, output_dev, trivial_penalty.
    """
    device = domain.device if domain is not None else torch.device("cpu")
    st = config["skip_threshold"]
    nl = config["n_eval_layers"]
    ss = config["skip_strategy"]
    mk = config["min_keep"]
    activations = torch.randn(16, 256, 512, device=device)
    if ss == "cosine":
        sims = torch.cosine_similarity(activations[:-1].flatten(1),
                                       activations[1:].flatten(1), dim=1)
        skip_mask = torch.zeros(16, dtype=torch.bool, device=device)
        skip_mask[1:] = sims > st
    elif ss == "norm":
        norms = activations.norm(dim=(1, 2))
        skip_mask = norms < st * norms.max()
    else:
        sims = torch.cosine_similarity(activations[:-1].flatten(1),
                                       activations[1:].flatten(1), dim=1)
        norms = activations.norm(dim=(1, 2))
        skip_mask = torch.zeros(16, dtype=torch.bool, device=device)
        skip_mask[1:] = sims > st
        skip_mask = skip_mask | (norms < st * norms.max())
    n_keep = max(int(mk * 16), 1)
    if skip_mask.sum() > 16 - n_keep:
        skip_mask[:16 - n_keep] = True
        skip_mask[16 - n_keep:] = False
    compute_saved = float(skip_mask.float().mean().item())
    full_out = activations.sum(dim=0)
    skipped_out = activations[~skip_mask].sum(dim=0) if skip_mask.any() else full_out
    dev = float((full_out - skipped_out).norm().item() / (full_out.norm().item() + 1e-8))
    trivial_penalty = 0.0
    if compute_saved < 0.05:
        trivial_penalty = -50.0 - (compute_saved * 20 - dev * 30)
    return {"compute_saved": compute_saved, "output_dev": dev,
            "trivial_penalty": trivial_penalty,
            "behavioral_0": compute_saved, "behavioral_1": dev}


@register("conv_simulate")
def conv_simulate(config: dict, domain=None) -> dict:
    """Conv: kernel_size, stride, dilation, groups, n_conv_layers.

    Metrics: receptive_field (normalized), param_ratio, out_quality.
    """
    device = domain.device if domain is not None else torch.device("cpu")
    ks = config["kernel_size"]
    st = config["stride"]
    dl = config["dilation"]
    gr = config["groups"]
    nl = config["n_conv_layers"]
    rf = ks * dl * nl
    d = 256
    params = ks * ks * d * d / gr * nl
    param_ratio = params / (d * d * 16)
    x = torch.randn(1, d, 32, 32, device=device)
    gr_safe = min(gr, d)
    while d % gr_safe != 0:
        gr_safe -= 1
    conv = torch.nn.Conv2d(d, d, ks, stride=st, dilation=dl, groups=gr_safe,
                           padding=ks * dl // 2)
    try:
        out = conv(x)
        out_quality = float(out.std().item() / (x.std().item() + 1e-8))
    except (RuntimeError, ValueError, TypeError):
        out_quality = 0.0
    rf_norm = min(1.0, rf / 100)
    return {"receptive_field": rf_norm, "param_ratio": param_ratio,
            "out_quality": out_quality,
            "behavioral_0": rf_norm, "behavioral_1": param_ratio}
