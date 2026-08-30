"""Training simulator functions — raw metric computation only.

Scoring is handled by RewardGuard + domain JSON specs, NOT here.
Each simulator returns a dict of raw metrics (convergence_speed, stability,
throughput, etc.).

These are PORT-EXACT replicas of the original domain evaluate() methods,
separated into pure metric computation (here) + declarative scoring (JSON).
"""
from __future__ import annotations

import numpy as np
import torch

from . import register


@register("optimizer_simulate")
def optimizer_simulate(config: dict, domain=None) -> dict:
    """Optimizer type, lr, betas, weight_decay. Metrics: convergence_speed, stability."""
    device = domain.device if domain is not None else torch.device("cpu")
    x = torch.tensor([3.0, -2.0, 1.5], device=device)
    lr = config["lr"]
    b1, b2 = config["beta1"], config["beta2"]
    wd = config["weight_decay"]
    m = torch.zeros(3, device=device)
    v = torch.zeros(3, device=device)
    losses = torch.zeros(50, device=device)
    for step in range(50):
        loss = (x ** 2).sum() + wd * (x ** 2).sum()
        grad = 2 * x + 2 * wd * x
        if config["opt_type"] == "adamw":
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            x = x - lr * m / (v.sqrt() + 1e-8)
        elif config["opt_type"] == "muon":
            g = grad
            g_norm = g.norm() + 1e-8
            g_orth = g / g_norm
            x = x - lr * g_orth
        elif config["opt_type"] == "lion":
            m = b1 * m + (1 - b1) * grad
            x = x - lr * torch.sign(m)
        else:  # sgd
            x = x - lr * grad
        losses[step] = loss
    conv_speed = 1.0 / (1.0 + float(losses[-1].item()))
    stability = 1.0 / (1.0 + float(losses[-10:].std().item()))
    return {"convergence_speed": conv_speed, "stability": stability,
            "behavioral_0": conv_speed, "behavioral_1": stability}


@register("scheduler_simulate")
def scheduler_simulate(config: dict, domain=None) -> dict:
    """Scheduler type, warmup, min_lr_ratio, decay_steps. Metrics: auc, final_decay, early_lr_jump."""
    device = domain.device if domain is not None else torch.device("cpu")
    base_lr = 1e-3
    ws = config["warmup_steps"]
    ds = config["decay_steps"]
    mr = config["min_lr_ratio"]
    lrs = []
    for step in range(ds):
        if step < ws:
            lr = base_lr * step / max(ws, 1)
        elif config["sched_type"] == "cosine":
            progress = (step - ws) / max(ds - ws, 1)
            lr = base_lr * (mr + (1 - mr) * 0.5 * (1 + np.cos(np.pi * progress)))
        elif config["sched_type"] == "linear":
            progress = (step - ws) / max(ds - ws, 1)
            lr = base_lr * (1 - progress * (1 - mr))
        elif config["sched_type"] == "warmup_decay":
            lr = base_lr * mr ** ((step - ws) / max(ds - ws, 1))
        else:
            lr = base_lr
        lrs.append(lr)
    auc = float(np.sum(lrs) / (base_lr * ds))
    final_decay = 1.0 - lrs[-1] / base_lr
    early_lr_jump = 0.0
    if ws > 0:
        early_lr_jump = abs(lrs[1] - lrs[0]) / base_lr
    return {"area_under_curve": auc, "final_decay": final_decay,
            "early_lr_jump": early_lr_jump,
            "behavioral_0": auc, "behavioral_1": final_decay}


@register("loss_simulate")
def loss_simulate(config: dict, domain=None) -> dict:
    """Loss type, label_smoothing, focal_gamma, temperature. Metrics: gradient_magnitude, smoothness, focus_ratio."""
    device = domain.device if domain is not None else torch.device("cpu")
    logits = torch.randn(32, 100, device=device) * 2
    targets = torch.randint(0, 100, (32,), device=device)
    temp = config["temperature"]
    ls = config["label_smoothing"]
    fg = config["focal_gamma"]
    logits_t = logits / temp
    if config["loss_type"] == "ce":
        loss = torch.nn.functional.cross_entropy(logits_t, targets)
    elif config["loss_type"] == "focal":
        ce = torch.nn.functional.cross_entropy(logits_t, targets, reduction="none")
        loss = ((1 - torch.exp(-ce)) ** fg * ce).mean()
    elif config["loss_type"] == "label_smooth":
        loss = torch.nn.functional.cross_entropy(logits_t, targets, label_smoothing=ls)
    else:  # kl
        log_p = torch.log_softmax(logits_t, dim=-1)
        p = torch.softmax(logits_t, dim=-1)
        target_dist = torch.zeros_like(p)
        target_dist[range(32), targets] = 1
        loss = (target_dist * (torch.log(target_dist + 1e-8) - log_p)).sum(dim=-1).mean()
    probs = torch.softmax(logits_t, dim=-1)
    grad_mag = float((probs * (1 - probs)).mean().item())
    smoothness = 1.0 / (1.0 + float(loss.item()))
    correct_mask = probs.argmax(dim=-1) == targets
    wrong_probs = probs[~correct_mask]
    correct_probs = probs[correct_mask].gather(1, targets[correct_mask].unsqueeze(1))
    if len(wrong_probs) > 0 and len(correct_probs) > 1:
        wrong_grad = float(wrong_probs.var(unbiased=False).item())
        correct_grad = float(correct_probs.var(unbiased=False).item())
        focus_ratio = wrong_grad / (correct_grad + 1e-8)
    else:
        focus_ratio = 1.0
    return {"gradient_magnitude": grad_mag, "smoothness": smoothness,
            "focus_ratio": focus_ratio,
            "behavioral_0": grad_mag, "behavioral_1": smoothness}


@register("muon_simulate")
def muon_simulate(config: dict, domain=None) -> dict:
    """Muon momentum, nesterov, weight_decay, ns_steps. Metrics: convergence, orthogonality."""
    device = domain.device if domain is not None else torch.device("cpu")
    x = torch.tensor([3.0, -2.0, 1.5, -1.0, 0.5], device=device)
    mom = config["momentum"]
    wd = config["weight_decay"]
    ns = min(config["ns_steps"], 3)
    buf = torch.zeros(5, device=device)
    final_loss = torch.tensor(0.0, device=device)
    for _ in range(10):
        loss = (x ** 2).sum()
        grad = 2 * x + 2 * wd * x
        buf = mom * buf + grad
        g = buf
        for _ in range(ns):
            g_norm = g.norm() + 1e-8
            g = g * (3 / g_norm) - g * (g @ g) / (g_norm ** 3)
        x = x - 0.01 * g
        final_loss = loss
    conv = 1.0 / (1.0 + float(final_loss.item()))
    orth = 1.0 / (1.0 + abs(float((buf @ buf).item()) - float((buf.norm() ** 2).item())))
    return {"convergence": conv, "orthogonality": orth,
            "behavioral_0": conv, "behavioral_1": orth}


@register("cpu_adamw_simulate")
def cpu_adamw_simulate(config: dict, domain=None) -> dict:
    """CPU offload ratio, prefetch depth, compression, update_freq. Metrics: throughput, latency."""
    oratio = config["offload_ratio"]
    pd = config["prefetch_depth"]
    comp = config["compression"]
    uf = config["update_freq"]
    base_throughput = 1000
    comp_factor = {"none": 1.0, "int8": 2.0, "int4": 3.0}[comp]
    throughput = base_throughput * (1 - oratio * 0.5) * comp_factor
    latency = oratio * 50 / max(pd, 1) * (1 / comp_factor) * uf
    return {"throughput": throughput, "latency": latency,
            "throughput_norm": throughput / 1000, "latency_ms": latency,
            "behavioral_0": throughput / 1000, "behavioral_1": latency}


@register("grad_accum_simulate")
def grad_accum_simulate(config: dict, domain=None) -> dict:
    """Gradient accumulation steps, micro_batch, grad_clip, sync_freq. Metrics: throughput, noise, effective_batch, grad_clip."""
    as_ = config["accum_steps"]
    mb = config["micro_batch"]
    gc = config["grad_clip"]
    eff_batch = as_ * mb
    noise = 1.0 / (eff_batch ** 0.5)
    throughput = mb / (mb + as_ * 0.1)
    return {"throughput": throughput, "noise": noise,
            "effective_batch": eff_batch / 512, "grad_clip": gc,
            "behavioral_0": eff_batch / 512, "behavioral_1": noise}


@register("fp8_training_simulate")
def fp8_training_simulate(config: dict, domain=None) -> dict:
    """FP8 autocast mode, smooth_swiglu, mu_scaling, loss_scale. Metrics: dynamic_range, overflow, quant_err."""
    device = domain.device if domain is not None else torch.device("cpu")
    if config["autocast_mode"] == "e4m3":
        max_val = 448
        mantissa_bits = 3
    else:
        max_val = 57344
        mantissa_bits = 2
    x = torch.randn(256, 512, device=device) * 0.02
    if config["mu_scaling"]:
        x = x / (x.std() + 1e-8)
    if config["smooth_swiglu"]:
        x = x / (x.abs().max(dim=-1, keepdim=True).values + 1e-8) * max_val * 0.5
    overflow = float((x.abs() > max_val).float().mean().item())
    dynamic_range = float(x.abs().max().item() / (x.abs().mean().item() + 1e-8))
    x_clamped = x.clamp(-max_val, max_val)
    scale = x_clamped.abs().max() / max_val if x_clamped.abs().max() > 0 else torch.tensor(1.0, device=device)
    x_scaled = x_clamped / scale
    n_levels = 2 ** mantissa_bits
    x_quantized = torch.round(x_scaled * n_levels) / n_levels
    x_dequantized = x_quantized * scale
    quant_err = float((x - x_dequantized).norm().item() / (x.norm().item() + 1e-8))
    mu_scaling_bonus = 3.0 if config["mu_scaling"] else 0.0
    smooth_swiglu_bonus = 2.0 if config["smooth_swiglu"] else 0.0
    loss_scale_bonus = 2.0 if config["loss_scale"] > 1024 else 0.0
    return {"dynamic_range": dynamic_range, "overflow": overflow,
            "quant_err": quant_err, "fp8_quant_error": quant_err,
            "mu_scaling_bonus": mu_scaling_bonus,
            "smooth_swiglu_bonus": smooth_swiglu_bonus,
            "loss_scale_bonus": loss_scale_bonus,
            "dynamic_range_norm": dynamic_range / 100, "overflow_risk": overflow,
            "behavioral_0": dynamic_range / 100, "behavioral_1": overflow}


@register("mod_simulate")
def mod_simulate(config: dict, domain=None) -> dict:
    """MoD keep_fraction, router_type, aux_loss_weight, n_skip_layers.
    Metrics: compute_saved, accuracy_loss, base_score (router-adjusted).
    """
    device = domain.device if domain is not None else torch.device("cpu")
    kf = config["keep_fraction"]
    nsl = config["n_skip_layers"]
    alw = config["aux_loss_weight"]
    rt = config["router_type"]
    tokens = torch.randn(128, 512, device=device)
    router = torch.randn(128, device=device)
    keep_mask = router > torch.quantile(router, 1 - kf)
    compute_saved = 1.0 - kf
    kept = tokens[keep_mask]
    acc_loss = 1.0 - float(kept.std().item() / (tokens.std().item() + 1e-8))
    # Router quality multiplier (mlp=1.0, linear=0.85) — applied to base score
    router_quality = 1.0 if rt == "mlp" else 0.85
    base_score = (compute_saved * 20 - acc_loss * 10) * router_quality
    return {"compute_saved": compute_saved, "accuracy_loss": acc_loss,
            "base_score": base_score,
            "behavioral_0": compute_saved, "behavioral_1": acc_loss}


@register("apollo_simulate")
def apollo_simulate(config: dict, domain=None) -> dict:
    """APOLLO optimizer: rank, scale_mode, lr_scale. Metrics: memory_savings, convergence_quality, rank_penalty, scale_bonus, trivial_penalty."""
    device = domain.device if domain is not None else torch.device("cpu")
    rank = config["rank"]
    scale = config["scale"]
    lr_scale = config["lr_scale"]
    cols = 4096
    x = torch.tensor([3.0, -2.0, 1.5, -1.0, 0.5], device=device)
    lr = 0.01 * lr_scale
    proj_factor = min(rank / 16.0, 1.0)
    losses = torch.zeros(30, device=device)
    for step in range(30):
        loss = (x ** 2).sum()
        grad = 2 * x
        x = x - lr * grad * (0.5 + proj_factor)
        losses[step] = loss
    conv_quality = 1.0 / (1.0 + float(losses[-1].item()))
    memory_savings = (8 - 2 * rank / cols) / 8
    memory_savings = float(np.clip(memory_savings, 0, 1))
    rank_penalty = 0.0
    if rank > 32:
        rank_penalty = (rank - 32) * 0.3
    if rank == 1:
        rank_penalty -= 2.0
    scale_bonus = 2.0 if scale == "channel" else 0.0
    trivial_penalty = 0.0
    if rank >= 64 and scale == "channel" and lr_scale > 8.0:
        trivial_penalty = 15.0
    return {"memory_savings": memory_savings, "convergence_quality": conv_quality,
            "rank_penalty": rank_penalty, "scale_bonus": scale_bonus,
            "trivial_penalty": trivial_penalty,
            "behavioral_0": memory_savings, "behavioral_1": conv_quality}


@register("bread_simulate")
def bread_simulate(config: dict, domain=None) -> dict:
    """BREAD landscape correction: correction_mode, sgd_lr_scale. Metrics: mode_bonus, lr_scale_quality, stability."""
    device = domain.device if domain is not None else torch.device("cpu")
    mode = config["correction_mode"]
    sgd_lr_scale = config["sgd_lr_scale"]
    x = torch.tensor([3.0, -2.0, 1.5, -1.0, 0.5], device=device)
    lr = 0.01 * sgd_lr_scale
    losses = torch.zeros(20, device=device)
    for step in range(20):
        loss = (x ** 2).sum()
        grad = 2 * x
        x = x - lr * grad
        losses[step] = loss
    stability = 1.0 / (1.0 + float(losses[-10:].std().item()))
    mode_bonus = 0.0
    if mode == "disabled":
        mode_bonus = 0.0
    elif mode == "partial":
        mode_bonus = 15.0
    else:  # "all"
        mode_bonus = -5.0
    lr_scale_quality = 0.0
    if sgd_lr_scale < 2.0:
        lr_scale_quality = -(2.0 - sgd_lr_scale) * 3
    elif sgd_lr_scale > 8.0:
        lr_scale_quality = -(sgd_lr_scale - 8.0) * 3
    else:
        lr_scale_quality = 5.0 - abs(sgd_lr_scale - 5.0) * 0.5
    return {"mode_bonus": mode_bonus, "lr_scale_quality": lr_scale_quality,
            "stability": stability, "correction_quality": mode_bonus / 15.0,
            "behavioral_0": mode_bonus / 15.0, "behavioral_1": stability}


@register("flash_optim_simulate")
def flash_optim_simulate(config: dict, domain=None) -> dict:
    """FlashOptim companded optimizer states: bits, companding_mode, strength, quantile_threshold. Metrics: memory_savings, precision, companding_bonus, bit_penalty."""
    device = domain.device if domain is not None else torch.device("cpu")
    bits = config["bits"]
    companding = config["companding"]
    strength = float(config.get("companding_strength", 1.0))
    quantile_thresh = float(config.get("quantile_threshold", 0.0))
    state = torch.randn(256, 512, device=device) * 0.1
    # Apply quantile-based clipping to handle outliers
    if quantile_thresh > 0:
        q = torch.quantile(state.abs().flatten(), 1.0 - quantile_thresh)
        state = state.clamp(-q, q)
    if companding == "sqrt":
        transformed = torch.sign(state) * (state.abs() ** (1.0 / (1.0 + strength)))
    elif companding == "log":
        transformed = torch.sign(state) * torch.log1p(state.abs() * strength) / max(strength, 1e-8)
    else:  # linear
        transformed = state
    n_levels = 2 ** bits
    max_val = transformed.abs().max().item() + 1e-8
    scale = max_val / n_levels
    quantized = torch.round(transformed / scale) * scale
    if companding == "sqrt":
        dequantized = torch.sign(quantized) * (quantized.abs() ** (1.0 + strength))
    elif companding == "log":
        dequantized = torch.sign(quantized) * (torch.expm1(quantized.abs() * strength) - 1.0) / max(strength, 1e-8)
    else:
        dequantized = quantized
    quant_err = float((state - dequantized).norm().item() / (state.norm().item() + 1e-8))
    precision = 1.0 / (1.0 + quant_err * 10)
    memory_savings = 0.75 if bits == 8 else 0.875
    # Strength bonus: optimal around 1.0, penalize extremes
    strength_bonus = 5.0 - abs(strength - 1.0) * 3.0
    companding_bonus = (5.0 if companding == "sqrt" else (3.0 if companding == "log" else 0.0)) + strength_bonus
    bit_penalty = -10.0 if bits == 4 else 0.0
    # Quantile clipping reduces error but also reduces effective precision
    quantile_bonus = quantile_thresh * 20.0 if quantile_thresh > 0 else 0.0
    return {"memory_savings": memory_savings, "precision": precision,
            "companding_bonus": companding_bonus, "bit_penalty": bit_penalty,
            "quantile_bonus": quantile_bonus,
            "behavioral_0": memory_savings, "behavioral_1": precision}


@register("triton_kernel_simulate")
def triton_kernel_simulate(config: dict, domain=None) -> dict:
    """Triton fused kernel block sizes: rms_block_size, swiglu_block_size. Metrics: base_speedup, mismatch_penalty, rms_match, swiglu_match."""
    device = domain.device if domain is not None else torch.device("cpu")
    rms_block = config["rms_block_size"]
    swiglu_block = config["swiglu_block_size"]
    rms_target = 4096
    swiglu_target = 16384
    _ = torch.randn(4096, device=device)  # match original randn call
    base_speedup = 20.0
    rms_mismatch = 0.0
    if rms_block < rms_target:
        rms_mismatch = (rms_target / rms_block) * 3
    elif rms_block > rms_target:
        rms_mismatch = (rms_block / rms_target) * 0.5
    swiglu_mismatch = 0.0
    if swiglu_block < swiglu_target:
        swiglu_mismatch = (swiglu_target / swiglu_block) * 3
    elif swiglu_block > swiglu_target:
        swiglu_mismatch = (swiglu_block / swiglu_target) * 0.5
    mismatch_penalty = rms_mismatch + swiglu_mismatch
    rms_match = 1.0 / (1.0 + rms_mismatch)
    swiglu_match = 1.0 / (1.0 + swiglu_mismatch)
    return {"base_speedup": base_speedup, "mismatch_penalty": mismatch_penalty,
            "rms_match": rms_match, "swiglu_match": swiglu_match,
            "behavioral_0": rms_match, "behavioral_1": swiglu_match}


@register("varlen_simulate")
def varlen_simulate(config: dict, domain=None) -> dict:
    """Varlen attention for packed sequences: use_varlen boolean. Metrics: varlen_bonus, speedup, vram_savings."""
    device = domain.device if domain is not None else torch.device("cpu")
    use_varlen = config["use_varlen"]
    _ = torch.randn(128, 64, device=device)  # match original randn call
    if use_varlen:
        speedup = 2.1
        vram_savings = 0.5
        varlen_bonus = 25.0
    else:
        speedup = 1.0
        vram_savings = 0.0
        varlen_bonus = 0.0
    return {"varlen_bonus": varlen_bonus,
            "speedup_norm": speedup / 3.0, "vram_savings": vram_savings,
            "behavioral_0": speedup / 3.0, "behavioral_1": vram_savings}
