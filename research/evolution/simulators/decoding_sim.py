"""Decoding domain simulators — pure metric computation, no scoring logic."""
from __future__ import annotations
import torch
import numpy as np
from . import register


@register("spec_decode_simulate")
def spec_decode_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    nd = int(config["n_draft_tokens"])
    dr = float(config["draft_model_ratio"])
    at = float(config["acceptance_threshold"])
    temp = float(config["temperature"])
    base_accept = 0.9 ** nd
    threshold_accept = max(0.0, 1.0 - (at - 0.5) * 1.4)
    acceptance = base_accept * threshold_accept * (1 - dr * 0.3)
    temp_factor = 1.0 / (1.0 + max(0, temp - 0.7) * 0.5)
    acceptance *= temp_factor
    speedup = (1 + nd * acceptance) / (1 + dr * nd * 0.1)
    temp_penalty = 0.0
    if temp > 1.5:
        temp_penalty = (temp - 1.5) * 3
    elif temp < 0.3:
        temp_penalty = (0.3 - temp) * 5
    return {"acceptance": acceptance, "speedup": speedup,
            "temp_penalty": temp_penalty,
            "behavioral_0": acceptance, "behavioral_1": speedup}


@register("mtp_simulate")
def mtp_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    nh = int(config["n_heads"])
    lw = float(config["loss_weight"])
    sw = bool(config["share_weights"])
    dr = float(config["depth_ratio"])
    try:
        from research.decoding.mtp import MTPModule
        d_model, vocab = 128, 100
        mtp = MTPModule(d_model=d_model, vocab_size=vocab, n_heads=nh,
                        share_weights=sw).to(device)
        hidden = torch.randn(4, 8, d_model, device=device)
        targets = torch.randint(0, vocab, (4, 8), device=device)
        token_embeds = torch.randn(4, 8, d_model, device=device)
        mtp_loss, logits_list = mtp(hidden, token_embeds, targets)
        pred_acc = 1.0 / (1.0 + float(mtp_loss))
        throughput = nh * (1.0 if sw else 0.8) * dr
        signal = lw * pred_acc
    except (ImportError, Exception):
        throughput = nh * (1.0 if sw else 0.8)
        pred_acc = 1.0 / (1.0 + nh * 0.15) * dr
        signal = lw * pred_acc
    lw_penalty = 0.0
    if lw > 0.5:
        lw_penalty = (lw - 0.5) * 20
    dr_latency = dr * nh * 0.5
    throughput_norm = throughput / 4.0
    return {"throughput": throughput_norm, "pred_acc": pred_acc,
            "signal": signal, "lw_penalty": lw_penalty, "dr_latency": dr_latency,
            "behavioral_0": throughput_norm, "behavioral_1": pred_acc}


@register("batched_decode_simulate")
def batched_decode_simulate(config: dict, domain=None) -> dict:
    bs = int(config["max_batch_size"])
    ps = config["padding_strategy"]
    mw = int(config["merge_window_ms"])
    msd = int(config["max_seq_diff"])
    throughput = bs * (1 - 0.03 * bs)
    if ps == "dynamic":
        padding_waste = 0.05
    else:
        padding_waste = min(0.5, msd / 1024)
    merge_eff = min(1.0, mw / 50)
    latency_penalty = 0.0
    if mw > 50:
        latency_penalty = (mw - 50) * 0.05
    throughput_norm = throughput / 16.0
    return {"throughput": throughput_norm, "padding_waste": padding_waste,
            "merge_eff": merge_eff, "latency_penalty": latency_penalty,
            "behavioral_0": throughput_norm, "behavioral_1": padding_waste}


@register("sampling_simulate")
def sampling_simulate(config: dict, domain=None) -> dict:
    device = domain.device if domain is not None else torch.device("cpu")
    logits = torch.randn(100, device=device) * 3
    temp = float(config["temperature"])
    tp = float(config["top_p"])
    tk = int(config["top_k"])
    rp = float(config["repetition_penalty"])
    fp = float(config["frequency_penalty"])
    logits = logits / temp
    if tk > 0:
        topk_vals = logits.topk(min(tk, 100)).values
        logits = torch.where(logits < topk_vals[-1], torch.full_like(logits, -1e9), logits)
    sorted_logits = logits.sort(descending=True)
    cum_probs = torch.softmax(sorted_logits.values, dim=-1).cumsum(dim=-1)
    cutoff = (cum_probs > tp).nonzero()[0].item() if tp < 1.0 else 99
    mask = torch.zeros(100, dtype=torch.bool, device=logits.device)
    mask[sorted_logits.indices[:cutoff + 1]] = True
    logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
    probs = torch.softmax(logits, dim=-1)
    diversity = float(-(probs * torch.log(probs + 1e-8)).sum().item()) / np.log(100)
    coherence = float(probs.max().item())
    if rp <= 1.3:
        rp_benefit = (rp - 1.0) * 8
    else:
        rp_benefit = 2.4 - (rp - 1.3) * 10
    if fp <= 1.0:
        fp_benefit = fp * 3
    else:
        fp_benefit = 3.0 - (fp - 1.0) * 4
    temp_penalty = 0.0
    if temp > 1.5:
        temp_penalty = (temp - 1.5) * 4
    elif temp < 0.3:
        temp_penalty = (0.3 - temp) * 6
    return {"diversity": diversity, "coherence": coherence,
            "rp_benefit": rp_benefit, "fp_benefit": fp_benefit,
            "temp_penalty": temp_penalty,
            "behavioral_0": diversity, "behavioral_1": coherence}


@register("beam_search_simulate")
def beam_search_simulate(config: dict, domain=None) -> dict:
    bw = int(config["beam_width"])
    lp = float(config["length_penalty"])
    es = bool(config["early_stopping"])
    dp = float(config["diversity_penalty"])
    accuracy = 1.0 - 1.0 / (1.0 + bw * 0.5)
    length_factor = 1.0 - abs(lp - 1.0) * 2
    compute = bw / 8.0
    if es:
        compute *= 0.7
    if dp <= 0.7:
        diversity_bonus = dp * 3
    else:
        diversity_bonus = 2.1 - (dp - 0.7) * 5
    beam_penalty = 0.0
    if bw == 1:
        beam_penalty = 5.0
    elif bw == 2:
        beam_penalty = 1.0
    return {"accuracy": accuracy, "compute": compute,
            "length_factor": length_factor, "diversity_bonus": diversity_bonus,
            "beam_penalty": beam_penalty,
            "behavioral_0": accuracy, "behavioral_1": compute}
