"""Decoding-related evolution domains (5 domains).

Each domain explores a decoding configuration search space.
All use small tensor operations for fast evaluation (no model loading).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any
from . import BaseDomain


class SpeculativeDecode(BaseDomain):
    """Speculative decoding: n_draft_tokens, draft_model_ratio, acceptance_threshold."""
    def name(self): return "speculative_decode"
    def output_dim(self): return 4
    def behavioral_dims(self):
        return [("acceptance_rate", 10, 0, 1), ("speedup", 10, 0, 4)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_draft_tokens": int(np.interp(p[0], [0, 1], [1, 8])),
                "draft_model_ratio": float(np.interp(p[1], [0, 1], [0.1, 0.5])),
                "acceptance_threshold": float(np.interp(p[2], [0, 1], [0.5, 0.95])),
                "temperature": float(np.interp(p[3], [0, 1], [0, 1]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("n_draft_tokens", 4) - 1) / 7, 0, 1),
            np.clip((config.get("draft_model_ratio", 0.3) - 0.1) / 0.4, 0, 1),
            np.clip((config.get("acceptance_threshold", 0.8) - 0.5) / 0.45, 0, 1),
            config.get("temperature", 0.7),
        ], dtype=torch.float32)
    def evaluate(self, config):
        nd = config["n_draft_tokens"]
        dr = config["draft_model_ratio"]
        at = config["acceptance_threshold"]
        temp = config["temperature"]
        # Acceptance rate: higher threshold = STRICTER acceptance = FEWER tokens accepted.
        # The old formula (at ** nd) was backwards — it increased acceptance with threshold.
        # Correct model: acceptance_threshold is the min probability to accept a draft token.
        # Higher threshold → reject more → lower acceptance rate.
        # Base acceptance decreases with more draft tokens (compounding error).
        base_accept = 0.9 ** nd  # base acceptance without threshold filtering
        # Threshold filtering: fraction of tokens with prob > threshold
        # Model: draft token probs follow a distribution where higher threshold
        # rejects more. At threshold=0.5, ~70% accepted; at 0.95, ~30%.
        threshold_accept = max(0.0, 1.0 - (at - 0.5) * 1.4)  # 0.5→1.0, 0.95→0.37
        acceptance = base_accept * threshold_accept * (1 - dr * 0.3)
        # Temperature effect: high temperature makes draft model less certain,
        # reducing acceptance. Low temperature = more confident = higher acceptance
        # but risk of degenerate repetition.
        temp_factor = 1.0 / (1.0 + max(0, temp - 0.7) * 0.5)  # temp<=0.7 → 1.0, temp=2.0 → 0.55
        acceptance *= temp_factor
        # Speedup = (1 + nd * acceptance) / (1 + dr * nd * 0.1)
        speedup = (1 + nd * acceptance) / (1 + dr * nd * 0.1)
        # Penalize extreme temperatures: temp>1.5 causes incoherent drafts,
        # temp<0.3 causes degenerate repetition (no diversity in drafts)
        temp_penalty = 0.0
        if temp > 1.5:
            temp_penalty = (temp - 1.5) * 3
        elif temp < 0.3:
            temp_penalty = (0.3 - temp) * 5
        score = speedup * 10 + acceptance * 5 - temp_penalty
        return {"score": float(score), "behavioral": (acceptance, speedup),
                "metadata": {"n_draft": nd, "acceptance": acceptance, "speedup": speedup,
                             "temp": temp, "temp_penalty": temp_penalty}}


class MtpConfig(BaseDomain):
    """Multi-token prediction: n_heads, loss_weight, share_weights, depth_ratio."""
    def name(self): return "mtp_config"
    def output_dim(self): return 4
    def behavioral_dims(self):
        return [("throughput", 10, 0, 1), ("prediction_acc", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_heads": int(np.interp(p[0], [0, 1], [1, 4])),
                "loss_weight": float(np.interp(p[1], [0, 1], [0.1, 0.5])),
                "share_weights": bool(p[2] > 0.5),
                "depth_ratio": float(np.interp(p[3], [0, 1], [0.5, 1.0]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("n_heads", 2) - 1) / 3, 0, 1),
            np.clip((config.get("loss_weight", 0.3) - 0.1) / 0.4, 0, 1),
            1.0 if config.get("share_weights", True) else 0.0,
            np.clip((config.get("depth_ratio", 0.75) - 0.5) / 0.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        nh = config["n_heads"]
        lw = config["loss_weight"]
        sw = config["share_weights"]
        dr = config["depth_ratio"]
        # Use ForgeEngine's MTPModule for real prediction accuracy measurement
        try:
            from research.decoding.mtp import MTPModule
            d_model, vocab = 128, 100
            mtp = MTPModule(d_model=d_model, vocab_size=vocab, n_heads=nh,
                            share_weights=sw).to(self.device)
            hidden = self._randn(4, 8, d_model)
            targets = torch.randint(0, vocab, (4, 8), device=self.device)
            token_embeds = self._randn(4, 8, d_model)
            mtp_loss, logits_list = mtp(hidden, token_embeds, targets)
            pred_acc = 1.0 / (1.0 + float(mtp_loss))
            throughput = nh * (1.0 if sw else 0.8) * dr
            signal = lw * pred_acc
        except (ImportError, Exception):
            # Fallback: analytical model
            throughput = nh * (1.0 if sw else 0.8)
            pred_acc = 1.0 / (1.0 + nh * 0.15) * dr
            signal = lw * pred_acc
        # Loss weight tradeoff: too high (>0.5) hurts main task loss.
        # Sweet spot is 0.3-0.5 (MTP auxiliary loss should not dominate).
        lw_penalty = 0.0
        if lw > 0.5:
            lw_penalty = (lw - 0.5) * 20  # linear penalty above 0.5
        # Depth ratio tradeoff: deeper MTP heads increase latency.
        # dr=1.0 means full-depth heads (same as main model) = 2x inference cost.
        # dr=0.5 means half-depth = 1.5x inference cost. Sweet spot 0.5-0.8.
        dr_latency = dr * nh * 0.5  # latency cost scales with depth * heads
        score = throughput * 5 + pred_acc * 10 + signal * 5 - lw_penalty - dr_latency
        return {"score": float(score), "behavioral": (throughput / 4, pred_acc),
                "metadata": {"n_heads": nh, "pred_acc": pred_acc,
                             "lw_penalty": lw_penalty, "dr_latency": dr_latency}}


class BatchedDecode(BaseDomain):
    """Batched decoding: max_batch, padding, merge_window, max_seq_diff."""
    def name(self): return "batched_decode"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"padding_strategy": ["left", "right", "dynamic"]}
    def behavioral_dims(self):
        return [("throughput", 10, 0, 1), ("padding_waste", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"max_batch_size": int(np.interp(p[0], [0, 1], [1, 16])),
                "padding_strategy": ["left", "right", "dynamic"][int(p[1] * 2.999)],
                "merge_window_ms": int(np.interp(p[2], [0, 1], [10, 100])),
                "max_seq_diff": int(np.interp(p[3], [0, 1], [0, 512]))}
    def encode(self, config):
        pmap = {"left": 0, "right": 0.5, "dynamic": 1.0}
        return torch.tensor([
            np.clip((config.get("max_batch_size", 8) - 1) / 15, 0, 1),
            pmap.get(config.get("padding_strategy", "left"), 0),
            np.clip((config.get("merge_window_ms", 50) - 10) / 90, 0, 1),
            np.clip(config.get("max_seq_diff", 256) / 512, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        bs = config["max_batch_size"]
        ps = config["padding_strategy"]
        mw = config["merge_window_ms"]
        msd = config["max_seq_diff"]
        # Throughput: scales with batch size but diminishing returns
        throughput = bs * (1 - 0.03 * bs)
        # Padding waste: dynamic = least, left/right = depends on seq diff
        if ps == "dynamic":
            padding_waste = 0.05
        else:
            padding_waste = min(0.5, msd / 1024)
        # Merge window: longer = more batching but more latency.
        # Latency penalty: users wait up to merge_window_ms before first token.
        # >50ms is noticeable, >100ms is annoying for interactive use.
        merge_eff = min(1.0, mw / 50)
        latency_penalty = 0.0
        if mw > 50:
            latency_penalty = (mw - 50) * 0.05  # 5ms over → -0.25, 100ms → -2.5
        score = throughput * 5 - padding_waste * 20 + merge_eff * 3 - latency_penalty
        return {"score": float(score), "behavioral": (throughput / 16, padding_waste),
                "metadata": {"batch_size": bs, "throughput": throughput,
                             "latency_penalty": latency_penalty}}


class SamplingConfig(BaseDomain):
    """Sampling: temperature, top_p, top_k, repetition_penalty, frequency_penalty."""
    def name(self): return "sampling_config"
    def output_dim(self): return 5
    def behavioral_dims(self):
        return [("diversity", 10, 0, 1), ("coherence", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"temperature": float(np.interp(p[0], [0, 1], [0.1, 2.0])),
                "top_p": float(np.interp(p[1], [0, 1], [0.5, 1.0])),
                "top_k": int(np.interp(p[2], [0, 1], [0, 100])),
                "repetition_penalty": float(np.interp(p[3], [0, 1], [1.0, 1.5])),
                "frequency_penalty": float(np.interp(p[4], [0, 1], [0, 2]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("temperature", 1.0) - 0.1) / 1.9, 0, 1),
            np.clip((config.get("top_p", 0.9) - 0.5) / 0.5, 0, 1),
            np.clip(config.get("top_k", 50) / 100, 0, 1),
            np.clip((config.get("repetition_penalty", 1.1) - 1.0) / 0.5, 0, 1),
            np.clip(config.get("frequency_penalty", 0.5) / 2, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        logits = self._randn(100) * 3
        temp = config["temperature"]
        tp = config["top_p"]
        tk = config["top_k"]
        rp = config["repetition_penalty"]
        fp = config["frequency_penalty"]
        logits = logits / temp
        # Top-k filtering
        if tk > 0:
            topk_vals = logits.topk(min(tk, 100)).values
            logits = torch.where(logits < topk_vals[-1], torch.full_like(logits, -1e9), logits)
        # Top-p filtering
        sorted_logits = logits.sort(descending=True)
        cum_probs = torch.softmax(sorted_logits.values, dim=-1).cumsum(dim=-1)
        cutoff = (cum_probs > tp).nonzero()[0].item() if tp < 1.0 else 99
        mask = torch.zeros(100, dtype=torch.bool, device=logits.device)
        mask[sorted_logits.indices[:cutoff + 1]] = True
        logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        probs = torch.softmax(logits, dim=-1)
        # Diversity: entropy of distribution
        diversity = float(-(probs * torch.log(probs + 1e-8)).sum().item()) / np.log(100)
        # Coherence: how peaked is the distribution
        coherence = float(probs.max().item())
        # Repetition penalty: has a BENEFIT (reduces repetition in long outputs)
        # but also a cost (over-penalizes legitimate repeated tokens like "the").
        # Model: benefit peaks at rp=1.1-1.3, then decreases.
        if rp <= 1.3:
            rp_benefit = (rp - 1.0) * 8  # benefit for mild penalty
        else:
            rp_benefit = 2.4 - (rp - 1.3) * 10  # diminishing then negative
        # Frequency penalty: similar — mild penalty helps diversity,
        # but too high causes incoherent outputs.
        if fp <= 1.0:
            fp_benefit = fp * 3
        else:
            fp_benefit = 3.0 - (fp - 1.0) * 4
        # Temperature tradeoff: extreme temps hurt quality
        temp_penalty = 0.0
        if temp > 1.5:
            temp_penalty = (temp - 1.5) * 4
        elif temp < 0.3:
            temp_penalty = (0.3 - temp) * 6
        score = diversity * 10 + coherence * 10 + rp_benefit + fp_benefit - temp_penalty
        return {"score": float(score), "behavioral": (diversity, coherence),
                "metadata": {"temp": temp, "diversity": diversity,
                             "rp_benefit": rp_benefit, "fp_benefit": fp_benefit}}


class BeamSearch(BaseDomain):
    """Beam search: beam_width, length_penalty, early_stopping, diversity_penalty."""
    def name(self): return "beam_search"
    def output_dim(self): return 4
    def behavioral_dims(self):
        return [("accuracy", 10, 0, 1), ("compute_cost", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"beam_width": int(np.interp(p[0], [0, 1], [1, 8])),
                "length_penalty": float(np.interp(p[1], [0, 1], [0.5, 2.0])),
                "early_stopping": bool(p[2] > 0.5),
                "diversity_penalty": float(np.interp(p[3], [0, 1], [0, 1]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("beam_width", 4) - 1) / 7, 0, 1),
            np.clip((config.get("length_penalty", 1.0) - 0.5) / 1.5, 0, 1),
            1.0 if config.get("early_stopping", True) else 0.0,
            config.get("diversity_penalty", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        bw = config["beam_width"]
        lp = config["length_penalty"]
        es = config["early_stopping"]
        dp = config["diversity_penalty"]
        # Simulate beam search quality
        # Accuracy increases with beam width but diminishing returns
        accuracy = 1.0 - 1.0 / (1.0 + bw * 0.5)
        # Length penalty: affects output length. lp=1.0 is neutral.
        # lp<1.0 favors shorter outputs (may truncate), lp>1.0 favors longer (may ramble).
        # Sweet spot is 0.8-1.2.
        length_factor = 1.0 - abs(lp - 1.0) * 2  # peaks at lp=1.0
        # Compute cost: linear in beam width
        compute = bw / 8.0
        # Early stopping saves compute
        if es:
            compute *= 0.7
        # Diversity penalty: improves variety between beams but too high
        # hurts quality (forces suboptimal tokens for diversity).
        # Sweet spot is 0.3-0.7.
        if dp <= 0.7:
            diversity_bonus = dp * 3
        else:
            diversity_bonus = 2.1 - (dp - 0.7) * 5  # diminishing then penalty
        # Beam width=1 is greedy search (no beam at all). Penalize it
        # — the whole point of beam search is multiple beams.
        beam_penalty = 0.0
        if bw == 1:
            beam_penalty = 5.0  # greedy = not beam search
        elif bw == 2:
            beam_penalty = 1.0  # minimal beam
        score = (accuracy * 15 - compute * 10 + diversity_bonus
                 + length_factor * 3 - beam_penalty)
        return {"score": float(score), "behavioral": (accuracy, compute),
                "metadata": {"beam_width": bw, "accuracy": accuracy,
                             "beam_penalty": beam_penalty}}
