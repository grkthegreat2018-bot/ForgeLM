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
        # Simulate: acceptance rate decreases with more draft tokens
        # but speedup increases if accepted
        acceptance = at ** nd * (1 - dr * 0.3)
        # Speedup = (1 + nd * acceptance) / (1 + dr * nd * 0.1)
        speedup = (1 + nd * acceptance) / (1 + dr * nd * 0.1)
        score = speedup * 10 + acceptance * 5
        return {"score": float(score), "behavioral": (acceptance, speedup),
                "metadata": {"n_draft": nd, "acceptance": acceptance, "speedup": speedup}}


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
        score = throughput * 5 + pred_acc * 10 + signal * 5
        return {"score": float(score), "behavioral": (throughput / 4, pred_acc),
                "metadata": {"n_heads": nh, "pred_acc": pred_acc}}


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
        # Merge window: longer = more batching but more latency
        merge_eff = min(1.0, mw / 50)
        score = throughput * 5 - padding_waste * 20 + merge_eff * 3
        return {"score": float(score), "behavioral": (throughput / 16, padding_waste),
                "metadata": {"batch_size": bs, "throughput": throughput}}


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
        # Penalize extreme repetition/frequency penalties
        penalty_cost = (rp - 1.0) * 5 + config["frequency_penalty"] * 2
        score = diversity * 10 + coherence * 10 - penalty_cost
        return {"score": float(score), "behavioral": (diversity, coherence),
                "metadata": {"temp": temp, "diversity": diversity}}


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
        # Length penalty affects output length
        length_factor = lp * 0.5 + 0.5
        # Compute cost: linear in beam width
        compute = bw / 8.0
        # Early stopping saves compute
        if es:
            compute *= 0.7
        # Diversity penalty improves variety
        diversity_bonus = dp * 2
        score = accuracy * 15 - compute * 10 + diversity_bonus + length_factor * 2
        return {"score": float(score), "behavioral": (accuracy, compute),
                "metadata": {"beam_width": bw, "accuracy": accuracy}}
