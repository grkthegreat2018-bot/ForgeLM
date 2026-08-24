"""Architecture-related evolution domains (5 domains).

Each domain explores an architecture configuration search space.
All use small tensor operations for fast evaluation (no model loading).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any
from . import BaseDomain


class MoeRouting(BaseDomain):
    """MoE: n_experts, top_k, router_mode, load_balance_weight, shared_expert."""
    def name(self): return "moe_routing"
    def output_dim(self): return 5
    def discrete_choices(self):
        return {"router_mode": ["switch", "aux_free"]}
    def behavioral_dims(self):
        return [("load_balance", 10, 0, 1), ("expert_util", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_experts": int(np.interp(p[0], [0, 1], [4, 16])),
                "top_k": int(np.interp(p[1], [0, 1], [1, 4])),
                "router_mode": ["switch", "aux_free"][int(p[2] * 1.999)],
                "load_balance_weight": float(np.interp(p[3], [0, 1], [0, 0.1])),
                "shared_expert": bool(p[4] > 0.5)}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("n_experts", 8) - 4) / 12, 0, 1),
            np.clip((config.get("top_k", 2) - 1) / 3, 0, 1),
            0.0 if config.get("router_mode", "switch") == "switch" else 1.0,
            np.clip(config.get("load_balance_weight", 0.01) / 0.1, 0, 1),
            1.0 if config.get("shared_expert", True) else 0.0,
        ], dtype=torch.float32)
    def evaluate(self, config):
        ne = config["n_experts"]
        tk = config["top_k"]
        lbw = config["load_balance_weight"]
        se = config["shared_expert"]
        # Use ForgeEngine's ElbowRouter for real dynamic routing evaluation
        try:
            from research.inference.scheduler.moe_optim import ElbowRouter
            router_logits = self._randn(128, ne)
            router = ElbowRouter(min_experts=1, max_experts=tk)
            expert_indices, expert_weights = router.route(router_logits)
            # Measure load balance from actual routing
            expert_counts = torch.zeros(ne, device=self.device)
            for i in range(ne):
                expert_counts[i] = (expert_indices == i).float().sum()
            balance = 1.0 - float(expert_counts.std().item() / (expert_counts.mean().item() + 1e-8))
            util = float((expert_counts > 0).float().mean().item())
        except (ImportError, Exception):
            # Fallback: standard top-k routing
            router = self._randn(128, ne)
            topk_vals, topk_idx = router.topk(tk, dim=-1)
            expert_counts = torch.zeros(ne, device=self.device)
            for i in range(ne):
                expert_counts[i] = (topk_idx == i).float().sum()
            balance = 1.0 - float(expert_counts.std().item() / (expert_counts.mean().item() + 1e-8))
            util = float((expert_counts > 0).float().mean().item())
        active_ratio = tk / ne + (0.2 if se else 0)
        score = balance * 15 + util * 10 - lbw * 20
        return {"score": float(score), "behavioral": (balance, util),
                "metadata": {"n_experts": ne, "balance": balance, "util": util}}


class FactorizedEmbed(BaseDomain):
    """Factorized embedding: rank, init_mode, tie_factor, vocab_size."""
    def name(self): return "factorized_embed"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"init_mode": ["svd", "random"]}
    def behavioral_dims(self):
        return [("param_reduction", 10, 0, 1), ("recon_err", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"rank": int(np.interp(p[0], [0, 1], [64, 512])),
                "init_mode": ["svd", "random"][int(p[1] * 1.999)],
                "tie_factor": float(p[2]),
                "vocab_size": int(np.interp(p[3], [0, 1], [32000, 65536]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("rank", 256) - 64) / 448, 0, 1),
            0.0 if config.get("init_mode", "svd") == "svd" else 1.0,
            config.get("tie_factor", 0.5),
            np.clip((config.get("vocab_size", 65536) - 32000) / 33536, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        r = config["rank"]
        vs = config["vocab_size"]
        d = 2048
        # Full embedding: vs × d params
        full_params = vs * d
        # Factorized: vs × r + r × d
        fact_params = vs * r + r * d
        reduction = 1.0 - fact_params / full_params
        # Reconstruction error: SVD gives ~0, random gives higher
        d_sample = min(d, 512)
        n_sample = min(vs, 64)
        emb = self._randn(n_sample, d_sample)
        if config["init_mode"] == "svd":
            U, S, Vh = torch.linalg.svd(emb, full_matrices=False)
            r_clip = min(r, S.shape[0])
            emb_recon = U[:, :r_clip] @ torch.diag(S[:r_clip]) @ Vh[:r_clip]
        else:
            proj = self._randn(d_sample, r) / (d_sample ** 0.5)
            back = self._randn(r, d_sample) / (r ** 0.5)
            emb_recon = emb @ proj @ back
        err = float((emb - emb_recon).norm().item() / (emb.norm().item() + 1e-8))
        score = reduction * 20 - err * 100
        return {"score": float(score), "behavioral": (reduction, err),
                "metadata": {"rank": r, "reduction": reduction, "err": err}}


class TitanMemory(BaseDomain):
    """TITAN: memory_rank, gate_init, n_memory_slots, update_freq."""
    def name(self): return "titan_memory"
    def output_dim(self): return 4
    def behavioral_dims(self):
        return [("memory_capacity", 10, 0, 1), ("param_ratio", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"memory_rank": int(np.interp(p[0], [0, 1], [64, 512])),
                "gate_init": float(np.interp(p[1], [0, 1], [0, 0.5])),
                "n_memory_slots": int(np.interp(p[2], [0, 1], [1, 8])),
                "update_freq": int(np.interp(p[3], [0, 1], [1, 10]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("memory_rank", 256) - 64) / 448, 0, 1),
            np.clip(config.get("gate_init", 0.0) / 0.5, 0, 1),
            np.clip((config.get("n_memory_slots", 4) - 1) / 7, 0, 1),
            np.clip((config.get("update_freq", 5) - 1) / 9, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        r = config["memory_rank"]
        gi = config["gate_init"]
        ns = config["n_memory_slots"]
        uf = config["update_freq"]
        d = 512
        # Simulate memory: store and retrieve from slots
        h = self._randn(32, d)
        memory = self._randn(ns, r, d) * 0.01
        gate = torch.sigmoid(torch.tensor(gi))
        # Retrieval: query memory slots
        retrieved = (h @ memory.view(-1, d).T).mean(dim=-1) * gate
        capacity = float(retrieved.std().item() / (h.std().item() + 1e-8))
        # Param ratio: memory params / model params
        param_ratio = (ns * r * d) / (d * d * 16)
        score = capacity * 15 - param_ratio * 10 + gate * 5
        return {"score": float(score), "behavioral": (capacity, param_ratio),
                "metadata": {"rank": r, "capacity": capacity, "gate": float(gate)}}


class FfnSkip(BaseDomain):
    """FFN skip: threshold, n_eval_layers, strategy, min_keep."""
    def name(self): return "ffn_skip"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"skip_strategy": ["cosine", "norm", "hybrid"]}
    def behavioral_dims(self):
        return [("compute_saved", 10, 0, 1), ("output_dev", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"skip_threshold": float(np.interp(p[0], [0, 1], [0, 0.5])),
                "n_eval_layers": int(np.interp(p[1], [0, 1], [1, 16])),
                "skip_strategy": ["cosine", "norm", "hybrid"][int(p[2] * 2.999)],
                "min_keep": float(np.interp(p[3], [0, 1], [0.5, 1.0]))}
    def encode(self, config):
        smap = {"cosine": 0, "norm": 0.5, "hybrid": 1.0}
        return torch.tensor([
            np.clip(config.get("skip_threshold", 0.3) / 0.5, 0, 1),
            np.clip((config.get("n_eval_layers", 8) - 1) / 15, 0, 1),
            smap.get(config.get("skip_strategy", "cosine"), 0),
            np.clip((config.get("min_keep", 0.8) - 0.5) / 0.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        st = config["skip_threshold"]
        nl = config["n_eval_layers"]
        ss = config["skip_strategy"]
        mk = config["min_keep"]
        # Simulate: 16 layers, decide which to skip
        activations = self._randn(16, 256, 512)
        if ss == "cosine":
            # Cosine similarity between consecutive layers (15 pairs → 16 mask)
            sims = torch.cosine_similarity(activations[:-1].flatten(1),
                                           activations[1:].flatten(1), dim=1)
            skip_mask = torch.zeros(16, dtype=torch.bool, device=self.device)
            skip_mask[1:] = sims > st
        elif ss == "norm":
            norms = activations.norm(dim=(1, 2))
            skip_mask = norms < st * norms.max()
        else:  # hybrid
            sims = torch.cosine_similarity(activations[:-1].flatten(1),
                                           activations[1:].flatten(1), dim=1)
            norms = activations.norm(dim=(1, 2))
            skip_mask = torch.zeros(16, dtype=torch.bool, device=self.device)
            skip_mask[1:] = sims > st
            skip_mask = skip_mask | (norms < st * norms.max())
        # Ensure min_keep fraction is kept
        n_keep = max(int(mk * 16), 1)
        if skip_mask.sum() > 16 - n_keep:
            skip_mask[:16 - n_keep] = True
            skip_mask[16 - n_keep:] = False
        compute_saved = float(skip_mask.float().mean().item())
        # Output deviation: skipped layers contribute 0
        full_out = activations.sum(dim=0)
        skipped_out = activations[~skip_mask].sum(dim=0) if skip_mask.any() else full_out
        dev = float((full_out - skipped_out).norm().item() / (full_out.norm().item() + 1e-8))
        if compute_saved < 0.05:
            score = -50
        else:
            score = compute_saved * 20 - dev * 30
        return {"score": float(score), "behavioral": (compute_saved, dev),
                "metadata": {"compute_saved": compute_saved, "deviation": dev}}


class ConvConfig(BaseDomain):
    """Conv: kernel_size, stride, dilation, groups, n_conv_layers."""
    def name(self): return "conv_config"
    def output_dim(self): return 5
    def discrete_choices(self):
        return {"kernel_size": [3, 5, 7, 9]}
    def behavioral_dims(self):
        return [("receptive_field", 10, 0, 1), ("param_count", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        ks_choices = [3, 5, 7, 9]
        return {"kernel_size": ks_choices[int(p[0] * 3.999)],
                "stride": int(np.interp(p[1], [0, 1], [1, 3])),
                "dilation": int(np.interp(p[2], [0, 1], [1, 4])),
                "groups": int(np.interp(p[3], [0, 1], [1, 8])),
                "n_conv_layers": int(np.interp(p[4], [0, 1], [1, 6]))}
    def encode(self, config):
        ks_map = {3: 0, 5: 0.33, 7: 0.66, 9: 1.0}
        return torch.tensor([
            ks_map.get(config.get("kernel_size", 3), 0),
            np.clip((config.get("stride", 1) - 1) / 2, 0, 1),
            np.clip((config.get("dilation", 1) - 1) / 3, 0, 1),
            np.clip((config.get("groups", 1) - 1) / 7, 0, 1),
            np.clip((config.get("n_conv_layers", 3) - 1) / 5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        ks = config["kernel_size"]
        st = config["stride"]
        dl = config["dilation"]
        gr = config["groups"]
        nl = config["n_conv_layers"]
        # Receptive field: grows with kernel, dilation, layers
        rf = ks * dl * nl
        # Param count: kernel^2 * in * out / groups
        d = 256
        params = ks * ks * d * d / gr * nl
        param_ratio = params / (d * d * 16)
        # Run actual conv on small input (groups must divide d)
        x = self._randn(1, d, 32, 32)
        gr_safe = min(gr, d)
        while d % gr_safe != 0:
            gr_safe -= 1
        conv = torch.nn.Conv2d(d, d, ks, stride=st, dilation=dl, groups=gr_safe, padding=ks * dl // 2)
        try:
            out = conv(x)
            out_quality = float(out.std().item() / (x.std().item() + 1e-8))
        except:
            out_quality = 0.0
        rf_norm = min(1.0, rf / 100)
        score = rf_norm * 10 + out_quality * 5 - param_ratio * 3
        return {"score": float(score), "behavioral": (rf_norm, param_ratio),
                "metadata": {"receptive_field": rf, "params": params}}
