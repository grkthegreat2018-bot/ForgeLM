"""Attention-related evolution domains (10 domains).

Each domain explores a different attention configuration search space.
All use small tensor operations for fast evaluation (no model loading).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any
from . import BaseDomain


class RopeConfig(BaseDomain):
    """RoPE theta, scaling type, and scaling factor."""
    def name(self): return "rope_config"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"scaling_type": ["none", "yarn", "linear"]}
    def behavioral_dims(self):
        return [("rotation_diversity", 10, 0, 1), ("stability", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        theta = float(np.interp(p[0], [0, 1], [1000, 10_000_000]))
        scaling_type = ["none", "yarn", "linear"][int(p[1] * 2.999)]
        scaling_factor = float(np.interp(p[2], [0, 1], [0.5, 4.0]))
        return {"theta": theta, "scaling_type": scaling_type, "scaling_factor": scaling_factor}
    def encode(self, config):
        theta = config.get("theta", 1_000_000)
        st_map = {"none": 0, "yarn": 0.5, "linear": 1.0}
        return torch.tensor([
            np.clip((np.log(theta) - np.log(1000)) / (np.log(10e6) - np.log(1000)), 0, 1),
            st_map.get(config.get("scaling_type", "none"), 0),
            np.clip((config.get("scaling_factor", 1.0) - 0.5) / 3.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        theta = config["theta"]
        scaling = config["scaling_factor"]
        d = 64
        # Test at BOTH short (32) and long (4096) range — catches theta that
        # only looks good at short range but fails at 32K context.
        pos_short = torch.arange(32, dtype=torch.float32)
        pos_long = torch.arange(4096, dtype=torch.float32)
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        if config["scaling_type"] == "yarn":
            freqs = freqs / scaling
        elif config["scaling_type"] == "linear":
            pos_short = pos_short / scaling
            pos_long = pos_long / scaling
        # Short-range rotation diversity
        angles_short = pos_short.unsqueeze(1) * freqs.unsqueeze(0)
        rot_div_short = float(angles_short.std().item() / (angles_short.abs().mean().item() + 1e-8))
        # Long-range: check for attention collapse (all angles converge → no diversity)
        angles_long = pos_long.unsqueeze(1) * freqs.unsqueeze(0)
        rot_div_long = float(angles_long.std().item() / (angles_long.abs().mean().item() + 1e-8))
        # Stability: penalize extreme angles that cause numerical issues
        stability = 1.0 / (1.0 + float((angles_long.abs() > 1e4).float().mean().item()) * 10)
        # Checkpoint compatibility penalty: deviating far from base theta (1M)
        # breaks RoPE for models trained with 1M. Only reward high theta if
        # the model is trained from scratch with that theta.
        base_theta = 1_000_000.0
        theta_ratio = max(theta, base_theta) / min(theta, base_theta)
        compat_penalty = 0.0
        if theta_ratio > 10:  # >10x deviation from base
            compat_penalty = (np.log10(theta_ratio) - 1) * 8  # log-scaled penalty
        # Long-range attention quality: high theta compresses low-freq dims
        # so much they barely rotate → attention becomes too local at long range.
        # Measure the fraction of dimensions that are "frozen" (angle change < 0.01 rad over 4096 positions)
        angle_change = (angles_long[-1] - angles_long[0]).abs()
        frozen_frac = float((angle_change < 0.01).float().mean().item())
        frozen_penalty = frozen_frac * 10  # penalize frozen dimensions
        score = (rot_div_short * 5 + rot_div_long * 5 + stability * 5
                 - compat_penalty - frozen_penalty)
        return {"score": float(score), "behavioral": (rot_div_short, stability),
                "metadata": {"theta": theta, "scaling": scaling,
                             "frozen_frac": frozen_frac,
                             "compat_penalty": compat_penalty}}
    def seed_configs(self):
        return [{"theta": 10000, "scaling_type": "none", "scaling_factor": 1.0},
                {"theta": 1000000, "scaling_type": "none", "scaling_factor": 1.0},
                {"theta": 500000, "scaling_type": "yarn", "scaling_factor": 2.0}]


class DiffAttnConfig(BaseDomain):
    """Differential attention lambda, head count, softmax separation."""
    def name(self): return "diff_attn"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("noise_cancellation", 10, 0, 1), ("signal_retention", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"lambda_init": float(p[0]), "n_heads": int(np.interp(p[1], [0, 1], [4, 32])),
                "softmax_sep": float(p[2])}
    def encode(self, config):
        return torch.tensor([
            config.get("lambda_init", 0.5),
            np.clip((config.get("n_heads", 16) - 4) / 28, 0, 1),
            config.get("softmax_sep", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        lam = config["lambda_init"]
        n_h = config["n_heads"]
        d = 64
        q = self._randn(4, n_h, 32, d)
        k = self._randn(4, n_h, 32, d)
        attn1 = torch.softmax(q @ k.transpose(-2, -1) / (d ** 0.5), dim=-1)
        attn2 = torch.softmax((q + 0.1 * torch.randn_like(q)) @ k.transpose(-2, -1) / (d ** 0.5), dim=-1)
        diff_attn = attn1 - lam * attn2
        noise_cancel = 1.0 - float(diff_attn.abs().std().item() / (attn1.abs().std().item() + 1e-8))
        signal_ret = 1.0 - float((attn1 - diff_attn).abs().mean().item() / (attn1.abs().mean().item() + 1e-8))
        score = noise_cancel * 10 + signal_ret * 10 - lam * 2
        return {"score": float(score), "behavioral": (noise_cancel, signal_ret),
                "metadata": {"lambda": lam, "n_heads": n_h}}


class CsaAttention(BaseDomain):
    """CSA top-k position selection, pattern type, block size."""
    def name(self): return "csa_attention"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"pattern_type": ["standard", "csa", "csa_hca_hybrid"]}
    def behavioral_dims(self):
        return [("sparsity", 10, 0, 1), ("coverage", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"top_k": int(np.interp(p[0], [0, 1], [64, 1024])),
                "pattern_type": ["standard", "csa", "csa_hca_hybrid"][int(p[1] * 2.999)],
                "block_size": int(np.interp(p[2], [0, 1], [8, 128]))}
    def encode(self, config):
        pt_map = {"standard": 0, "csa": 0.5, "csa_hca_hybrid": 1.0}
        return torch.tensor([
            np.clip((config.get("top_k", 256) - 64) / 960, 0, 1),
            pt_map.get(config.get("pattern_type", "standard"), 0),
            np.clip((config.get("block_size", 32) - 8) / 120, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        top_k = config["top_k"]
        bs = config["block_size"]
        seq = 2048
        mask = torch.zeros(seq, seq)
        if config["pattern_type"] == "csa":
            scores = self._randn(seq)
            top_idx = scores.topk(min(top_k, seq)).indices
            mask[:, top_idx] = 1.0
        elif config["pattern_type"] == "csa_hca_hybrid":
            for i in range(0, seq, bs):
                mask[i:i+bs, i:i+bs] = 1.0
            scores = self._randn(seq)
            top_idx = scores.topk(min(top_k // 2, seq)).indices
            mask[:, top_idx] = 1.0
        else:
            mask = torch.ones(seq, seq)
        sparsity = 1.0 - float(mask.mean().item())
        coverage = float(mask.sum(dim=0).clamp(0, 1).mean().item())
        if sparsity < 0.1:
            score = -50
        else:
            score = sparsity * 10 + coverage * 5
        return {"score": float(score), "behavioral": (sparsity, coverage),
                "metadata": {"top_k": top_k, "block_size": bs}}


class GlaAttention(BaseDomain):
    """GLA latent dim, heads, compression ratio."""
    def name(self): return "gla_attention"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("reconstruction_err", 10, 0, 1), ("compression", 10, 1, 8)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"latent_dim": int(np.interp(p[0], [0, 1], [64, 512])),
                "n_heads": int(np.interp(p[1], [0, 1], [4, 32])),
                "compression_ratio": float(np.interp(p[2], [0, 1], [1.0, 8.0]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("latent_dim", 256) - 64) / 448, 0, 1),
            np.clip((config.get("n_heads", 16) - 4) / 28, 0, 1),
            np.clip((config.get("compression_ratio", 2.0) - 1.0) / 7.0, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        ld = config["latent_dim"]
        n_h = config["n_heads"]
        d = 512
        # Use n_heads in the evaluation — more heads = better parallelism
        # but each head has smaller dim, affecting reconstruction quality.
        head_dim = d // max(n_h, 1)
        k = self._randn(4, n_h, 32, head_dim)
        # Semi-orthogonal projections for realistic reconstruction error,
        # matching gla_attn_simulate (random projections give ~1.4 error
        # regardless of latent dim — unrealistic).
        if ld >= head_dim:
            # Overcomplete: near-lossless reconstruction
            proj_down = self._randn(head_dim, ld)
            Q, _ = torch.linalg.qr(proj_down.T)
            proj_down = Q[:head_dim, :head_dim].T
            proj_up = proj_down.T
            recon_err = 0.02 + 0.01 * torch.rand(1, device=self._device).item()
        else:
            # Undercomplete: SVD/QR-based optimal projection
            proj_down = self._randn(head_dim, ld)
            Q, _ = torch.linalg.qr(proj_down)
            proj_down = Q
            proj_up = proj_down.T
            k_latent = k @ proj_down
            k_recon = k_latent @ proj_up
            recon_err = float((k - k_recon).norm().item() / (k.norm().item() + 1e-8))
        compression = d / ld
        # More heads = better parallelism but more projection params
        head_overhead = n_h * 0.02  # param cost per head
        # Scoring matches gla_attention.json spec: base components plus a
        # graduated trivial penalty (-50 at compression=1.0 → 0 at 2.0).
        # The spec's n_heads_score/head_overhead flag handlers cancel against
        # the simulator's flag_adjustment, so the legacy path just omits both.
        base_score = -recon_err * 30 + compression * 5 - head_overhead
        score = base_score
        if compression < 2.0:
            deficit = 2.0 - compression
            score += -50.0 * deficit - base_score * deficit
        return {"score": float(score), "behavioral": (recon_err, compression),
                "metadata": {"latent_dim": ld, "n_heads": n_h, "recon_err": recon_err}}


class GtaAttention(BaseDomain):
    """GTA V=K mixing ratio, KV heads, tie strength."""
    def name(self): return "gta_attention"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("kv_reduction", 10, 0, 1), ("output_deviation", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"v_k_mix": float(p[0]), "n_kv_heads": int(np.interp(p[1], [0, 1], [4, 16])),
                "tie_strength": float(p[2])}
    def encode(self, config):
        return torch.tensor([
            config.get("v_k_mix", 0.5),
            np.clip((config.get("n_kv_heads", 8) - 4) / 12, 0, 1),
            config.get("tie_strength", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        mix = config["v_k_mix"]
        n_kv = config["n_kv_heads"]
        tie = config["tie_strength"]
        d = 64
        # Use n_kv_heads: fewer KV heads = more compression but more deviation
        n_q_heads = 32
        k = self._randn(4, n_kv, 32, d)
        v = self._randn(4, n_kv, 32, d)
        # V=K mixing with tie_strength controlling how strongly V is pulled toward K
        # tie_strength=0: no tying (V independent), tie_strength=1: V fully tied to K
        v_tied = mix * k * tie + (1 - mix * tie) * v
        dev = float((v - v_tied).norm().item() / (v.norm().item() + 1e-8))
        # KV reduction: fewer KV heads = more compression (GQA ratio)
        kv_red = 1.0 - n_kv / n_q_heads
        # tie_strength tradeoff: high tying saves more params but increases deviation
        tying_savings = tie * 0.3  # param savings from tying
        if dev > 0.5:
            score = -50
        else:
            score = kv_red * 10 - dev * 20 + tying_savings * 3
        return {"score": float(score), "behavioral": (kv_red, dev),
                "metadata": {"mix": mix, "n_kv_heads": n_kv, "tie_strength": tie,
                             "deviation": dev, "kv_reduction": kv_red}}


class QkNormConfig(BaseDomain):
    """QK-norm type, epsilon, scale init."""
    def name(self): return "qk_norm"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"norm_type": ["rmsnorm", "layernorm"]}
    def behavioral_dims(self):
        return [("attention_change", 10, 0, 1), ("stability", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"norm_type": ["rmsnorm", "layernorm"][int(p[0] * 1.999)],
                "epsilon": float(np.interp(p[1], [0, 1], [1e-6, 1e-3])),
                "scale_init": float(np.interp(p[2], [0, 1], [0.5, 2.0]))}
    def encode(self, config):
        nt_map = {"rmsnorm": 0, "layernorm": 1.0}
        return torch.tensor([
            nt_map.get(config.get("norm_type", "rmsnorm"), 0),
            np.clip((np.log(config.get("epsilon", 1e-5)) - np.log(1e-6)) / (np.log(1e-3) - np.log(1e-6)), 0, 1),
            np.clip((config.get("scale_init", 1.0) - 0.5) / 1.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        d = 64
        q = self._randn(4, 8, 32, d) * 3
        k = self._randn(4, 8, 32, d) * 3
        eps = config["epsilon"]
        scale = config["scale_init"]
        # Use fused RoPE+QK-norm kernel from ForgeEngine when available
        try:
            from research.decoding.fused_rope_qknorm import fused_qk_norm_rope
            # fused_qk_norm_rope does RMSNorm + RoPE; we only want the norm part
            # so pass identity cos/sin (theta=0 → no rotation)
            cos = torch.ones(32, d, device=self.device)
            sin = torch.zeros(32, d, device=self.device)
            w = torch.ones(d, device=self.device) * scale
            if config["norm_type"] == "rmsnorm":
                q_n, k_n = fused_qk_norm_rope(q, k, w, w, cos, sin, eps)
            else:
                raise ImportError  # fall through for layernorm
        except (ImportError, Exception):
            # Fallback: manual norm
            if config["norm_type"] == "rmsnorm":
                q_n = q * (q.pow(2).mean(-1, keepdim=True) + eps).rsqrt() * scale
                k_n = k * (k.pow(2).mean(-1, keepdim=True) + eps).rsqrt() * scale
            else:
                q_n = (q - q.mean(-1, keepdim=True)) / (q.std(-1, keepdim=True) + eps) * scale
                k_n = (k - k.mean(-1, keepdim=True)) / (k.std(-1, keepdim=True) + eps) * scale
        attn_orig = torch.softmax(q @ k.transpose(-2, -1) / (d ** 0.5), dim=-1)
        attn_norm = torch.softmax(q_n @ k_n.transpose(-2, -1) / (d ** 0.5), dim=-1)
        change = float((attn_orig - attn_norm).abs().mean().item())
        stability = 1.0 / (1.0 + float(attn_norm.max().item()) * 0.1)
        score = stability * 10 - change * 5
        return {"score": float(score), "behavioral": (change, stability),
                "metadata": {"epsilon": eps, "scale": scale}}


class AttnResidual(BaseDomain):
    """AttnRes k layers, gate init, retrieval dim."""
    def name(self): return "attn_residual"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("info_recovery", 10, 0, 1), ("compute_cost", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"k_layers": int(np.interp(p[0], [0, 1], [1, 8])),
                "gate_init": float(p[1]),
                "retrieval_dim": int(np.interp(p[2], [0, 1], [64, 512]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("k_layers", 4) - 1) / 7, 0, 1),
            config.get("gate_init", 0.0),
            np.clip((config.get("retrieval_dim", 256) - 64) / 448, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        k = config["k_layers"]
        rd = config["retrieval_dim"]
        gate = config["gate_init"]
        d = 512
        h = self._randn(32, d)
        past = self._randn(k, 32, d)
        proj = self._randn(d, rd) / (d ** 0.5)
        retrieved = past[:k].mean(0) @ proj @ self._randn(rd, d) / (rd ** 0.5)
        info_rec = float(torch.cosine_similarity(h.flatten(), retrieved.flatten(), dim=0).abs().item())
        compute = k * rd / (d * 16)
        score = info_rec * 10 - compute * 2 + gate * 3
        return {"score": float(score), "behavioral": (info_rec, compute),
                "metadata": {"k": k, "gate": gate, "retrieval_dim": rd}}


class MhcConfig(BaseDomain):
    """MHC rank, gate init, n connections."""
    def name(self): return "mhc_config"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("expressivity", 10, 0, 1), ("param_ratio", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"rank": int(np.interp(p[0], [0, 1], [64, 512])),
                "gate_init": float(p[1]),
                "n_connections": int(np.interp(p[2], [0, 1], [1, 8]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("rank", 256) - 64) / 448, 0, 1),
            config.get("gate_init", 0.0),
            np.clip((config.get("n_connections", 4) - 1) / 7, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        r = config["rank"]
        nc = config["n_connections"]
        d = 512
        h = self._randn(32, d)
        proj = self._randn(d, r) / (d ** 0.5)
        back = self._randn(r, d) / (r ** 0.5)
        residual = h @ proj @ back
        expressivity = float(residual.std().item() / (h.std().item() + 1e-8))
        param_ratio = (r * nc * d * 2) / (d * d * 16)
        score = expressivity * 10 - param_ratio * 5
        return {"score": float(score), "behavioral": (expressivity, param_ratio),
                "metadata": {"rank": r, "n_connections": nc}}


class SlidingWindow(BaseDomain):
    """Sliding window size, stride, overlap ratio."""
    def name(self): return "sliding_window"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("coverage", 10, 0, 1), ("memory_ratio", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        ws = int(np.interp(p[0], [0, 1], [256, 4096]))
        stride = int(np.interp(p[1], [0, 1], [1, ws // 2]))
        return {"window_size": ws, "stride": max(1, stride), "overlap_ratio": float(p[2])}
    def encode(self, config):
        ws = config.get("window_size", 1024)
        return torch.tensor([
            np.clip((ws - 256) / 3840, 0, 1),
            np.clip(config.get("stride", 512) / max(ws // 2, 1), 0, 1),
            config.get("overlap_ratio", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        ws = config["window_size"]
        stride = config["stride"]
        seq = 512  # reduced from 4096 — coverage pattern is scale-invariant
        # Vectorized sliding window mask — no Python loop
        positions = torch.arange(seq, device=self.device, dtype=torch.float32)
        # For each position i, window covers [i-ws//2, i+ws//2]
        # Coverage = fraction of positions that see each column
        half = ws // 2
        # A column j is covered by row i if |i-j| <= half
        # Total coverage per column = number of rows within half
        # Vectorized: coverage[j] = min(half*2+1, j+half+1, seq-j+half) / seq
        dist = torch.arange(seq, device=self.device)
        coverage_per_col = torch.minimum(
            torch.tensor(half * 2 + 1, device=self.device),
            torch.minimum(dist + half + 1, seq - dist + half)
        ).clamp(max=seq).float() / seq
        coverage = float(coverage_per_col.mean().item())
        memory_ratio = ws / 4096  # report relative to original seq
        if memory_ratio > 0.9:
            score = -50
        else:
            score = coverage * 10 - memory_ratio * 5
        return {"score": float(score), "behavioral": (coverage, memory_ratio),
                "metadata": {"window_size": ws, "stride": stride}}


class LocalGlobal(BaseDomain):
    """Local window, global ratio, n global heads."""
    def name(self): return "local_global"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("receptive_field", 10, 0, 1), ("compute_ratio", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"local_window": int(np.interp(p[0], [0, 1], [256, 2048])),
                "global_ratio": float(p[1]),
                "n_global_heads": int(np.interp(p[2], [0, 1], [1, 16]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("local_window", 512) - 256) / 1792, 0, 1),
            config.get("global_ratio", 0.2),
            np.clip((config.get("n_global_heads", 4) - 1) / 15, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        lw = config["local_window"]
        gr = config["global_ratio"]
        ngh = config["n_global_heads"]
        seq = 2048
        rf = (lw + gr * seq) / seq
        compute = (lw / seq) * 0.7 + gr * 0.3
        if compute > 0.9:
            score = -50
        else:
            score = rf * 10 - compute * 5
        return {"score": float(score), "behavioral": (rf, compute),
                "metadata": {"local_window": lw, "global_ratio": gr, "n_global_heads": ngh}}
