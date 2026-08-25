"""8 compact evolution domains for KV cache-related search spaces.

Each domain uses small synthetic tensor operations (no model loading) to model
real tradeoffs: compression vs error, memory vs speed, param reduction vs
quality.  Designed for MAP-Elites / ForgeEvolve search.

Domains:
  1. RotorQuantKV   — rotation + quantization of K,V
  2. HadamardKV     — Hadamard transform on K,V
  3. StreamingKV    — chunked streaming KV with sink tokens
  4. KvZipKV        — vector-quantization KV compression
  5. XQuantKV       — rematerialization + quantization
  6. KvRecompute    — selective layer recomputation
  7. CrossLayerKV   — cross-layer KV sharing
  8. PagedEvictKV   — paged KV with eviction policies
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Any
from . import BaseDomain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hadamard(n: int, device=None) -> torch.Tensor:
    """Normalized Hadamard matrix of size n (power of 2)."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / np.sqrt(n)


def _quant_error(x: torch.Tensor, bits: int) -> float:
    """Simulate uniform quantization and return relative L2 error."""
    levels = 2 ** bits
    xmax = x.abs().amax().clamp(min=1e-8)
    x_q = torch.round(x / xmax * (levels / 2 - 1)) / (levels / 2 - 1) * xmax
    return float((x - x_q).norm() / x.norm().clamp(min=1e-8))


# ---------------------------------------------------------------------------
# 1. RotorQuantKV
# ---------------------------------------------------------------------------

class RotorQuantKV(BaseDomain):
    """Rotate K,V then quantize.  Search rotation type, n_rotations, quant_bits."""

    ROT_TYPES = ["hadamard", "dct", "random"]

    def __init__(self, d: int = 64, n_heads: int = 8, seq_len: int = 512, seed: int = 42):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = self._randn(seq_len, n_heads, d, generator=g) * 0.1
        self.d, self.n_heads, self.seq_len = d, n_heads, seq_len

    def name(self) -> str: return "rotor_quant_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        rot = self.ROT_TYPES[int(p[0] * 3) % 3]
        n_rot = int(1 + p[1] * 7)          # 1-8
        bits = [4, 8][int(p[2] * 2) % 2]   # 4 or 8
        return {"rot_type": rot, "n_rotations": n_rot, "quant_bits": bits}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        ri = self.ROT_TYPES.index(c.get("rot_type", "hadamard"))
        bi = 0 if int(c.get("quant_bits", 8)) == 4 else 1
        return torch.tensor([ri / 3, (int(c.get("n_rotations", 4)) - 1) / 7, bi], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        K = self.K.clone()
        d = self.d
        for _ in range(int(c["n_rotations"])):
            if c["rot_type"] == "hadamard":
                H = _hadamard(d, device=self.device); K = torch.einsum("shd,df->shf", K, H)
            elif c["rot_type"] == "dct":
                n = torch.arange(d, dtype=torch.float32, device=self.device)
                D = torch.cos(np.pi * (2 * n.unsqueeze(0) + 1) * n.unsqueeze(1) / (2 * d)) / np.sqrt(d)
                K = torch.einsum("shd,df->shf", K, D)
            else:
                R = self._randn(d, d) / np.sqrt(d); K = torch.einsum("shd,df->shf", K, R)
        err = _quant_error(K, int(c["quant_bits"]))
        comp = 16.0 / int(c["quant_bits"])                      # bf16 → bits
        compute = int(c["n_rotations"]) * (1.0 if c["rot_type"] == "hadamard" else 2.5)
        score = comp * 10 - err * 200 - compute * 0.5
        return {"score": float(score), "behavioral": (comp, err),
                "metadata": {"rot_type": c["rot_type"], "n_rotations": c["n_rotations"],
                             "quant_bits": c["quant_bits"], "compute": compute}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("compression", 8, 2.0, 4.0), ("quant_error", 10, 0.0, 0.5)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"rot_type": self.ROT_TYPES, "n_rotations": [1, 2, 4, 8], "quant_bits": [4, 8]}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"rot_type": r, "n_rotations": n, "quant_bits": b}
                for r in self.ROT_TYPES for n in [1, 4] for b in [4, 8]]

    def to_cpu(self) -> "RotorQuantKV": return self


# ---------------------------------------------------------------------------
# 2. HadamardKV
# ---------------------------------------------------------------------------

class HadamardKV(BaseDomain):
    """Apply Hadamard transform to K,V.  Search hadamard_dim, n_apply, quant_bits."""

    def __init__(self, d: int = 64, n_heads: int = 8, seq_len: int = 512, seed: int = 7):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = self._randn(seq_len, n_heads, d, generator=g) * 0.1
        self.d = d

    def name(self) -> str: return "hadamard_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        hd = int(64 + p[0] * 448)                          # 64-512
        hd = min(hd, self.d)                               # clamp to dim
        n_apply = int(1 + p[1] * 3)                        # 1-4
        bits = [4, 8][int(p[2] * 2) % 2]
        return {"hadamard_dim": hd, "n_apply": n_apply, "quant_bits": bits}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        bi = 0 if c.get("quant_bits", 8) == 4 else 1
        return torch.tensor([(c.get("hadamard_dim", 64) - 64) / 448,
                             (c.get("n_apply", 1) - 1) / 3, bi], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        K = self.K.clone(); d = self.d
        hd = min(c["hadamard_dim"], d)
        err_orig = _quant_error(K, c["quant_bits"])
        for _ in range(c["n_apply"]):
            H = _hadamard(hd, device=self.device)
            K[..., :hd] = torch.einsum("shd,df->shf", K[..., :hd], H)
        err_rot = _quant_error(K, c["quant_bits"])
        err_reduction = max(0.0, err_orig - err_rot)
        compute = c["n_apply"] * hd / 64.0
        comp = 16.0 / c["quant_bits"]
        score = err_reduction * 500 + comp * 5 - compute * 0.3
        return {"score": float(score), "behavioral": (err_reduction, compute),
                "metadata": {"hadamard_dim": c["hadamard_dim"], "n_apply": c["n_apply"],
                             "quant_bits": c["quant_bits"], "err_orig": err_orig, "err_rot": err_rot}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("err_reduction", 10, 0.0, 0.3), ("compute_cost", 10, 0.0, 16.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"hadamard_dim": [64, 128, 256, 512], "n_apply": [1, 2, 3, 4], "quant_bits": [4, 8]}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"hadamard_dim": hd, "n_apply": n, "quant_bits": 4}
                for hd in [64, 128, 256] for n in [1, 2, 4]]

    def to_cpu(self) -> "HadamardKV": return self


# ---------------------------------------------------------------------------
# 3. StreamingKV
# ---------------------------------------------------------------------------

class StreamingKV(BaseDomain):
    """Streaming KV with sink tokens.  Search chunk_size, n_sink, overlap."""

    def __init__(self, seq_len: int = 4096, d: int = 64, n_heads: int = 8, seed: int = 11):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = self._randn(seq_len, n_heads, d, generator=g) * 0.1
        self.seq_len = seq_len

    def name(self) -> str: return "streaming_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        chunk = int(128 + p[0] * 1920)                     # 128-2048
        n_sink = int(4 + p[1] * 28)                        # 4-32
        overlap = float(p[2] * 0.5)                        # 0-0.5
        return {"chunk_size": chunk, "n_sink": n_sink, "overlap": overlap}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([(c.get("chunk_size", 512) - 128) / 1920,
                             (c.get("n_sink", 16) - 4) / 28,
                             c.get("overlap", 0.25) / 0.5], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        chunk, n_sink, overlap = c["chunk_size"], c["n_sink"], c["overlap"]
        step = max(1, int(chunk * (1 - overlap)))
        n_chunks = max(1, (self.seq_len - n_sink) // step)
        # Coverage: fraction of tokens represented (sink + chunks with overlap)
        covered = n_sink + n_chunks * chunk
        coverage = min(1.0, covered / self.seq_len)
        # Memory: sink + last chunk (streaming keeps only sink + current)
        # Overlap means keeping parts of previous chunks, which increases memory.
        # overlap=0.5 means keeping 50% of previous chunk = +0.5*chunk memory.
        mem_ratio = (n_sink + chunk * (1 + overlap)) / self.seq_len
        # Attention quality: estimate via variance of covered tokens
        sink_idx = torch.linspace(0, self.seq_len - 1, n_sink).long()
        chunk_start = self.seq_len - chunk
        kept_idx = torch.cat([sink_idx, torch.arange(chunk_start, self.seq_len)])
        kept = self.K[kept_idx]
        coverage_err = 1.0 - float(kept.std() / self.K.std().clamp(min=1e-8))
        score = coverage * 100 - mem_ratio * 80 - coverage_err * 50
        return {"score": float(score), "behavioral": (coverage, mem_ratio),
                "metadata": {"chunk_size": chunk, "n_sink": n_sink, "overlap": overlap,
                             "n_chunks": n_chunks, "coverage_err": coverage_err,
                             "mem_with_overlap": mem_ratio}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("coverage", 10, 0.0, 1.0), ("mem_ratio", 10, 0.0, 1.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"chunk_size": [128, 256, 512, 1024, 2048], "n_sink": [4, 8, 16, 32],
                "overlap": [0.0, 0.1, 0.25, 0.5]}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"chunk_size": cs, "n_sink": ns, "overlap": 0.25}
                for cs in [256, 512, 1024] for ns in [8, 16]]

    def to_cpu(self) -> "StreamingKV": return self


# ---------------------------------------------------------------------------
# 4. KvZipKV
# ---------------------------------------------------------------------------

class KvZipKV(BaseDomain):
    """KV zip vector quantization.  Search compression_ratio, codebook_size, n_iter."""

    def __init__(self, d: int = 64, n_heads: int = 8, seq_len: int = 512, seed: int = 23):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = self._randn(seq_len * n_heads, d, generator=g) * 0.1
        self.d, self.n_tokens = d, seq_len * n_heads

    def name(self) -> str: return "kvzip_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        comp = int(2 + p[0] * 14)                          # 2-16
        codebook = int(64 + p[1] * 448)                    # 64-512
        n_iter = int(10 + p[2] * 90)                       # 10-100
        return {"compression_ratio": comp, "codebook_size": codebook, "n_iter": n_iter}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([(float(c.get("compression_ratio", 8)) - 2) / 14,
                             (int(c.get("codebook_size", 256)) - 64) / 448,
                             (int(c.get("n_iter", 50)) - 10) / 90], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        K = self.K[:256]  # reduced from 1024 for speed
        n_sub = K.shape[0]
        cb_size = min(int(c["codebook_size"]), n_sub)
        idx = torch.randperm(n_sub, device=K.device)[:cb_size]
        codebook = K[idx].clone()
        K_sub = K
        for _ in range(min(int(c["n_iter"]), 5)):  # capped at 5 (was 20)
            dist = torch.cdist(K_sub, codebook)   # (256, cb_size)
            assign = dist.argmin(dim=1)
            one_hot = torch.zeros(K_sub.shape[0], cb_size, device=K.device)
            one_hot.scatter_(1, assign.unsqueeze(1), 1.0)
            counts = one_hot.sum(dim=0).clamp(min=1)
            codebook = (one_hot.T @ K_sub) / counts.unsqueeze(1)
        dist_full = torch.cdist(K_sub, codebook)
        recon = codebook[dist_full.argmin(dim=1)]
        recon_err = float((K_sub - recon).norm() / K_sub.norm().clamp(min=1e-8))
        comp = c["compression_ratio"]
        # Actual compression: n_tokens * d * 2 bytes → cb_size * d * 2 + n_tokens * 4 (indices)
        mem_bytes = cb_size * self.d * 2 + 1024 * 4
        orig_bytes = 1024 * self.d * 2
        actual_comp = orig_bytes / max(mem_bytes, 1)
        score = actual_comp * 8 - recon_err * 300 - int(c["n_iter"]) * 0.02
        return {"score": float(score), "behavioral": (actual_comp, recon_err),
                "metadata": {"compression_ratio": comp, "codebook_size": cb_size,
                             "n_iter": c["n_iter"], "recon_err": recon_err, "actual_comp": actual_comp}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("compression", 10, 1.0, 16.0), ("recon_error", 10, 0.0, 0.5)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"compression_ratio": [2, 4, 8, 16], "codebook_size": [64, 128, 256, 512],
                "n_iter": [10, 25, 50, 100]}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"compression_ratio": cr, "codebook_size": cb, "n_iter": 50}
                for cr in [4, 8, 16] for cb in [128, 256]]

    def to_cpu(self) -> "KvZipKV": return self


# ---------------------------------------------------------------------------
# 5. XQuantKV
# ---------------------------------------------------------------------------

class XQuantKV(BaseDomain):
    """XQuant rematerialization.  Search recomputation_ratio, quant_bits, checkpoint_interval."""

    def __init__(self, n_layers: int = 16, seq_len: int = 2048, d: int = 64, n_heads: int = 8, seed: int = 31):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = [self._randn(seq_len, n_heads, d, generator=g) * 0.1 for _ in range(n_layers)]
        self.n_layers, self.seq_len = n_layers, seq_len

    def name(self) -> str: return "xquant_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        ratio = float(p[0])                                # 0-1
        bits = [4, 8][int(p[1] * 2) % 2]
        ckpt_interval = int(2 + p[2] * 14)                 # 2-16
        return {"recomputation_ratio": ratio, "quant_bits": bits, "checkpoint_interval": ckpt_interval}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        bi = 0 if c.get("quant_bits", 8) == 4 else 1
        return torch.tensor([c.get("recomputation_ratio", 0.5), bi,
                             (c.get("checkpoint_interval", 8) - 2) / 14], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        ratio, bits, interval = c["recomputation_ratio"], c["quant_bits"], c["checkpoint_interval"]
        # Layers with recomputation: don't store KV, recompute on backward
        n_recompute = int(self.n_layers * ratio)
        n_stored = self.n_layers - n_recompute
        # Memory: stored layers quantized, recomputed layers free
        mem_full = self.n_layers * self.seq_len * 64 * 2   # bf16
        mem_saved = n_recompute * self.seq_len * 64 * 2
        mem_quant = n_stored * self.seq_len * 64 * (bits / 16)
        mem_total = mem_quant                              # recomputed layers = 0 storage
        mem_ratio = mem_total / mem_full
        # Recomputation cost: proportional to n_recompute / interval
        recompute_cost = n_recompute / interval
        # Inference latency: recomputing KV during generation is MUCH more
        # expensive than during training (no parallelism, sequential token gen).
        # ratio=1.0 means NO KV cache = every token recomputes all layers = O(n^2) generation.
        # This is the critical penalty the old scoring missed.
        if ratio >= 1.0:
            # Full recompute = no KV cache = quadratic generation cost
            inference_penalty = 50.0
        elif ratio > 0.5:
            # High recompute = significant generation slowdown
            inference_penalty = (ratio - 0.5) * 60  # 0.5→0, 1.0→30
        else:
            inference_penalty = ratio * 10  # mild penalty for low recompute
        # Quantization error on stored layers (guard against n_stored=0)
        if n_stored > 0:
            err = np.mean([_quant_error(self.K[i][:512], bits) for i in range(min(n_stored, 4))])
            err = float(err) if np.isfinite(err) else 1.0
        else:
            err = 0.0  # no stored layers = no quant error
        score = ((1 - mem_ratio) * 100 - recompute_cost * 5 - err * 200
                 - inference_penalty)
        return {"score": float(score), "behavioral": (1 - mem_ratio, recompute_cost),
                "metadata": {"recomputation_ratio": ratio, "quant_bits": bits,
                             "checkpoint_interval": interval, "n_recompute": n_recompute,
                             "mem_ratio": mem_ratio, "quant_err": err,
                             "inference_penalty": inference_penalty}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("memory_saved", 10, 0.0, 1.0), ("recompute_cost", 10, 0.0, 8.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"recomputation_ratio": [0.0, 0.25, 0.5, 0.75, 1.0], "quant_bits": [4, 8],
                "checkpoint_interval": [2, 4, 8, 16]}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"recomputation_ratio": r, "quant_bits": 4, "checkpoint_interval": 8}
                for r in [0.0, 0.25, 0.5, 0.75, 1.0]]

    def to_cpu(self) -> "XQuantKV": return self


# ---------------------------------------------------------------------------
# 6. KvRecompute
# ---------------------------------------------------------------------------

class KvRecompute(BaseDomain):
    """Selective KV recomputation.  Search recompute_layers, strategy, threshold."""

    STRATEGIES = ["selective", "full"]

    def __init__(self, n_layers: int = 16, seq_len: int = 2048, d: int = 64, n_heads: int = 8, seed: int = 41):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = [self._randn(seq_len, n_heads, d, generator=g) * 0.1 for _ in range(n_layers)]
        self.n_layers = n_layers
        # Precompute per-layer attention entropy (proxy for importance)
        self.layer_entropy = [float(k.var(dim=0).mean()) for k in self.K]

    def name(self) -> str: return "kv_recompute"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        n_recomp = int(p[0] * 16)                          # 0-16
        strat = self.STRATEGIES[int(p[1] * 2) % 2]
        threshold = float(p[2])                            # 0-1
        return {"recompute_layers": n_recomp, "recompute_strategy": strat, "threshold": threshold}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        si = self.STRATEGIES.index(c.get("recompute_strategy", "selective"))
        return torch.tensor([c.get("recompute_layers", 8) / 16, si,
                             c.get("threshold", 0.5)], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        n_recomp, strat, threshold = c["recompute_layers"], c["recompute_strategy"], c["threshold"]
        # Selective: recompute layers with entropy below threshold
        if strat == "selective":
            entropies = torch.tensor(self.layer_entropy)
            norm_ent = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
            recomp_mask = norm_ent < threshold
            n_actual = min(n_recomp, int(recomp_mask.sum()))
        else:
            n_actual = min(n_recomp, self.n_layers)
        n_stored = self.n_layers - n_actual
        # Memory saved
        mem_full = self.n_layers * 2048 * 64 * 2
        mem_saved_ratio = n_actual / self.n_layers
        # Compute cost: full recompute = 2x, selective = 1.3x per layer
        cost_per_layer = 2.0 if strat == "full" else 1.3
        compute_cost = n_actual * cost_per_layer
        # Quality: recompute is exact (no quant), so error = 0 for recomputed, small for stored
        quality = 1.0 - n_stored * 0.01                    # stored layers have minor cache miss
        # Inference latency: recomputing all layers during generation is O(n^2).
        # n_actual=16 (all layers) = no KV cache = quadratic generation.
        # This is the critical penalty the old scoring missed.
        recomp_ratio = n_actual / self.n_layers
        if recomp_ratio >= 1.0:
            inference_penalty = 40.0  # full recompute = no KV cache
        elif recomp_ratio > 0.5:
            inference_penalty = (recomp_ratio - 0.5) * 50
        else:
            inference_penalty = recomp_ratio * 8
        score = (mem_saved_ratio * 100 - compute_cost * 2 + quality * 20
                 - inference_penalty)
        return {"score": float(score), "behavioral": (mem_saved_ratio, compute_cost),
                "metadata": {"recompute_layers": n_recomp, "strategy": strat,
                             "threshold": threshold, "n_actual_recomp": n_actual,
                             "quality": quality, "inference_penalty": inference_penalty}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("memory_saved", 10, 0.0, 1.0), ("compute_cost", 10, 0.0, 32.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"recompute_layers": [0, 4, 8, 12, 16], "recompute_strategy": self.STRATEGIES,
                "threshold": [0.1, 0.3, 0.5, 0.7, 0.9]}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"recompute_layers": n, "recompute_strategy": s, "threshold": 0.5}
                for n in [4, 8, 12] for s in self.STRATEGIES]

    def to_cpu(self) -> "KvRecompute": return self


# ---------------------------------------------------------------------------
# 7. CrossLayerKV
# ---------------------------------------------------------------------------

class CrossLayerKV(BaseDomain):
    """Cross-layer KV sharing.  Search share_ratio, n_share_groups, share_mode."""

    MODES = ["avg", "max", "learned"]

    def __init__(self, n_layers: int = 16, seq_len: int = 512, d: int = 64, n_heads: int = 8, seed: int = 53):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = self._randn(n_layers, seq_len, n_heads, d, generator=g) * 0.1
        self.n_layers = n_layers

    def name(self) -> str: return "cross_layer_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        ratio = float(p[0])                                # 0-1
        n_groups = int(1 + p[1] * 7)                       # 1-8
        mode = self.MODES[int(p[2] * 3) % 3]
        return {"share_ratio": ratio, "n_share_groups": n_groups, "share_mode": mode}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        mi = self.MODES.index(c.get("share_mode", "avg"))
        return torch.tensor([c.get("share_ratio", 0.5), (c.get("n_share_groups", 4) - 1) / 7, mi / 3],
                            dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        ratio, n_groups, mode = c["share_ratio"], c["n_share_groups"], c["share_mode"]
        K = self.K.clone()
        n_layers = self.n_layers
        # Group layers and share KV within groups
        group_size = max(1, n_layers // n_groups)
        n_shared = 0
        recon_err = 0.0
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, n_layers)
            group = K[start:end]
            if mode == "avg":
                shared = group.mean(dim=0, keepdim=True)
            elif mode == "max":
                shared = group.abs().argmax(dim=0)  # index of max
                shared = group.gather(0, shared.unsqueeze(0))
            else:  # learned → weighted average with learned weights (simulated)
                # Learned mode: optimal weighting minimizes recon error.
                # Simulate by using SVD-based optimal combination.
                group_flat = group.reshape(group.shape[0], -1)  # (n_layers, rest)
                # Optimal shared = weighted combo that minimizes ||target - shared||^2
                # Approximate: use the first principal component direction
                U, S, Vh = torch.linalg.svd(group_flat, full_matrices=False)
                weights = torch.softmax(S, dim=0)  # higher singular values = more weight
                shared_flat = (weights.unsqueeze(0) @ U.T @ group_flat)  # weighted combo
                shared = shared_flat.reshape(1, *group.shape[1:])
            # Share ratio: fraction of layers that use shared KV
            n_share_in_group = max(1, int((end - start) * ratio))
            target = group[-n_share_in_group:]
            recon = shared.expand(n_share_in_group, *group.shape[1:])
            # Relative L2 reconstruction error (always non-negative)
            recon_err += float((target - recon).norm().item() / (target.norm().item() + 1e-8))
            n_shared += n_share_in_group
        param_reduction = n_shared / n_layers
        recon_err /= max(1, n_groups)
        # Compute overhead for learned mode (higher due to weight params + compute)
        overhead = 1.5 if mode == "learned" else (1.2 if mode == "max" else 1.0)
        score = param_reduction * 100 - recon_err * 500 - overhead * 2
        return {"score": float(score), "behavioral": (param_reduction, recon_err),
                "metadata": {"share_ratio": ratio, "n_share_groups": n_groups, "share_mode": mode,
                             "param_reduction": param_reduction, "recon_err": recon_err}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("param_reduction", 10, 0.0, 1.0), ("recon_error", 10, 0.0, 0.5)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"share_ratio": [0.0, 0.25, 0.5, 0.75, 1.0], "n_share_groups": [1, 2, 4, 8],
                "share_mode": self.MODES}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"share_ratio": r, "n_share_groups": 4, "share_mode": m}
                for r in [0.25, 0.5, 0.75] for m in self.MODES]

    def to_cpu(self) -> "CrossLayerKV": return self


# ---------------------------------------------------------------------------
# 8. PagedEvictKV
# ---------------------------------------------------------------------------

class PagedEvictKV(BaseDomain):
    """Paged KV with eviction.  Search page_size, n_pages, eviction_policy."""

    POLICIES = ["lru", "lfu", "importance"]

    def __init__(self, seq_len: int = 4096, d: int = 64, n_heads: int = 8, seed: int = 67):
        super().__init__()
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.K = self._randn(seq_len, n_heads, d, generator=g) * 0.1
        self.seq_len = seq_len
        # Simulate access pattern: Zipfian-ish (some tokens accessed more)
        rng = np.random.RandomState(seed)
        self.access_pattern = rng.zipf(1.5, size=seq_len).clip(1, 100)

    def name(self) -> str: return "paged_evict_kv"
    def output_dim(self) -> int: return 3

    def decode(self, p: torch.Tensor) -> dict[str, Any]:
        p = p.detach().cpu().numpy()
        page_size = int(16 + p[0] * 240)                   # 16-256
        n_pages = int(64 + p[1] * 448)                     # 64-512
        policy = self.POLICIES[int(p[2] * 3) % 3]
        return {"page_size": page_size, "n_pages": n_pages, "eviction_policy": policy}

    def encode(self, c: dict[str, Any]) -> torch.Tensor:
        pi = self.POLICIES.index(c.get("eviction_policy", "lru"))
        return torch.tensor([(c.get("page_size", 64) - 16) / 240,
                             (c.get("n_pages", 256) - 64) / 448, pi / 3], dtype=torch.float32)

    def evaluate(self, c: dict[str, Any]) -> dict:
        page_size, n_pages, policy = c["page_size"], c["n_pages"], c["eviction_policy"]
        total_tokens = self.seq_len
        capacity = page_size * n_pages
        # Memory efficiency: how much of capacity is usefully filled
        mem_eff = min(1.0, total_tokens / capacity) if capacity > 0 else 0.0
        # Simulate eviction: compute hit rate based on policy
        n_total_pages = (total_tokens + page_size - 1) // page_size
        if n_total_pages <= n_pages:
            hit_rate = 1.0
        else:
            # Assign importance per page (mean access frequency)
            page_importance = []
            for i in range(n_total_pages):
                start = i * page_size
                end = min(start + page_size, total_tokens)
                page_importance.append(float(self.access_pattern[start:end].mean()))
            page_importance = np.array(page_importance)
            if policy == "lru":
                # Keep most recently accessed (last n_pages pages)
                hit_rate = n_pages / n_total_pages
            elif policy == "lfu":
                # Keep most frequently accessed
                top = np.argsort(page_importance)[-n_pages:]
                hit_rate = float(page_importance[top].sum() / max(page_importance.sum(), 1))
            else:  # importance → weighted by access pattern
                top = np.argsort(page_importance)[-n_pages:]
                hit_rate = float(page_importance[top].sum() / max(page_importance.sum(), 1))
        # Overhead: smaller pages = more metadata
        overhead = n_pages * 0.001
        score = hit_rate * 100 + mem_eff * 30 - overhead * 10 - (page_size / 256) * 5
        return {"score": float(score), "behavioral": (hit_rate, mem_eff),
                "metadata": {"page_size": page_size, "n_pages": n_pages, "eviction_policy": policy,
                             "capacity": capacity, "hit_rate": hit_rate, "mem_eff": mem_eff}}

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [("hit_rate", 10, 0.0, 1.0), ("mem_efficiency", 10, 0.0, 1.0)]

    def discrete_choices(self) -> dict[str, list] | None:
        return {"page_size": [16, 32, 64, 128, 256], "n_pages": [64, 128, 256, 512],
                "eviction_policy": self.POLICIES}

    def seed_configs(self) -> list[dict[str, Any]]:
        return [{"page_size": ps, "n_pages": 256, "eviction_policy": pol}
                for ps in [32, 64, 128] for pol in self.POLICIES]

    def to_cpu(self) -> "PagedEvictKV": return self
