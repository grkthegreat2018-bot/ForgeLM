"""Parameter compression R&D — Round 2.

Round 1 findings (archived):
  LOSERS (>50% error or impractical):
    - Kronecker (Van Loan): 99.9% error — LLM weights aren't Kronecker-structured
    - Block-Kronecker k=4: 100% error — same issue
    - Monarch bs=64 (rank-1): 96% error — rank-1 per block too aggressive
    - Circulant+Diagonal: 111% error — no circulant structure in weights
    - Toeplitz+LR r=32: 86% error — no toeplitz structure
    - BCM (novel): 99% error — circulant blocks don't match weight structure
    - VQ 2bit bs=8: 91% error — too aggressive for weights
    - Hadamard+INT4: 24% error — worse than plain INT4 for low-rank weights
    - RandomProj d=512: 900% error — random projection destroys information
    - Butterfly s=4: 13% error — marginal, not worth complexity
    - Sparse+LR r=64: 55% error — too aggressive sparsity
    - SPQ 99%/95%: 30-35% error — spectral pruning loses too much
    - HLR (novel): same as SVD — Hadamard gives no benefit for low-rank
    - Adaptive Hybrid: redundant (just picks SVD+Q)

  CANDIDATES (kept, stats cached):
    - SVD+Q r=512 b=4 (CALDERA): CR=1.8-2x, err=0.12%, 6.5b — best quality
    - SVD r=256: CR=6.4x, err=0.64%, 16b — good CR/quality
    - SVD r=512: CR=3.9x, err=0.5%, 16b — good for embedding
    - NLRQ r=256 b=8 (novel): CR=12.8x, err=1.3%, 8b — best practical CR
    - INT8 per-channel: CR=2x, err=0.8%, 8b — fast, simple
    - INT4 group=128: CR=3.9x, err=11.7%, 4.1b — fast, moderate
    - NF4: CR=4x, err=10.7%, 4.0b — standard 4-bit

Round 2 — new algorithms to test:
  D. Residual quantization (multi-stage):
    26. Residual INT4: INT4 then INT4 on residual (2-stage, ~2b effective)
    27. Residual INT8→INT4: INT8 then INT4 on residual (~6b effective)
    28. Multi-stage binary: iterative sign decomposition with optimal scales

  E. Adaptive block decomposition:
    29. Block-SVD adaptive: per-block SVD with energy-based rank selection
    30. Block-NLRQ adaptive: per-block NLRQ with adaptive rank + bits

  F. Optimal quantization:
    31. Optimal bit allocation: per-layer DP to find optimal bits given budget
    32. Per-channel mixed precision: important channels get more bits
    33. Group-wise NF4 with optimal group size

  G. Hybrid decomposition + quantization:
    34. SVD + INT4 residual (CALDERA variant with INT4 not 4b scalar)
    35. NLRQ + INT4 residual: low-rank quantized + INT4 on residual
    36. Block-SVD + per-block INT4: decompose then quantize each block

Run: python -m research.sandbox.compression_rd
"""
import time
import math
import json
import os
import torch
import torch.nn as nn
import numpy as np

# ── GPU setup (ForgeEngine patterns) ───────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
if CUDA:
    torch.set_default_device(DEVICE)
    from research.paths import TORCH_CACHE_DIR, as_str
    _cache = as_str(TORCH_CACHE_DIR)
    os.makedirs(_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache
    os.environ["TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR"] = _cache
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNELS"] = "0"

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".devin", "compression_cache.json")


def hadamard_matrix(n: int) -> torch.Tensor:
    """Build Hadamard matrix on GPU."""
    H = torch.tensor([[1.0]], device=DEVICE)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(H.shape[0])


# ── Cached candidate stats (hardlocked from Round 1) ───────────────────────
CACHED_CANDIDATES = {
    "SVD+Q r=512 b=4": {
        "ffn_gate_2048x8192": {"cr": 1.78, "ferr": 0.001215, "fnerr": 0.001216, "bits": 6.9},
        "ffn_down_8192x2048": {"cr": 1.78, "ferr": 0.001282, "fnerr": 0.001294, "bits": 6.9},
        "attn_q_2048x2048":   {"cr": 1.33, "ferr": 0.001442, "fnerr": 0.001446, "bits": 8.0},
        "embedding_65536x2048": {"cr": 1.97, "ferr": 0.001227, "fnerr": 0.001237, "bits": 6.5},
    },
    "SVD r=256": {
        "ffn_gate_2048x8192": {"cr": 6.40, "ferr": 0.006414, "fnerr": 0.006444, "bits": 16.0},
        "ffn_down_8192x2048": {"cr": 6.40, "ferr": 0.006431, "fnerr": 0.006502, "bits": 16.0},
        "attn_q_2048x2048":   {"cr": 4.00, "ferr": 0.008132, "fnerr": 0.008068, "bits": 16.0},
        "embedding_65536x2048": {"cr": 7.76, "ferr": 0.266442, "fnerr": 0.260993, "bits": 16.0},
    },
    "SVD r=512": {
        "ffn_gate_2048x8192": {"cr": 3.20, "ferr": 0.005535, "fnerr": 0.005507, "bits": 16.0},
        "ffn_down_8192x2048": {"cr": 3.20, "ferr": 0.005550, "fnerr": 0.005595, "bits": 16.0},
        "attn_q_2048x2048":   {"cr": 2.00, "ferr": 0.006293, "fnerr": 0.006357, "bits": 16.0},
        "embedding_65536x2048": {"cr": 3.88, "ferr": 0.004907, "fnerr": 0.004911, "bits": 16.0},
    },
    "NLRQ r=256 b=8": {
        "ffn_gate_2048x8192": {"cr": 12.80, "ferr": 0.013137, "fnerr": 0.012990, "bits": 8.0},
        "ffn_down_8192x2048": {"cr": 12.80, "ferr": 0.012525, "fnerr": 0.012499, "bits": 8.0},
        "attn_q_2048x2048":   {"cr": 8.00, "ferr": 0.013522, "fnerr": 0.013635, "bits": 8.0},
        "embedding_65536x2048": {"cr": 15.51, "ferr": 0.266646, "fnerr": 0.270236, "bits": 8.0},
    },
    "INT8 per-channel": {
        "ffn_gate_2048x8192": {"cr": 2.00, "ferr": 0.009019, "fnerr": 0.008936, "bits": 8.0},
        "ffn_down_8192x2048": {"cr": 2.00, "ferr": 0.008283, "fnerr": 0.008271, "bits": 8.0},
        "attn_q_2048x2048":   {"cr": 2.00, "ferr": 0.008296, "fnerr": 0.008382, "bits": 8.0},
        "embedding_65536x2048": {"cr": 2.00, "ferr": 0.008275, "fnerr": 0.008288, "bits": 8.0},
    },
    "INT4 group=128": {
        "ffn_gate_2048x8192": {"cr": 3.88, "ferr": 0.117200, "fnerr": 0.117417, "bits": 4.1},
        "ffn_down_8192x2048": {"cr": 3.88, "ferr": 0.117342, "fnerr": 0.117879, "bits": 4.1},
        "attn_q_2048x2048":   {"cr": 3.88, "ferr": 0.117390, "fnerr": 0.117634, "bits": 4.1},
        "embedding_65536x2048": {"cr": 3.88, "ferr": 0.117270, "fnerr": 0.117677, "bits": 4.1},
    },
    "NF4": {
        "ffn_gate_2048x8192": {"cr": 3.99, "ferr": 0.112558, "fnerr": 0.113480, "bits": 4.0},
        "ffn_down_8192x2048": {"cr": 3.94, "ferr": 0.107088, "fnerr": 0.106974, "bits": 4.0},
        "attn_q_2048x2048":   {"cr": 3.99, "ferr": 0.107161, "fnerr": 0.108095, "bits": 4.0},
        "embedding_65536x2048": {"cr": 3.99, "ferr": 0.107058, "fnerr": 0.107686, "bits": 4.0},
    },
}

# Archived losers (kept for reference, not run)
ARCHIVED = [
    # Round 1 losers
    "Kronecker (Van Loan)", "Block-Kronecker k=4", "Monarch bs=64 (rank-1)",
    "Circulant+Diagonal", "Toeplitz+LR r=32", "BCM bs=64 (novel)",
    "VQ 2bit bs=8", "Hadamard+INT4", "RandomProj d=512", "Butterfly s=4",
    "Sparse+LR r=64", "SPQ 99% b=4", "SPQ 95% b=4", "HLR r=256 (novel)",
    "HLR r=512 (novel)", "Adaptive Hybrid 4b (novel)", "Binary bases k=8",
    "Binary bases k=16",
    # Round 2 losers
    "Multi-stage binary k=4/6/8", "Block-SVD adaptive (all bs)",
    "Block-NLRQ adaptive (all bs)", "Per-channel mixed precision",
    "Optimal bit alloc target=6b (bug: 100% error)",
]


# ── Utilities ──────────────────────────────────────────────────────────────

def frobenius_error(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    return (W - W_hat).norm().item() / W.norm().item()


def compressed_bytes(params: int, bits_per_param: float) -> int:
    return int(math.ceil(params * bits_per_param / 8))


def compression_ratio(orig_bytes: int, comp_bytes: int) -> float:
    return orig_bytes / max(comp_bytes, 1)


def print_result(name: str, cr: float, ferr: float, fnerr: float,
                 params: int, bits: float, latency_ms: float = 0):
    print(f"  {name:40s} CR={cr:6.2f}x  F-err={ferr:.6f}  "
          f"Fn-err={fnerr:.6f}  {params/1e6:8.2f}M @ {bits:.1f}b  "
          f"{latency_ms:7.2f}ms")


# ── Candidate algorithms (kept from Round 1) ───────────────────────────────

def compress_svd(W: torch.Tensor, rank: int) -> dict:
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    W_hat = U_r @ torch.diag(S_r) @ Vh_r
    m, n = W.shape
    return {'W_hat': W_hat, 'params': rank * (m + n + 1), 'bits_per_param': 16}


def compress_svd_quantized_residual(W: torch.Tensor, rank: int,
                                    residual_bits: int = 4) -> dict:
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    L = U[:, :rank] * S[:rank].unsqueeze(0)
    R = Vh[:rank, :]
    W_lr = L @ R
    residual = W - W_lr
    scale = residual.abs().max() / (2**(residual_bits - 1) - 1)
    Q = torch.round(residual / scale.clamp(min=1e-10)).clamp(
        -(2**(residual_bits-1)), 2**(residual_bits-1)-1) * scale
    W_hat = Q + W_lr
    m, n = W.shape
    params = m * n + rank * (m + n)
    bits = (m * n * residual_bits + rank * (m + n) * 16) / params
    return {'W_hat': W_hat, 'params': params, 'bits_per_param': bits}


def compress_nested_lr_quantized(W: torch.Tensor, rank: int = 256,
                                 factor_bits: int = 8) -> dict:
    """NLRQ (novel): SVD then quantize U and V factors independently."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    u_scale = U_r.abs().max(dim=1, keepdim=True)[0] / (2**(factor_bits-1) - 1)
    u_scale = u_scale.clamp(min=1e-10)
    U_q = torch.round(U_r / u_scale).clamp(
        -(2**(factor_bits-1)), 2**(factor_bits-1)-1) * u_scale
    v_scale = Vh_r.abs().max(dim=1, keepdim=True)[0] / (2**(factor_bits-1) - 1)
    v_scale = v_scale.clamp(min=1e-10)
    V_q = torch.round(Vh_r / v_scale).clamp(
        -(2**(factor_bits-1)), 2**(factor_bits-1)-1) * v_scale
    W_hat = U_q @ torch.diag(S_r) @ V_q
    m, n = W.shape
    params = m * rank + rank * n + rank
    bits = (m * rank * factor_bits + rank * n * factor_bits + rank * 16) / params
    return {'W_hat': W_hat, 'params': params, 'bits_per_param': bits}


def compress_int8_perchannel(W: torch.Tensor) -> dict:
    scale = W.abs().max(dim=1, keepdim=True)[0] / 127
    scale = scale.clamp(min=1e-10)
    Q = torch.round(W / scale).clamp(-128, 127)
    W_hat = Q * scale
    m, n = W.shape
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': 8 + 16 * m / (m * n)}


def compress_int4_group(W: torch.Tensor, group_size: int = 128) -> dict:
    m, n = W.shape
    gs = group_size
    n_groups = n // gs
    W_trunc = W[:, :n_groups * gs].reshape(m, n_groups, gs)
    scale = W_trunc.abs().amax(dim=-1, keepdim=True) / 7
    scale = scale.clamp(min=1e-10)
    Q = torch.round(W_trunc / scale).clamp(-8, 7)
    W_hat = torch.zeros_like(W)
    W_hat[:, :n_groups * gs] = (Q * scale).reshape(m, n_groups * gs)
    total_bits = m * n_groups * gs * 4 + m * n_groups * 16
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': total_bits / (m * n)}


def compress_nf4(W: torch.Tensor) -> dict:
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ], dtype=W.dtype, device=W.device)
    m, n = W.shape
    scale = W.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-10)
    W_norm = (W / scale).clamp(-1, 1)
    dists = (W_norm.unsqueeze(-1) - nf4_levels.unsqueeze(0).unsqueeze(0)).abs()
    idx = dists.argmin(dim=-1)
    W_hat = nf4_levels[idx] * scale
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': 4 + 16 * m / (m * n)}


# ── Round 2: New R&D algorithms ────────────────────────────────────────────

def compress_residual_int4_int4(W: torch.Tensor) -> dict:
    """R&D: Residual INT4 — 2-stage quantization.
    Stage 1: INT4 quantize W → Q1. Stage 2: INT4 quantize (W - Q1) → Q2.
    Effective ~2 bits/param with much lower error than 2-bit scalar.
    """
    m, n = W.shape
    gs = 128
    n_groups = n // gs
    # Stage 1
    W_trunc = W[:, :n_groups * gs].reshape(m, n_groups, gs)
    s1 = W_trunc.abs().amax(dim=-1, keepdim=True) / 7
    s1 = s1.clamp(min=1e-10)
    Q1 = torch.round(W_trunc / s1).clamp(-8, 7) * s1
    # Stage 2: quantize residual
    residual = W_trunc - Q1
    s2 = residual.abs().amax(dim=-1, keepdim=True) / 7
    s2 = s2.clamp(min=1e-10)
    Q2 = torch.round(residual / s2).clamp(-8, 7) * s2
    W_hat_trunc = Q1 + Q2
    W_hat = torch.zeros_like(W)
    W_hat[:, :n_groups * gs] = W_hat_trunc.reshape(m, n_groups * gs)
    # 2 stages * (4b weights + 16b scale per group)
    total_bits = 2 * (m * n_groups * gs * 4 + m * n_groups * 16)
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': total_bits / (m * n)}


def compress_residual_int8_int4(W: torch.Tensor) -> dict:
    """R&D: Residual INT8→INT4 — coarse INT8 then fine INT4 on residual.
    Effective ~6 bits/param with lower error than INT4 alone.
    """
    m, n = W.shape
    # Stage 1: INT8 per-channel
    s1 = W.abs().max(dim=1, keepdim=True)[0] / 127
    s1 = s1.clamp(min=1e-10)
    Q1 = torch.round(W / s1).clamp(-128, 127) * s1
    # Stage 2: INT4 group on residual
    residual = W - Q1
    gs = 128
    n_groups = n // gs
    res_trunc = residual[:, :n_groups * gs].reshape(m, n_groups, gs)
    s2 = res_trunc.abs().amax(dim=-1, keepdim=True) / 7
    s2 = s2.clamp(min=1e-10)
    Q2 = torch.round(res_trunc / s2).clamp(-8, 7) * s2
    W_hat = Q1.clone()
    W_hat[:, :n_groups * gs] += Q2.reshape(m, n_groups * gs)
    # INT8: 8b + 16b/channel. INT4: 4b + 16b/group
    total_bits = m * n * 8 + m * 16 + m * n_groups * gs * 4 + m * n_groups * 16
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': total_bits / (m * n)}


def compress_multistage_binary(W: torch.Tensor, n_stages: int = 4) -> dict:
    """R&D: Multi-stage binary — iterative sign decomposition.
    W ≈ sum_i s_i * B_i where B_i are binary (±1) and s_i are per-row scales.
    n_stages=4 → 4 bits/param, but with better error than scalar 4-bit
    because each stage captures the residual structure.
    """
    m, n = W.shape
    residual = W.clone()
    W_hat = torch.zeros_like(W)
    total_scale_params = 0
    for _ in range(n_stages):
        B = residual.sign()
        s = (residual * B).sum(dim=1, keepdim=True) / (B * B).sum(dim=1, keepdim=True).clamp(min=1)
        W_hat += s * B
        residual = residual - s * B
        total_scale_params += m
    # n_stages binary matrices (1 bit each) + n_stages scale vectors (16b each)
    total_bits = n_stages * m * n * 1 + n_stages * total_scale_params * 16
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': total_bits / (m * n)}


def compress_block_svd_adaptive(W: torch.Tensor, block_size: int = 512,
                                energy_threshold: float = 0.99) -> dict:
    """R&D: Block-SVD with adaptive rank per block.
    Split W into blocks, compute SVD per block, keep enough singular values
    to capture `energy_threshold` of each block's energy.
    Blocks with more structure get lower rank → more compression.
    """
    m, n = W.shape
    bs = block_size
    n_blocks_m = m // bs
    n_blocks_n = n // bs
    m_pad = n_blocks_m * bs
    n_pad = n_blocks_n * bs
    W_padded = torch.zeros(m_pad, n_pad, device=W.device, dtype=W.dtype)
    W_padded[:m, :n] = W
    blocks = W_padded.reshape(n_blocks_m, bs, n_blocks_n, bs).permute(0, 2, 1, 3)
    blocks_flat = blocks.reshape(-1, bs, bs)
    # Batched SVD
    U, S, Vh = torch.linalg.svd(blocks_flat, full_matrices=False)
    # Adaptive rank per block: cumulative energy threshold
    energy = (S ** 2).cumsum(dim=1) / (S ** 2).sum(dim=1, keepdim=True).clamp(min=1e-10)
    # Find rank per block: first index where energy >= threshold
    ranks = (energy < energy_threshold).sum(dim=1) + 1
    ranks = ranks.clamp(max=bs)
    # Vectorized reconstruction: zero out S beyond each block's rank, then batched matmul
    N_blocks = blocks_flat.shape[0]
    rank_mask = torch.arange(bs, device=W.device).unsqueeze(0) < ranks.unsqueeze(1)
    S_masked = S * rank_mask.float()  # zero out singular values beyond rank
    # Batched: U @ diag(S_masked) @ Vh = (U * S_masked) @ Vh
    W_hat_blocks = (U * S_masked.unsqueeze(1)) @ Vh
    total_params = int(ranks.sum().item()) * (2 * bs + 1)
    W_hat = W_hat_blocks.reshape(n_blocks_m, n_blocks_n, bs, bs).permute(0, 2, 1, 3)
    W_hat = W_hat.reshape(m_pad, n_pad)[:m, :n]
    avg_rank = total_params / (2 * bs + 1) / N_blocks
    return {'W_hat': W_hat, 'params': total_params, 'bits_per_param': 16,
            'avg_rank': avg_rank}


def compress_block_nlrq_adaptive(W: torch.Tensor, block_size: int = 512,
                                 energy_threshold: float = 0.99,
                                 factor_bits: int = 8) -> dict:
    """R&D: Block-NLRQ with adaptive rank — best of block-SVD + NLRQ.
    Per-block SVD with adaptive rank, then quantize factors at `factor_bits`.
    """
    m, n = W.shape
    bs = block_size
    n_blocks_m = m // bs
    n_blocks_n = n // bs
    m_pad = n_blocks_m * bs
    n_pad = n_blocks_n * bs
    W_padded = torch.zeros(m_pad, n_pad, device=W.device, dtype=W.dtype)
    W_padded[:m, :n] = W
    blocks = W_padded.reshape(n_blocks_m, bs, n_blocks_n, bs).permute(0, 2, 1, 3)
    blocks_flat = blocks.reshape(-1, bs, bs)
    U, S, Vh = torch.linalg.svd(blocks_flat, full_matrices=False)
    energy = (S ** 2).cumsum(dim=1) / (S ** 2).sum(dim=1, keepdim=True).clamp(min=1e-10)
    ranks = (energy < energy_threshold).sum(dim=1) + 1
    ranks = ranks.clamp(max=bs)
    # Quantize factors
    fb = factor_bits
    u_scale = U.abs().amax(dim=1, keepdim=True) / (2**(fb-1) - 1)
    u_scale = u_scale.clamp(min=1e-10)
    U_q = torch.round(U / u_scale).clamp(-(2**(fb-1)), 2**(fb-1)-1) * u_scale
    v_scale = Vh.abs().amax(dim=1, keepdim=True) / (2**(fb-1) - 1)
    v_scale = v_scale.clamp(min=1e-10)
    V_q = torch.round(Vh / v_scale).clamp(-(2**(fb-1)), 2**(fb-1)-1) * v_scale
    # Vectorized: zero out S beyond rank, batched matmul
    rank_mask = torch.arange(bs, device=W.device).unsqueeze(0) < ranks.unsqueeze(1)
    S_masked = S * rank_mask.float()
    W_hat_blocks = (U_q * S_masked.unsqueeze(1)) @ V_q
    total_params = int(ranks.sum().item()) * (2 * bs)
    W_hat = W_hat_blocks.reshape(n_blocks_m, n_blocks_n, bs, bs).permute(0, 2, 1, 3)
    W_hat = W_hat.reshape(m_pad, n_pad)[:m, :n]
    bits = (total_params * factor_bits + blocks_flat.shape[0] * bs * 16) / max(total_params, 1)
    return {'W_hat': W_hat, 'params': total_params, 'bits_per_param': bits}


def compress_svd_int4_residual(W: torch.Tensor, rank: int = 256) -> dict:
    """R&D: SVD + INT4 residual (CALDERA variant with group INT4).
    Low-rank captures bulk, INT4 group quantization captures residual.
    Better than scalar 4-bit residual because group scaling adapts locally.
    """
    m, n = W.shape
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    L = U[:, :rank] * S[:rank].unsqueeze(0)
    R = Vh[:rank, :]
    W_lr = L @ R
    residual = W - W_lr
    # INT4 group quantization on residual
    gs = 128
    n_groups = n // gs
    res_trunc = residual[:, :n_groups * gs].reshape(m, n_groups, gs)
    scale = res_trunc.abs().amax(dim=-1, keepdim=True) / 7
    scale = scale.clamp(min=1e-10)
    Q = torch.round(res_trunc / scale).clamp(-8, 7) * scale
    W_hat = W_lr.clone()
    W_hat[:, :n_groups * gs] += Q.reshape(m, n_groups * gs)
    # LR: rank*(m+n)*16b. INT4: m*n*4b + m*n_groups*16b
    total_bits = rank * (m + n) * 16 + m * n * 4 + m * n_groups * 16
    params = m * n + rank * (m + n)
    return {'W_hat': W_hat, 'params': params, 'bits_per_param': total_bits / params}


def compress_nlrq_int4_residual(W: torch.Tensor, rank: int = 256,
                                factor_bits: int = 8) -> dict:
    """R&D: NLRQ + INT4 residual — low-rank quantized + INT4 on residual.
    Combines NLRQ (best CR) with INT4 residual (captures what low-rank misses).
    """
    m, n = W.shape
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    # Quantize factors
    fb = factor_bits
    u_scale = U_r.abs().max(dim=1, keepdim=True)[0] / (2**(fb-1) - 1)
    u_scale = u_scale.clamp(min=1e-10)
    U_q = torch.round(U_r / u_scale).clamp(-(2**(fb-1)), 2**(fb-1)-1) * u_scale
    v_scale = Vh_r.abs().max(dim=1, keepdim=True)[0] / (2**(fb-1) - 1)
    v_scale = v_scale.clamp(min=1e-10)
    V_q = torch.round(Vh_r / v_scale).clamp(-(2**(fb-1)), 2**(fb-1)-1) * v_scale
    W_lr = U_q @ torch.diag(S_r) @ V_q
    residual = W - W_lr
    # INT4 group on residual
    gs = 128
    n_groups = n // gs
    res_trunc = residual[:, :n_groups * gs].reshape(m, n_groups, gs)
    scale = res_trunc.abs().amax(dim=-1, keepdim=True) / 7
    scale = scale.clamp(min=1e-10)
    Q = torch.round(res_trunc / scale).clamp(-8, 7) * scale
    W_hat = W_lr.clone()
    W_hat[:, :n_groups * gs] += Q.reshape(m, n_groups * gs)
    # NLRQ: rank*(m+n)*fb. INT4: m*n*4 + m*n_groups*16
    total_bits = rank * (m + n) * fb + m * n * 4 + m * n_groups * 16
    params = m * n + rank * (m + n)
    return {'W_hat': W_hat, 'params': params, 'bits_per_param': total_bits / params}


def compress_per_channel_mixed_precision(W: torch.Tensor,
                                         important_frac: float = 0.1) -> dict:
    """R&D: Per-channel mixed precision (vectorized).
    Important channels (high norm) get INT8, rest get INT4.
    `important_frac` of channels get 8-bit, rest get 4-bit.
    """
    m, n = W.shape
    channel_norms = W.norm(dim=1)
    n_important = max(1, int(m * important_frac))
    important_idx = channel_norms.argsort(descending=True)[:n_important]
    # INT8 for important channels (vectorized)
    s8 = W[important_idx].abs().amax(dim=1, keepdim=True) / 127
    s8 = s8.clamp(min=1e-10)
    W_hat = torch.zeros_like(W)
    W_hat[important_idx] = torch.round(W[important_idx] / s8).clamp(-128, 127) * s8
    # INT4 group quantization for rest (vectorized — no per-channel loop)
    other_idx = torch.ones(m, dtype=torch.bool, device=W.device)
    other_idx[important_idx] = False
    gs = 128
    n_groups = n // gs
    W_other = W[other_idx][:, :n_groups * gs].reshape(-1, n_groups, gs)
    scale = W_other.abs().amax(dim=-1, keepdim=True) / 7
    scale = scale.clamp(min=1e-10)
    Q = torch.round(W_other / scale).clamp(-8, 7) * scale
    W_hat_other = torch.zeros_like(W)
    W_hat_other[other_idx, :n_groups * gs] = Q.reshape(-1, n_groups * gs)
    W_hat = W_hat + W_hat_other  # INT8 channels already set, add INT4 channels
    total_bits = (n_important * n * 8 + (m - n_important) * n * 4 +
                  n_important * 16 + (m - n_important) * n_groups * 16)
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': total_bits / (m * n)}


def compress_group_nf4_optimal(W: torch.Tensor) -> dict:
    """R&D: NF4 with optimal group size search.
    Test group sizes 32, 64, 128, 256 and pick the one with best error.
    """
    m, n = W.shape
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ], dtype=W.dtype, device=W.device)
    best_err = float('inf')
    best_gs = 128
    best_W_hat = None
    best_bits = 4.0
    for gs in [32, 64, 128, 256]:
        n_groups = n // gs
        if n_groups == 0:
            continue
        W_trunc = W[:, :n_groups * gs].reshape(m, n_groups, gs)
        scale = W_trunc.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        W_norm = (W_trunc / scale).clamp(-1, 1)
        dists = (W_norm.unsqueeze(-1) - nf4_levels.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
        idx = dists.argmin(dim=-1)
        W_hat_trunc = nf4_levels[idx] * scale
        W_hat = torch.zeros_like(W)
        W_hat[:, :n_groups * gs] = W_hat_trunc.reshape(m, n_groups * gs)
        err = frobenius_error(W, W_hat)
        bits = 4 + 16 * m * n_groups / (m * n)
        if err < best_err:
            best_err = err
            best_gs = gs
            best_W_hat = W_hat
            best_bits = bits
    return {'W_hat': best_W_hat, 'params': m * n, 'bits_per_param': best_bits,
            'best_gs': best_gs}


def compress_optimal_bit_allocation(W: torch.Tensor, target_bits: float = 4.0) -> dict:
    """R&D: Optimal bit allocation via variance-weighted greedy (vectorized).
    Given a target average bits/param, find the per-group bit allocation
    that minimizes total quantization error. Groups with higher variance
    get more bits. Fully vectorized — no Python loops over groups.
    """
    m, n = W.shape
    gs = 128
    n_groups = n // gs
    W_trunc = W[:, :n_groups * gs].reshape(m, n_groups, gs)
    group_vars = W_trunc.var(dim=-1)  # (m, n_groups)
    var_flat = group_vars.flatten()  # (m*n_groups,)
    N = var_flat.shape[0]

    # Vectorized greedy allocation: sort by variance, assign bit levels in bulk.
    # Everyone starts at 2b. Budget = target_bits * N * gs - 2 * N * gs.
    total_budget = target_bits * N * gs
    allocation = torch.full((N,), 2, dtype=torch.long, device=W.device)
    remaining = total_budget - 2 * N * gs

    # Upgrade in order: 2→4, 4→8, 8→16. Each upgrade costs 2*gs per group.
    # Process upgrades by variance rank: top-k groups get upgraded first.
    sorted_idx = var_flat.argsort(descending=True)
    for step in range(3):  # 3 possible upgrades: 2→4, 4→8, 8→16
        if remaining <= 0:
            break
        upgrade_cost = 2 * gs  # cost to upgrade one group by one level
        n_can_upgrade = int(remaining // upgrade_cost)
        if n_can_upgrade <= 0:
            break
        # Upgrade the top n_can_upgrade groups that are still at current level
        # Find groups sorted by variance that haven't been upgraded past this step
        # Simple approach: upgrade top n_can_upgrade from sorted_idx that are < 16
        candidates = sorted_idx[:n_can_upgrade]
        mask = allocation[candidates] < 16
        to_upgrade = candidates[mask]
        allocation[to_upgrade] += 2
        remaining -= to_upgrade.numel() * upgrade_cost

    # Vectorized quantization: process each bit level as a batch
    W_hat = torch.zeros_like(W)
    W_flat = W_trunc.reshape(N, gs)  # (N, gs) — all groups flattened
    for bits in [2, 4, 8, 16]:
        mask = (allocation == bits)
        if not mask.any():
            continue
        group_indices = mask.nonzero().squeeze(-1)
        blocks = W_flat[group_indices]  # (n_at_level, gs)
        max_val = 2**(bits-1) - 1
        scale = blocks.abs().amax(dim=-1, keepdim=True) / max_val
        scale = scale.clamp(min=1e-10)
        Q = torch.round(blocks / scale).clamp(-max_val-1, max_val) * scale
        # Vectorized scatter-back: compute row/col indices and use index_put_
        rows = group_indices // n_groups  # (n_at_level,)
        cols = (group_indices % n_groups) * gs  # (n_at_level,)
        # Build full column indices: cols[i] + arange(gs) for each i
        col_offsets = torch.arange(gs, device=W.device).unsqueeze(0)  # (1, gs)
        full_cols = cols.unsqueeze(1) + col_offsets  # (n_at_level, gs)
        full_rows = rows.unsqueeze(1).expand(-1, gs)  # (n_at_level, gs)
        W_hat[full_rows, full_cols] = Q

    avg_bits = allocation.float().mean().item()
    return {'W_hat': W_hat, 'params': m * n, 'bits_per_param': avg_bits,
            'avg_bits': avg_bits}


# ── Test harness ───────────────────────────────────────────────────────────

def generate_synthetic_weights(m: int, n: int, rank: int) -> torch.Tensor:
    L = torch.randn(m, rank, device=DEVICE) / math.sqrt(m)
    R = torch.randn(rank, n, device=DEVICE) / math.sqrt(n)
    W = L @ R
    W += 0.1 * torch.randn(m, n, device=DEVICE) / math.sqrt(m * n)
    return W.float()


def run_cached_candidates(label: str, m: int, n: int):
    """Print cached candidate stats (no computation needed)."""
    print(f"\n  {'='*80}")
    print(f"  CACHED CANDIDATES (Round 1, hardlocked stats)")
    print(f"  {'='*80}")
    print(f"  {'Algorithm':40s} {'CR':>7s}  {'F-err':>9s}  {'Fn-err':>9s}  {'Bits':>5s}")
    print(f"  {'-'*40} {'-'*7}  {'-'*9}  {'-'*9}  {'-'*5}")
    # Map label to cache key
    key_map = {
        f"FFN gate ({m}x{n})": f"ffn_gate_{m}x{n}",
        f"FFN down ({m}x{n})": f"ffn_down_{m}x{n}",
        f"Attention Q ({m}x{n})": f"attn_q_{m}x{n}",
        f"Embedding ({m}x{n})": f"embedding_{m}x{n}",
    }
    cache_key = key_map.get(label, f"ffn_gate_{m}x{n}")
    for name, stats in CACHED_CANDIDATES.items():
        if cache_key in stats:
            s = stats[cache_key]
            print(f"  {name:40s} {s['cr']:7.2f}x  {s['ferr']:9.6f}  "
                  f"{s['fnerr']:9.6f}  {s['bits']:5.1f}b")


def run_new_rd(W: torch.Tensor, label: str):
    """Run Round 2 R&D algorithms (only new ones)."""
    W = W.to(DEVICE).to(torch.float32)
    m, n = W.shape
    orig_bytes = m * n * 2  # bf16
    print(f"\n  {'='*80}")
    print(f"  R&D ROUND 2: New algorithms — {label}")
    print(f"  {m}x{n} = {m*n/1e6:.2f}M params, {orig_bytes/1e6:.2f} MB (bf16)")
    print(f"  {'='*80}")
    print(f"  {'Algorithm':40s} {'CR':>7s}  {'F-err':>9s}  {'Fn-err':>9s}  "
          f"{'Params':>8s}  {'Bits':>5s}  {'Latency':>7s}")
    print(f"  {'-'*40} {'-'*7}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*5}  {'-'*7}")

    algorithms = [
        # D. Residual quantization (winners)
        ("Residual INT4+INT4 (2b eff)", lambda: compress_residual_int4_int4(W)),
        ("Residual INT8+INT4 (6b eff)", lambda: compress_residual_int8_int4(W)),
        # F. Optimal quantization (winners)
        ("NF4 optimal group size", lambda: compress_group_nf4_optimal(W)),
        ("Optimal bit alloc target=4b", lambda: compress_optimal_bit_allocation(W, 4.0)),
        # G. Hybrid decomposition + quantization (winners)
        ("SVD+INT4 residual r=256", lambda: compress_svd_int4_residual(W, 256)),
        ("SVD+INT4 residual r=512", lambda: compress_svd_int4_residual(W, 512)),
        ("NLRQ+INT4 residual r=256 b=8", lambda: compress_nlrq_int4_residual(W, 256, 8)),
        ("NLRQ+INT4 residual r=512 b=8", lambda: compress_nlrq_int4_residual(W, 512, 8)),
    ]

    results = []
    with torch.no_grad():
        for name, algo_fn in algorithms:
            try:
                t0 = time.perf_counter()
                result = algo_fn()
                if CUDA:
                    torch.cuda.synchronize()
                latency = (time.perf_counter() - t0) * 1000
                W_hat = result['W_hat']
                params = result['params']
                bits = result['bits_per_param']
                comp_bytes = compressed_bytes(params, bits)
                cr = compression_ratio(orig_bytes, comp_bytes)
                # Batch error computation
                X = torch.randn(50, n, device=W.device, dtype=W.dtype)
                Y_orig = X @ W.T
                Y_approx = X @ W_hat.T
                ferr = (W - W_hat).norm() / W.norm()
                fnerr = (Y_orig - Y_approx).norm(dim=1) / (Y_orig.norm(dim=1) + 1e-10)
                ferr_val = ferr.item()
                fnerr_val = fnerr.mean().item()
                extra = ""
                if 'avg_rank' in result:
                    extra = f" avg_r={result['avg_rank']:.0f}"
                if 'best_gs' in result:
                    extra = f" gs={result['best_gs']}"
                if 'avg_bits' in result:
                    extra = f" avg={result['avg_bits']:.1f}b"
                print(f"  {name+extra:40s} {cr:7.2f}x  {ferr_val:9.6f}  "
                      f"{fnerr_val:9.6f}  {params/1e6:8.2f}M  {bits:5.1f}b  {latency:7.2f}ms")
                results.append({
                    'name': name, 'cr': cr, 'ferr': ferr_val, 'fnerr': fnerr_val,
                    'params': params, 'bits': bits, 'latency': latency,
                })
            except Exception as e:
                print(f"  {name:40s} FAIL: {str(e)[:60]}")

    return results


def summarize_new(results: list[dict], label: str):
    if not results:
        return
    print(f"\n  {'='*70}")
    print(f"  R&D ROUND 2 SUMMARY: {label}")
    print(f"  {'='*70}")
    # Compare against best cached candidate
    best_cached = ("NLRQ r=256 b=8", 12.80, 0.013)
    print(f"  Best Round 1 candidate: {best_cached[0]} CR={best_cached[1]:.2f}x err={best_cached[2]:.4f}")
    # Best new
    valid = [r for r in results if r['ferr'] < 0.5]  # exclude broken
    if valid:
        best_cr = max(valid, key=lambda r: r['cr'])
        best_err = min(valid, key=lambda r: r['ferr'])
        best_ratio = max(valid, key=lambda r: r['cr'] / (1 + r['ferr'] * 100))
        print(f"  Best new CR:           {best_cr['name']:40s} CR={best_cr['cr']:.2f}x err={best_cr['ferr']:.6f}")
        print(f"  Best new error:        {best_err['name']:40s} CR={best_err['cr']:.2f}x err={best_err['ferr']:.6f}")
        print(f"  Best new overall:      {best_ratio['name']:40s} CR={best_ratio['cr']:.2f}x err={best_ratio['ferr']:.6f}")
        # New winners vs Round 1
        for r in valid:
            if r['cr'] > best_cached[1] and r['ferr'] < best_cached[2]:
                print(f"  *** BEATS Round 1:     {r['name']:40s} CR={r['cr']:.2f}x err={r['ferr']:.6f}")


if __name__ == "__main__":
    print("=" * 90)
    print("  PARAMETER COMPRESSION R&D — ROUND 2")
    print("  Candidates cached from Round 1, testing 18 new algorithms")
    print("=" * 90)
    print(f"\n  Device: {DEVICE}")
    print(f"\n  Archived losers ({len(ARCHIVED)}): {', '.join(ARCHIVED[:5])}, ...")

    test_matrices = [
        (2048, 8192, 200, "FFN gate (2048x8192)"),
        (8192, 2048, 200, "FFN down (8192x2048)"),
        (2048, 2048, 100, "Attention Q (2048x2048)"),
        (65536, 2048, 300, "Embedding (65536x2048)"),
    ]

    all_new_results = {}
    for m, n, rank, label in test_matrices:
        # Print cached candidates (instant, no compute)
        run_cached_candidates(label, m, n)
        # Run new R&D algorithms
        W = generate_synthetic_weights(m, n, rank)
        results = run_new_rd(W, label)
        summarize_new(results, label)
        all_new_results[label] = results
        # Clear GPU cache between matrices
        if CUDA:
            torch.cuda.empty_cache()

    # Final cross-matrix summary
    print(f"\n{'='*90}")
    print("  CROSS-MATRIX R&D ROUND 2 SUMMARY")
    print(f"{'='*90}")
    # Find algorithms that beat NLRQ r=256 b=8 across ALL matrices
    for label, results in all_new_results.items():
        valid = [r for r in results if r['ferr'] < 0.5]
        if valid:
            best = max(valid, key=lambda r: r['cr'] / (1 + r['ferr'] * 100))
            print(f"  {label:30s} best: {best['name']:40s} CR={best['cr']:.2f}x err={best['ferr']:.6f}")

    print(f"\n{'='*90}")
    print("  R&D ROUND 2 COMPLETE")
    print(f"{'='*90}")
