"""Quantization domain: search FP4/NVFP4/AS-FP4/R-FP4 parameters.

Expanded search space (10 parameters):
  - block_size: {16, 32, 64, 128} (discrete)
  - scale_method: {absmax, mse_optimal, hybrid} (discrete)
  - residual_ratio: [0.0, 0.3] (continuous) — 0 = pure FP4, 0.3 = heavy residual
  - global_scale_factor: [0.5, 1.5] (continuous) — multiplier on global scale
  - scale_search_range: [0.3, 1.5] (continuous) — how far to search around absmax
  - scale_search_steps: {5, 10, 15, 20} (discrete) — grid search resolution
  - use_hadamard: {True, False} (discrete) — pre-quantization Hadamard rotation
  - hadamard_dim: {16, 32, 64} (discrete) — block size for rotation
  - scale_clip_min: [0.001, 0.1] (continuous) — zero-prevention floor
  - rounding_method: {rtn, stochastic, mse_round} (discrete) — rounding strategy

Evaluation: quantize a test weight matrix, measure (error, compression, speed).
Score = activation-weighted error (not Frobenius) + compression + speed.
Behavioral dims: (compression, fwd_error) → MAP-Elites explores quality/memory tradeoff.

CUDA: weight matrix, input, and all quantization ops run on GPU when available.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Any
from . import BaseDomain

# ── Triton fused scale-search kernel ──────────────────────────────────────
# Replaces the (out, n_blocks, n_steps, block_size) tensor expansion in
# _custom_scale_search with a single kernel that processes one block per
# program, iterating over candidate scales in registers. Eliminates OOM.

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _fused_scale_search_kernel(
        w_ptr,            # (out_f * n_blocks * block_size,) — flattened weights
        scale_out_ptr,    # (out_f, n_blocks) — best scale per block
        mse_out_ptr,      # (out_f, n_blocks) — best MSE per block
        out_f, n_blocks,
        BLOCK_SIZE: tl.constexpr,
        N_STEPS: tl.constexpr,
        SEARCH_LO: tl.constexpr,
        SEARCH_HI: tl.constexpr,
    ):
        """Each program handles one (row, block) pair.
        Iterates over N_STEPS candidate scales in registers, quantizes to
        FP4, measures MSE, writes the best. No 4D tensor expansion.
        """
        pid = tl.program_id(0)
        row = pid // n_blocks
        col = pid % n_blocks
        if row >= out_f:
            return

        # Load block of weights
        w_off = (row * n_blocks + col) * BLOCK_SIZE
        w = tl.load(w_ptr + w_off + tl.arange(0, BLOCK_SIZE))

        absmax = tl.max(tl.abs(w))
        base_scale = absmax / 6.0
        if base_scale < 1e-8:
            base_scale = 1e-8

        best_mse = float('inf')
        best_scale = base_scale

        for s in range(N_STEPS):
            frac = s / max(N_STEPS - 1, 1)
            mult = SEARCH_LO + frac * (SEARCH_HI - SEARCH_LO)
            scale = base_scale * mult
            if scale < 1e-12:
                scale = 1e-12

            w_norm = w / scale
            abs_norm = tl.abs(w_norm)

            # Inline FP4 quantization (searchsorted on 7 boundaries)
            mag = tl.where(abs_norm < 0.25, 0.0,
                  tl.where(abs_norm < 0.75, 0.5,
                  tl.where(abs_norm < 1.25, 1.0,
                  tl.where(abs_norm < 1.75, 1.5,
                  tl.where(abs_norm < 2.5,  2.0,
                  tl.where(abs_norm < 3.5,  3.0,
                  tl.where(abs_norm < 5.0,  4.0, 6.0)))))))

            w_dq = tl.where(w_norm >= 0, 1.0, -1.0) * mag * scale
            err = w - w_dq
            mse = tl.sum(err * err) / BLOCK_SIZE

            if mse < best_mse:
                best_mse = mse
                best_scale = scale

        tl.store(scale_out_ptr + row * n_blocks + col, best_scale)
        tl.store(mse_out_ptr + row * n_blocks + col, best_mse)


    def _triton_scale_search(w, block_size, search_range, n_steps, device):
        """Triton-accelerated scale search. Returns (best_scales, mse) per block."""
        out_f, in_f = w.shape
        pad = (block_size - in_f % block_size) % block_size
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        in_padded = w.shape[1]
        n_blocks = in_padded // block_size

        w_flat = w.contiguous().view(-1)
        best_scales = torch.empty(out_f, n_blocks, device=device, dtype=torch.float32)
        mse_out = torch.empty(out_f, n_blocks, device=device, dtype=torch.float32)

        grid = (out_f * n_blocks,)
        # Quantize search_range to constexpr (Triton needs compile-time consts for loop bounds)
        # Round n_steps to nearest power of 2 for efficiency
        n_steps_pow2 = max(4, 1 << (n_steps - 1).bit_length())
        lo = 1.0 - search_range
        hi = 1.0 + search_range

        _fused_scale_search_kernel[grid](
            w_flat, best_scales, mse_out,
            out_f, n_blocks,
            BLOCK_SIZE=block_size,
            N_STEPS=n_steps_pow2,
            SEARCH_LO=lo,
            SEARCH_HI=hi,
        )
        return best_scales, mse_out, w.view(out_f, n_blocks, block_size)


def _hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Generate a normalized Hadamard matrix of size n (n must be power of 2)."""
    if n == 1:
        return torch.ones(1, 1, device=device)
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h / np.sqrt(n)


def _apply_hadamard(w: torch.Tensor, dim: int,
                    device: torch.device) -> torch.Tensor:
    """Apply Hadamard rotation along the input dimension."""
    out_f, in_f = w.shape
    # Pad to multiple of dim
    pad = (dim - in_f % dim) % dim
    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad))
    in_padded = w.shape[1]
    n_blocks = in_padded // dim

    H = _hadamard_matrix(dim, device)  # (dim, dim)
    w_blocks = w.view(out_f, n_blocks, dim)  # (out, n_blocks, dim)
    # Rotate: w' = w @ H (each block gets rotated)
    w_rot = torch.bmm(w_blocks, H.unsqueeze(0).expand(out_f, -1, -1))
    return w_rot.view(out_f, in_padded)[:, :in_f]


def _quantize_with_options(w: torch.Tensor, block_size: int,
                           scale_method: str, scale_search_range: float,
                           scale_search_steps: int, scale_clip_min: float,
                           rounding_method: str,
                           device: torch.device) -> tuple:
    """Quantize with expanded options. Returns (packed, scales, global_scale)."""
    from research.inference.quant.nvfp4_quant import (
        _quantize_to_fp4, _dequantize_fp4, _FP4_MAGNITUDES, _FP4_BOUNDARIES,
    )

    if scale_method == "absmax":
        # Standard absmax with optional scale clipping
        packed, scales, global_scale = _quantize_to_fp4(w, block_size)
        if scale_clip_min > 0:
            # Clip global scale to prevent underflow
            global_scale = global_scale.clamp(min=scale_clip_min)
        return packed, scales, global_scale

    elif scale_method == "mse_optimal":
        from research.inference.quant.novel_quant import _quantize_to_fp4_adaptive
        packed, scales, global_scale = _quantize_to_fp4_adaptive(w, block_size)
        if scale_clip_min > 0:
            global_scale = global_scale.clamp(min=scale_clip_min)
        return packed, scales, global_scale

    else:  # hybrid — custom scale search with configurable range/steps
        # Use Triton kernel if available (eliminates OOM from 4D tensor expansion)
        if _HAS_TRITON and w.is_cuda:
            return _custom_scale_search_triton(w, block_size, scale_search_range,
                                               scale_search_steps, scale_clip_min,
                                               device)
        return _custom_scale_search(w, block_size, scale_search_range,
                                    scale_search_steps, scale_clip_min,
                                    rounding_method, device)


def _custom_scale_search_triton(w: torch.Tensor, block_size: int,
                                search_range: float, n_steps: int,
                                scale_clip_min: float,
                                device: torch.device) -> tuple:
    """Triton-accelerated scale search. Uses fused kernel for the hot loop,
    then does final quantization + packing in PyTorch (cheap, no 4D expansion).
    """
    from research.inference.quant.nvfp4_quant import _FP4_MAGNITUDES, _FP4_BOUNDARIES, _HAS_FP8

    best_scales, mse, w_blocks = _triton_scale_search(
        w, block_size, search_range, n_steps, device
    )
    out_f, n_blocks, _ = w_blocks.shape
    best_scale_exp = best_scales.unsqueeze(-1)  # (out, n_blocks, 1)

    # Two-level scaling
    global_scale = best_scales.amax(dim=1, keepdim=True).clamp(
        min=scale_clip_min if scale_clip_min > 0 else 1e-12)
    block_scale_normalized = best_scales / global_scale  # (out, n_blocks)

    # Re-quantize with best scale (small ops, no expansion)
    w_norm_final = w_blocks / best_scale_exp.clamp(min=1e-12)
    abs_norm_final = w_norm_final.abs()
    idx_final = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_final)
    idx_final = idx_final.clamp(0, 7)
    magnitude_final = _FP4_MAGNITUDES.to(device)[idx_final]
    # w_fp4 not needed for packing, just sign + idx
    sign_bit = (w_norm_final < 0).long() << 3
    mag_idx = idx_final.long()
    fp4_code = (sign_bit | mag_idx).to(torch.uint8)
    fp4_flat = fp4_code.view(out_f, -1)
    low = fp4_flat[:, 0::2] & 0x0F
    high = (fp4_flat[:, 1::2] << 4) & 0xF0
    packed = (low | high).contiguous()

    if _HAS_FP8:
        scales_fp8 = block_scale_normalized.to(torch.float8_e4m3fn)
    else:
        scales_fp8 = block_scale_normalized.to(torch.float16)

    global_scale_flat = global_scale.squeeze(-1).to(torch.float32)
    return packed, scales_fp8.contiguous(), global_scale_flat.contiguous()


def _custom_scale_search(w: torch.Tensor, block_size: int,
                         search_range: float, n_steps: int,
                         scale_clip_min: float, rounding_method: str,
                         device: torch.device) -> tuple:
    """Custom scale search with configurable range and resolution.

    Searches over candidate scales: absmax/6.0 * {1-search_range, ..., 1+search_range}
    Picks the scale that minimizes MSE per block.
    """
    from research.inference.quant.nvfp4_quant import (
        _FP4_MAGNITUDES, _FP4_BOUNDARIES, _HAS_FP8,
    )

    out_f, in_f = w.shape
    pad = (block_size - in_f % block_size) % block_size
    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad))
    in_padded = w.shape[1]
    n_blocks = in_padded // block_size

    w_blocks = w.view(out_f, n_blocks, block_size)

    absmax = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    base_scale = absmax / 6.0

    # Candidate multipliers: 1-search_range to 1+search_range
    candidates = torch.linspace(1 - search_range, 1 + search_range, n_steps,
                                dtype=w.dtype, device=device)
    scales = base_scale * candidates  # (out, n_blocks, n_steps)
    scales_exp = scales.unsqueeze(-1)  # (out, n_blocks, n_steps, 1)
    w_exp = w_blocks.unsqueeze(-2)  # (out, n_blocks, 1, block_size)

    w_norm = w_exp / scales_exp.clamp(min=1e-12)

    # Rounding
    if rounding_method == "stochastic":
        # Stochastic rounding: add uniform noise then round
        noise = torch.rand_like(w_norm) - 0.5
        abs_norm = (w_norm.abs() + noise).clamp(min=0)
    elif rounding_method == "mse_round":
        # MSE-optimal rounding: try both neighbors, pick the one with lower error
        abs_norm = w_norm.abs()
    else:  # rtn (round to nearest)
        abs_norm = w_norm.abs()

    idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm)
    idx = idx.clamp(0, 7)
    magnitude = _FP4_MAGNITUDES.to(device)[idx]
    w_q = torch.sign(w_norm) * magnitude
    w_dq = w_q * scales_exp

    # MSE per candidate
    mse = ((w_exp - w_dq) ** 2).mean(dim=-1)  # (out, n_blocks, n_steps)
    best_idx = mse.argmin(dim=-1, keepdim=True)
    best_scale = scales.gather(-1, best_idx)  # (out, n_blocks, 1)

    # Two-level scaling
    global_scale = best_scale.amax(dim=1, keepdim=True).clamp(min=scale_clip_min if scale_clip_min > 0 else 1e-12)
    block_scale_normalized = best_scale / global_scale

    # Re-quantize with best scale
    w_norm_final = w_blocks / best_scale.clamp(min=1e-12)
    abs_norm_final = w_norm_final.abs()
    idx_final = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_final)
    idx_final = idx_final.clamp(0, 7)
    magnitude_final = _FP4_MAGNITUDES.to(device)[idx_final]
    w_fp4 = torch.sign(w_norm_final) * magnitude_final

    # Pack
    sign_bit = (w_norm_final < 0).long() << 3
    mag_idx = idx_final.long()
    fp4_code = (sign_bit | mag_idx).to(torch.uint8)
    fp4_flat = fp4_code.view(out_f, -1)
    low = fp4_flat[:, 0::2] & 0x0F
    high = (fp4_flat[:, 1::2] << 4) & 0xF0
    packed = (low | high).contiguous()

    scale_flat = block_scale_normalized.squeeze(-1)
    if _HAS_FP8:
        scales_fp8 = scale_flat.to(torch.float8_e4m3fn)
    else:
        scales_fp8 = scale_flat.to(torch.float16)

    global_scale_flat = global_scale.squeeze(1).squeeze(-1).to(torch.float32)
    return packed, scales_fp8.contiguous(), global_scale_flat.contiguous()


class QuantDomain(BaseDomain):
    """Search quantization parameters on a test weight matrix (GPU-accelerated).

    Expanded search space with 10 parameters including Hadamard rotation,
    custom scale search, rounding methods, and zero-prevention.
    """

    BLOCK_SIZES = [16, 32, 64, 128]
    SCALE_METHODS = ["absmax", "mse_optimal", "hybrid"]
    ROUNDING_METHODS = ["rtn", "stochastic", "mse_round"]
    HADAMARD_DIMS = [16, 32, 64]

    def __init__(self, matrix_size: tuple = (256, 512), seed: int = 42,
                 weight_std: float = 0.02, outlier_frac: float = 0.01,
                 device: torch.device = None):
        self.matrix_size = matrix_size
        self.out_f, self.in_f = matrix_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))

        torch.manual_seed(seed)
        self.W = (torch.randn(self.out_f, self.in_f, device=self.device) * weight_std)
        outlier_mask = torch.rand(self.out_f, self.in_f, device=self.device) < outlier_frac
        self.W[outlier_mask] *= 5.0

        self.x = torch.randn(8, self.in_f, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            self.y_ref = torch.nn.functional.linear(self.x, self.W)

    def name(self) -> str:
        return "quant"

    def output_dim(self) -> int:
        return 10  # expanded search space

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()

        block_idx = int(p[0] * len(self.BLOCK_SIZES))
        block_idx = max(0, min(len(self.BLOCK_SIZES) - 1, block_idx))
        scale_idx = int(p[1] * len(self.SCALE_METHODS))
        scale_idx = max(0, min(len(self.SCALE_METHODS) - 1, scale_idx))
        round_idx = int(p[6] * len(self.ROUNDING_METHODS))
        round_idx = max(0, min(len(self.ROUNDING_METHODS) - 1, round_idx))
        hadamard_idx = int(p[7] * len(self.HADAMARD_DIMS))
        hadamard_idx = max(0, min(len(self.HADAMARD_DIMS) - 1, hadamard_idx))

        return {
            "block_size": self.BLOCK_SIZES[block_idx],
            "scale_method": self.SCALE_METHODS[scale_idx],
            "residual_ratio": float(np.clip(p[2], 0.0, 0.3)),
            "global_scale_factor": float(0.5 + p[3] * 1.0),  # [0.5, 1.5]
            "scale_search_range": float(0.3 + p[4] * 1.2),   # [0.3, 1.5]
            "scale_search_steps": [3, 5, 7, 10][int(p[5] * 4) % 4],
            "rounding_method": self.ROUNDING_METHODS[round_idx],
            "use_hadamard": bool(p[7] > 0.5),
            "hadamard_dim": self.HADAMARD_DIMS[hadamard_idx],
            "scale_clip_min": float(0.001 + p[8] * 0.099),   # [0.001, 0.1]
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        bs = config.get("block_size", 32)
        sm = config.get("scale_method", "absmax")
        rr = config.get("residual_ratio", 0.0)
        gs = config.get("global_scale_factor", 1.0)
        ssr = config.get("scale_search_range", 0.5)
        sss = config.get("scale_search_steps", 10)
        rm = config.get("rounding_method", "rtn")
        uh = config.get("use_hadamard", False)
        hd = config.get("hadamard_dim", 32)
        scm = config.get("scale_clip_min", 0.001)

        bs_idx = self.BLOCK_SIZES.index(bs) if bs in self.BLOCK_SIZES else 1
        sm_idx = self.SCALE_METHODS.index(sm) if sm in self.SCALE_METHODS else 0
        rm_idx = self.ROUNDING_METHODS.index(rm) if rm in self.ROUNDING_METHODS else 0
        hd_idx = self.HADAMARD_DIMS.index(hd) if hd in self.HADAMARD_DIMS else 1
        sss_idx = [5, 10, 15, 20].index(sss) if sss in [5, 10, 15, 20] else 1

        return torch.tensor([
            bs_idx / len(self.BLOCK_SIZES),
            sm_idx / len(self.SCALE_METHODS),
            rr / 0.3,
            (gs - 0.5) / 1.0,
            (ssr - 0.3) / 1.2,
            sss_idx / 4,
            rm_idx / len(self.ROUNDING_METHODS),
            (1.0 if uh else 0.0),
            (scm - 0.001) / 0.099,
            0.0,  # spare
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        """Quantize W with given config, measure activation-weighted error + compression + speed."""
        from research.inference.quant.nvfp4_quant import _dequantize_fp4

        block_size = config.get("block_size", 32)
        scale_method = config.get("scale_method", "absmax")
        residual_ratio = config.get("residual_ratio", 0.0)
        gs_factor = config.get("global_scale_factor", 1.0)
        scale_search_range = config.get("scale_search_range", 0.5)
        scale_search_steps = config.get("scale_search_steps", 10)
        rounding_method = config.get("rounding_method", "rtn")
        use_hadamard = config.get("use_hadamard", False)
        hadamard_dim = config.get("hadamard_dim", 32)
        scale_clip_min = config.get("scale_clip_min", 0.001)

        W = self.W.clone()

        try:
            # Apply Hadamard rotation if enabled
            if use_hadamard:
                W = _apply_hadamard(W, hadamard_dim, self.device)

            # Quantize with expanded options
            packed, scales, global_scale = _quantize_with_options(
                W, block_size, scale_method, scale_search_range,
                scale_search_steps, scale_clip_min, rounding_method,
                self.device,
            )

            # Apply global scale factor
            global_scale = global_scale * gs_factor

            # Dequantize
            W_dq = _dequantize_fp4(packed, scales, self.out_f, self.in_f,
                                   block_size, torch.float32,
                                   global_scale=global_scale)

            # Undo Hadamard rotation if applied
            if use_hadamard:
                H_inv = _hadamard_matrix(hadamard_dim, self.device).t()
                out_f_dq, in_f_dq = W_dq.shape
                pad = (hadamard_dim - in_f_dq % hadamard_dim) % hadamard_dim
                if pad > 0:
                    W_dq = torch.nn.functional.pad(W_dq, (0, pad))
                in_padded = W_dq.shape[1]
                n_blocks = in_padded // hadamard_dim
                w_blocks = W_dq.view(out_f_dq, n_blocks, hadamard_dim)
                w_rot = torch.bmm(w_blocks, H_inv.unsqueeze(0).expand(out_f_dq, -1, -1))
                W_dq = w_rot.view(out_f_dq, in_padded)[:, :self.in_f]

            # Activation-weighted error
            y_q = torch.nn.functional.linear(self.x, W_dq)
            fwd_err = (self.y_ref - y_q).norm().item() / self.y_ref.norm().item()
            # Compare to ORIGINAL self.W (not rotated W) for frob_err
            frob_err = (self.W - W_dq).norm().item() / self.W.norm().item()

            # Compression ratio
            w_bytes = self.out_f * self.in_f * 2  # bf16
            q_bytes = packed.numel() + scales.numel() + global_scale.numel() * 4
            if residual_ratio > 0:
                n_residual = int(self.out_f * self.in_f * residual_ratio)
                q_bytes += n_residual * 5
            compression = w_bytes / q_bytes

            # Speed: estimate from tensor sizes instead of benchmarking
            dequant_ms = (packed.numel() + scales.numel()) / 1e6  # proxy: MB → ms

            # Score: activation-weighted error + compression + speed
            score = -fwd_err * 100 + compression * 2 - dequant_ms * 0.01

            return {
                "score": float(score),
                "behavioral": (compression, fwd_err),
                "metadata": {
                    "frob_err": frob_err,
                    "fwd_err": fwd_err,
                    "compression": compression,
                    "dequant_ms": dequant_ms,
                    "q_bytes": q_bytes,
                    "block_size": block_size,
                    "scale_method": scale_method,
                    "rounding_method": rounding_method,
                    "use_hadamard": use_hadamard,
                    "hadamard_dim": hadamard_dim,
                    "scale_clip_min": scale_clip_min,
                    "scale_search_range": scale_search_range,
                    "scale_search_steps": scale_search_steps,
                },
            }

        except Exception as e:
            return {
                "score": -1000.0,
                "behavioral": (1.0, 1.0),
                "metadata": {"error": str(e)},
            }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [
            ("compression", 10, 1.0, 5.0),
            ("fwd_error", 10, 0.0, 0.3),
        ]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "block_size": self.BLOCK_SIZES,
            "scale_method": self.SCALE_METHODS,
            "rounding_method": self.ROUNDING_METHODS,
            "use_hadamard": [False, True],
            "hadamard_dim": self.HADAMARD_DIMS,
            "scale_search_steps": [5, 10, 15, 20],
            "residual_ratio": [0.0, 0.05, 0.10, 0.20],
            "global_scale_factor": [0.7, 0.85, 1.0, 1.15, 1.3],
            "scale_clip_min": [0.001, 0.01, 0.05, 0.1],
            "scale_search_range": [0.3, 0.5, 0.8, 1.0, 1.5],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        """Known-good configs to pre-evaluate and inject into the archive.
        These bootstrap the search so the surrogate has real data from gen 0.
        """
        seeds = []
        # Core combos: block_size × scale_method (the most impactful axes)
        for bs in [16, 32, 64]:
            for sm in ["absmax", "mse_optimal"]:
                seeds.append({
                    "block_size": bs, "scale_method": sm,
                    "residual_ratio": 0.0, "global_scale_factor": 1.0,
                    "scale_search_range": 0.5, "scale_search_steps": 10,
                    "rounding_method": "rtn", "use_hadamard": False,
                    "hadamard_dim": 32, "scale_clip_min": 0.001,
                })
        # Hadamard variants on the best known config
        for hd in [16, 32, 64]:
            seeds.append({
                "block_size": 16, "scale_method": "mse_optimal",
                "residual_ratio": 0.0, "global_scale_factor": 1.0,
                "scale_search_range": 0.5, "scale_search_steps": 10,
                "rounding_method": "rtn", "use_hadamard": True,
                "hadamard_dim": hd, "scale_clip_min": 0.001,
            })
        # Rounding method variants
        for rm in ["rtn", "stochastic", "mse_round"]:
            seeds.append({
                "block_size": 16, "scale_method": "mse_optimal",
                "residual_ratio": 0.0, "global_scale_factor": 1.0,
                "scale_search_range": 0.5, "scale_search_steps": 10,
                "rounding_method": rm, "use_hadamard": False,
                "hadamard_dim": 32, "scale_clip_min": 0.001,
            })
        # Scale search variants
        for ssr in [0.3, 0.8, 1.5]:
            for sss in [5, 20]:
                seeds.append({
                    "block_size": 16, "scale_method": "hybrid",
                    "residual_ratio": 0.0, "global_scale_factor": 1.0,
                    "scale_search_range": ssr, "scale_search_steps": sss,
                    "rounding_method": "rtn", "use_hadamard": False,
                    "hadamard_dim": 32, "scale_clip_min": 0.001,
                })
        # Residual variants
        for rr in [0.05, 0.10, 0.20]:
            seeds.append({
                "block_size": 16, "scale_method": "mse_optimal",
                "residual_ratio": rr, "global_scale_factor": 1.0,
                "scale_search_range": 0.5, "scale_search_steps": 10,
                "rounding_method": "rtn", "use_hadamard": False,
                "hadamard_dim": 32, "scale_clip_min": 0.001,
            })
        return seeds
