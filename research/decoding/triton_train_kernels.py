"""Triton fused training kernels: RMSNorm + SwiGLU.

Liger-Kernel-style fused kernels for training throughput. Replaces separate
PyTorch ops with single Triton kernels, reducing HBM traffic and kernel
launch overhead. Tuned for SM120 (RTX 5070, GDDR7 672 GB/s).

Community findings (2025-2026):
  - Liger-Kernel: 20% throughput, 60% VRAM reduction (LinkedIn)
  - Unsloth: 2-5x faster with RoPE/MLP Triton kernels
  - Fused RMSNorm: single kernel for x * rsqrt(mean(x^2) + eps) * weight
  - Fused SwiGLU: single kernel for silu(gate) * up → down (elementwise part)

Novel twist for SM120: GDDR7 has lower bandwidth than HBM (672 vs ~3000 GB/s),
so these kernels are MORE valuable here than on datacenter GPUs — the
bandwidth-bound ops dominate more. Block sizes tuned for SM120 occupancy:
  - RMSNorm: BLOCK = next_pow2(d_model), one program per (B, T) row
  - SwiGLU: BLOCK = next_pow2(intermediate), one program per (B, T) row

Usage:
    from research.decoding.triton_train_kernels import triton_rms_norm, triton_swiglu

    # RMSNorm (replaces F.rms_norm)
    out = triton_rms_norm(x, weight, eps)

    # SwiGLU elementwise (replaces F.silu(gate) * up)
    # NOTE: only fuses the activation, not the linear projections.
    # The w_gate/w_up/w_down linears stay as cuBLAS GEMMs (already optimal).
    # This fuses: silu(gate) * up into one kernel (saves 1 intermediate tensor).
    act = triton_swiglu_act(gate, up)
    out = w_down(act)

Fallback: pure PyTorch when Triton is unavailable (CPU, or kernel compile fail).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ─── PyTorch fallbacks ──────────────────────────────────────────────

def _pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure PyTorch RMSNorm (matches F.rms_norm output exactly)."""
    return F.rms_norm(x, [x.shape[-1]], weight, eps)


def _pytorch_swiglu_act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch SwiGLU activation: silu(gate) * up."""
    return F.silu(gate) * up


# ─── Triton kernels ─────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr, w_ptr, out_ptr,
        eps, N,
        stride_x_row, stride_x_col,
        stride_w, stride_out_row, stride_out_col,
        BLOCK: tl.constexpr,
    ):
        """RMSNorm kernel: one program per row of N elements.

        Computes: out = x * rsqrt(mean(x^2) + eps) * weight
        Uses fp32 accumulation for numerical stability.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)

        # Load x row (with masking for N < BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col,
                     mask=mask, other=0.0).to(tl.float32)

        # Compute mean(x^2) in fp32
        x2 = x * x
        mean_x2 = tl.sum(x2) / N

        # rsqrt(mean + eps)
        rstd = 1.0 / tl.sqrt(mean_x2 + eps)

        # Load weight
        w = tl.load(w_ptr + cols * stride_w, mask=mask, other=0.0).to(tl.float32)

        # Normalize and apply weight
        out = (x * rstd) * w

        # Store (cast back to input dtype)
        tl.store(out_ptr + row * stride_out_row + cols * stride_out_col,
                 out, mask=mask)

    @triton.jit
    def _swiglu_kernel(
        gate_ptr, up_ptr, out_ptr,
        N,
        stride_gate_row, stride_gate_col,
        stride_up_row, stride_up_col,
        stride_out_row, stride_out_col,
        BLOCK: tl.constexpr,
    ):
        """SwiGLU activation kernel: out = silu(gate) * up.

        silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        Fuses the silu + elementwise multiply into one kernel, saving one
        intermediate tensor read/write vs PyTorch's separate ops.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)

        mask = cols < N
        gate = tl.load(gate_ptr + row * stride_gate_row + cols * stride_gate_col,
                        mask=mask, other=0.0)
        up = tl.load(up_ptr + row * stride_up_row + cols * stride_up_col,
                      mask=mask, other=0.0)

        # silu(gate) = gate * sigmoid(gate)
        silu_gate = gate * tl.sigmoid(gate)

        # SwiGLU: silu(gate) * up
        out = silu_gate * up

        tl.store(out_ptr + row * stride_out_row + cols * stride_out_col,
                 out, mask=mask)


# ─── Public API ─────────────────────────────────────────────────────

_triton_rms_available = _TRITON_AVAILABLE and torch.cuda.is_available()
_triton_swiglu_available = _TRITON_AVAILABLE and torch.cuda.is_available()


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm via Triton. Falls back to F.rms_norm on CPU/no-Triton.

    Args:
        x: (..., D) input tensor. Normalized over last dim.
        weight: (D,) learnable scale.
        eps: small constant for numerical stability.

    Returns:
        (..., D) normalized tensor, same dtype as input.
    """
    if not _triton_rms_available or not x.is_cuda:
        return _pytorch_rms_norm(x, weight, eps)

    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    M = x_2d.shape[0]
    out = torch.empty_like(x_2d)

    # Block size: next power of 2 >= D, capped at 65536 (Triton limit).
    BLOCK = 1
    while BLOCK < D:
        BLOCK *= 2
    BLOCK = min(BLOCK, 65536)

    try:
        _rms_norm_kernel[(M,)](
            x_2d, weight, out,
            eps, D,
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(0),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK,
        )
        return out.reshape(orig_shape)
    except Exception:
        # Fallback on any Triton error (shape mismatch, OOM, etc.)
        return _pytorch_rms_norm(x, weight, eps)


def triton_swiglu_act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation via Triton: silu(gate) * up.

    Only fuses the elementwise activation — the w_gate/w_up/w_down linear
    projections stay as cuBLAS GEMMs (already optimal). This saves one
    intermediate tensor (the silu output) by fusing silu + multiply.

    Args:
        gate: (..., H) gate projection output.
        up: (..., H) up projection output.

    Returns:
        (..., H) SwiGLU activation: silu(gate) * up.
    """
    if not _triton_swiglu_available or not gate.is_cuda:
        return _pytorch_swiglu_act(gate, up)

    assert gate.shape == up.shape, f"gate {gate.shape} != up {up.shape}"
    orig_shape = gate.shape
    H = orig_shape[-1]
    gate_2d = gate.reshape(-1, H)
    up_2d = up.reshape(-1, H)
    M = gate_2d.shape[0]
    out = torch.empty_like(gate_2d)

    BLOCK = 1
    while BLOCK < H:
        BLOCK *= 2
    BLOCK = min(BLOCK, 65536)

    try:
        _swiglu_kernel[(M,)](
            gate_2d, up_2d, out,
            H,
            gate_2d.stride(0), gate_2d.stride(1),
            up_2d.stride(0), up_2d.stride(1),
            out.stride(0), out.stride(1),
            BLOCK=BLOCK,
        )
        return out.reshape(orig_shape)
    except Exception:
        return _pytorch_swiglu_act(gate, up)


def is_triton_available() -> bool:
    """Check if Triton kernels are available (CUDA + triton installed)."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()
