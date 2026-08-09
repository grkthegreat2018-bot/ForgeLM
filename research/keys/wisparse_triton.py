"""Triton kernel for WiSparse — fuses threshold + mask into one GPU kernel.

Eliminates Python dispatch overhead and GPU→CPU sync from the wisparse hot path.

The fused operation:
  gate_sparse[n, i] = gate[n, i] if |gate[n, i]| * weight_norms[i] > threshold else 0

This replaces 5 separate PyTorch ops (abs, mul, compare, to, mul) + 1 sync (.item())
with a single kernel launch.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    triton = None
    tl = None


@triton.jit
def _wisparse_fused_kernel(
    gate_ptr,          # (N, d_ff) — gate activations
    weight_norms_ptr,  # (d_ff,) — precomputed weight importance
    out_ptr,           # (N, d_ff) — sparse gate output
    N: tl.constexpr,   # number of rows
    D: tl.constexpr,   # d_ff (hidden dim)
    threshold,         # scalar — sparsity threshold
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused kernel: threshold check + masking in one pass. No GPU→CPU sync."""
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < D

    gate_ptrs = gate_ptr + offs_n[:, None] * D + offs_d[None, :]
    wnorm_ptrs = weight_norms_ptr + offs_d

    gate = tl.load(gate_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    wnorm = tl.load(wnorm_ptrs, mask=mask_d, other=0.0)

    # Fused: score = |gate| * weight_norm, keep if score > threshold
    abs_gate = tl.abs(gate)
    score = abs_gate * wnorm[None, :]
    keep = score > threshold

    out = tl.where(keep, gate, 0.0)

    out_ptrs = out_ptr + offs_n[:, None] * D + offs_d[None, :]
    tl.store(out_ptrs, out, mask=mask_n[:, None] & mask_d[None, :])


def wisparse_fused(
    gate: torch.Tensor,
    weight_norms: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Apply wisparse threshold + masking via fused Triton kernel.

    Falls back to PyTorch ops if Triton is not installed.

    Args:
        gate: (N, d_ff) gate activations
        weight_norms: (d_ff,) precomputed weight importance per neuron
        threshold: scalar sparsity threshold

    Returns:
        sparse_gate: (N, d_ff) — gate with low-contribution neurons zeroed
    """
    if not _HAS_TRITON:
        # PyTorch fallback: same math, more kernel launches but no triton dep.
        score = gate.abs() * weight_norms.unsqueeze(0)
        return torch.where(score > threshold, gate, torch.zeros_like(gate))

    N, D = gate.shape
    out = torch.empty_like(gate)

    BLOCK_N = 16
    BLOCK_D = 128

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))

    _wisparse_fused_kernel[grid](
        gate, weight_norms, out,
        N, D, threshold,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return out
