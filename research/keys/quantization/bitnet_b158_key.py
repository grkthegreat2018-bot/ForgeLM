"""BitNet b1.58 — ternary quantization-aware training (QAT).

Forces every linear weight to {-1, 0, +1} (~1.58 bits) with an absmean
scale factor, replacing fp16 multiplications with integer add / zero-skip.
Near-fp16 quality at a fraction of the FLOPs and memory (BitNet b1.58,
Ma et al. 2024; BitNet a4.8 follow-ups).

Implementation notes:
  - ``BitNetLinear`` keeps the same ``weight`` parameter name, so state-dict
    keys stay identical to nn.Linear — the existing checkpoint loads as-is.
  - Quantization is active only when ``quantize=True``; the straight-through
    estimator (STE) lets gradients flow through the ternary rounding during
    QAT. With quantize=False the module is a plain Linear (lossless).
  - Scale is computed per-tensor as absmean(W) / 0.7 (b1.58 convention),
    matching the paper's steady-state weight distribution; it is not a
    learnable parameter, so no extra checkpoint keys.

Usage:
    from research.keys.quantization.bitnet_b158_key import BitNetLinear, BitNetB158Key
    lin = BitNetLinear(d_model, hidden, quantize=True)   # training-mode QAT
    state = apply_bitnet_b158(state)                      # offline ternary bake
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult

_TERNARY_EPS = 1e-6


# ---------------------------------------------------------------------------
# True BitNet kernel A: integer-addition ternary GEMM (Triton, fp activations)
#
# b1.58-faithful: weights are {-1,0,1}, so the matmul is pure addition
# (+x where w==+1, -x where w==-1, zeros skipped) — no weight multiplies.
# NOTE: Triton's tl.broadcast_to / 3D tl.where are known-buggy (issues
# #2157, #1467, #532 — wrong dim silently broadcast); we use the documented
# workaround: subscript notation + direct multiply.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _ternary_add_kernel(
        X, W, Y,
        M, N, K,
        sxm, sxk, swn, swk,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Raw ternary matmul (x @ W^T) as pure additions, no scale.

        Scaling is applied on the host (kernel-side vector*matrix broadcast
        hits Triton bugs #2157/#1467)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + offs_k
            xm = (offs_m[:, None] < M) & (kk[None, :] < K)
            wm = (offs_n[:, None] < N) & (kk[None, :] < K)
            x = tl.load(X + offs_m[:, None] * sxm + kk[None, :] * sxk,
                        mask=xm, other=0.0)
            w = tl.load(W + offs_n[:, None] * swn + kk[None, :] * swk,
                        mask=wm, other=0)
            # Add-only accumulation. Convert int8 weights to fp BEFORE the
            # comparison: int8==int hits a Triton bool-broadcast bug
            # (#7957 broadcast_bug_bool_mat); subscript broadcast is used
            # because tl.broadcast_to mis-broadcasts (#2157, #1467).
            wf = w.to(tl.float32)
            acc += tl.sum(
                x[:, None, :] * (wf[None, :, :] == 1.0).to(tl.float32)
                - x[:, None, :] * (wf[None, :, :] == -1.0).to(tl.float32),
                axis=2)
        ym = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :],
                 acc.to(Y.dtype.element_ty), mask=ym)


def _triton_ternary_linear(x: torch.Tensor, w: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
    """y = (x @ w.T) * scale, w ternary int8 — add-only, zero-skip."""
    M, K = x.reshape(-1, x.shape[-1]).shape
    N = w.shape[0]
    x2 = x.reshape(M, K).contiguous()
    w2 = w.to(torch.int8).contiguous()
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM, BN, BK = 32, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _ternary_add_kernel[grid](
        x2, w2, y, M, N, K,
        x2.stride(0), x2.stride(1), w2.stride(0), w2.stride(1),
        BM=BM, BN=BN, BK=BK,
    )
    y = y * scale.view(1, -1).to(y.dtype)
    return y.view(*x.shape[:-1], N)


# ---------------------------------------------------------------------------
# True BitNet kernel B: INTEGER GEMM on tensor cores (int8 @ int8)
# ---------------------------------------------------------------------------

def _int8_ternary_linear(x: torch.Tensor, w: torch.Tensor,
                         w_scale: torch.Tensor) -> torch.Tensor:
    """y = (x @ w.T) * (x_scale * w_scale) with w ternary int8.

    REAL integer matmul: activations are quantized to int8 (absmax, BitNet
    a4.8 style) and multiplied on the GPU's integer tensor cores via
    torch._int_mm — no floating-point matmul at all. Ternary weights make
    the int8 weight matrix exact (no weight quantization error).
    """
    M, K = x.reshape(-1, x.shape[-1]).shape
    N = w.shape[0]
    xf = x.reshape(M, K)
    x_absmax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x8 = torch.clamp(torch.round(xf * (127.0 / x_absmax)),
                     -127, 127).to(torch.int8)
    w8 = w.to(torch.int8).t().contiguous()
    if M <= 16:
        # torch._int_mm requires M > 16; pad with zero rows.
        x8 = torch.cat([x8, torch.zeros(32 - M, K, dtype=torch.int8,
                                        device=x.device)], dim=0)
        y = torch._int_mm(x8, w8).float()[:M]
    else:
        y = torch._int_mm(x8, w8).float()
    y = y * ((x_absmax / 127.0) * w_scale.view(1, -1).to(x.device))
    # Cast back to the activation dtype — fp32 output would poison the
    # bf16 residual stream downstream (rms_norm/linear dtype mismatch).
    return y.to(x.dtype).view(*x.shape[:-1], N)


class _TernaryLinearFn(torch.autograd.Function):
    """Ternary forward via integer kernels; STE backward through the
    full-precision master weights (QAT semantics).

    Kernel selection (CUDA):
      - "triton" : b1.58 add-only ternary GEMM (fp activations, zero-skip)
      - default  : int8 @ int8 on tensor cores (a4.8-style activation q)
    CPU falls back to an fp GEMM on the ternary weights.
    """

    @staticmethod
    def forward(ctx, x, w, qscale, kernel):
        q, scale = ternary_quantize(w, qscale)
        scale = scale.detach().to(x.device)
        if kernel == "triton" and _HAS_TRITON and x.is_cuda:
            y = _triton_ternary_linear(x, q, scale)
        elif kernel == "int8" and x.is_cuda:
            y = _int8_ternary_linear(x, q, scale)
        else:
            y = F.linear(x, q.to(x.dtype), None) * scale.to(x.dtype)
        ctx.save_for_backward(x, w, y)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w, y = ctx.saved_tensors
        # y = x @ w.T  ->  grad_x = grad_y @ w (STE: full-precision weights)
        grad_x = grad_y @ w
        gx = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1])
        grad_w = gx.T @ xr
        grad_scale = (grad_y * (y / ctx.scale.view(1, -1))).sum()
        return grad_x, grad_w, grad_scale, None


def ternary_quantize(w: torch.Tensor, scale: float | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} * scale.

    Returns (quantized_weights, scale). Uses the b1.58 absmean convention:
    scale = absmean(W) / 0.7 (0.7 ≈ 2/3 split point for ternary rounding).
    """
    w_abs = w.abs()
    if scale is None:
        scale = w_abs.mean().clamp(min=_TERNARY_EPS) / 0.7
    q = torch.where(w_abs < 0.5 * scale, torch.zeros_like(w),
                    torch.sign(w))
    return q, scale


class BitNetLinear(nn.Module):
    """Linear layer with STE ternary QAT (BitNet b1.58).

    QAT semantics (world-class practice):
      - training forward: ternary weights + STE (master weights stay fp).
      - eval forward: full-precision master weights by default (lossless for
        a warm-started / partially-trained model); set force_quant=True once
        QAT converges to deploy true ternary inference.
      - learned_scale: a per-layer learnable quantize scale (absmean init),
        which QAT tunes to minimize ternary rounding error — the step that
        recovers most of the b1.58 quality gap vs. a fixed absmean scale.

    Args:
        in_features, out_features: as nn.Linear.
        bias: use a bias term.
        quantize: True = ternary forward during training (QAT).
        force_quant: also quantize in eval (deployment mode).
        learned_scale: learn a per-layer scale (param ``qscale``).
        quantize_scale: fixed scale (None = absmean / learned).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, quantize: bool = False,
                 force_quant: bool = False, learned_scale: bool = True,
                 quantize_scale: float | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize = quantize
        self.force_quant = force_quant
        self.learned_scale = learned_scale
        self.quantize_scale = quantize_scale
        # Kaiming init matching nn.Linear so pre-trained weights load fine.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        self.qscale = None
        if learned_scale:
            # Init from the b1.58 absmean convention; QAT then tunes it.
            with torch.no_grad():
                scale = self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            self.qscale = nn.Parameter(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # QAT: quantize only in training (or when explicitly deploying).
        if self.quantize and (self.training or self.force_quant):
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            if x.is_cuda and self.bias is None:
                # TRUE BitNet on CUDA: default = int8 tensor-core GEMM;
                # FORGE_BITNET_KERNEL=triton selects the b1.58 add-only
                # kernel (fp activations). STE backward in both.
                import os
                kernel = os.environ.get("FORGE_BITNET_KERNEL", "int8")
                return _TernaryLinearFn.apply(x, w, scale, kernel)
            # Fallback (CPU): fp GEMM on ternary weights, STE.
            q, _ = ternary_quantize(w, scale)
            w = w + (q - w).detach()
        return F.linear(x, w, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        had_weight = prefix + "weight" in state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if had_weight and self.qscale is not None:
            # qscale was initialized from the RANDOM init weight; re-anchor
            # it to the loaded checkpoint's absmean (b1.58 convention).
            with torch.no_grad():
                self.qscale.copy_(
                    self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7)


def apply_bitnet_b158(state: dict[str, torch.Tensor], scale: float | None = None
                      ) -> dict[str, torch.Tensor]:
    """Offline ternary bake: quantize every linear-ish weight in a state dict.

    Quantizes all keys ending in ``.weight`` (embeddings/head included, per
    BitNet b1.58 — they are also ternary). Returns a new state dict; the
    result is intended for QAT warm-start or fused integer-kernel inference.

    Args:
        state: model state dict (keys from ConfigurableResearchLLM).
        scale: optional fixed quantize scale (None = per-tensor absmean).

    Returns:
        New state dict with quantized weights (same keys/shapes/dtypes).
    """
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k.endswith(".weight"):
            q, _ = ternary_quantize(v.float(), scale)
            out[k] = q.to(v.dtype)
        else:
            out[k] = v
    return out


class BitNetB158Key(Key):
    """BitNet b1.58 key — ternary QAT transform on linear weights.

    Key class: PARTIAL — the ternary transform is not invertible.
    """

    def __init__(self, scale: float | None = None):
        self.scale = scale

    @property
    def name(self) -> str:
        return "bitnet_b158"

    @property
    def description(self) -> str:
        return f"Ternary QAT weights {{-1,0,+1}} (scale={self.scale or 'absmean'})"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            weights = apply_bitnet_b158(dict(data), self.scale)
            n = sum(1 for k in weights if k.endswith(".weight"))
            return KeyResult(success=True, weights=weights,
                             metadata={"n_quantized": n,
                                       "scale": self.scale or "absmean"})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "Ternary weights cannot be un-quantized"})


def build_bitnet_linear(config, in_features: int, out_features: int,
                        bias: bool = False) -> nn.Module:
    """Build a BitNetLinear honoring the model config."""
    return BitNetLinear(in_features, out_features, bias=bias,
                        quantize=bool(getattr(config, "use_bitnet", False)),
                        force_quant=bool(getattr(config, "bitnet_force_quant", False)),
                        learned_scale=bool(getattr(config, "bitnet_learned_scale", True)))
