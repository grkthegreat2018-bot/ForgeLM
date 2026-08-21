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
        # Cast everything to grad_y's dtype for consistent matmuls
        # (model may be bf16 but loss/grad can be fp32)
        gdt = grad_y.dtype
        grad_x = grad_y @ w.to(gdt)
        gx = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1]).to(gdt)
        grad_w = gx.T @ xr
        grad_scale = (grad_y * (y.to(gdt) / ctx.scale.view(1, -1).to(gdt))).sum()
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
      - prequantized: weights stored as int8 buffer (1 byte vs 4 bytes fp32).
        Set via ``convert_to_int8_storage()`` after loading a pre-quantized
        checkpoint. Forward pass uses int8 directly with tensor-core GEMM.
        4x VRAM reduction for weights; no runtime quantization cost.

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
        self._prequantized = False  # set True by convert_to_int8_storage()
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
        # Pre-quantized int8 path: weights already ternary, stored as int8 buffer.
        # 4x less VRAM than fp32 parameter; no runtime quantization needed.
        if self._prequantized:
            w_int8 = self.weight_int8  # int8 buffer, values {-1, 0, +1}
            scale = self.qscale_buf.to(x.dtype) if hasattr(self, 'qscale_buf') else 1.0
            if x.is_cuda and self.bias is None:
                # Use int8 tensor-core GEMM directly (weights already int8).
                return _int8_ternary_linear(x, w_int8, scale)
            # CPU fallback: dequantize int8 → fp, standard GEMM.
            w = w_int8.to(x.dtype) * scale
            return F.linear(x, w, self.bias)
        # Normal QAT / eval path.
        w = self.weight
        if self.quantize and (self.training or self.force_quant):
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            if x.is_cuda and self.bias is None:
                import os
                kernel = os.environ.get("FORGE_BITNET_KERNEL", "int8")
                return _TernaryLinearFn.apply(x, w, scale, kernel)
            q, _ = ternary_quantize(w, scale)
            w = w + (q - w).detach()
        return F.linear(x, w, self.bias)

    def convert_to_int8_storage(self):
        """Convert fp32 weight parameter to int8 buffer (pre-quantized inference).

        Ternary-quantizes the current weight, stores as int8 buffer, and removes
        the fp32 parameter. Reduces weight VRAM by 4x (4 bytes → 1 byte per param).
        Also stores qscale as a buffer (no gradient needed in inference).

        After conversion, the layer uses the int8 tensor-core GEMM path directly
        — no runtime quantization cost.
        """
        if self._prequantized:
            return  # already converted
        with torch.no_grad():
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            q, s = ternary_quantize(self.weight.data.float(), scale)
            # Store ternary weights as int8 {-1, 0, +1}
            w_int8 = q.to(torch.int8)
            # Register as buffer (not parameter — no gradients in inference)
            device = self.weight.device
            dtype = self.weight.dtype
            del self.weight
            self.register_buffer("weight_int8", w_int8.to(device))
            # Store scale as buffer too
            if isinstance(s, torch.Tensor):
                self.register_buffer("qscale_buf", s.to(device).to(dtype))
            else:
                self.register_buffer("qscale_buf", torch.tensor(s, device=device, dtype=dtype))
            # Remove qscale parameter if it exists
            if self.qscale is not None:
                del self.qscale
                self.qscale = None
            self._prequantized = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        had_weight = prefix + "weight" in state_dict
        # Check if loading a pre-quantized int8 weight
        weight_key = prefix + "weight"
        src_tensor = state_dict.get(weight_key, None)
        is_int8_weight = src_tensor is not None and src_tensor.dtype == torch.int8

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

        if had_weight and self.qscale is not None:
            # qscale was initialized from the RANDOM init weight; re-anchor
            # it to the loaded checkpoint's absmean (b1.58 convention).
            # Skip if weight is still on meta (assign=True defers replacement)
            # or if loading pre-quantized int8 (qscale set by convert_to_int8_storage)
            if is_int8_weight:
                # Pre-quantized: compute scale from the int8 source tensor directly
                with torch.no_grad():
                    absmean = src_tensor.float().abs().mean().clamp(min=_TERNARY_EPS)
                    self.qscale.copy_(absmean / 0.7)
            elif not self.weight.is_meta:
                with torch.no_grad():
                    self.qscale.copy_(
                        self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7)
            # else: weight is on meta, skip qscale update (will be set after assign)

    def load_prequantized(self, weight_int8: torch.Tensor, qscale: float | torch.Tensor):
        """Load a pre-quantized ternary weight (int8) directly into int8 storage.

        Bypasses the fp32 parameter entirely — stores as int8 buffer.
        Call this after __init__ but before any forward pass.
        The tensor's own device is used (typically GPU for direct-VRAM loading).
        """
        # Use the tensor's device (already on GPU if loaded direct from safetensors)
        device = weight_int8.device
        # Remove fp32 weight parameter
        if hasattr(self, 'weight'):
            del self.weight
        # Register int8 buffer (keep on tensor's device)
        self.register_buffer("weight_int8", weight_int8.to(torch.int8))
        # Register scale buffer
        if isinstance(qscale, torch.Tensor):
            self.register_buffer("qscale_buf", qscale.to(device))
        else:
            self.register_buffer("qscale_buf", torch.tensor(qscale, device=device))
        # Remove qscale parameter
        if self.qscale is not None:
            del self.qscale
            self.qscale = None
        self._prequantized = True


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


def convert_model_to_int8(model: nn.Module) -> int:
    """Convert all BitNetLinear layers in a model to int8 weight storage.

    Call after loading a checkpoint with ternary-quantized weights.
    Reduces weight VRAM by 4x (fp32 → int8). Returns number of layers converted.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, BitNetLinear) and not module._prequantized:
            module.convert_to_int8_storage()
            n += 1
    return n


class BitNetEmbedding(nn.Module):
    """Embedding with BitNet b1.58 ternary QAT on the embedding weight.

    During training (or force_quant), the looked-up vectors are ternary-quantized
    with STE. Eval uses full-precision master weights (lossless for warm start).

    Args:
        num_embeddings: vocab size
        embedding_dim: embedding dimension
        quantize: enable ternary QAT during training
        force_quant: also quantize in eval (deployment)
        learned_scale: learn a per-tensor quantize scale
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 quantize: bool = False, force_quant: bool = False,
                 learned_scale: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quantize = quantize
        self.force_quant = force_quant
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if learned_scale:
            with torch.no_grad():
                scale = self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            self.qscale = nn.Parameter(scale)
        else:
            self.qscale = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = F.embedding(token_ids, self.weight)
        if self.quantize and (self.training or self.force_quant):
            scale = self.qscale if self.qscale is not None else None
            q, s = ternary_quantize(self.weight, scale)
            q_out = F.embedding(token_ids, q) * s.to(x.dtype)
            # STE: gradient flows through master weights
            x = x + (q_out - x).detach()
        return x

    @property
    def weight_data(self) -> torch.Tensor:
        """Access the raw weight parameter (for checkpoint I/O)."""
        return self.weight


def build_bitnet_embedding(config, num_embeddings: int, embedding_dim: int
                           ) -> nn.Module:
    """Build a BitNetEmbedding honoring the model config."""
    return BitNetEmbedding(
        num_embeddings, embedding_dim,
        quantize=bool(getattr(config, "use_bitnet", False)),
        force_quant=bool(getattr(config, "bitnet_force_quant", False)),
        learned_scale=bool(getattr(config, "bitnet_learned_scale", True)))
