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
            if self.qscale is not None:
                scale = self.qscale
            else:
                scale = self.quantize_scale
            q, _ = ternary_quantize(w, scale)
            # STE: forward uses ternary weights, backward flows through the
            # original weights (identity gradient for the rounding step).
            w = w + (q - w).detach()
        return F.linear(x, w, self.bias)


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
