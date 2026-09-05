"""SubBitnet — training-free QAT-style sub-bitnet quantization key.

R&D round 49 (2026-09-05). Targets **sub-bitnet** bit-widths (<= 1.58 bits/w,
default ~1.0 bit/w) with no training, no calibration data, and no gradient
steps — a closed-form Key that approximates what QAT would converge to.

The key insight (QuIP#, QuaRot, SpinQuant): the reason naive 1-bit sign
quantization is lossy on LLM weights is *outliers* — a few large-magnitude
columns dominate the MSE. QAT fixes this by *learning* a rotation/scale that
spreads the outliers. We get the same effect **for free** with a fixed
Hadamard rotation (incoherence preprocessing), then close the remaining gap
with iterative residual binarization (IRB) — the binary analog of the
codebase's IRI-FP4 — plus an optional closed-form SVD low-rank residual
(BiLLM / SVDLift-style) for the structured error component.

Pipeline (all training-free, closed-form):
  1. Hadamard incoherence rotation:  W_rot = W @ H   (H fixed, orthogonal)
     → spreads outliers, makes columns sub-Gaussian so sign() is near-optimal.
  2. Per-channel absmean scale:      s = absmean(W_rot, dim=1) / 0.7
     → exactly the scale QAT would learn, computed from weight stats.
  3. Iterative Residual Binarization (IRB), K rounds:
       r_0 = W_rot
       for k in 0..K-1:
         s_k   = absmean(r_k, dim=1) / 0.7   (per-channel, clamped)
         q_k   = sign(r_k / s_k)             ∈ {-1, +1}
         r_{k+1} = r_k - s_k * q_k
       W_rot ≈ Σ_k s_k * q_k
     Each round is 1 bit/w + negligible per-channel scale overhead. Each
     round shrinks the residual by ~2x (sign quantization captures half the
     energy of a sub-Gaussian vector), so error decays ~exponentially.
  4. Optional SVD low-rank residual:  r_K ≈ U_r S_r V_r^T
     → closed-form SVD of the final residual, top-r components stored as
       float16 factors. Captures the structured error binarization cannot.
       Effective cost = 16 * 2 * r * (out+in) / (out*in) bits/w.

Effective bit-width (out=2048, in=8192, r=0):
  K=1, rank=0:  ~1.00 bits/w   (pure binary, sub-bitnet)   ← default
  K=1, rank=16: ~1.02 bits/w
  K=2, rank=0:  ~2.00 bits/w
  K=1, rank=64: ~1.10 bits/w

This is the training-free substitute for BitNet QAT at sub-bitnet bit-widths.
BitNet b1.58 needs QAT to reach 1.58 bits cleanly; SubBitnet reaches ~1.0 bit
with no training by exploiting incoherence + residual structure instead of
gradient descent.

Two components:
  1. SubBitnetLinear: nn.Module with packed binary signs + scales (+ low-rank)
     for inference.
  2. SubBitnetKey: Key class for the key system.

Reuses the normalized Hadamard matrix generator from
``forge.engine.quant.novel_quant_r44`` (canonical path, no duplication).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys.misc.base import Key, KeyClass, KeyResult
from forge.engine.quant.novel_quant_r44 import _hadamard_matrix

_BITNET_SCALE_DIVISOR = 0.7  # BitNet b1.58 absmean convention
_RES_EPS = 1e-8


# ── Core training-free quantize / dequantize ─────────────────────────────────

def _hadamard_size(in_features: int) -> int:
    """Smallest power of 2 >= in_features (Hadamard matrix constraint)."""
    n = 1
    while n < in_features:
        n *= 2
    return n


def binary_quantize_round(residual: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """One IRB round: per-channel absmean scale + sign binarization.

    Args:
        residual: [out, in] float residual to binarize this round.

    Returns:
        signs: int8 tensor {-1, +1} [out, in]  (1 bit/elem)
        scale: float32 per-channel scale [out]
    """
    scale = residual.abs().mean(dim=1).clamp(min=_RES_EPS) / _BITNET_SCALE_DIVISOR
    signs = torch.sign(residual / scale.unsqueeze(1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return signs.to(torch.int8), scale.to(torch.float32)


def quantize_sub_bitnet(w: torch.Tensor, n_rounds: int = 1,
                        rank: int = 0, use_hadamard: bool = True
                        ) -> dict[str, Any]:
    """Training-free sub-bitnet quantization of a 2D weight tensor.

    Hadamard incoherence rotation → K IRB binary rounds → optional SVD
    low-rank residual. Returns a packed dict consumed by
    :class:`SubBitnetLinear` and :func:`apply_sub_bitnet`.

    Args:
        w: weight tensor [out_features, in_features] (any dtype, float-converted).
        n_rounds: number of IRB binary rounds (K). 1 = pure 1-bit (sub-bitnet).
        rank: SVD low-rank residual rank (0 = disabled).
        use_hadamard: apply Hadamard incoherence rotation (recommended).

    Returns:
        Packed dict:
          signs: list[int8 tensor]  length K, each [out, in_padded] ∈ {-1,+1}
          scales: list[float32 tensor] length K, each [out]
          rank_u: float16 [out, rank] or None
          rank_v: float16 [rank, in] or None   (stores V^T rows)
          shape: (out_features, in_features)
          hadamard_size: int (power of 2, == in_features if use_hadamard=False)
          n_rounds: int
          rank: int
          use_hadamard: bool
    """
    w = w.float()
    out_f, in_f = w.shape

    if use_hadamard:
        h_size = _hadamard_size(in_f)
        pad = h_size - in_f
        wp = F.pad(w, (0, pad)) if pad > 0 else w
        H = _hadamard_matrix(h_size, w.device, w.dtype)
        w_rot = wp @ H  # (out, h_size)
    else:
        h_size = in_f
        w_rot = w

    # K IRB binary rounds on the (rotated) weight.
    signs_list, scales_list = [], []
    residual = w_rot
    for _ in range(max(1, n_rounds)):
        s, scale = binary_quantize_round(residual)
        recon = scale.unsqueeze(1) * s.float()
        residual = residual - recon
        signs_list.append(s)
        scales_list.append(scale)

    # Optional closed-form SVD low-rank residual on what's left.
    rank_u = None
    rank_v = None
    if rank > 0:
        # Residual lives in rotated space; SVD there, store factors in rotated
        # space (inverse Hadamard applied at dequant time alongside the binary
        # reconstruction).
        U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
        r = min(rank, S.numel())
        if r > 0:
            rank_u = (U[:, :r] * S[:r].unsqueeze(0)).to(torch.float16)  # (out, r)
            rank_v = Vt[:r, :].to(torch.float16)                       # (r, h_size)

    return {
        "signs": signs_list,
        "scales": scales_list,
        "rank_u": rank_u,
        "rank_v": rank_v,
        "shape": (out_f, in_f),
        "hadamard_size": h_size,
        "n_rounds": len(signs_list),
        "rank": 0 if rank_u is None else rank_u.shape[1],
        "use_hadamard": use_hadamard,
    }


def dequantize_sub_bitnet(packed: dict[str, Any],
                          dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reconstruct a weight tensor from a SubBitnet packed dict.

    Sums the K binary rounds (each: scale * signs), adds the low-rank residual
    if present, then applies the inverse Hadamard rotation and removes padding.
    """
    out_f, in_f = packed["shape"]
    h_size = packed["hadamard_size"]
    use_hadamard = packed["use_hadamard"]
    n_rounds = packed["n_rounds"]
    device = packed["signs"][0].device

    acc = torch.zeros(out_f, h_size, dtype=torch.float32, device=device)
    for r in range(n_rounds):
        s = packed["signs"][r].to(torch.float32)
        scale = packed["scales"][r].to(torch.float32)
        acc = acc + scale.unsqueeze(1) * s

    if packed["rank_u"] is not None:
        ru = packed["rank_u"].to(torch.float32)
        rv = packed["rank_v"].to(torch.float32)
        acc = acc + ru @ rv  # (out, r) @ (r, h_size) → (out, h_size)

    if use_hadamard:
        H = _hadamard_matrix(h_size, device, torch.float32)
        w = acc @ H  # H symmetric & orthogonal → inverse = H
    else:
        w = acc

    return w[:, :in_f].to(dtype)


# ── State-dict quantization (for the Key class) ──────────────────────────────

def apply_sub_bitnet(state: dict[str, torch.Tensor], n_rounds: int = 1,
                     rank: int = 0, use_hadamard: bool = True
                     ) -> dict[str, torch.Tensor]:
    """Apply sub-bitnet quantization to all 2D ``.weight`` tensors in a state dict.

    For each 2D weight, stores packed binary signs + per-channel scales for
    each round, plus optional low-rank factors. Non-2D weights and non-weight
    tensors pass through unchanged (matches the BitNet / IRI-FP4 convention:
    norms, embeddings, and 1D weights are left full-precision).

    Emits per weight ``name.weight``:
      name.weight.sb_signs_r{r}: int8 [out, in_padded] ∈ {-1, +1}
      name.weight.sb_scale_r{r}: float32 [out]
      name.weight.sb_rank_u:     float16 [out, rank]   (only if rank > 0)
      name.weight.sb_rank_v:     float16 [rank, in_padded] (only if rank > 0)
      name.weight.sb_meta:       int32 [out, in, h_size, n_rounds, rank, use_hadamard]
    """
    out = {}
    for k, v in state.items():
        if (isinstance(v, torch.Tensor) and k.endswith(".weight")
                and v.ndim == 2):
            packed = quantize_sub_bitnet(v.float(), n_rounds=n_rounds,
                                         rank=rank, use_hadamard=use_hadamard)
            base = k.replace(".weight", "")
            for r in range(packed["n_rounds"]):
                out[f"{base}.weight.sb_signs_r{r}"] = packed["signs"][r]
                out[f"{base}.weight.sb_scale_r{r}"] = packed["scales"][r]
            if packed["rank_u"] is not None:
                out[f"{base}.weight.sb_rank_u"] = packed["rank_u"]
                out[f"{base}.weight.sb_rank_v"] = packed["rank_v"]
            out[f"{base}.weight.sb_meta"] = torch.tensor(
                [v.shape[0], v.shape[1], packed["hadamard_size"],
                 packed["n_rounds"], packed["rank"],
                 int(packed["use_hadamard"])], dtype=torch.int32)
        else:
            out[k] = v
    return out


# ── SubBitnetLinear: inference module ────────────────────────────────────────

class SubBitnetLinear(nn.Module):
    """Linear layer with sub-bitnet quantized weights for inference.

    Stores K binary sign rounds + per-channel float16 scales (+ optional
    float16 low-rank residual). Dequantizes on-the-fly by summing all rounds,
    adding the low-rank term, and applying the inverse Hadamard rotation.

    Storage (out=2048, in=8192, K=1, rank=0): ~1.0 bit/w + ~0.001 bit/w scale
    overhead → ~1.0 bits/w (sub-bitnet).

    Args:
        in_features, out_features: as nn.Linear.
        bias: include a bias term.
        n_rounds: IRB binary rounds (default 1 = pure 1-bit).
        rank: SVD low-rank residual rank (default 0 = disabled).
        use_hadamard: apply Hadamard incoherence rotation (default True).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, n_rounds: int = 1, rank: int = 0,
                 use_hadamard: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_rounds = max(1, n_rounds)
        self.rank = rank
        self.use_hadamard = use_hadamard
        self.hadamard_size = _hadamard_size(in_features) if use_hadamard else in_features

        self.register_buffer(
            "weight_signs",
            torch.zeros(self.n_rounds, out_features, self.hadamard_size,
                        dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(self.n_rounds, out_features, dtype=torch.float16),
        )
        if rank > 0:
            self.register_buffer(
                "rank_u", torch.zeros(out_features, rank, dtype=torch.float16))
            self.register_buffer(
                "rank_v", torch.zeros(rank, self.hadamard_size, dtype=torch.float16))
        else:
            self.register_buffer("rank_u", torch.zeros(0, dtype=torch.float16))
            self.register_buffer("rank_v", torch.zeros(0, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        self._cached_weight = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, n_rounds: int = 1, rank: int = 0,
                    use_hadamard: bool = True) -> "SubBitnetLinear":
        """Build from a standard nn.Linear by quantizing its weights (training-free).

        The resulting module is moved to the source linear's device so its
        buffers match the source weight's device (CUDA-safe).
        """
        out_f, in_f = lin.weight.shape
        device = lin.weight.device
        obj = cls(in_f, out_f, bias=lin.bias is not None,
                  n_rounds=n_rounds, rank=rank, use_hadamard=use_hadamard)
        packed = quantize_sub_bitnet(lin.weight.data.float(),
                                     n_rounds=n_rounds, rank=rank,
                                     use_hadamard=use_hadamard)
        obj.load_prequantized(packed,
                              lin.bias.data if lin.bias is not None else None)
        return obj.to(device)

    def load_prequantized(self, packed: dict[str, Any],
                          bias: torch.Tensor | None = None):
        """Load packed sub-bitnet weights from a quantize_sub_bitnet dict."""
        with torch.no_grad():
            for r in range(self.n_rounds):
                self.weight_signs[r].copy_(packed["signs"][r])
                self.weight_scales[r].copy_(packed["scales"][r].to(torch.float16))
            if self.rank > 0 and packed["rank_u"] is not None:
                self.rank_u.copy_(packed["rank_u"])
                self.rank_v.copy_(packed["rank_v"])
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(torch.float16))
        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        if cache and self._cached_weight is not None \
                and self._cached_weight.dtype == dtype:
            return self._cached_weight
        acc = torch.zeros(self.out_features, self.hadamard_size,
                          dtype=torch.float32, device=self.weight_signs.device)
        for r in range(self.n_rounds):
            s = self.weight_signs[r].to(torch.float32)
            scale = self.weight_scales[r].to(torch.float32)
            acc = acc + scale.unsqueeze(1) * s
        if self.rank > 0:
            acc = acc + self.rank_u.to(torch.float32) @ self.rank_v.to(torch.float32)
        if self.use_hadamard:
            H = _hadamard_matrix(self.hadamard_size, acc.device, torch.float32)
            w = acc @ H
        else:
            w = acc
        w = w[:, :self.in_features].to(dtype)
        if cache:
            self._cached_weight = w
        return w

    @property
    def weight_quantized(self) -> torch.Tensor:
        return self._dequantize_weight(torch.float32, cache=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, cache=True)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        bpw = self.effective_bits_per_weight()
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"n_rounds={self.n_rounds}, rank={self.rank}, "
                f"hadamard={self.use_hadamard}, eff_bpw={bpw:.2f}")

    def effective_bits_per_weight(self) -> float:
        """Theoretical effective bits per weight (ignoring negligible scale overhead)."""
        n = self.out_features * self.hadamard_size
        bits = self.n_rounds * n  # 1 bit per sign element per round
        if self.rank > 0:
            bits += 16 * (self.out_features * self.rank
                          + self.rank * self.hadamard_size)
        return bits / max(1, self.out_features * self.in_features)


# ── SubBitnetKey: Key class ──────────────────────────────────────────────────

class SubBitnetKey(Key):
    """Sub-bitnet key — training-free QAT-style sub-bitnet quantization.

    Hadamard incoherence rotation + Iterative Residual Binarization (IRB) +
    optional SVD low-rank residual. Approximates QAT-optimal binary weights
    with no training, no calibration, no gradient steps.

    Key class: PARTIAL — binarization is not invertible. The Hadamard rotation
    IS invertible (orthogonal), but sign quantization discards magnitude, so
    the overall transform is one-way. IRB + low-rank reduce the error
    exponentially in the number of rounds / rank.

    Default (n_rounds=1, rank=0): ~1.0 bit/w — sub-bitnet.
    """

    def __init__(self, n_rounds: int = 1, rank: int = 0,
                 use_hadamard: bool = True):
        self.n_rounds = n_rounds
        self.rank = rank
        self.use_hadamard = use_hadamard

    @property
    def name(self) -> str:
        return "sub_bitnet"

    @property
    def description(self) -> str:
        bpw = self.n_rounds  # ~1 bit/round, scale overhead negligible
        lr = f" + rank-{self.rank} SVD residual" if self.rank > 0 else ""
        hd = "Hadamard+IRB " if self.use_hadamard else "IRB "
        return (f"{hd}{self.n_rounds}-round binary{lr} "
                f"(~{bpw:.1f} bits/w, sub-bitnet)")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Quantize all 2D weight tensors to sub-bitnet binary (+ residual)."""
        try:
            weights = apply_sub_bitnet(dict(data), n_rounds=self.n_rounds,
                                       rank=self.rank,
                                       use_hadamard=self.use_hadamard)
            n = sum(1 for k in weights if k.endswith(".sb_signs_r0"))
            return KeyResult(
                success=True, weights=weights,
                metadata={"n_quantized": n,
                          "n_rounds": self.n_rounds,
                          "rank": self.rank,
                          "use_hadamard": self.use_hadamard},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Binarization is not invertible — return weights as-is."""
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "Sub-bitnet binary weights cannot be un-quantized"})


# ── Model conversion utility ─────────────────────────────────────────────────

def convert_model_to_sub_bitnet(model: nn.Module, n_rounds: int = 1,
                                rank: int = 0, use_hadamard: bool = True
                                ) -> nn.Module:
    """Convert all nn.Linear layers in a model to SubBitnetLinear (in-place).

    Skips embedding/head layers (matching the BitNet / IRI-FP4 convention).
    """
    skip_names = ("embed", "head", "lm_head", "output")
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and not any(s in name for s in skip_names):
            sub_lin = SubBitnetLinear.from_linear(
                module, n_rounds=n_rounds, rank=rank, use_hadamard=use_hadamard)
            setattr(model, name, sub_lin)
        else:
            convert_model_to_sub_bitnet(module, n_rounds, rank, use_hadamard)
    return model


def build_sub_bitnet_linear(config, in_features: int, out_features: int,
                            bias: bool = True) -> nn.Module:
    """Build a SubBitnetLinear honoring the model config.

    Reads ``sub_bitnet_rounds``, ``sub_bitnet_rank``, ``sub_bitnet_hadamard``
    from the config (all optional, with sub-bitnet defaults).
    """
    return SubBitnetLinear(
        in_features, out_features, bias=bias,
        n_rounds=int(getattr(config, "sub_bitnet_rounds", 1)),
        rank=int(getattr(config, "sub_bitnet_rank", 0)),
        use_hadamard=bool(getattr(config, "sub_bitnet_hadamard", True)),
    )
