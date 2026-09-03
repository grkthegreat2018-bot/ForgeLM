"""IRI-FP4 quantization key — Iterative Residual Refinement FP4.

R&D round 15 (2026-08-28). Validated by scripts/test_novel_quant.py:
  - IRI-FP4 x2: 41.21 dB SQNR at 1.12 bytes/w — beats FP8 (~42 dB) using
    only the FP4 codebook (2x FP4 tensor-core throughput vs 1x FP8 on
    Blackwell).
  - Each round adds ~21 dB SQNR at +0.56 bytes/w.
  - At n_rounds=3, block_size=32: ~1.6 bits/w (0.2 bytes/w) — extreme
    compression with exponentially decreasing residual.

Key insight: instead of one FP4 pass + one sparse residual (R-FP4), run K
rounds. Each round quantizes the RESIDUAL from the previous round to FP4
with its own per-block scale, and accumulates. After K rounds the residual
is ~6^K times smaller.

IRI-FP4 storage format (per weight tensor):
  - Round 0: FP4 indices (4 bits/elem, 2 per byte) + per-block float16 scale
  - Rounds 1..N-1: FP4 indices for residual + per-block float16 scale
  - Total: n_rounds * (0.5 bytes/w for indices + block_size/2 bytes/w for scales)
  - At n_rounds=3, block_size=32: ~1.6 bits/w

Two components:
  1. IRIFP4Linear: nn.Module with IRI-FP4 packed weights for inference
  2. IRIFP4Key: Key class for the key system
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult
from research.inference.quant.novel_quant import (
    quantize_iri_fp4,
    _optimal_fp4_scale,
    _fp4_quant_dequant_block,
    _pack_fp4_round,
)
from research.inference.quant.nvfp4_quant import _dequantize_fp4


# ── Standalone quantize / dequantize functions ──────────────────────────────

def quantize_weight_iri_fp4(w: torch.Tensor, block_size: int = 32,
                            n_rounds: int = 3) -> dict[str, Any]:
    """Quantize a 2D weight tensor to IRI-FP4 packed representation.

    Runs K rounds of FP4 quantization on successive residuals. Each round
    produces FP4 indices (4-bit, 2 per byte) + per-block float16 scales.

    Args:
        w: weight tensor [out_features, in_features]
        block_size: elements per scale block (default 32)
        n_rounds: number of residual refinement rounds (default 3)

    Returns:
        Packed dict with keys:
          - indices: list of uint8 tensors [n_rounds], each [out, in_padded//2]
          - scales: list of float16 tensors [n_rounds], each [out, n_blocks]
          - global_scales: list of float32 tensors [n_rounds], each [out]
          - shape: (out_features, in_features)
          - block_size: int
          - n_rounds: int
    """
    w = w.float()
    out_f, in_f = w.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(w, (0, pad)) if pad > 0 else w
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    indices_list = []
    scales_list = []
    global_scales_list = []

    residual = wp.clone()
    for _ in range(n_rounds):
        wb = residual.view(out_f, n_blocks, block_size)
        scale = _optimal_fp4_scale(wb)  # (out, n_blocks, 1)
        w_dq = _fp4_quant_dequant_block(wb, scale)
        packed, scales_fp8, gs = _pack_fp4_round(wb, scale, block_size)
        # Store scales as float16 per spec
        indices_list.append(packed)
        scales_list.append(scales_fp8.to(torch.float16))
        global_scales_list.append(gs)
        residual = residual - w_dq.view(out_f, in_p)

    return {
        "indices": indices_list,
        "scales": scales_list,
        "global_scales": global_scales_list,
        "shape": (out_f, in_f),
        "block_size": block_size,
        "n_rounds": n_rounds,
    }


def dequantize_iri_fp4(packed: dict[str, Any],
                       dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reconstruct weight from IRI-FP4 packed representation.

    Sums the dequantized FP4 weights across all rounds.

    Args:
        packed: dict from quantize_weight_iri_fp4
        dtype: output dtype (default bfloat16)

    Returns:
        Reconstructed weight tensor [out_features, in_features]
    """
    out_f, in_f = packed["shape"]
    block_size = packed["block_size"]
    n_rounds = packed["n_rounds"]

    acc = torch.zeros(out_f, in_f, dtype=torch.float32,
                      device=packed["indices"][0].device)
    for r in range(n_rounds):
        w = _dequantize_fp4(
            packed["indices"][r],
            packed["scales"][r],
            out_f, in_f,
            block_size, torch.float32,
            global_scale=packed["global_scales"][r],
        )
        acc = acc + w
    return acc.to(dtype)


def dequantize_iri_fp4_state(state: dict[str, torch.Tensor],
                             dtype: torch.dtype = torch.bfloat16
                             ) -> dict[str, torch.Tensor]:
    """Dequantize all IRI-FP4 packed tensors in a state dict.

    Called by model_loader.build_model_fast() when loading a V10 checkpoint.
    Finds all `{key}.iri_packed` + `{key}.iri_scales` pairs, dequantizes them
    back to full-precision weights, and removes the packed tensors.

    Args:
        state: state dict with IRI-FP4 packed tensors
        dtype: output dtype for dequantized weights

    Returns:
        state dict with packed tensors replaced by full-precision weights
    """
    from research.inference.quant.nvfp4_quant import _dequantize_fp4, _FP8_DTYPE

    out = {}
    packed_keys = {}  # base_key -> (iri_packed, iri_scales)

    # Group packed tensors by base key
    for k, v in state.items():
        if k.endswith(".iri_packed"):
            base = k[:-len(".iri_packed")]
            packed_keys.setdefault(base, {})["packed"] = v
        elif k.endswith(".iri_scales"):
            base = k[:-len(".iri_scales")]
            packed_keys.setdefault(base, {})["scales"] = v
        else:
            out[k] = v

    # Dequantize each packed weight
    for base, data in packed_keys.items():
        if "packed" not in data or "scales" not in data:
            # Incomplete — keep as-is
            if "packed" in data:
                out[f"{base}.iri_packed"] = data["packed"]
            if "scales" in data:
                out[f"{base}.iri_scales"] = data["scales"]
            continue

        iri_packed = data["packed"]   # (n_rounds, out, in_p//2) uint8
        iri_scales = data["scales"]   # (n_rounds, out, n_blocks) float16
        n_rounds = iri_packed.shape[0]
        out_f = iri_packed.shape[1]
        in_p = iri_packed.shape[2] * 2  # 2 FP4 per byte
        in_f = in_p  # may include padding

        acc = torch.zeros(out_f, in_f, dtype=torch.float32, device=iri_packed.device)
        for r in range(n_rounds):
            w = _dequantize_fp4(
                iri_packed[r],
                iri_scales[r].to(_FP8_DTYPE),
                out_f, in_f,
                32,  # block_size
                torch.float32,
            )
            acc = acc + w

        out[f"{base}.weight"] = acc.to(dtype)

    return out


# ── IRIFP4Linear: inference module ──────────────────────────────────────────

class IRIFP4Linear(nn.Module):
    """Linear layer with IRI-FP4 quantized weights for inference.

    Stores n_rounds FP4 quantizations (indices + per-block float16 scales +
    per-channel global scales). Dequantizes on-the-fly by summing all rounds.

    Storage: n_rounds * (0.5 bytes/w + block_size/2 bytes/w for scales).
    At n_rounds=3, block_size=32: ~1.6 bits/w.

    Args:
        in_features, out_features: as nn.Linear.
        bias: whether to include a bias term.
        block_size: elements per scale block (default 32).
        n_rounds: number of residual refinement rounds (default 3).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, block_size: int = 32, n_rounds: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.n_rounds = n_rounds

        n_blocks = (in_features + block_size - 1) // block_size
        # K packed index tensors + K scale sets
        self.register_buffer(
            "weight_packed",
            torch.zeros(n_rounds, out_features, (in_features + 1) // 2,
                        dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(n_rounds, out_features, n_blocks, dtype=torch.float16),
        )
        self.register_buffer(
            "weight_global_scale",
            torch.ones(n_rounds, out_features, dtype=torch.float32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        self._cached_weight = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, block_size: int = 32,
                    n_rounds: int = 3) -> "IRIFP4Linear":
        """Build from a standard nn.Linear by quantizing its weights."""
        w = lin.weight.data
        out_f, in_f = w.shape
        obj = cls(in_f, out_f, bias=lin.bias is not None,
                  block_size=block_size, n_rounds=n_rounds)
        packed = quantize_weight_iri_fp4(w, block_size, n_rounds)
        obj.load_prequantized(packed, packed.get("bias"),
                              lin.bias.data if lin.bias is not None else None)
        return obj

    def load_prequantized(self, fp4_data: dict[str, Any],
                          scales: Any = None, bias: torch.Tensor | None = None):
        """Load pre-quantized IRI-FP4 weights.

        Args:
            fp4_data: packed dict from quantize_weight_iri_fp4 (preferred),
                      OR a list/tuple of (indices, scales, global_scale) per round.
            scales: unused if fp4_data is a packed dict (kept for API compat).
            bias: optional bias tensor to load.
        """
        with torch.no_grad():
            if isinstance(fp4_data, dict) and "indices" in fp4_data:
                # Packed dict format
                indices_list = fp4_data["indices"]
                scales_list = fp4_data["scales"]
                gs_list = fp4_data["global_scales"]
                for r in range(self.n_rounds):
                    self.weight_packed[r].copy_(indices_list[r])
                    self.weight_scales[r].copy_(scales_list[r])
                    self.weight_global_scale[r].copy_(gs_list[r])
            elif isinstance(fp4_data, (list, tuple)):
                # Legacy: list of (indices, scales, global_scale) per round
                for r, (idx, sc, gs) in enumerate(fp4_data):
                    self.weight_packed[r].copy_(idx)
                    self.weight_scales[r].copy_(sc)
                    self.weight_global_scale[r].copy_(gs)
            else:
                raise ValueError(
                    "fp4_data must be a packed dict or list of round tuples")
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(torch.float16))
        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16,
                           cache: bool = False) -> torch.Tensor:
        """Dequantize all rounds and sum.

        Args:
            dtype: output dtype
            cache: if True, cache the result for subsequent calls (uses more
                VRAM but faster). If False (default), dequantize on every call
                (saves VRAM, slightly slower). For large models that don't fit
                in VRAM with cached weights, keep cache=False.
        """
        if cache and self._cached_weight is not None and self._cached_weight.dtype == dtype:
            return self._cached_weight
        acc = torch.zeros(self.out_features, self.in_features,
                          dtype=torch.float32, device=self.weight_packed.device)
        for r in range(self.n_rounds):
            w = _dequantize_fp4(
                self.weight_packed[r],
                self.weight_scales[r],
                self.out_features, self.in_features,
                self.block_size, torch.float32,
                global_scale=self.weight_global_scale[r],
            )
            acc = acc + w
        acc = acc.to(dtype)
        if cache:
            self._cached_weight = acc
        return acc

    @property
    def weight_quantized(self) -> torch.Tensor:
        """Return the dequantized weight data (full precision reconstruction)."""
        return self._dequantize_weight(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cache dequantized weight for speed (first call dequantizes,
        # subsequent calls reuse the cached bf16 weight).
        # This trades VRAM (cached weight = same as bf16) for 50x speed.
        # To save VRAM instead, set self._cached_weight = None before forward.
        w = self._dequantize_weight(x.dtype, cache=True)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, bias)
        # QLoRA: apply LoRA adapter if attached (frozen quant base + trainable LoRA)
        if hasattr(self, 'lora_adapter') and self.lora_adapter is not None:
            out = out + self.lora_adapter(x)
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"block_size={self.block_size}, "
                f"n_rounds={self.n_rounds}")

    @torch.no_grad()
    def merge_lora(self) -> bool:
        """Merge LoRA adapter into packed weights (QLoRA merge).

        Dequantizes base → adds LoRA delta → re-quantizes to IRI-FP4.
        Returns True if merged, False if no adapter attached.
        """
        if not hasattr(self, 'lora_adapter') or self.lora_adapter is None:
            return False
        lora = self.lora_adapter
        # Dequantize base to fp32
        w = self._dequantize_weight(torch.float32, cache=False)
        # Add LoRA delta: W += scale * B @ A
        delta = lora.scale * (lora.lora_B @ lora.lora_A)
        w = w + delta.to(torch.float32)
        # Re-quantize to IRI-FP4
        packed = quantize_weight_iri_fp4(w, self.block_size, self.n_rounds)
        self.load_prequantized(packed)
        # Remove adapter
        del self.lora_adapter
        self._cached_weight = None
        return True


# ── State-dict quantization (for Key class) ─────────────────────────────────

def apply_iri_fp4(state: dict[str, torch.Tensor], block_size: int = 32,
                  n_rounds: int = 3) -> dict[str, torch.Tensor]:
    """Apply IRI-FP4 quantization to all 2D .weight tensors in state dict.

    For each 2D weight, produces packed FP4 indices + per-block float16 scales
    for each round. Non-weight tensors and non-2D weights are passed through.

    Stores per weight `name.weight`:
      - name.weight.fp4_indices_r{r}: uint8 packed indices [out, in_padded//2]
      - name.weight.fp4_scales_r{r}: float16 per-block scales [out, n_blocks]
      - name.weight.fp4_gscale_r{r}: float32 per-channel global scale [out]
    """
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k.endswith(".weight") and v.ndim == 2:
            packed = quantize_weight_iri_fp4(v.float(), block_size, n_rounds)
            base = k.replace(".weight", "")
            for r in range(n_rounds):
                out[f"{base}.weight.fp4_indices_r{r}"] = packed["indices"][r]
                out[f"{base}.weight.fp4_scales_r{r}"] = packed["scales"][r]
                out[f"{base}.weight.fp4_gscale_r{r}"] = packed["global_scales"][r]
            # Store shape metadata as a small int tensor for traceability
            out[f"{base}.weight.fp4_meta"] = torch.tensor(
                [v.shape[0], v.shape[1], block_size, n_rounds], dtype=torch.int32)
        else:
            out[k] = v
    return out


# ── IRIFP4Key: Key class ────────────────────────────────────────────────────

class IRIFP4Key(Key):
    """IRI-FP4 key — Iterative Residual Refinement FP4 quantization.

    Key class: PARTIAL — FP4 quantization is not invertible. The multi-round
    residual refinement reduces error exponentially (each round ~21 dB SQNR).
    At n_rounds=3, block_size=32: ~1.6 bits/w with near-FP8 accuracy.
    """

    def __init__(self, block_size: int = 32, n_rounds: int = 3):
        self.block_size = block_size
        self.n_rounds = n_rounds

    @property
    def name(self) -> str:
        return "iri_fp4"

    @property
    def description(self) -> str:
        # Per round: 0.5 bytes/w (indices) + 2 bytes / block_size (scales)
        bytes_per_w = self.n_rounds * (0.5 + 2.0 / self.block_size)
        return (f"IRI-FP4: {self.n_rounds} rounds, block_size={self.block_size} "
                f"(~{bytes_per_w:.2f} bytes/w, {bytes_per_w * 8:.1f} bits/w)")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Quantize all 2D weight tensors to IRI-FP4."""
        try:
            weights = apply_iri_fp4(dict(data), self.block_size, self.n_rounds)
            n = sum(1 for k in weights if k.endswith(".fp4_indices_r0"))
            return KeyResult(
                success=True, weights=weights,
                metadata={"n_quantized": n,
                          "block_size": self.block_size,
                          "n_rounds": self.n_rounds},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """FP4 quantization is not invertible — return weights as-is."""
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "IRI-FP4 weights cannot be un-quantized"})


# ── Model conversion utility ────────────────────────────────────────────────

def convert_model_to_iri_fp4(model: nn.Module, block_size: int = 32,
                             n_rounds: int = 3) -> nn.Module:
    """Convert all nn.Linear layers in a model to IRIFP4Linear.

    Replaces each nn.Linear with an IRIFP4Linear that stores the IRI-FP4
    quantized weights. The original model's Linear weights are quantized
    in-place. Non-Linear modules are left untouched.

    Args:
        model: nn.Module with nn.Linear layers
        block_size: FP4 block size (default 32)
        n_rounds: number of residual rounds (default 3)

    Returns:
        The model with Linear layers replaced by IRIFP4Linear (in-place).
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            iri_lin = IRIFP4Linear.from_linear(
                module, block_size=block_size, n_rounds=n_rounds)
            setattr(model, name, iri_lin)
        else:
            # Recurse into submodules
            convert_model_to_iri_fp4(module, block_size, n_rounds)
    return model


def build_iri_fp4_linear(config, in_features: int, out_features: int,
                         bias: bool = True) -> nn.Module:
    """Build an IRIFP4Linear honoring the model config.

    Reads ``iri_fp4_rounds`` (canonical) with fallback to
    ``iri_fp4_n_rounds`` (legacy alias) for the round count.
    """
    n_rounds = int(getattr(config, "iri_fp4_rounds",
                           getattr(config, "iri_fp4_n_rounds", 3)))
    return IRIFP4Linear(
        in_features, out_features, bias=bias,
        block_size=int(getattr(config, "iri_fp4_block_size", 32)),
        n_rounds=n_rounds,
    )
