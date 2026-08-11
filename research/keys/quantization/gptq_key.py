"""GPTQ Key — 4-bit weight quantization via second-order error compensation.

GPTQ (Frantar et al., 2023) quantizes weights column-by-column, using the
Hessian inverse to compensate for quantization error. This gives near-lossless
4-bit quantization with minimal calibration data.

As a Key: this transforms 2D weight tensors from bf16 → INT4 + scales.
The quantized weights are stored alongside per-group scales.

This is a PARTIAL key — quantization is lossy (not reversible).

Usage:
    from research.keys.gptq_key import GPTQKey, quantize_gptq
    # Quantize model weights to INT4
    quantized_state = quantize_gptq(state, group_size=128)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class GPTQKey(Key):
    """GPTQ 4-bit quantization key."""

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    @property
    def name(self) -> str:
        return "gptq"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL  # Lossy quantization

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Not used — GPTQ operates on full model with calibration."""
        return KeyResult(success=False, error="GPTQ operates on full model with calibration")

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=False, error="GPTQ is not reversible (lossy)")


def quantize_gptq(state: dict[str, torch.Tensor], group_size: int = 128,
                  bits: int = 4) -> dict[str, torch.Tensor]:
    """Apply GPTQ-style 4-bit quantization to all 2D weight tensors.

    Simplified GPTQ: per-group symmetric quantization with scale compensation.
    Full GPTQ uses Hessian-based error correction; this uses the simpler
    RTN (Round-To-Nearest) with per-group scales, which is close to GPTQ
    for group_size=128.

    Args:
        state: Model state dict
        group_size: Quantization group size (smaller = better quality)
        bits: Target bit width (4 or 8)

    Returns:
        State dict with quantized weights (stored as int8 + scales)
    """
    quantized = {}
    n_quantized = 0
    n_skipped = 0

    for name, tensor in state.items():
        # Only quantize 2D weight tensors (Linear layers)
        if tensor.dim() != 2 or tensor.numel() < 10000:
            quantized[name] = tensor
            n_skipped += 1
            continue

        # Skip norms, embeddings, and small tensors
        if "norm" in name or "embed" in name or "head" in name:
            quantized[name] = tensor
            n_skipped += 1
            continue

        q, scales = _quantize_tensor_gptq(tensor, group_size, bits)
        quantized[f"{name}__q"] = q
        quantized[f"{name}__scale"] = scales
        n_quantized += 1

    orig_size = sum(t.numel() * t.element_size() for t in state.values())
    new_size = sum(t.numel() * t.element_size() for t in quantized.values())
    print(f"  [GPTQ] Quantized {n_quantized} tensors, skipped {n_skipped}")
    print(f"  [GPTQ] {orig_size/1e9:.2f} GB → {new_size/1e9:.2f} GB ({new_size/orig_size:.1%})")

    return quantized


def _quantize_tensor_gptq(tensor: torch.Tensor, group_size: int = 128,
                           bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to INT4 with per-group scales.

    Args:
        tensor: [out_features, in_features] weight tensor
        group_size: Number of input features per quantization group
        bits: Target bits (4 or 8)

    Returns:
        (quantized_int8, scales) where quantized values are in [-2^(b-1), 2^(b-1)-1]
    """
    out_features, in_features = tensor.shape

    # Pad to multiple of group_size
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad))

    # Reshape to groups
    n_groups = tensor.shape[1] // group_size
    t = tensor.float().reshape(out_features, n_groups, group_size)

    # Per-group scale (symmetric)
    max_val = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    max_int = 2 ** (bits - 1) - 1  # 7 for 4-bit, 127 for 8-bit
    scale = max_val / max_int

    # Quantize
    q = (t / scale).round().clamp(-max_int - 1, max_int).to(torch.int8)

    # Reshape back
    q = q.reshape(out_features, -1)
    if pad > 0:
        q = q[:, :in_features]

    scales = scale.squeeze(-1).to(torch.float16)  # [out_features, n_groups]
    return q, scales


def dequantize_gptq(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize GPTQ state dict back to bf16.

    Args:
        state: State dict with __q and __scale suffixes

    Returns:
        State dict with dequantized bf16 tensors
    """
    dequantized = {}
    quant_keys = {k for k in state if k.endswith("__q")}

    for key, tensor in state.items():
        if key.endswith("__q"):
            base_name = key.replace("__q", "")
            scale_key = key.replace("__q", "__scale")
            scales = state[scale_key]
            dequantized[base_name] = _dequantize_tensor_gptq(tensor, scales)
        elif key.endswith("__scale"):
            continue  # Skip scale tensors (used by __q)
        else:
            dequantized[key] = tensor

    return dequantized


def _dequantize_tensor_gptq(q: torch.Tensor, scales: torch.Tensor,
                             group_size: int = 128) -> torch.Tensor:
    """Dequantize a GPTQ tensor."""
    out_features = q.shape[0]
    in_features = q.shape[1]
    n_groups = scales.shape[1]

    # Pad if needed
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        q = torch.nn.functional.pad(q, (0, pad))

    q = q.float().reshape(out_features, n_groups, group_size)
    s = scales.float().unsqueeze(-1)  # [out_features, n_groups, 1]

    deq = q * s
    deq = deq.reshape(out_features, -1)

    if pad > 0:
        deq = deq[:, :in_features]

    return deq.to(torch.bfloat16)
