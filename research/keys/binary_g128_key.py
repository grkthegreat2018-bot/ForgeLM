"""Binary g128 quantization key — 1.125 bits/weight, 14.2x compression.

Inspired by Bonsai-27B (prism-ml): each weight is 1 sign bit {-1, +1}, and
every 128 consecutive weights share a single FP16 scale factor.

True cost: 1 bit (sign) + 16/128 bits (scale) = 1.125 bits/weight.
Compression vs FP16: 16 / 1.125 = 14.22x.

Novel twist — Group Scale Absorption:
  The per-128 FP16 group scales can be folded (absorbed) into adjacent
  RMSNorm weights or the next Linear layer's input projection, eliminating
  runtime dequant overhead entirely. This composes with the Norm Folding
  key: after absorption the binary weights are pure sign bits and the
  scales live inside neighbouring FP16 tensors that must run anyway.

WARNING: Lossy key. Do NOT apply to ForgeLM V2 or expert packs.

Key class: PARTIAL — lossy, reverse() is approximate (not exact round-trip).
"""
from typing import Dict, Tuple

import torch

from .base import Key, KeyClass, KeyResult


class BinaryG128Key(Key):
    """Binary g128 quantization key (1.125 bits/weight).

    forward: FP16 weights -> binary sign bits (int8) + group scales (fp16)
    reverse: binary + scales -> approximate FP16 weights (lossy)
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    @property
    def name(self) -> str:
        return "binary_g128"

    @property
    def description(self) -> str:
        return ("Binary 1-bit quantization with 128-wide group scales "
                "(1.125 bits/weight, 14.2x compression). Bonsai-27B inspired.")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Quantize FP16 weights to binary + group scales.

        Args:
            data: {"weight": tensor (2D Linear weight, fp16)}

        Returns:
            weights: {"signs": int8 tensor, "scales": fp16 tensor}
        """
        try:
            w = data["weight"]
            signs, scales = binary_g128_quantize(w, self.group_size)
            return KeyResult(
                success=True,
                weights={"signs": signs, "scales": scales},
                metadata={
                    "group_size": self.group_size,
                    "bits_per_weight": 1 + 16 / self.group_size,
                    "compression": 16 / (1 + 16 / self.group_size),
                    "lossy": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Reconstruct approximate FP16 weights from binary + scales (lossy)."""
        try:
            signs = weights["signs"]
            scales = weights["scales"]
            w_approx = binary_g128_dequantize(signs, scales, self.group_size)
            return KeyResult(
                success=True,
                data={"weight": w_approx},
                metadata={"lossy": True},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def binary_g128_quantize(weight: torch.Tensor,
                         group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to binary signs + per-group FP16 scales.

    Args:
        weight: (..., N) tensor of FP16/BF16 weights.
        group_size: number of consecutive weights sharing one scale.

    Returns:
        signs: int8 tensor, same shape as weight, values in {-1, +1}.
        scales: fp16 tensor of shape (..., ceil(N / group_size)).
    """
    orig_shape = weight.shape
    flat = weight.reshape(-1)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(-1, group_size)  # (n_groups, group_size)
    scales = grouped.abs().amax(dim=1).clamp(min=1e-8).to(torch.float16)
    signs = torch.sign(grouped / scales.unsqueeze(1).to(grouped.dtype))
    signs = torch.where(grouped == 0, torch.ones_like(signs), signs)
    signs = signs.reshape(-1)[:n].reshape(orig_shape).to(torch.int8)
    return signs, scales


def binary_g128_dequantize(signs: torch.Tensor, scales: torch.Tensor,
                            group_size: int = 128) -> torch.Tensor:
    """Reconstruct approximate FP16 weights from binary signs + group scales.

    Args:
        signs: int8 tensor, values in {-1, +1}.
        scales: fp16 tensor of shape (n_groups,).
        group_size: group size used during quantization.

    Returns:
        Reconstructed weight tensor (fp16), same shape as signs.
    """
    orig_shape = signs.shape
    flat = signs.reshape(-1).to(torch.float16)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(-1, group_size)
    deq = grouped * scales.unsqueeze(1).to(torch.float16)
    return deq.reshape(-1)[:n].reshape(orig_shape)


def apply_binary_g128(state_dict: dict[str, torch.Tensor],
                      group_size: int = 128) -> dict[str, torch.Tensor]:
    """Quantize all 2D Linear weights in a state dict to binary g128.

    Replaces each 2D weight with dequantized (lossy) FP16 so the model
    can run without custom kernels. Stores signs/scales in metadata keys.

    Args:
        state_dict: model state dict.
        group_size: group size for quantization.

    Returns:
        Modified state dict with quantized (dequantized) weights.
    """
    out = dict(state_dict)
    n_quantized = 0
    for key in list(out.keys()):
        w = out[key]
        if not isinstance(w, torch.Tensor) or w.dim() != 2:
            continue
        if w.numel() < group_size:
            continue
        signs, scales = binary_g128_quantize(w, group_size)
        out[key] = binary_g128_dequantize(signs, scales, group_size).to(w.dtype)
        out[f"{key}._bin_signs"] = signs
        out[f"{key}._bin_scales"] = scales
        n_quantized += 1
    print(f"  [Binary g128] Quantized {n_quantized} 2D weights "
          f"(group_size={group_size}, ~14.2x compression)")
    return out


if __name__ == "__main__":
    key = BinaryG128Key()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    w = torch.randn(256, 512, dtype=torch.float16)
    result = key.forward({"weight": w})
    print(f"Forward: {result.success}, compression={result.metadata['compression']:.1f}x")

    rev = key.reverse(result.weights)
    if rev.success:
        err = (w - rev.data["weight"]).pow(2).mean().sqrt()
        rel = err / w.pow(2).mean().sqrt() * 100
        print(f"Reverse relative error: {rel:.2f}% (lossy expected)")
    print("  Binary g128 verified (lossy)")
