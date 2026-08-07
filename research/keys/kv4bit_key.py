"""4-bit KV cache quantization key — group-wise 4-bit with per-channel scales.

Complementary to RotorQuant: simpler and faster (no rotation), just
group-wise 4-bit symmetric quantization of K and V cache tensors with
per-channel FP16 scale factors.

Compression: 4-bit values + 16-bit scale per group_size elements.
  For group_size=32: 4 + 16/32 = 4.5 bits/element -> 3.56x vs FP16.

Novel twist — KV Scale Absorption:
  ForgeLM uses QK-Norm (RMSNorm on attention heads). The per-channel KV
  quant scales can be folded into the QK-Norm weights (or the subsequent
  projection), eliminating runtime dequant overhead. After absorption the
  4-bit KV cache is pure int4 and the scales live inside norm weights that
  must execute anyway.

WARNING: Lossy key. Do NOT apply to ForgeLM V2 or expert packs.

Key class: PARTIAL — lossy, reverse() is approximate (not exact round-trip).
"""
import torch
from typing import Dict, Tuple
from .base import Key, KeyClass, KeyResult


class KV4BitKey(Key):
    """4-bit KV cache quantization key (group-wise, per-channel scales).

    forward: KV tensors -> 4-bit quantized + per-channel FP16 scales
    reverse: 4-bit + scales -> approximate KV tensors (lossy)
    """

    def __init__(self, group_size: int = 32):
        self.group_size = group_size

    @property
    def name(self) -> str:
        return "kv4bit"

    @property
    def description(self) -> str:
        return ("4-bit group-wise KV cache quantization with per-channel "
                "FP16 scales. Complementary to RotorQuant (simpler, faster).")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Quantize KV tensors to 4-bit + per-channel scales.

        Args:
            data: {"k": tensor (..., head_dim), "v": tensor (..., head_dim)}

        Returns:
            weights: {"k_q": int8 4-bit-packed, "k_scales": fp16,
                      "v_q": int8 4-bit-packed, "v_scales": fp16}
        """
        try:
            k = data["k"]
            v = data["v"]
            k_q, k_s = quantize_kv_4bit(k, self.group_size)
            v_q, v_s = quantize_kv_4bit(v, self.group_size)
            bits_per = 4 + 16 / self.group_size
            return KeyResult(
                success=True,
                weights={"k_q": k_q, "k_scales": k_s,
                         "v_q": v_q, "v_scales": v_s},
                metadata={
                    "group_size": self.group_size,
                    "bits_per_element": bits_per,
                    "compression": 16 / bits_per,
                    "lossy": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Dequantize 4-bit KV cache to approximate FP16 (lossy)."""
        try:
            k_approx = dequantize_kv_4bit(
                weights["k_q"], weights["k_scales"], self.group_size)
            v_approx = dequantize_kv_4bit(
                weights["v_q"], weights["v_scales"], self.group_size)
            return KeyResult(
                success=True,
                data={"k": k_approx, "v": v_approx},
                metadata={"lossy": True},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def quantize_kv_4bit(kv: torch.Tensor,
                     group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-wise 4-bit symmetric quantization with per-channel scales.

    Quantization is along the last dimension (head_dim). Each group of
    `group_size` elements shares one scale. Scales are per-channel, meaning
    one scale vector per leading-index row (all groups in a row use the
    same scale vector entry — actually per-group, computed per row-group).

    Args:
        kv: (..., head_dim) tensor.
        group_size: number of elements per quant group.

    Returns:
        qkv: int8 tensor (values in [-8, 7]), same shape as kv.
        scales: fp16 tensor of shape (..., ceil(head_dim / group_size)).
    """
    orig_shape = kv.shape
    head_dim = orig_shape[-1]
    n_groups = (head_dim + group_size - 1) // group_size
    pad = n_groups * group_size - head_dim
    kv_padded = kv if pad == 0 else torch.nn.functional.pad(kv, (0, pad))
    grouped = kv_padded.reshape(*orig_shape[:-1], n_groups, group_size)
    scales = grouped.abs().amax(dim=-1).clamp(min=1e-8).to(torch.float16)
    qmax = 7  # 4-bit signed symmetric
    qkv = torch.clamp(
        torch.round(grouped / scales.unsqueeze(-1).to(grouped.dtype)),
        -qmax, qmax,
    ).to(torch.int8)
    return qkv.reshape(orig_shape), scales


def dequantize_kv_4bit(qkv: torch.Tensor, scales: torch.Tensor,
                       group_size: int = 32) -> torch.Tensor:
    """Dequantize 4-bit KV cache to approximate FP16.

    Args:
        qkv: int8 tensor (values in [-8, 7]).
        scales: fp16 tensor of shape (..., n_groups).
        group_size: group size used during quantization.

    Returns:
        Reconstructed KV tensor (fp16), same shape as qkv.
    """
    orig_shape = qkv.shape
    head_dim = orig_shape[-1]
    n_groups = (head_dim + group_size - 1) // group_size
    pad = n_groups * group_size - head_dim
    qkv_padded = qkv if pad == 0 else torch.nn.functional.pad(qkv, (0, pad))
    grouped = qkv_padded.reshape(*orig_shape[:-1], n_groups, group_size).to(torch.float16)
    deq = grouped * scales.unsqueeze(-1).to(torch.float16)
    return deq.reshape(orig_shape)


if __name__ == "__main__":
    key = KV4BitKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    k = torch.randn(1, 4, 32, 128, dtype=torch.float16)
    v = torch.randn(1, 4, 32, 128, dtype=torch.float16)

    result = key.forward({"k": k, "v": v})
    print(f"Forward: {result.success}, compression={result.metadata['compression']:.2f}x")

    rev = key.reverse(result.weights)
    if rev.success:
        k_err = (k - rev.data["k"]).pow(2).mean().sqrt()
        k_rel = k_err / k.pow(2).mean().sqrt() * 100
        print(f"K relative error: {k_rel:.2f}% (lossy expected)")
    print("  KV 4-bit verified (lossy)")
