"""KIVI key — 2-bit asymmetric KV cache quantization without training.

KIVI (2024) quantizes the KV cache to 2 bits with asymmetric strategies:
- Key cache: per-CHANNEL quantization (group along channel/head_dim dimension)
- Value cache: per-TOKEN quantization (group along sequence dimension)

This asymmetry comes from the observation that K cache has outlier channels
while V cache has outlier tokens. Per-channel K handles channel outliers;
per-token V handles token-specific variation.

This is a TRIVIAL key — pure quantization formula, no data or training.

Reference: KIVI, arxiv 2402.02750
"""
import torch

from research.keys.base import Key, KeyClass, KeyResult


def asymmetric_quantize(x, bits=2, dim=-1):
    """Asymmetric quantization to `bits` bits along dimension `dim`.

    Uses min/max scaling (asymmetric, unsigned-like for signed tensors).
    Returns quantized tensor (dequantized to original dtype for simulation).
    """
    qmax = 2 ** bits - 1
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    scale = (x_max - x_min).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round((x - x_min) / scale), 0, qmax)
    return q * scale + x_min


class KIVIKey(Key):
    """KIVI 2-bit KV cache quantization key.

    Per-channel K quantization + per-token V quantization.
    No training — pure quantization formula.

    Key class: TRIVIAL — fixed formula, no data or training.
    """

    @property
    def name(self) -> str:
        return "kivi"

    @property
    def description(self) -> str:
        return "2-bit asymmetric KV cache quantization (per-channel K, per-token V)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Quantize KV cache to 2 bits.

        Args:
            data: {"k": tensor (..., seq_len, head_dim),
                   "v": tensor (..., seq_len, head_dim),
                   "bits": int (default 2)}

        Returns:
            {"k_quant": tensor (same shape, dequantized),
             "v_quant": tensor (same shape, dequantized)}
        """
        try:
            k = data["k"]
            v = data["v"]
            bits = data.get("bits", 2)

            # K: per-channel quantization (along head_dim, last dim)
            k_quant = asymmetric_quantize(k, bits=bits, dim=-1)

            # V: per-token quantization (along seq_len, second-to-last dim)
            v_quant = asymmetric_quantize(v, bits=bits, dim=-2)

            return KeyResult(
                success=True,
                weights={"k_quant": k_quant, "v_quant": v_quant},
                metadata={
                    "bits": bits,
                    "compression": 16 / bits,  # bf16 → bits
                    "k_method": "per_channel",
                    "v_method": "per_token",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Dequantize (already dequantized in forward for simulation)."""
        try:
            return KeyResult(
                success=True,
                data={"k": weights["k_quant"], "v": weights["v_quant"]},
                metadata={"lossy": True},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


class KIVICache:
    """KV cache with KIVI 2-bit quantization.

    K cache: per-channel 2-bit quantized
    V cache: per-token 2-bit quantized
    """

    def __init__(self, n_heads, head_dim, max_seq_len=2048, bits=2, dtype=torch.bfloat16):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bits = bits
        self.dtype = dtype
        self.k_cache = []  # list of per-token K vectors
        self.v_cache = []

    def append(self, k, v):
        """Add one token's K, V. k: (n_heads, head_dim), v: same."""
        self.k_cache.append(k.detach())
        self.v_cache.append(v.detach())

    def get(self):
        """Return quantized K, V tensors."""
        if not self.k_cache:
            return None, None
        k = torch.stack(self.k_cache, dim=0)  # (seq, n_heads, head_dim)
        v = torch.stack(self.v_cache, dim=0)
        # Per-channel K (along head_dim), per-token V (along seq)
        k_q = asymmetric_quantize(k, bits=self.bits, dim=-1)
        v_q = asymmetric_quantize(v, bits=self.bits, dim=0)
        return k_q.to(self.dtype), v_q.to(self.dtype)

    def __len__(self):
        return len(self.k_cache)


if __name__ == "__main__":
    key = KIVIKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test with synthetic KV cache
    k = torch.randn(1, 4, 128, 64)  # (batch, heads, seq, head_dim)
    v = torch.randn(1, 4, 128, 64)

    r = key.forward({"k": k, "v": v, "bits": 2})
    print(f"Forward: {r.success}")
    print(f"  Compression: {r.metadata['compression']}x")
    print(f"  K method: {r.metadata['k_method']}")
    print(f"  V method: {r.metadata['v_method']}")

    # Measure quantization error
    k_err = (k - r.weights["k_quant"]).pow(2).mean().sqrt()
    v_err = (v - r.weights["v_quant"]).pow(2).mean().sqrt()
    k_rel = k_err / k.pow(2).mean().sqrt() * 100
    v_rel = v_err / v.pow(2).mean().sqrt() * 100
    print(f"  K relative error: {k_rel:.2f}%")
    print(f"  V relative error: {v_rel:.2f}%")
