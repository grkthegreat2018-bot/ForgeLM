"""RotorQuant key — fixed Givens rotation for KV cache compression.

No weights to learn. The rotation is data-oblivious (fixed random angles).
This is a TRIVIAL key — just wraps the existing RotorQuant rotation logic
in the Key interface.

Key class: TRIVIAL — fixed formula, no data or training needed.
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult
from research.quantization.rotorquant import make_givens_rotations, rot2_apply, rot2_inverse


class RotorQuantKey(Key):
    """RotorQuant KV cache compression key.

    Generates fixed Givens rotations (2D block-diagonal) for KV cache
    decorrelation before scalar quantization. No learning needed —
    the rotation angles are random but fixed (seeded).

    The key converts: raw KV vectors → rotated + quantized KV vectors.
    Reverse: rotated KV → raw KV (lossy due to quantization).
    """

    @property
    def name(self) -> str:
        return "rotorquant"

    @property
    def description(self) -> str:
        return "Fixed Givens rotation for KV cache compression (PlanarQuant variant)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict) -> KeyResult:
        """data -> rotated KV (ready for quantization).

        Args:
            data: {"k": tensor (..., head_dim), "v": tensor (..., head_dim),
                   "head_dim": int, "bits": int}

        Returns:
            {"k_compressed": tensor, "v_compressed": tensor, "rotations": tensor}
        """
        try:
            k = data["k"]
            v = data["v"]
            head_dim = data.get("head_dim", k.shape[-1])
            bits = data.get("bits", 4)

            n_groups = head_dim // 2
            rotations = make_givens_rotations(n_groups)  # (n_groups, 2)

            # Reshape to pairs, rotate, reshape back
            def apply_rot(t):
                shape = t.shape
                t_pairs = t.reshape(*shape[:-1], n_groups, 2)  # (..., n_groups, 2)
                # Broadcast rotations to t_pairs shape
                rot = rotations.to(t.device, t.dtype)  # (n_groups, 2)
                c = rot[..., 0]  # (n_groups,)
                s = rot[..., 1]  # (n_groups,)
                v0 = t_pairs[..., 0]
                v1 = t_pairs[..., 1]
                r0 = c * v0 - s * v1
                r1 = s * v0 + c * v1
                return torch.stack([r0, r1], dim=-1).reshape(*shape)

            k_rot = apply_rot(k)
            v_rot = apply_rot(v)

            # Quantize (uniform, per-vector scale)
            qmax = 2 ** (bits - 1) - 1
            k_scale = k_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            v_scale = v_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            k_quant = torch.clamp(torch.round(k_rot / k_scale), -qmax, qmax) * k_scale
            v_quant = torch.clamp(torch.round(v_rot / v_scale), -qmax, qmax) * v_scale

            return KeyResult(
                success=True,
                weights={"k_compressed": k_quant, "v_compressed": v_quant,
                         "rotations": rotations, "k_scale": k_scale, "v_scale": v_scale},
                metadata={"bits": bits, "head_dim": head_dim, "compression": 16 / bits},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """rotated KV -> approximate raw KV (lossy)."""
        try:
            k_quant = weights["k_compressed"]
            v_quant = weights["v_compressed"]
            rotations = weights["rotations"]
            head_dim = k_quant.shape[-1]
            n_groups = head_dim // 2

            def apply_inverse(t):
                shape = t.shape
                t_pairs = t.reshape(*shape[:-1], n_groups, 2)
                rot = rotations.to(t.device, t.dtype)
                c = rot[..., 0]
                s = rot[..., 1]
                v0 = t_pairs[..., 0]
                v1 = t_pairs[..., 1]
                r0 = c * v0 + s * v1
                r1 = -s * v0 + c * v1
                return torch.stack([r0, r1], dim=-1).reshape(*shape)

            k_approx = apply_inverse(k_quant)
            v_approx = apply_inverse(v_quant)

            return KeyResult(
                success=True,
                data={"k": k_approx, "v": v_approx},
                metadata={"lossy": True},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


if __name__ == "__main__":
    key = RotorQuantKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test round-trip
    d = 128
    k = torch.randn(1, 4, 32, d)  # (batch, heads, seq, head_dim)
    v = torch.randn(1, 4, 32, d)

    result = key.forward({"k": k, "v": v, "head_dim": d, "bits": 4})
    print(f"Forward: {result.success}, compression={result.metadata.get('compression')}x")

    rev = key.reverse(result.weights)
    print(f"Reverse: {rev.success}")

    if rev.success:
        k_err = (k - rev.data["k"]).pow(2).mean().sqrt()
        v_err = (v - rev.data["v"]).pow(2).mean().sqrt()
        k_rel = k_err / k.pow(2).mean().sqrt() * 100
        print(f"K relative error: {k_rel:.2f}%")
        print(f"V relative error: {v_err / v.pow(2).mean().sqrt() * 100:.2f}%")
