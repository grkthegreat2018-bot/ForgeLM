"""QuaRot R2-R4 key — additional Hadamard rotation positions for full quantization.

QuaRot (NeurIPS 2024) applies Hadamard rotations at 4 positions in a transformer:
  R1: residual stream (inputs of q/k/v/gate/up_proj) — already in spinquant_key.py
  R2: attention head value projection (per-head, head_dim) — OFFLINE, fused into weights
  R3: Q/K after RoPE (head_dim) — ONLINE, improves KV cache quantization
  R4: down_proj input (intermediate_size) — ONLINE, improves FFN activation quant

R2 is offline (fused into weights, no runtime cost).
R3/R4 are online (small runtime cost, but fast Hadamard transform).

This key handles R2 (offline fusion). R3/R4 are runtime hooks, not weight transforms.

Key class: TRIVIAL — fixed Hadamard, no data or training.

Reference: QuaRot, arxiv 2404.00456
"""
import torch
from research.keys.base import Key, KeyClass, KeyResult
from research.keys.spinquant_key import hadamard_matrix


class QuaRotR2Key(Key):
    """QuaRot R2 rotation — per-head Hadamard on V projection.

    Applies a Hadamard rotation to each attention head's V output,
    fused into the V projection weight. The inverse is fused into
    the O projection weight. No runtime cost.

    Key class: TRIVIAL — fixed Hadamard per head.
    """

    @property
    def name(self) -> str:
        return "quarot_r2"

    @property
    def description(self) -> str:
        return "Per-head Hadamard rotation on V/O (offline, fused into weights)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict) -> KeyResult:
        """Apply R2 rotation: rotate V output, inverse-rotate O input.

        Args:
            data: {"v_weight": tensor (n_heads * head_dim, d_model),
                   "o_weight": tensor (d_model, n_heads * head_dim),
                   "n_heads": int,
                   "head_dim": int}

        Returns:
            {"v_weight_rotated": tensor, "o_weight_rotated": tensor,
             "rotation": tensor (head_dim, head_dim)}
        """
        try:
            v_weight = data["v_weight"]
            o_weight = data["o_weight"]
            n_heads = data["n_heads"]
            head_dim = data["head_dim"]

            # Generate per-head Hadamard (shared across heads for simplicity)
            H = hadamard_matrix(head_dim).to(dtype=v_weight.dtype)
            # Block-diagonal: each head gets the same H
            # V_rotated = block_diag(H, H, ..., H) @ V
            # O_rotated = O @ block_diag(H^T, H^T, ..., H^T)

            # Reshape V weight to per-head blocks
            # v_weight: (n_heads * head_dim, d_model)
            v_reshaped = v_weight.reshape(n_heads, head_dim, -1)  # (n_heads, head_dim, d_model)
            H_batch = H.unsqueeze(0).expand(n_heads, -1, -1).contiguous()
            v_rotated = torch.bmm(H_batch, v_reshaped)
            v_weight_rotated = v_rotated.reshape(n_heads * head_dim, -1)

            # O weight: (d_model, n_heads * head_dim)
            d_model = o_weight.shape[0]
            o_reshaped = o_weight.reshape(d_model, n_heads, head_dim).permute(1, 0, 2)  # (n_heads, d_model, head_dim)
            Ht_batch = H.T.unsqueeze(0).expand(n_heads, -1, -1).contiguous()
            o_rotated = torch.bmm(o_reshaped, Ht_batch)  # (n_heads, d_model, head_dim)
            o_weight_rotated = o_rotated.permute(1, 0, 2).reshape(d_model, n_heads * head_dim)

            return KeyResult(
                success=True,
                weights={
                    "v_weight_rotated": v_weight_rotated,
                    "o_weight_rotated": o_weight_rotated,
                    "rotation": H,
                },
                metadata={"n_heads": n_heads, "head_dim": head_dim, "position": "R2"},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Undo R2 rotation (apply inverse Hadamard)."""
        try:
            v_rot = weights["v_weight_rotated"]
            o_rot = weights["o_weight_rotated"]
            H = weights["rotation"]
            n_heads = weights.get("n_heads", v_rot.shape[0] // H.shape[0])
            head_dim = H.shape[0]
            d_model = o_rot.shape[0]  # o_weight: (d_model, n_heads * head_dim)

            v_reshaped = v_rot.reshape(n_heads, head_dim, d_model)
            Ht_batch = H.T.unsqueeze(0).expand(n_heads, -1, -1).contiguous()
            v_orig = torch.bmm(Ht_batch, v_reshaped)
            v_weight = v_orig.reshape(n_heads * head_dim, d_model)

            o_reshaped = o_rot.reshape(d_model, n_heads, head_dim).permute(1, 0, 2)
            H_batch = H.unsqueeze(0).expand(n_heads, -1, -1).contiguous()
            o_orig = torch.bmm(o_reshaped, H_batch)
            o_weight = o_orig.permute(1, 0, 2).reshape(d_model, n_heads * head_dim)

            return KeyResult(success=True, data={"v_weight": v_weight, "o_weight": o_weight})
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def apply_quarot_r2_to_model(model):
    """Apply QuaRot R2 rotation to all attention layers (in-place).

    Fuses Hadamard rotation into V and O projection weights.
    No runtime cost — the rotation is absorbed into the weights.

    Returns:
        Number of layers modified.
    """
    modified = 0
    for block in model.blocks:
        attn = block.attn
        # Find V and O projections
        v_weight = None
        o_weight = None
        for name in ["v_proj", "v", "value", "v_down_proj"]:
            if hasattr(attn, name):
                v_weight = getattr(attn, name).weight.data
                break
        for name in ["o_proj", "o", "out_proj"]:
            if hasattr(attn, name):
                o_weight = getattr(attn, name).weight.data
                break

        if v_weight is None or o_weight is None:
            continue

        # Infer head config
        n_heads = getattr(attn, 'n_heads', 12)
        head_dim = v_weight.shape[0] // n_heads

        # Skip if head_dim is not power of 2
        if head_dim & (head_dim - 1) != 0:
            continue

        key = QuaRotR2Key()
        result = key.forward({
            "v_weight": v_weight, "o_weight": o_weight,
            "n_heads": n_heads, "head_dim": head_dim,
        })
        if result.success:
            # Write back
            for name in ["v_proj", "v", "value", "v_down_proj"]:
                if hasattr(attn, name):
                    getattr(attn, name).weight.data = result.weights["v_weight_rotated"]
                    break
            for name in ["o_proj", "o", "out_proj"]:
                if hasattr(attn, name):
                    getattr(attn, name).weight.data = result.weights["o_weight_rotated"]
                    break
            modified += 1

    return modified


if __name__ == "__main__":
    key = QuaRotR2Key()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    n_heads, head_dim, d_model = 4, 64, 256
    v_weight = torch.randn(n_heads * head_dim, d_model)
    o_weight = torch.randn(d_model, n_heads * head_dim)

    r = key.forward({"v_weight": v_weight, "o_weight": o_weight,
                     "n_heads": n_heads, "head_dim": head_dim})
    print(f"Forward: {r.success}")
    print(f"  V rotated: {r.weights['v_weight_rotated'].shape}")
    print(f"  O rotated: {r.weights['o_weight_rotated'].shape}")

    # Verify round-trip
    rv = key.reverse({**r.weights, "n_heads": n_heads})
    print(f"Reverse: {rv.success}")
    v_err = (rv.data["v_weight"] - v_weight).abs().max().item()
    o_err = (rv.data["o_weight"] - o_weight).abs().max().item()
    print(f"  V round-trip err: {v_err:.2e}")
    print(f"  O round-trip err: {o_err:.2e}")
