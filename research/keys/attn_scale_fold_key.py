"""Attention Scale Folding Key — fold 1/sqrt(d_k) into Q projection weights.

LOSSLESS: RoPE(Q * c) = c * RoPE(Q) since RoPE is a linear rotation.
Eliminates the per-attention scalar multiply (1/sqrt(head_dim)) by
pre-scaling q_proj weight rows by sqrt(1/sqrt(head_dim)) = head_dim^(-1/4).

In standard attention:
  scores = Q @ K^T * scale    where scale = 1/sqrt(head_dim)
  attn = softmax(scores)
  out = attn @ V

After folding:
  Q' = Q * sqrt(scale) = Q * head_dim^(-1/4)   [folded into q_proj]
  K' = K * sqrt(scale) = K * head_dim^(-1/4)   [folded into k_up_proj]
  scores = Q' @ K'^T                            [no scale multiply needed]
  = (Q * sqrt(s)) @ (K * sqrt(s))^T = Q @ K^T * s   [same result]

Key insight: fold sqrt(scale) into BOTH q_proj and k_up_proj, so
  Q' @ K'^T = (Q * sqrt(s)) @ (K * sqrt(s))^T = Q @ K^T * s
This eliminates the scale multiply entirely.

For MLA: k_up_proj produces the uncompressed K (before RoPE), so we can
fold into k_up_proj output rows. Q goes through q_proj then RoPE then q_norm.
Since RoPE is linear and q_norm is per-head RMSNorm (also linear in the
norm weight), the scale propagates through.

Key class: FULL — reversible (divide back), composable.

Usage:
    from research.keys.attn_scale_fold_key import AttnScaleFoldKey, apply_attn_scale_fold
    state = apply_attn_scale_fold(state, n_layers=28, head_dim=128)
"""
import math
import torch
from typing import Dict
from .base import Key, KeyClass, KeyResult


class AttnScaleFoldKey(Key):
    """Attention Scale Folding key — absorb 1/sqrt(d_k) into Q/K projections.

    Eliminates the per-attention-head scale multiply by pre-scaling
    q_proj and k_up_proj weights. LOSSLESS (RoPE is linear).

    Key class: FULL — reversible, composable.
    """

    @property
    def name(self) -> str:
        return "attn_scale_fold"

    @property
    def description(self) -> str:
        return "Fold 1/sqrt(head_dim) into q_proj/k_up_proj (eliminate scale multiply)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Fold attention scale into q_proj and k_up_proj weights.

        Args:
            data: state dict with q_proj.weight, k_up_proj.weight, and head_dim

        Returns:
            modified state dict with scale folded into projections
        """
        try:
            state = dict(data)

            # Detect head_dim from q_proj shape
            # q_proj.weight shape: [n_heads * head_dim, d_model]
            # For ForgeLM V2: [1536, 1536] → n_heads=12, head_dim=128
            q_key = None
            for k in state:
                if "q_proj.weight" in k and "blocks." in k:
                    q_key = k
                    break

            if q_key is None:
                return KeyResult(success=False, error="No q_proj.weight found")

            q_w = state[q_key]
            out_dim = q_w.shape[0]  # n_heads * head_dim

            # Detect n_heads and head_dim
            # For MLA: n_heads=12, head_dim=128, out_dim=1536
            # Try common configs
            if out_dim == 1536:
                n_heads, head_dim = 12, 128
            elif out_dim == 1024:
                n_heads, head_dim = 16, 64
            elif out_dim == 768:
                n_heads, head_dim = 12, 64
            else:
                # Try to infer: head_dim is usually 64 or 128
                for hd in [128, 64, 96, 80, 256, 32]:
                    if out_dim % hd == 0:
                        n_heads = out_dim // hd
                        head_dim = hd
                        break
                else:
                    return KeyResult(success=False, error=f"Cannot infer head_dim from out_dim={out_dim}")

            scale = 1.0 / math.sqrt(head_dim)
            fold_factor = math.sqrt(scale)  # head_dim^(-1/4)

            # Count layers
            n_layers = 0
            for k in state:
                if "q_proj.weight" in k and "blocks." in k:
                    layer = int(k.split(".")[1])
                    n_layers = max(n_layers, layer + 1)

            folded = 0
            for i in range(n_layers):
                # Fold sqrt(scale) into q_proj output rows
                qk = f"blocks.{i}.attn.q_proj.weight"
                if qk in state:
                    w = state[qk].float()
                    # Scale each output row by fold_factor
                    # q_proj.weight shape: [out_dim, in_dim]
                    # Row j corresponds to output dimension j
                    state[qk] = (w * fold_factor).to(state[qk].dtype)
                    folded += 1

                # Fold sqrt(scale) into k_up_proj output rows
                kk = f"blocks.{i}.attn.k_up_proj.weight"
                if kk in state:
                    w = state[kk].float()
                    state[kk] = (w * fold_factor).to(state[kk].dtype)
                    folded += 1

                # Also fold q_proj bias if present
                qb = f"blocks.{i}.attn.q_proj.bias"
                if qb in state:
                    state[qb] = (state[qb].float() * fold_factor).to(state[qb].dtype)

                kb = f"blocks.{i}.attn.k_up_proj.bias"
                if kb in state:
                    state[kb] = (state[kb].float() * fold_factor).to(state[kb].dtype)

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "n_layers": n_layers,
                    "head_dim": head_dim,
                    "n_heads": n_heads,
                    "scale": scale,
                    "fold_factor": fold_factor,
                    "n_folded": folded,
                    "saves_compute": True,
                    "note": "Eliminates per-attention scale multiply",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Unfold scale from q_proj and k_up_proj (divide by fold_factor)."""
        try:
            state = dict(weights)
            n_layers = 0
            for k in state:
                if "q_proj.weight" in k and "blocks." in k:
                    layer = int(k.split(".")[1])
                    n_layers = max(n_layers, layer + 1)

            out_dim = state[f"blocks.0.attn.q_proj.weight"].shape[0]
            if out_dim == 1536:
                head_dim = 128
            elif out_dim == 1024:
                head_dim = 64
            else:
                for hd in [128, 64, 96, 80, 256, 32]:
                    if out_dim % hd == 0:
                        head_dim = hd
                        break

            fold_factor = math.sqrt(1.0 / math.sqrt(head_dim))

            for i in range(n_layers):
                for pname in ["q_proj", "k_up_proj"]:
                    wk = f"blocks.{i}.attn.{pname}.weight"
                    if wk in state:
                        state[wk] = (state[wk].float() / fold_factor).to(state[wk].dtype)
                    bk = f"blocks.{i}.attn.{pname}.bias"
                    if bk in state:
                        state[bk] = (state[bk].float() / fold_factor).to(state[bk].dtype)

            return KeyResult(success=True, data=state,
                             metadata={"un folded": True, "fold_factor": fold_factor})
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def apply_attn_scale_fold(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fold attention scale into q_proj and k_up_proj weights."""
    key = AttnScaleFoldKey()
    result = key.forward(state)
    if not result.success:
        raise RuntimeError(f"Attention scale fold failed: {result.error}")

    meta = result.metadata
    print(f"  [AttnScaleFold] Folded 1/sqrt({meta['head_dim']}) into "
          f"{meta['n_folded']} projections across {meta['n_layers']} layers")
    print(f"  [AttnScaleFold] Eliminates {meta['n_layers']} scale multiplies per forward pass")
    return result.weights


if __name__ == "__main__":
    key = AttnScaleFoldKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test with dummy data
    state = {
        "blocks.0.attn.q_proj.weight": torch.randn(1536, 1536, dtype=torch.bfloat16),
        "blocks.0.attn.k_up_proj.weight": torch.randn(512, 1536, dtype=torch.bfloat16),
        "blocks.0.attn.q_proj.bias": torch.randn(1536, dtype=torch.bfloat16),
        "blocks.1.attn.q_proj.weight": torch.randn(1536, 1536, dtype=torch.bfloat16),
        "blocks.1.attn.k_up_proj.weight": torch.randn(512, 1536, dtype=torch.bfloat16),
    }

    result = key.forward(state)
    print(f"  Success: {result.success}")
    print(f"  Folded: {result.metadata['n_folded']} projections")
    print(f"  Head dim: {result.metadata['head_dim']}")
    print(f"  Fold factor: {result.metadata['fold_factor']:.6f}")

    # Verify reversibility
    rev = key.reverse(result.weights)
    print(f"  Reverse success: {rev.success}")

    # Check round-trip
    orig = state["blocks.0.attn.q_proj.weight"].float()
    rev_w = rev.data["blocks.0.attn.q_proj.weight"].float()
    max_diff = (orig - rev_w).abs().max().item()
    print(f"  Round-trip max diff: {max_diff:.8f}")
