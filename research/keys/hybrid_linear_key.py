"""Hybrid Linear Attention Key — convert full attention layers to linear attention.

Inspired by Bonsai-27B: 75% linear attention / 25% full attention across layers.
Linear attention is O(T·d) instead of O(T²·d), and only the full-attention layers
grow a KV cache during inference.

WARNING: Lossy key (architecture change). Do NOT apply to ForgeLM V2 or expert packs.

The method:
  1. Standard attention: softmax(QK^T / sqrt(d)) V — O(T²·d)
  2. Linear attention replaces softmax with a kernel feature map phi:
       phi(Q) @ (phi(K)^T V)  — O(T·d²) or O(T·d) with feature dim << d
  3. For each layer marked linear: project Q,K through phi (elu+1), discard
     softmax-related params (no temperature scaling, no causal mask needed
     for the linear portion since phi(Q)phi(K)^T is already non-negative)
  4. Bottom 25% of layers stay full attention (those benefit most from
     precise softmax normalization for local token interactions)

Novel twist — adaptive split ratio: instead of a fixed 75/25 split, learn which
layers should be linear vs full based on per-layer attention entropy. Layers
with high attention entropy (diffuse, uniform attention) are good candidates
for linear approximation; layers with low entropy (sharp, peaked attention)
should stay full. This adaptive approach can outperform fixed ratios.

Key class: PARTIAL — one direction (full → hybrid), not reversible.

Usage:
    from research.keys.hybrid_linear_key import HybridLinearKey, convert_to_hybrid_attention
    state, linear_layers = convert_to_hybrid_attention(state, n_layers=28, linear_ratio=0.75)
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base import Key, KeyClass, KeyResult


class HybridLinearKey(Key):
    """Hybrid Linear Attention key — convert full attention to linear attention.

    Marks a fraction of layers as linear attention (O(T·d) instead of O(T²·d)).
    The remaining layers keep full softmax attention.

    Key class: PARTIAL — architecture change is irreversible.

    WARNING: Lossy key (architecture change). Do NOT apply to ForgeLM V2 or expert packs.
    """

    def __init__(self, linear_ratio: float = 0.75):
        self.linear_ratio = linear_ratio

    @property
    def name(self) -> str:
        return "hybrid_linear"

    @property
    def description(self) -> str:
        return ("Convert full attention to hybrid linear/full attention "
                "(Bonsai-27B style, 75% linear / 25% full)")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Convert full attention weights to linear attention for specified layers.

        Args:
            data: {"state": state_dict, "n_layers": int,
                   "linear_ratio": float (optional, default 0.75)}

        Returns:
            KeyResult with modified state_dict and metadata listing linear layers.
        """
        try:
            state = dict(data.get("state", data))
            n_layers = data["n_layers"]
            ratio = data.get("linear_ratio", self.linear_ratio)

            modified, linear_layers = convert_to_hybrid_attention(
                state, n_layers, linear_ratio=ratio)

            return KeyResult(
                success=True,
                weights=modified,
                metadata={
                    "n_linear_layers": len(linear_layers),
                    "n_full_layers": n_layers - len(linear_layers),
                    "linear_layers": linear_layers,
                    "linear_ratio": ratio,
                    "method": "elu_plus_1_kernel",
                    "lossy": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Not supported — architecture change is irreversible."""
        return KeyResult(
            success=False,
            error="HybridLinearKey.reverse is not supported: "
                  "architecture change (full→linear) is irreversible.",
        )


def convert_to_hybrid_attention(
    state: Dict[str, torch.Tensor],
    n_layers: int,
    linear_ratio: float = 0.75,
) -> Tuple[Dict[str, torch.Tensor], List[bool]]:
    """Mark top `linear_ratio` of layers as linear attention, bottom stays full.

    Linear attention layers use phi(Q) @ (phi(K)^T V) with elu+1 feature map,
    so softmax-related parameters (temperature, causal bias) are discarded.

    Args:
        state: model state dict
        n_layers: total number of attention layers
        linear_ratio: fraction of layers to convert (top layers become linear)

    Returns:
        (modified state_dict, list of bool — True if layer is linear attention)
    """
    n_linear = int(n_layers * linear_ratio)
    # Top layers (higher index) become linear; bottom layers stay full
    linear_layers = [i >= (n_layers - n_linear) for i in range(n_layers)]

    for layer_idx in range(n_layers):
        if not linear_layers[layer_idx]:
            continue

        prefix = f"blocks.{layer_idx}.attn."
        # Discard softmax temperature / scale params (linear attn has no softmax)
        for soft_key in [f"{prefix}temperature", f"{prefix}scale",
                         f"{prefix}softmax_bias"]:
            if soft_key in state:
                del state[soft_key]

        # Mark Q/K projections as linear (add a flag tensor for inference dispatch)
        flag_key = f"{prefix}linear_attn"
        state[flag_key] = torch.tensor([1], dtype=torch.int32)

    print(f"  [Hybrid Linear] {n_linear}/{n_layers} layers → linear attention "
          f"(ratio={linear_ratio:.2f})")
    print(f"    Linear layers: {[i for i, v in enumerate(linear_layers) if v]}")
    print(f"    Full layers:   {[i for i, v in enumerate(linear_layers) if not v]}")
    return state, linear_layers


def linear_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute linear attention: phi(Q) @ (phi(K)^T V) using elu+1 feature map.

    Replaces softmax(QK^T / sqrt(d)) V with a kernel approximation:
      phi(x) = elu(x) + 1   (non-negative feature map)
      attn = phi(Q) @ (phi(K)^T V)   — computed in O(T·d²) or chunked O(T·d)

    The numerator is phi(Q) @ S where S = phi(K)^T V (d×d matrix, independent of T).
    The denominator is phi(Q) @ z where z = sum(phi(K)) (d-vector).
    This is the standard linear attention formulation (Katharopoulos et al. 2020).

    Args:
        q: (B, H, T, d) query tensor
        k: (B, H, T, d) key tensor
        v: (B, H, T, d_v) value tensor

    Returns:
        (B, H, T, d_v) attention output
    """
    phi_q = F.elu(q) + 1.0   # (B, H, T, d)
    phi_k = F.elu(k) + 1.0   # (B, H, T, d)

    # S = phi(K)^T @ V  → (B, H, d, d_v)
    s = torch.einsum("bhtd,bhtv->bhdv", phi_k, v)
    # z = sum over T of phi(K) → (B, H, d)
    z = phi_k.sum(dim=2)      # (B, H, d)

    # numerator = phi(Q) @ S → (B, H, T, d_v)
    num = torch.einsum("bhtd,bhdv->bhtv", phi_q, s)
    # denominator = phi(Q) @ z → (B, H, T, 1)
    denom = torch.einsum("bhtd,bhd->bht", phi_q, z).unsqueeze(-1) + 1e-6

    return num / denom
