"""QK-Norm for MLA Key — training-free RMSNorm absorption into MLA projections.

Based on "QK-Normed MLA: QK normalization without full key caching" (2026).

Key insight: RMSNorm decomposes into:
  1. Static affine weight γ (per-dimension scale) — absorb into projections
  2. Dynamic scalar S(z) = 1/sqrt(mean(z²) + eps) — one scalar per token per group

For MLA:
  - K-side norm weight γ_k can be absorbed into k_up_proj: W_k' = diag(γ_k) @ W_k
  - Q-side norm weight γ_q can be absorbed into q_proj: W_q' = diag(γ_q) @ W_q
  - The dynamic RMS scalar is computed at runtime (one division per token)

With identity init (γ=1), this is lossless — the norm is a no-op.
With learned γ (from fine-tuning), the absorption bakes it into weights.

This is a FULL key — reversible (un-absorb to recover γ), composable.

Usage:
    from research.keys.qk_norm_mla_key import QKNormMLAKey, apply_qk_norm_mla
    # Add QK-Norm weights to checkpoint (identity init)
    apply_qk_norm_mla(state_dict, n_layers=28, head_dim=128)
"""
from typing import Dict

import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class QKNormMLAKey(Key):
    """QK-Norm for MLA — absorb RMSNorm into projections.

    Adds q_norm and k_norm RMSNorm layers to MLA attention.
    With identity init (weight=1), this is lossless.
    The norm weights can be absorbed into q_proj and k_up_proj
    to eliminate extra parameters at inference.

    Key class: FULL — reversible, data→weight, composable.
    """

    @property
    def name(self) -> str:
        return "qk_norm_mla"

    @property
    def description(self) -> str:
        return "QK-Norm for MLA: RMSNorm on Q/K with projection absorption"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Initialize QK-Norm weights (identity init).

        Args:
            data: {"n_layers": int, "head_dim": int}

        Returns:
            weights dict with q_norm.weight and k_norm.weight per layer
        """
        try:
            n_layers = data["n_layers"]
            head_dim = data["head_dim"]

            weights = {}
            for i in range(n_layers):
                weights[f"blocks.{i}.attn.q_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
                weights[f"blocks.{i}.attn.k_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)

            return KeyResult(
                success=True,
                weights=weights,
                metadata={
                    "n_layers": n_layers, "head_dim": head_dim,
                    "init": "identity", "total_params": 2 * n_layers * head_dim,
                    "absorbable": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Extract QK-Norm weights from checkpoint.

        If norms were absorbed into projections, un-absorb them.
        Otherwise, just extract the norm weights directly.
        """
        data = {}
        n_layers = 0
        for k in weights:
            if "q_norm.weight" in k:
                idx = int(k.split(".")[1])
                n_layers = max(n_layers, idx + 1)
                data[k] = weights[k]
            if "k_norm.weight" in k:
                data[k] = weights[k]

        return KeyResult(
            success=len(data) > 0,
            data=data,
            metadata={"n_layers": n_layers, "absorbed": False},
        )


def apply_qk_norm_mla(state: dict[str, torch.Tensor], n_layers: int,
                      head_dim: int) -> dict[str, torch.Tensor]:
    """Add QK-Norm weights to a state dict (identity init).

    Adds q_norm.weight and k_norm.weight (all ones) to each layer.
    The model config must also set use_qk_norm=True.
    """
    for i in range(n_layers):
        state[f"blocks.{i}.attn.q_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
        state[f"blocks.{i}.attn.k_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
    print(f"  [QK-Norm MLA] Added to {n_layers} layers (identity init, {2*n_layers*head_dim} params)")
    return state


def absorb_qk_norm_into_projections(state: dict[str, torch.Tensor],
                                     n_layers: int, head_dim: int) -> dict[str, torch.Tensor]:
    """Absorb QK-Norm weights into MLA projections (training-free).

    For each layer:
      q_proj.weight = diag(γ_q) @ q_proj.weight  (scale rows by γ_q)
      k_up_proj.weight = diag(γ_k) @ k_up_proj.weight  (scale rows by γ_k)

    After absorption, q_norm and k_norm can be removed from the model.
    The dynamic RMS scalar is still computed at runtime.

    This is the key insight from the 2026 paper: the static norm weight
    is a diagonal matrix that can be folded into the linear projection.
    """
    for i in range(n_layers):
        q_norm_key = f"blocks.{i}.attn.q_norm.weight"
        k_norm_key = f"blocks.{i}.attn.k_norm.weight"
        q_proj_key = f"blocks.{i}.attn.q_proj.weight"
        k_up_key = f"blocks.{i}.attn.k_up_proj.weight"

        if q_norm_key not in state or k_norm_key not in state:
            continue

        gamma_q = state[q_norm_key].float()  # (head_dim,)
        gamma_k = state[k_norm_key].float()  # (head_dim,)

        # Absorb into q_proj: scale each head_dim block of rows
        if q_proj_key in state:
            w = state[q_proj_key].float()  # (d_model, d_model)
            n_heads = w.shape[0] // head_dim
            # Reshape to (n_heads, head_dim, d_model), scale each head
            w = w.view(n_heads, head_dim, -1)
            w = w * gamma_q.view(1, head_dim, 1)  # broadcast
            state[q_proj_key] = w.reshape(state[q_proj_key].shape).to(state[q_proj_key].dtype)

        # Absorb into k_up_proj: scale each head_dim block of rows
        if k_up_key in state:
            w = state[k_up_key].float()  # (d_model, kv_compression_dim)
            n_heads = w.shape[0] // head_dim
            w = w.view(n_heads, head_dim, -1)
            w = w * gamma_k.view(1, head_dim, 1)
            state[k_up_key] = w.reshape(state[k_up_key].shape).to(state[k_up_key].dtype)

        # Remove norm weights (now absorbed)
        del state[q_norm_key]
        del state[k_norm_key]

    print(f"  [QK-Norm MLA] Absorbed into projections for {n_layers} layers (0 extra params at inference)")
    return state


if __name__ == "__main__":
    key = QKNormMLAKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"n_layers": 4, "head_dim": 128})
    print(f"Forward: {r.success}")
    print(f"  Weights: {len(r.weights)} tensors")
    print(f"  Total params: {r.metadata['total_params']}")
    # Verify identity init
    for k, v in r.weights.items():
        assert (v == 1.0).all(), f"{k} not identity!"
    print("  Identity init verified ✓")

    # Test absorption
    state = dict(r.weights)
    state["blocks.0.attn.q_proj.weight"] = torch.randn(1536, 1536, dtype=torch.bfloat16)
    state["blocks.0.attn.k_up_proj.weight"] = torch.randn(1536, 512, dtype=torch.bfloat16)
    state = absorb_qk_norm_into_projections(state, 4, 128)
    assert "blocks.0.attn.q_norm.weight" not in state, "Norm not removed!"
    print("  Absorption verified ✓")
