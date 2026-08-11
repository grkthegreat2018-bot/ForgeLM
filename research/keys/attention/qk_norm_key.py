"""QK-Norm Key — add RMSNorm to Q and K projections.

QK-Norm (used in Gemma 2/3, QK-Norm paper) normalizes Q and K vectors
before attention. This stabilizes training by preventing attention score
explosion in deep models.

As a Key: we add RMSNorm layers with identity initialization (weight=1, eps=1e-6).
At init, QK-Norm is a no-op (RMSNorm with weight=1 preserves the input).
The model can then be fine-tuned to learn the optimal norm scales.

This is a TRIVIAL key — no weight transformation needed, just architecture
addition with identity init.

Usage:
    from research.keys.qk_norm_key import QKNormKey, apply_qk_norm
    # Add QK-Norm to model
    apply_qk_norm(model)
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class QKNorm(nn.Module):
    """RMSNorm for Q/K vectors. Identity at init (weight=1).

    Uses F.rms_norm for a single fused kernel instead of 3 separate ops.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = [dim]

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


class QKNormKey(Key):
    """QK-Norm key — add RMSNorm to Q and K."""

    @property
    def name(self) -> str:
        return "qk_norm"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL  # Identity init, no weight transform

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """QK-Norm adds new parameters (identity init), doesn't transform existing."""
        return KeyResult(success=True, weights={}, metadata={"note": "QK-Norm is identity init"})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data={}, metadata={"note": "QK-Norm is identity init"})


def apply_qk_norm(model):
    """Add QK-Norm to all attention layers in the model.

    Modifies the model in-place. Each attention layer gets:
      - q_norm: RMSNorm(head_dim) after Q projection
      - k_norm: RMSNorm(head_dim) after K projection

    Both initialized to identity (weight=1).
    """
    n_added = 0
    for block in model.blocks:
        attn = block.attn

        # Get head_dim from attention module
        if hasattr(attn, 'head_dim'):
            head_dim = attn.head_dim
        elif hasattr(attn, 'n_heads') and hasattr(attn, 'q_proj'):
            head_dim = attn.q_proj.out_features // attn.n_heads
        else:
            continue

        # Add QK-Norm (identity init)
        attn.q_norm = QKNorm(head_dim).to(next(attn.parameters()).device)
        attn.k_norm = QKNorm(head_dim).to(next(attn.parameters()).device)
        n_added += 1

    print(f"  [QK-Norm] Added to {n_added} attention layers (identity init)")
    return n_added


def apply_qk_norm_to_state(state: dict[str, torch.Tensor], n_layers: int,
                           n_heads: int, head_dim: int) -> dict[str, torch.Tensor]:
    """Add QK-Norm weights to a state dict (identity init).

    Adds q_norm.weight and k_norm.weight (all ones) to each layer.
    """
    for i in range(n_layers):
        state[f"blocks.{i}.attn.q_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
        state[f"blocks.{i}.attn.k_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
    return state
