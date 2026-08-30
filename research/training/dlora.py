"""DLoRA (DoRA + LoRA) adapter for warm-starting V8 from V7-8B-B.

Decomposes weights into magnitude + direction (DoRA), then applies LoRA
to the direction component. This allows efficient adaptation of a pre-trained
model with fewer trainable parameters while preserving the magnitude-direction
structure that DoRA showed improves quality.

At init, DLoRA is a no-op: direction=0, lora_B=0, magnitude=1 -> output=0.
For warm-start, copy pre-trained weights into direction, freeze it, and
train only LoRA + magnitude.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DLoRAAdapter(nn.Module):
    """DLoRA (DoRA + LoRA) adapter.

    Weight decomposition: W_eff = magnitude * (direction + scale * (lora_B @ lora_A))

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.
        alpha: LoRA alpha (scale = alpha / rank). Defaults to rank (scale=1.0).

    Attributes:
        magnitude: Scalar magnitude parameter (init=1.0, trainable).
        direction: Base direction matrix (out_features, in_features), zero-init.
        lora_A: LoRA A matrix (rank, in_features), kaiming init.
        lora_B: LoRA B matrix (out_features, rank), zero init (no-op at start).
        scale: LoRA scaling factor = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        if alpha is None:
            alpha = rank
        self.scale = alpha / rank

        # Magnitude (scalar, init=1.0, trainable)
        self.magnitude = nn.Parameter(torch.tensor(1.0))

        # Direction (base weight matrix, zero-init so adapter is no-op at init)
        self.direction = nn.Parameter(torch.zeros(out_features, in_features))

        # LoRA matrices: A is kaiming-init, B is zero-init (no-op at start)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: magnitude * F.linear(x, direction + scale * (lora_B @ lora_A)).

        Supports input shapes (B, in_features) and (B, T, in_features).
        """
        lora_delta = self.scale * (self.lora_B @ self.lora_A)  # (out, in)
        weight = self.direction + lora_delta  # (out, in)
        return self.magnitude * F.linear(x, weight)

    def merge(self) -> torch.Tensor:
        """Merge DLoRA into a single effective weight matrix.

        Returns:
            effective_w (out_features, in_features):
                magnitude * (direction + scale * (lora_B @ lora_A))
        """
        with torch.no_grad():
            lora_delta = self.scale * (self.lora_B @ self.lora_A)
            return self.magnitude * (self.direction + lora_delta)

    def load_direction(self, weight: torch.Tensor) -> None:
        """Load a pre-trained weight matrix into the direction component.

        Used for warm-starting: copy V7 weights into direction, freeze it,
        then train only LoRA + magnitude.
        """
        with torch.no_grad():
            self.direction.copy_(weight)
