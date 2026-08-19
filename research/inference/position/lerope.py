"""LeRoPE: Learnable RoPE frequencies.

Based on "LeRoPE: Learnable RoPE Frequencies Improve Language Modeling"
(arXiv 2607.10134).

Key insight: RoPE frequencies are typically a fixed geometric sequence
specified by a base-frequency hyperparameter. LeRoPE makes these frequencies
LEARNABLE parameters, adding one scalar per frequency band (32 parameters
total for a typical model).

Results:
  - Consistently outperforms RoPE and partial RoPE across all scales
    (52M to 2.5B parameters)
  - RoPE requires 3.4% more compute (FLOPs) to match LeRoPE at 2.5B scale
  - Emergence of a high-norm "positional LeRoPE band" (learned structure)
  - Only 32 extra parameters (negligible memory)

For our model (head_dim=64, 32 frequency bands):
  - 32 learnable scalars per attention layer
  - 6 attention layers × 32 = 192 total learnable parameters
  - Negligible memory, better performance
  - Compatible with existing checkpoints (initialize scalars to 1.0 = base RoPE)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LeRoPE(nn.Module):
    """LeRoPE: Learnable RoPE frequencies.

    Replaces fixed RoPE frequencies with learnable scalars (one per band).
    Initialized to 1.0 (= standard RoPE), so loading an existing checkpoint
    is lossless. Training learns to adjust frequencies for better performance.
    """

    def __init__(self, head_dim: int, base: float = 10000.0,
                 n_frequency_bands: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        d = head_dim // 2
        self.n_bands = n_frequency_bands or d

        # Base frequencies (standard RoPE)
        base_freqs = 1.0 / (base ** (torch.arange(0, d).float() / d))
        self.register_buffer('base_freqs', base_freqs)

        # Learnable frequency scalars (one per band)
        # Initialize to 1.0 → identical to standard RoPE
        self.freq_scalars = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, offset: int = 0,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply LeRoPE to a tensor.

        Args:
            x: (B, n_heads, T, head_dim)
            offset: position offset
            position_ids: explicit position IDs

        Returns:
            rotated: (B, n_heads, T, head_dim)
        """
        B, n_h, T, hd = x.shape
        device = x.device
        d = hd // 2

        if position_ids is not None:
            positions = position_ids.float()
        else:
            positions = torch.arange(offset, offset + T, device=device, dtype=torch.float32)

        # Learnable frequencies: base_freqs * freq_scalars
        freqs = self.base_freqs.to(device) * self.freq_scalars.to(device)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (T, d)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        # Apply rotation
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

    def get_learned_freqs(self) -> torch.Tensor:
        """Return the learned frequency values."""
        return self.base_freqs * self.freq_scalars

    def stats(self) -> dict:
        learned = self.get_learned_freqs()
        return {
            "n_bands": self.n_bands,
            "n_params": self.freq_scalars.numel(),
            "base_freq_range": (self.base_freqs.min().item(), self.base_freqs.max().item()),
            "learned_freq_range": (learned.min().item(), learned.max().item()),
            "scalar_range": (self.freq_scalars.min().item(), self.freq_scalars.max().item()),
            "scalar_mean": self.freq_scalars.mean().item(),
        }


def integrate_lerope_into_model(model: nn.Module, base: float = 10000.0):
    """Replace standard RoPE in a model with LeRoPE.

    Finds all attention layers with RoPE and replaces the rope module
    with LeRoPE. Existing checkpoints load losslessly (scalars init to 1.0).
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'rope') and module.rope is not None:
            # Get head_dim from the existing RoPE
            head_dim = getattr(module.rope, 'head_dim',
                              getattr(module, 'head_dim', 64))
            # Create LeRoPE
            lerope = LeRoPE(head_dim, base=base)
            lerope = lerope.to(next(module.parameters()).device)
            module.rope = lerope
            count += 1

    print(f"  [LeRoPE] Replaced RoPE in {count} attention layers "
          f"({count * 32} learnable params total)")
    return count
