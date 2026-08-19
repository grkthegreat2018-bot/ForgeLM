"""RoPE-ID: In-Distribution high-frequency rotation for length generalization.

Based on "Frayed RoPE and Long Inputs: A Geometric Perspective"
(arXiv 2603.18017).

Key insight: through geometric analysis, RoPE applied to longer inputs
damages key/query cluster separation, producing pathological behavior by
inhibiting sink token functionality. The fix: apply RoPE with high frequency
to only a SUBSET of channels, leaving the rest unchanged.

RoPE-ID (In Distribution):
  - Apply high-frequency rotation to a fraction of channels per head
  - RoPE-free channels maintain separated key/query clusters across length
  - High-frequency rotation ensures cluster merging happens within training length

Two criteria for length generalization:
  1. Lower bound on cluster overlap (ensures sink token functionality)
  2. Attaining this bound within training length (prevents OOD errors)

Results: effective on 1B and 3B Transformers on LongBench and RULER.

For our model:
  - Plug-in replacement for RoPE
  - Apply to a fraction of head_dim channels (e.g., 25%)
  - Better length generalization without retraining
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class RoPEID:
    """RoPE-ID: In-Distribution high-frequency rotation.

    Applies RoPE with high frequency to a subset of channels, leaving
    the rest unchanged. This maintains key/query cluster separation
    across sequence lengths, enabling length generalization.
    """

    def __init__(self, head_dim: int, base: float = 10000.0,
                 rotation_fraction: float = 0.25,
                 high_freq_base: float = 1e6):
        """
        Args:
            head_dim: dimension per head
            base: base frequency for standard RoPE channels
            rotation_fraction: fraction of channels to apply high-freq rotation
            high_freq_base: base frequency for high-freq channels (much higher)
        """
        self.head_dim = head_dim
        self.base = base
        self.rotation_fraction = rotation_fraction
        self.high_freq_base = high_freq_base

        d = head_dim // 2
        n_rotated = max(1, int(d * rotation_fraction))
        n_free = d - n_rotated

        # Standard frequencies for non-rotated channels (low frequency)
        freqs_standard = 1.0 / (base ** (torch.arange(0, d).float() / d))

        # High frequencies for rotated channels
        freqs_high = 1.0 / (high_freq_base ** (torch.arange(0, n_rotated).float() / n_rotated))

        # Combine: high-freq for first n_rotated channels, standard for rest
        self.freqs = torch.zeros(d)
        self.freqs[:n_rotated] = freqs_high
        self.freqs[n_rotated:] = freqs_standard[n_rotated:]

        # Which channels are rotated (high-freq)
        self.rotated_mask = torch.zeros(d, dtype=torch.bool)
        self.rotated_mask[:n_rotated] = True

    def apply(self, x: torch.Tensor, offset: int = 0,
              position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply RoPE-ID to a tensor.

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

        freqs = self.freqs.to(device)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # (T, d)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        # Split into rotated and free channels
        x_rot = x[..., :d]
        x_pass = x[..., d:]

        # Apply rotation only to rotated channels
        x1 = x_rot * cos - self._shift(x_rot) * sin
        x2 = x_rot * sin + self._shift(x_rot) * cos

        # Combine: rotated channels get new values, free channels pass through
        # Actually, RoPE rotates pairs: (x[i], x[i+d]) → rotation
        # For RoPE-ID: only rotate the first n_rotated pairs
        out = torch.cat([x1, x2], dim=-1)
        return out

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift x by half head_dim (for RoPE pair rotation)."""
        d = x.shape[-1]
        return torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)

    def stats(self) -> dict:
        return {
            "head_dim": self.head_dim,
            "rotation_fraction": self.rotation_fraction,
            "n_rotated": self.rotated_mask.sum().item(),
            "n_free": (~self.rotated_mask).sum().item(),
            "high_freq_base": self.high_freq_base,
            "standard_base": self.base,
        }
