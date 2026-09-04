"""AdaLN-zero conditioning (DiT — Diffusion Transformers).

AdaLN-zero is a lossless conditioning injection method:

1. A conditioning embedding (class label, timestep, or signal) is passed
   through an MLP: ``cond -> SiLU -> Linear -> (gamma, beta)``.
2. The final Linear layer is **zero-initialized** so that at init the
   output is all zeros.
3. ``gamma`` has 1 added to it, so at init ``scale = 1`` and ``shift = 0``
   — the adaptive layer norm is the identity.
4. The model gradually *learns* to use the conditioning signal; at
   initialization there is zero quality degradation.

This module provides:

- :class:`AdaLNZeroModulation` — produces ``(scale, shift)`` from a cond
  embedding.  Zero-init final layer → identity at start.
- :class:`AdaLNZero` — full adaptive layer norm that normalizes ``x``
  and applies the scale/shift from the modulation network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNZeroModulation(nn.Module):
    """Produce adaptive layer-norm scale (gamma) and shift (beta) from a
    conditioning embedding.

    Architecture::

        cond -> SiLU -> Linear(cond_dim, 2 * dim) -> (gamma, beta)

    The final ``Linear`` is zero-initialized (weights and biases), so at
    init ``gamma = 0`` and ``beta = 0``.  After adding 1 to ``gamma`` in
    :meth:`forward`, the effective scale is 1 and shift is 0 — identity.

    Args:
        dim: hidden dimension (``d_model``).
        cond_dim: conditioning input dimension.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 2 * dim, bias=True)
        # Zero-init: weights = 0, biases = 0  →  output = 0 at init.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and shift from conditioning.

        Args:
            cond: conditioning embedding, shape ``(B, cond_dim)`` or
                ``(B, 1, cond_dim)``.

        Returns:
            (scale, shift) each of shape ``(B, 1, dim)`` or ``(B, dim)``
            depending on input rank.  ``scale`` has 1 added so zero-init
            gives identity.
        """
        h = self.silu(cond)
        out = self.linear(h)  # (..., 2 * dim)
        gamma, beta = out.chunk(2, dim=-1)
        # Add 1 to gamma so zero-init → scale=1 (identity).
        scale = gamma + 1.0
        shift = beta
        return scale, shift


class AdaLNZero(nn.Module):
    """Full adaptive layer norm with zero-init conditioning.

    Normalizes ``x`` (via RMSNorm or LayerNorm) then applies the
    scale/shift produced by :class:`AdaLNZeroModulation` from ``cond``.

    At initialization (zero-init modulation), this is exactly the
    identity w.r.t. the base normalization — no quality degradation.

    Args:
        dim: hidden dimension (``d_model``).
        cond_dim: conditioning input dimension.
        norm_type: ``"rmsnorm"`` or ``"layernorm"``.
        eps: epsilon for normalization.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        norm_type: str = "rmsnorm",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.norm_type = norm_type
        self.eps = eps
        self.modulation = AdaLNZeroModulation(dim, cond_dim)
        if norm_type == "rmsnorm":
            self.weight = nn.Parameter(torch.ones(dim))
            self.normalized_shape = [dim]
        else:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
            self.normalized_shape = [dim]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "rmsnorm":
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalize ``x`` and apply AdaLN-zero scale/shift.

        Args:
            x: hidden states, shape ``(B, T, dim)``.
            cond: conditioning embedding, shape ``(B, cond_dim)`` or
                ``(B, 1, cond_dim)``.  If ``None``, only normalization
                is applied (backward-compatible fallback).

        Returns:
            Normalized (and modulated) hidden states, same shape as ``x``.
        """
        h = self._normalize(x)
        if cond is None:
            return h
        scale, shift = self.modulation(cond)
        # Broadcast scale/shift to match x's shape.
        # cond is (B, cond_dim) → scale is (B, dim); unsqueeze for T dim.
        if scale.dim() == x.dim() - 1:
            scale = scale.unsqueeze(1)  # (B, 1, dim)
            shift = shift.unsqueeze(1)  # (B, 1, dim)
        return scale * h + shift
