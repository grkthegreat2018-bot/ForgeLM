"""Shared tensor helpers used across attention / RoPE / cache modules.

Consolidates the duplicated ``_rotate_half`` and ``_repeat_kv`` helpers that
were previously copy-pasted into ``model_loader.py``, the various key modules
under ``forge/keys/``, and ``cacheblend.py``.

All implementations are functionally identical across the original sites;
the versions below preserve the exact behaviour of the originals.
"""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to the front (negated).

    Used by every RoPE variant to compute the rotated component:
        rotate_half(x) = concat(-x2, x1)  where x = concat(x1, x2)

    Args:
        x: tensor whose last dimension is split in half.

    Returns:
        Tensor of the same shape as *x* with the two halves swapped and the
        former second half negated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA broadcast).

    Args:
        x:    ``[B, n_kv, T, hd]`` key or value tensor.
        n_rep: number of query heads per KV head (``n_heads // n_kv_heads``).

    Returns:
        ``[B, n_kv * n_rep, T, hd]`` tensor with each KV head repeated
        ``n_rep`` times.  If ``n_rep == 1`` the input is returned unchanged.
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(
        B, n_kv * n_rep, T, hd)
