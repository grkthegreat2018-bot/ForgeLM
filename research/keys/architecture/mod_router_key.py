"""Mixture-of-Depths (MoD) — per-token depth routing.

MoD (Raposo et al., 2024) establishes a static compute budget per block:
a router scores each token, the top-k fraction goes through the expensive
attention/FFN compute, and the rest bypass via a cheap residual connection.
This caps total FLOPs while spending compute only on tokens that need deep
reasoning. Combined with MoE ("Staged MoDE") you route for depth first,
then for experts.

This implementation is lossless-at-start:
  - router linear is zero-initialized (all scores equal)
  - keep_fraction=1.0 keeps every token
so a pretrained checkpoint loads with identical outputs until trained.

Inference note: with KV-cache generation, skipped tokens would desynchronize
cache positions, so during inference (use_cache or T<=1) all tokens are kept.
During training the mask is applied with straight-through semantics.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class ModRouter(nn.Module):
    """Per-block token router: keep top-k tokens, bypass the rest.

    Args:
        d_model: hidden size.
        keep_fraction: fraction of tokens allowed through (1.0 = all).
        temperature: router logit sharpening (0 = hard top-k).
    """

    def __init__(self, d_model: int = 2048, keep_fraction: float = 1.0,
                 temperature: float = 0.0):
        super().__init__()
        self.keep_fraction = keep_fraction
        self.temperature = temperature
        # Zero-init => uniform scores => deterministic, lossless at 1.0.
        self.router = nn.Linear(d_model, 1, bias=False)

    def token_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        """Return a (B, T) bool mask of kept tokens, or None if all kept."""
        if self.keep_fraction >= 1.0 or x.shape[1] <= 1:
            return None
        B, T, _ = x.shape
        scores = self.router(x).squeeze(-1)  # (B, T)
        if self.temperature > 0:
            scores = scores / self.temperature
        k = max(1, min(T - 1, math.ceil(T * self.keep_fraction)))
        top = torch.topk(scores, k, dim=-1).indices
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask.scatter_(1, top, True)
        return mask

    def apply(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Gate a block's update: kept tokens get it, others pass through.

        x, update: (B, T, d_model). Returns the GATED UPDATE (caller adds it
        to the residual). With keep_fraction=1.0 the update is returned
        unchanged (lossless).
        """
        mask = self.token_mask(x)
        if mask is None:
            return update
        # STE: forward value is the hard top-k mask, gradients flow through
        # the soft router scores (gate = soft + (hard - soft).detach()).
        hard = mask.unsqueeze(-1).to(update.dtype)  # (B, T, 1)
        soft = self.router(x).sigmoid().to(update.dtype)  # (B, T, 1)
        gate = soft + (hard - soft).detach()
        return gate * update

    def aux_loss(self, x: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """Load-balancing-ish auxiliary loss for the TRUE-SKIP path.

        In the skip path the hard top-k selection is non-differentiable, so
        the router would never receive gradients. This loss pushes the soft
        scores of SKIPPED tokens toward 0 (be confident about skipping) and
        kept tokens toward 1. Attached via the block's `_last_aux_loss` and
        aggregated into the model loss like the MoE aux term.
        """
        scores = self.router(x).squeeze(-1)  # (B, T)
        if mask is None:
            return scores.sigmoid().mean() * 0.0
        kept = mask
        skipped = ~mask
        aux = (scores[skipped].sigmoid().mean()
               + (1.0 - scores[kept].sigmoid()).mean())
        return 0.5 * aux


class ModRouterKey(Key):
    """Adds/removes the MoD router parameter on a state dict.

    forward: inject a zero-init router weight per block (lossless).
    reverse: strip router keys.
    Key class: PARTIAL.
    """

    @property
    def name(self) -> str:
        return "mod_router"

    @property
    def description(self) -> str:
        return "Mixture-of-Depths token router (zero-init, lossless)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if k.startswith("blocks.") and ".ln1.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)
            d_model = None
            for k, v in state.items():
                if k.startswith("blocks.") and ".ln1.weight" in k:
                    d_model = v.shape[0]
                    break
            if d_model is None:
                raise ValueError("cannot infer d_model from state dict")
            for i in range(n_layers):
                state[f"blocks.{i}.mod.router.weight"] = torch.zeros(1, d_model)
            return KeyResult(success=True, weights=state,
                             metadata={"n_layers": n_layers, "d_model": d_model})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(weights)
            removed = 0
            for k in list(state.keys()):
                if ".mod.router.weight" in k:
                    del state[k]
                    removed += 1
            return KeyResult(success=True, weights=state,
                             metadata={"keys_removed": removed})
        except Exception as e:
            return KeyResult(success=False, error=str(e))
