"""Norm-Gated Mixture-of-Depths Key — skip layers by residual-delta norm.

A novel alternative to learned Mixture-of-Depths (MoD) routers: instead of
training a router network to decide which layers to skip, use the residual
stream's delta norm as a free, parameterless gating signal.

Core idea:
    If a transformer block barely changes the residual stream
    (||block(x) - x|| is small), the block is contributing little
    information for that input.  Skip it.

The residual delta for a standard pre-norm block is:

    delta = (x + attn_out + ffn_out) - x = attn_out + ffn_out

Implementation:
    1. Calibration pass — feed representative data through the model,
       measure ||delta|| per layer, and set a per-layer threshold at
       the p-th percentile of observed deltas.
    2. Inference — if the residual-delta norm falls below the layer's
       threshold, skip the block (return x unchanged).

Lossless at init:
    A threshold of 0.0 means *never skip* (the delta norm is always >= 0),
    so the patched model is bit-identical to the original until
    ``calibrate()`` is called.

Classification: TRIVIAL (runtime, no weights).

Usage:
    from research.keys.norm_gated_mod_key import (
        NormGatedMoDKey, apply_norm_gated_mod, calibrate,
    )
    # 1. Patch the model (lossless — threshold defaults to 0.0)
    apply_norm_gated_mod(model)
    # 2. Calibrate thresholds on representative data
    calibrate(model, calibration_data, percentile=50.0)
    # 3. Run inference — layers with small residual deltas are skipped
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import Key, KeyClass, KeyResult


# ---------------------------------------------------------------------------
# Key class
# ---------------------------------------------------------------------------
class NormGatedMoDKey(Key):
    """Norm-Gated Mixture-of-Depths key.

    No weights are learned — the "router" is the residual-delta norm,
    which is computed for free from the existing forward pass.
    """

    @property
    def name(self) -> str:
        return "norm_gated_mod"

    @property
    def description(self) -> str:
        return (
            "Skip transformer layers whose residual-delta norm falls below "
            "a per-layer threshold. No router parameters; lossless at init."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL  # runtime only, no weights

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """No weights to produce — this key is a runtime patch."""
        return KeyResult(success=True, weights={}, metadata={"type": "runtime_patch"})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Nothing to extract — no weights exist."""
        return KeyResult(success=True, data={})


# ---------------------------------------------------------------------------
# Patching utilities
# ---------------------------------------------------------------------------
def _make_gated_forward(original_forward):
    """Create a patched forward that checks residual-delta norm.

    The patched forward mirrors the original ``ModularBlock.forward`` logic
    but inserts a norm-gate check after computing the block output.  If
    ``||delta|| < skip_threshold`` the block is skipped (x returned
    unchanged) and the skip counter is incremented.

    The KV cache (``present``) is still returned even when the block is
    skipped, because attention has already been computed and the cache
    must remain consistent for subsequent tokens.
    """

    def gated_forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # --- Activation checkpointing path (training only) ----------------
        if (
            self.training
            and not use_cache
            and getattr(self, "_gradient_checkpointing", False)
        ):
            # During training we never skip — just delegate to the original.
            return original_forward(self, x, past_key_value, use_cache, **kwargs)

        # --- Inference path with norm-gate --------------------------------
        threshold = self.skip_threshold

        # Fast path: threshold == 0 → never skip → identical to original.
        if threshold <= 0.0:
            return original_forward(self, x, past_key_value, use_cache, **kwargs)

        # Compute attention (always needed for KV cache consistency).
        attn_out, present = self.attn(
            self.ln1(x), past_key_value=past_key_value, use_cache=use_cache
        )

        # Compute FFN output.
        x_after_attn = x + attn_out
        ffn_out = self.ffn(self.ln2(x_after_attn))
        if isinstance(ffn_out, tuple):  # MoE returns (output, aux_loss)
            ffn_out = ffn_out[0]

        # Residual delta for the whole block = attn_out + ffn_out.
        delta = attn_out + ffn_out
        delta_norm = delta.norm(dim=-1).mean().item()

        if delta_norm < threshold:
            # Skip: return the *original* x (no attn_out, no ffn_out).
            # ``present`` is still returned so the KV cache stays consistent.
            self._norm_gated_skip_count += 1
            self._norm_gated_last_delta = delta_norm
            return x, present

        # Normal path: apply both residuals.
        self._norm_gated_last_delta = delta_norm
        return x_after_attn + ffn_out, present

    return gated_forward


def apply_norm_gated_mod(model, percentile: float = 0.0) -> nn.Module:
    """Patch every transformer block with norm-gated skipping.

    After calling this, each block in ``model.blocks`` gains:

    * ``skip_threshold`` (float) — per-layer delta-norm threshold.
      Defaults to ``0.0`` which means *never skip* (bit-identical to
      the unpatched model).
    * ``_norm_gated_skip_count`` (int) — running count of skipped
      forwards (useful for profiling).
    * ``_norm_gated_last_delta`` (float) — delta norm from the most
      recent forward.

    Parameters
    ----------
    model : nn.Module
        Model with a ``blocks`` attribute (``nn.ModuleList`` of
        ``ModularBlock``).
    percentile : float, optional
        Initial percentile for threshold.  ``0.0`` (default) keeps the
        model lossless.  Use ``calibrate()`` to set data-driven
        thresholds.

    Returns
    -------
    nn.Module
        The same model (patched in-place).
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError(
            "Model has no 'blocks' attribute — apply_norm_gated_mod "
            "expects a ModularModel-style transformer."
        )

    for i, block in enumerate(blocks):
        # Store the original forward so we can delegate when threshold == 0.
        # Store as unbound function to avoid double-self when calling.
        if not hasattr(block, "_norm_gated_original_forward"):
            # Get the unbound forward from the class (not the bound method)
            original = type(block).forward
            block._norm_gated_original_forward = original

        # Per-layer threshold (0.0 = never skip = lossless).
        block.skip_threshold = 0.0  # always start lossless

        # Profiling counters.
        block._norm_gated_skip_count = 0
        block._norm_gated_last_delta = 0.0

        # Patch the forward method.
        block.forward = _make_gated_forward(block._norm_gated_original_forward).__get__(
            block, type(block)
        )

    print(
        f"  [NormGatedMoD] Patched {len(blocks)} blocks "
        f"(threshold=0.0, lossless). Call calibrate() to enable skipping."
    )
    return model


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate(
    model,
    data: torch.Tensor,
    percentile: float = 50.0,
    batch_size: int = 4,
) -> dict[int, float]:
    """Run a calibration pass and set per-layer skip thresholds.

    Feeds ``data`` through the model, measures the residual-delta norm
    ``||block(x) - x||`` for every layer on every sample, and sets each
    layer's ``skip_threshold`` to the ``percentile``-th percentile of
    observed deltas.

    After calibration, layers will skip inputs whose delta norm falls
    below the threshold — i.e. the bottom ``percentile``% of deltas.

    Parameters
    ----------
    model : nn.Module
        A model already patched with ``apply_norm_gated_mod()`` (if not
        patched, this function patches it first).
    data : torch.Tensor
        Calibration token IDs of shape ``(N, seq_len)``.
    percentile : float, optional
        Percentile of delta norms to use as the skip threshold.
        ``50.0`` → skip ~50% of inputs (aggressive).
        ``10.0`` → skip only the ~10% smallest deltas (conservative).
        ``0.0``  → never skip (lossless).
    batch_size : int, optional
        Mini-batch size for the calibration forward passes.

    Returns
    -------
    dict
        Mapping ``{layer_index: threshold}`` for inspection/logging.
    """
    # Ensure the model is patched.
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Model has no 'blocks' attribute.")

    if not hasattr(blocks[0], "_norm_gated_original_forward"):
        apply_norm_gated_mod(model)

    n_layers = len(blocks)
    device = next(model.parameters()).device
    data = data.to(device)

    # Accumulate delta norms per layer.
    delta_norms: list[list[float]] = [[] for _ in range(n_layers)]

    # Temporarily set thresholds to 0 so nothing is skipped during calibration.
    original_thresholds = [blk.skip_threshold for blk in blocks]
    for blk in blocks:
        blk.skip_threshold = 0.0

    model.eval()

    n_samples = data.shape[0]
    for start in range(0, n_samples, batch_size):
        batch = data[start : start + batch_size]
        # Run a forward pass with hooks to capture residual deltas.
        _calibration_forward(model, batch, blocks, delta_norms)

    # Restore original thresholds (will be overwritten below).
    for i, blk in enumerate(blocks):
        blk.skip_threshold = original_thresholds[i]

    # Compute per-layer thresholds.
    thresholds: dict[int, float] = {}
    for i, norms in enumerate(delta_norms):
        if len(norms) == 0:
            thresholds[i] = 0.0
            continue
        tensor = torch.tensor(norms)
        thresh = torch.quantile(tensor, percentile / 100.0).item()
        blocks[i].skip_threshold = thresh
        thresholds[i] = thresh

    # Print summary.
    skipped_pct = percentile
    print(f"  [NormGatedMoD] Calibration complete ({len(delta_norms[0])} samples):")
    print(f"    percentile={percentile:.1f}  (~{skipped_pct:.0f}% of inputs will be skipped)")
    for i, t in thresholds.items():
        print(f"    layer {i:3d}: threshold={t:.6f}")

    return thresholds


def _calibration_forward(
    model,
    batch: torch.Tensor,
    blocks: nn.ModuleList,
    delta_norms: list[list[float]],
) -> None:
    """Run one forward batch and record per-layer residual-delta norms.

    Instead of relying on hooks (which are tricky with the patched forward),
    this manually walks the blocks and computes the delta norm for each.
    """
    x = model.embed(batch)
    for i, block in enumerate(blocks):
        # Compute the block output.
        x_out, _ = block._norm_gated_original_forward(
            x, past_key_value=None, use_cache=False
        )
        # delta = x_out - x  (== attn_out + ffn_out for pre-norm blocks).
        delta = x_out - x
        # Mean L2 norm across the batch and sequence dimensions.
        norm_val = delta.norm(dim=-1).mean().item()
        delta_norms[i].append(norm_val)
        x = x_out


# ---------------------------------------------------------------------------
# Profiling / introspection
# ---------------------------------------------------------------------------
def get_skip_stats(model) -> dict[int, dict[str, int]]:
    """Return per-layer skip statistics after inference.

    Returns
    -------
    dict
        ``{layer_idx: {"skips": int, "last_delta": float, "threshold": float}}``
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Model has no 'blocks' attribute.")

    stats: dict[int, dict[str, int]] = {}
    for i, blk in enumerate(blocks):
        stats[i] = {
            "skips": getattr(blk, "_norm_gated_skip_count", 0),
            "last_delta": getattr(blk, "_norm_gated_last_delta", 0.0),
            "threshold": getattr(blk, "skip_threshold", 0.0),
        }
    return stats


def reset_skip_stats(model) -> None:
    """Reset all per-layer skip counters to zero."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Model has no 'blocks' attribute.")
    for blk in blocks:
        blk._norm_gated_skip_count = 0


def remove_norm_gated_mod(model) -> nn.Module:
    """Restore original forward methods (un-patch the model).

    Removes the norm-gate and restores each block's original forward.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Model has no 'blocks' attribute.")

    for block in blocks:
        if hasattr(block, "_norm_gated_original_forward"):
            block.forward = block._norm_gated_original_forward
            del block._norm_gated_original_forward
            if hasattr(block, "skip_threshold"):
                del block.skip_threshold
            if hasattr(block, "_norm_gated_skip_count"):
                del block._norm_gated_skip_count
            if hasattr(block, "_norm_gated_last_delta"):
                del block._norm_gated_last_delta

    print("  [NormGatedMoD] Removed norm-gate patches (model restored).")
    return model
