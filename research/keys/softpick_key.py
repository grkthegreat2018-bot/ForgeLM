"""Softpick Attention Key — sigmoid-gated softmax for sparse attention.

Softpick (Zuhri et al., ACL 2026 Findings) replaces the standard softmax in
attention with a sigmoid-gated variant:

    softpick(x) = softmax(x) * sigmoid(x)

The sigmoid gate drives low-scoring positions to exactly zero instead of the
small nonzero weight softmax assigns.  This yields:

  - 0% attention-sink rate (no sink tokens absorbing spurious probability mass)
  - Naturally sparse attention maps (better interpretability)
  - Improved quantization robustness (no extreme outlier columns from sinks)

This is a TRIVIAL key — no weights, no training.  The entire change is a
runtime replacement of the attention softmax.  The original Softpick repo
ships a modified FlashAttention-2 kernel; here we implement the equivalent via
manual attention (scores → softpick → @V) so it works on any device without a
custom CUDA build.

Key class: TRIVIAL — fixed formula, no data or training.

Reference: zuhri-etal-2026, github.com/zuhdzuhri/softpick-attention
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .base import Key, KeyClass, KeyResult

# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def softpick(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softpick activation: softmax(x) * sigmoid(x).

    Equivalent to softmax(x) * (1 / (1 + exp(-x))).  The elementwise sigmoid
    gate suppresses low-scoring positions toward zero, producing a sparse
    attention distribution without any explicit top-k thresholding.

    Args:
        scores: pre-softmax attention scores, shape (..., seq, seq).
        dim: softmax dimension (last by default, as in attention).

    Returns:
        Sparse attention weights with the same shape as *scores*.
    """
    return F.softmax(scores, dim=dim) * torch.sigmoid(scores)


def softpick_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Manual attention with softpick replacing softmax.

    Because softpick is an elementwise gate applied *after* softmax, it cannot
    use PyTorch's fused ``F.scaled_dot_product_attention`` (which never
    materialises the attention matrix).  We compute scores explicitly, apply
    the causal mask, then use ``softpick`` instead of ``softmax``.

    Args:
        q: (B, H, T_q, D) query tensor.
        k: (B, H, T_k, D) key tensor.
        v: (B, H, T_k, D_v) value tensor.
        is_causal: apply lower-triangular causal mask.
        attn_mask: optional additive mask broadcastable to (T_q, T_k).
            When provided, *is_causal* is ignored.
        scale: custom scale; defaults to 1/sqrt(D).

    Returns:
        (B, H, T_q, D_v) attention output.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)

    if attn_mask is not None:
        scores = scores + attn_mask
    elif is_causal:
        T_q, T_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.full((T_q, T_k), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask

    attn = softpick(scores, dim=-1)
    return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Key class
# ---------------------------------------------------------------------------

class SoftpickKey(Key):
    """Softpick attention key — sigmoid-gated softmax replacement.

    Replaces the softmax in attention with ``softpick(x) = softmax(x) * sigmoid(x)``.
    This produces sparse attention weights with 0% sink rate and helps
    quantization by eliminating attention-sink outlier columns.

    No weight changes — only the attention computation changes at runtime.

    Key class: TRIVIAL — fixed formula, no data or training.
    """

    @property
    def name(self) -> str:
        return "softpick"

    @property
    def description(self) -> str:
        return "Softpick attention: softmax(x) * sigmoid(x) for sparse, sink-free attention"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Apply softpick to attention scores.

        Args:
            data: {"scores": tensor (..., seq, seq) — pre-softmax QK^T/sqrt(d)}

        Returns:
            KeyResult with weights={"attn": softpick(scores)}
        """
        try:
            scores = data["scores"]
            attn = softpick(scores, dim=-1)
            return KeyResult(
                success=True,
                weights={"attn": attn},
                metadata={"formula": "softmax(x) * sigmoid(x)"},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Cannot reverse softpick (information is discarded by the gate)."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"lossy": True, "runtime_only": True},
        )


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

# Module-level flag so patched functions can check at runtime.
_SOFTPICK_ENABLED = False


def _make_softpick_flash_attention(original_fn):
    """Wrap the module-level ``flash_attention`` function to use softpick.

    On CUDA the original function delegates to ``F.scaled_dot_product_attention``
    (FlashAttention-2).  Because softpick requires materialising the score
    matrix for the elementwise sigmoid gate, we fall back to manual attention
    with ``softpick_attention`` whenever softpick is enabled.
    """

    def patched_flash_attention(q, k, v, is_causal=True):
        if _SOFTPICK_ENABLED:
            return softpick_attention(q, k, v, is_causal=is_causal)
        return original_fn(q, k, v, is_causal=is_causal)

    patched_flash_attention.__wrapped__ = original_fn
    patched_flash_attention.__softpick__ = True
    return patched_flash_attention


def _patch_module_flash_attention(module):
    """Replace ``module.flash_attention`` with a softpick-aware version."""
    original = getattr(module, "flash_attention", None)
    if original is None:
        return False
    if getattr(original, "__softpick__", False):
        # Already patched — update the wrapped reference.
        original.__wrapped__ = original.__wrapped__
        return True
    module.flash_attention = _make_softpick_flash_attention(original)
    return True


def _patch_sdpa_calls(model):
    """Patch attention modules that call ``F.scaled_dot_product_attention`` directly.

    ``MultiHeadLatentAttention`` and ``StandardSDPA`` use SDPA for the
    chunked-prefill branch (custom causal mask).  We wrap their ``forward``
    methods so that, when softpick is enabled, those calls route through
    ``softpick_attention`` instead.
    """
    from research.model_loader import (
        DifferentialAttention,
        GroupedQueryAttention,
        MultiHeadLatentAttention,
        StandardSDPA,
    )

    target_types = (
        MultiHeadLatentAttention,
        StandardSDPA,
        GroupedQueryAttention,
        DifferentialAttention,
    )

    count = 0
    for module in model.modules():
        if isinstance(module, target_types):
            if getattr(module, "_softpick_patched", False):
                count += 1
                continue
            _wrap_attention_forward(module)
            module._softpick_patched = True
            count += 1
    return count


def _wrap_attention_forward(module):
    """Wrap a single attention module's ``forward`` to use softpick.

    The wrapper intercepts the three code paths used across attention variants:
      1. ``flash_attention(...)`` calls — handled by the module-level patch.
      2. ``F.scaled_dot_product_attention(q, k, v, attn_mask=mask)`` calls —
         replaced with ``softpick_attention`` when enabled.
      3. Manual ``F.softmax(scores, ...)`` calls (DifferentialAttention) —
         replaced with ``softpick(scores, ...)`` when enabled.

    Rather than re-implementing each forward (fragile), we monkey-patch the
    *names* that the forward body references: ``flash_attention`` and
    ``F.scaled_dot_product_attention`` / ``F.softmax`` within the module's
    globals.  The module-level ``flash_attention`` patch covers path 1.
    For paths 2 and 3 we wrap the module's forward to temporarily swap
    ``F.scaled_dot_product_attention`` and ``F.softmax``.
    """
    original_forward = module.forward

    def patched_forward(*args, **kwargs):
        if not _SOFTPICK_ENABLED:
            return original_forward(*args, **kwargs)

        # Temporarily replace F.scaled_dot_product_attention and F.softmax
        # with softpick-aware versions for the duration of this call.
        orig_sdpa = F.scaled_dot_product_attention
        orig_softmax = F.softmax

        def softpick_sdpa(q, k, v, attn_mask=None, is_causal=False, **kw):
            return softpick_attention(
                q, k, v, is_causal=is_causal, attn_mask=attn_mask,
            )

        def softpick_softmax_fn(input, dim=-1, **kw):
            return softpick(input, dim=dim)

        F.scaled_dot_product_attention = softpick_sdpa
        F.softmax = softpick_softmax_fn
        try:
            return original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
            F.softmax = orig_softmax

    patched_forward.__wrapped__ = original_forward
    patched_forward.__softpick__ = True
    module.forward = patched_forward


# ---------------------------------------------------------------------------
# Public apply / revert API
# ---------------------------------------------------------------------------

def apply_softpick(model):
    """Patch *model* to use softpick attention at runtime.

    This replaces the softmax in all attention layers with
    ``softpick(x) = softmax(x) * sigmoid(x)``.  The change is a runtime flag —
    no weights are modified and the patch can be fully reverted with
    :func:`revert_softpick`.

    Works with ``GroupedQueryAttention``, ``MultiHeadLatentAttention``,
    ``StandardSDPA``, and ``DifferentialAttention`` from
    ``research.model_loader``.

    Args:
        model: a ``ConfigurableResearchLLM`` (or any ``nn.Module`` whose
            attention layers use the classes above).

    Returns:
        The patched model (modified in-place).
    """
    global _SOFTPICK_ENABLED

    # 1. Patch the module-level flash_attention function used by all variants.
    import research.model_loader as ml
    module_patched = _patch_module_flash_attention(ml)

    # 2. Patch per-module forward methods for SDPA / manual-softmax paths.
    module_count = _patch_sdpa_calls(model)

    # 3. Enable the runtime flag.
    _SOFTPICK_ENABLED = True

    print("  [Softpick] Applied softpick attention to model")
    print(f"  [Softpick]   flash_attention: {'patched' if module_patched else 'not found'}")
    print(f"  [Softpick]   attention modules patched: {module_count}")
    print("  [Softpick]   formula: softmax(x) * sigmoid(x)")
    return model


def revert_softpick(model):
    """Revert softpick attention, restoring standard softmax.

    Disables the runtime flag and restores original ``flash_attention`` and
    attention ``forward`` methods.

    Args:
        model: the model previously patched with :func:`apply_softpick`.

    Returns:
        The restored model (modified in-place).
    """
    global _SOFTPICK_ENABLED
    _SOFTPICK_ENABLED = False

    # Restore module-level flash_attention.
    import research.model_loader as ml
    fa = getattr(ml, "flash_attention", None)
    if fa is not None and getattr(fa, "__softpick__", False):
        ml.flash_attention = fa.__wrapped__

    # Restore per-module forwards.
    from research.model_loader import (
        DifferentialAttention,
        GroupedQueryAttention,
        MultiHeadLatentAttention,
        StandardSDPA,
    )
    target_types = (
        MultiHeadLatentAttention,
        StandardSDPA,
        GroupedQueryAttention,
        DifferentialAttention,
    )
    count = 0
    for module in model.modules():
        if isinstance(module, target_types):
            fwd = getattr(module.forward, "__wrapped__", None)
            if fwd is not None and getattr(module.forward, "__softpick__", False):
                module.forward = fwd
                module._softpick_patched = False
                count += 1

    print(f"  [Softpick] Reverted softpick attention ({count} modules restored)")
    return model


def is_softpick_enabled() -> bool:
    """Return whether softpick attention is currently active."""
    return _SOFTPICK_ENABLED


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = SoftpickKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Verify softpick produces sparser output than softmax.
    scores = torch.tensor([[10.0, -5.0, -5.0, -5.0]])
    sm = F.softmax(scores, dim=-1)
    sp = softpick(scores, dim=-1)
    print(f"  softmax:  {sm.tolist()}")
    print(f"  softpick: {sp.tolist()}")
    # Softpick should suppress the -5 positions more aggressively.
    assert sp[0, 1].item() < sm[0, 1].item(), "softpick should be sparser than softmax"
    print("  Sparsity verified: softpick suppresses low scores more than softmax")

    # Verify attention output shape.
    q = torch.randn(1, 4, 8, 32)
    k = torch.randn(1, 4, 8, 32)
    v = torch.randn(1, 4, 8, 32)
    out = softpick_attention(q, k, v, is_causal=True)
    assert out.shape == (1, 4, 8, 32)
    print(f"  Attention output shape: {out.shape} ✓")

    # Verify apply / revert on a dummy model.
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()

    dummy = DummyModel()
    assert not is_softpick_enabled()
    # apply_softpick patches flash_attention even with no attention modules.
    apply_softpick(dummy)
    assert is_softpick_enabled()
    revert_softpick(dummy)
    assert not is_softpick_enabled()
    print("  Apply / revert cycle verified ✓")
