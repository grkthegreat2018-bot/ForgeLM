"""Per-Query Adaptive Temperature Key — learnable per-query attention scale.

Standard attention uses a fixed temperature T = sqrt(d_head), giving the
familiar scale 1/sqrt(d_head).  This key replaces that constant with a
*query-dependent* temperature:

    T(Q) = softplus(w · Q + b)          # per head, per token

At initialisation w = 0 and b = sqrt(d_head), so

    T(Q) = softplus(0 + sqrt(d)) ≈ sqrt(d)      # (softplus(x) ≈ x for x ≫ 1)

which is exactly the standard attention scale — the model is **lossless at
init**.  Fine-tuning w lets each head specialise its temperature to the
content of the query, sharpening or softening attention as needed.

This is classified as **TRIVIAL**: the key itself introduces only two tiny
parameter tensors (w, b) that are identity-initialised, so there is nothing
meaningful to convert between data and weights — it is a runtime
configuration / patch.

Usage::

    from research.keys.per_query_temp_key import (
        PerQueryTempKey,
        apply_per_query_temp,
    )
    apply_per_query_temp(model)          # patch every attention layer
    # … train normally; w will specialise per head …

References:
    - docs/research/KEY_MAPPING_MASTER.md  (F5 — Per-Query Adaptive Temperature)
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Key, KeyClass, KeyResult

# ---------------------------------------------------------------------------
# Attention layer types that this key knows how to patch.
# ---------------------------------------------------------------------------
# We import lazily inside _is_supported_attn so that this module can be
# imported without pulling in the full model_loader dependency graph.
_SUPPORTED_ATTN_CLASS_NAMES = {
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "StandardSDPA",
    "DifferentialAttention",
}


def _is_supported_attn(layer: nn.Module) -> bool:
    """Return *True* if *layer* is an attention module we can patch."""
    return type(layer).__name__ in _SUPPORTED_ATTN_CLASS_NAMES


# ---------------------------------------------------------------------------
# Core module: per-query adaptive temperature
# ---------------------------------------------------------------------------
class PerQueryTemp(nn.Module):
    """Compute a query-dependent attention temperature per head.

    T(Q) = softplus(w · Q + b)

    where ``w`` has shape ``(n_heads, head_dim)`` and ``b`` has shape
    ``(n_heads,)``.

    At init ``w = 0`` and ``b = sqrt(head_dim)``, giving
    ``T = softplus(sqrt(d)) ≈ sqrt(d)`` — the standard attention scale.

    The module returns the **temperature** T (not the scale).  Callers
    should divide attention scores by T (equivalently, multiply by 1/T).
    """

    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        # w: (n_heads, head_dim) — zero-init → lossless at init
        self.w = nn.Parameter(torch.zeros(n_heads, head_dim))
        # b: (n_heads,) — sqrt(d) so that softplus(sqrt(d)) ≈ sqrt(d)
        sqrt_d = math.sqrt(head_dim)
        self.b = nn.Parameter(torch.full((n_heads,), sqrt_d))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute per-query temperature.

        Args:
            q: Query tensor of shape ``(B, n_heads, T, head_dim)``.

        Returns:
            Temperature tensor of shape ``(B, n_heads, T, 1)``.
        """
        # w · Q  →  (B, n_heads, T)
        #   w: (n_heads, head_dim)  broadcast over batch & seq
        wq = (q * self.w.unsqueeze(0).unsqueeze(2)).sum(dim=-1)  # (B, n_heads, T)
        # + b  →  (B, n_heads, T)
        logits = wq + self.b.unsqueeze(0).unsqueeze(-1)          # (B, n_heads, T)
        # softplus → temperature (always positive)
        temp = F.softplus(logits)                                # (B, n_heads, T)
        # Cast to query dtype to avoid mixed-dtype matmul errors
        temp = temp.to(q.dtype)
        # (B, n_heads, T, 1) for broadcasting over key positions
        return temp.unsqueeze(-1)

    @property
    def is_identity(self) -> bool:
        """True when w is still all-zeros (lossless / standard attention)."""
        return bool((self.w.abs().max() < 1e-8).item())


# ---------------------------------------------------------------------------
# Manual attention with per-query temperature
# ---------------------------------------------------------------------------
def _adaptive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    temp: torch.Tensor,
    is_causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention with a *per-query* temperature.

    Args:
        q: ``(B, H, Tq, D)``
        k: ``(B, H, Tk, D)``
        v: ``(B, H, Tk, Dv)``
        temp: ``(B, H, Tq, 1)`` — temperature per query position.
        is_causal: Apply causal mask (prefill with no explicit mask).
        attn_mask: Optional additive mask ``(Tq, Tk)`` or ``(B, 1, Tq, Tk)``.

    Returns:
        ``(B, H, Tq, Dv)``
    """
    # scores: (B, H, Tq, Tk)
    scores = torch.matmul(q, k.transpose(-2, -1))
    # Divide by per-query temperature (broadcast over Tk).
    # Keep same dtype as scores (bf16) to avoid upcast.
    scores = scores / temp.to(scores.dtype)

    if attn_mask is not None:
        # attn_mask is additive (-inf where masked).
        scores = scores + attn_mask
    elif is_causal:
        Tq, Tk = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.full((Tq, Tk), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Key class
# ---------------------------------------------------------------------------
class PerQueryTempKey(Key):
    """Per-Query Adaptive Temperature key.

    Classified as **TRIVIAL**: the key introduces only two tiny
    identity-initialised parameter tensors (``w``, ``b``).  There is no
    meaningful data ↔ weight conversion — it is a runtime patch that makes
    the attention temperature query-dependent.
    """

    def __init__(self, n_heads: int = 0, head_dim: int = 0):
        # Stored for metadata; the actual module is created during apply.
        self._n_heads = n_heads
        self._head_dim = head_dim

    @property
    def name(self) -> str:
        return "per_query_temp"

    @property
    def description(self) -> str:
        return (
            "Per-Query Adaptive Temperature: T(Q)=softplus(w·Q+b), "
            "lossless at init (w=0, b=sqrt(d))."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL  # minimal identity-init weights, runtime patch

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """No-op — the key is a runtime configuration, not a weight generator."""
        return KeyResult(
            success=True,
            weights={},
            metadata={
                "n_heads": self._n_heads,
                "head_dim": self._head_dim,
                "init_w": "zeros",
                "init_b": "sqrt(head_dim)",
            },
        )

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """No-op — nothing to extract (identity-init parameters)."""
        return KeyResult(success=True, data={})


# ---------------------------------------------------------------------------
# Patching utilities
# ---------------------------------------------------------------------------
def _make_patched_forward(original_forward, temp_module: PerQueryTemp):
    """Create a patched ``forward`` that uses adaptive temperature.

    The patched forward intercepts *after* q/k/v projections and RoPE but
    *before* the SDPA call, replacing the fixed-scale attention with the
    per-query temperature.
    """

    def patched_forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        cls_name = type(self).__name__

        # ---- project q / k / v (replicate the original logic) ----
        if cls_name == "GroupedQueryAttention":
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

            past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            q = self.rope(q, offset=past_len)
            k = self.rope(k, offset=past_len)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)
            new_kv = (k, v) if use_cache else None

            k = self._repeat_kv(k)
            v = self._repeat_kv(v)

            temp = temp_module(q)  # (B, n_heads, T, 1)
            total_len = k.shape[-2]
            if T == 1 and total_len > 1:
                out = _adaptive_attention(q, k, v, temp, is_causal=False)
            else:
                out = _adaptive_attention(q, k, v, temp, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out), new_kv

        elif cls_name == "MultiHeadLatentAttention":
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            c_kv = self.kv_down_proj(x)
            k = self.k_up_proj(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_up_proj(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            if self.use_qk_norm and not getattr(self, "_qk_norm_identity", True):
                q = self.q_norm(q)
                k = self.k_norm(k)

            past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            q = self.rope(q, offset=past_len)
            k = self.rope(k, offset=past_len)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)

            temp = temp_module(q)
            total_len = k.shape[-2]
            if T == 1 and total_len > 1:
                out = _adaptive_attention(q, k, v, temp, is_causal=False)
            elif past_len == 0 and T == total_len:
                out = _adaptive_attention(q, k, v, temp, is_causal=True)
            else:
                from research.model_loader import _causal_mask
                mask = _causal_mask(T, total_len, past_len, x.device)
                out = _adaptive_attention(q, k, v, temp, is_causal=False, attn_mask=mask)

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.out_proj(out)
            if use_cache:
                return out, (k, v)
            return out, None

        elif cls_name == "StandardSDPA":
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            q = self.rope(q, offset=past_len)
            k = self.rope(k, offset=past_len)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)

            temp = temp_module(q)
            total_len = k.shape[-2]
            if T == 1 and total_len > 1:
                out = _adaptive_attention(q, k, v, temp, is_causal=False)
            elif past_len == 0 and T == total_len:
                out = _adaptive_attention(q, k, v, temp, is_causal=True)
            else:
                from research.model_loader import _causal_mask
                mask = _causal_mask(T, total_len, past_len, x.device)
                out = _adaptive_attention(q, k, v, temp, is_causal=False, attn_mask=mask)

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.out_proj(out)
            if use_cache:
                return out, (k, v)
            return out, None

        else:
            # Fallback: unsupported layer — use original forward.
            return original_forward(x, past_key_value=past_key_value, use_cache=use_cache)

    return patched_forward


def apply_per_query_temp(model: nn.Module) -> nn.Module:
    """Patch every supported attention layer in *model* with per-query
    adaptive temperature.

    For each ``GroupedQueryAttention``, ``MultiHeadLatentAttention``, or
    ``StandardSDPA`` layer found in the model:

    1.  A :class:`PerQueryTemp` module is created and registered as a
        sub-module (``layer.per_query_temp``) so its parameters are
        trainable and serialised with the model.
    2.  The layer's ``forward`` is replaced with a version that computes
        ``T(Q) = softplus(w·Q + b)`` and uses it as the attention
        temperature instead of the fixed ``1/sqrt(head_dim)``.

    At init the patch is **lossless**: ``w = 0`` and ``b = sqrt(head_dim)``
    so ``T ≈ sqrt(d)`` and attention is identical to the standard scale.

    Args:
        model: The model to patch (modified in-place).

    Returns:
        The patched model (same object, for chaining).
    """
    patched_count = 0

    for module in model.modules():
        if not _is_supported_attn(module):
            continue

        n_heads = module.n_heads
        head_dim = module.head_dim

        # Create and register the temperature module.
        temp_module = PerQueryTemp(n_heads=n_heads, head_dim=head_dim)
        # Move to same device as the parent module
        try:
            device = next(module.parameters()).device
            temp_module = temp_module.to(device)
        except StopIteration:
            pass  # module has no parameters, leave on CPU
        module.per_query_temp = temp_module

        # Patch forward (store original for later restoration).
        original_forward = module.forward
        module._original_forward = original_forward
        module.forward = _make_patched_forward(original_forward, temp_module).__get__(module, type(module))

        patched_count += 1

    print(
        f"  [PerQueryTemp] Patched {patched_count} attention layer(s) "
        f"with adaptive temperature (lossless at init: T=sqrt(d_head))."
    )
    return model


def remove_per_query_temp(model: nn.Module) -> nn.Module:
    """Remove the per-query adaptive temperature patch from *model*.

    Restores the original ``forward`` for every patched layer and deletes
    the ``per_query_temp`` sub-module.

    .. note::
        This requires that the original forward was stored.  The patch
        stores it as ``layer._original_forward``.

    Args:
        model: The patched model.

    Returns:
        The restored model (same object).
    """
    restored = 0
    for module in model.modules():
        if hasattr(module, "per_query_temp"):
            if hasattr(module, "_original_forward"):
                module.forward = module._original_forward
                del module._original_forward
            del module.per_query_temp
            restored += 1

    if restored:
        print(f"  [PerQueryTemp] Restored original forward for {restored} layer(s).")
    return model
