"""R&D Round 23: HyperCloning — function-preserving model expansion.

Expands a smaller pre-trained transformer into a larger one such that the
cloned (larger) model produces EXACTLY the same logits as the source at
initialization. The larger model starts at the smaller model's accuracy,
then improves from there with more capacity.

Paper: arXiv:2409.12903 (Apple NeurIPS 2024).

Expansion modes
---------------
- WIDTH  (embedding_dim_multiplier=2): d_model and n_heads double,
  head_dim stays fixed.  Uses a DUPLICATION strategy so that the hidden
  state is ``[x, x]`` (source activations tiled) at every layer:
    * embedding / layernorm weights are TILED  → keeps norm statistics
      identical to the source (mean/var of ``[x, x]`` == mean/var of ``x``)
    * every linear weight becomes block-diagonal ``[[W, 0], [0, W]]``
      → ``[x, x] @ blkdiag = [W@x, W@x]`` (first half matches source)
    * the output head is zero-padded ``[W_src, 0]`` → logits depend only on
      the first (== source) half of the hidden state, reproducing source
      logits exactly.
- DEPTH  (depth_multiplier=2): n_layers multiplies.  The first
  ``src_n_layers`` layers get the (width-expanded) source weights; every
  extra layer is zero-initialised so it is an exact identity
  (``x + 0 + 0 = x``) at init — function-preserving while still trainable.

Works with both plain ``nn.Linear`` transformers and BitNet b1.58
(``BitNetLinear``) transformers via duck-typing on the ``.qscale``
attribute.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["clone_model"]


# ── weight expansion primitives ─────────────────────────────────────────────

def _block_diag(W: torch.Tensor, m: int) -> torch.Tensor:
    """Block-diagonal tile: ``[[W, 0], [0, W], ...]`` → (m*out, m*in).

    ``[x, x, ...] @ blkdiag(W, m).T`` → ``[W@x, W@x, ...]`` so the first
    block of the output matches the source linear exactly.
    """
    if m == 1:
        return W.clone()
    out, inp = W.shape
    new = torch.zeros(m * out, m * inp, dtype=W.dtype, device=W.device)
    for i in range(m):
        new[i * out:(i + 1) * out, i * inp:(i + 1) * inp] = W
    return new


def _tile(W: torch.Tensor, m: int, dim: int = 1) -> torch.Tensor:
    """Tile a weight tensor ``m`` times along ``dim`` (duplicate)."""
    if m == 1:
        return W.clone()
    reps = [1] * W.dim()
    reps[dim] = m
    return W.repeat(*reps)


def _pad_zeros(W: torch.Tensor, m: int, dim: int = 1) -> torch.Tensor:
    """Copy ``W`` into the first slice and zero-pad the rest along ``dim``."""
    if m == 1:
        return W.clone()
    shape = list(W.shape)
    shape[dim] = m * shape[dim]
    new = torch.zeros(*shape, dtype=W.dtype, device=W.device)
    new.narrow(dim, 0, W.shape[dim]).copy_(W)
    return new


# ── layer helpers (work for nn.Linear and BitNetLinear) ─────────────────────

_LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "w_gate", "w_up", "w_down")
_NORM_KEYS = ("attn_norm", "ffn_norm")


def _is_bitnet_layer(mod) -> bool:
    return hasattr(mod, "qscale")


def _get_sub(layer, key):
    """Access ``layer.key`` (TinyTransformerLayer) or ``layer[key]`` (ModuleDict)."""
    if isinstance(layer, nn.ModuleDict):
        return layer[key]
    return getattr(layer, key)


def _refresh_qscale(lin) -> None:
    """Re-anchor a BitNetLinear qscale to its (new) weight absmean."""
    if not _is_bitnet_layer(lin) or lin.qscale is None:
        return
    with torch.no_grad():
        lin.qscale.copy_(lin.weight.abs().mean().clamp(min=1e-6) / 0.7)


def _expand_layer(dst_layer, src_layer, m: int) -> None:
    """Copy width-expanded weights from ``src_layer`` into ``dst_layer``."""
    for key in _LINEAR_KEYS:
        src_lin = _get_sub(src_layer, key)
        dst_lin = _get_sub(dst_layer, key)
        dst_lin.weight.data = _block_diag(src_lin.weight.data, m)
        _refresh_qscale(dst_lin)
    for key in _NORM_KEYS:
        src_norm = _get_sub(src_layer, key)
        dst_norm = _get_sub(dst_layer, key)
        dst_norm.weight.data = _tile(src_norm.weight.data, m, dim=0)
        dst_norm.bias.data = _tile(src_norm.bias.data, m, dim=0)


def _zero_layer(layer) -> None:
    """Zero all linear weights → identity residual layer (function-preserving)."""
    for key in _LINEAR_KEYS:
        lin = _get_sub(layer, key)
        lin.weight.data.zero_()
        _refresh_qscale(lin)


# ── main API ────────────────────────────────────────────────────────────────

def clone_model(src_model: nn.Module,
                embedding_dim_multiplier: int = 2,
                depth_multiplier: int = 1) -> nn.Module:
    """Function-preserving expansion of ``src_model``.

    Args:
        src_model: a transformer exposing ``d_model``, ``n_layers``,
            ``n_heads``, ``head_dim``, ``vocab`` and constructed via
            ``type(src_model)(d_model=..., n_layers=..., n_heads=...,
            head_dim=..., vocab=...)``.
        embedding_dim_multiplier: width factor (2 → double d_model & n_heads,
            head_dim fixed).
        depth_multiplier: depth factor (2 → double n_layers; extra layers are
            zero-initialised identity at init).

    Returns:
        A new model of the same class with expanded dimensions whose logits
        match the source at initialisation (first ``vocab`` output dims for
        width expansion; exact for depth expansion).
    """
    m = embedding_dim_multiplier
    d = depth_multiplier
    src_d = src_model.d_model
    src_n = src_model.n_layers
    new_d = src_d * m
    new_n = src_n * d
    new_heads = src_model.n_heads * m
    new_head_dim = src_model.head_dim
    vocab = src_model.vocab

    model_cls = type(src_model)
    dst = model_cls(d_model=new_d, n_layers=new_n, n_heads=new_heads,
                    head_dim=new_head_dim, vocab=vocab)
    # ensure vocab attr exists (some classes don't store it)
    dst.vocab = vocab

    with torch.no_grad():
        # ── embedding: tile (duplicate) so hidden state is [e, e] ──
        dst.embed.weight.data = _tile(src_model.embed.weight.data, m, dim=1)

        # ── final norm: tile weight/bias ──
        dst.norm.weight.data = _tile(src_model.norm.weight.data, m, dim=0)
        dst.norm.bias.data = _tile(src_model.norm.bias.data, m, dim=0)

        # ── output head: zero-pad (logits depend only on source half) ──
        dst.head.weight.data = _pad_zeros(src_model.head.weight.data, m, dim=1)
        _refresh_qscale(dst.head)

        # ── layers: first src_n get expanded source weights ──
        for i in range(src_n):
            _expand_layer(dst.layers[i], src_model.layers[i], m)

        # ── extra depth layers: zero-init → identity at init ──
        for i in range(src_n, new_n):
            _zero_layer(dst.layers[i])

    return dst
