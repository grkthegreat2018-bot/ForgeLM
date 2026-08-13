"""Attention Residuals (AttnRes) — Kimi K3 cross-layer retrieval.

Kimi K3 uses Attention Residuals to selectively retrieve representations
across depth rather than accumulating them uniformly via standard residual
connections. This gives a 2.5x scaling efficiency improvement over K2.

Standard residual: x_out = x + sublayer(x)
AttnRes:          x_out = x + sublayer(x) + g_i * retrieve(x_{i-k}, x_i)

where g_i is a learned scalar gate (init=0 → lossless at start) and
retrieve() is an attention-based retrieval from past layer representations.

This generalizes DenseFormer DWA (which uses fixed weighted averaging) —
AttnRes uses ATTENTION to dynamically select WHICH past layers to retrieve
from based on the current query, not just static weighted averaging.

Architecture:
  - For each layer i, maintain a buffer of past layer outputs.
  - AttnRes gate: g_i (scalar, init=0).
  - AttnRes retrieval: a small cross-attention from x_i (query) to the
    buffer of past outputs (keys/values), with a learned projection.
  - Output: x_i + sublayer(x_i) + g_i * retrieval_i

Identity init: g_i=0 → no retrieval → standard transformer. Lossless.

Usage:
    from research.keys.architecture.attn_residual_key import AttnResKey, AttnResModule

    # In model construction:
    attn_res = AttnResModule(d_model, n_layers, k=4)  # retrieve from last 4 layers
    # After each block: x = attn_res(x, layer_idx, past_outputs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class AttnResModule(nn.Module):
    """Attention Residual module for cross-layer representation retrieval.

    Maintains a buffer of past layer outputs and retrieves from them via
    cross-attention, gated by a learned scalar (init=0 = lossless).

    Args:
        d_model: model dimension.
        n_layers: total number of layers (for pre-allocating gates).
        k: number of past layers to retrieve from (window size).
        n_heads: number of attention heads for retrieval.
    """

    def __init__(self, d_model: int, n_layers: int, k: int = 4,
                 n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.k = k
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        self.head_dim = head_dim

        # Per-layer gates (init=0 → lossless).
        self.gates = nn.Parameter(torch.zeros(n_layers))

        # Query/key/value projections for retrieval attention.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Scale for attention.
        self.scale = head_dim ** -0.5

        # Identity init for projections: q_proj = I, k_proj = I, v_proj = I,
        # out_proj = I. This makes retrieval = identity at init (but gated
        # by 0, so it doesn't matter — still lossless).
        with torch.no_grad():
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
            nn.init.eye_(self.v_proj.weight)
            nn.init.eye_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, layer_idx: int,
                past_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Compute AttnRes retrieval for the current layer.

        Args:
            x: (B, seq_len, d_model) — current layer input.
            layer_idx: current layer index (0-based).
            past_outputs: list of past layer outputs (x_0, x_1, ..., x_{i-1}).

        Returns:
            (B, seq_len, d_model) — retrieval result (to be added to x + sublayer(x)).
        """
        if layer_idx == 0 or not past_outputs:
            return torch.zeros_like(x)

        # Select last k past outputs.
        k_past = past_outputs[-self.k:]
        if not k_past:
            return torch.zeros_like(x)

        B, S, D = x.shape
        n_past = len(k_past)

        # Stack past outputs: (B, n_past * S, D)
        past_stack = torch.cat(k_past, dim=1)  # (B, n_past*S, D)

        # Project query (from current x) and key/value (from past).
        q = self.q_proj(x)  # (B, S, D)
        kv = self.k_proj(past_stack)  # (B, n_past*S, D)
        vv = self.v_proj(past_stack)  # (B, n_past*S, D)

        # Reshape for multi-head attention.
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        kv = kv.view(B, n_past * S, self.n_heads, self.head_dim).transpose(1, 2)
        vv = vv.view(B, n_past * S, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (causal within each past layer, but
        # past layers are already computed — full attention to past is fine).
        attn = F.scaled_dot_product_attention(q, kv, vv, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection + gate.
        out = self.out_proj(attn)
        gate = self.gates[layer_idx]
        return gate * out


class AttnResKey(Key):
    """AttnRes key — initializes cross-layer attention residual gates.

    Identity init: all gates = 0 → no retrieval → standard transformer.
    Fine-tuning learns which layers benefit from cross-layer retrieval.

    Key class: TRIVIAL — identity init, no data or training needed.
    """

    @property
    def name(self) -> str:
        return "attn_residual"

    @property
    def description(self) -> str:
        return ("Attention Residuals (Kimi K3): cross-layer retrieval via "
                "attention, gated (init=0 = lossless)")

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize AttnRes gates for all layers.

        Args:
            data: {"n_layers": int, "k": int (window, default 4),
                   "d_model": int, "n_heads": int (default 4)}

        Returns:
            {"gates": tensor of shape (n_layers,) — all zeros (identity init)}
        """
        try:
            n_layers = data["n_layers"]
            gates = torch.zeros(n_layers)
            return KeyResult(
                success=True,
                weights={"gates": gates},
                metadata={
                    "k": data.get("k", 4),
                    "n_heads": data.get("n_heads", 4),
                    "d_model": data.get("d_model", 768),
                    "init": "identity (gates=0)",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Extract gates from a trained model.

        Args:
            weights: {"gates": tensor of shape (n_layers,)}

        Returns:
            {"n_layers": int, "gates": tensor}
        """
        try:
            gates = weights.get("gates")
            if gates is None:
                return KeyResult(success=False, error="Missing 'gates' in weights")
            return KeyResult(
                success=True,
                data={"n_layers": len(gates), "gates": gates.clone()},
                metadata={"had_retrieval": float(gates.abs().max()) > 1e-6},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def apply_attn_res_to_model(model: nn.Module, n_layers: int,
                            d_model: int, k: int = 4,
                            n_heads: int = 4,
                            test_input=None, safe: bool = True) -> AttnResModule:
    """Create and attach an AttnRes module to a model.

    Uses safety validation. AttnRes gates init=0, so the module returns zeros
    and the model output should be identical after attachment (the module is
    not called during forward — it must be wired into the training loop).

    Note: this function only ATTACHES the module. The training loop must
    call it after each block. The safety check verifies that attaching the
    module doesn't corrupt existing parameters.

    Args:
        model: the model to attach to.
        n_layers: number of transformer layers.
        d_model: model dimension.
        k: number of past layers to retrieve from.
        n_heads: number of retrieval attention heads.
        test_input: optional input for forward validation.
        safe: if True, use safe_apply with rollback.

    Returns:
        The AttnResModule (also attached as model.attn_res).
    """
    def _apply(m):
        attn_res = AttnResModule(d_model, n_layers, k=k, n_heads=n_heads)
        m.attn_res = attn_res
        return m

    if safe:
        from research.keys.safety import safe_apply
        # identity_init=True because attaching a zero-gate module doesn't
        # change the forward pass (the module isn't called during forward).
        safe_apply(model, _apply, identity_init=True,
                   test_input=test_input, atol=1e-5, rtol=1e-4)
    else:
        _apply(model)

    return model.attn_res
