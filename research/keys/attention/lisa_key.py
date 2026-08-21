"""LiSA — Cross-Layer Attention Sharing (TACL 2026).

Share Q/K projection weights across attention layers with a tiny alignment
FFN per layer. 6x Q/K compression, 19-40% throughput improvement.

LOSSLESS AT INIT: per-layer Q/K are loaded from the checkpoint (unchanged).
A shared Q/K component is added with gate=0 (contributes nothing at start).
Training opens the gate, and the shared Q/K can gradually replace per-layer
Q/K (which can then be pruned).

Paper: ACL 2026, TACL. Code: github.com/takagi97/tisa

Usage:
    # In ModelConfig:
    config.use_lisa = True
    config.lisa_compress = 6
    config.lisa_align_dim = 0  # 0 = auto (d_model // 4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LisaSharedQK(nn.Module):
    """Shared Q/K projections with per-layer alignment + gating.

    Holds:
    - shared_q: (d_model, n_heads * head_dim) — shared Q projection
    - shared_k: (d_model, n_kv_heads * head_dim) — shared K projection
    - per-layer alignment FFN: small (d_model → align_dim → d_model) that
      corrects the shared Q/K for each layer's specific needs
    - per-layer gate (scalar, init=0 → lossless at start)

    At init (gate=0): each layer uses its own Q/K (from checkpoint).
    The shared Q/K + alignment contributes nothing.
    After training (gate→1): shared Q/K + alignment can replace per-layer Q/K.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, n_attn_layers: int, align_dim: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_attn_layers = n_attn_layers
        self.align_dim = align_dim if align_dim > 0 else d_model // 4

        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        # Shared Q/K (initialized to zeros — will be trained)
        self.shared_q = nn.Linear(d_model, q_dim, bias=False)
        self.shared_k = nn.Linear(d_model, kv_dim, bias=False)
        nn.init.zeros_(self.shared_q.weight)
        nn.init.zeros_(self.shared_k.weight)

        # Per-layer alignment: small FFN that adapts shared Q/K for each layer
        # align_q[i]: (d_model, align_dim) + (align_dim, q_dim)
        # align_k[i]: (d_model, align_dim) + (align_dim, kv_dim)
        self.align_q_down = nn.ModuleList([
            nn.Linear(d_model, self.align_dim, bias=False)
            for _ in range(n_attn_layers)
        ])
        self.align_q_up = nn.ModuleList([
            nn.Linear(self.align_dim, q_dim, bias=False)
            for _ in range(n_attn_layers)
        ])
        self.align_k_down = nn.ModuleList([
            nn.Linear(d_model, self.align_dim, bias=False)
            for _ in range(n_attn_layers)
        ])
        self.align_k_up = nn.ModuleList([
            nn.Linear(self.align_dim, kv_dim, bias=False)
            for _ in range(n_attn_layers)
        ])

        # Zero-init alignment up-projections so alignment contributes nothing at start
        for i in range(n_attn_layers):
            nn.init.zeros_(self.align_q_up[i].weight)
            nn.init.zeros_(self.align_k_up[i].weight)

        # Per-layer gate (scalar, init=0 → lossless at start)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_attn_layers)
        ])

    def get_q_correction(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Get the LiSA Q correction for a given layer.

        Returns the shared Q + alignment output, scaled by the gate.
        At init (gate=0), returns zeros (no correction).
        """
        gate = self.gates[layer_idx]
        h = F.silu(self.align_q_down[layer_idx](x))
        aligned = self.align_q_up[layer_idx](h)
        shared = self.shared_q(x)
        return gate * (shared + aligned)

    def get_k_correction(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Get the LiSA K correction for a given layer."""
        gate = self.gates[layer_idx]
        h = F.silu(self.align_k_down[layer_idx](x))
        aligned = self.align_k_up[layer_idx](h)
        shared = self.shared_k(x)
        return gate * (shared + aligned)


class LisaKey:
    """Key interface for LiSA checkpoint conversion (lossless at init).

    LiSA requires NO checkpoint conversion — the shared Q/K and alignment
    modules are zero-initialized and contribute nothing at load time.
    The per-layer Q/K are loaded from the checkpoint unchanged.
    Missing keys (lisa.shared_q.weight, lisa.gates.*, etc.) are handled by
    strict=False loading.
    """

    @staticmethod
    def apply(model: nn.Module, config) -> LisaSharedQK | None:
        """Attach a LisaSharedQK module to the model.

        Returns the LisaSharedQK module, or None if no attention layers.
        """
        n_attn = sum(1 for block in model.blocks
                     if getattr(block, 'layer_type', 'attention') == 'attention')
        if n_attn == 0:
            return None

        d_model = config.d_model
        n_heads = config.n_heads
        n_kv = getattr(config, 'n_kv_heads', None) or n_heads
        head_dim = d_model // n_heads
        align_dim = getattr(config, 'lisa_align_dim', 0)

        lisa = LisaSharedQK(d_model, n_heads, n_kv, head_dim,
                            n_attn, align_dim=align_dim)
        model.lisa = lisa
        return lisa
