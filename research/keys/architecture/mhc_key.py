"""Manifold-Constrained Hyper-Connections (mHC) — DeepSeek-V4.

DeepSeek-V4 uses mHC to enhance conventional residual connections. Instead
of the standard x + sublayer(x), mHC projects the sublayer output through
a learned low-rank manifold before adding:

    x_out = x + W_proj(sublayer(x))

where W_proj is a low-rank projection (rank = d_model // 4) that constrains
the sublayer output to a learned manifold. This improves information flow
across depth by:
  1. Projecting the residual update into a lower-dimensional subspace,
     reducing noise and redundancy.
  2. Learning a task-specific manifold that best preserves useful information.
  3. Identity init (W_proj = I on the diagonal) makes it lossless at start.

This is a generalization of the ValueResidual key (which copies V from
layer 0) — mHC applies a learned projection to EVERY sublayer output.

Architecture:
  - W_proj = U @ V^T where U ∈ R^{D×r}, V ∈ R^{D×r}, r = D//4.
  - Identity init: U = [I_r; 0], V = [I_r; 0] → W_proj = [[I_r, 0], [0, 0]].
    This is NOT full identity — it projects to the first r dimensions.
    For TRUE identity init, use r = D (full rank). But the paper uses r=D//4
    with a warmup where the gate starts at 0.
  - Gate g (scalar, init=0): x_out = x + g * W_proj(sublayer(x)) + sublayer(x).
    At g=0, this is standard residual. As g increases, the manifold projection
    is gradually mixed in.

Usage:
    from research.keys.architecture.mhc_key import MHCModule, MHCKey

    # In model construction:
    mhc = MHCModule(d_model, rank=d_model//4)
    # Replace: x = x + sublayer(x)
    # With:    x = mhc(x, sublayer(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class MHCModule(nn.Module):
    """Manifold-Constrained Hyper-Connection module.

    Projects sublayer output through a low-rank learned manifold before
    adding to the residual stream. Gate init=0 → lossless at start.

    Args:
        d_model: model dimension D.
        rank: manifold dimension r (default D//4).
    """

    def __init__(self, d_model: int, rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        if rank is None:
            rank = max(1, d_model // 4)
        self.rank = rank

        # Low-rank projection: W_proj = U @ V^T
        # U: (D, r), V: (D, r)
        # Init: U = [I_r; 0], V = [I_r; 0] → W_proj = diag(I_r, 0)
        U = torch.zeros(d_model, rank)
        V = torch.zeros(d_model, rank)
        U[:rank, :rank] = torch.eye(rank)
        V[:rank, :rank] = torch.eye(rank)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)

        # Gate (init=0 → lossless: x_out = x + sublayer(x)).
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """Compute manifold-constrained residual.

        x_out = x + sublayer_out + gate * (U @ V^T @ sublayer_out)

        At gate=0: x_out = x + sublayer_out (standard residual).
        At gate>0: the manifold projection is mixed in.

        Args:
            x: (B, seq_len, D) — residual stream.
            sublayer_out: (B, seq_len, D) — sublayer output.

        Returns:
            (B, seq_len, D) — updated residual stream.
        """
        # Low-rank projection: (D, r) @ (r, D) -> (D, D)
        # Applied as: sublayer_out @ (U @ V^T) = (sublayer_out @ U) @ V^T
        # More efficient: (B, S, D) @ (D, r) -> (B, S, r) @ (r, D) -> (B, S, D)
        projected = (sublayer_out @ self.U) @ self.V.transpose(-1, -2)
        return x + sublayer_out + self.gate * projected


class MHCKey(Key):
    """mHC key — initializes manifold-constrained hyper-connection parameters.

    Identity init: gate=0 → standard residual. Lossless at start.
    Fine-tuning learns the optimal manifold projection.

    Key class: TRIVIAL — identity init, no data or training needed.
    """

    @property
    def name(self) -> str:
        return "mhc"

    @property
    def description(self) -> str:
        return ("Manifold-Constrained Hyper-Connections (DeepSeek-V4): "
                "low-rank residual projection, gated (init=0 = lossless)")

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize mHC parameters for a layer.

        Args:
            data: {"d_model": int, "rank": int (optional, default d_model//4)}

        Returns:
            {"U": (D, r) tensor, "V": (D, r) tensor, "gate": scalar tensor}
        """
        try:
            D = data["d_model"]
            r = data.get("rank", max(1, D // 4))

            U = torch.zeros(D, r)
            V = torch.zeros(D, r)
            U[:r, :r] = torch.eye(r)
            V[:r, :r] = torch.eye(r)
            gate = torch.tensor(0.0)

            return KeyResult(
                success=True,
                weights={"U": U, "V": V, "gate": gate},
                metadata={"d_model": D, "rank": r, "init": "identity (gate=0)"},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Extract mHC parameters from a trained model.

        Args:
            weights: {"U": tensor, "V": tensor, "gate": scalar}

        Returns:
            {"d_model": int, "rank": int, "gate": float}
        """
        try:
            U = weights.get("U")
            V = weights.get("V")
            gate = weights.get("gate")
            if U is None or V is None or gate is None:
                return KeyResult(success=False,
                                 error="Missing U, V, or gate in weights")
            D, r = U.shape
            return KeyResult(
                success=True,
                data={"d_model": D, "rank": r, "gate": float(gate)},
                metadata={"had_projection": abs(float(gate)) > 1e-6},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def apply_mhc_to_model(model: nn.Module, d_model: int,
                       rank: int | None = None,
                       test_input=None, safe: bool = True) -> list[MHCModule]:
    """Create MHC modules for each layer of a model.

    Uses safety validation. MHC modules are attached but not called during
    forward (the training loop must wire them in). Gate init=0 means they
    would be identity even if called.

    Args:
        model: the model.
        d_model: model dimension.
        rank: manifold rank (default d_model//4).
        test_input: optional input for forward validation.
        safe: if True, use safe_apply with rollback.

    Returns:
        List of MHCModule (one per layer). Also attached as model.mhc_modules.
    """
    def _apply(m):
        n_layers = len(m.blocks) if hasattr(m, "blocks") else 1
        modules = [MHCModule(d_model, rank=rank) for _ in range(n_layers)]
        m.mhc_modules = nn.ModuleList(modules)
        return m

    if safe:
        from research.keys.safety import safe_apply
        safe_apply(model, _apply, identity_init=True,
                   test_input=test_input, atol=1e-5, rtol=1e-4)
    else:
        _apply(model)

    return list(model.mhc_modules)
