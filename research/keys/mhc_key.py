"""mHC (Manifold-Constrained Hyper-Connections) weight-stealing key.

DeepSeek V4 (arxiv 2512.24880) replaces standard residual connections with
hyper-connections parameterized by three matrices:
  - H_res : doubly stochastic (Birkhoff polytope) — mixes residual streams
  - H_pre : non-negative mixing map — pre-processing
  - H_post: non-negative mixing map — post-processing

Key insight: initialize H_res as IDENTITY (which is doubly stochastic by
construction) and H_pre/H_post as identity-like.  This preserves the original
residual behavior *exactly* — the model is unchanged.  Fine-tuning then
learns optimal mixing on the Birkhoff polytope via Sinkhorn-Knopp projection.

Key class: TRIVIAL — identity init, no data or training needed.
"""
from typing import Dict

import torch

from .base import Key, KeyClass, KeyResult


def sinkhorn_knopp_project(matrix: torch.Tensor, n_iters: int = 10) -> torch.Tensor:
    """Project a matrix onto the Birkhoff polytope (doubly stochastic).

    Alternately normalize rows and columns to sum to 1.
    Used during fine-tuning to keep H_res on the manifold.
    """
    m = matrix.detach().clone().abs()
    n = m.shape[0]
    for _ in range(n_iters):
        # Row normalize
        row_sum = m.sum(dim=1, keepdim=True).clamp_min(1e-12)
        m = m / row_sum
        # Column normalize
        col_sum = m.sum(dim=0, keepdim=True).clamp_min(1e-12)
        m = m / col_sum
    return m


def _padded_identity(rows: int, cols: int) -> torch.Tensor:
    """Return an identity matrix of shape (rows, cols), zero-padded."""
    m = torch.zeros(rows, cols)
    n = min(rows, cols)
    m[:n, :n] = torch.eye(n)
    return m


class MHCKey(Key):
    """Manifold-Constrained Hyper-Connections key.

    Initializes H_res as identity (doubly stochastic by construction),
    H_pre and H_post as identity-like matrices.  This preserves the
    original residual behavior exactly — the model is unchanged.

    Fine-tuning then learns optimal mixing on the Birkhoff polytope
    via Sinkhorn-Knopp projection.

    Key class: TRIVIAL — identity init, no data or training.
    """

    @property
    def name(self) -> str:
        return "mhc"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights.  Expects {"d_model": int, "expansion": int (default 1)}."""
        try:
            d_model = int(data["d_model"])
            expansion = int(data.get("expansion", 1))
            d_exp = d_model * expansion
            weights = {
                "H_res": torch.eye(d_model),
                "H_pre": _padded_identity(d_exp, d_model),
                "H_post": _padded_identity(d_model, d_exp),
            }
            return KeyResult(success=True, weights=weights,
                             metadata={"d_model": d_model, "expansion": expansion})
        except Exception as exc:
            return KeyResult(success=False, error=str(exc))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data.  Identity is its own inverse — passthrough."""
        try:
            d_model = weights["H_res"].shape[0]
            d_exp = weights["H_pre"].shape[0]
            expansion = d_exp // d_model
            return KeyResult(success=True,
                             data={"d_model": d_model, "expansion": expansion},
                             metadata={"note": "identity passthrough"})
        except Exception as exc:
            return KeyResult(success=False, error=str(exc))


def init_mhc_in_model(model, expansion: int = 1) -> None:
    """Initialize mHC connections in a model (replaces residual connections).

    For each block, adds H_res/H_pre/H_post as identity matrices.
    The model behaves identically to before — mHC is a no-op at init.
    """
    key = MHCKey()
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    if d_model is None:
        raise ValueError("Cannot determine d_model from model.config")
    result = key.forward({"d_model": d_model, "expansion": expansion})
    if not result.success:
        raise RuntimeError(f"mHC init failed: {result.error}")
    blocks = getattr(model, "layers", None) or getattr(model, "blocks", None) or getattr(model, "h", None)
    if blocks is None:
        raise ValueError("Cannot find transformer blocks (layers/blocks/h)")
    for block in blocks:
        block.H_res = torch.nn.Parameter(result.weights["H_res"].clone())
        block.H_pre = torch.nn.Parameter(result.weights["H_pre"].clone())
        block.H_post = torch.nn.Parameter(result.weights["H_post"].clone())
