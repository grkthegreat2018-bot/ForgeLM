"""SliceGPT key — computational invariance + PCA slicing for model compression.

SliceGPT (ICLR 2024, Microsoft) exploits the fact that the residual stream is
computationally invariant under orthogonal transforms. By applying a PCA-based
orthogonal transform Q to the residual stream, then slicing off low-variance
dimensions, we get a smaller dense model with identical behaviour on the
kept subspace.

Key class: FULL — pure math (PCA + slice), no training needed.
"""
import torch
import torch.nn as nn
from typing import Dict
from .base import Key, KeyClass, KeyResult


def compute_pca_transform(activations: torch.Tensor, sparsity: float = 0.25):
    """Compute PCA transform Q from residual stream activations.

    Args:
        activations: (n_tokens, d_model) residual stream activations
        sparsity: fraction of dimensions to remove

    Returns:
        Q: (d_model, d_model) orthogonal transform (sorted by eigenvalue desc)
        keep_indices: which columns to keep
        new_d_model: reduced dimension
    """
    assert activations.dim() == 2, f"Expected 2D, got {activations.dim()}D"
    n, d = activations.shape
    centered = activations - activations.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)          # ascending
    order = torch.argsort(eigvals, descending=True)
    Q = eigvecs[:, order]
    new_d = max(1, int(d * (1.0 - sparsity)))
    keep = order[:new_d]
    return Q, keep, new_d


class SliceGPTKey(Key):
    """SliceGPT: computational invariance + PCA slicing for model compression.

    Applies an orthogonal transform to the residual stream (PCA of activations),
    then slices off low-variance dimensions to reduce d_model.

    Key class: FULL — pure math (PCA + slice), no training needed.
    Needs a small calibration set to compute the PCA transform.
    """

    @property
    def name(self) -> str:
        return "slicegpt"

    @property
    def description(self) -> str:
        return "PCA orthogonal transform + dimension slicing (SliceGPT, ICLR 2024)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights. Computes PCA transform and slices weight matrices.

        Expected data: residual_activations (n_tokens, d_model),
                       sparsity (float), weights (dict of 2D tensors).
        """
        try:
            acts = data["residual_activations"]
            sparsity = float(data.get("sparsity", 0.25))
            weights = data.get("weights", {})
        except KeyError as e:
            return KeyResult(success=False, error=f"Missing key: {e}")

        Q, keep, new_d = compute_pca_transform(acts, sparsity)
        d = Q.shape[0]
        sliced: Dict[str, torch.Tensor] = {}
        for name, W in weights.items():
            if not isinstance(W, torch.Tensor) or W.dim() < 2:
                sliced[name] = W
                continue
            Wt = Q @ W @ Q.T                    # transform
            sliced[name] = Wt[keep][:, keep]    # slice kept dims
        return KeyResult(
            success=True,
            weights={"transform": Q, "new_d_model": new_d, "sliced_weights": sliced},
            metadata={"original_d_model": d, "new_d_model": new_d,
                      "sparsity": sparsity, "keep_indices": keep},
        )

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data. Pads sliced weights with zeros, applies inverse Q."""
        try:
            Q = weights["transform"]
            sliced = weights["sliced_weights"]
        except KeyError as e:
            return KeyResult(success=False, error=f"Missing key: {e}")
        d = Q.shape[0]
        new_d = weights.get("new_d_model", d)
        Q_inv = Q.T
        restored: Dict[str, torch.Tensor] = {}
        for name, Ws in sliced.items():
            if not isinstance(Ws, torch.Tensor) or Ws.dim() < 2:
                restored[name] = Ws
                continue
            pad = torch.zeros(d, d, dtype=Ws.dtype, device=Ws.device)
            pad[:new_d, :new_d] = Ws
            restored[name] = Q_inv @ pad @ Q     # inverse transform
        return KeyResult(
            success=True,
            data={"weights": restored, "transform": Q, "new_d_model": new_d},
            metadata={"original_d_model": d, "new_d_model": new_d, "approximate": True},
        )


def apply_slicegpt_to_model(model, residual_activations: torch.Tensor,
                            sparsity: float = 0.25) -> float:
    """Apply SliceGPT compression to a model (in-place).

    Transforms all weight matrices with PCA and slices low-variance dims.
    Returns the compression ratio achieved.
    """
    key = SliceGPTKey()
    weights = {n: p.data.clone() for n, p in model.named_parameters() if p.dim() >= 2}
    result = key.forward({"residual_activations": residual_activations,
                          "sparsity": sparsity, "weights": weights})
    if not result.success:
        raise RuntimeError(f"SliceGPT forward failed: {result.error}")
    sliced = result.weights["sliced_weights"]
    d, new_d = result.metadata["original_d_model"], result.metadata["new_d_model"]
    for name, Ws in sliced.items():
        obj = model
        parts = name.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], nn.Parameter(Ws))
    return 1.0 - (new_d / d)
