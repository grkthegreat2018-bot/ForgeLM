"""Per-Query Interpolation Key — interpolate full model weights per query.

Novel insight: Instead of choosing between a specialist (model A) and a
generalist (model B) at deployment time, we can interpolate their *full*
weight matrices per-query based on query features. A lightweight router
predicts an interpolation coefficient alpha in [0, 1], and we compute
W_mixed = alpha * W_A + (1-alpha) * W_B. High-norm queries (domain-specific)
route toward the specialist; low-norm queries route toward the generalist.
This is a runtime operation — no training, no permanent weight changes —
but the interpolation itself is lossy and irreversible.

Key class: TRIVIAL — runtime weight mixing, no training, irreversible.
"""
import torch
from typing import Dict, Callable, Optional
from .base import Key, KeyClass, KeyResult


class PerQueryInterp:
    """Per-query full-weight interpolation between two models.

    A router function maps query features to an interpolation coefficient
    alpha in [0, 1]. Weights are mixed as W_mixed = alpha * W_A + (1-alpha) * W_B.
    """

    def __init__(self, alpha_router: Optional[Callable[[torch.Tensor], float]] = None):
        self.alpha_router = alpha_router or self._default_router

    def _default_router(self, query_features: torch.Tensor) -> float:
        """Default router: use L2 norm of query embedding.

        High norm -> specialist (alpha=1), low norm -> generalist (alpha=0).
        """
        norm = query_features.float().norm().item()
        return max(0.0, min(1.0, norm / 10.0))

    def interpolate_weights(
        self,
        weights_a: Dict[str, torch.Tensor],
        weights_b: Dict[str, torch.Tensor],
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        """Interpolate two state dicts linearly.

        W_mixed = alpha * W_A + (1-alpha) * W_B

        Args:
            weights_a: specialist state dict
            weights_b: generalist state dict
            alpha: interpolation coefficient in [0, 1]

        Returns:
            Mixed state dict
        """
        mixed = {}
        for key in weights_a:
            if key in weights_b and weights_a[key].shape == weights_b[key].shape:
                wa = weights_a[key].float()
                wb = weights_b[key].float()
                mixed[key] = (alpha * wa + (1.0 - alpha) * wb).to(weights_a[key].dtype)
            else:
                mixed[key] = weights_a[key]
        return mixed

    def apply_to_model(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        query_features: torch.Tensor,
    ) -> float:
        """Interpolate model weights in-place on model_a and return alpha.

        Args:
            model_a: specialist model (modified in-place)
            model_b: generalist model (read-only)
            query_features: 1-D tensor of query embedding features

        Returns:
            alpha used for interpolation
        """
        alpha = self.alpha_router(query_features)
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        mixed = self.interpolate_weights(state_a, state_b, alpha)
        model_a.load_state_dict(mixed)
        return alpha


class PerQueryInterpKey(Key):
    """Per-Query Interpolation key — interpolate full weights per query.

    Key class: TRIVIAL — runtime weight mixing, no training, irreversible.
    """

    @property
    def name(self) -> str:
        return "per_query_interp"

    @property
    def description(self) -> str:
        return ("Per-query full-weight interpolation: router predicts alpha, "
                "W_mixed = alpha*W_A + (1-alpha)*W_B")

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Predict alpha from query features and interpolate weights.

        Args:
            data: {"query_features": tensor, "weights_a": dict, "weights_b": dict}
        """
        try:
            interp = PerQueryInterp()
            alpha = interp.alpha_router(data["query_features"])
            mixed = interp.interpolate_weights(
                data["weights_a"], data["weights_b"], alpha
            )
            return KeyResult(
                success=True,
                weights={"mixed_weights": mixed, "alpha": torch.tensor(alpha)},
                metadata={"alpha": alpha, "n_params": len(mixed)},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Interpolation is lossy — cannot recover original weights."""
        return KeyResult(
            success=False,
            error="Interpolation is irreversible (lossy: two models -> one)",
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 64

    # Synthetic weights for two "models"
    weights_a = {f"layer_{i}.weight": torch.randn(d_model, d_model) * 0.5
                 for i in range(3)}
    weights_b = {f"layer_{i}.weight": torch.randn(d_model, d_model) * 0.3
                 for i in range(3)}

    # Test 1: high-norm query -> alpha near 1 (specialist)
    key = PerQueryInterpKey()
    high_norm = torch.randn(d_model) * 8.0
    result_hi = key.forward({
        "query_features": high_norm,
        "weights_a": weights_a,
        "weights_b": weights_b,
    })
    assert result_hi.success, f"Forward failed: {result_hi.error}"
    alpha_hi = result_hi.metadata["alpha"]
    assert alpha_hi > 0.5, f"Expected alpha > 0.5 for high norm, got {alpha_hi}"
    print(f"[PerQueryInterpKey] high-norm alpha={alpha_hi:.4f} (specialist)")

    # Test 2: low-norm query -> alpha near 0 (generalist)
    low_norm = torch.randn(d_model) * 0.1
    result_lo = key.forward({
        "query_features": low_norm,
        "weights_a": weights_a,
        "weights_b": weights_b,
    })
    alpha_lo = result_lo.metadata["alpha"]
    assert alpha_lo < 0.5, f"Expected alpha < 0.5 for low norm, got {alpha_lo}"
    print(f"[PerQueryInterpKey] low-norm alpha={alpha_lo:.4f} (generalist)")

    # Test 3: alpha=1 -> mixed == weights_a
    interp = PerQueryInterp()
    mixed_1 = interp.interpolate_weights(weights_a, weights_b, 1.0)
    for k in weights_a:
        assert torch.allclose(mixed_1[k], weights_a[k], atol=1e-6), "alpha=1 mismatch"
    # alpha=0 -> mixed == weights_b
    mixed_0 = interp.interpolate_weights(weights_a, weights_b, 0.0)
    for k in weights_b:
        assert torch.allclose(mixed_0[k], weights_b[k], atol=1e-6), "alpha=0 mismatch"
    print("[PerQueryInterpKey] boundary checks: alpha=1->A, alpha=0->B (OK)")

    # Test 4: reverse is irreversible
    rev = key.reverse(result_hi.weights)
    assert not rev.success, "Reverse should fail (irreversible)"
    print(f"[PerQueryInterpKey] reverse correctly fails: {rev.error}")
    print("[PerQueryInterpKey] all tests passed")
