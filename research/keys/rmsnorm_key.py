"""RMSNorm key — per-feature scale after RMS normalization.

Architecture: y = (x / sqrt(mean(x^2) + eps)) * weight

Forward key: weight = mean(y / x_normalized, dim=0)  per feature
Reverse key: y = x_normalized * weight

Classification: Bi (exact both directions)
"""
import torch
from .base import Key, KeyClass, KeyResult


class RMSNormKey(Key):
    @property
    def name(self) -> str:
        return "rmsnorm"

    @property
    def description(self) -> str:
        return "RMSNorm scale. weight = mean(y / x_normalized). Linear given normalized input."

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def forward(self, data: dict) -> KeyResult:
        """data -> weights. Expects: 'X' [n, d], 'Y' [n, d], 'eps' (optional)."""
        try:
            X = data['X']
            Y = data['Y']
            eps = data.get('eps', 1e-6)

            # Normalize X
            x_norm = X / torch.sqrt(X.pow(2).mean(dim=-1, keepdim=True) + eps)
            # weight = y / x_norm, averaged over samples
            weight = (Y / x_norm).mean(dim=0)

            return KeyResult(
                success=True,
                weights={'weight': weight},
                metadata={'eps': eps, 'd_model': X.shape[-1], 'n_samples': X.shape[0]}
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data. Returns weight as data (it's the learned scale)."""
        weight = weights.get('weight')
        if weight is None:
            return KeyResult(success=False, error="Missing 'weight' in weights")
        return KeyResult(
            success=True,
            data={'weight': weight, 'd_model': weight.shape[-1]},
            metadata={'d_model': weight.shape[-1]}
        )

    def apply(self, weights: dict, X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply RMSNorm: y = (x / rms(x)) * weight."""
        weight = weights['weight']
        x_norm = X / torch.sqrt(X.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x_norm * weight
