"""Linear + MSE key — the foundation key.

Architecture: y = x @ W^T (no bias, no activation)
Loss: MSE

Forward key: W = pinv(X^T X) @ X^T Y   (normal equation, instant)
Reverse key: Y = X @ W^T                (forward pass)
             X = Y @ pinv(W^T)          (if d_out >= d_in)

Classification: Partial (forward exact, reverse limited by dimensionality)
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class LinearMSEKey(Key):
    @property
    def name(self) -> str:
        return "linear_mse"

    @property
    def description(self) -> str:
        return "Linear layer with MSE loss. Normal equation: W = pinv(X^T X) @ X^T Y"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """data -> weights. Expects keys: 'X' [n, d_in], 'Y' [n, d_out]."""
        try:
            X = data['X']
            Y = data['Y']
            # Normal equation: W = pinv(X^T X) @ X^T Y, shape [d_out, d_in]
            XtX = X.T @ X
            W = torch.linalg.pinv(XtX) @ X.T @ Y
            W = W.T  # [d_out, d_in]
            return KeyResult(
                success=True,
                weights={'W': W},
                metadata={'method': 'normal_equation', 'n_samples': X.shape[0],
                         'd_in': X.shape[1], 'd_out': Y.shape[1]}
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data. Limited: weights alone cannot recover training data."""
        W = weights.get('W')
        if W is None:
            return KeyResult(success=False, error="Missing 'W' in weights")
        # Cannot recover X, Y from W alone — weights are a function, not a database
        return KeyResult(
            success=False,
            error="Cannot recover training data from weights alone. "
                  "Weights encode the transform, not the data. "
                  "Provide X to recover Y, or Y to recover X (if d_out >= d_in).",
            metadata={'W_shape': list(W.shape)}
        )

    def reverse_with_input(self, weights: dict, X: torch.Tensor) -> KeyResult:
        """Given weights + input, recover output: Y = X @ W^T."""
        W = weights['W']
        Y = X @ W.T
        return KeyResult(success=True, data={'X': X, 'Y': Y})

    def reverse_with_output(self, weights: dict, Y: torch.Tensor) -> KeyResult:
        """Given weights + output, recover input: X = Y @ pinv(W^T). Only if d_out >= d_in."""
        W = weights['W']
        d_out, d_in = W.shape
        if d_out < d_in:
            return KeyResult(success=False,
                error=f"Cannot recover X: d_out={d_out} < d_in={d_in}, information lost")
        X = Y @ torch.linalg.pinv(W.T)
        return KeyResult(success=True, data={'X': X, 'Y': Y})
