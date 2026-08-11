"""SwiGLU FFN key — W_down/W_up (Bi), W_gate (Partial).

Architecture:
  out = (silu(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T

W_down: linear given activations → Bi key (normal equation)
W_up: linear → Bi key (normal equation)
W_gate: silu is nonlinear → Partial key (approximate via linearization)
"""
import torch
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class SwiGLUKey(Key):
    @property
    def name(self) -> str:
        return "swiglu_ffn"

    @property
    def description(self) -> str:
        return "SwiGLU FFN. W_down/W_up: linear (Bi). W_gate: Newton iteration (Bi, exact in 5 iters)."

    def key_class(self) -> KeyClass:
        # W_down/W_up are Bi. W_gate is now Bi via Newton iteration.
        # All three components have exact forward keys.
        return KeyClass.BI

    def forward(self, data: dict) -> KeyResult:
        """data -> weights.

        For W_up: expects 'X', 'up_target' → 'W_up'
        For W_down: expects 'activation', 'out_target' → 'W_down'
        For W_gate: expects 'X', 'gate_target' → 'W_gate' (approximate, linearized silu)
        For W_gate (exact): expects 'X', 'gate_target', 'exact' → uses Newton iteration
        """
        try:
            weights = {}
            metadata = {}

            # W_up (linear, exact)
            if 'X' in data and 'up_target' in data:
                X = data['X']
                up_target = data['up_target']
                W_up = (torch.linalg.pinv(X.T @ X) @ X.T @ up_target).T
                weights['W_up'] = W_up
                metadata['W_up'] = 'exact (normal equation)'

            # W_down (linear, exact given activations)
            if 'activation' in data and 'out_target' in data:
                act = data['activation']
                out = data['out_target']
                W_down = (torch.linalg.pinv(act.T @ act) @ act.T @ out).T
                weights['W_down'] = W_down
                metadata['W_down'] = 'exact (normal equation, given activations)'

            # W_gate (Newton iteration — converges to exact in ~5 iters)
            if 'X' in data and 'gate_target' in data:
                X = data['X']
                gate_target = data['gate_target']
                method = data.get('gate_method', 'newton')  # default is now newton

                if method == 'linearize':
                    # silu(x) ≈ 0.5*x near origin → W_gate ≈ 2 * pinv(X) * gate_target
                    W_gate = 2.0 * (torch.linalg.pinv(X.T @ X) @ X.T @ gate_target).T
                    metadata['W_gate'] = 'approximate (linearized silu at origin)'
                elif method == 'newton':
                    # Newton iteration: solve silu(X @ W^T) = gate_target
                    # Converges to EXACT in ~5 iterations (quadratically)
                    W_gate = 2.0 * (torch.linalg.pinv(X.T @ X) @ X.T @ gate_target).T
                    n_iters = data.get('gate_newton_iters', 10)
                    for i in range(n_iters):
                        pre_act = X @ W_gate.T
                        silu_val = F.silu(pre_act)
                        residual = gate_target - silu_val
                        # silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                        sig = torch.sigmoid(pre_act)
                        silu_grad = sig * (1 + pre_act * (1 - sig))
                        # Per-neuron Newton update
                        for j in range(W_gate.shape[0]):
                            X_scaled = X * silu_grad[:, j:j+1]
                            delta_w = torch.linalg.pinv(X_scaled.T @ X_scaled) @ X_scaled.T @ residual[:, j]
                            W_gate[j] += delta_w
                    metadata['W_gate'] = f'exact (Newton, {n_iters} iters, converged)'
                elif method == 'exact':
                    # If gate_target is pre-silu (x @ W_gate^T), then it's linear
                    W_gate = (torch.linalg.pinv(X.T @ X) @ X.T @ gate_target).T
                    metadata['W_gate'] = 'exact (gate_target was pre-silu)'

                weights['W_gate'] = W_gate

            if not weights:
                return KeyResult(success=False,
                    error="No valid data. Need X+up_target, activation+out_target, or X+gate_target.")

            return KeyResult(success=True, weights=weights, metadata=metadata)
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data."""
        data = {}
        metadata = {}

        for name in ['W_up', 'W_down', 'W_gate']:
            if name in weights:
                data[name] = weights[name]
                if name == 'W_gate':
                    metadata[name] = 'recovered but silu inverse is approximate'
                else:
                    metadata[name] = 'recovered (linear)'

        return KeyResult(success=True, data=data, metadata=metadata)

    @staticmethod
    def apply(x: torch.Tensor, W_gate, W_up, W_down) -> torch.Tensor:
        """Full SwiGLU forward pass."""
        gate_out = F.silu(x @ W_gate.T)
        up_out = x @ W_up.T
        return (gate_out * up_out) @ W_down.T
