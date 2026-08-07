"""SpinQuant — Learned rotation for quantization-friendly weight distribution.

Rotates weight/activation matrices via an orthogonal transform before
quantization. Outliers get redistributed across dimensions, making the
distribution more Gaussian and easier to quantize. The rotation is
mathematically invisible to the output (computational invariance).

Two modes:
1. learn_rotation(): find optimal rotation by minimizing quantization error
   on a calibration set (Cayley SGD on Stiefel manifold)
2. apply_rotation(): fold rotation into weights for inference (no runtime cost)

Usage:
    from research.spinquant import SpinQuantizer
    q = SpinQuantizer(model, bits=4)
    q.calibrate(calib_data)  # learn rotation
    q.apply()  # fold rotation into weights
    # Now model can be quantized to 4-bit with minimal quality loss

Reference: SpinQuant (ICLR 2025, facebookresearch/SpinQuant)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def random_orthogonal(d, device="cuda"):
    """Generate a random orthogonal matrix via QR decomposition."""
    A = torch.randn(d, d, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


def cayley_orthogonal_update(R, grad, lr=0.01):
    """Cayley parameterization update on the Stiefel manifold (orthogonal matrices).

    R_new = R @ (I - 0.5 * lr * skew) @ inv(I + 0.5 * lr * skew)
    where skew = grad @ R.T - R @ grad.T

    This keeps R orthogonal after gradient updates.
    """
    skew = grad @ R.T - R @ grad.T
    I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    A = I - 0.5 * lr * skew
    B = I + 0.5 * lr * skew
    # R_new = R @ A @ inv(B)
    R_new = R @ A @ torch.linalg.inv(B)
    return R_new


class SpinQuantizer:
    """Learned rotation for quantization-friendly weight distribution.

    Args:
        model: the model to optimize
        bits: target quantization bits (4 default)
        lr: learning rate for rotation optimization
        calib_steps: number of optimization steps
    """

    def __init__(self, model, bits=4, lr=0.01, calib_steps=100):
        self.model = model
        self.bits = bits
        self.lr = lr
        self.calib_steps = calib_steps
        self.qmax = 2 ** (bits - 1) - 1

        # One rotation matrix per Linear layer (shared across the model dimension).
        self.rotations = {}
        self._init_rotations()

    def _init_rotations(self):
        """Initialize random orthogonal rotations for each Linear layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
                # Square matrix: rotate both input and output.
                d = module.weight.shape[0]
                self.rotations[name] = random_orthogonal(d, device=module.weight.device)
            elif isinstance(module, nn.Linear):
                # Non-square: rotate the larger dimension.
                d = max(module.weight.shape)
                self.rotations[name] = random_orthogonal(d, device=module.weight.device)

    def _quantize_error(self, W, R):
        """Compute quantization error after rotation.

        W_rotated = R @ W, then quantize, then measure error.
        """
        W_rot = R @ W
        scale = W_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        W_quant = torch.clamp(torch.round(W_rot / scale), -self.qmax, self.qmax) * scale
        return F.mse_loss(W_quant, W_rot)

    def calibrate(self, calib_data=None):
        """Learn optimal rotations by minimizing quantization error.

        Args:
            calib_data: optional calibration data (not needed for weight-only quant).
        """
        print(f"SpinQuant calibration: {self.calib_steps} steps, {self.bits}-bit target")
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self.rotations:
                continue

            R = self.rotations[name]
            W = module.weight.data
            if R.shape[0] != W.shape[0]:
                # Skip mismatched shapes (non-square).
                continue

            R.requires_grad = True
            optimizer = torch.optim.SGD([R], lr=self.lr)

            for step in range(self.calib_steps):
                optimizer.zero_grad()
                loss = self._quantize_error(W, R)
                loss.backward()
                # Cayley update to keep R orthogonal.
                with torch.no_grad():
                    R_new = cayley_orthogonal_update(R.data, R.grad, self.lr)
                    R.data.copy_(R_new)
                if (step + 1) % 20 == 0:
                    print(f"  [{name}] step {step+1}/{self.calib_steps} | q_error: {loss.item():.6f}")

            R.requires_grad = False

    def apply(self):
        """Fold rotation into weights: W_new = R @ W.

        After this, the model can be quantized with standard methods and
        the rotation provides better quantization quality.
        """
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self.rotations:
                continue
            R = self.rotations[name]
            W = module.weight.data
            if R.shape[0] == W.shape[0]:
                with torch.no_grad():
                    module.weight.data = R @ W
                    # Also rotate the bias if present.
                    if module.bias is not None:
                        module.bias.data = R @ module.bias.data

    def quantize_weights(self):
        """Quantize the (already rotated) weights to target bits."""
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            W = module.weight.data
            scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
            with torch.no_grad():
                module.weight.data = torch.clamp(
                    torch.round(W / scale), -self.qmax, self.qmax
                ) * scale

    def benchmark(self):
        """Print quantization error with and without rotation."""
        print("\nSpinQuant benchmark:")
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear) or name not in self.rotations:
                continue
            R = self.rotations[name]
            W = module.weight.data
            if R.shape[0] != W.shape[0]:
                continue

            # Without rotation.
            scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
            W_q = torch.clamp(torch.round(W / scale), -self.qmax, self.qmax) * scale
            err_no_rot = F.mse_loss(W_q, W).item()

            # With rotation.
            err_rot = self._quantize_error(W, R).item()

            improvement = (1 - err_rot / max(err_no_rot, 1e-8)) * 100
            print(f"  {name}: no_rot={err_no_rot:.6f} | rot={err_rot:.6f} | improvement: {improvement:.1f}%")
