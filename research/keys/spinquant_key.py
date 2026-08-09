"""SpinQuant Hadamard key — fixed Hadamard rotation, no calibration needed.

QuaRot (ICLR 2025) proved that a random Hadamard rotation is near-optimal for
outlier suppression.  This key replaces SpinQuant's learned Cayley-SGD rotation
with a deterministic Hadamard matrix, making it a TRIVIAL key (fixed formula,
no data needed).
"""
from typing import Dict

import torch
import torch.nn as nn

from .base import Key, KeyClass, KeyResult


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_matrix(d: int) -> torch.Tensor:
    """Generate a Hadamard matrix of size d (must be power of 2).

    Uses the recursive Sylvester construction:
        H_1 = [1]
        H_{2n} = [[H_n,  H_n],
                  [H_n, -H_n]]

    For non-power-of-2 dimensions the matrix is built at the next power of two
    and then cropped back to *d* × *d*.
    """
    if d <= 0:
        raise ValueError("dimension must be positive")
    size = _next_power_of_two(d)
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < size:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    # Normalise to be orthogonal (H @ H.T == I)
    H = H / torch.sqrt(torch.tensor(float(size), dtype=H.dtype))
    return H[:d, :d].contiguous()


class SpinQuantHadamardKey(Key):
    """SpinQuant with fixed Hadamard rotation — no calibration needed.

    Replaces learned Cayley SGD rotation with a deterministic Hadamard matrix.
    The Hadamard matrix H_d is orthogonal and distributes outliers uniformly
    across all dimensions, making quantization easier.

    Key class: TRIVIAL — fixed formula, no data needed.
    """

    @property
    def name(self) -> str:
        return "spinquant_hadamard"

    @property
    def description(self) -> str:
        return "Fixed Hadamard rotation for outlier suppression (QuaRot-style)."

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    # -- forward: data -> weights -------------------------------------------

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Rotate a weight tensor with a fixed Hadamard matrix.

        Expected input:  {"weight": tensor, "dim": int}
        Returned output: {"weight_rotated": tensor, "rotation": tensor}
        """
        try:
            weight = data["weight"]
            dim = int(data["dim"])
        except KeyError as exc:
            return KeyResult(success=False, error=f"missing key: {exc}")

        # Hadamard size = weight dimension along the rotation axis
        hdim = weight.shape[dim]
        H = hadamard_matrix(hdim).to(dtype=weight.dtype, device=weight.device)

        # Rotate along *dim* axis.  For nn.Linear weights (out_features,
        # in_features) dim=0 rotates rows (outputs), dim=1 rotates columns.
        if dim == 0:
            weight_rotated = H @ weight
        elif dim == 1:
            weight_rotated = weight @ H.t()
        else:
            return KeyResult(success=False, error=f"dim must be 0 or 1, got {dim}")

        return KeyResult(
            success=True,
            weights={"weight_rotated": weight_rotated, "rotation": H},
        )

    # -- reverse: weights -> data -------------------------------------------

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Invert the Hadamard rotation.

        Expected input:  {"weight_rotated": tensor, "rotation": tensor}
        Returned output: {"weight": tensor}

        Hadamard is symmetric (H.T == H) and orthogonal, so H @ H == I.
        """
        try:
            weight_rotated = weights["weight_rotated"]
            H = weights["rotation"]
        except KeyError as exc:
            return KeyResult(success=False, error=f"missing key: {exc}")

        # Determine rotation axis from tensor shapes
        if H.shape[0] == weight_rotated.shape[0] and H.shape[0] != weight_rotated.shape[1]:
            weight = H.t() @ weight_rotated          # undo row rotation
        elif H.shape[0] == weight_rotated.shape[1]:
            weight = weight_rotated @ H               # undo column rotation
        else:
            # Ambiguous — assume row rotation by default
            weight = H.t() @ weight_rotated

        return KeyResult(success=True, data={"weight": weight})


@torch.no_grad()
def apply_hadamard_to_model(model: nn.Module, bits: int = 4) -> nn.Module:
    """Apply a Hadamard rotation to every ``nn.Linear`` in *model* in-place.

    For each linear layer the output dimension is rotated (dim=0).  The
    rotation matrix is stored in a buffer named ``"hadamard_rotation"`` so the
    transformation can be inverted later.

    Returns the model (modified in-place).
    """
    key = SpinQuantHadamardKey()
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.data
        result = key.forward({"weight": weight, "dim": 0})
        if not result.success:
            raise RuntimeError(f"Hadamard forward failed: {result.error}")
        module.weight.data = result.weights["weight_rotated"]
        H = result.weights["rotation"]
        # Register as persistent buffer for later inversion
        if "hadamard_rotation" in dict(module.named_buffers()):
            module.hadamard_rotation = H
        else:
            module.register_buffer("hadamard_rotation", H)
    return model
