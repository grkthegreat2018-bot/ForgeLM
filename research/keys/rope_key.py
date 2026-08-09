"""RoPE key — Rotary Position Embedding.

Architecture: rotate(x, pos) = x * cos(pos*freq) + rotate90(x) * sin(pos*freq)

No learned weights. The key IS the formula. Data doesn't change it.

Classification: Bi (exact, no weights — it's a fixed rotation)
"""
import torch

from .base import Key, KeyClass, KeyResult


class RoPEKey(Key):
    @property
    def name(self) -> str:
        return "rope"

    @property
    def description(self) -> str:
        return "Rotary Position Embedding. Fixed rotation, no learned weights."

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def forward(self, data: dict) -> KeyResult:
        """No weights to produce — RoPE is a fixed formula."""
        return KeyResult(
            success=True,
            weights={},
            metadata={'note': 'RoPE has no learned weights. The key is the formula.'}
        )

    def reverse(self, weights: dict) -> KeyResult:
        """No weights to reverse — RoPE is a fixed formula."""
        return KeyResult(
            success=True,
            data={},
            metadata={'note': 'RoPE has no learned weights.'}
        )

    @staticmethod
    def compute_freqs(d_head: int, theta: float = 10000.0) -> torch.Tensor:
        """Compute RoPE frequencies."""
        return 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))

    @staticmethod
    def apply(x: torch.Tensor, positions: torch.Tensor,
              d_head: int, theta: float = 10000.0) -> torch.Tensor:
        """Apply RoPE rotation. x: [..., seq, d_head], positions: [seq]."""
        freqs = RoPEKey.compute_freqs(d_head, theta)
        angles = positions[:, None] * freqs[None, :]  # [seq, d_head/2]
        cos = angles.cos()
        sin = angles.sin()

        x1 = x[..., 0::2]  # even
        x2 = x[..., 1::2]  # odd
        # Rotate pairs
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

    @staticmethod
    def undo(x_rotated: torch.Tensor, positions: torch.Tensor,
             d_head: int, theta: float = 10000.0) -> torch.Tensor:
        """Inverse RoPE rotation (rotate by negative angle)."""
        freqs = RoPEKey.compute_freqs(d_head, theta)
        angles = positions[:, None] * freqs[None, :]
        cos = angles.cos()
        sin = -angles.sin()  # negative angle for inverse

        x1 = x_rotated[..., 0::2]
        x2 = x_rotated[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)
