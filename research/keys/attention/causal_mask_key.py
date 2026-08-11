"""Causal mask key — fixed lower-triangular pattern.

No learned weights. The mask is a runtime pattern applied to attention scores
before softmax. It constrains what tokens can attend to what.

Classification: Trivial (no weights)
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class CausalMaskKey(Key):
    @property
    def name(self) -> str:
        return "causal_mask"

    @property
    def description(self) -> str:
        return "Causal mask. Fixed lower-triangular pattern. No learned weights."

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def forward(self, data: dict) -> KeyResult:
        """No weights to produce."""
        return KeyResult(success=True, weights={},
                        metadata={'note': 'Causal mask has no learned weights'})

    def reverse(self, weights: dict) -> KeyResult:
        """No weights to reverse."""
        return KeyResult(success=True, data={},
                        metadata={'note': 'Causal mask has no learned weights'})

    @staticmethod
    def create(seq_len: int, device=None) -> torch.Tensor:
        """Create causal mask: lower triangular, 1=attend, 0=blocked."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device))

    @staticmethod
    def apply(scores: torch.Tensor) -> torch.Tensor:
        """Apply causal mask to attention scores (set blocked to -inf)."""
        seq_len = scores.shape[-1]
        mask = CausalMaskKey.create(seq_len, device=scores.device)
        return scores.masked_fill(mask == 0, float('-inf'))
