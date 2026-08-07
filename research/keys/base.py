"""Key base class — the interface every key implements.

A Key converts between data and weights for one architectural component.
Each key has:
  - forward(data) -> weights   (replace training)
  - reverse(weights) -> data   (extract what was learned)
  - classify() -> KeyClass     (Partial / Bi / Full)

Keys are discovered by training a minimal probe, then finding the
closed-form algorithm that produces the same weights instantly.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch


class KeyClass(Enum):
    """Classification of key completeness."""
    PARTIAL = "partial"   # one direction only (forward OR reverse)
    BI = "bi"             # both directions work, round-trip is identity
    FULL = "full"         # Bi + composable with other Full keys for cross-arch
    TRIVIAL = "trivial"   # no weights to convert (fixed pattern / formula)


@dataclass
class KeyResult:
    """Result of a key operation."""
    success: bool
    weights: Optional[Dict[str, torch.Tensor]] = None
    data: Optional[Dict[str, torch.Tensor]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Key:
    """Base class for all keys.

    Subclasses must implement:
      - name: str (component name, e.g. "embedding", "rmsnorm")
      - forward(data: dict) -> KeyResult  (data -> weights)
      - reverse(weights: dict) -> KeyResult  (weights -> data)
      - key_class() -> KeyClass

    Optional:
      - cross_arch(weights_a: dict, key_b: 'Key') -> KeyResult  (weights A -> weights B)
    """

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def description(self) -> str:
        return ""

    def key_class(self) -> KeyClass:
        raise NotImplementedError

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights (replace training)."""
        raise NotImplementedError

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data (extract what was learned)."""
        raise NotImplementedError

    def cross_arch(self, weights_a: Dict[str, torch.Tensor],
                   key_b: 'Key') -> KeyResult:
        """weights(A) -> data -> weights(B). Requires both keys to be Bi."""
        # Default implementation: reverse through self, forward through key_b
        if self.key_class() not in (KeyClass.BI, KeyClass.FULL):
            return KeyResult(success=False, error=f"{self.name} is not Bi, cannot cross-arch")
        if key_b.key_class() not in (KeyClass.BI, KeyClass.FULL):
            return KeyResult(success=False, error=f"{key_b.name} is not Bi, cannot cross-arch")

        # Decode A to data
        decode = self.reverse(weights_a)
        if not decode.success:
            return KeyResult(success=False, error=f"Reverse {self.name} failed: {decode.error}")

        # Encode data to B
        encode = key_b.forward(decode.data)
        if not encode.success:
            return KeyResult(success=False, error=f"Forward {key_b.name} failed: {encode.error}")

        return KeyResult(
            success=True,
            weights=encode.weights,
            metadata={"source_arch": self.name, "target_arch": key_b.name,
                     "intermediate_data_keys": list(decode.data.keys())}
        )

    def __repr__(self):
        return f"Key({self.name}, {self.key_class().value})"
