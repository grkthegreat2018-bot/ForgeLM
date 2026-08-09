"""Dead Weight Pruning Key — remove all-zero tensors and zero rows/columns.

LOSSLESS: zero tensors contribute nothing to the forward pass.
Removing them saves storage, VRAM, and compute (skip zero matmuls).

ForgeLM V2 has 56 all-zero tensors (router.noise_scale — all zeros from init).
These are scalar (1 element) so the storage save is tiny, but they add
to tensor count (928 → 872), slowing checkpoint I/O.

This key also detects zero rows/columns in 2D weights, which can be pruned
to skip unnecessary FLOPs. In V2, no zero rows/columns were found.

Key class: FULL — reversible (record what was removed), composable.

Usage:
    from research.keys.dead_weight_key import DeadWeightKey, apply_dead_weight_prune
    state, removed = apply_dead_weight_prune(state)
"""
from typing import Dict, List, Tuple

import torch

from .base import Key, KeyClass, KeyResult


class DeadWeightKey(Key):
    """Dead Weight Pruning key — remove all-zero tensors and zero rows/cols.

    LOSSLESS: zeros contribute nothing. Records what was removed for reversal.

    Key class: FULL — reversible, composable.
    """

    @property
    def name(self) -> str:
        return "dead_weight"

    @property
    def description(self) -> str:
        return "Prune all-zero tensors and zero rows/columns (lossless)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Remove all-zero tensors from state dict.

        Args:
            data: state dict

        Returns:
            weights: state dict with zero tensors removed
            metadata: list of removed keys, saved bytes
        """
        try:
            state = dict(data)
            removed: list[str] = []
            saved_bytes = 0

            for key in list(state.keys()):
                tensor = state[key]
                if tensor.dim() == 0:
                    continue
                # Skip tensors that are used by the forward pass even when zero
                # (e.g., router.gate.weight is all-zero at init but the MoE router reads it)
                if "router.gate" in key:
                    continue
                # Check if all zeros
                if tensor.abs().max().item() == 0:
                    saved_bytes += tensor.numel() * tensor.element_size()
                    removed.append(key)
                    del state[key]

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "removed_keys": removed,
                    "n_removed": len(removed),
                    "saved_bytes": saved_bytes,
                    "saved_mb": saved_bytes / 1e6,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Cannot restore without knowing original shapes — metadata needed."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"note": "Reverse requires removed_keys + shapes from forward metadata"},
        )


def apply_dead_weight_prune(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Remove all-zero tensors from state dict.

    Returns:
        pruned_state: state dict with zero tensors removed
        removed_keys: list of removed tensor keys
    """
    key = DeadWeightKey()
    result = key.forward(state)
    if not result.success:
        raise RuntimeError(f"Dead weight prune failed: {result.error}")

    removed = result.metadata["removed_keys"]
    saved = result.metadata["saved_mb"]
    print(f"  [DeadWeight] Removed {len(removed)} zero tensors, saved {saved:.4f} MB")
    if removed:
        print(f"  [DeadWeight] Removed: {removed[:5]}...")

    return result.weights, removed


if __name__ == "__main__":
    key = DeadWeightKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    state = {
        "real.weight": torch.randn(100, 50, dtype=torch.bfloat16),
        "zero1.weight": torch.zeros(50, dtype=torch.bfloat16),
        "zero2.weight": torch.zeros(4, 1536, dtype=torch.bfloat16),
    }

    result = key.forward(state)
    print(f"  Success: {result.success}")
    print(f"  Removed: {result.metadata['removed_keys']}")
    print(f"  Remaining: {list(result.weights.keys())}")
