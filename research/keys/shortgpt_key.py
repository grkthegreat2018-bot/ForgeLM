"""ShortGPT Key — layer pruning based on representational similarity.

ShortGPT (Men et al., 2024) removes layers that contribute least to the
model's output. Layers that produce similar hidden states to the previous
layer are "redundant" and can be safely removed.

This is a training-free key: we measure layer similarity on calibration
data, identify the least important layers, and remove them.

Key insight: deeper layers in LLMs often have high cosine similarity with
their input — they're nearly identity functions. Removing them saves
compute proportional to the number of removed layers.

Usage:
    from research.keys.shortgpt_key import ShortGPTKey, prune_layers
    # Measure similarity and prune
    pruned_state = prune_layers(state, model, n_remove=4)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import Key, KeyClass, KeyResult


class ShortGPTKey(Key):
    """ShortGPT layer pruning key."""

    def __init__(self, n_remove: int = 4):
        self.n_remove = n_remove

    @property
    def name(self) -> str:
        return "shortgpt"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL  # Lossy — removed layers can't be recovered

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Not used — ShortGPT operates on the full model, not per-tensor."""
        return KeyResult(success=False, error="ShortGPT operates on full model, not per-tensor")

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=False, error="ShortGPT is not reversible (lossy)")


def compute_layer_similarity(model, input_ids: torch.Tensor) -> list[float]:
    """Compute cosine similarity between consecutive layer outputs.

    High similarity → layer is nearly identity → candidate for removal.

    Args:
        model: ConfigurableResearchLLM
        input_ids: [1, T] calibration tokens

    Returns:
        List of similarity scores per layer (0 to n_layers-1)
    """
    model.eval()
    similarities = []

    with torch.inference_mode():
        x = model.embed(input_ids)
        prev_hidden = x.clone()

        for i, block in enumerate(model.blocks):
            # Run block (no cache)
            out = block(x, past_key_value=None, use_cache=False)
            if isinstance(out, tuple):
                out = out[0]

            # Cosine similarity between input and output of this block
            cos = torch.nn.functional.cosine_similarity(
                prev_hidden.flatten(1), out.flatten(1), dim=-1
            ).mean().item()
            similarities.append(cos)

            prev_hidden = out.clone()
            x = out

    return similarities


def prune_layers(state: dict[str, torch.Tensor], model, input_ids: torch.Tensor,
                 n_remove: int = 4) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Prune the least important layers from a checkpoint.

    Args:
        state: Full model state dict
        model: Model instance (for measuring similarity)
        input_ids: Calibration tokens [1, T]
        n_remove: Number of layers to remove

    Returns:
        (pruned_state, removed_layer_indices)
    """
    print("  [ShortGPT] Computing layer similarities...")
    sims = compute_layer_similarity(model, input_ids)

    print("  [ShortGPT] Layer similarities:")
    for i, s in enumerate(sims):
        marker = " ← REMOVE" if s > sorted(sims, reverse=True)[-n_remove:][0] else ""
        print(f"    Layer {i:2d}: {s:.4f}{marker}")

    # Find layers to remove (highest similarity = most redundant)
    remove_indices = sorted(np.argsort(sims)[-n_remove:].tolist())
    print(f"  [ShortGPT] Removing layers: {remove_indices}")

    # Build pruned state dict
    pruned = {}
    kept_layers = [i for i in range(len(model.blocks)) if i not in remove_indices]

    for key, tensor in state.items():
        if key.startswith("blocks."):
            parts = key.split(".")
            layer_idx = int(parts[1])

            if layer_idx in remove_indices:
                continue  # Skip removed layers

            # Renumber: map old layer index to new
            new_idx = kept_layers.index(layer_idx)
            parts[1] = str(new_idx)
            new_key = ".".join(parts)
            pruned[new_key] = tensor
        else:
            pruned[key] = tensor

    n_orig = len([k for k in state if k.startswith("blocks.")])
    n_pruned = len([k for k in pruned if k.startswith("blocks.")])
    print(f"  [ShortGPT] {n_orig} block tensors → {n_pruned} block tensors")
    print(f"  [ShortGPT] Model: {len(model.blocks)}L → {len(kept_layers)}L")

    return pruned, remove_indices
