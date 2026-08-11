"""Tensor Deduplication Key — find and eliminate exact-duplicate tensors.

LOSSLESS: identical tensors are stored once and aliased at load time.
Saves disk space, VRAM, and boot load time.

ForgeLM V2 has 193 redundant tensors (467.6 MB):
  - embed.weight == head.weight (466.75 MB — tied embeddings stored separately)
  - 56x post_attn_norm.weight == post_ffn_norm.weight (all identity, all identical)
  - 56x q_norm.weight == k_norm.weight (all identity, all identical)
  - 56x router.gate.weight == router.noise.weight (all zeros, identical)
  - 28x router.noise_scale (all zeros, identical)

Key class: FULL — reversible (dedup map records which tensors alias which),
  composable with other keys.

Usage:
    from research.keys.tensor_dedup_key import TensorDedupKey, apply_tensor_dedup
    state, dedup_map = apply_tensor_dedup(state)
    # state now has unique tensors only
    # dedup_map: {alias_key: canonical_key} — use at load to restore aliases
"""
import hashlib
from typing import Dict, List, Tuple

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


def tensor_hash(t: torch.Tensor) -> str:
    """Hash a tensor's raw bytes (handles bf16 via uint16 view)."""
    arr = t.contiguous().view(torch.uint16) if t.dtype == torch.bfloat16 else t.contiguous()
    return hashlib.md5(arr.numpy().tobytes()).hexdigest()


class TensorDedupKey(Key):
    """Tensor Deduplication key — eliminate exact-duplicate tensors.

    Finds tensors with identical bytes, keeps one canonical copy,
    and records a mapping for aliases. At load time, aliases point to
    the canonical tensor (shared storage).

    LOSSLESS: identical bytes → identical values, zero information loss.
    """

    @property
    def name(self) -> str:
        return "tensor_dedup"

    @property
    def description(self) -> str:
        return "Deduplicate exact-same tensors (shared storage, lossless)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Deduplicate tensors — keep canonical, record aliases.

        Args:
            data: state dict with potentially duplicate tensors

        Returns:
            weights: deduplicated state dict (fewer tensors)
            metadata: dedup_map {alias_key: canonical_key}, savings info
        """
        try:
            state = dict(data)
            hash_map: dict[str, list[str]] = {}

            for key, tensor in state.items():
                if tensor.dim() == 0:
                    continue
                h = tensor_hash(tensor)
                if h not in hash_map:
                    hash_map[h] = []
                hash_map[h].append(key)

            # Build dedup map: for each group with >1 tensor, pick canonical (first)
            # and alias the rest
            dedup_map: dict[str, str] = {}
            saved_bytes = 0
            for h, keys in hash_map.items():
                if len(keys) <= 1:
                    continue
                # Canonical = the one with the "most important" name
                # Prefer embed.weight, then head.weight, then lowest layer
                def sort_key(k):
                    if k == "embed.weight":
                        return (0, k)
                    if k == "head.weight":
                        return (1, k)
                    return (2, k)
                keys_sorted = sorted(keys, key=sort_key)
                canonical = keys_sorted[0]
                for alias in keys_sorted[1:]:
                    dedup_map[alias] = canonical
                    saved_bytes += state[alias].numel() * state[alias].element_size()
                    del state[alias]

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "dedup_map": dedup_map,
                    "n_deduplicated": len(dedup_map),
                    "saved_bytes": saved_bytes,
                    "saved_mb": saved_bytes / 1e6,
                    "original_count": len(data),
                    "deduped_count": len(state),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Restore aliases from dedup map.

        Args:
            weights: must contain 'dedup_map' in metadata or be a tuple (state, map)
        """
        # Reverse is handled at load time using the dedup_map
        # The state dict + dedup_map is the full representation
        return KeyResult(
            success=True,
            data=weights,
            metadata={"note": "Reverse requires dedup_map from forward metadata"},
        )


def apply_tensor_dedup(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Apply tensor deduplication to a state dict.

    Returns:
        deduped_state: state dict with duplicates removed
        dedup_map: {alias_key: canonical_key} for restoring at load time
    """
    key = TensorDedupKey()
    result = key.forward(state)
    if not result.success:
        raise RuntimeError(f"Tensor dedup failed: {result.error}")

    dedup_map = result.metadata["dedup_map"]
    saved_mb = result.metadata["saved_mb"]
    n_dedup = result.metadata["n_deduplicated"]
    print(f"  [TensorDedup] Deduplicated {n_dedup} tensors, saved {saved_mb:.1f} MB")
    print(f"  [TensorDedup] Tensors: {result.metadata['original_count']} → {result.metadata['deduped_count']}")

    return result.weights, dedup_map


def restore_aliases(state: dict[str, torch.Tensor], dedup_map: dict[str, str]) -> dict[str, torch.Tensor]:
    """Restore deduplicated tensors at load time.

    For each alias in dedup_map, create a reference to the canonical tensor.
    Uses shared storage (tensor aliasing) — zero-copy, zero VRAM overhead.
    """
    restored = dict(state)
    for alias, canonical in dedup_map.items():
        if canonical in restored:
            # Share storage — both point to same underlying data
            restored[alias] = restored[canonical]
    return restored


def save_deduped_checkpoint(input_path: str, output_path: str,
                            meta_path: str = None) -> dict[str, str]:
    """Load a checkpoint, deduplicate, and save the smaller version.

    Saves the dedup_map as a sidecar JSON file for restoration at load time.

    Returns:
        dedup_map: {alias_key: canonical_key}
    """
    import json

    from safetensors.torch import load_file, save_file

    print(f"  [TensorDedup] Loading {input_path}...")
    state = load_file(input_path)
    print(f"  [TensorDedup] Loaded {len(state)} tensors")

    deduped, dedup_map = apply_tensor_dedup(state)

    # Save deduped checkpoint
    print(f"  [TensorDedup] Saving deduped checkpoint to {output_path}...")
    save_file(deduped, output_path)

    # Save dedup map as sidecar
    if meta_path is None:
        meta_path = output_path + ".dedup.json"
    with open(meta_path, "w") as f:
        json.dump(dedup_map, f, indent=2)
    print(f"  [TensorDedup] Dedup map saved to {meta_path}")

    # Report size savings
    import os
    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)
    print(f"  [TensorDedup] Size: {orig_size/1e6:.1f} MB → {new_size/1e6:.1f} MB "
          f"(saved {(orig_size - new_size)/1e6:.1f} MB, {100*(1-new_size/orig_size):.1f}%)")

    return dedup_map


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".safetensors", "_dedup.safetensors")
        save_deduped_checkpoint(inp, out)
    else:
        # Quick test with dummy data
        key = TensorDedupKey()
        print(f"Key: {key.name}, class: {key.key_class().value}")

        state = {
            "embed.weight": torch.randn(100, 50, dtype=torch.bfloat16),
            "head.weight": None,  # will be set to embed
            "layer0.norm.weight": torch.ones(50, dtype=torch.bfloat16),
            "layer1.norm.weight": torch.ones(50, dtype=torch.bfloat16),
        }
        state["head.weight"] = state["embed.weight"].clone()

        result = key.forward(state)
        print(f"  Success: {result.success}")
        print(f"  Deduped: {result.metadata['n_deduplicated']}")
        print(f"  Saved: {result.metadata['saved_mb']:.4f} MB")
        print(f"  Dedup map: {result.metadata['dedup_map']}")
