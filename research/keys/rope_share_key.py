"""RoPE Buffer Sharing Key — share RoPE cos/sin buffers across all layers.

LOSSLESS: all layers use the same RoPE (same head_dim, same base, same max_seq_len).
Currently each layer creates its own RotaryEmbedding with its own cos_cached/sin_cached
buffers. These are identical across layers and can be shared.

For ForgeLM V2: 28 layers × 2 buffers (cos, sin) × 32768 × 128 × 4 bytes = 940 MB
of redundant buffers if they were persistent. They're currently non-persistent
(register_buffer with persistent=False), so they don't appear in the checkpoint,
but they ARE recreated per-layer in VRAM at boot time.

This key makes RoPE buffers shared (single instance, all layers reference it),
saving VRAM and reducing boot-time allocation.

Key class: TRIVIAL — runtime optimization, no weight changes.

Usage:
    from research.keys.rope_share_key import RoPEShareKey, apply_rope_sharing
    apply_rope_sharing(model)
"""
from typing import Dict

import torch
import torch.nn as nn

from .base import Key, KeyClass, KeyResult


class RoPEShareKey(Key):
    """RoPE Buffer Sharing key — share cos/sin buffers across all layers.

    All layers with the same head_dim and RoPE base use identical cos/sin
    buffers. This key makes them share a single buffer instance.

    LOSSLESS: identical buffers → identical values.
    Saves VRAM (28x fewer buffers) and boot-time allocation.

    Key class: TRIVIAL — runtime optimization, no weight changes.
    """

    @property
    def name(self) -> str:
        return "rope_share"

    @property
    def description(self) -> str:
        return "Share RoPE cos/sin buffers across all layers (save VRAM, faster boot)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """No weight transformation — this is a runtime optimization."""
        return KeyResult(
            success=True,
            weights=data,
            metadata={"note": "Runtime optimization — use apply_rope_sharing(model) instead"},
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data=weights)


def apply_rope_sharing(model: nn.Module) -> int:
    """Share RoPE buffers across all attention layers in a model.

    Makes all layers reference the first layer's RoPE buffers.
    Saves (n_layers - 1) × 2 × max_seq_len × head_dim × 4 bytes of VRAM.

    Args:
        model: ConfigurableResearchLLM with .blocks

    Returns:
        Number of layers that now share buffers
    """
    # Find the first layer's RoPE
    first_rope = None
    n_layers = 0

    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, 'rope'):
            if first_rope is None:
                first_rope = attn.rope
                n_layers += 1
            else:
                # Share buffers — point this layer's rope to the first one's buffers
                attn.rope.cos_cached = first_rope.cos_cached
                attn.rope.sin_cached = first_rope.sin_cached
                attn.rope.inv_freq = first_rope.inv_freq
                n_layers += 1

    if first_rope is not None:
        # Calculate VRAM save
        cos_bytes = first_rope.cos_cached.numel() * first_rope.cos_cached.element_size()
        sin_bytes = first_rope.sin_cached.numel() * first_rope.sin_cached.element_size()
        saved_mb = (n_layers - 1) * (cos_bytes + sin_bytes) / 1e6
        print(f"  [RoPEShare] Shared RoPE buffers across {n_layers} layers")
        print(f"  [RoPEShare] VRAM saved: {saved_mb:.1f} MB "
              f"({n_layers-1} layers × {(cos_bytes+sin_bytes)/1e6:.1f} MB/buffers)")

    return n_layers


if __name__ == "__main__":
    key = RoPEShareKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print(f"  Description: {key.description}")
    print("  Note: apply_rope_sharing(model) shares RoPE buffers at runtime")
