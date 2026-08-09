"""DoRA: Weight-Decomposed Low-Rank Adaptation.

Decomposes pre-trained weights into magnitude and direction, then applies
LoRA only to the direction. This separates the magnitude (which is frozen)
from the direction (which is adapted), giving better quality than vanilla LoRA.

Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (NVIDIA, 2024)
Result: Matches or exceeds full fine-tuning quality with 0.1% of params.

Key insight: vanilla LoRA updates W = W₀ + BA, which entangles magnitude
and direction. DoRA keeps magnitude fixed and only adapts direction:
  W = m * (V / ||V||), where V = W₀ + BA, m = ||W₀|| (frozen)

Usage:
    from research.architecture.dora import apply_dora_to_linear, apply_dora_to_model

    # Apply DoRA to all attention/FFN projections
    n = apply_dora_to_model(model, rank=16, alpha=32)
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn


class DoRALinear(nn.Module):
    """DoRA-wrapped Linear layer.

    Decomposes weight into magnitude (frozen) + direction (LoRA-adapted).
    Forward: y = x @ (m * normalize(W₀ + BA)).T + b

    Args:
        original: the nn.Linear to wrap
        rank: LoRA rank
        alpha: LoRA scaling factor (effective_scale = alpha / rank)
        dropout: dropout on LoRA input
    """

    def __init__(self, original: nn.Linear, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Detect device from original weight
        device = original.weight.data.device

        # Store frozen base weight (transposed for F.linear convention).
        self.base_weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False) if original.bias is not None else None

        # LoRA matrices (B @ A, low-rank) — create on same device as base weight.
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B starts at zero so initial output = base (no change).

        # Magnitude vector (frozen, computed from base weight).
        # m_i = ||W₀_i|| (norm of each output row).
        weight_norm = original.weight.data.norm(dim=1, keepdim=True)  # (out, 1)
        self.magnitude = nn.Parameter(weight_norm.clone(), requires_grad=True)  # learnable magnitude

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Compute adapted direction: V = W₀ + scaling * B @ A
        # Cast LoRA params to base weight dtype to avoid mixed-dtype errors
        bw_dtype = self.base_weight.dtype
        lora_delta = (self.lora_B.to(bw_dtype) @ self.lora_A.to(bw_dtype)) * self.scaling
        direction = self.base_weight + lora_delta  # (out, in)

        # Normalize direction and apply magnitude: W = m * V / ||V||
        direction_norm = direction.norm(dim=1, keepdim=True)  # (out, 1)
        direction_norm = direction_norm.clamp(min=1e-8)
        adapted_weight = self.magnitude.to(bw_dtype) * direction / direction_norm

        # Forward pass.
        x = self.dropout(x)
        return F.linear(x, adapted_weight, self.bias.to(bw_dtype) if self.bias is not None else None)

    def merge_and_unload(self) -> nn.Linear:
        """Merge DoRA weights back into a single Linear (for inference)."""
        with torch.no_grad():
            lora_delta = (self.lora_B @ self.lora_A) * self.scaling
            direction = self.base_weight + lora_delta
            direction_norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
            merged_weight = self.magnitude.to(self.base_weight.dtype) * direction / direction_norm
            merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
            merged.weight.data = merged_weight.to(self.base_weight.dtype)
            if self.bias is not None:
                merged.bias.data = self.bias.data.to(self.base_weight.dtype)
            # Ensure merged layer is on same device as original
            merged = merged.to(self.base_weight.device)
        return merged


# F is used in forward, import at module level.
import torch.nn.functional as F


def apply_dora_to_linear(module: nn.Linear, rank=16, alpha=32, dropout=0.0) -> DoRALinear:
    """Wrap a single Linear layer with DoRA."""
    return DoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)


def apply_dora_to_model(model, rank=16, alpha=32, dropout=0.0,
                        target_modules=None, verbose=True) -> int:
    """Apply DoRA to matching Linear layers in the model.

    Args:
        model: the model
        rank: LoRA rank (4-64 typical, higher = more capacity)
        alpha: scaling factor (usually 2x rank)
        dropout: LoRA dropout
        target_modules: list of module name substrings to target.
                       Default: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'w1', 'w2', 'w3', 'fc1', 'fc2']
        verbose: print stats

    Returns:
        number of layers wrapped
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'w1', 'w2', 'w3', 'fc1', 'fc2', 'qkv_proj', 'out_proj',
                         'kv_down_proj', 'k_up_proj', 'v_up_proj']

    n_wrapped = 0
    n_trainable = 0

    for name, module in model.named_modules():
        # Find parent module to replace the child.
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent.
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)

                # Replace with DoRA.
                dora = DoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, dora)
                n_wrapped += 1
                n_trainable += sum(p.numel() for p in dora.parameters() if p.requires_grad)
                break

    # Freeze all non-DoRA parameters.
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze DoRA params.
    for module in model.modules():
        if isinstance(module, DoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            module.magnitude.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  [DoRA] wrapped {n_wrapped} layers, rank={rank}, alpha={alpha}")
        print(f"  [DoRA] trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return n_wrapped


def merge_dora(model) -> int:
    """Merge all DoRA layers back into plain Linear (for inference speed)."""
    n_merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, DoRALinear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            merged = module.merge_and_unload()
            setattr(parent, child_name, merged)
            n_merged += 1
    print(f"  [DoRA] merged {n_merged} layers back to Linear")
    return n_merged
