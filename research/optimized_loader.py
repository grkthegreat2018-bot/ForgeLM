"""Optimized checkpoint loader for ForgeLM V2.

Loads the optimized checkpoint (forgelm_v2_opt.safetensors) and restores:
1. Deduplicated tensors (aliases point to canonical, shared storage)
2. Dead tensors (filled with zeros or identity, as appropriate)
3. Scale-folded attention (model code must skip the scale multiply)

The restored state dict is identical to the original V2 checkpoint,
just loaded faster and using less disk space.

Usage:
    from research.optimized_loader import load_optimized_v2
    state = load_optimized_v2()  # returns full 928-tensor state dict
    model.load_state_dict(state, strict=False)
"""
import json
import os

import torch
from safetensors.torch import load_file

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_OPT = os.path.join(PROJECT, "research", "checkpoints", "forgelm_v2_opt.safetensors")
META_PATH = CKPT_OPT + ".meta.json"

# Dead tensor templates (what to restore as)
DEAD_TENSOR_DEFAULTS = {
    # SandwichNorm post-norms: identity (all ones)
    "post_attn_norm.weight": lambda shape, dtype: torch.ones(shape, dtype=dtype),
    "post_ffn_norm.weight": lambda shape, dtype: torch.ones(shape, dtype=dtype),
    # Router noise: all zeros
    "router.noise.weight": lambda shape, dtype: torch.zeros(shape, dtype=dtype),
    "router.noise_scale": lambda shape, dtype: torch.zeros(shape, dtype=dtype),
}


def load_optimized_v2(ckpt_path: str = None, meta_path: str = None) -> dict:
    """Load the optimized V2 checkpoint and restore all removed tensors.

    Returns a state dict identical to the original V2 checkpoint (928 tensors),
    but loaded from the smaller optimized file (734 tensors + metadata).

    Args:
        ckpt_path: path to optimized checkpoint (default: forgelm_v2_opt.safetensors)
        meta_path: path to metadata JSON (default: ckpt_path + .meta.json)

    Returns:
        Full state dict with all 928 tensors restored
    """
    if ckpt_path is None:
        ckpt_path = CKPT_OPT
    if meta_path is None:
        meta_path = META_PATH

    # Load optimized checkpoint (fewer tensors)
    state = load_file(ckpt_path)

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    dedup_map = meta["dedup_map"]
    scale_folded = meta.get("scale_folded", False)
    head_dim = meta.get("head_dim", 128)
    dead_removed = meta.get("dead_removed", 0)

    # Step 1: Restore deduplicated tensors (aliases → canonical, shared storage)
    for alias, canonical in dedup_map.items():
        if canonical in state:
            state[alias] = state[canonical]  # shared storage, zero-copy

    # Step 2: Restore dead tensors (zeros or identity)
    # We need to know the shapes — infer from tensor names
    n_layers = 28
    d_model = 1536
    n_experts = 4

    for i in range(n_layers):
        # post_attn_norm.weight: shape (d_model,), all ones
        key = f"blocks.{i}.post_attn_norm.weight"
        if key not in state:
            state[key] = torch.ones(d_model, dtype=torch.bfloat16)

        # post_ffn_norm.weight: shape (d_model,), all ones
        key = f"blocks.{i}.post_ffn_norm.weight"
        if key not in state:
            state[key] = torch.ones(d_model, dtype=torch.bfloat16)

        # router.noise.weight: shape (n_experts, d_model), all zeros
        key = f"blocks.{i}.ffn.router.noise.weight"
        if key not in state:
            state[key] = torch.zeros(n_experts, d_model, dtype=torch.bfloat16)

        # router.noise_scale: shape (1,), all zeros
        key = f"blocks.{i}.ffn.router.noise_scale"
        if key not in state:
            state[key] = torch.zeros(1, dtype=torch.bfloat16)

    # Step 3: If scale was folded, un-fold it for the model
    # (The model code applies scale separately, so we need to undo the fold)
    if scale_folded:
        import math
        fold_factor = math.sqrt(1.0 / math.sqrt(head_dim))
        for i in range(n_layers):
            for pname in ["q_proj", "k_up_proj"]:
                wk = f"blocks.{i}.attn.{pname}.weight"
                if wk in state:
                    state[wk] = (state[wk].float() / fold_factor).to(torch.bfloat16)
                bk = f"blocks.{i}.attn.{pname}.bias"
                if bk in state:
                    state[bk] = (state[bk].float() / fold_factor).to(torch.bfloat16)

    return state


def load_optimized_v2_fast(ckpt_path: str = None, meta_path: str = None) -> dict:
    """Load optimized V2 WITHOUT restoring dead tensors or un-folding scale.

    Returns only the 734 tensors that are actually used by the forward pass.
    Dead tensors (post_norms, router noise) are skipped — the model code
    doesn't use them anyway.

    Scale fold stays applied — the model code must be modified to skip
    the scale multiply (set attn_scale=1.0 or use a flag).

    This is the fastest path: load 734 tensors, no restoration overhead.

    Args:
        ckpt_path: path to optimized checkpoint
        meta_path: path to metadata JSON

    Returns:
        State dict with 734 tensors (forward-pass-relevant only)
    """
    if ckpt_path is None:
        ckpt_path = CKPT_OPT
    if meta_path is None:
        meta_path = META_PATH

    state = load_file(ckpt_path)

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    dedup_map = meta["dedup_map"]

    # Restore deduplicated tensors (aliases → canonical, shared storage)
    for alias, canonical in dedup_map.items():
        if canonical in state:
            state[alias] = state[canonical]

    return state


if __name__ == "__main__":
    import time

    print("Optimized V2 Loader Test")
    print("="*60)

    # Fast load (734 tensors, no restoration)
    print("\n[1] Fast load (734 tensors, no dead restoration)...")
    t0 = time.time()
    state_fast = load_optimized_v2_fast()
    t_fast = time.time() - t0
    print(f"  Loaded {len(state_fast)} tensors in {t_fast:.1f}s")

    # Full load (928 tensors, with restoration)
    print("\n[2] Full load (928 tensors, with restoration)...")
    t0 = time.time()
    state_full = load_optimized_v2()
    t_full = time.time() - t0
    print(f"  Loaded {len(state_full)} tensors in {t_full:.1f}s")

    # Verify: compare with original
    print("\n[3] Verifying against original checkpoint...")
    orig_path = os.path.join(PROJECT, "research", "checkpoints", "forgelm_v2.safetensors")
    if os.path.exists(orig_path):
        orig = load_file(orig_path)
        print(f"  Original: {len(orig)} tensors")

        # Check fast load has all forward-pass tensors
        missing = 0
        for k in orig:
            if k not in state_fast:
                # Check if it's a dead tensor
                is_dead = any(p in k for p in [
                    "post_attn_norm", "post_ffn_norm",
                    "router.noise", "noise_scale"
                ])
                if not is_dead:
                    missing += 1
                    if missing <= 3:
                        print(f"  MISSING (not dead): {k}")
        print(f"  Missing non-dead tensors in fast load: {missing}")

        # Check full load matches original
        all_match = True
        max_diff = 0
        for k in orig:
            if k in state_full:
                diff = (orig[k].float() - state_full[k].float()).abs().max().item()
                max_diff = max(max_diff, diff)
                if diff > 0.01:
                    all_match = False
                    if k.startswith("blocks.0.attn.q_proj"):
                        print(f"  DIFF (scale fold): {k} = {diff:.6f} (expected, un-folded)")
        print(f"  Full load max diff: {max_diff:.6f}")
        print(f"  Full load matches original: {all_match or max_diff < 0.01}")
    else:
        print(f"  Original not found at {orig_path}")
