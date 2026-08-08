"""Find all dead-weight tensors in V2 — keys in checkpoint but NOT used by forward pass.

V2 was built by applying identity-init keys to weights, but several keys
(SandwichNorm, DenseFormer, LogitCap, SwiGLUClamp) were never wired into
the model's forward pass. Their tensors sit in the checkpoint doing nothing.

This script identifies all such dead tensors so they can be pruned (losslessly).
"""
import os
import sys
import torch
from safetensors import safe_open

CKPT = os.path.join("research", "checkpoints", "forgelm_v2.safetensors")

# Tensors used by ConfigurableResearchLLM forward pass:
# - embed.weight
# - blocks.{i}.ln1.weight, blocks.{i}.ln2.weight
# - blocks.{i}.attn.q_proj.weight/bias, kv_down_proj.weight/bias, k_up_proj.weight/bias, v_up_proj.weight/bias
# - blocks.{i}.attn.q_norm.weight, k_norm.weight (if use_qk_norm, but skipped if identity)
# - blocks.{i}.ffn.w_gate.weight, w_up.weight, w_down.weight (dense) OR
# - blocks.{i}.ffn.experts.{e}.w1.weight, w3.weight, w2.weight (MoE)
# - blocks.{i}.ffn.shared.w1.weight, w3.weight, w2.weight (shared expert)
# - blocks.{i}.ffn.router.gate.weight (router)
# - ln_f.weight
# - head.weight (tied with embed)

USED_PATTERNS = [
    "embed.weight",
    "head.weight",  # tied
    "ln_f.weight",
    "blocks.",  # check per-block below
]

# Per-block used patterns (suffixes)
BLOCK_USED = [
    ".ln1.weight",
    ".ln2.weight",
    ".attn.q_proj.weight",
    ".attn.q_proj.bias",
    ".attn.kv_down_proj.weight",
    ".attn.kv_down_proj.bias",
    ".attn.k_up_proj.weight",
    ".attn.k_up_proj.bias",
    ".attn.v_up_proj.weight",
    ".attn.v_up_proj.bias",
    ".attn.q_norm.weight",   # identity, skipped at runtime, but loaded
    ".attn.k_norm.weight",   # identity, skipped at runtime, but loaded
    ".ffn.w_gate.weight",
    ".ffn.w_up.weight",
    ".ffn.w_down.weight",
    ".ffn.experts.",  # MoE experts
    ".ffn.shared.w1.weight",
    ".ffn.shared.w3.weight",
    ".ffn.shared.w2.weight",
    ".ffn.router.gate.weight",  # router (even if zeros, loaded for MoE)
]

# Dead patterns (in checkpoint but NOT in forward pass)
DEAD_PATTERNS = [
    ".post_attn_norm.weight",   # SandwichNorm — never in forward
    ".post_ffn_norm.weight",    # SandwichNorm — never in forward
    ".ffn.router.noise.weight", # MoE noise — not used in forward
    ".ffn.router.noise_scale",  # MoE noise scale — not used
    ".dwa_weights",             # DenseFormer — never in forward
    ".logit_cap",               # LogitCap — runtime flag, no weights
    ".swiglu_clamp",            # SwiGLUClamp — runtime flag, no weights
]

def main():
    print("Dead Tensor Analysis — ForgeLM V2")
    print("="*60)

    state = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    print(f"Total tensors: {len(state)}")

    # Categorize
    dead_tensors = []
    used_tensors = []
    unknown_tensors = []

    for key in state:
        is_dead = False
        for pattern in DEAD_PATTERNS:
            if pattern in key:
                is_dead = True
                break

        if is_dead:
            dead_tensors.append(key)
            continue

        # Check if it matches any used pattern
        is_used = False
        if key in ("embed.weight", "head.weight", "ln_f.weight"):
            is_used = True
        elif key.startswith("blocks."):
            for pattern in BLOCK_USED:
                if key.endswith(pattern) or pattern in key:
                    is_used = True
                    break
            if not is_used:
                # Check for expert weights more carefully
                if ".ffn.experts." in key and (".w1.weight" in key or ".w3.weight" in key or ".w2.weight" in key):
                    is_used = True

        if is_used:
            used_tensors.append(key)
        else:
            unknown_tensors.append(key)

    # Calculate sizes
    dead_bytes = sum(state[k].numel() * state[k].element_size() for k in dead_tensors)
    used_bytes = sum(state[k].numel() * state[k].element_size() for k in used_tensors)
    unknown_bytes = sum(state[k].numel() * state[k].element_size() for k in unknown_tensors)

    print(f"\nUsed tensors: {len(used_tensors)} ({used_bytes/1e6:.1f} MB)")
    print(f"Dead tensors: {len(dead_tensors)} ({dead_bytes/1e6:.1f} MB)")
    print(f"Unknown tensors: {len(unknown_tensors)} ({unknown_bytes/1e6:.1f} MB)")

    # Group dead tensors by pattern
    print(f"\nDead tensor breakdown:")
    for pattern in DEAD_PATTERNS:
        matches = [k for k in dead_tensors if pattern in k]
        if matches:
            sz = sum(state[k].numel() * state[k].element_size() for k in matches)
            print(f"  {pattern}: {len(matches)} tensors, {sz/1e6:.4f} MB")

    # Show unknown tensors
    if unknown_tensors:
        print(f"\nUnknown tensors (first 10):")
        for k in unknown_tensors[:10]:
            print(f"  {k}: shape={tuple(state[k].shape)}, {state[k].numel()*state[k].element_size()/1e6:.4f} MB")

    # Summary
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(state)} tensors, {total_bytes/1e6:.1f} MB")
    print(f"  Dead:  {len(dead_tensors)} tensors, {dead_bytes/1e6:.1f} MB ({100*dead_bytes/total_bytes:.2f}%)")
    print(f"  After pruning dead: {len(used_tensors)} tensors, {used_bytes/1e6:.1f} MB")
    print(f"  Save: {dead_bytes/1e6:.1f} MB")

    # Also check: are post_attn/post_ffn norms all identity (all-ones)?
    print(f"\n  Verify dead norms are identity (all-ones):")
    post_norms = [k for k in dead_tensors if "post_" in k and "norm" in k]
    all_identity = True
    for k in post_norms[:3]:
        w = state[k]
        is_id = (w == 1.0).all().item()
        print(f"    {k}: all_ones={is_id}, shape={tuple(w.shape)}")
        if not is_id:
            all_identity = False
    print(f"  All post-norms identity: {all_identity}")

    # Check router noise is all zeros
    print(f"\n  Verify router noise is all-zero:")
    noise_keys = [k for k in dead_tensors if "noise" in k]
    all_zero = True
    for k in noise_keys[:3]:
        w = state[k]
        is_zero = (w == 0).all().item()
        print(f"    {k}: all_zero={is_zero}, shape={tuple(w.shape)}")
        if not is_zero:
            all_zero = False
    print(f"  All noise tensors zero: {all_zero}")

if __name__ == "__main__":
    main()
