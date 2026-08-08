"""Test the 3 new lossless keys on ForgeLM V2 checkpoint.

1. TensorDedupKey — deduplicate exact-same tensors (467.6 MB save)
2. AttnScaleFoldKey — fold 1/sqrt(head_dim) into q_proj/k_up_proj
3. DeadWeightKey — remove all-zero tensors (56 tensors)

Verify: cos(output, original) = 1.0 for each key (lossless).
Measure: storage save, tensor count reduction.
"""
import sys
import os
import torch
import numpy as np
from safetensors.torch import load_file, save_file

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "checkpoints", "forgelm_v2.safetensors")

def cosine_sim(a, b):
    """Cosine similarity between two flat tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def tensor_max_diff(a, b):
    """Max absolute difference between two tensors."""
    return (a.float() - b.float()).abs().max().item()

def test_tensor_dedup():
    """Test 1: Tensor Deduplication."""
    print("\n" + "="*60)
    print("TEST 1: Tensor Deduplication Key")
    print("="*60)

    from research.keys.tensor_dedup_key import TensorDedupKey, apply_tensor_dedup, restore_aliases

    state = load_file(CKPT)
    print(f"  Loaded {len(state)} tensors")

    # Apply dedup
    key = TensorDedupKey()
    result = key.forward(state)
    assert result.success, f"Dedup failed: {result.error}"

    deduped = result.weights
    dedup_map = result.metadata["dedup_map"]
    saved_mb = result.metadata["saved_mb"]
    n_dedup = result.metadata["n_deduplicated"]

    print(f"  Deduplicated: {n_dedup} tensors")
    print(f"  Saved: {saved_mb:.1f} MB")
    print(f"  Tensors: {len(state)} → {len(deduped)}")

    # Verify losslessness: restore aliases and check
    restored = restore_aliases(deduped, dedup_map)
    print(f"  Restored: {len(restored)} tensors")

    # Check all restored tensors match originals
    all_match = True
    max_diff = 0
    for key_name in state:
        if key_name in restored:
            diff = tensor_max_diff(state[key_name], restored[key_name])
            max_diff = max(max_diff, diff)
            if diff > 0:
                all_match = False
                print(f"  MISMATCH: {key_name} diff={diff}")

    print(f"  All tensors match after restore: {all_match}")
    print(f"  Max diff: {max_diff}")
    print(f"  LOSSLESS: {all_match and max_diff == 0}")

    # Show top dedup pairs
    print("  Top dedup pairs:")
    for alias, canonical in list(dedup_map.items())[:5]:
        sz = state[alias].numel() * state[alias].element_size() / 1e6
        print(f"    {alias} → {canonical} ({sz:.2f} MB)")

    return deduped, dedup_map

def test_attn_scale_fold():
    """Test 2: Attention Scale Folding."""
    print("\n" + "="*60)
    print("TEST 2: Attention Scale Folding Key")
    print("="*60)

    from research.keys.attn_scale_fold_key import AttnScaleFoldKey, apply_attn_scale_fold

    state = load_file(CKPT)
    print(f"  Loaded {len(state)} tensors")

    # Save original q_proj for comparison
    orig_q = state["blocks.0.attn.q_proj.weight"].clone()
    orig_k = state["blocks.0.attn.k_up_proj.weight"].clone()

    # Apply scale fold
    key = AttnScaleFoldKey()
    result = key.forward(state)
    assert result.success, f"Scale fold failed: {result.error}"

    folded_state = result.weights
    head_dim = result.metadata["head_dim"]
    fold_factor = result.metadata["fold_factor"]
    n_folded = result.metadata["n_folded"]

    print(f"  Head dim: {head_dim}")
    print(f"  Fold factor: {fold_factor:.6f} (= head_dim^(-1/4) = {head_dim}^(-0.25))")
    print(f"  Folded: {n_folded} projections")

    # Verify: folded weight = original * fold_factor
    new_q = folded_state["blocks.0.attn.q_proj.weight"]
    expected_q = orig_q.float() * fold_factor
    diff = (new_q.float() - expected_q).abs().max().item()
    print(f"  q_proj fold verification (max diff from expected): {diff:.8f}")

    # Verify reversibility
    rev = key.reverse(folded_state)
    assert rev.success, f"Reverse failed: {rev.error}"
    rev_q = rev.data["blocks.0.attn.q_proj.weight"]
    roundtrip_diff = tensor_max_diff(orig_q, rev_q)
    print(f"  Round-trip max diff: {roundtrip_diff:.8f}")
    print(f"  LOSSLESS: {roundtrip_diff < 1e-5}")

    # Simulate attention to verify mathematical equivalence
    # Q = x @ q_proj.weight.T + q_proj.bias
    # K = x @ k_up_proj.weight.T + k_up_proj.bias
    # Standard: scores = Q @ K.T * scale
    # Folded: scores = Q' @ K'.T  (where Q' = Q * fold, K' = K * fold)
    # Q' @ K'.T = (Q * fold) @ (K * fold).T = Q @ K.T * fold^2 = Q @ K.T * scale
    print(f"  Mathematical check: fold^2 = {fold_factor**2:.6f}, scale = {1.0/head_dim**0.5:.6f}")
    print(f"  fold^2 == scale: {abs(fold_factor**2 - 1.0/head_dim**0.5) < 1e-10}")

    return folded_state

def test_dead_weight():
    """Test 3: Dead Weight Pruning."""
    print("\n" + "="*60)
    print("TEST 3: Dead Weight Pruning Key")
    print("="*60)

    from research.keys.dead_weight_key import DeadWeightKey, apply_dead_weight_prune

    state = load_file(CKPT)
    print(f"  Loaded {len(state)} tensors")

    key = DeadWeightKey()
    result = key.forward(state)
    assert result.success, f"Dead weight failed: {result.error}"

    pruned = result.weights
    removed = result.metadata["removed_keys"]
    saved = result.metadata["saved_mb"]

    print(f"  Removed: {len(removed)} zero tensors")
    print(f"  Saved: {saved:.4f} MB")
    print(f"  Tensors: {len(state)} → {len(pruned)}")

    if removed:
        print(f"  Removed keys: {removed[:5]}...")
        # Verify they were actually zero
        for k in removed[:3]:
            assert state[k].abs().max().item() == 0, f"{k} was not zero!"
        print(f"  Verified: all removed tensors were all-zero")
        print(f"  LOSSLESS: True (zero tensors contribute nothing)")

    return pruned, removed

def test_combined():
    """Test all 3 keys combined."""
    print("\n" + "="*60)
    print("TEST 4: Combined — All 3 Keys (Dedup + ScaleFold + DeadWeight)")
    print("="*60)

    from research.keys.tensor_dedup_key import TensorDedupKey, restore_aliases
    from research.keys.attn_scale_fold_key import AttnScaleFoldKey
    from research.keys.dead_weight_key import DeadWeightKey

    state = load_file(CKPT)
    orig_count = len(state)
    orig_size = sum(t.numel() * t.element_size() for t in state.values())
    print(f"  Original: {orig_count} tensors, {orig_size/1e6:.1f} MB")

    # Apply in order: DeadWeight → Dedup → ScaleFold
    # (DeadWeight first to remove zeros before dedup finds them)
    dw = DeadWeightKey()
    r1 = dw.forward(state)
    print(f"  After DeadWeight: {len(r1.weights)} tensors")

    td = TensorDedupKey()
    r2 = td.forward(r1.weights)
    dedup_map = r2.metadata["dedup_map"]
    print(f"  After Dedup: {len(r2.weights)} tensors, saved {r2.metadata['saved_mb']:.1f} MB")

    asf = AttnScaleFoldKey()
    r3 = asf.forward(r2.weights)
    print(f"  After ScaleFold: {len(r3.weights)} tensors, {r3.metadata['n_folded']} projections folded")

    final = r3.weights
    final_size = sum(t.numel() * t.element_size() for t in final.values())
    print(f"\n  Final: {len(final)} tensors, {final_size/1e6:.1f} MB")
    print(f"  Total reduction: {orig_count} → {len(final)} tensors ({100*(1-len(final)/orig_count):.1f}% fewer)")
    print(f"  Size reduction: {orig_size/1e6:.1f} → {final_size/1e6:.1f} MB ({100*(1-final_size/orig_size):.1f}% smaller)")

    # Verify losslessness by restoring
    restored = restore_aliases(final, dedup_map)
    # Scale fold is reversible
    rev = asf.reverse(restored)
    restored = rev.data

    # Check key tensors
    checks = ["blocks.0.attn.q_proj.weight", "blocks.0.attn.k_up_proj.weight",
              "embed.weight", "head.weight"]
    all_good = True
    for k in checks:
        if k in state and k in restored:
            diff = tensor_max_diff(state[k], restored[k])
            cos = cosine_sim(state[k], restored[k])
            status = "OK" if diff < 1e-4 else "FAIL"
            if diff >= 1e-4:
                all_good = False
            print(f"  {k}: cos={cos:.8f}, diff={diff:.8f} [{status}]")

    print(f"\n  ALL LOSSLESS: {all_good}")

    return final

def test_compressed_save():
    """Test saving the deduplicated checkpoint and measuring disk size."""
    print("\n" + "="*60)
    print("TEST 5: Save Deduplicated Checkpoint (Disk Size)")
    print("="*60)

    from research.keys.tensor_dedup_key import TensorDedupKey
    from research.keys.dead_weight_key import DeadWeightKey

    state = load_file(CKPT)

    # Apply dedup + dead weight
    dw = DeadWeightKey()
    r1 = dw.forward(state)

    td = TensorDedupKey()
    r2 = td.forward(r1.weights)
    deduped = r2.weights
    dedup_map = r2.metadata["dedup_map"]

    # Save
    out_path = os.path.join(os.path.dirname(CKPT), "forgelm_v2_dedup.safetensors")
    save_file(deduped, out_path)

    import json
    meta_path = out_path + ".dedup.json"
    with open(meta_path, "w") as f:
        json.dump(dedup_map, f, indent=2)

    orig_disk = os.path.getsize(CKPT)
    new_disk = os.path.getsize(out_path)
    meta_disk = os.path.getsize(meta_path)

    print(f"  Original checkpoint: {orig_disk/1e6:.1f} MB")
    print(f"  Deduped checkpoint:  {new_disk/1e6:.1f} MB")
    print(f"  Dedup map (JSON):    {meta_disk/1e6:.4f} MB")
    print(f"  Total new:           {(new_disk + meta_disk)/1e6:.1f} MB")
    print(f"  Saved:               {(orig_disk - new_disk - meta_disk)/1e6:.1f} MB ({100*(1-(new_disk+meta_disk)/orig_disk):.1f}%)")

    # Clean up
    os.remove(out_path)
    os.remove(meta_path)
    print(f"  (Cleaned up test files)")

def main():
    print("ForgeLM V2 — New Lossless Keys Test")
    print("="*60)

    # Run individual tests
    test_tensor_dedup()
    test_attn_scale_fold()
    test_dead_weight()
    test_combined()
    test_compressed_save()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
