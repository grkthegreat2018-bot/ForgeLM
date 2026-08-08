"""Analyze ForgeLM V2 checkpoint for novel LOSSLESS optimization opportunities.

Tests:
1. Tensor hash deduplication (exact duplicates from identity-init keys)
2. Exact low-rank decomposition (numerically rank-deficient weights)
3. Zero row/column pruning
4. Symmetric matrix detection
5. Block-diagonal structure detection
6. Router bias uniformity (softmax shift-invariance)
7. Embedding/head tying verification
8. Weight reordering for disk compression
9. Attention scale folding feasibility
10. Cross-layer weight similarity (partial sharing)
11. Expert weight similarity (shared vs routed)
12. Norm weight distribution (all-ones from identity init?)
"""
import sys
import os
import hashlib
import numpy as np
import torch
from safetensors import safe_open
from collections import defaultdict

CKPT = "research/checkpoints/forgelm_v2.safetensors"

def load_state():
    """Load all tensors from V2 checkpoint."""
    state = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state

def tensor_hash(t):
    """Hash a tensor's raw bytes (convert bf16 → uint16 view for numpy)."""
    arr = t.contiguous().view(torch.uint16) if t.dtype == torch.bfloat16 else t.contiguous()
    return hashlib.md5(arr.numpy().tobytes()).hexdigest()

def test_hash_dedup(state):
    """Test 1: Find exact duplicate tensors via hashing."""
    print("\n" + "="*60)
    print("TEST 1: Tensor Hash Deduplication")
    print("="*60)

    hash_map = defaultdict(list)
    for key, tensor in state.items():
        if tensor.dim() == 0:
            continue
        h = tensor_hash(tensor)
        hash_map[h].append(key)

    duplicates = {h: keys for h, keys in hash_map.items() if len(keys) > 1}
    total_dup_tensors = sum(len(keys) - 1 for keys in duplicates.values())
    dup_bytes = 0
    for h, keys in duplicates.items():
        # Size of one instance (all dupes are same size)
        t = state[keys[0]]
        dup_bytes += t.numel() * t.element_size() * (len(keys) - 1)

    print(f"  Total tensors: {len(state)}")
    print(f"  Unique hashes: {len(hash_map)}")
    print(f"  Duplicate groups: {len(duplicates)}")
    print(f"  Redundant tensors: {total_dup_tensors}")
    print(f"  Potential storage save: {dup_bytes / 1e6:.1f} MB")

    if duplicates:
        print("  Top duplicate groups:")
        for h, keys in sorted(duplicates.items(), key=lambda x: -len(x[1]))[:10]:
            t = state[keys[0]]
            sz = t.numel() * t.element_size() / 1e6
            print(f"    {len(keys)}x {keys[0]} (size: {sz:.2f} MB, shape: {tuple(t.shape)})")
            if len(keys) > 1:
                print(f"      also: {keys[1:3]}...")

    return duplicates, dup_bytes

def test_low_rank(state, threshold=1e-6):
    """Test 2: Find numerically rank-deficient weight matrices."""
    print("\n" + "="*60)
    print("TEST 2: Exact Low-Rank Decomposition")
    print("="*60)

    low_rank_found = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or tensor.shape[0] < 4 or tensor.shape[1] < 4:
            continue
        if tensor.numel() > 2_000_000:  # skip very large for speed
            continue
        # Use float32 for SVD
        t = tensor.float()
        try:
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        except Exception:
            continue
        # Check if numerically low-rank
        if S[0] < 1e-10:
            continue  # all-zero matrix
        rel_singular = S / S[0]
        # Find effective rank (where singular values drop to ~0)
        effective_rank = (rel_singular > threshold).sum().item()
        full_rank = min(tensor.shape)
        if effective_rank < full_rank * 0.5 and effective_rank < full_rank - 1:
            # Significant rank reduction possible
            original_bytes = tensor.numel() * tensor.element_size()
            # Store U[:, :r], S[:r], Vh[:r, :] — but in bf16
            r = effective_rank
            compressed_bytes = (tensor.shape[0] * r + r + r * tensor.shape[1]) * 2  # bf16
            ratio = compressed_bytes / original_bytes
            low_rank_found.append((key, effective_rank, full_rank, ratio, original_bytes / 1e6))

    low_rank_found.sort(key=lambda x: x[3])  # sort by compression ratio
    total_save = sum((1 - x[3]) * x[4] for x in low_rank_found)

    print(f"  Low-rank tensors found: {len(low_rank_found)}")
    print(f"  Potential storage save: {total_save:.1f} MB")
    if low_rank_found:
        print("  Top candidates (best compression):")
        for key, rank, full, ratio, sz in low_rank_found[:10]:
            print(f"    {key}: rank {rank}/{full}, ratio={ratio:.3f}, size={sz:.2f} MB")

    return low_rank_found

def test_zeros(state):
    """Test 3: Find zero rows/columns in weight matrices."""
    print("\n" + "="*60)
    print("TEST 3: Zero Row/Column Pruning")
    print("="*60)

    zero_rows = 0
    zero_cols = 0
    zero_tensors = 0
    for key, tensor in state.items():
        if tensor.dim() != 2:
            continue
        if tensor.abs().max().item() == 0:
            zero_tensors += 1
            continue
        # Zero rows
        row_zeros = (tensor.abs().sum(dim=1) == 0).sum().item()
        col_zeros = (tensor.abs().sum(dim=0) == 0).sum().item()
        zero_rows += row_zeros
        zero_cols += col_zeros

    print(f"  All-zero tensors: {zero_tensors}")
    print(f"  Zero rows found: {zero_rows}")
    print(f"  Zero columns found: {zero_cols}")

    return zero_rows, zero_cols

def test_symmetry(state):
    """Test 4: Find symmetric weight matrices (W = W^T)."""
    print("\n" + "="*60)
    print("TEST 4: Symmetric Matrix Detection")
    print("="*60)

    symmetric = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
            continue
        if tensor.shape[0] < 4:
            continue
        t = tensor.float()
        diff = (t - t.T).abs().max().item()
        if diff < 1e-6:
            symmetric.append((key, tensor.numel() * tensor.element_size() / 1e6))

    save = sum(s * 0.5 for _, s in symmetric)  # save half by storing upper triangle
    print(f"  Symmetric matrices: {len(symmetric)}")
    print(f"  Potential save: {save:.2f} MB")
    for key, sz in symmetric[:5]:
        print(f"    {key}: {sz:.2f} MB")

    return symmetric

def test_router_bias(state):
    """Test 6: Check if MoE router bias is uniform (can be removed)."""
    print("\n" + "="*60)
    print("TEST 6: Router Bias Uniformity")
    print("="*60)

    router_biases = {k: v for k, v in state.items() if "router" in k and "bias" in k}
    router_weights = {k: v for k, v in state.items() if "router" in k and "weight" in k}

    print(f"  Router bias tensors: {len(router_biases)}")
    print(f"  Router weight tensors: {len(router_weights)}")

    uniform_biases = 0
    for key, bias in router_biases.items():
        if bias.dim() == 1:
            is_uniform = (bias - bias[0]).abs().max().item() < 1e-7
            all_zero = bias.abs().max().item() < 1e-7
            if is_uniform:
                uniform_biases += 1
            print(f"    {key}: shape={tuple(bias.shape)}, uniform={is_uniform}, "
                  f"all_zero={all_zero}, range=[{bias.min():.4f}, {bias.max():.4f}]")

    print(f"  Uniform biases (removable): {uniform_biases}/{len(router_biases)}")

    return router_biases, uniform_biases

def test_embedding_tying(state):
    """Test 7: Verify embedding/head tying."""
    print("\n" + "="*60)
    print("TEST 7: Embedding / LM Head Tying")
    print("="*60)

    embed = state.get("embed.weight", state.get("embedding.weight"))
    head = state.get("head.weight", state.get("lm_head.weight"))

    if embed is not None and head is not None:
        if embed.shape == head.shape:
            diff = (embed.float() - head.float()).abs().max().item()
            print(f"  embed.weight shape: {tuple(embed.shape)}")
            print(f"  head.weight shape: {tuple(head.shape)}")
            print(f"  Max difference: {diff:.8f}")
            print(f"  Tied: {diff < 1e-7}")
            if diff < 1e-7:
                save = head.numel() * head.element_size() / 1e6
                print(f"  Already tied (save {save:.1f} MB if sharing storage)")
        else:
            print(f"  Different shapes — not tied")
    else:
        print(f"  embed: {embed is not None}, head: {head is not None}")

def test_norm_weights(state):
    """Test 12: Check norm weight distribution (identity init = all ones?)."""
    print("\n" + "="*60)
    print("TEST 12: Norm Weight Distribution (Identity Init Check)")
    print("="*60)

    norm_keys = [k for k in state if "norm" in k.lower() and "weight" in k]
    identity_norms = 0
    non_identity_norms = []
    for key in norm_keys:
        w = state[key]
        if w.dim() == 1:
            is_identity = (w - 1.0).abs().max().item() < 1e-6
            if is_identity:
                identity_norms += 1
            else:
                non_identity_norms.append((key, (w - 1.0).abs().max().item(),
                                           w.min().item(), w.max().item()))

    print(f"  Total norm weights: {len(norm_keys)}")
    print(f"  Identity (all-ones): {identity_norms}")
    print(f"  Non-identity: {len(non_identity_norms)}")
    if non_identity_norms:
        print("  Non-identity norms (top 5):")
        for key, maxdev, mn, mx in sorted(non_identity_norms, key=lambda x: -x[1])[:5]:
            print(f"    {key}: max_dev={maxdev:.6f}, range=[{mn:.4f}, {mx:.4f}]")

    return identity_norms, len(norm_keys)

def test_cross_layer_similarity(state, n_layers=28):
    """Test 10: Check cross-layer weight similarity for storage sharing."""
    print("\n" + "="*60)
    print("TEST 10: Cross-Layer Weight Similarity")
    print("="*60)

    # Group tensors by component type
    component_groups = defaultdict(list)
    for key in state:
        if key.startswith("blocks."):
            parts = key.split(".")
            layer_idx = int(parts[1])
            component = ".".join(parts[2:])
            component_groups[component].append((layer_idx, key))

    # For each component type, compute pairwise cosine similarity between layers
    high_sim_pairs = []
    for comp, layers in component_groups.items():
        if len(layers) != n_layers:
            continue
        # Get all tensors for this component
        tensors = []
        for layer_idx, key in sorted(layers):
            t = state[key].float().flatten()
            tensors.append((layer_idx, key, t))

        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                li, ki, ti = tensors[i]
                lj, kj, tj = tensors[j]
                # Cosine similarity
                cos = torch.nn.functional.cosine_similarity(
                    ti.unsqueeze(0), tj.unsqueeze(0)).item()
                if cos > 0.999:
                    sz = state[ki].numel() * state[ki].element_size() / 1e6
                    high_sim_pairs.append((ki, kj, cos, sz))

    high_sim_pairs.sort(key=lambda x: -x[2])
    print(f"  High-similarity pairs (cos > 0.999): {len(high_sim_pairs)}")
    if high_sim_pairs:
        total_save = sum(sz for _, _, _, sz in high_sim_pairs[:20])
        print(f"  Top pairs (potential sharing, NOT exact):")
        for ki, kj, cos, sz in high_sim_pairs[:10]:
            print(f"    cos={cos:.6f}, size={sz:.2f} MB")
            print(f"      {ki}")
            print(f"      {kj}")

    return high_sim_pairs

def test_expert_similarity(state):
    """Test 11: Check expert weight similarity within layers."""
    print("\n" + "="*60)
    print("TEST 11: Expert Weight Similarity (Within Layer)")
    print("="*60)

    # Find expert weights
    expert_keys = [k for k in state if "experts." in k and ".weight" in k]
    # Group by layer + part (w1, w2, w3)
    groups = defaultdict(list)
    for key in expert_keys:
        parts = key.split(".")
        layer = parts[1]
        # Find expert index and part
        expert_idx = None
        part = None
        for p in parts:
            if p in ("w1", "w2", "w3", "w_gate", "w_up", "w_down"):
                part = p
        for p in parts:
            if p.isdigit() and "expert" in str(parts[max(0, parts.index(p)-1)]):
                expert_idx = int(p)
        if expert_idx is not None and part is not None:
            groups[(layer, part)].append((expert_idx, key))

    identical_experts = 0
    for (layer, part), experts in groups.items():
        if len(experts) < 2:
            continue
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                ei, ki = experts[i]
                ej, kj = experts[j]
                ti = state[ki].float().flatten()
                tj = state[kj].float().flatten()
                cos = torch.nn.functional.cosine_similarity(
                    ti.unsqueeze(0), tj.unsqueeze(0)).item()
                if cos > 0.9999:
                    identical_experts += 1
                    sz = state[ki].numel() * state[ki].element_size() / 1e6
                    print(f"    IDENTICAL: {ki} == {kj} (cos={cos:.6f}, {sz:.2f} MB)")

    print(f"  Identical expert pairs (cos > 0.9999): {identical_experts}")

    # Also check shared vs routed
    shared_keys = [k for k in state if "shared." in k and ".weight" in k]
    print(f"  Shared expert weights: {len(shared_keys)}")

    return identical_experts

def test_compression_potential(state):
    """Test 8: Weight reordering for disk compression."""
    print("\n" + "="*60)
    print("TEST 8: Weight Reordering for Disk Compression")
    print("="*60)

    import io
    import zlib

    # Test: current ordering vs sorted-by-magnitude ordering
    total_original = 0
    total_compressed = 0
    total_sorted_compressed = 0

    for key, tensor in state.items():
        if tensor.numel() < 1000:
            continue
        raw = tensor.contiguous().numpy().tobytes()
        original_size = len(raw)
        compressed = len(zlib.compress(raw, level=6))

        # Sort by absolute value (group similar magnitudes)
        t_np = tensor.float().numpy()
        flat = t_np.flatten()
        sorted_idx = np.argsort(np.abs(flat))
        # Use float32 for sorted (can't cast back to bf16 via numpy)
        sorted_raw = flat[sorted_idx].astype(np.float32).tobytes()
        sorted_compressed = len(zlib.compress(sorted_raw, level=6))

        total_original += original_size
        total_compressed += compressed
        total_sorted_compressed += sorted_compressed

    print(f"  Total raw size: {total_original / 1e6:.1f} MB")
    print(f"  zlib compressed: {total_compressed / 1e6:.1f} MB (ratio: {total_compressed/total_original:.3f})")
    print(f"  zlib + sort: {total_sorted_compressed / 1e6:.1f} MB (ratio: {total_sorted_compressed/total_original:.3f})")
    print(f"  NOTE: sorted version is lossy (reordering changes output) — only for storage, must un-sort at load")

    return total_compressed / total_original

def test_attention_scale_folding(state):
    """Test 9: Attention scale folding feasibility."""
    print("\n" + "="*60)
    print("TEST 9: Attention Scale Folding")
    print("="*60)

    # Check if q_proj exists and what head_dim is
    q_proj_keys = [k for k in state if "q_proj" in k and "weight" in k]
    if q_proj_keys:
        q = state[q_proj_keys[0]]
        print(f"  q_proj shape: {tuple(q.shape)}")
        # For MLA: q_proj outputs n_heads * head_dim
        # head_dim = q.shape[0] / n_heads
        # scale = 1/sqrt(head_dim)
        # Can fold sqrt(scale) into q_proj rows
        print(f"  Scale = 1/sqrt(head_dim) can be folded into q_proj weight rows")
        print(f"  This eliminates 1 multiply per attention computation")
        print(f"  LOSSLESS: RoPE(Q * c) = c * RoPE(Q) since RoPE is linear")

    return q_proj_keys

def main():
    print("Loading ForgeLM V2 checkpoint...")
    state = load_state()
    print(f"Loaded {len(state)} tensors")

    # Total size
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"Total size: {total_bytes / 1e6:.1f} MB ({total_bytes / 1e9:.2f} GB)")

    # Run all tests
    test_hash_dedup(state)
    test_low_rank(state)
    test_zeros(state)
    test_symmetry(state)
    test_router_bias(state)
    test_embedding_tying(state)
    test_norm_weights(state)
    test_cross_layer_similarity(state)
    test_expert_similarity(state)
    test_attention_scale_folding(state)
    test_compression_potential(state)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
