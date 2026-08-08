"""Deep lossless analysis — find more optimization opportunities in V2.

Tests:
1. Delta encoding: store layer N+1 as diff from layer N (if similar)
2. int8-exact weights: are any weights exactly representable in int8?
3. Embedding compression: SVD, low-rank, or structured patterns in embed
4. Router skip: router is all-zero, can we remove it entirely?
5. Norm fusion: fold ln1/ln2 into attention/FFN (v3 technique, check if applicable)
6. Out_proj + ln1 fusion: fuse out_proj with next layer's ln1
7. KV cache structure: can compressed KV be compressed further?
8. Weight magnitude distribution: find tensors with limited dynamic range
9. Block-diagonal structure: are any weights block-diagonal?
10. Per-channel quantization: can individual channels be quantized losslessly?
11. Expert weight sharing: can experts share a base + delta?
12. Attention head pruning: do any heads contribute nothing?
13. fp8-eexact weights: are any weights exactly representable in fp8 (e4m3)?
14. Repeated rows/cols: are there repeated patterns within tensors?
15. Sparse residual: weight = dense_approx + sparse_residual (SharQ style)
"""
import os
import sys
import math
import torch
import numpy as np
from safetensors import safe_open
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
CKPT = "research/checkpoints/forgelm_v2.safetensors"


def load_state():
    state = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def test_delta_encoding(state):
    """Test 1: Delta encoding for consecutive layers."""
    print("\n" + "="*60)
    print("TEST 1: Delta Encoding (consecutive layers)")
    print("="*60)

    # Group by component type
    groups = defaultdict(list)
    for key in state:
        if key.startswith("blocks."):
            parts = key.split(".")
            layer = int(parts[1])
            comp = ".".join(parts[2:])
            groups[comp].append((layer, key))

    total_save = 0
    good_candidates = []
    for comp, layers in groups.items():
        if len(layers) < 2:
            continue
        layers.sort()
        # Compare consecutive layers
        for i in range(len(layers) - 1):
            _, k1 = layers[i]
            _, k2 = layers[i + 1]
            t1 = state[k1].float()
            t2 = state[k2].float()
            delta = t2 - t1
            # If delta is small, we can store delta in fewer bits
            delta_norm = delta.norm().item()
            orig_norm = t2.norm().item()
            ratio = delta_norm / (orig_norm + 1e-10)
            if ratio < 0.5:  # delta is significantly smaller
                sz = state[k2].numel() * state[k2].element_size() / 1e6
                # Estimate: delta in int8 = 1 byte vs 2 bytes bf16
                save = sz * 0.5
                total_save += save
                good_candidates.append((k1, k2, ratio, sz, save))

    good_candidates.sort(key=lambda x: x[2])
    print(f"  Delta-encodable pairs (ratio < 0.5): {len(good_candidates)}")
    print(f"  Estimated save (int8 delta): {total_save:.1f} MB")
    if good_candidates:
        print("  Top candidates (lowest delta ratio):")
        for k1, k2, ratio, sz, save in good_candidates[:10]:
            print(f"    {k2}: delta_ratio={ratio:.4f}, size={sz:.2f} MB, save={save:.2f} MB")

    return good_candidates


def test_int8_exact(state):
    """Test 2: Are any weights exactly representable in int8?"""
    print("\n" + "="*60)
    print("TEST 2: int8-Exact Weights")
    print("="*60)

    int8_exact = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or tensor.numel() < 100:
            continue
        t = tensor.float()
        # Check if all values are exactly representable as int8 * scale
        # int8 range: -128 to 127
        t_max = t.abs().max().item()
        if t_max == 0:
            continue
        scale = t_max / 127.0
        quantized = torch.round(t / scale).clamp(-128, 127)
        reconstructed = quantized * scale
        diff = (t - reconstructed).abs().max().item()
        if diff < 1e-6:
            int8_exact.append((key, t_max, scale, tensor.numel() * tensor.element_size() / 1e6))

    total_save = sum(sz * 0.5 for _, _, _, sz in int8_exact)  # int8 = 1 byte vs 2 bytes
    print(f"  int8-exact tensors: {len(int8_exact)}")
    print(f"  Estimated save: {total_save:.1f} MB")
    if int8_exact:
        for key, tmax, scale, sz in int8_exact[:5]:
            print(f"    {key}: max={tmax:.4f}, scale={scale:.6f}, size={sz:.2f} MB")

    return int8_exact


def test_fp8_exact(state):
    """Test 13: Are any weights exactly representable in fp8 (e4m3)?"""
    print("\n" + "="*60)
    print("TEST 13: fp8-Exact Weights (e4m3)")
    print("="*60)

    # fp8 e4m3: 4 exponent bits, 3 mantissa bits
    # Values: 0, subnormals, and normals up to 448
    # Check if bf16 values round-trip through fp8
    fp8_exact = []
    for key, tensor in state.items():
        if tensor.numel() < 100:
            continue
        t = tensor.float()
        # Simulate fp8 e4m3 quantization
        # fp8 e4m3fn: max=448, min normal=2^-6
        t_max = t.abs().max().item()
        if t_max == 0:
            continue
        if t_max > 448:
            continue  # can't represent in fp8 e4m3
        # Cast to fp8 via torch (if available) or simulate
        try:
            t_fp8 = t.to(torch.float8_e4m3fn)
            reconstructed = t_fp8.float()
            diff = (t - reconstructed).abs().max().item()
            if diff < 1e-6:
                save = tensor.numel() * (tensor.element_size() - 1) / 1e6
                fp8_exact.append((key, t_max, diff, save))
        except Exception:
            pass

    total_save = sum(s for _, _, _, s in fp8_exact)
    print(f"  fp8-exact tensors: {len(fp8_exact)}")
    print(f"  Estimated save: {total_save:.1f} MB")
    if fp8_exact:
        for key, tmax, diff, save in fp8_exact[:5]:
            print(f"    {key}: max={tmax:.4f}, diff={diff:.8f}, save={save:.2f} MB")

    return fp8_exact


def test_embedding_compression(state):
    """Test 3: Embedding compression analysis."""
    print("\n" + "="*60)
    print("TEST 3: Embedding Compression")
    print("="*60)

    embed = state.get("embed.weight")
    if embed is None:
        print("  No embedding found")
        return

    print(f"  Embedding shape: {tuple(embed.shape)}")
    print(f"  Embedding size: {embed.numel() * embed.element_size() / 1e6:.1f} MB")

    # SVD analysis
    t = embed.float()
    U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    # How many singular values for 99% energy?
    total_energy = (S ** 2).sum().item()
    cumsum = (S ** 2).cumsum(0) / total_energy
    r99 = (cumsum < 0.99).sum().item() + 1
    r999 = (cumsum < 0.999).sum().item() + 1
    r9999 = (cumsum < 0.9999).sum().item() + 1
    full_rank = min(t.shape)

    print(f"  Full rank: {full_rank}")
    print(f"  Rank for 99% energy: {r99} ({100*r99/full_rank:.1f}%)")
    print(f"  Rank for 99.9% energy: {r999} ({100*r999/full_rank:.1f}%)")
    print(f"  Rank for 99.99% energy: {r9999} ({100*r9999/full_rank:.1f}%)")

    # Check for exact low-rank
    rel_s = S / S[0]
    effective_rank = (rel_s > 1e-6).sum().item()
    print(f"  Effective rank (rel > 1e-6): {effective_rank}")

    # Check for repeated rows (vocab entries)
    print(f"  Checking for repeated embedding rows...")
    # Sample 1000 rows and check for duplicates
    n_checked = 0
    n_dupes = 0
    seen = {}
    for i in range(min(10000, t.shape[0])):
        row_hash = hash(t[i].numpy().tobytes())
        if row_hash in seen:
            n_dupes += 1
        else:
            seen[row_hash] = i
        n_checked += 1
    print(f"  Duplicates in first {n_checked} rows: {n_dupes}")

    # Check for zero rows
    zero_rows = (t.abs().sum(dim=1) == 0).sum().item()
    print(f"  Zero rows: {zero_rows}")

    return {"r99": r99, "r999": r999, "full_rank": full_rank, "zero_rows": zero_rows}


def test_router_skip(state):
    """Test 4: Router analysis — can we skip routing entirely?"""
    print("\n" + "="*60)
    print("TEST 4: Router Skip Analysis")
    print("="*60)

    router_keys = [k for k in state if "router" in k]
    print(f"  Router tensors: {len(router_keys)}")

    for key in router_keys[:5]:
        t = state[key]
        is_zero = (t.abs().max().item() == 0)
        print(f"    {key}: shape={tuple(t.shape)}, all_zero={is_zero}")

    # If router is all zeros, softmax(zeros) = uniform = 1/n_experts
    # With dense_bypass (moe_top_k=0), router is skipped entirely
    # So router weights are dead weight IF using dense_bypass
    router_bytes = sum(state[k].numel() * state[k].element_size() for k in router_keys)
    print(f"  Router total size: {router_bytes/1e6:.2f} MB")
    print(f"  If using dense_bypass: router is DEAD WEIGHT (can be pruned)")
    print(f"  If using routed MoE: router needs fine-tuning (currently untrained)")

    return router_keys


def test_norm_fusion(state):
    """Test 5: Norm fusion — check if ln1/ln2 can be fused into attention/FFN."""
    print("\n" + "="*60)
    print("TEST 5: Norm Fusion (ln1/ln2 into adjacent weights)")
    print("="*60)

    # Check if all ln1/ln2 are identity (all-ones)
    n_layers = 28
    all_identity = True
    for i in range(n_layers):
        for norm in ["ln1", "ln2"]:
            key = f"blocks.{i}.{norm}.weight"
            if key in state:
                w = state[key]
                if not (w == 1.0).all().item():
                    all_identity = False
                    print(f"    NON-IDENTITY: {key}: max_dev={(w-1).abs().max():.6f}")

    ln_f = state.get("ln_f.weight")
    ln_f_identity = ln_f is not None and (ln_f == 1.0).all().item()

    print(f"  ln1/ln2 all identity: {all_identity}")
    print(f"  ln_f identity: {ln_f_identity}")

    if all_identity:
        print(f"  → Norm Folding (v3 technique) is LOSSLESS and applicable")
        print(f"  → Would eliminate {n_layers * 2 + 1} norm operations per forward pass")
        print(f"  → But norms are needed for dynamic RMS scalar (can't fully remove)")
        print(f"  → Can skip the * weight multiply (weight=1.0 → no-op)")

    return all_identity


def test_repeated_rows_cols(state):
    """Test 14: Repeated rows/columns within tensors."""
    print("\n" + "="*60)
    print("TEST 14: Repeated Rows/Columns Within Tensors")
    print("="*60)

    repeated = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or tensor.shape[0] < 4 or tensor.shape[1] < 4:
            continue
        if tensor.numel() > 500000:  # skip large for speed
            continue
        t = tensor.float()

        # Check for repeated rows
        row_hashes = {}
        n_rep_rows = 0
        for i in range(t.shape[0]):
            h = hash(t[i].numpy().tobytes())
            if h in row_hashes:
                n_rep_rows += 1
            else:
                row_hashes[h] = i

        # Check for repeated columns
        col_hashes = {}
        n_rep_cols = 0
        for j in range(t.shape[1]):
            h = hash(t[:, j].numpy().tobytes())
            if h in col_hashes:
                n_rep_cols += 1
            else:
                col_hashes[h] = j

        if n_rep_rows > 0 or n_rep_cols > 0:
            repeated.append((key, n_rep_rows, t.shape[0], n_rep_cols, t.shape[1]))

    print(f"  Tensors with repeated rows/cols: {len(repeated)}")
    if repeated:
        for key, nr, tr, nc, tc in repeated[:10]:
            print(f"    {key}: {nr}/{tr} repeated rows, {nc}/{tc} repeated cols")

    return repeated


def test_weight_magnitude(state):
    """Test 8: Weight magnitude distribution — find low-dynamic-range tensors."""
    print("\n" + "="*60)
    print("TEST 8: Weight Magnitude Distribution")
    print("="*60)

    low_range = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or tensor.numel() < 100:
            continue
        t = tensor.float()
        t_max = t.abs().max().item()
        t_min = t.abs()[t.abs() > 0].min().item() if (t.abs() > 0).any() else 0
        if t_max == 0:
            continue
        dynamic_range = t_max / (t_min + 1e-10)
        # Low dynamic range = good quantization candidate
        if dynamic_range < 100:  # less than 100x range
            low_range.append((key, t_max, t_min, dynamic_range,
                              tensor.numel() * tensor.element_size() / 1e6))

    low_range.sort(key=lambda x: x[3])
    print(f"  Low-dynamic-range tensors (< 100x): {len(low_range)}")
    if low_range:
        print("  Top candidates (lowest dynamic range):")
        for key, tmax, tmin, dr, sz in low_range[:10]:
            print(f"    {key}: range=[{tmin:.6f}, {tmax:.6f}], DR={dr:.1f}, size={sz:.2f} MB")

    return low_range


def test_expert_sharing(state):
    """Test 11: Expert weight sharing — can experts share a base + delta?"""
    print("\n" + "="*60)
    print("TEST 11: Expert Base+Delta Sharing")
    print("="*60)

    # Group experts by layer and part
    groups = defaultdict(list)
    for key in state:
        if "experts." not in key or ".weight" not in key:
            continue
        parts = key.split(".")
        layer = parts[1]
        for p in parts:
            if p in ("w1", "w2", "w3", "w_gate", "w_up", "w_down"):
                part = p
        for p in parts:
            if p.isdigit() and int(p) < 10:
                expert_idx = int(p)
        groups[(layer, part)].append((expert_idx, key))

    shareable = []
    for (layer, part), experts in groups.items():
        if len(experts) < 2:
            continue
        # Compare each pair
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                ei, ki = experts[i]
                ej, kj = experts[j]
                t1 = state[ki].float()
                t2 = state[kj].float()
                delta = t2 - t1
                delta_norm = delta.norm().item()
                orig_norm = t2.norm().item()
                ratio = delta_norm / (orig_norm + 1e-10)
                if ratio < 0.3:  # delta is < 30% of original
                    sz = state[kj].numel() * state[kj].element_size() / 1e6
                    shareable.append((ki, kj, ratio, sz))

    shareable.sort(key=lambda x: x[2])
    print(f"  Shareable expert pairs (delta < 30%): {len(shareable)}")
    if shareable:
        total_save = sum(s * 0.75 for _, _, _, s in shareable)  # base + int8 delta
        print(f"  Estimated save: {total_save:.1f} MB")
        for k1, k2, ratio, sz in shareable[:10]:
            print(f"    {k2} vs {k1}: delta_ratio={ratio:.4f}, size={sz:.2f} MB")

    return shareable


def test_out_proj_fusion(state):
    """Test 6: Out_proj + next ln1 fusion."""
    print("\n" + "="*60)
    print("TEST 6: Out_proj + ln1 Fusion Analysis")
    print("="*60)

    # out_proj: [d_model, d_model], ln1: [d_model]
    # If ln1 is identity (weight=1.0), then out_proj is already "fused"
    # The dynamic RMS scalar can't be fused (it's data-dependent)
    n_layers = 28
    all_ln1_identity = True
    for i in range(n_layers):
        key = f"blocks.{i}.ln1.weight"
        if key in state:
            if not (state[key] == 1.0).all().item():
                all_ln1_identity = False

    print(f"  All ln1 identity: {all_ln1_identity}")
    if all_ln1_identity:
        print(f"  → ln1 weight multiply is a no-op (weight=1.0)")
        print(f"  → Can skip {n_layers} ln1 weight multiplies at runtime (lossless)")
        print(f"  → out_proj + ln1 fusion: ln1 weight already absorbed (identity)")

    # Check out_proj sizes
    out_proj_bytes = 0
    for i in range(n_layers):
        key = f"blocks.{i}.attn.out_proj.weight"
        if key in state:
            out_proj_bytes += state[key].numel() * state[key].element_size()
    print(f"  out_proj total: {out_proj_bytes/1e6:.1f} MB")

    return all_ln1_identity


def test_block_diagonal(state):
    """Test 9: Block-diagonal structure detection."""
    print("\n" + "="*60)
    print("TEST 9: Block-Diagonal Structure")
    print("="*60)

    block_diag = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
            continue
        if tensor.shape[0] < 64:
            continue
        if tensor.numel() > 500000:
            continue
        t = tensor.float()
        # Check if off-diagonal blocks are zero
        # Try block sizes that divide the dimension
        for bs in [64, 128, 256, 512]:
            if t.shape[0] % bs != 0:
                continue
            n_blocks = t.shape[0] // bs
            if n_blocks < 2:
                continue
            # Check off-diagonal blocks
            off_diag_max = 0
            for i in range(n_blocks):
                for j in range(n_blocks):
                    if i == j:
                        continue
                    block = t[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                    off_diag_max = max(off_diag_max, block.abs().max().item())
            if off_diag_max < 1e-6:
                block_diag.append((key, bs, n_blocks))
                break

    print(f"  Block-diagonal tensors: {len(block_diag)}")
    if block_diag:
        for key, bs, nb in block_diag[:5]:
            print(f"    {key}: block_size={bs}, n_blocks={nb}")

    return block_diag


def test_attn_head_pruning(state):
    """Test 12: Attention head pruning — do any heads contribute nothing?"""
    print("\n" + "="*60)
    print("TEST 12: Attention Head Analysis")
    print("="*60)

    # Check if any q_proj output rows (heads) are all zero
    n_layers = 28
    n_heads = 12
    head_dim = 128
    dead_heads = []

    for i in range(n_layers):
        q_key = f"blocks.{i}.attn.q_proj.weight"
        if q_key not in state:
            continue
        q = state[q_key].float()
        # q_proj shape: [n_heads * head_dim, d_model]
        # Each head occupies rows [h*head_dim : (h+1)*head_dim]
        for h in range(n_heads):
            head_weights = q[h*head_dim:(h+1)*head_dim]
            if head_weights.abs().max().item() < 1e-8:
                dead_heads.append((i, h))
                print(f"    DEAD HEAD: layer {i}, head {h} (all-zero q_proj rows)")

    # Also check out_proj columns (each head writes to specific columns)
    for i in range(n_layers):
        o_key = f"blocks.{i}.attn.out_proj.weight"
        if o_key not in state:
            continue
        o = state[o_key].float()
        # out_proj shape: [d_model, n_heads * head_dim]
        for h in range(n_heads):
            head_cols = o[:, h*head_dim:(h+1)*head_dim]
            if head_cols.abs().max().item() < 1e-8:
                dead_heads.append((i, h, "out_proj"))
                print(f"    DEAD HEAD (out_proj): layer {i}, head {h}")

    print(f"  Dead heads: {len(dead_heads)}")
    if not dead_heads:
        print(f"  All {n_layers * n_heads} heads are active (no pruning possible)")

    return dead_heads


def test_kv_compression(state):
    """Test 7: Can compressed KV (MLA) be compressed further?"""
    print("\n" + "="*60)
    print("TEST 7: KV Compression Analysis (MLA)")
    print("="*60)

    # kv_down_proj: [d_model, kv_compression_dim] = [1536, 512]
    # k_up_proj: [kv_compression_dim, n_heads * head_dim] = [512, 1536]
    # v_up_proj: [kv_compression_dim, n_heads * head_dim] = [512, 1536]
    n_layers = 28
    for i in range(min(3, n_layers)):
        kvd = state.get(f"blocks.{i}.attn.kv_down_proj.weight")
        kup = state.get(f"blocks.{i}.attn.k_up_proj.weight")
        vup = state.get(f"blocks.{i}.attn.v_up_proj.weight")
        if kvd is not None:
            print(f"  Layer {i}:")
            print(f"    kv_down_proj: {tuple(kvd.shape)}, rank={torch.linalg.matrix_rank(kvd.float()).item()}")
            if kup is not None:
                print(f"    k_up_proj: {tuple(kup.shape)}, rank={torch.linalg.matrix_rank(kup.float()).item()}")
            if vup is not None:
                print(f"    v_up_proj: {tuple(vup.shape)}, rank={torch.linalg.matrix_rank(vup.float()).item()}")

    # Check if kv_down_proj can be further compressed
    kvd = state.get("blocks.0.attn.kv_down_proj.weight")
    if kvd is not None:
        t = kvd.float()
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        rel_s = S / S[0]
        eff_rank = (rel_s > 1e-6).sum().item()
        print(f"  kv_down_proj effective rank: {eff_rank}/{min(t.shape)}")
        # Energy at various ranks
        total_e = (S**2).sum().item()
        for pct in [0.9, 0.95, 0.99]:
            cumsum = (S**2).cumsum(0) / total_e
            r = (cumsum < pct).sum().item() + 1
            print(f"    {pct*100:.0f}% energy at rank {r}")


def main():
    print("Deep Lossless Analysis — ForgeLM V2")
    print("="*60)

    state = load_state()
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"Loaded {len(state)} tensors, {total_bytes/1e6:.1f} MB")

    test_delta_encoding(state)
    test_int8_exact(state)
    test_fp8_exact(state)
    test_embedding_compression(state)
    test_router_skip(state)
    test_norm_fusion(state)
    test_out_proj_fusion(state)
    test_repeated_rows_cols(state)
    test_weight_magnitude(state)
    test_expert_sharing(state)
    test_block_diagonal(state)
    test_attn_head_pruning(state)
    test_kv_compression(state)

    print("\n" + "="*60)
    print("DEEP ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
