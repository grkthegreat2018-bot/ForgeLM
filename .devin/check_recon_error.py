"""Check SVD+int4 reconstruction error vs original expert weights."""
import sys, os, torch
sys.path.insert(0, '.')
from safetensors.torch import load_file
from research.airmoe_infinite import InfiniteAirMoE
from research.bake_v4 import compress_expert_svd_int4

print("=" * 60)
print("SVD+int4 Reconstruction Error Check")
print("=" * 60)

# Load original v2 expert weights
state = load_file("research/checkpoints/forgelm_v2.safetensors")

for layer in [0, 14, 27]:
    for part in ["w1", "w2", "w3"]:
        key = f"blocks.{layer}.ffn.experts.0.{part}.weight"
        if key not in state:
            continue
        w_orig = state[key].float()
        
        # Compress
        comp = compress_expert_svd_int4(w_orig, svd_energy=0.99, use_int4=False)
        
        # Decompress (SVD only format)
        U = comp["U"].float()
        Vh = comp["Vh"].float()
        W_recon = (U * comp["S"].float().unsqueeze(0)) @ Vh
        
        # Error metrics
        diff = (w_orig - W_recon).abs()
        rel_error = diff.mean().item() / w_orig.abs().mean().item()
        max_error = diff.max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            w_orig.flatten().unsqueeze(0), W_recon.flatten().unsqueeze(0)).item()
        
        print(f"\n  Layer {layer} {part}: shape={w_orig.shape}")
        print(f"    SVD rank: {comp['rank'].item()}")
        print(f"    Cosine similarity: {cos_sim:.6f}")
        print(f"    Relative error: {rel_error:.4f} ({rel_error*100:.2f}%)")
        print(f"    Max abs error: {max_error:.6f}")
        print(f"    Orig: mean={w_orig.mean():.4f}, std={w_orig.std():.4f}")
        print(f"    Recon: mean={W_recon.mean():.4f}, std={W_recon.std():.4f}")
