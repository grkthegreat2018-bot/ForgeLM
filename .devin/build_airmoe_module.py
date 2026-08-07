"""Build AirMoE module from ForgeLM v2.

Loads the model (which creates MoE experts at load time), extracts expert
weights from the live model, then saves them as individual files for
modular distribution.

Users download:
  1. base_model.safetensors (required — shared weights, no routed experts)
  2. Only the expert files they need (optional, ~50-200 MB each)
  3. Or topic bundles (group of experts in one file)

All output goes to D: drive.
"""
import sys, os, time, torch
sys.path.insert(0, '.')

from research.config import get_config
from research.model_loader import ModelLoader
from research.keys.airmoe_key import build_airmoe_module, build_topic_bundle, AirMoEManifest
from safetensors.torch import save_file
from pathlib import Path

SRC = "research/checkpoints/forgelm_v2.safetensors"
OUT = "D:/windsurf/ForgeAI/research/checkpoints/forgelm_v2_airmoe"

N_LAYERS = 28
N_EXPERTS = 4

# Assign topics to experts (for selective download)
EXPERT_TOPICS = {
    0: "math",
    1: "code",
    2: "reasoning",
    3: "general",
}


def main():
    print("=" * 70)
    print("Build AirMoE Module from ForgeLM V2")
    print("=" * 70)

    # Phase 1: Load model (creates MoE experts at load time)
    print(f"\n[1] Loading ForgeLM V2 (creates MoE experts)...")
    t0 = time.time()
    cfg = get_config("forgelm_v2", device="cpu")
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=SRC)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Phase 2: Extract expert weights from live model
    print(f"\n[2] Extracting expert weights from live model...")
    expert_state = {}  # layer_expert → {w1, w2, w3}
    base_state = {}    # everything else

    for name, param in model.named_parameters():
        if "ffn.experts." in name:
            # Parse: blocks.{i}.ffn.experts.{ei}.{w1/w2/w3}.weight
            expert_state[name] = param.data.clone()
        else:
            base_state[name] = param.data.clone()

    n_expert_tensors = len(expert_state)
    n_base_tensors = len(base_state)
    print(f"  Base tensors: {n_base_tensors}")
    print(f"  Expert tensors: {n_expert_tensors} "
          f"({N_LAYERS} layers x {N_EXPERTS} experts x 3 weights)")

    # Phase 3: Build AirMoE module
    print(f"\n[3] Building modular AirMoE module at {OUT}...")

    # First save base model (shared weights only)
    out = Path(OUT)
    experts_dir = out / "experts"
    out.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)

    base_path = out / "base_model.safetensors"
    print(f"  Saving base model ({n_base_tensors} tensors)...")
    save_file(base_state, str(base_path))
    base_size = base_path.stat().st_size
    print(f"  Base model: {base_size/1e6:.0f} MB")

    # Create manifest
    manifest = AirMoEManifest(
        model_name="ForgeLM-v2-airmoe",
        base_model="base_model.safetensors",
        n_layers=N_LAYERS,
        n_experts=N_EXPERTS,
        expert_dim=1792,
        compressed=True,
    )

    # Phase 4: Save each expert as individual file (with SVD compression)
    print(f"\n[4] Saving {N_LAYERS * N_EXPERTS} expert files (SVD compressed)...")
    total_expert_size = 0

    for i in range(N_LAYERS):
        for ei in range(N_EXPERTS):
            # Collect this expert's weights
            expert_weights = {}
            for part in ["w1", "w2", "w3"]:
                k = f"blocks.{i}.ffn.experts.{ei}.{part}.weight"
                if k in expert_state:
                    expert_weights[part] = expert_state[k]

            if not expert_weights:
                continue

            # SVD compress each weight
            compressed = {}
            svd_rank = 0
            for part, w in expert_weights.items():
                U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                cumsum = (S ** 2).cumsum(0)
                total_energy = cumsum[-1]
                k = max(1, (cumsum < 0.9 * total_energy).sum().item() + 1)
                svd_rank = max(svd_rank, k)
                compressed[f"{part}_U"] = U[:, :k].to(torch.bfloat16).contiguous()
                compressed[f"{part}_S"] = S[:k].to(torch.float16).contiguous()
                compressed[f"{part}_Vh"] = Vh[:k, :].to(torch.bfloat16).contiguous()

            shard_name = f"expert_l{i}_e{ei}.safetensors"
            shard_path = experts_dir / shard_name
            save_file(compressed, str(shard_path))
            shard_size = shard_path.stat().st_size
            total_expert_size += shard_size

            # Compute hash
            import hashlib
            h = hashlib.sha256()
            with open(shard_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            sha = h.hexdigest()[:16]

            topic = EXPERT_TOPICS.get(ei, "")
            manifest.add_expert(
                layer=i, expert_idx=ei,
                file_path=f"experts/{shard_name}",
                size_bytes=shard_size, sha256=sha,
                topic=topic, compressed=True, svd_rank=svd_rank,
            )

        if (i + 1) % 7 == 0:
            print(f"    Layer {i+1}/{N_LAYERS} done "
                  f"({total_expert_size/1e6:.0f} MB so far)")

    print(f"  Total expert size: {total_expert_size/1e6:.0f} MB")

    # Phase 5: Build topic bundles
    print(f"\n[5] Building topic bundles...")
    for topic in EXPERT_TOPICS.values():
        build_topic_bundle(manifest, topic, str(out))

    # Phase 6: Save manifest
    manifest_path = out / "manifest.json"
    manifest.save(str(manifest_path))

    # Summary
    print(f"\n{'='*70}")
    print(f"AirMoE Module Complete")
    print(f"{'='*70}")
    manifest.print_summary()

    print(f"\n  File layout:")
    print(f"    {OUT}/")
    print(f"      base_model.safetensors  ({base_size/1e6:.0f} MB) — REQUIRED")
    print(f"      manifest.json           — expert registry")
    print(f"      experts/")
    print(f"        expert_l0_e0.safetensors  (~{total_expert_size/(N_LAYERS*N_EXPERTS)/1e6:.0f} MB each)")
    print(f"        expert_l0_e1.safetensors")
    print(f"        ... ({N_LAYERS*N_EXPERTS} files)")
    print(f"        bundle_math.safetensors     (topic bundle)")
    print(f"        bundle_code.safetensors")
    print(f"        bundle_reasoning.safetensors")
    print(f"        bundle_general.safetensors")

    print(f"\n  Selective download:")
    print(f"    # Minimal: base + 1 topic")
    print(f"    # Download: base_model.safetensors + bundle_math.safetensors")
    print(f"    # Skip: bundle_code, bundle_reasoning, bundle_general")
    print(f"    # Saves: ~75% of download size")

    print(f"\n  VRAM usage at inference:")
    print(f"    Base model in VRAM: ~{base_size/1e9:.1f} GB")
    print(f"    + 2 experts cached: ~{2*total_expert_size/(N_LAYERS*N_EXPERTS)/1e6:.0f} MB")
    print(f"    + KV cache + activations: ~0.8 GB")
    print(f"    Total: ~{base_size/1e9 + 0.8:.1f} GB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
