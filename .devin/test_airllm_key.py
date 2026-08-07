"""Test AirLLM key: split XP model into shards, verify round-trip."""
import sys, torch
sys.path.insert(0, '.')
from research.keys.airllm_key import AirLLMKey

key = AirLLMKey()
print(f"Key: {key.name}, class: {key.key_class().value}")
print(f"Description: {key.description}")
print()

# Test 1: Split XP model into per-layer shards (no compression)
print("=== Test 1: Split XP model (no compression) ===")
result = key.forward({
    "checkpoint_path": "research/checkpoints/xp_full_no_mqa.safetensors",
    "output_dir": "research/checkpoints/xp_shards",
    "compression": None,
    "layer_prefix": "blocks",
    "extra_keys": ["embed.weight", "head.weight", "ln_f.weight"],
})
print(f"  Success: {result.success}")
if result.success:
    n = result.metadata["n_shards"]
    sz = result.metadata["total_size_gb"]
    names = result.weights["shard_names"][:3]
    print(f"  Shards: {n}")
    print(f"  Total size: {sz:.2f} GB")
    print(f"  First 3 shards: {names}")
else:
    print(f"  Error: {result.error}")
    sys.exit(1)

# Test 2: Verify round-trip (reassemble and compare)
print()
print("=== Test 2: Round-trip (reassemble shards) ===")
rev = key.reverse({
    "shard_dir": result.weights["shard_dir"],
    "shard_names": result.weights["shard_names"],
    "output_path": "research/checkpoints/xp_reassembled.safetensors",
})
print(f"  Success: {rev.success}")
if rev.success:
    print(f"  Tensors: {rev.data['n_tensors']}")

    # Compare a few tensors
    from safetensors import safe_open
    with safe_open("research/checkpoints/xp_full_no_mqa.safetensors", framework="pt") as f_orig:
        with safe_open("research/checkpoints/xp_reassembled.safetensors", framework="pt") as f_new:
            keys = list(f_orig.keys())[:5]
            all_match = True
            for kn in keys:
                t_orig = f_orig.get_tensor(kn)
                t_new = f_new.get_tensor(kn)
                match = torch.equal(t_orig, t_new)
                if not match:
                    diff = (t_orig.float() - t_new.float()).abs().max().item()
                    print(f"  MISMATCH {kn}: max diff = {diff}")
                    all_match = False
                else:
                    print(f"  OK {kn}: identical")
            if all_match:
                print("  All 5 sampled tensors identical!")
else:
    print(f"  Error: {rev.error}")

# Test 3: Check shard sizes (should be ~equal per layer)
print()
print("=== Test 3: Shard sizes ===")
from pathlib import Path
shard_dir = Path("research/checkpoints/xp_shards")
shards = sorted(shard_dir.glob("shard_*.safetensors"))
for s in shards[:5]:
    sz_mb = s.stat().st_size / 1e6
    print(f"  {s.name}: {sz_mb:.1f} MB")
print(f"  ... ({len(shards)} total shards)")
last = shards[-1]
print(f"  {last.name}: {last.stat().st_size / 1e6:.1f} MB")
