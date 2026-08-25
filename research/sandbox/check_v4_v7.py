"""Check V4_Base checkpoint dimensions and V7_best quality."""
import sys, os
sys.path.insert(0, "D:\\windsurf\\ForgeAI")
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

import torch
from safetensors.torch import load_file as load_safetensors

# 1. Check V4_Base dimensions
print("=== V4_Base checkpoint ===")
v4_state = load_safetensors("research/checkpoints/ForgeLM_V4_Base.safetensors")
print(f"Total keys: {len(v4_state)}")
# Find d_model from first weight
for k in sorted(v4_state.keys())[:5]:
    print(f"  {k}: shape={v4_state[k].shape}, dtype={v4_state[k].dtype}")

# Check key patterns
import re
patterns = {}
for k in v4_state:
    pat = re.sub(r'blocks\.\d+\.', 'blocks.N.', k)
    pat = re.sub(r'layers\.\d+\.', 'layers.N.', pat)
    patterns[pat] = patterns.get(pat, 0) + 1

print(f"\nKey patterns:")
for pat, count in sorted(patterns.items()):
    print(f"  {pat}: {count}")

# Check dimensions
for k in sorted(v4_state.keys()):
    if 'ln1' in k or 'ln2' in k or 'post_attn' in k:
        print(f"  {k}: shape={v4_state[k].shape}")
        break

# Find embed/head
for k in sorted(v4_state.keys()):
    if 'embed' in k and 'weight' in k:
        print(f"  {k}: shape={v4_state[k].shape}")
    if 'head' in k and 'weight' in k:
        print(f"  {k}: shape={v4_state[k].shape}")

# Count layers
layer_nums = set()
for k in v4_state:
    m = re.match(r'blocks\.(\d+)\.', k)
    if m:
        layer_nums.add(int(m.group(1)))
print(f"  Layer count: {len(layer_nums)} (max={max(layer_nums) if layer_nums else 'N/A'})")

# Total params
total = sum(v.numel() for v in v4_state.values())
print(f"  Total params: {total/1e6:.1f}M")

# 2. Check V7_best
print(f"\n=== V7_best checkpoint ===")
v7_state = load_safetensors("research/checkpoints/ForgeLM_V7_best.safetensors")
print(f"Total keys: {len(v7_state)}")
for k in sorted(v7_state.keys())[:5]:
    print(f"  {k}: shape={v7_state[k].shape}, dtype={v7_state[k].dtype}")

# Check embed
for k in sorted(v7_state.keys()):
    if 'embed' in k and 'weight' in k:
        print(f"  {k}: shape={v7_state[k].shape}")

# Count layers
layer_nums7 = set()
for k in v7_state:
    m = re.match(r'blocks\.(\d+)\.', k)
    if m:
        layer_nums7.add(int(m.group(1)))
print(f"  Layer count: {len(layer_nums7)} (max={max(layer_nums7) if layer_nums7 else 'N/A'})")

total7 = sum(v.numel() for v in v7_state.values())
print(f"  Total params: {total7/1e6:.1f}M")

# Check if V7_best has same key structure as V7_8B_final
v7_patterns = {}
for k in v7_state:
    pat = re.sub(r'blocks\.\d+\.', 'blocks.N.', k)
    pat = re.sub(r'mtp_module\.heads\.\d+\.', 'mtp_module.heads.N.', pat)
    v7_patterns[pat] = v7_patterns.get(pat, 0) + 1

print(f"\nV7_best key patterns:")
for pat, count in sorted(v7_patterns.items())[:20]:
    print(f"  {pat}: {count}")
