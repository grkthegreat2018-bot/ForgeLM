"""Inspect the Jamba V2 checkpoint structure."""
from safetensors.torch import load_file

sd = load_file("research/checkpoints/ForgeLM_V2.safetensors")
keys = sorted(sd.keys())

# Block structure
blocks = set()
for k in keys:
    if k.startswith("blocks."):
        blocks.add(int(k.split(".")[1]))
print(f"Blocks: {sorted(blocks)} ({len(blocks)} total)")

# Block 0 (mamba)
b0 = [k for k in keys if k.startswith("blocks.0.")]
print(f"\nBlock 0 keys ({len(b0)}):")
for k in b0:
    print(f"  {k}: {sd[k].shape}")

# Block 7 (first attention per config)
b7 = [k for k in keys if k.startswith("blocks.7.")]
print(f"\nBlock 7 keys ({len(b7)}):")
for k in b7:
    print(f"  {k}: {sd[k].shape}")

# Block 21 (second attention)
b21 = [k for k in keys if k.startswith("blocks.21.")]
print(f"\nBlock 21 keys ({len(b21)}):")
for k in b21:
    print(f"  {k}: {sd[k].shape}")

# Top-level
print(f"\nembed.weight: {sd['embed.weight'].shape}")
head = [k for k in keys if "head" in k or "lm_head" in k or "unembed" in k]
print(f"Head keys: {head}")
norm = [k for k in keys if "norm" in k and "blocks" not in k]
print(f"Top-level norm: {norm}")

# Total
total = sum(v.numel() for v in sd.values())
print(f"\nTotal params: {total/1e9:.2f}B")
