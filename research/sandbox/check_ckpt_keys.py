"""Check checkpoint keys to diagnose model loading issues."""
import sys, os
sys.path.insert(0, "D:\\windsurf\\ForgeAI")
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

from safetensors import safe_open

ckpt = "research/checkpoints/ForgeLM_V7_8B_final.safetensors"
with safe_open(ckpt, framework="pt") as f:
    keys = list(f.keys())

print(f"Total keys: {len(keys)}")
print(f"\n=== First 30 keys ===")
for k in sorted(keys)[:30]:
    print(f"  {k}")

print(f"\n=== FFN-related keys (first 20) ===")
ffn_keys = [k for k in keys if 'ffn' in k]
print(f"Total FFN keys: {len(ffn_keys)}")
for k in sorted(ffn_keys)[:20]:
    print(f"  {k}")

print(f"\n=== Attention-related keys (first 20) ===")
attn_keys = [k for k in keys if 'attn' in k]
print(f"Total attn keys: {len(attn_keys)}")
for k in sorted(attn_keys)[:20]:
    print(f"  {k}")

print(f"\n=== MTP keys ===")
mtp_keys = [k for k in keys if 'mtp' in k]
for k in sorted(mtp_keys):
    print(f"  {k}")

print(f"\n=== Embed/head keys ===")
eh_keys = [k for k in keys if 'embed' in k or 'head' in k]
for k in sorted(eh_keys):
    print(f"  {k}")

print(f"\n=== Key pattern summary ===")
# Group by pattern
patterns = {}
for k in keys:
    # Replace block numbers with N
    import re
    pat = re.sub(r'blocks\.\d+\.', 'blocks.N.', k)
    pat = re.sub(r'mtp_module\.heads\.\d+\.', 'mtp_module.heads.N.', pat)
    patterns[pat] = patterns.get(pat, 0) + 1

for pat, count in sorted(patterns.items()):
    print(f"  {pat}: {count}")
