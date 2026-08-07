"""Compare v1 and v2 checkpoint tensors to find what changed."""
import torch
from safetensors import safe_open

v1_path = "research/checkpoints/forgelm_v1.safetensors"
v2_path = "research/checkpoints/forgelm_v2.safetensors"

# Load keys
with safe_open(v1_path, framework="pt") as f:
    v1_keys = set(f.keys())
with safe_open(v2_path, framework="pt") as f:
    v2_keys = set(f.keys())

print(f"V1: {len(v1_keys)} tensors")
print(f"V2: {len(v2_keys)} tensors")

# Keys only in v2 (new transforms)
only_v2 = v2_keys - v1_keys
print(f"\nNew in v2 ({len(only_v2)}):")
for k in sorted(only_v2)[:20]:
    print(f"  {k}")
if len(only_v2) > 20:
    print(f"  ... and {len(only_v2)-20} more")

# Keys only in v1 (shouldn't be any)
only_v1 = v1_keys - v2_keys
print(f"\nMissing in v2 ({len(only_v1)}):")
for k in sorted(only_v1)[:10]:
    print(f"  {k}")

# Compare shared keys — check if any base weights were modified
print(f"\nComparing shared tensors...")
with safe_open(v1_path, framework="pt") as f1:
    with safe_open(v2_path, framework="pt") as f2:
        shared = v1_keys & v2_keys
        modified = []
        identical = 0
        for k in sorted(shared):
            t1 = f1.get_tensor(k)
            t2 = f2.get_tensor(k)
            if t1.shape != t2.shape:
                modified.append((k, "SHAPE MISMATCH", t1.shape, t2.shape))
            elif not torch.equal(t1, t2):
                diff = (t1.float() - t2.float()).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(
                    t1.float().flatten().unsqueeze(0),
                    t2.float().flatten().unsqueeze(0)).item()
                modified.append((k, f"diff={diff:.6f}, cos={cos:.6f}"))
            else:
                identical += 1

        print(f"  Identical: {identical}/{len(shared)}")
        print(f"  Modified:  {len(modified)}/{len(shared)}")
        
        if modified:
            print(f"\n  MODIFIED TENSORS (first 20):")
            for item in modified[:20]:
                if len(item) == 4:
                    print(f"    {item[0]}: {item[1]} {item[2]} -> {item[3]}")
                else:
                    print(f"    {item[0]}: {item[1]}")
            if len(modified) > 20:
                print(f"    ... and {len(modified)-20} more")
