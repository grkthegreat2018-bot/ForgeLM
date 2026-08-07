"""Verify that XP model actually stole all weights from Qwen."""
import sys, torch
sys.path.insert(0, '.')
from safetensors import safe_open
from safetensors.torch import load_file

orig_path = 'research/checkpoints/qwen25_coder_1.5b_ported.safetensors'
xp_path = 'research/checkpoints/xp_model_keystack.safetensors'

with safe_open(orig_path, framework='pt') as f:
    orig_keys = set(f.keys())
    orig_state = {k: f.get_tensor(k) for k in orig_keys}

xp_state = load_file(xp_path)
xp_keys = set(xp_state.keys())

print(f"Original Qwen: {len(orig_keys)} tensors")
print(f"XP model:      {len(xp_keys)} tensors")
print(f"Extra in XP (KeyStack additions): {sorted(xp_keys - orig_keys)}")
print(f"Missing in XP: {sorted(orig_keys - xp_keys)}")
print()

shared = orig_keys & xp_keys
print(f"Shared keys: {len(shared)}")
print()

mismatches = []
identical_count = 0
for k in sorted(shared):
    o = orig_state[k].float()
    x = xp_state[k].float()
    if o.shape != x.shape:
        mismatches.append((k, f"shape {o.shape} vs {x.shape}"))
        continue
    diff = (o - x).abs().max().item()
    if diff < 1e-6:
        identical_count += 1
    else:
        mismatches.append((k, f"maxdiff={diff:.6f}"))

print(f"Identical weights: {identical_count}/{len(shared)}")
print(f"Mismatches: {len(mismatches)}")
print()
if mismatches:
    print("MISMATCHED WEIGHTS (not properly stolen):")
    for k, reason in mismatches:
        print(f"  {k}: {reason}")
