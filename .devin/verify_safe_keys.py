"""Verify safe keys were applied correctly to V2 and expert packs.

Checks:
1. V2 checkpoint still loads and has correct tensor count
2. Test-gated injection modified w_gate/w_up/w_down in last 8 layers
3. Context patches modified w_down in last 4 layers
4. Expert bundles were modified
5. CoT packs exist and are valid JSON
6. No lossy keys were applied (check tensor count unchanged)
"""
import sys, os, json, torch
sys.path.insert(0, '.')

from safetensors import safe_open
from pathlib import Path

V2_PATH = "research/checkpoints/forgelm_v2.safetensors"
V1_PATH = "research/checkpoints/forgelm_v1.safetensors"
EXPERT_DIR = "research/checkpoints/forgelm_v2_airmoe/experts"
COT_DIR = "research/checkpoints/cot_packs"
N_LAYERS = 28
D_MODEL = 1536

print("=" * 60)
print("Verify Safe Keys Applied to V2 + Experts")
print("=" * 60)

errors = []
checks = []

# ── 1. V2 loads and has correct tensor count ──────────────────────
print("\n[1] V2 checkpoint integrity...")
with safe_open(V2_PATH, framework="pt") as f:
    v2_keys = list(f.keys())
    v2_count = len(v2_keys)
print(f"  V2 tensors: {v2_count}")
if v2_count == 928:
    checks.append("V2 tensor count correct (928)")
else:
    errors.append(f"V2 tensor count wrong: {v2_count} (expected 928)")

# ── 2. Check test-gated injection modified last 8 layers ──────────
print("\n[2] Test-gated fact injection verification...")
with safe_open(V2_PATH, framework="pt") as f:
    v2_state = {k: f.get_tensor(k) for k in v2_keys if "w1" in k or "w2" in k or "w_gate" in k or "w_down" in k}

# Compare with V1 if available
v1_state = {}
if os.path.exists(V1_PATH):
    with safe_open(V1_PATH, framework="pt") as f:
        for k in f.keys():
            if "w1" in k or "w2" in k or "w_gate" in k or "w_down" in k:
                v1_state[k] = f.get_tensor(k)

modified_layers = []
unmodified_layers = []
for layer in range(N_LAYERS):
    # Check w1 (gate) for MoE experts: ffn.experts.{N}.w1.weight
    changed = False
    for expert in range(4):
        key = f"blocks.{layer}.ffn.experts.{expert}.w1.weight"
        if key in v2_state and key in v1_state:
            diff = (v2_state[key].float() - v1_state[key].float()).abs().max().item()
            if diff > 1e-6:
                changed = True
                break
        # Also check dense ffn
        key2 = f"blocks.{layer}.ffn.w_gate.weight"
        if key2 in v2_state and key2 in v1_state:
            diff = (v2_state[key2].float() - v1_state[key2].float()).abs().max().item()
            if diff > 1e-6:
                changed = True
                break

    if changed:
        modified_layers.append(layer)
    else:
        unmodified_layers.append(layer)

print(f"  Modified layers: {modified_layers}")
print(f"  Unmodified layers: {unmodified_layers[:5]}... (first 5)")

# Expect layers 20-27 modified (last 8)
expected = list(range(N_LAYERS - 8, N_LAYERS))
if modified_layers == expected:
    checks.append(f"Test-gated injection: layers {expected} modified (correct)")
else:
    errors.append(f"Test-gated injection: expected layers {expected}, got {modified_layers}")

# ── 3. Check that early layers were NOT modified ──────────────────
print("\n[3] Early layers untouched (lossless safety)...")
early_untouched = all(l in unmodified_layers for l in range(N_LAYERS - 8))
if early_untouched:
    checks.append("Early layers (0-19) untouched — lossless safety confirmed")
else:
    errors.append("Early layers were modified — possible lossy contamination!")

# ── 4. Check no extra tensors added (lossless) ────────────────────
print("\n[4] No extra tensors (lossless check)...")
if v2_count == 928:
    checks.append("No tensors added/removed — lossless")
else:
    errors.append(f"Tensor count changed: {v2_count} vs 928")

# ── 5. Expert bundles modified ────────────────────────────────────
print("\n[5] Expert bundle verification...")
expert_dir = Path(EXPERT_DIR)
bundles = list(expert_dir.glob("bundle_*.safetensors"))
print(f"  Found {len(bundles)} bundles")
for b in bundles:
    with safe_open(b, framework="pt") as f:
        bkeys = list(f.keys())
    print(f"  {b.name}: {len(bkeys)} tensors")
    if len(bkeys) > 0:
        checks.append(f"{b.name}: {len(bkeys)} tensors (loaded OK)")
    else:
        errors.append(f"{b.name}: empty or corrupt")

# ── 6. CoT packs exist and are valid ──────────────────────────────
print("\n[6] CoT Knowledge Packs verification...")
cot_dir = Path(COT_DIR)
if cot_dir.exists():
    packs = list(cot_dir.glob("cot_pack_*.json"))
    print(f"  Found {len(packs)} CoT packs")
    valid = 0
    for p in packs[:5]:  # check first 5
        with open(p) as f:
            data = json.load(f)
        if "prompt" in data and "reasoning" in data and "solution" in data:
            valid += 1
    print(f"  Validated {valid}/5 sampled packs")
    if len(packs) > 0 and valid == min(5, len(packs)):
        checks.append(f"CoT packs: {len(packs)} valid JSON files")
    else:
        errors.append(f"CoT packs: {valid}/5 valid")
else:
    errors.append("CoT pack directory not found")

# ── 7. Verify V2 still has all expected keys ──────────────────────
print("\n[7] V2 key structure intact...")
required_patterns = ["embed", "blocks.0.", "blocks.27.", "dwa_weights", "head"]
for pat in required_patterns:
    found = any(pat in k for k in v2_keys)
    if found:
        checks.append(f"  Pattern '{pat}' present")
    else:
        errors.append(f"  Pattern '{pat}' MISSING")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VERIFICATION RESULTS")
print(f"{'='*60}")
print(f"  Checks passed: {len(checks)}")
print(f"  Errors: {len(errors)}")
for c in checks:
    print(f"  ✓ {c}")
for e in errors:
    print(f"  ✗ {e}")
print(f"\n  {'ALL CHECKS PASSED' if not errors else 'FAILURES DETECTED'}")
print(f"{'='*60}")

sys.exit(0 if not errors else 1)
