"""Calibrate FFN skip using cosine similarity (correct FFN-SkipLLM metric).

Measures cosine similarity between FFN input and FFN output per layer.
High similarity = FFN is redundant (safe to skip).
Low similarity = FFN is important (cold region, never skip).
"""
import os, torch
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.model_loader import load_default_model

model, tok = load_default_model("forgelm_v3",
    checkpoint_path="research/checkpoints/ForgeLM_V3_Base.safetensors",
    device="cuda", dtype=torch.bfloat16)
model.eval()

# Hook to measure cosine similarity between FFN input and output
cos_sims = {}
hooks = []

for i, block in enumerate(model.blocks):
    def make_hook(idx):
        def hook(mod, args, output):
            # args[0] = FFN input (after ln2), output = FFN output
            ffn_in = args[0] if isinstance(args, tuple) else args
            if isinstance(output, tuple):
                ffn_out = output[0]
            else:
                ffn_out = output
            # Cosine similarity per token, then average
            # cos_sim(a, b) = dot(a,b) / (||a|| * ||b||)
            if ffn_in.dim() == 3 and ffn_out.dim() == 3:
                # Flatten to (B*T, C) for cosine sim
                flat_in = ffn_in.float().reshape(-1, ffn_in.size(-1))
                flat_out = ffn_out.float().reshape(-1, ffn_out.size(-1))
                # Per-token cosine similarity
                dot = (flat_in * flat_out).sum(-1)
                norm_in = flat_in.norm(dim=-1)
                norm_out = flat_out.norm(dim=-1)
                cos = (dot / (norm_in * norm_out + 1e-8)).mean().item()
                cos_sims.setdefault(idx, []).append(cos)
        return hook
    h = block.ffn.register_forward_hook(make_hook(i))
    hooks.append(h)

# Run prompts (prefill + decode)
prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Write a function to sort a list",
    "What is 2 + 2?",
    "Explain how binary search works",
]

print("=== PREFILL (all tokens) ===")
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        _ = model(ids, use_cache=False)

for h in hooks:
    h.remove()

print("\nLayer | Type  | Mean Cos Sim | Min    | Max    | Std")
print("-" * 60)
for i in range(16):
    if i in cos_sims and cos_sims[i]:
        vals = cos_sims[i]
        m = sum(vals) / len(vals)
        std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        ltype = model.blocks[i].layer_type
        flag = " <-- SKIP (high sim)" if m > 0.95 else ""
        flag2 = " <-- COLD (low sim)" if m < 0.85 else ""
        print(f"  {i:2d}  | {ltype:5s} | {m:.4f}      | {min(vals):.4f} | {max(vals):.4f} | {std:.4f}{flag}{flag2}")

# Identify cold and non-cold regions
all_sims = {i: sum(v) / len(v) for i, v in cos_sims.items() if v}
sorted_layers = sorted(all_sims.items(), key=lambda x: x[0])

print("\n=== REGION ANALYSIS ===")
print("Cold regions (low cos sim, FFN important — NEVER skip):")
for i, sim in sorted_layers:
    if sim < 0.85:
        print(f"  Layer {i}: {sim:.4f}")

print("\nNon-cold region (high cos sim, FFN redundant — SKIP candidates):")
skip_candidates = []
for i, sim in sorted_layers:
    if sim >= 0.95:
        print(f"  Layer {i}: {sim:.4f}")
        skip_candidates.append(i)

print(f"\nSkip candidates: {skip_candidates}")
print(f"Skip ratio: {len(skip_candidates)}/16 = {len(skip_candidates)/16*100:.0f}%")

# Suggest config
if skip_candidates:
    print(f"\n# Recommended config (static skip set):")
    print(f"ffn_skip_threshold = 0.95  # cosine similarity threshold")
    print(f"ffn_skip_layers = {skip_candidates}  # non-cold region layers")
