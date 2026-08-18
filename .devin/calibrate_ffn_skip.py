"""Calibrate FFN skip threshold by measuring per-layer FFN input norms."""
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

# Hook to measure FFN input norms per layer
norms = {}
hooks = []
for i, block in enumerate(model.blocks):
    def make_hook(idx):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3:
                rms = x.float().pow(2).mean(dim=-1).sqrt().mean().item()
                norms.setdefault(idx, []).append(rms)
        return hook
    h = block.ln2.register_forward_pre_hook(
        lambda mod, args, idx=i: norms.setdefault(idx, []).append(
            args[0].float().pow(2).mean(dim=-1).sqrt().mean().item()))
    hooks.append(h)

# Run a few prompts
prompts = ["The capital of France is", "def fibonacci(n):", "Write a function to sort a list"]
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        _ = model(ids, use_cache=False)

# Remove hooks
for h in hooks:
    h.remove()

# Print stats
print("Layer | Mean RMS | Min | Max | Std")
print("-" * 50)
for i in range(16):
    if i in norms and norms[i]:
        vals = norms[i]
        m = sum(vals) / len(vals)
        print(f"  {i:2d}  | {m:.4f} | {min(vals):.4f} | {max(vals):.4f} | {(sum((v-m)**2 for v in vals)/len(vals))**0.5:.4f}")

# Suggest threshold (bottom 30th percentile)
all_norms = [v for vals in norms.values() for v in vals]
if all_norms:
    all_norms.sort()
    p30 = all_norms[int(len(all_norms) * 0.3)]
    p25 = all_norms[int(len(all_norms) * 0.25)]
    print(f"\nSuggested threshold (skip bottom 30%): {p30:.4f}")
    print(f"Suggested threshold (skip bottom 25%): {p25:.4f}")
