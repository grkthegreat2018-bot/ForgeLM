"""Check FFN input norms during decode (single token steps)."""
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

# Hook to measure FFN input norms during decode
norms_decode = {}
hooks = []
for i, block in enumerate(model.blocks):
    if block.layer_type != "conv":
        continue
    def make_hook(idx):
        def hook(mod, args):
            x = args[0]
            rms = x.float().pow(2).mean(dim=-1).sqrt().mean().item()
            norms_decode.setdefault(idx, []).append(rms)
        return hook
    h = block.ln2.register_forward_pre_hook(make_hook(i))
    hooks.append(h)

# Run decode steps
ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    out = model(ids, use_cache=True)
    past = out[2] if len(out) > 2 else out[1]
    next_tok = out[0][:, -1:].argmax(-1)
    for _ in range(5):
        out = model(next_tok, use_cache=True, past_key_values=past)
        past = out[2] if len(out) > 2 else out[1]
        next_tok = out[0][:, -1:].argmax(-1)

for h in hooks:
    h.remove()

print("Decode FFN input norms (conv layers):")
for i in sorted(norms_decode.keys()):
    vals = norms_decode[i]
    m = sum(vals) / len(vals)
    below = sum(1 for v in vals if v < 0.013)
    print(f"  Layer {i}: mean={m:.4f}, below_threshold={below}/{len(vals)}, vals={[round(v,4) for v in vals]}")
