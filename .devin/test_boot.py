"""Test boot with Triton kernels + clean fastsafetensors fallback."""
import os
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

bk = os.environ.get("FORGE_BITNET_KERNEL", "")
fq = os.environ.get("FORGE_FUSED_ROPE_QKNORM", "")
print(f"FORGE_BITNET_KERNEL={bk}")
print(f"FORGE_FUSED_ROPE_QKNORM={fq}")

import torch, time
from research.model_loader import load_default_model
from research.inference.forge_engine import ForgeEngine

model, tok = load_default_model("forgelm_v3",
    checkpoint_path="research/checkpoints/ForgeLM_V3_Base.safetensors",
    device="cuda", dtype=torch.bfloat16)
model.eval()
engine = ForgeEngine(model, tok)

# Warmup
_ = engine.generate("Hello", max_new_tokens=5, temperature=0.0)
torch.cuda.synchronize()

# Benchmark
t0 = time.time()
result = engine.generate("The capital of France is", max_new_tokens=50, temperature=0.0)
torch.cuda.synchronize()
gen_time = time.time() - t0
print(f"Generation: {50/gen_time:.1f} tok/s")
print(f"Output: {result[:80]}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
