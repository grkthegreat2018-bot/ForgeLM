"""Compare base vs SFT model outputs."""
import os, sys, time
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ["FORGE_BITNET_KERNEL"] = "triton"
os.environ["FORGE_FUSED_ROPE_QKNORM"] = "1"

import torch
from research.config import get_config
from research.model_loader import ModelLoader
from research.inference.forge_engine import ForgeEngine
from research.tokenizer_cache import get_tokenizer

device = "cuda"
dtype = torch.bfloat16
tok = get_tokenizer()

def chat_prompt(msg):
    return f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"

def boot_and_test(checkpoint, label):
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"{'='*60}")
    cfg = get_config("forgelm_v3", device=device)
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=checkpoint, dtype=dtype)
    model.to(device).eval()
    engine = ForgeEngine(model, tok, device=device)

    for msg in ["What is the capital of France?", "What is 2+2?"]:
        prompt = chat_prompt(msg)
        out = engine.generate(prompt, max_new_tokens=64, temperature=0.0)
        print(f"  Q: {msg}")
        print(f"  A: {out.strip()[:200]}")
        print()
    del model, engine
    torch.cuda.empty_cache()

# Test base model
boot_and_test("research/checkpoints/ForgeLM_V3_Base.safetensors", "BASE (pre-SFT)")

# Test SFT model
boot_and_test("research/checkpoints/ForgeLM_V3_SFT.safetensors", "SFT (1000 steps)")
