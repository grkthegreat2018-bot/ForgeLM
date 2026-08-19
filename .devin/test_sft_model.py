"""Boot ForgeLM_V3_SFT on ForgeEngine — test with chat format."""
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
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from research.config import get_config
from research.model_loader import ModelLoader
from research.inference.forge_engine import ForgeEngine
from research.tokenizer_cache import get_tokenizer

device = "cuda"
dtype = torch.bfloat16

# ── Boot ──
t0 = time.perf_counter()
cfg = get_config("forgelm_v3", device=device)
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/ForgeLM_V3_SFT.safetensors", dtype=dtype)
model.to(device).eval()
tok = get_tokenizer()
engine = ForgeEngine(model, tok, device=device)
t_boot = time.perf_counter() - t0
print(f"Boot: {t_boot*1000:.0f}ms")

# ── Test generation with chat format ──
def chat_prompt(user_msg):
    """Format as chat conversation."""
    return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

prompts = [
    chat_prompt("What is the capital of France?"),
    chat_prompt("Write a Python function to reverse a list."),
    chat_prompt("What is 2+2?"),
    chat_prompt("Hello! How are you?"),
]

for prompt in prompts:
    user = prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    print(f"\n{'='*60}")
    print(f"User: {user}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    out = engine.generate(
        prompt,
        max_new_tokens=128,
        temperature=0.0,
    )
    t_gen = time.perf_counter() - t0
    # Clean output
    out_clean = out.replace("<|im_end|>", "").strip()
    n_tokens = len(tok.encode(out)) if out else 0
    print(f"Assistant: {out_clean[:300]}")
    print(f"Gen: {t_gen*1000:.0f}ms | {n_tokens} tokens | {n_tokens/t_gen:.0f} tok/s")

print(f"\nVRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
