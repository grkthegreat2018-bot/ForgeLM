"""Quick response-quality check on a finished checkpoint."""
import torch
from transformers import AutoTokenizer
from research.model_loader import ModelLoader
from research.config import get_config
from research.fast_infer import FastInferenceEngine

import sys
CKPT = sys.argv[1] if len(sys.argv) > 1 else "research/checkpoints/distilled_llm.safetensors"
CFG = "360m_mla"
DEVICE = "cuda"

print(f"Loading {CKPT} ...")
cfg = get_config(CFG)
model = ModelLoader.build_model(cfg, checkpoint_path=CKPT)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

engine = FastInferenceEngine(model, tok, device=DEVICE)

PROMPTS = [
    "The capital of France is",
    "Once upon a time, a little girl named",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Question: What is 2 + 2?\nAnswer:",
]

print("\n" + "="*70)
print(f"GREEDY (temperature=0) — {CKPT}")
print("="*70)
import traceback
for p in PROMPTS:
    try:
        out = engine.generate(p, max_new_tokens=50, temperature=0.0)
        gen = out[len(p):].strip()
        print(f"\n[P] {p}")
        print(f"[G] {gen!r}")
    except Exception as e:
        print(f"\n[P] {p}")
        print(f"[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        break
