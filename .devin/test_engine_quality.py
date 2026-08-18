"""Test ForgeLM V3 with ForgeEngine (proper inference path)."""
import os, torch
from pathlib import Path
env_path = Path("D:/windsurf/ForgeAI/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from research.model_loader import load_default_model
from research.tokenizer_cache import get_tokenizer
from research.inference.forge_engine import ForgeEngine

model, tokenizer = load_default_model("forgelm_v3",
    checkpoint_path="research/checkpoints/ForgeLM_V3_SFT.safetensors",
    device="cuda", dtype=torch.bfloat16)
model.eval()

engine = ForgeEngine(model, tokenizer)

# Test with ForgeEngine
prompts = [
    "def is_palindrome(s):\n    return",
    "Write a Python function to check if a string is a palindrome.\n\ndef is_palindrome(s):\n    ",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
]

for prompt in prompts:
    print(f"=== Prompt: {prompt[:60]}... ===")
    result = engine.generate(prompt, max_new_tokens=80, temperature=0.0)
    print(result)
    print()

# Also test with the model's own generate if it exists
print("=== Model.generate check ===")
print(f"Has generate: {hasattr(model, 'generate')}")
if hasattr(model, 'generate'):
    import inspect
    sig = inspect.signature(model.generate)
    print(f"Generate signature: {sig}")
