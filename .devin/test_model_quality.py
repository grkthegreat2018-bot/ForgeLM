"""Test ForgeLM V3 output quality directly — is the model corrupted?"""
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

model, tokenizer = load_default_model("forgelm_v3",
    checkpoint_path="research/checkpoints/ForgeLM_V3_SFT.safetensors",
    device="cuda", dtype=torch.bfloat16)
model.eval()

# Test 1: Simple completion
prompt1 = "def is_palindrome(s):\n    return"
ids = tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=50, temperature=0.0, do_sample=False) if hasattr(model, 'generate') else None
if out is not None:
    print("=== Test 1: def is_palindrome(s): return ===")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
else:
    # Manual generation
    from research.inference.forge_engine import ForgeEngine
    engine = ForgeEngine(model, tokenizer)
    print("=== Test 1: def is_palindrome(s): return ===")
    result = engine.generate(prompt1, max_new_tokens=50, temperature=0.0)
    print(result)

print()

# Test 2: Chat-style prompt
prompt2 = "Write a Python function to check if a string is a palindrome.\n\ndef is_palindrome(s):\n"
print("=== Test 2: palindrome function ===")
if hasattr(model, 'generate'):
    ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out2 = model.generate(ids2, max_new_tokens=100, temperature=0.0, do_sample=False)
    print(tokenizer.decode(out2[0], skip_special_tokens=True))
else:
    result2 = engine.generate(prompt2, max_new_tokens=100, temperature=0.0)
    print(result2)

print()

# Test 3: Simple math
prompt3 = "What is 2 + 2? The answer is "
print("=== Test 3: simple math ===")
if hasattr(model, 'generate'):
    ids3 = tokenizer(prompt3, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out3 = model.generate(ids3, max_new_tokens=20, temperature=0.0, do_sample=False)
    print(tokenizer.decode(out3[0], skip_special_tokens=True))
else:
    result3 = engine.generate(prompt3, max_new_tokens=20, temperature=0.0)
    print(result3)

print()

# Test 4: Check what the tokenizer is doing
print("=== Tokenizer info ===")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# Test 5: Raw forward pass — check logits
print("\n=== Test 5: raw logits check ===")
ids5 = tokenizer("def fibonacci", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    logits = model(ids5)
    if isinstance(logits, tuple):
        logits = logits[0]
print(f"Logits shape: {logits.shape}")
print(f"Logits finite: {torch.isfinite(logits).all().item()}")
print(f"Top-5 tokens: {logits[0, -1].topk(5).indices.tolist()}")
print(f"Top-5 words: {[tokenizer.decode([t]) for t in logits[0, -1].topk(5).indices.tolist()]}")
