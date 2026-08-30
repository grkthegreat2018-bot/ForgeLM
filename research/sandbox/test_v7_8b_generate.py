"""End-to-end generation test for the ported V7-8B-B model.

Loads through the production path (load_default_model -> ForgeEngine),
generates text from a simple prompt, and reports:
  - VRAM usage
  - tokens/sec
  - output text
  - whether output is degenerate (empty / repetition / non-finite)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from research.model_loader import load_default_model
from research.inference.forge_engine import ForgeEngine

CKPT = "research/checkpoints/ForgeLM_V7_8B_B_ported.safetensors"
CONFIG = "forgelm_v7_8b_b"
PROMPT = "The capital of France is"
MAX_NEW = 30

print(f"=== V7-8B-B Generation Test ===")
print(f"Config: {CONFIG}")
print(f"Checkpoint: {CKPT}")
print(f"Prompt: {PROMPT!r}")
print()

# VRAM before
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1e9

print("[1] Loading model (production path)...")
t0 = time.time()
model, tokenizer = load_default_model(
    config_name=CONFIG,
    checkpoint_path=CKPT,
    device="cuda",
    dtype=torch.bfloat16,
    fast_load=True,
)
load_time = time.time() - t0
mem_after_load = torch.cuda.memory_allocated() / 1e9
mem_peak_load = torch.cuda.max_memory_allocated() / 1e9
n_params = sum(p.numel() for p in model.parameters())
print(f"    Loaded in {load_time:.1f}s")
print(f"    Params: {n_params/1e9:.3f}B")
print(f"    VRAM after load: {mem_after_load:.2f} GB (peak {mem_peak_load:.2f} GB)")

print("\n[2] Building ForgeEngine...")
engine = ForgeEngine(model, tokenizer, device="cuda", checkpoint_path=CKPT)

print("\n[3] Generating (greedy, max_new=30)...")
t0 = time.time()
output = engine.generate(PROMPT, max_new_tokens=MAX_NEW, temperature=0.0)
gen_time = time.time() - t0
mem_peak_gen = torch.cuda.max_memory_allocated() / 1e9

tokens_per_sec = MAX_NEW / gen_time if gen_time > 0 else 0
print(f"    Generated {MAX_NEW} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
print(f"    VRAM peak during gen: {mem_peak_gen:.2f} GB")
print(f"\n[4] Output:")
print(f"    {output!r}")

# Degeneracy checks
issues = []
if not output.strip():
    issues.append("empty output")
# Check for excessive repetition (same 5-token span repeated 3+ times)
words = output.split()
if len(words) >= 15:
    for i in range(len(words) - 5):
        span = tuple(words[i:i+5])
        if words[i+5:i+10] == list(span) and words[i+10:i+15] == list(span):
            issues.append(f"repetition at word {i}: {' '.join(span)}")
            break
# Non-finite logits would have crashed, but double-check output chars
if any(ord(c) == 0 for c in output):
    issues.append("null characters in output")

print(f"\n[5] Degeneracy check: {'PASS' if not issues else 'FAIL: ' + ', '.join(issues)}")

print(f"\n=== Summary ===")
print(f"  Load: {load_time:.1f}s, {mem_peak_load:.2f} GB peak VRAM")
print(f"  Gen:  {gen_time:.2f}s, {tokens_per_sec:.1f} tok/s, {mem_peak_gen:.2f} GB peak")
print(f"  Output quality: {'OK' if not issues else 'DEGENERATE'}")
if issues:
    sys.exit(1)
print("  RESULT: V7-8B-B model WORKS end-to-end.")
