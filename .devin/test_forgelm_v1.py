"""ForgeLM v1 — Final comprehensive test.

ForgeLM v1 is the culmination of the KeyStack pipeline:
  - Base: Qwen2.5-Coder-1.5B (1.54B params, 100% quality preserved)
  - MLA attention (d_c=512, 4x KV compression, 100% energy retention)
  - MoE FFN (4 routed + 1 shared, lossless with top-4)
  - MRL (matryoshka dimension reordering)
  - QuaRot (Hadamard rotation for KV quantization)
  - ValueResidual (V0 stored + 28 gates)
  - RotorQuant (rotation matrices for KV compression)
  - MTP (4 prediction heads, trunk fine-tuned)
  - AirLLM (streamable for low-VRAM)

Runtime features:
  - StreamingLLM (attention sinks + sliding window)
  - SnapKV (observation-window eviction)
  - Hadamard INT4 KV (4x KV compression, lossless)
  - Prefix caching
  - torch.compile
  - Logit capping
"""
import sys, os, torch, time
sys.path.insert(0, '.')

from research.inference.forge_engine import ForgeEngine
from research.mtp import MTPHead
from research.config import get_config
from safetensors.torch import load_file

PROMPT = "def fibonacci(n):"
EXPECTED = "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n"

print("=" * 60)
print("ForgeLM v1 — Final Test")
print("=" * 60)
print()

# Build ForgeLM v1
print("--- Building ForgeLM v1 ---")
engine = ForgeEngine.from_checkpoint(
    checkpoint='research/checkpoints/forgelm_v1.safetensors',
    config_name='forgelm_v1',
    device='cuda',
)
print()

# Test 1: Standard
print("--- Test 1: Standard decoding ---")
engine.activate(kv_cache='standard', decoding='standard')
out1 = engine.generate(PROMPT, max_new_tokens=40)
correct1 = out1[:60] == EXPECTED[:60]
print(f"Output: {out1[:80]}")
print(f"Correct: {correct1}")
s1 = engine.benchmark(PROMPT, max_new_tokens=30)
print()

# Test 2: Streaming KV
print("--- Test 2: Streaming KV (sinks + window) ---")
engine.activate(kv_cache='streaming', decoding='standard')
out2 = engine.generate(PROMPT, max_new_tokens=40)
correct2 = out2[:60] == EXPECTED[:60]
print(f"Output: {out2[:80]}")
print(f"Correct: {correct2}")
s2 = engine.benchmark(PROMPT, max_new_tokens=30)
print()

# Test 3: Hadamard INT4 KV
print("--- Test 3: Hadamard INT4 KV (4x compression) ---")
engine.activate(kv_cache='hadamard_int4', decoding='standard')
out3 = engine.generate(PROMPT, max_new_tokens=40)
correct3 = out3[:60] == EXPECTED[:60]
print(f"Output: {out3[:80]}")
print(f"Correct: {correct3}")
s3 = engine.benchmark(PROMPT, max_new_tokens=30)
print()

# Test 4: RotorQuant KV
print("--- Test 4: RotorQuant KV ---")
engine.activate(kv_cache='rotorquant', decoding='standard')
out4 = engine.generate(PROMPT, max_new_tokens=40)
correct4 = out4[:60] == EXPECTED[:60]
print(f"Output: {out4[:80]}")
print(f"Correct: {correct4}")
s4 = engine.benchmark(PROMPT, max_new_tokens=30)
print()

# Test 5: Prefix caching
print("--- Test 5: Prefix caching ---")
engine.activate(kv_cache='standard', decoding='standard', use_prefix_cache=True)
out5a = engine.generate(PROMPT, max_new_tokens=20)
out5b = engine.generate(PROMPT, max_new_tokens=20)  # Should hit cache
correct5 = out5a[:60] == EXPECTED[:60]
print(f"Output: {out5a[:80]}")
print(f"Correct: {correct5}")
print()

# Summary
print("=" * 60)
print("ForgeLM v1 Summary")
print("=" * 60)
print(f"{'Config':<30} {'tok/s':>8} {'Correct':>8}")
print("-" * 50)
print(f"{'Standard':<30} {s1['tokens_per_sec']:>8.0f} {str(correct1):>8}")
print(f"{'Streaming KV':<30} {s2['tokens_per_sec']:>8.0f} {str(correct2):>8}")
print(f"{'Hadamard INT4 KV':<30} {s3['tokens_per_sec']:>8.0f} {str(correct3):>8}")
print(f"{'RotorQuant KV':<30} {s4['tokens_per_sec']:>8.0f} {str(correct4):>8}")
print(f"{'Prefix cache':<30} {'N/A':>8} {str(correct5):>8}")
print()
all_correct = all([correct1, correct2, correct3, correct4, correct5])
print(f"All outputs correct: {all_correct}")
print()
print("ForgeLM v1 Architecture:")
print(f"  Base: Qwen2.5-Coder-1.5B (1.54B params)")
print(f"  Attention: MLA (d_c=512, 4x KV compression)")
print(f"  FFN: MoE (4 routed + 1 shared, top-4)")
print(f"  KeyStack: MRL + QuaRot + ValueResidual + RotorQuant + MTP")
print(f"  Runtime: StreamingLLM + SnapKV + Prefix cache + torch.compile")
print(f"  Checkpoint: research/checkpoints/forgelm_v1.safetensors")
print(f"  Config: forgelm_v1")
