"""Test ForgeEngine unified backend on XP model."""
import sys, torch
sys.path.insert(0, '.')
from research.inference.forge_engine import ForgeEngine

# Build engine from XP checkpoint
print("=== Building ForgeEngine ===")
engine = ForgeEngine.from_checkpoint(
    checkpoint="research/checkpoints/xp_full_no_mqa.safetensors",
    config_name="qwen25_coder_1.5b",
    tokenizer_path="research/checkpoints/qwen_hf",
    device="cuda",
)

# Test 1: Baseline (standard KV, standard decoding)
print("\n=== Test 1: Baseline (standard) ===")
engine.activate(kv_cache="standard", decoding="standard")
out = engine.generate("def fibonacci(n):", max_new_tokens=40)
print(f"  Output: {out[:120]}")
print(f"  Stats: {engine.stats()}")

# Test 2: Hadamard INT4 KV cache
print("\n=== Test 2: Hadamard INT4 KV cache ===")
engine.activate(kv_cache="hadamard_int4", decoding="standard")
out2 = engine.generate("def fibonacci(n):", max_new_tokens=40)
print(f"  Output: {out2[:120]}")
match = out == out2
print(f"  Match with baseline: {match}")

# Test 3: RotorQuant KV cache
print("\n=== Test 3: RotorQuant KV cache ===")
engine.activate(kv_cache="rotorquant", decoding="standard")
out3 = engine.generate("def fibonacci(n):", max_new_tokens=40)
print(f"  Output: {out3[:120]}")

# Test 4: MTP self-speculative decoding
print("\n=== Test 4: MTP self-speculative decoding ===")
engine.activate(kv_cache="standard", decoding="mtp_selfspec")
out4 = engine.generate("def fibonacci(n):", max_new_tokens=40)
print(f"  Output: {out4[:120]}")

# Test 5: Full stack (Hadamard KV + MTP self-spec)
print("\n=== Test 5: Full stack (Hadamard KV + MTP self-spec) ===")
engine.activate(kv_cache="hadamard_int4", decoding="mtp_selfspec",
                use_v0_warm=True)
out5 = engine.generate("The meaning of life is", max_new_tokens=40)
print(f"  Output: {out5[:120]}")
print(f"  Full stats: {engine.stats()}")

# Test 6: Benchmark
print("\n=== Test 6: Benchmark ===")
engine.activate(kv_cache="standard", decoding="standard")
engine.benchmark("def fibonacci(n):", max_new_tokens=30)

print("\n=== All tests passed ===")
