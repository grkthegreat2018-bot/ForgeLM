"""Test base v2 model quality + hot expert loading.

Phase 1: Load v2 base model, generate text, verify Qwen-level quality
Phase 2: Hot-load an expert from v4 library, generate again, compare
Phase 3: Hot-swap to a different topic expert, verify swap works
"""
import sys, os, time, torch
sys.path.insert(0, '.')

from research.config import get_config
from research.model_loader import ModelLoader
from transformers import AutoTokenizer
from research.airmoe_infinite import InfiniteAirMoE

print("=" * 70)
print("Base v2 Quality Test + Hot Expert Loading")
print("=" * 70)

# ── Phase 1: Load v2 and test quality ──────────────────────────────
print("\n[1] Loading ForgeLM v2 base model...")
t0 = time.time()
cfg = get_config("forgelm_v2", device="cuda")
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
model.to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")
print(f"  Loaded in {time.time()-t0:.1f}s")

# VRAM check
if torch.cuda.is_available():
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM allocated: {vram_gb:.2f} GB")

# Generate text from prompts
def generate(prompt, max_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits, _ = model(input_ids)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

test_prompts = [
    '"""Check if a number is prime"""\ndef is_prime(n):\n    ',
    '"""Compute fibonacci"""\ndef fib(n):\n    ',
    '"""Reverse a string"""\ndef reverse(s):\n    ',
    "The capital of France is",
    "def sort_list(lst):\n    return",
]

print("\n[2] Generating text (base v1, no expert hotswap)...")
base_outputs = {}
for prompt in test_prompts:
    t0 = time.time()
    out = generate(prompt, max_tokens=40)
    gen_time = time.time() - t0
    base_outputs[prompt] = out
    print(f"\n  Prompt: {prompt[:50]}")
    print(f"  Output: {out[len(prompt):].strip()[:100]}")
    print(f"  Time: {gen_time:.1f}s")

# ── Phase 2: Hot-load expert from v4 library ───────────────────────
print(f"\n{'='*70}")
print("[3] Hot-loading expert from v4 library...")
print(f"{'='*70}")

v4_dir = "D:/windsurf/ForgeAI/research/checkpoints/forgelm_v4"
manifest = os.path.join(v4_dir, "manifest.json")

if not os.path.exists(manifest):
    print(f"  ERROR: v4 manifest not found at {manifest}")
    print("  Run: python -m research.bake_v4")
    sys.exit(1)

airmoe = InfiniteAirMoE(model, tokenizer, v4_dir,
                         device="cuda",
                         vram_budget_gb=2.0,
                         max_cached_topics=5)

print(f"  Available topics: {airmoe.router.list_topics()}")

# Route a code query → should pick python_general or python_algorithms
test_query = "Write a Python function to sort a list"
topic = airmoe.route_and_load(test_query)
print(f"\n  Query: {test_query}")
print(f"  Routed to topic: {topic}")

# Generate with expert loaded
print("\n[4] Generating with hot expert loaded...")
expert_outputs = {}
for prompt in test_prompts:
    t0 = time.time()
    out = generate(prompt, max_tokens=40)
    gen_time = time.time() - t0
    expert_outputs[prompt] = out
    print(f"\n  Prompt: {prompt[:50]}")
    print(f"  Output: {out[len(prompt):].strip()[:100]}")
    print(f"  Time: {gen_time:.1f}s")

# ── Phase 3: Hot-swap to different topic ───────────────────────────
print(f"\n{'='*70}")
print("[5] Hot-swapping to math topic...")
print(f"{'='*70}")

math_query = "Solve the equation x^2 + 5x + 6 = 0"
topic2 = airmoe.route_and_load(math_query)
print(f"\n  Query: {math_query}")
print(f"  Routed to topic: {topic2}")

print("\n[6] Generating with math expert...")
math_prompt = "Solve x^2 + 5x + 6 = 0\n"
t0 = time.time()
out = generate(math_prompt, max_tokens=40)
print(f"  Output: {out[len(math_prompt):].strip()[:100]}")
print(f"  Time: {time.time()-t0:.1f}s")

# ── Comparison ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("[7] Comparison: base vs expert-loaded")
print(f"{'='*70}")

for prompt in test_prompts:
    base_out = base_outputs[prompt][len(prompt):].strip()[:80]
    exp_out = expert_outputs[prompt][len(prompt):].strip()[:80]
    changed = "CHANGED" if base_out != exp_out else "same"
    print(f"\n  Prompt: {prompt[:40]}")
    print(f"  Base:   {base_out}")
    print(f"  Expert: {exp_out}")
    print(f"  Status: {changed}")

# Stats
airmoe.print_stats()

# Final VRAM
if torch.cuda.is_available():
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"\n  Final VRAM: {vram_gb:.2f} GB")

print(f"\n{'='*70}")
print("Test Complete")
print(f"{'='*70}")
