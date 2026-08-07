"""Build ForgeLM v2 and compare with V1.

Loads both checkpoints, runs identical prompts, measures:
  - Logit cosine similarity (should be 1.0 — lossless)
  - Generation quality (should be identical)
  - Load time, param count, tensor count
  - Per-layer hidden state divergence
"""
import sys, time, torch, os
sys.path.insert(0, '.')

from research.config import get_config
from research.model_loader import ModelLoader
from transformers import AutoTokenizer

V1_PATH = "research/checkpoints/forgelm_v1.safetensors"
V2_PATH = "research/checkpoints/forgelm_v2.safetensors"

print("=" * 70)
print("ForgeLM V2 vs V1 — Build & Compare")
print("=" * 70)

# Load tokenizer
print("\n[1] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")
print(f"  Vocab: {tok.vocab_size}")

# Load V1
print("\n[2] Loading ForgeLM V1...")
t0 = time.time()
cfg1 = get_config("forgelm_v1", device="cuda")
model_v1 = ModelLoader.build_model_fast(cfg1, checkpoint_path=V1_PATH)
model_v1.to("cuda").eval()
t_v1 = time.time() - t0
n_params_v1 = sum(p.numel() for p in model_v1.parameters())
print(f"  Loaded in {t_v1:.1f}s — {n_params_v1/1e6:.1f}M params")

# Load V2
print("\n[3] Loading ForgeLM V2...")
t0 = time.time()
cfg2 = get_config("forgelm_v2", device="cuda")
model_v2 = ModelLoader.build_model_fast(cfg2, checkpoint_path=V2_PATH)
model_v2.to("cuda").eval()
t_v2 = time.time() - t0
n_params_v2 = sum(p.numel() for p in model_v2.parameters())
print(f"  Loaded in {t_v2:.1f}s — {n_params_v2/1e6:.1f}M params")

print(f"\n  Param difference: +{(n_params_v2 - n_params_v1)/1e3:.1f}K (QK-Norm scales)")

# ─── Comparison Tests ────────────────────────────────────────────────

print("\n[4] Running comparison tests...")

prompts = [
    "def fibonacci(n):",
    "The capital of France is",
    "import numpy as np\nnp.array([1,2,3]).",
    "2 + 2 =",
    "def quicksort(arr):",
]

all_cos_sims = []
all_max_diffs = []

for prompt in prompts:
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.inference_mode():
        logits_v1, hidden_v1 = model_v1(input_ids)
        logits_v2, hidden_v2 = model_v2(input_ids)

    # Logit comparison
    cos_sim = torch.nn.functional.cosine_similarity(
        logits_v1.flatten().float().unsqueeze(0),
        logits_v2.flatten().float().unsqueeze(0)
    ).item()
    max_diff = (logits_v1.float() - logits_v2.float()).abs().max().item()

    all_cos_sims.append(cos_sim)
    all_max_diffs.append(max_diff)

    # Generate from both
    with torch.inference_mode():
        gen_v1 = input_ids.clone()
        gen_v2 = input_ids.clone()
        for _ in range(30):
            lg1, _ = model_v1(gen_v1)
            nt1 = lg1[0, -1].argmax()
            gen_v1 = torch.cat([gen_v1, nt1.unsqueeze(0).unsqueeze(0)], dim=1)

            lg2, _ = model_v2(gen_v2)
            nt2 = lg2[0, -1].argmax()
            gen_v2 = torch.cat([gen_v2, nt2.unsqueeze(0).unsqueeze(0)], dim=1)

    text_v1 = tok.decode(gen_v1[0], skip_special_tokens=True)
    text_v2 = tok.decode(gen_v2[0], skip_special_tokens=True)
    match = "✓" if text_v1 == text_v2 else "≈"

    print(f"\n  Prompt: {prompt[:50]}")
    print(f"  cos={cos_sim:.8f}  max_diff={max_diff:.2e}  gen_match={match}")
    if text_v1 != text_v2:
        print(f"    V1: {text_v1[len(prompt):][:80]}")
        print(f"    V2: {text_v2[len(prompt):][:80]}")

# ─── Per-layer hidden state divergence ───────────────────────────────

print("\n[5] Per-layer hidden state divergence...")
input_ids = tok("def hello_world():", return_tensors="pt").input_ids.to("cuda")

with torch.inference_mode():
    x1 = model_v1.embed(input_ids)
    x2 = model_v2.embed(input_ids)

    for i, (b1, b2) in enumerate(zip(model_v1.blocks, model_v2.blocks)):
        x1, _ = b1(x1)
        x2, _ = b2(x2)
        diff = (x1.float() - x2.float()).abs().max().item()
        if i < 5 or i >= len(model_v1.blocks) - 2 or diff > 1e-5:
            print(f"  Layer {i:2d}: max_diff = {diff:.2e}")

# ─── Summary ─────────────────────────────────────────────────────────

avg_cos = sum(all_cos_sims) / len(all_cos_sims)
avg_diff = sum(all_max_diffs) / len(all_max_diffs)
gen_matches = sum(1 for p in prompts
                  if tok.decode(model_v1(tok(p, return_tensors="pt").input_ids.to("cuda"))[0][0, -1].argmax().unsqueeze(0).unsqueeze(0), skip_special_tokens=True) ==
                  tok.decode(model_v2(tok(p, return_tensors="pt").input_ids.to("cuda"))[0][0, -1].argmax().unsqueeze(0).unsqueeze(0), skip_special_tokens=True))

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  V1: {n_params_v1/1e6:.1f}M params, {t_v1:.1f}s load")
print(f"  V2: {n_params_v2/1e6:.1f}M params, {t_v2:.1f}s load")
print(f"  Avg logit cosine similarity: {avg_cos:.10f}")
print(f"  Avg max abs difference:      {avg_diff:.2e}")
if avg_cos > 0.99999:
    print(f"  ✓ V2 is a LOSSLESS transform of V1 (identical output)")
elif avg_cos > 0.999:
    print(f"  ~ V2 is near-lossless (minor numerical differences)")
else:
    print(f"  ✗ V2 differs from V1!")
print(f"{'='*70}")
