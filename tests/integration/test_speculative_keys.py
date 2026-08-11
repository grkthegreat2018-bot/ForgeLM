"""Test L1 Speculative Attention, L6 Speculative FFN, L7 Redundant Layer Skip.
All should be lossless (cos=1.0)."""
import sys, os, torch, torch.nn as nn
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def get_input(model):
    tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf", fix_mistral_regex=True)
    text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
    enc = tok(text, return_tensors="pt")
    return enc.input_ids.to(next(model.parameters()).device)

def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a[0].flatten().unsqueeze(0).float(),
        b[0].flatten().unsqueeze(0).float(), dim=-1).item()

def fresh_model():
    from research.config import get_config
    from research.model_loader import ModelLoader
    cfg = get_config('forgelm_v2', device='cuda')
    m = ModelLoader.build_model_fast(
        cfg, checkpoint_path='research/checkpoints/forgelm_v2.safetensors',
        moe_top_k=0, dtype=torch.bfloat16)
    m.to('cuda').eval()
    return m

# Baseline
print("Loading baseline...")
model = fresh_model()
x = get_input(model)
with torch.no_grad():
    baseline, _ = model(x, use_cache=False)
print(f"Baseline shape: {baseline.shape}")
del model
torch.cuda.empty_cache()

# ── L7: Redundant Layer Skip ──
print("\n" + "=" * 60)
print("L7: Redundant Layer Skip (lossless)")
print("=" * 60)
from research.keys.speculative.speculative_keys import RedundantLayerSkipKey

model = fresh_model()
x = get_input(model)
key = RedundantLayerSkipKey(threshold=0.999)
key.calibrate(model, x)
with torch.no_grad():
    out, _ = model(x, use_cache=False)
cos = cos_sim(baseline, out)
print(f"  cos={cos:.6f} (expect 1.0 — only skips near-identity layers)")
del model, key
torch.cuda.empty_cache()

# Try with lower threshold
print("\n--- threshold=0.99 ---")
model = fresh_model()
x = get_input(model)
key = RedundantLayerSkipKey(threshold=0.99)
key.calibrate(model, x)
with torch.no_grad():
    out, _ = model(x, use_cache=False)
cos = cos_sim(baseline, out)
print(f"  cos={cos:.6f}")
del model, key
torch.cuda.empty_cache()

# ── L6: Speculative FFN ──
print("\n" + "=" * 60)
print("L6: Speculative FFN (lossless)")
print("=" * 60)
from research.keys.speculative.speculative_keys import SpeculativeFFNKey

model = fresh_model()
x = get_input(model)
key = SpeculativeFFNKey(tolerance=0.01)
key.apply(model)
with torch.no_grad():
    out, _ = model(x, use_cache=False)
cos = cos_sim(baseline, out)
key.print_stats()
print(f"  cos={cos:.6f} (expect 1.0 — full output always computed)")
del model, key
torch.cuda.empty_cache()

# ── L1: Speculative Attention ──
print("\n" + "=" * 60)
print("L1: Speculative Attention (lossless)")
print("=" * 60)
from research.keys.speculative.speculative_keys import SpeculativeAttentionKey

model = fresh_model()
x = get_input(model)
key = SpeculativeAttentionKey(draft_rank=32)
key.apply(model)
with torch.no_grad():
    out, _ = model(x, use_cache=False)
cos = cos_sim(baseline, out)
key.print_stats()
print(f"  cos={cos:.6f} (expect 1.0 — full attention always computed)")
del model, key
torch.cuda.empty_cache()

# ── All 3 stacked ──
print("\n" + "=" * 60)
print("All 3 lossless keys stacked")
print("=" * 60)
model = fresh_model()
x = get_input(model)

# L7 first (calibrate + skip)
l7 = RedundantLayerSkipKey(threshold=0.999)
l7.calibrate(model, x)

# L6 (speculative FFN)
l6 = SpeculativeFFNKey(tolerance=0.01)
l6.apply(model)

# L1 (speculative attention)
l1 = SpeculativeAttentionKey(draft_rank=32)
l1.apply(model)

with torch.no_grad():
    out, _ = model(x, use_cache=False)
cos = cos_sim(baseline, out)
l6.print_stats()
l1.print_stats()
print(f"  Stacked cos={cos:.6f} (expect 1.0)")
del model
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
