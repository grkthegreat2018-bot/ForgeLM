"""Verify ForgeLM v2 loads and produces output identical to v1.

Since all new keys are identity-init (lossless), the model output
should be identical to v1 (within floating point tolerance).
"""
import sys, torch
sys.path.insert(0, '.')

from research.config import get_config
from research.model_loader import ModelLoader

print("=" * 60)
print("ForgeLM v2 Verification")
print("=" * 60)

# Load v2 model
print("\n[1] Loading ForgeLM v2...")
cfg = get_config("forgelm_v2", device="cuda")
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
model.to("cuda")
model.eval()

# Check QK-Norm weights
print("\n[2] Checking QK-Norm weights...")
for i in range(min(3, len(model.blocks))):
    attn = model.blocks[i].attn
    if hasattr(attn, 'q_norm'):
        q_w = attn.q_norm.weight
        k_w = attn.k_norm.weight
        print(f"  Layer {i}: q_norm={q_w.shape} (mean={q_w.mean():.4f}), "
              f"k_norm={k_w.shape} (mean={k_w.mean():.4f})")
        assert (q_w == 1.0).all(), f"q_norm layer {i} not identity!"
        assert (k_w == 1.0).all(), f"k_norm layer {i} not identity!"
print("  QK-Norm: identity init verified ✓")

# Run a test forward pass
print("\n[3] Running test forward pass...")
test_input = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
with torch.inference_mode():
    logits, _ = model(test_input)
print(f"  Input shape: {test_input.shape}")
print(f"  Logits shape: {logits.shape}")
print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

# Generate a few tokens
print("\n[4] Generating 20 tokens...")
with torch.inference_mode():
    for _ in range(20):
        logits, _ = model(test_input)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        test_input = torch.cat([test_input, next_token], dim=1)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")
generated = tok.decode(test_input[0].tolist())
print(f"  Generated: {generated[:200]}")

# Compare with v1 if available
print("\n[5] Comparing v2 vs v1 output...")
v1_path = "research/checkpoints/forgelm_v1.safetensors"
import os
if os.path.exists(v1_path):
    cfg1 = get_config("forgelm_v1", device="cuda")
    model1 = ModelLoader.build_model_fast(cfg1, checkpoint_path=v1_path)
    model1.to("cuda")
    model1.eval()

    test = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.inference_mode():
        l1, _ = model1(test)
        l2, _ = model(test)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        l1.flatten().unsqueeze(0).float(),
        l2.flatten().unsqueeze(0).float()
    ).item()
    max_diff = (l1.float() - l2.float()).abs().max().item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max abs diff: {max_diff:.8f}")
    if cos_sim > 0.9999:
        print("  ✓ v2 output matches v1 (lossless transform verified)")
    elif cos_sim > 0.999:
        print("  ~ v2 output very close to v1 (minor numerical differences)")
    else:
        print("  ✗ v2 output differs from v1!")
else:
    print("  v1 checkpoint not found, skipping comparison")

print("\n" + "=" * 60)
print("Verification complete")
print("=" * 60)
