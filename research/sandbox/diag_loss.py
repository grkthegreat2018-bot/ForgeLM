"""Diagnose why training loss is ~30 (above random ln(65536)≈11).

Checks:
1. Model forward pass produces sensible logits
2. Head/embed weights are properly loaded (not zero/random)
3. v_mix_gate impact on loss
4. Plain CE loss (no entropy weighting) on a real example
5. Compare with/without entropy_alpha
"""
import sys, os
sys.path.insert(0, "D:\\windsurf\\ForgeAI")
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

import torch
torch.set_grad_enabled(False)

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer

print("=== Loading model ===")
cfg = get_config("forgelm_v7_8b_b", device="cuda")
cfg.mtp_n_heads = 2  # match checkpoint
cfg.use_mtp = False  # disable MTP to isolate CE loss
cfg.use_chunked_ce = True
cfg.ce_chunk_size = 128
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path="research/checkpoints/ForgeLM_V7_8B_final.safetensors",
    dtype=torch.bfloat16,
)
tok = get_tokenizer()
model.eval()

# Check head and embed weights
print("\n=== Weight diagnostics ===")
head_w = model.head.weight
embed_w = model.embed.weight
print(f"head.weight: shape={head_w.shape}, dtype={head_w.dtype}, "
      f"norm={head_w.norm().item():.4f}, mean={head_w.mean().item():.6f}, "
      f"std={head_w.std().item():.6f}")
print(f"embed.weight: shape={embed_w.shape}, dtype={embed_w.dtype}, "
      f"norm={embed_w.norm().item():.4f}, mean={embed_w.mean().item():.6f}, "
      f"std={embed_w.std().item():.6f}")
print(f"head == embed (tied): {head_w.data_ptr() == embed_w.data_ptr()}")

# Check a few block weights (only attention blocks have q_proj)
layer_types = getattr(cfg, 'layer_types', None)
print(f"Layer types: {layer_types[:6] if layer_types else 'all attention'}...")
for i in [2, 15, 31]:  # block 2 is first attention block
    blk = model.blocks[i]
    attn = blk.attn
    attn_name = type(attn).__name__
    if hasattr(attn, 'q_proj'):
        q = attn.q_proj
        print(f"block {i} ({attn_name}) q_proj: norm={q.weight.norm().item():.4f} "
              f"(shape={q.weight.shape})")
    else:
        print(f"block {i} ({attn_name}): no q_proj (type={attn_name})")
    if hasattr(attn, 'v_mix_gate'):
        gate = attn.v_mix_gate
        print(f"  v_mix_gate: val={gate.item():.6f}, device={gate.device}, "
              f"requires_grad={gate.requires_grad}")
    else:
        print(f"  v_mix_gate: NOT FOUND")

# Simple forward pass with a known token sequence
print("\n=== Forward pass test ===")
# Use a simple English sentence tokenized
text = "The quick brown fox jumps over the lazy dog."
ids = tok.encode(text)
print(f"Tokenized: {len(ids)} tokens: {ids[:20]}...")
input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)

# Forward without targets to get logits
with torch.no_grad():
    out = model(input_ids, targets=None)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

print(f"Logits: shape={logits.shape}, dtype={logits.dtype}")
print(f"  min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
      f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
print(f"  any NaN: {torch.isnan(logits).any().item()}, any Inf: {torch.isinf(logits).any().item()}")

# Check next-token prediction quality
probs = torch.softmax(logits[0, -1], dim=-1)
top5 = torch.topk(probs, 5)
print(f"Top-5 next tokens after 'dog.':")
for idx, prob in zip(top5.indices, top5.values):
    t = tok.decode([idx.item()])
    print(f"  token {idx.item()}: '{t}' (p={prob.item():.4f})")

# Now compute actual CE loss on this sequence
print("\n=== CE loss test ===")
shift_labels = torch.full_like(input_ids, -100)
shift_labels[:, :-1] = input_ids[:, 1:]  # predict next token

# Plain CE (entropy_alpha=0)
model.config.entropy_alpha = 0.0
with torch.no_grad():
    out = model(input_ids, targets=shift_labels)
    loss_plain = out[1] if isinstance(out, tuple) else out
print(f"Plain CE loss (entropy_alpha=0): {loss_plain.item():.4f}")

# Entropy-weighted CE (entropy_alpha=0.5)
model.config.entropy_alpha = 0.5
with torch.no_grad():
    out = model(input_ids, targets=shift_labels)
    loss_ent = out[1] if isinstance(out, tuple) else out
print(f"Entropy-weighted CE (alpha=0.5): {loss_ent.item():.4f}")

# Entropy-weighted CE (entropy_alpha=1.0)
model.config.entropy_alpha = 1.0
with torch.no_grad():
    out = model(input_ids, targets=shift_labels)
    loss_ent1 = out[1] if isinstance(out, tuple) else out
print(f"Entropy-weighted CE (alpha=1.0): {loss_ent1.item():.4f}")

# Check what entropy the model produces
model.config.entropy_alpha = 0.0
with torch.no_grad():
    out = model(input_ids, targets=None)
    logits = out[0] if isinstance(out, tuple) else out
    # Per-token entropy
    log_probs = torch.log_softmax(logits[0], dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    print(f"Per-token entropy: mean={entropy.mean().item():.4f}, "
          f"max={entropy.max().item():.4f}, min={entropy.min().item():.4f}")
    print(f"  (ln(65536)={torch.log(torch.tensor(65536.0)).item():.4f} = max possible)")

print("\n=== Done ===")
