"""Trace model forward pass to find where garbage output starts."""
import sys, os
sys.path.insert(0, "D:\\windsurf\\ForgeAI")
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

import torch
torch.set_grad_enabled(False)

from research.config import get_config
from research.model_loader import ModelLoader, ConfigurableResearchLLM
from research.tokenizer_cache import get_tokenizer

print("=== Loading model ===")
cfg = get_config("forgelm_v7_8b_b", device="cuda")
cfg.mtp_n_heads = 2
cfg.use_mtp = False
cfg.use_chunked_ce = True
cfg.ce_chunk_size = 128
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path="research/checkpoints/ForgeLM_V7_8B_final.safetensors",
    dtype=torch.bfloat16,
)
tok = get_tokenizer()
model.eval()

# Check missing/unexpected keys explicitly
print("\n=== Checking state dict loading ===")
# Rebuild to get missing/unexpected
import safetensors.torch as st
state = st.load_file("research/checkpoints/ForgeLM_V7_8B_final.safetensors")
model_keys = set(dict(model.named_parameters()).keys())
model_keys.update(set(dict(model.named_buffers()).keys()))
ckpt_keys = set(state.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys
print(f"Missing keys (in model, not in checkpoint): {len(missing)}")
if missing:
    for k in sorted(missing)[:30]:
        print(f"  MISSING: {k}")
print(f"Unexpected keys (in checkpoint, not in model): {len(unexpected)}")
if unexpected:
    for k in sorted(unexpected)[:30]:
        print(f"  UNEXPECTED: {k}")

# Trace forward pass
print("\n=== Forward pass trace ===")
text = "The quick brown fox jumps over the lazy dog."
ids = tok.encode(text)
input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)

# 1. Check embedding output
with torch.no_grad():
    embed_out = model.embed(input_ids)
    print(f"Embedding output: shape={embed_out.shape}, norm={embed_out.norm().item():.4f}, "
          f"mean={embed_out.mean().item():.6f}, std={embed_out.std().item():.6f}")
    print(f"  any NaN: {torch.isnan(embed_out).any().item()}, any Inf: {torch.isinf(embed_out).any().item()}")

# 2. Check output after each block
with torch.no_grad():
    x = embed_out
    for i, block in enumerate(model.blocks):
        layer_type = getattr(block, 'layer_type', 'attention')
        out = block(x)
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out
        if i < 5 or i >= 30 or i == 10 or i == 20:
            print(f"Block {i} ({layer_type}): norm={x.norm().item():.4f}, "
                  f"mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
                  f"any_nan={torch.isnan(x).any().item()}")

# 3. Check final norm + logits
with torch.no_grad():
    x = model.ln_f(x)
    print(f"ln_f: norm={x.norm().item():.4f}, mean={x.mean().item():.6f}, std={x.std().item():.6f}")
    logits = model.head(x)
    print(f"Logits: shape={logits.shape}, norm={logits.norm().item():.4f}, "
          f"min={logits.min().item():.4f}, max={logits.max().item():.4f}")

# 4. Check NLRQ FFN on block 2 (first attention block)
print("\n=== NLRQ FFN check (block 2) ===")
blk2 = model.blocks[2]
ffn = blk2.ffn
print(f"FFN type: {type(ffn).__name__}")
for name, param in ffn.named_parameters():
    print(f"  {name}: shape={param.shape}, norm={param.norm().item():.4f}, "
          f"device={param.device}, is_meta={param.is_meta}")

# Test FFN forward
with torch.no_grad():
    test_x = torch.randn(1, 4, 4096, device="cuda", dtype=torch.bfloat16)
    ffn_out = ffn(test_x)
    print(f"FFN output: shape={ffn_out.shape}, norm={ffn_out.norm().item():.4f}, "
          f"mean={ffn_out.mean().item():.6f}, std={ffn_out.std().item():.6f}")

# 5. Check attention on block 2
print("\n=== Attention check (block 2) ===")
attn = blk2.attn
print(f"Attention type: {type(attn).__name__}")
for name, param in attn.named_parameters():
    if param.is_meta:
        print(f"  {name}: META DEVICE!")
    else:
        print(f"  {name}: shape={param.shape}, norm={param.norm().item():.4f}, device={param.device}")

print("\n=== Done ===")
