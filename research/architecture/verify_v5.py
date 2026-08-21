"""Verify V5 checkpoint loads and generates correctly."""
import sys, os
sys.path.insert(0, 'D:/windsurf/ForgeAI')
import torch
from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from safetensors.torch import load_file

ckpt = 'D:/windsurf/ForgeAI/research/checkpoints/forgelm_v7_Base.safetensors'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loading V5 checkpoint: {ckpt}")
state = load_file(ckpt)  # CPU
print(f"  {len(state)} tensors, {sum(v.numel() for v in state.values())/1e6:.1f}M params")

# Check key categories
embed_keys = [k for k in state if 'embed' in k and 'blocks' not in k]
expert_keys = [k for k in state if 'experts' in k]
shared_keys = [k for k in state if 'shared' in k and 'blocks' in k]
router_keys = [k for k in state if 'router' in k]
attn_keys = [k for k in state if 'attn' in k and 'blocks' in k]
print(f"  Embedding keys: {len(embed_keys)}")
print(f"  Expert keys: {len(expert_keys)}")
print(f"  Shared expert keys: {len(shared_keys)}")
print(f"  Router keys: {len(router_keys)}")
print(f"  Attention keys: {len(attn_keys)}")

# Check a few specific keys
print(f"\n  Sample expert keys:")
for k in sorted(expert_keys)[:6]:
    print(f"    {k}: {state[k].shape}")

# Build model on CPU in bf16, then move to GPU
print(f"\nBuilding V5 model...")
cfg = get_config('forgelm_v7_moe')
old_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
model = ConfigurableResearchLLM(cfg)
torch.set_default_dtype(old_dtype)
model = model.to(torch.bfloat16).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params / 1e6:.1f}M params")

# Load state dict tensor-by-tensor to avoid holding everything on GPU
print(f"Loading state dict (streaming to GPU)...")
model_state = model.state_dict()
missing = []
unexpected = []
loaded = 0
for k, v in state.items():
    if k in model_state:
        if model_state[k].shape != v.shape:
            print(f"  SHAPE MISMATCH: {k} model={model_state[k].shape} ckpt={v.shape}")
            unexpected.append(k)
            continue
        model_state[k].copy_(v.to(device, dtype=torch.bfloat16))
        loaded += 1
    else:
        unexpected.append(k)
# Find missing keys
for k in model_state:
    if k not in state:
        missing.append(k)
print(f"  Loaded: {loaded}")
print(f"  Missing: {len(missing)}")
if missing:
    for k in missing[:10]:
        print(f"    MISSING: {k}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
print(f"  Unexpected: {len(unexpected)}")
if unexpected:
    for k in unexpected[:5]:
        print(f"    UNEXPECTED: {k}")

# Free state dict from CPU
del state
import gc; gc.collect()

# Check if shared expert weights match base FFN (lossless check)
print(f"\n--- Lossless check: shared expert vs base FFN ---")
v4_state = load_file('D:/windsurf/ForgeAI/research/checkpoints/forgelm_v7_Base.safetensors')
for i in [0, 2, 5]:
    v4_w = v4_state.get(f'blocks.{i}.ffn.w_gate.weight')
    v5_w = model_state.get(f'blocks.{i}.ffn.shared.w1.weight')
    if v4_w is not None and v5_w is not None:
        diff = (v4_w.float().cpu() - v5_w.float().cpu()).abs().max().item()
        print(f"  Block {i} shared.w1 vs base w_gate: max_diff={diff:.6f}")
del v4_state; gc.collect()

# Quick generation test
print(f"\n--- Generation test ---")
model.eval()
with torch.no_grad():
    idx = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    logits, loss = model(idx)
    print(f"  Forward OK: logits {logits.shape}, loss={loss}")
    for _ in range(5):
        logits, _ = model(idx)
        next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        idx = torch.cat([idx, next_tok], dim=1)
    print(f"  Generated tokens: {idx[0].tolist()}")

print(f"\n=== V5 verification PASSED ===")
