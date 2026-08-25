"""Check auxiliary module parameters for misconfiguration."""
import sys, os
sys.path.insert(0, "D:\\windsurf\\ForgeAI")
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

import torch
torch.set_grad_enabled(False)

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer

cfg = get_config("forgelm_v7_8b_b", device="cuda")
cfg.mtp_n_heads = 2
cfg.use_mtp = False
cfg.use_chunked_ce = True
cfg.ce_chunk_size = 128
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path="research/checkpoints/ForgeLM_V7_8B_final.safetensors",
    dtype=torch.bfloat16,
)
model.eval()

print("=== LiSA gates ===")
if hasattr(model, 'lisa') and model.lisa is not None:
    for name, param in model.lisa.named_parameters():
        if 'gate' in name:
            print(f"  lisa.{name}: val={param.data.flatten()[:5].tolist()}, "
                  f"norm={param.norm().item():.6f}")
        elif 'shared' in name:
            print(f"  lisa.{name}: shape={param.shape}, norm={param.norm().item():.4f}")

print("\n=== Hyperloop gates ===")
if hasattr(model, 'loop_gates'):
    for name, buf in model.named_buffers():
        if 'loop_gate' in name or 'middle_gate' in name:
            print(f"  {name}: val={buf.flatten()[:5].tolist()}")
    # Check parameters
    for name, param in model.named_parameters():
        if 'loop_gate' in name or 'middle_gate' in name:
            print(f"  {name}: val={param.data.flatten()[:5].tolist()}")

# Check loop_gates and middle_gates as attributes
for attr in ['loop_gates', 'middle_gates']:
    val = getattr(model, attr, None)
    if val is not None:
        if isinstance(val, torch.Tensor):
            print(f"  {attr}: shape={val.shape}, val={val.flatten()[:5].tolist()}")
        elif isinstance(val, (list, tuple)):
            print(f"  {attr}: len={len(val)}, val={[v.item() if hasattr(v, 'item') else v for v in val[:5]]}")
        elif hasattr(val, 'weight'):
            print(f"  {attr}: weight shape={val.weight.shape}, norm={val.weight.norm().item():.4f}")
        else:
            print(f"  {attr}: type={type(val).__name__}")

print("\n=== Value residual gates (_v0_gates) ===")
if hasattr(model, '_v0_gates'):
    v0g = model._v0_gates
    if isinstance(v0g, torch.Tensor):
        print(f"  _v0_gates: shape={v0g.shape}, val={v0g.flatten()[:5].tolist()}")
    elif isinstance(v0g, nn.Parameter):
        print(f"  _v0_gates: shape={v0g.shape}, val={v0g.data.flatten()[:5].tolist()}")
    else:
        # It might be a ModuleDict or ParameterDict
        for k, v in v0g.items():
            if hasattr(v, 'item'):
                print(f"  _v0_gates.{k}: {v.item():.6f}")
            elif hasattr(v, 'data'):
                print(f"  _v0_gates.{k}: {v.data.item():.6f}")

print("\n=== AttnRes ===")
if hasattr(model, '_attn_res'):
    ar = model._attn_res
    for name, param in ar.named_parameters():
        print(f"  _attn_res.{name}: shape={param.shape}, norm={param.norm().item():.4f}, "
              f"val={param.data.flatten()[:3].tolist()}")
    for name, buf in ar.named_buffers():
        print(f"  _attn_res.{name} (buffer): shape={buf.shape}, val={buf.flatten()[:3].tolist()}")

print("\n=== MoD routers (first 3 blocks) ===")
for i in range(min(3, len(model.blocks))):
    blk = model.blocks[i]
    if hasattr(blk, '_mod') and blk._mod is not None:
        mod = blk._mod
        for name, param in mod.named_parameters():
            print(f"  block {i} _mod.{name}: shape={param.shape}, norm={param.norm().item():.4f}")

print("\n=== TITAN memory (first 3 blocks) ===")
for i in range(min(3, len(model.blocks))):
    blk = model.blocks[i]
    if hasattr(blk, '_memory') and blk._memory is not None:
        mem = blk._memory
        for name, param in mem.named_parameters():
            print(f"  block {i} _memory.{name}: shape={param.shape}, norm={param.norm().item():.6f}")

print("\n=== MHC (first 3 blocks) ===")
for i in range(min(3, len(model.blocks))):
    blk = model.blocks[i]
    if hasattr(blk, '_mhc') and blk._mhc is not None:
        mhc = blk._mhc
        for name, param in mhc.named_parameters():
            print(f"  block {i} _mhc.{name}: shape={param.shape}, norm={param.norm().item():.6f}")

print("\n=== Block 2 detailed forward ===")
blk2 = model.blocks[2]
# Check what the block actually does
print(f"Block 2 type: {type(blk2).__name__}")
print(f"  layer_type: {blk2.layer_type}")
print(f"  has _mod: {hasattr(blk2, '_mod') and blk2._mod is not None}")
print(f"  has _memory: {hasattr(blk2, '_memory') and blk2._memory is not None}")
print(f"  has _mhc: {hasattr(blk2, '_mhc') and blk2._mhc is not None}")
print(f"  has _attn_res: {hasattr(blk2, '_attn_res') and blk2._attn_res is not None}")

# Check ln1 and ln2 weights
print(f"  ln1.weight: norm={blk2.ln1.weight.norm().item():.4f}")
print(f"  ln2.weight: norm={blk2.ln2.weight.norm().item():.4f}")
print(f"  post_attn_norm.weight: norm={blk2.post_attn_norm.weight.norm().item():.4f}")
print(f"  post_ffn_norm.weight: norm={blk2.post_ffn_norm.weight.norm().item():.4f}")

# Check if block forward skips anything
import inspect
fwd_src = inspect.getsource(blk2.forward)
print(f"\n  forward source (first 500 chars):")
print(fwd_src[:500])

print("\n=== Done ===")
