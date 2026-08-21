"""Check V4 checkpoint keys for V5 port planning."""
import sys, os
sys.path.insert(0, 'D:/windsurf/ForgeAI')
from safetensors.torch import load_file

ckpt = 'D:/windsurf/ForgeAI/research/checkpoints/ForgeLM_V4_Base.safetensors'
state = load_file(ckpt)

# Group keys by prefix
prefixes = {}
for k in sorted(state.keys()):
    parts = k.split('.')
    if parts[0] == 'blocks':
        prefix = '.'.join(parts[:3])  # blocks.N.xxx
        cat = parts[2] if len(parts) > 2 else '?'
    else:
        prefix = parts[0]
        cat = parts[0]
    prefixes.setdefault(cat, []).append(k)

# Show block 0 keys in detail
print("=== Block 0 keys ===")
for k in sorted(state.keys()):
    if k.startswith('blocks.0.'):
        print(f"  {k}: {state[k].shape}")

print("\n=== Non-block keys ===")
for k in sorted(state.keys()):
    if not k.startswith('blocks.'):
        print(f"  {k}: {state[k].shape}")

# Check which blocks are conv vs attention
print("\n=== Layer types (from keys) ===")
for i in range(16):
    has_attn = any(k.startswith(f'blocks.{i}.attn.') for k in state)
    has_conv = any(k.startswith(f'blocks.{i}.attn.conv') for k in state)
    has_qk = any('qk_norm' in k for k in state if k.startswith(f'blocks.{i}.'))
    ffn_keys = [k for k in state if k.startswith(f'blocks.{i}.ffn.')]
    layer_type = "conv" if has_conv else ("attn" if has_attn else "?")
    print(f"  Block {i}: {layer_type}, FFN keys: {len(ffn_keys)}")
