"""Debug checkpoint loading — check where weights end up."""
import torch
from research.config import get_config
from safetensors import safe_open

cfg = get_config('forgelm_v7_8b_b', device='cuda', mtp_n_heads=2)
ckpt = 'research/checkpoints/ForgeLM_V7_8B_final.safetensors'

# Load state dict directly to CUDA
state = {}
with safe_open(ckpt, framework='pt', device='cuda') as f:
    for key in f.keys():
        state[key] = f.get_tensor(key)

embed_key = 'embed.embed.weight'
if embed_key in state:
    print(f'embed device: {state[embed_key].device}, absmax: {state[embed_key].abs().max().item():.6f}')
else:
    print(f'embed key not found! Available embed keys: {[k for k in state if "embed" in k]}')

# Apply GTA key transform
from research.keys.attention.gta_key import GTAKey
res = GTAKey(n_layers=cfg.n_layers, n_heads=cfg.n_heads).forward(state)
if res.success:
    state = res.weights
    print(f'After GTA - embed device: {state[embed_key].device}, absmax: {state[embed_key].abs().max().item():.6f}')
    vg = state.get('blocks.2.attn.v_mix_gate')
    if vg is not None:
        print(f'After GTA - v_mix_gate device: {vg.device}, value: {vg.item()}')
else:
    print(f'GTA key failed: {res.error}')

# Now build model with meta init and assign
from research.model_loader import ConfigurableResearchLLM, ModelLoader
cfg_meta = get_config('forgelm_v7_8b_b', device='meta', mtp_n_heads=2)
with torch.device('meta'):
    model = ConfigurableResearchLLM(cfg_meta)

print(f'\nBefore load - embed device: {model.embed.embed.weight.device}')
missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
print(f'After assign - embed device: {model.embed.embed.weight.device}')
print(f'After assign - embed absmax: {model.embed.embed.weight.abs().max().item():.6f}')
print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')
