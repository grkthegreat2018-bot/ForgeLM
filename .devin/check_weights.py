import sys; sys.path.insert(0, 'D:/windsurf/ForgeAI')
from safetensors.torch import load_file
import torch

base = load_file('research/checkpoints/ForgeLM_V3_Base.safetensors')
final = load_file('research/checkpoints/ForgeLM_V3_SFT.safetensors')

# Check head weight diff
diff = (base['head.weight'].float() - final['head.weight'].float()).abs().mean().item()
print(f'head.weight diff: {diff:.6f}')
print(f'head.weight base absmean: {base["head.weight"].abs().mean().item():.6f}')
print(f'head.weight final absmean: {final["head.weight"].abs().mean().item():.6f}')

# Check if head has qscale in final but not base
print(f'head.qscale in base: {"head.qscale" in base}')
print(f'head.qscale in final: {"head.qscale" in final}')
print(f'head.qscale final: {final["head.qscale"].item():.6f}')

# The head is NOT BitNet in the base (it's a regular Linear)
# But convert_to_bitnet_everywhere converts it to BitNetLinear
# Check if head weight is tied to embed
embed_diff = (base['embed.weight'].float() - base['head.weight'].float()).abs().mean().item()
print(f'embed vs head (base): {embed_diff:.8f}')
embed_diff_final = (final['embed.weight'].float() - final['head.weight'].float()).abs().mean().item()
print(f'embed vs head (final): {embed_diff_final:.8f}')

# Check weight tying — the model uses tied embeddings
# If head was converted to BitNetLinear with learned_scale, the qscale
# re-anchors on load. But if the head weight was modified by LoRA merge,
# the qscale (from absmean) would be different from what was trained.
print(f'\nhead.weight absmean base: {base["head.weight"].abs().mean().item():.8f}')
print(f'head.weight absmean final: {final["head.weight"].abs().mean().item():.8f}')
print(f'Expected qscale base: {base["head.weight"].abs().mean().item()/0.7:.8f}')
print(f'Expected qscale final: {final["head.weight"].abs().mean().item()/0.7:.8f}')
print(f'Actual qscale final: {final["head.qscale"].item():.8f}')
