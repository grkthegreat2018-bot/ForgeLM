"""Patch embed.project.weight shape in ported checkpoint."""
from safetensors.torch import load_file, save_file
import torch

path = 'research/checkpoints/ForgeLM_V7_8B_B_ported.safetensors'
s = load_file(path)
w = s['embed.project.weight']
print(f'Before: {w.shape}')
s['embed.project.weight'] = w.T.contiguous()
key = 'embed.project.weight'
print(f'After: {s[key].shape}')
save_file(s, path)
print('Saved')
