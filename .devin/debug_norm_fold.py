"""Debug norm folding — check if embed/head sharing causes issues."""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from safetensors.torch import load_file
from research.keys.tensor_dedup_key import TensorDedupKey
from research.keys.norm_folding_v2_key import NormFoldingV2Key

state = load_file('research/checkpoints/forgelm_v2.safetensors')

# Step 1: dedup first
td = TensorDedupKey()
r = td.forward(state)
state = r.weights
dedup_map = r.metadata['dedup_map']
print(f'Dedup: {len(dedup_map)} aliases')

# Check head.weight status
head_canonical = dedup_map.get('head.weight', 'head.weight')
print(f'head.weight canonical: {head_canonical}')
print(f'head.weight in state: {"head.weight" in state}')
print(f'embed.weight in state: {"embed.weight" in state}')

# Save original embed for comparison
orig_embed = state.get('embed.weight', state.get(head_canonical)).clone()

# Step 2: norm folding
nf = NormFoldingV2Key()
r2 = nf.forward(state)
state = r2.weights

# Check if embed was modified
if 'embed.weight' in state:
    diff = (orig_embed.float() - state['embed.weight'].float()).abs().max().item()
    print(f'embed.weight diff vs pre-fold: {diff:.6f}')
    if diff > 0.001:
        print('BUG: embed.weight was modified by norm folding!')
        print('  This happens because head.weight was deduped to embed.weight')
        print('  and norm folding modifies head.weight (folds ln_f into it)')

# Check ln_f
lnf_key = 'ln_f.weight'
if lnf_key in state:
    is_identity = (state[lnf_key] == 1.0).all().item()
    print(f'ln_f.weight identity: {is_identity}')

# The fix: norm folding should fold ln_f into head.weight
# but if head.weight is deduped (shares storage with embed.weight),
# it will corrupt embed.weight too!
# Solution: clone head.weight before folding, or fold ln_f into embed instead
