"""Debug norm folding layer count."""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from safetensors.torch import load_file
from research.keys.norm_folding_v2_key import NormFoldingV2Key

state = load_file('research/checkpoints/forgelm_v2.safetensors')

ln1_keys = [k for k in state if 'ln1.weight' in k and 'blocks.' in k]
print(f'ln1 keys: {len(ln1_keys)}')

nf = NormFoldingV2Key()
r = nf.forward(state)
print(f'Folded: {r.metadata["n_folded"]}')
print(f'Identity norms: {r.metadata["n_identity_norms"]}/{r.metadata["n_total_norms"]}')

state = r.weights
for k in sorted(ln1_keys):
    is_id = (state[k] == 1.0).all().item()
    if not is_id:
        print(f'  NOT IDENTITY: {k}')
