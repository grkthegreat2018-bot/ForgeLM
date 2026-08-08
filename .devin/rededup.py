"""Re-dedup the distilled checkpoint to save space."""
import torch, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from safetensors.torch import load_file, save_file
from research.keys.tensor_dedup_key import TensorDedupKey

state = load_file('research/checkpoints/forgelm_v2_opt_distilled.safetensors')
before = len(state)
before_mb = sum(t.numel()*t.element_size() for t in state.values()) / 1e6
print(f'Before dedup: {before} tensors, {before_mb:.1f} MB')

td = TensorDedupKey()
r = td.forward(state)
state = r.weights
after = len(state)
after_mb = sum(t.numel()*t.element_size() for t in state.values()) / 1e6
print(f'After dedup: {after} tensors, {after_mb:.1f} MB')
print(f'Saved: {r.metadata["n_deduplicated"]} tensors, {r.metadata["saved_mb"]:.1f} MB')

save_file(state, 'research/checkpoints/forgelm_v2_opt_distilled_dedup.safetensors')
meta = json.load(open('research/checkpoints/forgelm_v2_opt.safetensors.meta.json'))
meta['dedup_map'].update(r.metadata['dedup_map'])
json.dump(meta, open('research/checkpoints/forgelm_v2_opt_distilled.safetensors.meta.json', 'w'), indent=2)
sz = os.path.getsize('research/checkpoints/forgelm_v2_opt_distilled.safetensors')
print(f'Saved: {sz/1e6:.1f} MB')
