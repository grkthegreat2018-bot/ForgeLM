"""Test AirLLM streaming inference: run XP model from shards, one layer at a time.

Simplified: load full model, then simulate streaming by freeing/reloding
each layer's weights from disk shards during forward pass.
"""
import sys, torch, time
sys.path.insert(0, '.')
from pathlib import Path
from safetensors.torch import load_file
from research.config import get_config
from research.model_loader import ModelLoader
from transformers import AutoTokenizer

SHARD_DIR = Path("research/checkpoints/xp_shards")
DEVICE = 'cuda'
DTYPE = torch.bfloat16

tok = AutoTokenizer.from_pretrained('research/checkpoints/qwen_hf')

# Load full model normally (we'll simulate streaming by tracking VRAM)
cfg = get_config('qwen25_coder_1.5b', device=DEVICE)
model = ModelLoader.build_model(cfg, checkpoint_path='research/checkpoints/xp_full_no_mqa.safetensors').to(DEVICE, dtype=DTYPE).eval()

# List shards
shards = sorted(SHARD_DIR.glob("shard_*.safetensors"))
print(f"Shards: {len(shards)}")

# Build param name → param map
param_map = {name: p for name, p in model.named_parameters()}

# Load shard 0 (embed, head, ln_f, extras) — keep these resident
shard0_state = load_file(str(shards[0]))
for kn, t in shard0_state.items():
    if kn in param_map:
        param_map[kn].data = t.to(DEVICE, dtype=DTYPE)

# Free all layer weights (move to CPU to simulate not in VRAM)
layer_shards = shards[1:]
print(f"Layer shards: {len(layer_shards)}")

# Save original layer weights to CPU, then free from VRAM
layer_weights_cpu = {}
for li in range(len(model.blocks)):
    for name, p in model.blocks[li].named_parameters():
        layer_weights_cpu[f"blocks.{li}.{name}"] = p.data.cpu()
        p.data = torch.empty(0, device='meta') if p.data.numel() == 0 else p.data.cpu()

# Actually, let's just test that loading from shards produces same output
# as the full model. Load full model separately for comparison.
model_full = ModelLoader.build_model(cfg, checkpoint_path='research/checkpoints/xp_full_no_mqa.safetensors').to(DEVICE, dtype=DTYPE).eval()

# Streaming generation: load each layer from shard, compute, free
def generate_streaming(prompt, max_new=30):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    for step in range(max_new):
        with torch.inference_mode():
            x = model.embed(ids)
            for li, block in enumerate(model.blocks):
                # Load this layer's weights from shard
                state = load_file(str(layer_shards[li]))
                for kn, t in state.items():
                    if kn in param_map:
                        param_map[kn].data = t.to(DEVICE, dtype=DTYPE)
                # Compute
                x = block(x)
                if isinstance(x, tuple):
                    x = x[0]
                # Free layer weights back to CPU
                for kn in state:
                    if kn in param_map:
                        param_map[kn].data = param_map[kn].data.cpu()
                del state
            x = model.ln_f(x)
            logits = model.head(x)
        nid = logits[0, -1].argmax().item()
        if nid == 151645:
            break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEVICE)], dim=-1)
    return ids[0]

def generate_full(prompt, max_new=30):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    for _ in range(max_new):
        with torch.inference_mode():
            o = model_full(ids); o = o[0] if isinstance(o, tuple) else o
        nid = o[0,-1].argmax().item()
        if nid == 151645: break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEVICE)], -1)
    return ids[0]

PROMPTS = ["def fibonacci(n):", "The meaning of life is"]
print("\n=== Streaming vs Full Inference ===")
for p in PROMPTS:
    t0 = time.time()
    out_s = generate_streaming(p, max_new=30)
    dt_s = time.time() - t0

    t0 = time.time()
    out_f = generate_full(p, max_new=30)
    dt_f = time.time() - t0

    match = torch.equal(out_s, out_f)
    print(f"\nPrompt: {p!r}")
    print(f"  Streaming: {dt_s:.1f}s | {tok.decode(out_s, skip_special_tokens=True)[:100]}")
    print(f"  Full:      {dt_f:.1f}s | {tok.decode(out_f, skip_special_tokens=True)[:100]}")
    print(f"  Match: {match}")
