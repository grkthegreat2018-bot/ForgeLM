"""Verify ported V7-8B-B checkpoint produces coherent output."""
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
    cfg, checkpoint_path="research/checkpoints/ForgeLM_V7_8B_B_ported.safetensors",
    dtype=torch.bfloat16,
)
tok = get_tokenizer()
model.eval()

for prompt in ["The capital of France is", "def fibonacci(n):", "Hello, how are you?"]:
    ids = tok.encode(prompt)
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, targets=None)
        logits = out[0] if isinstance(out, tuple) else out
    probs = torch.softmax(logits[0, -1], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\nPrompt: '{prompt}'")
    for idx, prob in zip(top5.indices, top5.values):
        t = tok.decode([idx.item()])
        print(f"  token {idx.item()}: '{t}' (p={prob.item():.4f})")

# Generate 30 tokens
prompt = "The capital of France is"
ids = tok.encode(prompt)
gen_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
print(f"\nGenerating 30 tokens from '{prompt}':")
with torch.no_grad():
    for _ in range(30):
        out = model(gen_ids, targets=None)
        logits = out[0] if isinstance(out, tuple) else out
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
print(f"  '{tok.decode(gen_ids[0].tolist())}'")
