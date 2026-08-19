import sys; sys.path.insert(0, 'D:/windsurf/ForgeAI')
import os; os.environ['FORGE_BITNET_KERNEL']='triton'; os.environ['FORGE_FUSED_ROPE_QKNORM']='1'
import torch
from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer

tok = get_tokenizer()
device = 'cuda'

# Load final SFT
cfg = get_config('forgelm_v3', device=device)
model = ModelLoader.build_model_fast(cfg, checkpoint_path='research/checkpoints/ForgeLM_V3_SFT.safetensors', dtype=torch.bfloat16)
model.to(device).eval()

# Load base for comparison
del model; torch.cuda.empty_cache()
cfg2 = get_config('forgelm_v3', device=device)
model_base = ModelLoader.build_model_fast(cfg2, checkpoint_path='research/checkpoints/ForgeLM_V3_Base.safetensors', dtype=torch.bfloat16)
model_base.to(device).eval()

# Test: forward pass with "The capital of France is" and check next token
prompt = "The capital of France is"
ids = tok.encode(prompt, return_tensors='pt').to(device)
print(f"Prompt: {prompt}")
print(f"Token IDs: {ids[0].tolist()}")

with torch.no_grad():
    out_base = model_base(ids)
    logits_base = out_base[0] if isinstance(out_base, tuple) else out_base
    next_token_base = logits_base[0, -1, :].argmax().item()

del model_base; torch.cuda.empty_cache()
cfg3 = get_config('forgelm_v3', device=device)
model_sft = ModelLoader.build_model_fast(cfg3, checkpoint_path='research/checkpoints/ForgeLM_V3_SFT.safetensors', dtype=torch.bfloat16)
model_sft.to(device).eval()

with torch.no_grad():
    out_sft = model_sft(ids)
    logits_sft = out_sft[0] if isinstance(out_sft, tuple) else out_sft
    next_token_sft = logits_sft[0, -1, :].argmax().item()

print(f"\nBase model next token: {next_token_base} = '{tok.decode([next_token_base])}'")
print(f"SFT model next token:  {next_token_sft} = '{tok.decode([next_token_sft])}'")

# Top 5 for each
with torch.no_grad():
    top5_base = torch.topk(logits_base[0, -1, :], 5)
    top5_sft = torch.topk(logits_sft[0, -1, :], 5)

print(f"\nBase top-5: {[(tok.decode([t.item()]), p.item()) for t, p in zip(top5_base.indices, top5_base.values)]}")
print(f"SFT top-5:  {[(tok.decode([t.item()]), p.item()) for t, p in zip(top5_sft.indices, top5_sft.values)]}")

# Check logit statistics
print(f"\nBase logits: mean={logits_base[0,-1,:].float().mean():.4f} std={logits_base[0,-1,:].float().std():.4f}")
print(f"SFT logits:  mean={logits_sft[0,-1,:].float().mean():.4f} std={logits_sft[0,-1,:].float().std():.4f}")
