"""Final comprehensive test of all performance improvements."""
import sys, os, torch
sys.path.insert(0, '.')

from research.inference.forge_engine import ForgeEngine
from research.mtp import MTPHead
from research.config import get_config
from safetensors.torch import load_file

print('=== FINAL COMPREHENSIVE TEST ===')
print()

# Test 1: MLA + MoE model
print('--- Test 1: MLA + MoE (top-4) ---')
engine = ForgeEngine.from_checkpoint(
    checkpoint='research/checkpoints/xp_mla_moe.safetensors',
    config_name='xp_1.5b_mla_moe',
    device='cuda',
)
engine.activate(kv_cache='standard', decoding='standard')
out1 = engine.generate('def fibonacci(n):', max_new_tokens=40)
print(f'Output: {out1[:80]}')
s1 = engine.benchmark('def fibonacci(n):', max_new_tokens=30)
print()

# Test 2: MLA + MoE with streaming KV
print('--- Test 2: MLA + MoE + Streaming KV ---')
engine.activate(kv_cache='streaming', decoding='standard')
out2 = engine.generate('def fibonacci(n):', max_new_tokens=40)
print(f'Output: {out2[:80]}')
s2 = engine.benchmark('def fibonacci(n):', max_new_tokens=30)
print()

# Test 3: MLA + MoE with Hadamard INT4 KV
print('--- Test 3: MLA + MoE + Hadamard INT4 KV ---')
engine.activate(kv_cache='hadamard_int4', decoding='standard')
out3 = engine.generate('def fibonacci(n):', max_new_tokens=40)
print(f'Output: {out3[:80]}')
s3 = engine.benchmark('def fibonacci(n):', max_new_tokens=30)
print()

# Test 4: Original GQA with MTP self-spec
print('--- Test 4: Original GQA + MTP self-spec ---')
del engine
torch.cuda.empty_cache()

engine2 = ForgeEngine.from_checkpoint(
    checkpoint='research/checkpoints/xp_full_no_mqa.safetensors',
    config_name='qwen25_coder_1.5b',
    device='cuda',
)
cfg = get_config('qwen25_coder_1.5b')
mtp_state = load_file('research/checkpoints/xp_finetuned_mtp.safetensors')
mtp_head = MTPHead(d_model=cfg.d_model, vocab_size=cfg.vocab_size, n_predict=4).to('cuda')
mtp_head.load_state_dict(mtp_state)
engine2.model.mtp_head = mtp_head

engine2.activate(kv_cache='standard', decoding='mtp_selfspec')
out4 = engine2.generate('def fibonacci(n):', max_new_tokens=40)
print(f'Output: {out4[:80]}')
s4 = engine2.benchmark('def fibonacci(n):', max_new_tokens=30)
print()

print('=== SUMMARY ===')
print(f'MLA+MoE standard:     {s1["tokens_per_sec"]:.0f} tok/s | {out1[:40]}')
print(f'MLA+MoE streaming:    {s2["tokens_per_sec"]:.0f} tok/s | {out2[:40]}')
print(f'MLA+MoE hadamard:     {s3["tokens_per_sec"]:.0f} tok/s | {out3[:40]}')
print(f'GQA MTP self-spec:    {s4["tokens_per_sec"]:.0f} tok/s | {out4[:40]}')
all_correct = out1[:30] == out2[:30] == out3[:30] == out4[:30]
print(f'All outputs match: {all_correct}')
