"""Verify chunked entropy-weighted CE is fast + correct."""
import os, sys, time, random
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import torch
import torch.nn.functional as F
os.environ["FORGE_BITNET_KERNEL"] = "triton"
os.environ["FORGE_FUSED_ROPE_QKNORM"] = "1"

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer
from research.training.sft_train import load_examples, tokenize_example, collate_batch, compute_loss
from research.training.bitnet_lora import convert_to_bitnet_everywhere, add_lora_adapters
from research.training.training_utils import configure_optimizer

device = "cuda"
dtype = torch.bfloat16
BATCH = 2
SEQ_LEN = 256

examples = load_examples(["research/data/finetune/forgelm_v3_train.jsonl"])
tok = get_tokenizer()
pad_id = tok.pad_token_id or 0
dataset = []
for ex in examples[:5000]:
    dataset.extend(tokenize_example(ex, tok, SEQ_LEN))
print(f"Dataset: {len(dataset)} examples")

def get_batch():
    batch = random.sample(dataset, BATCH)
    return collate_batch(batch, pad_id, device)

# Build model with chunked CE + entropy
cfg = get_config("forgelm_v3", device=device)
cfg.use_gradient_checkpointing = False
cfg.use_chunked_ce = True
cfg.ce_chunk_size = 128
cfg.entropy_alpha = 0.5  # Will be set dynamically by compute_loss
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/ForgeLM_V3_Base.safetensors", dtype=dtype)
model.to(device).train()
convert_to_bitnet_everywhere(model)
add_lora_adapters(model, rank=32, alpha=64, target_modules=None)
optimizer = configure_optimizer(model, 3e-4, 0.01, optimizer_name="muon_sf_plain")

print(f"\n{'='*60}")
print(f"TEST 1: Chunked entropy-weighted CE (entropy=0.5)")
print(f"{'='*60}")

# Warmup
for _ in range(3):
    input_ids, labels, attn_mask, _ = get_batch()
    loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=0.5)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

# Measure
times = []
losses = []
for step in range(10):
    input_ids, labels, attn_mask, _ = get_batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=0.5)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    times.append(t)
    losses.append(loss.item())

avg = sum(times) / len(times) * 1000
print(f"  Avg step: {avg:.0f}ms (fwd+bwd+opt)")
print(f"  Loss: {losses[0]:.2f} -> {losses[-1]:.2f}")
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"  With grad_mixup=3: {avg*3:.0f}ms/step")
print(f"  1000 steps: {avg*3*1000/1000/60:.1f}min")

print(f"\n{'='*60}")
print(f"TEST 2: Plain chunked CE (entropy=0.0, fast path)")
print(f"{'='*60}")

# Warmup
for _ in range(3):
    input_ids, labels, attn_mask, _ = get_batch()
    loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=0.0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

times2 = []
for step in range(10):
    input_ids, labels, attn_mask, _ = get_batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=0.0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    times2.append(t)

avg2 = sum(times2) / len(times2) * 1000
print(f"  Avg step: {avg2:.0f}ms (fwd+bwd+opt)")
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"  With grad_mixup=3: {avg2*3:.0f}ms/step")
print(f"  1000 steps: {avg2*3*1000/1000/60:.1f}min")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Old (full logits entropy):     ~64500ms/step -> 3229min (54hrs)")
print(f"  New chunked entropy=0.5:       {avg:.0f}ms/step -> {avg*3*1000/1000/60:.1f}min (with mixup=3)")
print(f"  New chunked entropy=0.0:       {avg2:.0f}ms/step -> {avg2*3*1000/1000/60:.1f}min (with mixup=3)")
print(f"  Speedup (entropy=0.5):         {64500/avg:.1f}x")
print(f"  Speedup (entropy=0.0):         {64500/avg2:.1f}x")
