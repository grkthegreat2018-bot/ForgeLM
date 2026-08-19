"""Profile sft_train.py bottlenecks: time each phase of a training step."""
import os, sys, time, json
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import torch
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

# ── Phase 1: Data loading ──
t0 = time.perf_counter()
examples = load_examples(["research/data/finetune/forgelm_v3_train.jsonl"])
t1 = time.perf_counter()
print(f"Data load:        {(t1-t0)*1000:7.0f}ms ({len(examples)} examples)")

# ── Phase 2: Tokenization ──
tok = get_tokenizer()
pad_id = tok.pad_token_id or 0
t0 = time.perf_counter()
dataset = []
for ex in examples[:5000]:  # sample for profiling
    dataset.extend(tokenize_example(ex, tok, 256))
t1 = time.perf_counter()
print(f"Tokenize 5K:      {(t1-t0)*1000:7.0f}ms ({len(dataset)} usable, {(t1-t0)/5000*1e6:.0f}us/ex)")

# ── Phase 3: Model build ──
t0 = time.perf_counter()
cfg = get_config("forgelm_v3", device=device)
cfg.use_gradient_checkpointing = True
cfg.use_chunked_ce = True
cfg.ce_chunk_size = 128
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/ForgeLM_V3_Base.safetensors", dtype=dtype)
model.to(device).train()
t1 = time.perf_counter()
print(f"Model build:      {(t1-t0)*1000:7.0f}ms")

# ── Phase 4: BitNet + LoRA ──
t0 = time.perf_counter()
n_conv, n_already = convert_to_bitnet_everywhere(model)
n_adapters, lora_params = add_lora_adapters(model, rank=32, alpha=64, target_modules=None)
t1 = time.perf_counter()
print(f"BitNet+LoRA:      {(t1-t0)*1000:7.0f}ms ({n_conv} converted, {n_adapters} adapters)")

# ── Phase 5: Optimizer ──
t0 = time.perf_counter()
optimizer = configure_optimizer(model, 3e-4, 0.01, optimizer_name="muon_sf_plain")
t1 = time.perf_counter()
print(f"Optimizer setup:  {(t1-t0)*1000:7.0f}ms")

# ── Phase 6: Training step profiling ──
print(f"\n--- Training step profiling (5 steps, grad_mixup=3) ---")

import random
random.seed(42)

def get_batch(dataset, batch_size=2):
    batch = random.sample(dataset, batch_size)
    return collate_batch(batch, pad_id, device)

# Warmup
print("  Warmup...")
for _ in range(2):
    input_ids, labels, attn_mask, rewards = get_batch(dataset)
    loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=0.5)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
torch.cuda.synchronize()

# Profile each sub-step
step_times = {"data": [], "forward": [], "backward": [], "optimizer": [], "total": []}

for step in range(5):
    torch.cuda.synchronize()
    t_step_start = time.perf_counter()

    # Grad mixup: 3 forward+backward, 1 optimizer step
    for mix in range(3):
        t_data = time.perf_counter()
        input_ids, labels, attn_mask, rewards = get_batch(dataset)
        torch.cuda.synchronize()
        t_data = time.perf_counter() - t_data
        step_times["data"].append(t_data)

        t_fwd = time.perf_counter()
        loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=0.5)
        loss_val = loss.item()
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_fwd
        step_times["forward"].append(t_fwd)

        t_bwd = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd
        step_times["backward"].append(t_bwd)

    t_opt = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_opt = time.perf_counter() - t_opt
    step_times["optimizer"].append(t_opt)

    t_total = time.perf_counter() - t_step_start
    step_times["total"].append(t_total)

    print(f"  Step {step+1}: loss={loss_val:.4f} data={step_times['data'][-3]*1000:.0f}+{step_times['data'][-2]*1000:.0f}+{step_times['data'][-1]*1000:.0f}ms "
         "fwd={:.0f}+{:.0f}+{:.0f}ms bwd={:.0f}+{:.0f}+{:.0f}ms opt={:.0f}ms total={:.0f}ms".format(
              step_times["forward"][-3]*1000, step_times["forward"][-2]*1000, step_times["forward"][-1]*1000,
              step_times["backward"][-3]*1000, step_times["backward"][-2]*1000, step_times["backward"][-1]*1000,
              t_opt*1000, t_total*1000))

# Summary
print(f"\n--- Summary (averages) ---")
avg_data = sum(step_times["data"]) / len(step_times["data"])
avg_fwd = sum(step_times["forward"]) / len(step_times["forward"])
avg_bwd = sum(step_times["backward"]) / len(step_times["backward"])
avg_opt = sum(step_times["optimizer"]) / len(step_times["optimizer"])
avg_total = sum(step_times["total"]) / len(step_times["total"])
print(f"  Data (per mixup):  {avg_data*1000:.0f}ms")
print(f"  Forward (per mixup): {avg_fwd*1000:.0f}ms")
print(f"  Backward (per mixup): {avg_bwd*1000:.0f}ms")
print(f"  3x fwd+bwd:        {(avg_data+avg_fwd+avg_bwd)*3*1000:.0f}ms")
print(f"  Optimizer:         {avg_opt*1000:.0f}ms")
print(f"  Total per step:    {avg_total*1000:.0f}ms")
print(f"  Est. 1000 steps:   {avg_total*1000:.0f}s = {avg_total*1000/60:.1f}min")
print(f"  VRAM:              {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
print(f"  VRAM reserved:     {torch.cuda.memory_reserved()/1e9:.2f} GB")

# GPU utilization check
print(f"\n  GPU compute utilization:")
print(f"  Forward+backward: {(avg_fwd+avg_bwd)*3 / avg_total * 100:.0f}% of step time")
print(f"  Data loading:     {avg_data*3 / avg_total * 100:.0f}% of step time")
print(f"  Optimizer:        {avg_opt / avg_total * 100:.0f}% of step time")
