"""Benchmark mixed technology combinations for the ForgeAI 360M MLA model.

Tests training and inference combos and prints a results table. Runs everything
in a single process to avoid orphaned-process issues. Keeps batch sizes small
(max 4) to stay within the 12 GB VRAM budget and avoid system RAM pressure.

Combos tested:
  TRAINING:
    T1: baseline (batch 2, no flags)
    T2: checkpointing + chunked CE (batch 4)
    T3: checkpointing + chunked CE + YaRN 4x (batch 1, seq 4096)
    T4: GaLore + checkpointing + chunked CE (batch 2)
  INFERENCE (all on the 20% Wanda-pruned model):
    I1: BF16 baseline
    I2: TorchAO Int8 weight-only
    I3: TorchAO FP8 weight-only
    I4: Wanda 20% + Int8 (stacked)
    I5: Wanda 20% + FP8 (stacked)
"""
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from research.config import get_config
from research.model_loader import ModelLoader
from research.training_utils import BinaryDataset

device = torch.device("cuda")
cfg = get_config("360m_mla")
SEQ = 256  # short seq for fast benchmark
N_STEPS = 10
N_INFER = 20

results = []


def log(label, **kw):
    line = f"{label:30s} | " + " | ".join(f"{k}={v}" for k, v in kw.items())
    print(line)
    results.append((label, kw))


def benchmark_training(label, batch_size, seq_len, use_ckpt=False, use_chunked=False, ce_chunk=256, yarn_factor=None, optimizer_name="bnb"):
    """Run N_STEPS training steps and report tok/s + peak VRAM."""
    c = get_config("360m_mla")
    c.seq_len = seq_len
    c.max_seq_len = max(c.max_seq_len, seq_len)
    if use_ckpt:
        c.use_gradient_checkpointing = True
    if use_chunked:
        c.use_chunked_ce = True
        c.ce_chunk_size = ce_chunk
    if yarn_factor is not None:
        c.rope_scaling = {"type": "yarn", "factor": yarn_factor, "original_max_position_embeddings": 1024, "beta_fast": 32.0, "beta_slow": 1.0}

    model = ModelLoader.build_model(c).to(device)
    model.train()

    from research.training_utils import configure_optimizer, get_lr
    opt = configure_optimizer(model, c.max_lr, c.weight_decay, optimizer_name)

    ds = BinaryDataset("research/data/train.bin", seq_len, c.vocab_size)

    # Warmup (2 steps)
    for _ in range(2):
        x, y = ds.get_batch(batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        x, y = ds.get_batch(batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_per_s = N_STEPS * batch_size * seq_len / elapsed
    vram = torch.cuda.max_memory_allocated() / 1e9

    del model, opt
    torch.cuda.empty_cache()
    log(label, batch=batch_size, seq=seq_len, tok_s=f"{tok_per_s:.0f}", vram_GB=f"{vram:.2f}")
    return tok_per_s, vram


def benchmark_inference(label, model, seq_len=256):
    """Run N_INFER forward passes and report ms/step + peak VRAM."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(N_INFER):
            out = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    ms = elapsed / N_INFER * 1000
    vram = torch.cuda.max_memory_allocated() / 1e9
    log(label, ms_step=f"{ms:.2f}", vram_GB=f"{vram:.2f}")
    return ms, vram


def eval_loss(model, seq_len=256, n_batches=5):
    """Quick val loss on a few batches."""
    ds = BinaryDataset("research/data/val.bin", seq_len, cfg.vocab_size)
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = ds.get_batch(2, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
            total += loss.item()
            n += 1
    return total / n


print("=" * 80)
print("FORGEAI MIX BENCHMARK — 360M MLA on RTX 5070")
print("=" * 80)

# ---------------------------------------------------------------------------
# TRAINING COMBOS
# ---------------------------------------------------------------------------
print("\n--- TRAINING COMBOS ---")
benchmark_training("T1: baseline", batch_size=2, seq_len=SEQ)
benchmark_training("T2: ckpt+chunkedCE", batch_size=4, seq_len=SEQ, use_ckpt=True, use_chunked=True)
benchmark_training("T3: ckpt+CE+YaRN4x", batch_size=1, seq_len=1024, use_ckpt=True, use_chunked=True, ce_chunk=128, yarn_factor=4.0)
benchmark_training("T4: GaLore+ckpt+CE", batch_size=2, seq_len=SEQ, use_ckpt=True, use_chunked=True, optimizer_name="galore")

# ---------------------------------------------------------------------------
# INFERENCE COMBOS (on unpruned and pruned models)
# ---------------------------------------------------------------------------
print("\n--- INFERENCE COMBOS ---")

# Unpruned BF16 baseline
m = ModelLoader.build_model(cfg, checkpoint_path="research/checkpoints/pretrained_llm.safetensors").to(device).eval()
loss_unpruned = eval_loss(m)
benchmark_inference("I1: BF16 unpruned", m)
del m; torch.cuda.empty_cache()

# Unpruned + Int8
from torchao.quantization import quantize_, Int8WeightOnlyConfig
m = ModelLoader.build_model(cfg, checkpoint_path="research/checkpoints/pretrained_llm.safetensors").to(device).eval()
quantize_(m, Int8WeightOnlyConfig())
benchmark_inference("I2: Int8 unpruned", m)
del m; torch.cuda.empty_cache()

# Unpruned + FP8
from torchao.quantization import Float8WeightOnlyConfig
m = ModelLoader.build_model(cfg, checkpoint_path="research/checkpoints/pretrained_llm.safetensors").to(device).eval()
quantize_(m, Float8WeightOnlyConfig())
benchmark_inference("I3: FP8 unpruned", m)
del m; torch.cuda.empty_cache()

# Pruned BF16
m = ModelLoader.build_model(cfg, checkpoint_path="research/checkpoints/pruned_llm.safetensors").to(device).eval()
loss_pruned = eval_loss(m)
benchmark_inference("I4: BF16 pruned20%", m)
del m; torch.cuda.empty_cache()

# Pruned + Int8 (stacked)
m = ModelLoader.build_model(cfg, checkpoint_path="research/checkpoints/pruned_llm.safetensors").to(device).eval()
quantize_(m, Int8WeightOnlyConfig())
benchmark_inference("I5: Int8 pruned20%", m)
del m; torch.cuda.empty_cache()

# Pruned + FP8 (stacked)
m = ModelLoader.build_model(cfg, checkpoint_path="research/checkpoints/pruned_llm.safetensors").to(device).eval()
quantize_(m, Float8WeightOnlyConfig())
benchmark_inference("I6: FP8 pruned20%", m)
del m; torch.cuda.empty_cache()

print(f"\n--- QUALITY (val loss, seq 256) ---")
print(f"unpruned:  {loss_unpruned:.4f} | ppl {math.exp(loss_unpruned):.2f}")
print(f"pruned20%: {loss_pruned:.4f} | ppl {math.exp(loss_pruned):.2f}")

# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Combo':30s} | {'Metric':20s} | {'Value':>10s}")
print("-" * 70)
for label, kw in results:
    for k, v in kw.items():
        print(f"{label:30s} | {k:20s} | {str(v):>10s}")
