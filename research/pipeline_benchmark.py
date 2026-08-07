"""Full alignment pipeline benchmark: SFT(LISA) -> ORPO -> Wanda prune -> eval.

Tests the end-to-end pipeline combining LISA (memory-efficient SFT), ORPO
(no-reference preference alignment), and Wanda pruning (inference speedup).
Measures loss at each stage and final inference throughput.
"""
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader
from research.training_utils import BinaryDataset, configure_optimizer, get_lr
from research.dpo_align import build_preference_sample, _logp_for_completion, orpo_loss
from research.wanda_prune import _capture_layer_inputs, wanda_prune

device = torch.device("cuda")
cfg = get_config("360m_mla")
cfg.seq_len = 256
cfg.max_seq_len = max(cfg.max_seq_len, 256)

CKPT = "research/checkpoints/pretrained_llm.safetensors"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

print("=" * 70)
print("FULL PIPELINE: Pretrain -> SFT(LISA) -> ORPO -> Wanda -> Eval")
print("=" * 70)

# --- Stage 0: Load pretrained model and measure baseline ---
print("\n[Stage 0] Loading pretrained model...")
model = ModelLoader.build_model(cfg, checkpoint_path=CKPT).to(device)
model.train()

def quick_eval(m, tag, seq_len=256, n=5):
    ds = BinaryDataset("research/data/val.bin", seq_len, cfg.vocab_size)
    m.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(n):
            x, y = ds.get_batch(2, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = m(x); logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
            tot += loss.item()
    avg = tot / n
    print(f"  [{tag}] val loss: {avg:.4f} | ppl: {math.exp(avg):.2f}")
    return avg

def quick_infer(m, tag, seq_len=256, n=20):
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        for _ in range(3): _ = m(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): _ = m(x)
        torch.cuda.synchronize(); ms = (time.perf_counter()-t0)/n*1000
    vram = torch.cuda.max_memory_allocated()/1e9
    print(f"  [{tag}] inference: {ms:.2f} ms/step | {vram:.2f} GB")
    return ms, vram

loss0 = quick_eval(model, "pretrained")
ms0, vram0 = quick_infer(model, "pretrained")

# --- Stage 1: SFT with LISA (synthetic data, 30 steps) ---
print("\n[Stage 1] SFT with LISA (top-4 of 19 layers, 30 steps)...")
model.train()
blocks = list(model.blocks)
lisa_active = set()
LISA_K = 4
LISA_INTERVAL = 10

# Synthetic SFT samples (prompt -> completion)
sft_samples = [
    ("The capital of France is", " Paris, the city of light."),
    ("Python is a", " programming language."),
    ("The sky is", " blue during the day."),
    ("2 + 2 equals", " 4."),
    ("Machine learning is", " a subset of artificial intelligence."),
]
sft_tokens = []
for prompt, completion in sft_samples:
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (p + c)[:cfg.seq_len]
    labels = [-100] * len(p) + c
    labels = labels[:len(ids)]
    sft_tokens.append((torch.tensor(ids), torch.tensor(labels)))

opt = configure_optimizer(model, 2e-4, cfg.weight_decay, "bnb")

def lisa_recompute(model, x_dummy, y_dummy):
    """Probe importance and select top-k layers."""
    global lisa_active
    for p in model.parameters(): p.requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x_dummy, targets=y_dummy)
    loss.backward()
    imps = []
    for i, blk in enumerate(blocks):
        g2 = sum((p.grad.norm()**2).item() if p.grad is not None else 0.0 for p in blk.parameters())
        imps.append((g2**0.5, i))
    imps.sort(reverse=True)
    new_active = {i for _, i in imps[:LISA_K]}
    for i in lisa_active - new_active:
        for p in blocks[i].parameters(): p.requires_grad_(False)
    for i in new_active - lisa_active:
        for p in blocks[i].parameters(): p.requires_grad_(True)
    lisa_active = new_active
    opt.zero_grad(set_to_none=True)
    print(f"  LISA active layers: {sorted(new_active)}")

for step in range(30):
    ids, labels = sft_tokens[step % len(sft_tokens)]
    x = ids.unsqueeze(0).to(device)
    y = labels.unsqueeze(0).to(device)
    # Pad to same length
    if x.shape[1] < cfg.seq_len:
        pad = torch.full((1, cfg.seq_len - x.shape[1]), tokenizer.pad_token_id or 0, device=device)
        x = torch.cat([x, pad], dim=1)
        ypad = torch.full((1, cfg.seq_len - y.shape[1]), -100, device=device)
        y = torch.cat([y, ypad], dim=1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, targets=y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    lr = get_lr(step, 30, 2e-4, 2e-5, 5)
    for g in opt.param_groups: g["lr"] = lr
    opt.step()
    opt.zero_grad(set_to_none=True)
    if (step + 1) % LISA_INTERVAL == 0 and step > 0:
        lisa_recompute(model, x, y)
    if (step+1) % 10 == 0:
        print(f"  SFT step {step+1}/30 | loss {loss.item():.4f}")

# Re-enable all grads for next stage
for p in model.parameters(): p.requires_grad_(True)
loss1 = quick_eval(model, "after SFT(LISA)")

# --- Stage 2: ORPO alignment (20 steps) ---
print("\n[Stage 2] ORPO alignment (20 steps)...")
model.train()
pref_samples = [
    build_preference_sample(tokenizer, "The sky is", " blue.", " green.", cfg.seq_len),
    build_preference_sample(tokenizer, "Paris is the capital of", " France.", " Germany.", cfg.seq_len),
    build_preference_sample(tokenizer, "2 + 2 =", " 4.", " 5.", cfg.seq_len),
    build_preference_sample(tokenizer, "Python is a", " programming language.", " type of snake.", cfg.seq_len),
    build_preference_sample(tokenizer, "The sun rises in the", " east.", " west.", cfg.seq_len),
]

opt2 = configure_optimizer(model, 5e-7, cfg.weight_decay, "bnb")
for step in range(20):
    s = pref_samples[step % len(pref_samples)]
    chosen_ids = torch.tensor(s["chosen_ids"], dtype=torch.long)
    rejected_ids = torch.tensor(s["rejected_ids"], dtype=torch.long)
    chosen_logp = _logp_for_completion(model, chosen_ids, s["chosen_start"], device)
    rejected_logp = _logp_for_completion(model, rejected_ids, s["rejected_start"], device)
    comp_len_c = max(1, len(s["chosen_ids"]) - s["chosen_start"])
    comp_len_r = max(1, len(s["rejected_ids"]) - s["rejected_start"])
    loss = orpo_loss(chosen_logp/comp_len_c, rejected_logp/comp_len_r, -chosen_logp/comp_len_c, lam=1.0)
    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    lr = get_lr(step, 20, 5e-7, 5e-8, 5)
    for g in opt2.param_groups: g["lr"] = lr
    opt2.step()
    if (step+1) % 5 == 0:
        print(f"  ORPO step {step+1}/20 | loss {loss.item():.4f}")

loss2 = quick_eval(model, "after ORPO")

# --- Stage 3: Wanda prune 20% ---
print("\n[Stage 3] Wanda pruning (20% sparsity)...")
model.eval()
target_layers = [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and mod.weight.ndim == 2]
ds = BinaryDataset("research/data/train.bin", cfg.seq_len, cfg.vocab_size)
calib = [ds.get_batch(2, device)[0] for _ in range(8)]
norms = _capture_layer_inputs(model, target_layers, calib, device)
actual_sparsity = wanda_prune(model, 0.2, norms, device)

loss3 = quick_eval(model, "after Wanda 20%")
ms3, vram3 = quick_infer(model, "after Wanda 20%")

# --- Save the final pipeline model ---
from research.checkpoint_io import save_checkpoint
out = "research/checkpoints/pipeline_llm.safetensors"
save_checkpoint(model.state_dict(), out)

# --- Summary ---
print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"{'Stage':30s} | {'Val Loss':>10s} | {'PPL':>10s} | {'ms/step':>8s} | {'VRAM':>8s}")
print("-" * 75)
print(f"{'0: Pretrained':30s} | {loss0:10.4f} | {math.exp(loss0):10.2f} | {ms0:8.2f} | {vram0:7.2f}G")
print(f"{'1: + SFT(LISA)':30s} | {loss1:10.4f} | {math.exp(loss1):10.2f} | {'--':>8s} | {'--':>8s}")
print(f"{'2: + ORPO':30s} | {loss2:10.4f} | {math.exp(loss2):10.2f} | {'--':>8s} | {'--':>8s}")
print(f"{'3: + Wanda 20%':30s} | {loss3:10.4f} | {math.exp(loss3):10.2f} | {ms3:8.2f} | {vram3:7.2f}G")
print(f"\nInference speedup: {ms0/ms3:.2f}x | VRAM delta: {vram3-vram0:+.2f} GB")
print(f"Quality delta: {loss3-loss0:+.4f} loss ({math.exp(loss3)/math.exp(loss0)-1:+.2%} ppl)")
print(f"Final model saved to: {out}")
