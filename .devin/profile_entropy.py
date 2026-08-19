"""Micro-benchmark: isolate what's slow in the entropy weighting path."""
import os, sys, time
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

device = "cuda"
B, T, V = 2, 256, 65536

# Simulate logits from model
print(f"Logits shape: [{B}, {T}, {V}] = {B*T*V/1e6:.1f}M elements")
print(f"  bf16: {B*T*V*2/1e6:.1f}MB, fp32: {B*T*V*4/1e6:.1f}MB\n")

# Test A: Just model forward (no loss)
from research.config import get_config
from research.model_loader import ModelLoader
from research.training.bitnet_lora import convert_to_bitnet_everywhere, add_lora_adapters

cfg = get_config("forgelm_v3", device=device)
cfg.use_gradient_checkpointing = False
cfg.use_chunked_ce = False
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/ForgeLM_V3_Base.safetensors", dtype=torch.bfloat16)
model.to(device).train()
convert_to_bitnet_everywhere(model)
add_lora_adapters(model, rank=32, alpha=64, target_modules=None)

input_ids = torch.randint(0, V, (B, T), device=device)
attn_mask = torch.ones(B, T, device=device, dtype=torch.bfloat16)
labels = torch.randint(0, V, (B, T), device=device)
labels[:, :T//2] = -100  # completion-only

# Warmup
for _ in range(2):
    out = model(input_ids, attention_mask=attn_mask)
    loss = F.cross_entropy(out[0].float().view(-1, V), labels.view(-1), ignore_index=-100)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
torch.cuda.synchronize()

# Test A: Model forward only
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    out = model(input_ids, attention_mask=attn_mask)
torch.cuda.synchronize()
t_fwd = (time.perf_counter() - t0) / 5 * 1000
print(f"[A] Model forward only:              {t_fwd:.0f}ms")

# Test B: Forward + CE loss (no entropy)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    out = model(input_ids, attention_mask=attn_mask)
    logits = out[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(shift_logits.view(-1, V).float(), shift_labels.view(-1), ignore_index=-100, reduction="none")
torch.cuda.synchronize()
t_ce = (time.perf_counter() - t0) / 5 * 1000
print(f"[B] Forward + CE (no entropy):       {t_ce:.0f}ms  (CE alone: {t_ce-t_fwd:.0f}ms)")

# Test C: Forward + CE + softmax for entropy
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    out = model(input_ids, attention_mask=attn_mask)
    logits = out[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(shift_logits.view(-1, V).float(), shift_labels.view(-1), ignore_index=-100, reduction="none").view(shift_labels.size())
    with torch.no_grad():
        probs = F.softmax(shift_logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
torch.cuda.synchronize()
t_ent = (time.perf_counter() - t0) / 5 * 1000
print(f"[C] Forward + CE + entropy:          {t_ent:.0f}ms  (entropy alone: {t_ent-t_ce:.0f}ms)")

# Test D: Just the entropy computation (no model forward)
shift_logits = torch.randn(B, T-1, V, device=device, dtype=torch.bfloat16, requires_grad=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        probs = F.softmax(shift_logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
torch.cuda.synchronize()
t_ent_only = (time.perf_counter() - t0) / 10 * 1000
print(f"[D] Entropy only (no model fwd):     {t_ent_only:.0f}ms")

# Test E: Just softmax
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        probs = F.softmax(shift_logits.float(), dim=-1)
torch.cuda.synchronize()
t_softmax = (time.perf_counter() - t0) / 10 * 1000
print(f"[E] Softmax only:                    {t_softmax:.0f}ms")

# Test F: Just log + multiply + sum
probs = F.softmax(shift_logits.float(), dim=-1)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
torch.cuda.synchronize()
t_logsum = (time.perf_counter() - t0) / 10 * 1000
print(f"[F] Log+mul+sum only:                {t_logsum:.0f}ms")

# Test G: Full compute_loss with entropy (backward too)
from research.training.sft_train import compute_loss
attn_mask_bf = torch.ones(B, T, device=device, dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    loss = compute_loss(model, input_ids, labels, attn_mask_bf, entropy_alpha=0.5)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
torch.cuda.synchronize()
t_full = (time.perf_counter() - t0) / 3 * 1000
print(f"[G] Full compute_loss(entropy=0.5):  {t_full:.0f}ms  (fwd+bwd)")

# Test H: Full compute_loss without entropy
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    loss = compute_loss(model, input_ids, labels, attn_mask_bf, entropy_alpha=0.0)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()
torch.cuda.synchronize()
t_no_ent = (time.perf_counter() - t0) / 3 * 1000
print(f"[H] Full compute_loss(entropy=0.0):  {t_no_ent:.0f}ms  (fwd+bwd)")

print(f"\n  Entropy overhead: {t_full - t_no_ent:.0f}ms per step")
print(f"  With grad_mixup=3: {t_full * 3:.0f}ms/step vs {t_no_ent * 3:.0f}ms/step")
print(f"  1000 steps: {t_full * 3 * 1000 / 1000 / 60:.1f}min vs {t_no_ent * 3 * 1000 / 1000 / 60:.1f}min")
