"""Test DiffusionBlocks on ForgeLM V3 (16 layers, 1.2B params).

Meures actual memory savings on the real model.
"""
import os, torch, gc, time
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.config import get_config
from research.model_loader import load_default_model
from research.checkpoint_io import load_checkpoint
from research.diffusion_blocks import DiffusionBlocks, DiffusionBlockConfig
import torch.nn.functional as F

# ── Build V3 model ──
print("=== Building ForgeLM V3 ===")
model, tok = load_default_model("forgelm_v3")
ckpt_path = "research/checkpoints/ForgeLM_V3_Base.safetensors"
if not os.path.exists(ckpt_path):
    ckpt_path = "research/checkpoints/ForgeLM_V2_BSP.safetensors"
print(f"Loading checkpoint: {ckpt_path}")
sd = load_checkpoint(ckpt_path)
model.load_state_dict({k: v for k, v in sd.items()}, strict=False)
model = model.to("cuda").to(torch.bfloat16)
model.eval()
config = model.config
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params, {len(model.blocks)} layers")
torch.cuda.empty_cache()

# ── Create DiffusionBlocks with B=4 ──
print("\n=== DiffusionBlocks B=4 ===")
db_config = DiffusionBlockConfig(
    num_blocks=4,
    use_noise_conditioning=True,
    cond_dim=256,
)
dblock = DiffusionBlocks(
    model=model,
    config=db_config,
    d_model=config.d_model,
    num_layers=len(model.blocks),
)

# ── Memory comparison ──
print("\n=== Memory Comparison (batch=2, seq=512) ===")
input_ids = torch.randint(0, config.vocab_size, (2, 512), device="cuda")
labels = torch.randint(0, config.vocab_size, (2, 512), device="cuda")

# Standard training
model.train()
for p in model.parameters():
    p.requires_grad = True
optimizer_full = torch.optim.AdamW(model.parameters(), lr=1e-4)
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model(input_ids, targets=labels)
loss = out[1] if isinstance(out, tuple) and len(out) > 1 else None
if loss is None:
    logits = out[0] if isinstance(out, tuple) else out
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
loss.backward()
t1 = time.time()
mem_full = torch.cuda.max_memory_allocated() / 1e9
print(f"Standard training: {mem_full:.2f} GB, {t1-t0:.2f}s")
optimizer_full.zero_grad()
del loss, out
torch.cuda.empty_cache()
gc.collect()

# DiffusionBlocks training (1 block = 4 layers)
for b in range(4):
    dblock.freeze_all_except_block(b)
    block_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in block_params)
    optimizer_block = torch.optim.AdamW(block_params, lr=1e-4)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = dblock.train_step(input_ids, labels, optimizer_block, block_idx=b)
    t1 = time.time()
    mem_block = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Block {b}: {mem_block:.2f} GB, {t1-t0:.2f}s, "
          f"{n_params/1e6:.1f}M params, loss={result['ce_loss']:.3f}")
    optimizer_block.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()

# ── Memory comparison with larger batch ──
print("\n=== Memory: DiffusionBlocks allows B× larger batch ===")
dblock.freeze_all_except_block(0)
block_params = [p for p in model.parameters() if p.requires_grad]
optimizer_block = torch.optim.AdamW(block_params, lr=1e-4)

for batch_size in [2, 4, 8, 16]:
    torch.cuda.empty_cache()
    gc.collect()
    try:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 512), device="cuda")
        labels = torch.randint(0, config.vocab_size, (batch_size, 512), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        result = dblock.train_step(input_ids, labels, optimizer_block, block_idx=0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Batch {batch_size}: {mem:.2f} GB, loss={result['ce_loss']:.3f}")
        optimizer_block.zero_grad()
        del input_ids, labels
        torch.cuda.empty_cache()
        gc.collect()
    except torch.cuda.OutOfMemoryError:
        print(f"  Batch {batch_size}: OOM")
        torch.cuda.empty_cache()
        gc.collect()
        break

print("\n=== DONE ===")
