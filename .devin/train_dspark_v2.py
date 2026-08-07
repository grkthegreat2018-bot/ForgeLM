"""Thorough DSpark training for ForgeLM v1.

Trains for 500 steps with cosine LR schedule, gradient accumulation,
and periodic checkpointing. Saves best model by validation loss.

Usage:
    python -u .devin/train_dspark_v2.py
"""
import sys, os, time, math, torch
sys.path.insert(0, '.')

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch.nn.functional as F
from research.config import get_config
from research.model_loader import ModelLoader
from research.dspark import DSparkHead
from safetensors.torch import save_file

DEVICE = "cuda"
D_MODEL = 1536
VOCAB = 151936
N_PREDICT = 4
SEQ_LEN = 128
BATCH_SIZE = 1
GRAD_ACCUM = 4          # effective batch = 4
STEPS = 500
LR = 3e-4
WARMUP = 50
TV_WEIGHT = 0.2
CONF_WEIGHT = 0.5
SAVE_PATH = "research/checkpoints/dspark_forgelm_v1.safetensors"
SAVE_EVERY = 100

print("=" * 60)
print("DSpark v2 Training — ForgeLM v1")
print("=" * 60)
print(f"  Steps: {STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LR: {LR} (cosine, warmup={WARMUP}) | Seq: {SEQ_LEN}")

# 1. Load model (frozen)
print("\n[1/4] Loading ForgeLM v1...")
cfg = get_config("forgelm_v1", device=DEVICE)
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v1.safetensors")
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (frozen)")

# 2. DSpark head
print("\n[2/4] Creating DSparkHead...")
dspark = DSparkHead(
    d_model=D_MODEL, vocab_size=VOCAB, n_predict=N_PREDICT,
    n_layers=1, seq_rank=256, seq_mode="rnn",
).to(DEVICE)
n_dspark = sum(p.numel() for p in dspark.parameters())
print(f"  DSparkHead: {n_dspark/1e6:.1f}M params")

# 3. Data (CPU, lazy GPU transfer)
print("\n[3/4] Loading data...")
train_data = torch.frombuffer(
    open("research/data/train.bin", "rb").read(), dtype=torch.uint32
).long()  # CPU
val_data = torch.frombuffer(
    open("research/data/val.bin", "rb").read(), dtype=torch.uint32
).long()  # CPU
print(f"  Train: {train_data.numel()/1e6:.1f}M | Val: {val_data.numel()/1e6:.1f}M tokens")

# 4. Optimizer + scheduler
print(f"\n[4/4] Training {STEPS} steps...")
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(dspark.parameters(), lr=LR, weight_decay=0.01)
    print("  8-bit AdamW (bitsandbytes)")
except ImportError:
    optimizer = torch.optim.AdamW(dspark.parameters(), lr=LR, weight_decay=0.01)

def cosine_lr(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

pos_weights = torch.exp(-torch.arange(1, N_PREDICT + 1, dtype=torch.float) / N_PREDICT).to(DEVICE)


def compute_loss(batch):
    """Three-term DSpark loss."""
    with torch.no_grad():
        out = model(batch, return_hidden=True)
        hidden = out[2] if len(out) > 2 else out[0]

    logits_list, confidences = dspark(hidden, batch)

    # Target logits from main model
    with torch.no_grad():
        target_logits_list = []
        for k in range(N_PREDICT):
            offset = k + 1
            if offset >= batch.shape[1]:
                break
            tgt_hidden = hidden[:, :-offset, :]
            tgt_logits = model.head(tgt_hidden)
            target_logits_list.append(tgt_logits)

    # L_ce
    ce_loss = 0.0
    for k in range(min(N_PREDICT, len(target_logits_list))):
        offset = k + 1
        pred = logits_list[k][:, :-offset, :].contiguous()
        target = batch[:, offset:].contiguous()
        ce_loss = ce_loss + F.cross_entropy(
            pred.view(-1, pred.size(-1)), target.view(-1),
            ignore_index=-100) * pos_weights[k]

    # L_tv
    tv_loss = 0.0
    for k in range(min(N_PREDICT, len(target_logits_list))):
        offset = k + 1
        draft_probs = F.softmax(logits_list[k][:, :-offset, :].contiguous(), dim=-1)
        target_probs = F.softmax(target_logits_list[k].contiguous(), dim=-1)
        tv_loss = tv_loss + 0.5 * (draft_probs - target_probs).abs().sum(-1).mean() * pos_weights[k]

    # L_conf
    conf_loss = 0.0
    for k in range(min(N_PREDICT, len(target_logits_list))):
        offset = k + 1
        draft_probs = F.softmax(logits_list[k][:, :-offset, :].contiguous(), dim=-1)
        target_probs = F.softmax(target_logits_list[k].contiguous(), dim=-1)
        c_star = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(-1)
        c_pred = confidences[:, :-offset, k].contiguous()
        conf_loss = conf_loss + F.binary_cross_entropy(
            c_pred.clamp(1e-6, 1 - 1e-6), c_star.clamp(1e-6, 1 - 1e-6),
            reduction="mean") * pos_weights[k]

    return ce_loss + TV_WEIGHT * tv_loss + CONF_WEIGHT * conf_loss, {
        "ce": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
        "tv": tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss,
        "conf": conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss,
    }


@torch.no_grad()
def validate(n_batches=10):
    dspark.eval()
    total_loss = 0
    for _ in range(n_batches):
        idx = torch.randint(0, val_data.numel() - SEQ_LEN - 1, (BATCH_SIZE,))
        batch = torch.stack([val_data[i:i+SEQ_LEN] for i in idx]).to(DEVICE)
        loss, _ = compute_loss(batch)
        total_loss += loss.item()
    dspark.train()
    return total_loss / n_batches


best_val = float("inf")
t0 = time.time()
optimizer.zero_grad()

for step in range(STEPS):
    dspark.train()

    # Gradient accumulation
    for _ in range(GRAD_ACCUM):
        idx = torch.randint(0, train_data.numel() - SEQ_LEN - 1, (BATCH_SIZE,))
        batch = torch.stack([train_data[i:i+SEQ_LEN] for i in idx]).to(DEVICE)
        loss, stats = compute_loss(batch)
        (loss / GRAD_ACCUM).backward()

    torch.nn.utils.clip_grad_norm_(dspark.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        torch.cuda.empty_cache()

    if step % 25 == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / (step + 1) * (STEPS - step - 1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  step {step:4d}/{STEPS} | loss={loss.item():.4f} "
              f"(ce={stats['ce']:.3f} tv={stats['tv']:.3f} "
              f"conf={stats['conf']:.3f}) | lr={lr_now:.2e} | "
              f"{elapsed:.0f}s, ~{eta:.0f}s left")

    # Periodic validation + checkpoint
    if step > 0 and step % SAVE_EVERY == 0:
        val_loss = validate()
        print(f"  >> val_loss={val_loss:.4f} (best={best_val:.4f})")
        if val_loss < best_val:
            best_val = val_loss
            state = {f"dspark.{k}": v.cpu() for k, v in dspark.state_dict().items()}
            save_file(state, SAVE_PATH)
            print(f"  >> Saved best checkpoint (val={val_loss:.4f})")

# Final save
val_loss = validate()
print(f"\nFinal val_loss={val_loss:.4f} (best={best_val:.4f})")
if val_loss <= best_val:
    state = {f"dspark.{k}": v.cpu() for k, v in dspark.state_dict().items()}
    save_file(state, SAVE_PATH)
    print(f"Saved final checkpoint to {SAVE_PATH}")

total = time.time() - t0
print(f"\nDone in {total:.0f}s ({total/60:.1f} min)")
print(f"  DSparkHead: {n_dspark/1e6:.1f}M params")
print(f"  Best val loss: {best_val:.4f}")
print(f"  Checkpoint: {SAVE_PATH}")
