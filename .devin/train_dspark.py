"""Train a DSparkHead for ForgeLM v1.

Custom training loop that calls model with return_hidden=True.
Saves to research/checkpoints/dspark_forgelm_v1.safetensors.
"""
import sys, os, time, torch
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
SEQ_LEN = 64
BATCH_SIZE = 1
STEPS = 100
LR = 1e-4
TV_WEIGHT = 0.2
CONF_WEIGHT = 0.5
SAVE_PATH = "research/checkpoints/dspark_forgelm_v1.safetensors"

print("=" * 60)
print("DSpark Head Training for ForgeLM v1")
print("=" * 60)

# 1. Load ForgeLM v1 model (frozen)
print("\n[1/4] Loading ForgeLM v1 model...")
cfg = get_config("forgelm_v1", device=DEVICE)
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v1.safetensors")
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (frozen)")

# 2. Create DSpark head — use n_layers=1 (single linear per head) to fit 12GB VRAM
# Tie W1 embedding with model's embedding to save ~78M params
print("\n[2/4] Creating DSparkHead...")
share_embed = model.embed  # tie W1 with model embedding (1536-dim, not 256 — use separate)
dspark = DSparkHead(
    d_model=D_MODEL,
    vocab_size=VOCAB,
    n_predict=N_PREDICT,
    n_layers=1,       # single Linear per head (no hidden layer) — ~234M × 4 = 936M → ~586M
    seq_rank=256,
    seq_mode="rnn",
).to(DEVICE)
n_dspark = sum(p.numel() for p in dspark.parameters())
print(f"  DSparkHead: {n_dspark/1e6:.1f}M params, n_predict={N_PREDICT}, n_layers=1")

# 3. Load training data
print("\n[3/4] Loading training data...")
# Keep data on CPU to avoid RAM/VRAM spike; move batches to GPU on demand
data = torch.frombuffer(
    open("research/data/train.bin", "rb").read(), dtype=torch.uint32
).long()  # CPU
print(f"  Train data: {data.numel()/1e6:.1f}M tokens (CPU)")

# 4. Custom training loop
print(f"\n[4/4] Training {STEPS} steps (batch={BATCH_SIZE}, seq={SEQ_LEN})...")
optimizer = torch.optim.AdamW(dspark.parameters(), lr=LR)
# Use 8-bit AdamW to save VRAM (1B param DSpark head)
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(dspark.parameters(), lr=LR)
    print("  Using 8-bit AdamW (bitsandbytes)")
except ImportError:
    print("  Using standard AdamW (bitsandbytes not available)")

# Position weights: w_k = exp(-(k-1)/gamma)
pos_weights = torch.exp(-torch.arange(1, N_PREDICT + 1, dtype=torch.float) / N_PREDICT).to(DEVICE)

t0 = time.time()
for step in range(STEPS):
    # Random batch — sample on CPU, move to GPU
    idx = torch.randint(0, data.numel() - SEQ_LEN - 1, (BATCH_SIZE,))
    batch = torch.stack([data[i:i+SEQ_LEN] for i in idx]).to(DEVICE)

    # Forward through frozen main model — get hidden states
    with torch.no_grad():
        out = model(batch, return_hidden=True)
        # (logits, loss, hidden) when return_hidden=True, no use_cache
        hidden = out[2]  # (B, T, D_MODEL)

    # DSpark forward
    dspark.train()
    logits_list, confidences = dspark(hidden, batch)
    # logits_list: list of (B, T, V), confidences: (B, T, gamma)

    # ── L_ce: cross-entropy ──
    ce_loss = 0.0
    for k in range(N_PREDICT):
        offset = k + 1
        if offset >= SEQ_LEN:
            break
        pred = logits_list[k][:, :-offset, :].contiguous()
        target = batch[:, offset:].contiguous()
        loss_k = F.cross_entropy(
            pred.view(-1, pred.size(-1)),
            target.view(-1),
            ignore_index=-100,
        )
        ce_loss = ce_loss + loss_k * pos_weights[k]

    # ── L_tv: total variation distance ──
    tv_loss = 0.0
    with torch.no_grad():
        target_logits_list = []
        for k in range(N_PREDICT):
            offset = k + 1
            if offset >= SEQ_LEN:
                break
            tgt_hidden = hidden[:, :-offset, :]
            tgt_logits = model.head(tgt_hidden)
            target_logits_list.append(tgt_logits)

    for k in range(min(N_PREDICT, len(target_logits_list))):
        offset = k + 1
        draft_logits = logits_list[k][:, :-offset, :].contiguous()
        target_logits = target_logits_list[k].contiguous()
        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        tv_dist = 0.5 * (draft_probs - target_probs).abs().sum(dim=-1).mean()
        tv_loss = tv_loss + tv_dist * pos_weights[k]

    # ── L_conf: confidence loss ──
    conf_loss = 0.0
    for k in range(min(N_PREDICT, len(target_logits_list))):
        offset = k + 1
        draft_logits = logits_list[k][:, :-offset, :].contiguous()
        target_logits = target_logits_list[k].contiguous()
        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        c_star = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
        c_pred = confidences[:, :-offset, k].contiguous()
        conf_loss = conf_loss + F.binary_cross_entropy(
            c_pred.clamp(1e-6, 1 - 1e-6),
            c_star.clamp(1e-6, 1 - 1e-6),
            reduction="mean",
        ) * pos_weights[k]

    total_loss = ce_loss + TV_WEIGHT * tv_loss + CONF_WEIGHT * conf_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(dspark.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0:
        torch.cuda.empty_cache()

    if step % 20 == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / (step + 1) * (STEPS - step - 1)
        print(f"  step {step:4d}/{STEPS} | loss={total_loss.item():.4f} "
              f"(ce={ce_loss.item():.3f} tv={tv_loss.item():.3f} "
              f"conf={conf_loss.item():.3f}) | {elapsed:.0f}s elapsed, ~{eta:.0f}s left")

# 5. Save
print(f"\nSaving DSparkHead to {SAVE_PATH}...")
state = {f"dspark.{k}": v.cpu() for k, v in dspark.state_dict().items()}
save_file(state, SAVE_PATH)
print(f"Saved! ({os.path.getsize(SAVE_PATH)/1e6:.1f} MB)")

total = time.time() - t0
print(f"\nDone in {total:.0f}s ({total/60:.1f} min)")
print(f"  DSparkHead: {n_dspark/1e6:.1f}M params")
print(f"  Final loss: {total_loss.item():.4f}")
print(f"  Checkpoint: {SAVE_PATH}")
