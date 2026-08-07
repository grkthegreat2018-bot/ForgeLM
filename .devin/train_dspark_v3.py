"""DSparkLite v3 — optimized training with hidden state precomputation.

Two-phase approach:
  Phase 1: Precompute hidden states from frozen model (one-time, ~2 min)
  Phase 2: Train adapters on cached features (very fast, no model forward)

This eliminates the 1.5B param model forward pass from the training loop,
turning 6.7s/step into ~0.05s/step (100x+ speedup).

Features:
  - Hidden state caching to disk (mmap, no RAM pressure)
  - Periodic checkpointing with full optimizer state (interrupt-safe)
  - Cosine LR schedule with warmup
  - bf16 model forward, fp32 adapter training

Usage:
    python -u .devin/train_dspark_v3.py
    python -u .devin/train_dspark_v3.py --resume  # resume from last checkpoint
"""
import sys, os, time, math, json, signal, torch, argparse
sys.path.insert(0, '.')

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch.nn.functional as F
import numpy as np
from research.config import get_config
from research.model_loader import ModelLoader
from research.dspark_lite import DSparkLite
from safetensors.torch import save_file, load_file

DEVICE = "cuda"
D_MODEL = 1536
VOCAB = 151936
N_PREDICT = 4
SEQ_LEN = 128
N_SAMPLES = 1000       # precompute this many sequences
BATCH_SIZE = 8         # training batch size (no model forward, can be large)
STEPS = 1000
LR = 5e-4
WARMUP = 50
SAVE_PATH = "research/checkpoints/dspark_lite_forgelm_v1.safetensors"
SAVE_EVERY = 100       # checkpoint every N steps
CACHE_DIR = "research/checkpoints/dspark_cache"
META_PATH = "research/checkpoints/dspark_lite_meta.json"

# ─── Helpers ─────────────────────────────────────────────────────────

def cosine_lr(step):
    if step < WARMUP:
        return step / max(WARMUP, 1)
    progress = (step - WARMUP) / max(STEPS - WARMUP, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(head, optimizer, scheduler, step, val_loss, path):
    """Save full training state for lossless resume."""
    state = {
        f"adapters.{k}": v.cpu()
        for k, v in head.adapters.state_dict().items()
    }
    save_file(state, path)
    # Save optimizer + scheduler + step to sidecar
    sidecar = {
        "step": step,
        "val_loss": val_loss,
        "optimizer_state": {k: v.cpu() for k, v in optimizer.state_dict().items()},
        "scheduler_state": scheduler.state_dict(),
        "lr": optimizer.param_groups[0]["lr"],
    }
    torch.save(sidecar, path + ".train.pt")
    meta = {"step": step, "val_loss": val_loss, "saved_at": time.time()}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)


def load_checkpoint(head, optimizer, scheduler, path):
    """Load full training state for resume."""
    state = load_file(path)
    clean = {k.replace("adapters.", "", 1): v
             for k, v in state.items() if k.startswith("adapters.")}
    head.adapters.load_state_dict(clean)
    sidecar_path = path + ".train.pt"
    if os.path.exists(sidecar_path):
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(sidecar["optimizer_state"])
        scheduler.load_state_dict(sidecar["scheduler_state"])
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        return sidecar["step"], sidecar.get("val_loss", float("inf"))
    return 0, float("inf")


# ─── Phase 1: Precompute hidden states ───────────────────────────────

def precompute_hidden_states(model, n_samples, seq_len, cache_dir):
    """Run frozen model over random sequences, cache hidden states to disk.

    Saves:
      cache_dir/hidden.npy  — (N, seq_len, d_model) float16
      cache_dir/tokens.npy  — (N, seq_len) int32
    """
    os.makedirs(cache_dir, exist_ok=True)
    hidden_path = os.path.join(cache_dir, "hidden.npy")
    tokens_path = os.path.join(cache_dir, "tokens.npy")

    if os.path.exists(hidden_path) and os.path.exists(tokens_path):
        hidden = np.load(hidden_path, mmap_mode="r")
        print(f"  Cached hidden states: {hidden.shape} (already exists)")
        return hidden_path, tokens_path

    print(f"  Precomputing {n_samples} hidden states (seq={seq_len})...")
    data = torch.frombuffer(
        open("research/data/train.bin", "rb").read(), dtype=torch.uint32
    ).long()  # CPU

    # Allocate mmap arrays
    hidden_arr = np.lib.format.open_memmap(
        hidden_path, mode="w+", dtype=np.float16,
        shape=(n_samples, seq_len, D_MODEL))
    tokens_arr = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.int32,
        shape=(n_samples, seq_len))

    model.eval()
    batch_precompute = 16  # larger batch for precomputation
    t0 = time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n_samples, batch_precompute):
            n = min(batch_precompute, n_samples - i)
            # Sample random sequences
            idxs = torch.randint(0, data.numel() - seq_len - 1, (n,))
            batches = torch.stack([data[i:i+seq_len] for i in idxs]).to(DEVICE)

            out = model(batches, return_hidden=True)
            hidden = out[2] if len(out) > 2 else out[0]  # (n, seq, d_model)

            hidden_arr[i:i+n] = hidden.float().cpu().numpy().astype(np.float16)
            tokens_arr[i:i+n] = batches.cpu().numpy().astype(np.int32)

            if i % (batch_precompute * 5) == 0:
                elapsed = time.time() - t0
                pct = (i + n) / n_samples * 100
                eta = elapsed / (i + n) * (n_samples - i - n)
                print(f"    {i+n}/{n_samples} ({pct:.0f}%) | {elapsed:.0f}s, ~{eta:.0f}s left")

    hidden_arr.flush()
    tokens_arr.flush()
    del hidden_arr, tokens_arr

    elapsed = time.time() - t0
    print(f"  Precompute done in {elapsed:.0f}s")
    return hidden_path, tokens_path


# ─── Phase 2: Train adapters on cached features ──────────────────────

def train_on_cache(head, hidden_path, tokens_path, steps, lr, batch_size,
                   warmup, save_every, save_path, resume=False):
    """Train DSparkLite adapters on precomputed hidden states.

    No model forward pass — just adapter forward + backward.
    """
    dev = torch.device(DEVICE)

    # Load cached data via mmap (no RAM copy)
    hidden_arr = np.load(hidden_path, mmap_mode="r")
    tokens_arr = np.load(tokens_path, mmap_mode="r")
    n_samples = hidden_arr.shape[0]
    print(f"  Cache: {n_samples} samples, hidden={hidden_arr.shape}")

    # Optimizer — only adapters
    optimizer = torch.optim.AdamW(head.adapters.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    # Position weights
    pos_weights = torch.exp(
        -torch.arange(1, N_PREDICT + 1, dtype=torch.float) / N_PREDICT
    ).to(dev)

    # Resume
    start_step = 0
    best_loss = float("inf")
    if resume and os.path.exists(save_path):
        start_step, best_loss = load_checkpoint(head, optimizer, scheduler, save_path)
        print(f"  Resumed from step {start_step} (best_loss={best_loss:.4f})")

    # Signal handler for graceful interrupt
    interrupted = [False]
    def on_sigint(sig, frame):
        print("\n  [Interrupt] Saving checkpoint...")
        interrupted[0] = True
    signal.signal(signal.SIGINT, on_sigint)

    n_params = sum(p.numel() for p in head.adapters.parameters())
    print(f"  Training: {n_params/1e6:.1f}M params, {steps} steps (from {start_step})")
    print(f"  lr={lr}, batch={batch_size}, save_every={save_every}")

    t0 = time.time()
    for step in range(start_step, steps):
        head.train()

        # Sample random batch from cache
        idxs = np.random.randint(0, n_samples, batch_size)
        hidden_batch = torch.from_numpy(
            hidden_arr[idxs].astype(np.float32)).to(dev)  # (B, T, D)
        token_batch = torch.from_numpy(
            tokens_arr[idxs]).long().to(dev)  # (B, T)

        # Adapter forward — only compute adapter outputs (no LM head)
        # Each adapter: h_k = hidden + adapter(hidden)  → (B, T, D)
        # Then use fused linear CE: loss = CE(lm_head_weight @ h_k, target)
        # This avoids materializing (B, T, 151936) logits
        lm_weight = head.lm_head.weight  # (V, D), frozen, no grad

        ce_loss = 0.0
        for k in range(N_PREDICT):
            offset = k + 1
            if offset >= hidden_batch.shape[1]:
                break
            # Adapter forward (tiny: 18.9M params)
            h_k = hidden_batch + head.adapters[k](hidden_batch)  # (B, T, D)
            # Slice to valid positions
            h_k = h_k[:, :-offset, :].contiguous()  # (B, T-offset, D)
            target = token_batch[:, offset:].contiguous()  # (B, T-offset)

            # Fused linear cross-entropy (avoids full logits materialization)
            # Manual: logits = h_k @ lm_weight.T, then CE
            # Use chunked CE to save VRAM
            flat_h = h_k.view(-1, D_MODEL)  # (N, D)
            flat_t = target.view(-1)  # (N,)

            # Chunked: process in groups of 512 tokens
            chunk = 512
            total_ce = 0.0
            n_chunks = 0
            for ci in range(0, flat_h.shape[0], chunk):
                h_chunk = flat_h[ci:ci+chunk]  # (C, D)
                t_chunk = flat_t[ci:ci+chunk]  # (C,)
                logits_chunk = F.linear(h_chunk, lm_weight)  # (C, V)
                total_ce = total_ce + F.cross_entropy(
                    logits_chunk, t_chunk, ignore_index=-100,
                    reduction="sum")
                n_chunks += t_chunk.numel()

            ce_loss = ce_loss + (total_ce / max(n_chunks, 1)) * pos_weights[k]

        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.adapters.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            n_done = step - start_step + 1
            eta = elapsed / n_done * (steps - step - 1)
            lr_now = optimizer.param_groups[0]["lr"]
            loss_val = ce_loss.item()
            print(f"  step {step:4d}/{steps} | ce={loss_val:.4f} | lr={lr_now:.2e} | "
                  f"{elapsed:.0f}s, ~{eta:.0f}s left")
            if loss_val < best_loss:
                best_loss = loss_val

        # Periodic checkpoint
        if step > 0 and (step % save_every == 0 or step == steps - 1):
            save_checkpoint(head, optimizer, scheduler, step, best_loss, save_path)
            print(f"  >> Checkpoint saved (step={step}, best={best_loss:.4f})")

        if interrupted[0]:
            save_checkpoint(head, optimizer, scheduler, step, best_loss, save_path)
            print(f"  >> Emergency checkpoint saved (step={step})")
            break

    if not interrupted[0]:
        save_checkpoint(head, optimizer, scheduler, steps - 1, best_loss, save_path)

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f} min)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoint: {save_path}")
    return head


# ─── Main ────────────────────────────────────────────────────────────

def main():
    global STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    # Update global STEPS for cosine_lr function
    STEPS = args.steps

    print("=" * 60)
    print("DSparkLite v3 — Optimized Training")
    print("=" * 60)
    print(f"  Steps: {args.steps} | Samples: {args.n_samples} | Batch: {BATCH_SIZE}")
    print(f"  LR: {LR} (cosine, warmup={WARMUP}) | Save every: {SAVE_EVERY}")

    # Phase 1: Load model + precompute hidden states
    print("\n[Phase 1] Precompute hidden states...")
    cfg = get_config("forgelm_v1", device=DEVICE)
    model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v1.safetensors")
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_path, tokens_path = precompute_hidden_states(
        model, args.n_samples, SEQ_LEN, CACHE_DIR)

    # Create DSparkLite with shared LM head
    head = DSparkLite(
        d_model=D_MODEL, vocab_size=VOCAB, n_predict=N_PREDICT,
        lm_head=model.head,
    ).to(DEVICE)

    # Free model from VRAM — only need LM head for adapter forward
    # Keep LM head reference but move rest to CPU
    lm_head_weight = model.head.weight.data.clone()
    del model
    torch.cuda.empty_cache()
    print(f"  Model freed from VRAM, LM head weight cached ({lm_head_weight.shape})")

    # Create a lightweight LM head for the DSparkLite (just weight, no full model)
    import torch.nn as nn
    lite_lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
    lite_lm_head.weight = nn.Parameter(lm_head_weight)
    lite_lm_head.to(DEVICE)
    lite_lm_head.eval()
    for p in lite_lm_head.parameters():
        p.requires_grad = False
    head.lm_head = lite_lm_head

    # Phase 2: Train on cached features
    print(f"\n[Phase 2] Train adapters on cached features...")
    head = train_on_cache(
        head, hidden_path, tokens_path,
        steps=args.steps, lr=LR, batch_size=BATCH_SIZE,
        warmup=WARMUP, save_every=SAVE_EVERY,
        save_path=SAVE_PATH, resume=args.resume,
    )

    print(f"\nCheckpoint: {SAVE_PATH}")
    print(f"Meta: {META_PATH}")


if __name__ == "__main__":
    main()
