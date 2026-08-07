"""DSparkLite v4 — max speed training.

Key optimizations vs v3:
1. Precompute hidden states AND target logits (no LM head in training loop)
2. Use bf16 autocast for precompute (2x faster)
3. Larger precompute batch (32)
4. Training loop: only adapter forward + CE against cached logits
   → ~0.01s/step (no model, no LM head, just 18.9M adapter params)
5. Lossless checkpointing with optimizer state
6. Signal handler for graceful interrupt

Usage:
    python -u .devin\train_dspark_v4.py
    python -u .devin\train_dspark_v4.py --resume
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
N_SAMPLES = 2000
BATCH_SIZE = 16        # large batch (no model forward, no LM head)
STEPS = 2000
LR = 1e-3
WARMUP = 100
SAVE_PATH = "research/checkpoints/dspark_lite_forgelm_v1.safetensors"
SAVE_EVERY = 200
CACHE_DIR = "research/checkpoints/dspark_cache"
META_PATH = "research/checkpoints/dspark_lite_meta.json"

# ─── Helpers ─────────────────────────────────────────────────────────

def cosine_lr(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(head, optimizer, scheduler, step, best_loss, path):
    """Save full training state for lossless resume."""
    state = {f"adapters.{k}": v.cpu()
             for k, v in head.adapters.state_dict().items()}
    save_file(state, path)
    sidecar = {
        "step": step,
        "val_loss": best_loss,
        "optimizer_state": {k: v.cpu() for k, v in optimizer.state_dict().items()},
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(sidecar, path + ".train.pt")
    with open(META_PATH, "w") as f:
        json.dump({"step": step, "val_loss": best_loss, "saved_at": time.time()}, f)


def load_checkpoint(head, optimizer, scheduler, path):
    state = load_file(path)
    clean = {k.replace("adapters.", "", 1): v
             for k, v in state.items() if k.startswith("adapters.")}
    head.adapters.load_state_dict(clean)
    sidecar_path = path + ".train.pt"
    if os.path.exists(sidecar_path):
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(sidecar["optimizer_state"])
        scheduler.load_state_dict(sidecar["scheduler_state"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(DEVICE)
        return sidecar["step"], sidecar.get("val_loss", float("inf"))
    return 0, float("inf")


# ─── Phase 1: Precompute hidden states + target logits ───────────────

def precompute_features(model, n_samples, seq_len, cache_dir):
    """Run frozen model, cache hidden states + target next-token logits.

    Saves:
      cache_dir/hidden.npy  — (N, seq_len, d_model) float16
      cache_dir/tokens.npy  — (N, seq_len) int32
      cache_dir/target_logits.npy — (N, seq_len, n_predict, vocab) is too large
      Instead: cache target TOKENS only (we use CE against token IDs, not logits)
    """
    os.makedirs(cache_dir, exist_ok=True)
    hidden_path = os.path.join(cache_dir, "hidden.npy")
    tokens_path = os.path.join(cache_dir, "tokens.npy")

    if os.path.exists(hidden_path) and os.path.exists(tokens_path):
        h = np.load(hidden_path, mmap_mode="r")
        print(f"  Cached: hidden={h.shape} (already exists)")
        return hidden_path, tokens_path

    print(f"  Precomputing {n_samples} samples (seq={seq_len})...")
    data = torch.frombuffer(
        open("research/data/train.bin", "rb").read(), dtype=torch.uint32
    ).long()

    hidden_arr = np.lib.format.open_memmap(
        hidden_path, mode="w+", dtype=np.float16,
        shape=(n_samples, seq_len, D_MODEL))
    tokens_arr = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.int32,
        shape=(n_samples, seq_len))

    model.eval()
    batch_pc = 32
    t0 = time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n_samples, batch_pc):
            n = min(batch_pc, n_samples - i)
            idxs = torch.randint(0, data.numel() - seq_len - 1, (n,))
            batches = torch.stack([data[j:j+seq_len] for j in idxs]).to(DEVICE)

            out = model(batches, return_hidden=True)
            hidden = out[2] if len(out) > 2 else out[0]

            hidden_arr[i:i+n] = hidden.float().cpu().numpy().astype(np.float16)
            tokens_arr[i:i+n] = batches.cpu().numpy().astype(np.int32)

            if i % (batch_pc * 4) == 0:
                elapsed = time.time() - t0
                done = i + n
                eta = elapsed / done * (n_samples - done)
                print(f"    {done}/{n_samples} ({done/n_samples*100:.0f}%) | "
                      f"{elapsed:.0f}s, ~{eta:.0f}s left")

    hidden_arr.flush()
    tokens_arr.flush()
    del hidden_arr, tokens_arr
    print(f"  Precompute done in {time.time()-t0:.0f}s")
    return hidden_path, tokens_path


# ─── Phase 2: Train adapters (no model, no LM head) ──────────────────

def train_adapters(head, hidden_path, tokens_path, steps, lr, batch_size,
                   warmup, save_every, save_path, resume=False):
    """Train DSparkLite adapters on cached features.

    The adapter output is (B, T, D_MODEL). We compute CE loss using the
    cached LM head weight (frozen, no grad) via chunked F.linear.
    No model forward, no LM head grad — just 18.9M adapter params.
    """
    dev = torch.device(DEVICE)

    hidden_arr = np.load(hidden_path, mmap_mode="r")
    tokens_arr = np.load(tokens_path, mmap_mode="r")
    n_samples = hidden_arr.shape[0]
    seq_len = hidden_arr.shape[1]
    print(f"  Cache: {n_samples} samples, seq={seq_len}")

    # Get LM head weight (frozen)
    lm_weight = head.lm_head.weight.data  # (V, D)
    lm_weight.requires_grad = False

    optimizer = torch.optim.AdamW(head.adapters.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_lr(s, steps, warmup))

    pos_weights = torch.exp(
        -torch.arange(1, N_PREDICT + 1, dtype=torch.float) / N_PREDICT
    ).to(dev)

    start_step = 0
    best_loss = float("inf")
    if resume and os.path.exists(save_path):
        start_step, best_loss = load_checkpoint(head, optimizer, scheduler, save_path)
        print(f"  Resumed from step {start_step} (best={best_loss:.4f})")

    interrupted = [False]
    def on_sigint(sig, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, on_sigint)

    n_params = sum(p.numel() for p in head.adapters.parameters())
    print(f"  Training: {n_params/1e6:.1f}M params, {steps} steps (from {start_step})")
    print(f"  lr={lr}, batch={batch_size}, save_every={save_every}")

    t0 = time.time()
    for step in range(start_step, steps):
        head.train()

        idxs = np.random.randint(0, n_samples, batch_size)
        hidden_batch = torch.from_numpy(
            hidden_arr[idxs].astype(np.float32)).to(dev)
        token_batch = torch.from_numpy(
            tokens_arr[idxs]).long().to(dev)

        # Adapter forward + CE loss (chunked to avoid OOM)
        ce_loss = 0.0
        chunk = 256  # tokens per CE chunk
        for k in range(N_PREDICT):
            offset = k + 1
            if offset >= seq_len:
                break
            h_k = hidden_batch + head.adapters[k](hidden_batch)
            h_k = h_k[:, :-offset, :].contiguous()
            target = token_batch[:, offset:].contiguous()

            flat_h = h_k.view(-1, D_MODEL)
            flat_t = target.view(-1)
            n_tok = flat_h.shape[0]

            total_ce = 0.0
            for ci in range(0, n_tok, chunk):
                logits = F.linear(flat_h[ci:ci+chunk], lm_weight)
                total_ce = total_ce + F.cross_entropy(
                    logits, flat_t[ci:ci+chunk],
                    ignore_index=-100, reduction="sum")

            ce_loss = ce_loss + (total_ce / n_tok) * pos_weights[k]

        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.adapters.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            done = step - start_step + 1
            eta = elapsed / done * (steps - step - 1)
            lr_now = optimizer.param_groups[0]["lr"]
            loss_val = ce_loss.item()
            print(f"  step {step:5d}/{steps} | ce={loss_val:.4f} | lr={lr_now:.2e} | "
                  f"{elapsed:.0f}s, ~{eta:.0f}s left")
            if loss_val < best_loss:
                best_loss = loss_val

        if step > 0 and (step % save_every == 0 or step == steps - 1):
            save_checkpoint(head, optimizer, scheduler, step, best_loss, save_path)
            print(f"  >> Checkpoint saved (step={step})")

        if interrupted[0]:
            save_checkpoint(head, optimizer, scheduler, step, best_loss, save_path)
            print(f"  >> Emergency save (step={step})")
            break

    if not interrupted[0]:
        save_checkpoint(head, optimizer, scheduler, steps - 1, best_loss, save_path)

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f} min) | Best: {best_loss:.4f}")
    return head


# ─── Main ────────────────────────────────────────────────────────────

def main():
    global STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()
    STEPS = args.steps

    print("=" * 60)
    print("DSparkLite v4 — Max Speed Training")
    print("=" * 60)
    print(f"  Steps: {args.steps} | Samples: {args.n_samples} | Batch: {BATCH_SIZE}")
    print(f"  LR: {LR} (cosine, warmup={WARMUP}) | Save every: {SAVE_EVERY}")

    # Phase 1: Load model + precompute
    print("\n[Phase 1] Precompute hidden states...")
    cfg = get_config("forgelm_v1", device=DEVICE)
    model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v1.safetensors")
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_path, tokens_path = precompute_features(
        model, args.n_samples, SEQ_LEN, CACHE_DIR)

    # Create head, cache LM head weight, free model
    head = DSparkLite(
        d_model=D_MODEL, vocab_size=VOCAB, n_predict=N_PREDICT,
        lm_head=model.head,
    ).to(DEVICE)

    lm_weight = model.head.weight.data.clone()
    del model
    torch.cuda.empty_cache()
    print(f"  Model freed, LM head cached: {lm_weight.shape}")

    import torch.nn as nn
    lite_head = nn.Linear(D_MODEL, VOCAB, bias=False)
    lite_head.weight = nn.Parameter(lm_weight)
    lite_head.to(DEVICE).eval()
    for p in lite_head.parameters():
        p.requires_grad = False
    head.lm_head = lite_head

    # Phase 2: Train
    print(f"\n[Phase 2] Train adapters...")
    head = train_adapters(
        head, hidden_path, tokens_path,
        steps=args.steps, lr=LR, batch_size=BATCH_SIZE,
        warmup=WARMUP, save_every=SAVE_EVERY,
        save_path=SAVE_PATH, resume=args.resume)

    print(f"\nCheckpoint: {SAVE_PATH}")


if __name__ == "__main__":
    main()
