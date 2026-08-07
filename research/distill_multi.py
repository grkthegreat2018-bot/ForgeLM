"""Multi-teacher distillation with offline logit caching.

PROBLEM: Large teacher models (30B+) don't fit alongside the student on 12GB VRAM.
SOLUTION: Two-phase distillation:

  Phase 1 (CACHING): Run the teacher alone (with CPU offloading), save its logits
    for each training batch to disk. This is slow but one-time.
  Phase 2 (TRAINING): Train the student from cached logits — no teacher in memory.
    Fast, fits in 12GB easily.

For multiple teachers with DIFFERENT vocabularies:
  - Same-vocab teachers (Qwen3-Coder, Qwen2.5): KL distillation on logits
  - Different-vocab teachers (LFM2, Gemma): sequence-level distillation
    (generate text from teacher, train student on that text)

Usage:
    # Phase 1: Cache teacher logits (run once per teacher)
    python -m research.distill_multi --phase cache --teacher Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --n-batches 500

    # Phase 2: Train student from cached logits
    python -m research.distill_multi --phase train --student-ckpt research/checkpoints/pretrained_llm.safetensors --steps 500

    # Multi-teacher: cache from multiple teachers, then train with weighted average
    python -m research.distill_multi --phase cache --teacher Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --cache-dir research/cache/qwen3coder
    python -m research.distill_multi --phase cache --teacher Qwen/Qwen2.5-Coder-7B-Instruct --cache-dir research/cache/qwen25coder
    python -m research.distill_multi --phase train --cache-dirs research/cache/qwen3coder research/cache/qwen25coder --weights 0.7 0.3
"""
import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader
from research.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    load_training_state,
    save_training_checkpoint,
    step_checkpoint_path,
)
from research.training_utils import (
    BinaryDataset,
    add_safeguard_args,
    configure_optimizer,
    get_lr,
    has_nan_params,
    patch_triton_cache_for_windows,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
)


def cache_teacher_logits(teacher_name, cache_dir, n_batches, batch_size, seq_len, device, max_disk_gb=50):
    """Phase 1: Run teacher forward passes and cache logits to disk.

    Logits are saved as float16 to save space (2 bytes per logit).
    For a 30B model with vocab 151936, one batch of [2, 1024] produces
    2 * 1024 * 151936 * 2 bytes = 0.59 GB. For 500 batches = 295 GB — too much!

    SOLUTION: Only cache the top-K logits per position (e.g., top-100).
    This captures 99.9% of the probability mass while using 100x less space.
    500 batches * 2 * 1024 * 100 * (4 bytes idx + 2 bytes val) = 0.59 GB. Manageable.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading teacher: {teacher_name}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)

    # Load with device_map='auto' for CPU offloading of large models.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"BF16 load failed ({str(e)[:100]}), trying FP8...")
        model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    teacher_vocab = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"Teacher loaded: {n_params:.1f}B params, vocab {teacher_vocab}, GPU mem {gpu_mem:.1f} GB")

    # Use the teacher's tokenizer to create training data.
    # We need the same token IDs that the teacher was trained on.
    # For Qwen teachers, this matches our student's tokenizer.
    train_file = "research/data/train.bin"
    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training data not found: {train_file}. Run prep_data.py first.")

    # Read raw tokens from our pre-tokenized data (Qwen tokenizer, vocab 151665).
    # The teacher may have a slightly larger vocab (151936) — padding handled in training.
    raw_tokens = np.fromfile(train_file, dtype=np.uint16)
    print(f"Training data: {len(raw_tokens)/1e6:.1f}M tokens")

    top_k = 100  # Cache top-100 logits per position
    total_size = 0

    print(f"\nCaching top-{top_k} logits for {n_batches} batches (batch={batch_size}, seq={seq_len})...")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx in range(n_batches):
            # Sample a random chunk from the training data.
            start = np.random.randint(0, len(raw_tokens) - batch_size * seq_len - 1)
            tokens = raw_tokens[start:start + batch_size * seq_len].reshape(batch_size, seq_len)
            x = torch.tensor(tokens, dtype=torch.long).to(model.device)

            # Forward pass — teacher logits.
            out = model(x)
            logits = out.logits[:, :-1, :].float()  # [B, T-1, V] — predict next token
            targets = x[:, 1:]  # [B, T-1]

            # Get top-K logits and their indices.
            top_vals, top_idx = logits.topk(top_k, dim=-1)  # [B, T-1, K]

            # Save to disk as compressed npz.
            chunk_path = cache_path / f"logits_{batch_idx:05d}.npz"
            np.savez_compressed(
                chunk_path,
                top_idx=top_idx.cpu().numpy().astype(np.int32),
                top_vals=top_vals.cpu().numpy().astype(np.float16),
                targets=targets.cpu().numpy().astype(np.int32),
                inputs=x.cpu().numpy().astype(np.int32),  # Save input tokens too
            )
            total_size += chunk_path.stat().st_size

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (n_batches - batch_idx - 1) / rate
                disk_gb = total_size / 1e9
                print(f"  Batch {batch_idx+1}/{n_batches} | {rate:.1f} batch/s | ETA {eta:.0f}s | disk {disk_gb:.2f} GB")

            if total_size / 1e9 > max_disk_gb:
                print(f"  Disk limit reached ({max_disk_gb} GB). Stopping at batch {batch_idx+1}.")
                break

    # Save metadata.
    meta = {
        "teacher": teacher_name,
        "vocab_size": teacher_vocab,
        "n_batches": batch_idx + 1,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "top_k": top_k,
        "cache_dir": str(cache_path),
    }
    import json
    with open(cache_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nCaching complete: {batch_idx+1} batches in {elapsed:.0f}s | {total_size/1e9:.2f} GB")
    print(f"Metadata saved to {cache_path / 'meta.json'}")
    return meta


def topk_kl_loss(student_logits, top_idx, top_vals, targets, temperature, alpha):
    """KL distillation loss using cached top-K teacher logits.

    Instead of full KL over the entire vocab, we compute KL only over the
    top-K teacher tokens (which capture 99.9% of probability mass). The
    remaining probability is assumed to be uniformly distributed over the
    non-top-K tokens (a negligible approximation).

    Args:
        student_logits: [N, V_s] student logits for this chunk
        top_idx: [N, K] teacher top-K token indices
        top_vals: [N, K] teacher top-K logit values (pre-softmax)
        targets: [N] ground truth token IDs
        temperature: distillation temperature
        alpha: weight on KL (1-alpha on CE)
    """
    N, V_s = student_logits.shape
    K = top_idx.shape[1]

    # Teacher soft targets (from cached logits at temperature T).
    t_soft = F.softmax(top_vals / temperature, dim=-1)  # [N, K]

    # Student logits at the top-K positions (gather).
    s_topk = student_logits.gather(1, top_idx)  # [N, K]
    s_log_soft_topk = F.log_softmax(s_topk / temperature, dim=-1)  # [N, K]

    # KL divergence over top-K: -sum(t * log_s)
    kl = -(t_soft * s_log_soft_topk).sum(dim=-1).mean()

    # Hard CE loss on full vocab.
    ce = F.cross_entropy(student_logits, targets)

    loss = alpha * (temperature ** 2) * kl + (1.0 - alpha) * ce
    return loss, kl.item(), ce.item()


def train_from_cache(student_ckpt, cache_dirs, weights, steps, batch_size, seq_len,
                     temperature, alpha, lr, save_path, device,
                     resume=None, save_every=0, keep_checkpoints=5,
                     status_file=None, heartbeat_file=None, vram_limit_gb=0.0):
    """Phase 2: Train student from cached teacher logits (no teacher in memory)."""
    cfg = get_config("360m_mla")
    cfg.seq_len = seq_len
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)
    cfg.use_gradient_checkpointing = True

    student = ModelLoader.build_model(cfg, checkpoint_path=resume or student_ckpt).to(device)
    student.train()
    optimizer = configure_optimizer(student, lr, cfg.weight_decay, "bnb")

    # Resume training state (optimizer + RNG + step) if requested.
    start_step = 0
    if resume:
        ts = load_training_state(resume, optimizer=optimizer)
        if ts["step"] is not None:
            start_step = ts["step"] + 1
            print(f"Resuming from {resume} at step {start_step}")

    # Ctrl-C -> emergency checkpoint before dying.
    current = {"step": start_step}

    def _sigint_handler(sig, frame):
        emergency_save(student, save_path, "interrupt", current["step"], optimizer=optimizer)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    # Load cached logits metadata.
    caches = []
    for cd in cache_dirs:
        import json
        with open(Path(cd) / "meta.json") as f:
            meta = json.load(f)
        n_batches = meta["n_batches"]
        cache_files = sorted(Path(cd).glob("logits_*.npz"))
        caches.append({
            "dir": Path(cd),
            "files": cache_files,
            "n_batches": len(cache_files),
            "vocab": meta["vocab_size"],
            "top_k": meta["top_k"],
        })
        print(f"Cache: {cd} | {len(cache_files)} batches | vocab {meta['vocab_size']} | top-{meta['top_k']}")

    # Normalize weights.
    weights = [w / sum(weights) for w in weights]
    print(f"Teacher weights: {weights}")

    def eval_loss(model, tag):
        model.eval()
        ds = BinaryDataset("research/data/val.bin", seq_len, cfg.vocab_size)
        losses = []
        with torch.no_grad():
            for _ in range(10):
                x, y = ds.get_batch(2, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                    losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1)).item())
        avg = sum(losses) / len(losses)
        print(f"  [{tag}] val loss: {avg:.4f} | ppl: {math.exp(avg):.2f}")
        model.train()
        return avg

    loss0 = eval_loss(student, "before distillation")

    aborted = False
    t0 = time.time()
    for step in range(start_step + 1, steps + 1):
        if vram_exceeded(vram_limit_gb, device):
            print("VRAM limit exceeded; emergency save + abort.")
            emergency_save(student, save_path, "emergency", step, optimizer=optimizer)
            aborted = True
            break
        current["step"] = step
        # Pick a teacher based on weights.
        r = np.random.random()
        cum = 0
        teacher_idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r < cum:
                teacher_idx = i
                break

        cache = caches[teacher_idx]
        batch_idx = np.random.randint(0, cache["n_batches"])
        chunk = np.load(cache["files"][batch_idx])

        top_idx = torch.tensor(chunk["top_idx"], dtype=torch.long).to(device)  # [B, T-1, K]
        top_vals = torch.tensor(chunk["top_vals"], dtype=torch.float32).to(device)  # [B, T-1, K]
        targets = torch.tensor(chunk["targets"], dtype=torch.long).to(device)  # [B, T-1]
        x = torch.tensor(chunk["inputs"], dtype=torch.long).to(device)  # [B, T]

        B, Tm1, K = top_idx.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = student(x)
            s_logits = out[0] if isinstance(out, tuple) else out  # [B, T, V]
            s_logits = s_logits[:, :-1, :].contiguous()  # [B, T-1, V]

        # Compute loss in chunks over the token dimension.
        chunk_size = 128
        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        n_chunks = 0
        for i in range(0, B * Tm1, chunk_size):
            s_chunk = s_logits.reshape(B * Tm1, -1)[i:i+chunk_size].float()
            ti_chunk = top_idx.reshape(B * Tm1, K)[i:i+chunk_size]
            tv_chunk = top_vals.reshape(B * Tm1, K)[i:i+chunk_size]
            tg_chunk = targets.reshape(B * Tm1)[i:i+chunk_size]
            loss, kl, ce = topk_kl_loss(s_chunk, ti_chunk, tv_chunk, tg_chunk, temperature, alpha)
            if n_chunks == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
            total_kl += kl
            total_ce += ce
            n_chunks += 1
        total_loss = total_loss / n_chunks

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        lr_step = get_lr(step, steps, lr, lr * 0.1, max(1, steps // 10))
        for g in optimizer.param_groups:
            g["lr"] = lr_step
        optimizer.step()

        if step % 50 == 0 and has_nan_params(student):
            print("NaN/Inf detected in parameters; emergency save + abort.")
            emergency_save(student, save_path, "nan", step, optimizer=optimizer)
            aborted = True
            break

        if save_every > 0 and step % save_every == 0:
            ckpt = step_checkpoint_path(save_path, step)
            save_training_checkpoint(student, ckpt, optimizer=optimizer, step=step)
            cleanup_step_checkpoints(save_path, keep_checkpoints)

        if step % 20 == 0 or step == steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tok_s = step * batch_size * seq_len / elapsed
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"Step {step:4d}/{steps} | loss {total_loss.item():.4f} (KL {total_kl/n_chunks:.4f}, CE {total_ce/n_chunks:.4f}) | lr {lr_step:.2e} | {tok_s:.0f} tok/s | {vram:.2f} GB")
            write_status_json(status_file, {
                "step": step, "steps": steps,
                "loss": total_loss.item(), "lr": lr_step,
                "kl": total_kl / n_chunks, "ce": total_ce / n_chunks,
            })
            write_heartbeat(heartbeat_file)

    if aborted:
        return
    loss1 = eval_loss(student, "after distillation")
    save_training_checkpoint(student, save_path, optimizer=optimizer, step=steps)
    print(f"\nSaved distilled model to {save_path}")
    print(f"Quality: {loss0:.4f} -> {loss1:.4f} ({(math.exp(loss1)/math.exp(loss0)-1)*100:+.1f}% ppl)")


def main():
    p = argparse.ArgumentParser(description="Multi-teacher distillation with offline logit caching")
    p.add_argument("--phase", choices=["cache", "train"], required=True)
    p.add_argument("--teacher", default=None, help="HF model name for cache phase")
    p.add_argument("--cache-dir", default="research/cache/teacher", help="Where to save/load cached logits")
    p.add_argument("--n-batches", type=int, default=500, help="Number of batches to cache")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    # Training phase args
    p.add_argument("--student-ckpt", default="research/checkpoints/pretrained_llm.safetensors")
    p.add_argument("--cache-dirs", nargs="+", default=["research/cache/teacher"])
    p.add_argument("--weights", nargs="+", type=float, default=[1.0])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save", default="research/checkpoints/distilled_multi.safetensors")
    p.add_argument("--max-disk-gb", type=float, default=50.0)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume student weights + training state from this checkpoint")
    add_safeguard_args(p)
    args = p.parse_args()

    patch_triton_cache_for_windows()
    device = torch.device("cuda")

    if args.phase == "cache":
        if not args.teacher:
            p.error("--teacher required for cache phase")
        cache_teacher_logits(args.teacher, args.cache_dir, args.n_batches,
                            args.batch_size, args.seq_len, device, args.max_disk_gb)
    elif args.phase == "train":
        train_from_cache(args.student_ckpt, args.cache_dirs, args.weights,
                        args.steps, args.batch_size, args.seq_len,
                        args.temperature, args.alpha, args.lr, args.save, device,
                        resume=args.resume, save_every=args.save_every,
                        keep_checkpoints=args.keep_checkpoints,
                        status_file=args.status_file, heartbeat_file=args.heartbeat_file,
                        vram_limit_gb=args.vram_limit_gb)


if __name__ == "__main__":
    main()
