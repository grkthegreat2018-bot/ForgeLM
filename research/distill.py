"""Offline knowledge distillation: Qwen 2.5-0.5B (teacher) -> ForgeAI 360M (student).

Distillation loss (Hinton et al. 2015):
    L = alpha * T^2 * KL(softmax(s/T) || softmax(t/T)) + (1 - alpha) * CE(s, y)

The teacher's soft targets carry "dark knowledge" — relative probabilities
between near-correct tokens — that hard one-hot labels don't. A 360M student
distilled from a 500M teacher can match the teacher's quality, effectively
getting 500M-level reasoning in a 360M package.

The KL term is computed in chunks over the vocab dimension to avoid
materializing the full [B, T, V] teacher + student logits simultaneously
(2 * 2 * 1024 * 151936 * 4 bytes = 2.5 GB in float32 — too much alongside
both models).

Usage:
    python -m research.distill --config 360m_mla --steps 1000 --batch-size 2 \
        --temperature 4.0 --alpha 0.7 --checkpoint research/checkpoints/pretrained_llm.safetensors
"""
import argparse
import math
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    load_training_state,
    save_checkpoint,
    save_training_checkpoint,
    step_checkpoint_path,
)
from research.chunked_ce import chunked_linear_cross_entropy
from research.config import get_config
from research.model_loader import ModelLoader
from research.task_logger import task_scope
from research.training_utils import (
    BinaryDataset,
    configure_optimizer,
    get_lr,
    has_nan_params,
    init_ema,
    patch_triton_cache_for_windows,
    update_ema,
    vram_exceeded,
    write_status_json,
)


def chunked_kl_loss(student_logits, teacher_logits, targets, temperature, alpha,
                    top_k_kl=100):
    """Compute distillation loss without materializing full [B*T, V] float32 tensors.

    top_k_kl: if set, only compute KL over the top-K teacher tokens per position.
    This captures 99.9% of probability mass while reducing KL compute by ~1500x
    (151936 vocab -> 100 tokens). Set to None for full-vocab KL.

    Memory optimization: topk and gather operate on the original dtype (bf16),
    and only the small [N, K] slices are upcast to float32 for softmax/KL.
    This avoids creating [N, 151936] float32 copies (1.2 GB each).
    """
    N, V_s = student_logits.shape  # student vocab
    V_t = teacher_logits.shape[1]  # teacher vocab (may be >= V_s)

    if top_k_kl is not None and top_k_kl < V_t:
        # Top-K truncation: operate in original dtype (bf16) to save VRAM.
        t_top_vals, t_top_idx = teacher_logits.topk(top_k_kl, dim=-1)  # [N, K] in bf16
        # Gather student logits at the same top-K positions.
        if V_s < V_t:
            pad = torch.full((N, V_t - V_s), -1e4, device=student_logits.device, dtype=student_logits.dtype)
            s_full = torch.cat([student_logits, pad], dim=1)
        else:
            s_full = student_logits[:, :V_t]
        s_top = s_full.gather(1, t_top_idx)  # [N, K] in bf16

        # Only upcast the small [N, K] slices to float32 (K=100, not 151936).
        t_soft = F.softmax(t_top_vals.float() / temperature, dim=-1)  # [N, K] float32
        s_log_soft = F.log_softmax(s_top.float() / temperature, dim=-1)  # [N, K] float32
        kl = -(t_soft * s_log_soft).sum(dim=-1).mean()

        # Free the large bf16 tensors before CE (reduces peak VRAM).
        del t_top_vals, t_top_idx, s_full, s_top, t_soft, s_log_soft
    else:
        # Full-vocab KL (original slow path) — upcast only here where needed.
        if V_s < V_t:
            pad = torch.full((N, V_t - V_s), -1e4, device=student_logits.device, dtype=student_logits.dtype)
            student_logits_full = torch.cat([student_logits, pad], dim=1)
        else:
            student_logits_full = student_logits[:, :V_t]
        s_log_soft = F.log_softmax(student_logits_full.float() / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits.float() / temperature, dim=-1)
        kl = -(t_soft * s_log_soft).sum(dim=-1).mean()

    # Hard CE loss on student's original vocab (no padding needed).
    # CE handles bf16 inputs internally without extra copies.
    ce = F.cross_entropy(student_logits.float(), targets)

    # Combined loss with T^2 scaling on the KL term.
    loss = alpha * (temperature ** 2) * kl + (1.0 - alpha) * ce
    return loss, kl.item(), ce.item()


def student_hidden_states(student, idx):
    """Run the student transformer without the LM head, returning [B, T, H].

    Distillation only needs (a) CE — handled by fused chunked CE on the hidden
    states — and (b) student logits at the teacher's top-K positions — computed
    with a small gathered matmul. Skipping the full [B*T, 151936] head output
    removes the single largest activation in the training graph.
    """
    x = student.embed(idx)
    for block in student.blocks:
        x, _ = block(x)
    return student.ln_f(x)


def hidden_distill_loss(hidden_flat, head_weight, t_top_vals, t_top_idx, targets,
                        temperature, alpha, ce_chunk_size=512):
    """Top-K KL + fused CE directly from hidden states (no full-vocab logits).

    hidden_flat: [N, H] student hidden states (pre-head, requires grad)
    head_weight: [V_s, H] student LM head weight (tied with embeddings)
    t_top_vals / t_top_idx: [N, K] teacher top-K logits / token ids (bf16)
    targets: [N] ground-truth token ids
    """
    N, H = hidden_flat.shape
    K = t_top_idx.shape[1]

    # Student logits at teacher top-K positions, gathered in chunks so the
    # [c, K, H] weight gather stays small (~100 MB per chunk at K=100).
    s_top_chunks = []
    gather_chunk = 1024
    for start in range(0, N, gather_chunk):
        h_c = hidden_flat[start : start + gather_chunk]              # [c, H]
        w_c = head_weight[t_top_idx[start : start + gather_chunk]]   # [c, K, H]
        s_top_chunks.append(torch.bmm(h_c.unsqueeze(1), w_c.transpose(1, 2)).squeeze(1))
    s_top = torch.cat(s_top_chunks, dim=0)                           # [N, K]

    t_soft = F.softmax(t_top_vals.float() / temperature, dim=-1)
    s_log_soft = F.log_softmax(s_top.float() / temperature, dim=-1)
    kl = -(t_soft * s_log_soft).sum(dim=-1).mean()
    del s_top, t_soft, s_log_soft, s_top_chunks

    # Fused linear + CE: never materializes [N, V] logits.
    ce = chunked_linear_cross_entropy(hidden_flat, head_weight, targets, chunk_size=ce_chunk_size)

    loss = alpha * (temperature ** 2) * kl + (1.0 - alpha) * ce
    return loss, kl.item(), ce.item()


def main():
    p = argparse.ArgumentParser(description="Distill Qwen 2.5-0.5B into ForgeAI 360M")
    p.add_argument("--config", default="360m_mla")
    p.add_argument("--checkpoint", default="research/checkpoints/pretrained_llm.safetensors", help="Student checkpoint to start from (or 'scratch' for random init)")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B", help="HF teacher model name")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature (higher = softer targets)")
    p.add_argument("--alpha", type=float, default=0.7, help="Weight on KL loss (1-alpha on CE). 0.7 = 70% distillation, 30% hard labels.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save", default="research/checkpoints/distilled_llm.safetensors")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--top-k-kl", type=int, default=100,
                   help="Only compute KL over teacher's top-K tokens (default 100, captures 99.9% of mass). Set to 0 for full-vocab KL.")
    p.add_argument("--save-every", type=int, default=1000,
                   help="Save checkpoint every N steps (default 1000). Set to 0 to disable periodic saves.")
    p.add_argument("--resume", default=None,
                   help="Resume from a periodic checkpoint (e.g. distilled_v2.step5000.safetensors). "
                        "Loads model weights and continues from the step number in the filename.")
    p.add_argument("--keep-checkpoints", type=int, default=5,
                   help="Keep only the last N periodic checkpoints (auto-delete older). 0 = keep all. Default 5.")
    p.add_argument("--status-file", default="research/checkpoints/distill_status.json",
                   help="Write progress JSON to this file every 10 steps (for GUI monitor). Set to empty to disable.")
    p.add_argument("--vram-limit-gb", type=float, default=11.0,
                   help="Abort training if VRAM exceeds this (default 11.0 GB on 12 GB card).")
    p.add_argument("--heartbeat-file", default="research/checkpoints/distill_heartbeat.txt",
                   help="Write a heartbeat timestamp every 10 steps. If stale >60s, monitor can detect hang.")
    p.add_argument("--no-hidden-loss", action="store_true", default=False,
                   help="Disable the hidden-state loss path and materialize full-vocab student logits (legacy, slower).")
    p.add_argument("--teacher-compile", action="store_true", default=False,
                   help="torch.compile the teacher forward (~1.3x faster; needs patched triton-windows).")
    p.add_argument("--ema-decay", type=float, default=None,
                   help="Enable EMA weight averaging with this decay (e.g. 0.999). Final save uses EMA weights.")
    args = p.parse_args()
    if args.top_k_kl == 0:
        args.top_k_kl = None

    patch_triton_cache_for_windows()
    cfg = get_config(args.config)
    cfg.seq_len = args.seq_len
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    cfg.use_gradient_checkpointing = args.gradient_checkpointing
    device = torch.device("cuda")

    # --- Load teacher (frozen) ---
    print(f"Loading teacher: {args.teacher}...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    if args.teacher_compile:
        try:
            teacher = torch.compile(teacher)
            print("Teacher compiled with torch.compile.")
        except Exception as e:
            print(f"Teacher compile failed ({e}); using eager mode.")
    teacher_vocab = teacher.config.vocab_size
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"Teacher: {teacher_params:.1f}M params, vocab {teacher_vocab}, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # --- Load student (trainable) ---
    # Resume takes priority: load from the periodic checkpoint and parse step number.
    start_step = 0
    if args.resume:
        ckpt = args.resume
        # Parse step number from filename (e.g. "distilled_v2.step5000.safetensors" -> 5000).
        m = re.search(r'\.step(\d+)\.', args.resume)
        if m:
            start_step = int(m.group(1))
            print(f"Resuming from step {start_step}")
        else:
            print(f"Warning: could not parse step from {args.resume}, starting from step 0")
    else:
        ckpt = None if args.checkpoint == "scratch" else args.checkpoint
    print(f"Loading student from {ckpt or 'scratch'}...")
    student = ModelLoader.build_model(cfg, checkpoint_path=ckpt).to(device)
    student.train()
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student: {student_params:.1f}M params, vocab {cfg.vocab_size}")

    # --- Optimizer ---
    optimizer = configure_optimizer(student, args.lr, cfg.weight_decay, "bnb")

    # --- EMA weight averaging (optional quality boost) ---
    ema_state = None
    if args.ema_decay is not None:
        ema_state = init_ema(student)
        print(f"EMA enabled (decay={args.ema_decay}).")

    # Full-state resume: restore optimizer momentum + EMA + step counter.
    if args.resume:
        ts = load_training_state(args.resume, optimizer=optimizer)
        if ts.get("step"):
            start_step = ts["step"]
        if ema_state is not None and ts.get("ema") is not None:
            ema_state = {k: v.to(device) for k, v in ts["ema"].items()}
        print(f"  [RESUME] optimizer={'restored' if ts['has_optimizer'] else 'fresh'}, "
              f"ema={'restored' if ts.get('ema') is not None else 'fresh'}, step={start_step}")

    # --- Data ---
    train_ds = BinaryDataset("research/data/train.bin", args.seq_len, cfg.vocab_size)
    val_ds = BinaryDataset("research/data/val.bin", args.seq_len, cfg.vocab_size)

    print(f"\nDistillation config: T={args.temperature}, alpha={args.alpha}, steps={args.steps}")
    print(f"Loss = {args.alpha} * T^2 * KL(student||teacher) + {1-args.alpha} * CE(student, ground_truth)")
    if start_step > 0:
        print(f"  [RESUME] Continuing from step {start_step} | {start_step}/{args.steps} ({start_step/args.steps*100:.1f}%) already done | {args.steps - start_step} steps remaining\n")
    else:
        print(f"  Starting fresh: 0/{args.steps} (0.0%) | {args.steps} steps remaining\n")

    def eval_loss(model, tag):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(10):
                x, y = val_ds.get_batch(2, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
                    losses.append(loss.item())
        avg = sum(losses) / len(losses)
        print(f"  [{tag}] val loss: {avg:.4f} | ppl: {math.exp(avg):.2f}")
        model.train()
        return avg

    with task_scope("distill") as log:
        log.log(f"Teacher: {args.teacher} ({teacher_params:.1f}M), Student: 360M MLA, T={args.temperature}, alpha={args.alpha}")

        # Baseline eval.
        loss0 = eval_loss(student, "student before distillation")

        t0 = time.perf_counter()
        last_ckpt_step = start_step
        try:
            for step in range(start_step + 1, args.steps + 1):
                # --- VRAM watchdog: abort before OOM freezes the PC ---
                if vram_exceeded(args.vram_limit_gb, device):
                    vram_now = torch.cuda.memory_allocated() / 1e9
                    print(f"!! VRAM watchdog: {vram_now:.2f} GB > limit {args.vram_limit_gb} GB. Saving emergency checkpoint and aborting.", flush=True)
                    emergency = emergency_save(student, args.save, "emergency", step,
                                               optimizer=optimizer, ema_state=ema_state)
                    write_status_json(args.status_file, {
                        "step": step, "total_steps": args.steps, "running": False,
                        "aborted": True, "reason": f"VRAM {vram_now:.2f} GB exceeded limit",
                        "emergency_ckpt": emergency, "timestamp": time.time()})
                    break

                x, y = train_ds.get_batch(args.batch_size, device)

                # --- Teacher forward (inference_mode: faster than no_grad) ---
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    t_out = teacher(x)
                    teacher_logits = t_out.logits[:, :-1, :]  # [B, T-1, V_t] bf16

                # Top-K on teacher logits immediately, then free the full
                # [B*T, V_t] tensor before the student forward (saves ~600 MB).
                B, Tm1, V_t = teacher_logits.shape
                t_flat = teacher_logits.reshape(B * Tm1, V_t)
                if args.top_k_kl is not None and args.top_k_kl < V_t:
                    t_top_vals, t_top_idx = t_flat.topk(args.top_k_kl, dim=-1)  # [N, K]
                else:
                    t_top_vals = t_top_idx = None
                del t_flat, teacher_logits, t_out

                # --- Student forward + loss ---
                if not args.no_hidden_loss and t_top_idx is not None:
                    # Hidden-state path: no full-vocab student logits anywhere.
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        hidden = student_hidden_states(student, x)[:, :-1, :]  # [B, T-1, H]
                    hidden_flat = hidden.reshape(B * Tm1, hidden.size(-1))
                    y_flat = y[:, 1:].reshape(B * Tm1)
                    total_loss, total_kl, total_ce = hidden_distill_loss(
                        hidden_flat, student.head.weight, t_top_vals, t_top_idx,
                        y_flat, args.temperature, args.alpha,
                        ce_chunk_size=cfg.ce_chunk_size,
                    )
                    del hidden, hidden_flat, t_top_vals, t_top_idx, x, y, y_flat
                else:
                    # Legacy path: full-vocab student logits (needed when top-K is off).
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = student(x)
                        student_logits = out[0] if isinstance(out, tuple) else out
                        student_logits = student_logits[:, :-1, :].contiguous()
                    V_s = student_logits.shape[2]
                    s_flat = student_logits.reshape(B * Tm1, V_s)
                    y_flat = y[:, 1:].reshape(B * Tm1)
                    # Legacy loss expects full teacher logits; rerun the teacher
                    # forward cheaply in inference mode (they were freed above).
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        t_full = teacher(x).logits[:, :-1, :].reshape(B * Tm1, V_t)
                    total_loss, total_kl, total_ce = chunked_kl_loss(
                        s_flat, t_full, y_flat, args.temperature, args.alpha,
                        top_k_kl=args.top_k_kl,
                    )
                    del s_flat, t_full, student_logits, out, x, y, y_flat

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                # Single gradient clip (clipping twice distorts the update).
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)

                # NaN/Inf detection — catch weight corruption before it crashes CUDA.
                if step % 50 == 0 and has_nan_params(student):
                    print(f"!! NaN/Inf detected in weights at step {step}. Saving emergency checkpoint and aborting.", flush=True)
                    emergency = emergency_save(student, args.save, "nan", step,
                                               optimizer=optimizer, ema_state=ema_state)
                    write_status_json(args.status_file, {
                        "step": step, "total_steps": args.steps, "running": False,
                        "crashed": True, "error": "NaN/Inf in weights",
                        "emergency_ckpt": emergency, "timestamp": time.time()})
                    break
                lr = get_lr(step, args.steps, args.lr, args.lr * 0.1, max(1, args.steps // 10))
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.step()

                # EMA update: ema = decay * ema + (1 - decay) * raw
                if ema_state is not None:
                    update_ema(ema_state, student, args.ema_decay)

                # --- Periodic cache clear to prevent fragmentation ---
                if step % 200 == 0:
                    torch.cuda.empty_cache()

                if step % 10 == 0 or step == args.steps:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    steps_done_this_run = step - start_step
                    tok_s = steps_done_this_run * args.batch_size * args.seq_len / elapsed
                    vram = torch.cuda.max_memory_allocated() / 1e9
                    # True progress: percentage, remaining steps, ETA.
                    pct = step / args.steps * 100
                    remaining = args.steps - step
                    step_rate = steps_done_this_run / elapsed if elapsed > 0 else 0
                    eta_sec = remaining / step_rate if step_rate > 0 else 0
                    eta_h = int(eta_sec // 3600)
                    eta_m = int((eta_sec % 3600) // 60)
                    msg = (f"Step {step}/{args.steps} ({pct:.1f}%) | {remaining} left | "
                           f"ETA {eta_h}h{eta_m:02d}m | loss {total_loss.item():.4f} "
                           f"(KL {total_kl:.4f}, CE {total_ce:.4f}) | "
                           f"lr {lr:.2e} | {tok_s:.0f} tok/s | {vram:.2f} GB")
                    print(msg, flush=True)
                    log.log(msg)

                    # Write status JSON for GUI monitor (keys kept for monitor.py).
                    write_status_json(args.status_file, {
                        "step": step,
                        "total_steps": args.steps,
                        "pct": round(pct, 2),
                        "remaining": remaining,
                        "eta_hours": eta_h,
                        "eta_minutes": eta_m,
                        "eta_seconds": int(eta_sec),
                        "loss": round(total_loss.item(), 4),
                        "kl": round(total_kl, 4),
                        "ce": round(total_ce, 4),
                        "lr": lr,
                        "tok_s": round(tok_s, 0),
                        "vram_gb": round(vram, 2),
                        "elapsed_seconds": round(elapsed, 1),
                        "timestamp": time.time(),
                        "running": True,
                    })

                    # Heartbeat file (separate from status) for hang detection.
                    if args.heartbeat_file:
                        try:
                            with open(args.heartbeat_file, "w") as f:
                                f.write(str(time.time()))
                        except Exception:
                            pass

                # --- Periodic checkpoint (full training state, for pause/resume) ---
                if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
                    periodic_path = step_checkpoint_path(args.save, step)
                    save_training_checkpoint(student, periodic_path, optimizer=optimizer,
                                             ema_state=ema_state, step=step)
                    last_ckpt_step = step
                    print(f"  [checkpoint] Saved to {periodic_path} (resume with --resume {periodic_path})", flush=True)
                    cleanup_step_checkpoints(args.save, args.keep_checkpoints)

            # Final eval (only if we completed all steps).
            if step >= args.steps:
                loss1 = eval_loss(student, "student after distillation")

                # Full resumable state at args.save; EMA deliverable alongside if enabled.
                save_training_checkpoint(student, args.save, optimizer=optimizer,
                                         ema_state=ema_state, step=args.steps)
                print(f"\nSaved distilled model to {args.save}")
                if ema_state is not None:
                    ema_path = args.save.replace(".safetensors", ".ema.safetensors")
                    save_checkpoint({k: ema_state.get(k, v) for k, v in student.state_dict().items()}, ema_path)
                    print(f"Saved EMA weights to {ema_path}")
                print(f"Quality: {loss0:.4f} -> {loss1:.4f} ({(math.exp(loss1)/math.exp(loss0)-1)*100:+.1f}% ppl)")

                # Write final status (running=False) so GUI knows we're done.
                write_status_json(args.status_file, {
                    "step": args.steps, "total_steps": args.steps, "pct": 100.0,
                    "remaining": 0, "running": False, "finished": True,
                    "final_loss": round(loss1, 4), "final_ppl": round(math.exp(loss1), 2),
                    "timestamp": time.time()})

        except KeyboardInterrupt:
            print(f"\n!! Interrupted at step {step}. Saving emergency checkpoint...", flush=True)
            emergency = emergency_save(student, args.save, "interrupt", step,
                                       optimizer=optimizer, ema_state=ema_state)
            print(f"   Resume with: --resume {emergency}", flush=True)
            write_status_json(args.status_file, {
                "step": step, "total_steps": args.steps, "running": False,
                "interrupted": True, "resume_ckpt": emergency,
                "timestamp": time.time()})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n!! CRASH at step {step}: {e}", flush=True)
            # Save whatever we have from the last periodic checkpoint.
            if last_ckpt_step > 0:
                print(f"   Last periodic checkpoint: step {last_ckpt_step}", flush=True)
                print(f"   Find checkpoints with: Get-ChildItem research\\checkpoints\\distilled_v2.step*.safetensors", flush=True)
            else:
                print("   No periodic checkpoint was saved yet. Restart from scratch.", flush=True)
            write_status_json(args.status_file, {
                "step": step, "total_steps": args.steps, "running": False,
                "crashed": True, "error": str(e),
                "last_checkpoint_step": last_ckpt_step,
                "timestamp": time.time()})
            raise


if __name__ == "__main__":
    main()
