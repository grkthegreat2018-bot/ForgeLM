"""Train student model on synthetic data from any teacher (sequence-level distillation).

This is the simplest form of distillation: collect (prompt, response) pairs
from any source (API models, GLM-5.2 subagents, other LLMs), then train the
student with standard cross-entropy loss on the responses.

Unlike logit-level distillation (distill.py), this works with ANY teacher
regardless of vocabulary or architecture — you just need text outputs.

Usage:
    # Train on synthetic data from multiple JSONL files
    python -m research.distill_synthetic --student-ckpt research/checkpoints/distilled_llm.safetensors \
        --data research/data/synthetic_coding.jsonl research/data/synthetic_reasoning.jsonl \
        --steps 500 --lr 2e-4 --save research/checkpoints/synthetic_distilled.safetensors

    # With online learning safety (EMA + quality monitoring)
    python -m research.distill_synthetic --student-ckpt research/checkpoints/distilled_llm.safetensors \
        --data research/data/synthetic_*.jsonl --steps 1000 --ema-decay 0.999 --quality-check
"""
import argparse
import glob
import json
import math
import signal
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader
from research.training_utils import (
    BinaryDataset,
    configure_optimizer,
    evaluate_loss,
    init_ema,
    update_ema,
    restore_ema,
    get_lr,
    patch_triton_cache_for_windows,
    add_safeguard_args,
    has_nan_params,
    vram_exceeded,
    write_status_json,
    write_heartbeat,
)
from research.checkpoint_io import (
    save_checkpoint,
    save_training_checkpoint,
    load_training_state,
    step_checkpoint_path,
    cleanup_step_checkpoints,
    emergency_save,
)


def _tokenize_chat(tokenizer, messages, max_length):
    """Tokenize a chat-formatted conversation with proper assistant masking.

    Uses the tokenizer's apply_chat_template to render the full conversation,
    then computes the assistant_mask by re-rendering the prompt-only portion
    and marking everything after it as assistant tokens (training signal).

    This is the critical step for instruction tuning: without role markers,
    the model treats user+assistant text as one blob and learns to continue
    rather than answer. With Qwen chat template (<|im_start|>user ... <|im_end|>
    / <|im_start|>assistant ...), the model learns the boundary.

    Returns (ids, assistant_mask) or None if invalid.
    """
    # Render full conversation (user + assistant).
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Render prompt-only (user turn + generation prompt for assistant).
    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Assistant mask: True for tokens after the prompt prefix.
    n_prompt = len(prompt_ids)
    if n_prompt >= len(full_ids):
        return None  # No assistant tokens
    mask = [False] * n_prompt + [True] * (len(full_ids) - n_prompt)

    # Truncate.
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        mask = mask[:max_length]

    return full_ids, mask


def _tokenize_plain(tokenizer, prompt, completion, max_length):
    """Tokenize a plain (non-chat) prompt+completion pair with assistant mask."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        comp_ids = comp_ids + [tokenizer.eos_token_id]

    ids = prompt_ids + comp_ids
    mask = [False] * len(prompt_ids) + [True] * len(comp_ids)

    if len(ids) > max_length:
        ids = ids[:max_length]
        mask = mask[:max_length]

    return ids, mask


def load_synthetic_data(file_paths, tokenizer, max_length=1024, use_chat_template=True):
    """Load JSONL files with {prompt, completion} or {messages: [...]} pairs.

    When use_chat_template=True, applies the tokenizer's chat template
    (Qwen: <|im_start|>user ... <|im_end|> / <|im_start|>assistant ...) and
    masks prompt tokens so loss is only computed on assistant responses.

    Returns a list of (input_ids, assistant_mask) tuples where:
      - input_ids: token IDs for the full conversation
      - assistant_mask: True for assistant tokens (training signal), False for prompt
    """
    samples = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"  Warning: {fp} not found, skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Warning: {fp}:{line_num} invalid JSON: {str(e)[:60]}")
                    continue

                ids, mask = None, None

                # Chat-format input: {messages: [...]}
                if use_chat_template and "messages" in data:
                    msgs = data["messages"]
                    if not msgs or not any(m.get("role") == "assistant" for m in msgs):
                        continue
                    result = _tokenize_chat(tokenizer, msgs, max_length)
                    if result is not None:
                        ids, mask = result

                # Plain format: {prompt, completion}
                elif "prompt" in data and "completion" in data:
                    if use_chat_template:
                        # Wrap in chat format on the fly.
                        msgs = [
                            {"role": "user", "content": data["prompt"]},
                            {"role": "assistant", "content": data["completion"]},
                        ]
                        result = _tokenize_chat(tokenizer, msgs, max_length)
                        if result is not None:
                            ids, mask = result
                    else:
                        ids, mask = _tokenize_plain(
                            tokenizer, data["prompt"], data["completion"], max_length
                        )

                if ids is None or sum(mask) < 5:
                    continue

                samples.append((ids, mask))

        print(f"  Loaded {len(samples)} total samples (after {path.name})")

    return samples


def train_on_synthetic(student_ckpt, data_files, steps, lr, max_length, temperature,
                       alpha, save_path, ema_decay, quality_check, device,
                       mix_pretrain_ratio=0.0, pretrain_seq_len=512, use_chat_template=True,
                       resume=None, save_every=0, keep_checkpoints=5,
                       status_file=None, heartbeat_file=None, vram_limit_gb=0.0):
    """Train student on synthetic data with optional EMA + quality monitoring.

    mix_pretrain_ratio: fraction of each batch that's pretraining data (0.0-1.0).
        E.g., 0.5 means half the batch is synthetic, half is pretraining data.
        This prevents catastrophic forgetting when the synthetic dataset is small.
    resume: checkpoint path to resume full training state from (takes priority
        over student_ckpt for the weight load).
    """
    cfg = get_config("360m_mla")
    cfg.seq_len = max_length
    cfg.max_seq_len = max(cfg.max_seq_len, max_length)
    cfg.use_gradient_checkpointing = True

    # Load student (resume checkpoint takes priority over --student-ckpt).
    if resume:
        student_ckpt = resume
    print(f"Loading student from {student_ckpt}...")
    student = ModelLoader.build_model(cfg, checkpoint_path=student_ckpt).to(device)
    student.train()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Load synthetic data.
    print(f"\nLoading synthetic data from {len(data_files)} files...")
    print(f"  Chat template: {'ON (Qwen <|im_start|> format)' if use_chat_template else 'OFF (plain prompt+completion)'}")
    samples = load_synthetic_data(data_files, tokenizer, max_length, use_chat_template=use_chat_template)
    print(f"Total samples: {len(samples)}")

    if len(samples) == 0:
        print("No valid samples found. Exiting.")
        return

    # Load pretraining data for mixing (optional).
    pretrain_ds = None
    if mix_pretrain_ratio > 0:
        try:
            pretrain_ds = BinaryDataset("research/data/train.bin", pretrain_seq_len, cfg.vocab_size)
            print(f"Pretrain mixing enabled: {mix_pretrain_ratio*100:.0f}% of each batch is pretraining data")
            print(f"  Pretrain dataset: research/data/train.bin (seq_len={pretrain_seq_len})")
        except Exception as e:
            print(f"  Warning: could not load pretrain data for mixing: {e}")
            print(f"  Continuing without pretrain mixing.")
            mix_pretrain_ratio = 0.0

    # Optimizer.
    optimizer = configure_optimizer(student, lr, cfg.weight_decay, "bnb")

    # EMA (optional safety net).
    ema_state = None
    if ema_decay is not None:
        ema_state = init_ema(student)
        print(f"EMA enabled (decay={ema_decay})")

    # Resume full training state (optimizer + EMA + step) if requested.
    start_step = 0
    if resume:
        ts = load_training_state(resume, optimizer=optimizer)
        if ts["step"] is not None:
            start_step = ts["step"]
        if ts["ema"] is not None:
            if ema_state is None:
                ema_state = init_ema(student)
            for name, tensor in ts["ema"].items():
                if name in ema_state:
                    ema_state[name].copy_(tensor.to(device))
        print(f"Resumed from {resume}: start_step={start_step}, "
              f"optimizer={'loaded' if ts['has_optimizer'] else 'fresh'}, "
              f"EMA={'loaded' if ts['ema'] is not None else 'fresh'}")

    # Val dataset for quality monitoring.
    val_ds = BinaryDataset("research/data/val.bin", 512, cfg.vocab_size)

    def eval_loss(model, tag):
        return evaluate_loss(model, val_ds, device, n_batches=10, batch_size=2, tag=tag)

    def update_ema_local():
        update_ema(ema_state, student, ema_decay)

    loss0 = eval_loss(student, "before synthetic distillation")

    # Training loop.
    batch_size = 2
    t0 = time.perf_counter()
    best_val = loss0

    # Ctrl-C emergency save: dump full training state, then re-raise.
    current = {"step": start_step}

    def _sigint_handler(signum, frame):
        emergency_save(student, save_path, "interrupt", current["step"],
                       optimizer=optimizer, ema_state=ema_state)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    for step in range(start_step + 1, steps + 1):
        current["step"] = step

        # VRAM watchdog: abort with an emergency checkpoint before OOM.
        if vram_exceeded(vram_limit_gb, device):
            print(f"  [SAFETY] VRAM limit {vram_limit_gb} GB exceeded at step {step}")
            emergency_save(student, save_path, "emergency", step,
                           optimizer=optimizer, ema_state=ema_state)
            break

        # Determine how many synthetic vs pretrain samples in this batch.
        n_pretrain = int(round(batch_size * mix_pretrain_ratio)) if pretrain_ds else 0
        n_synthetic = batch_size - n_pretrain

        # Sample synthetic examples.
        batch_ids = []
        batch_masks = []
        max_len = 0
        if n_synthetic > 0:
            batch_indices = np.random.choice(len(samples), n_synthetic, replace=len(samples) < n_synthetic)
            for idx in batch_indices:
                ids, mask = samples[idx]
                batch_ids.append(ids)
                batch_masks.append(mask)
                max_len = max(max_len, len(ids))

        # Add pretraining data samples (full sequence, all tokens are training signal).
        if n_pretrain > 0 and pretrain_ds is not None:
            for _ in range(n_pretrain):
                px, py = pretrain_ds.get_batch(1, device)
                ids = px[0].cpu().tolist()
                batch_ids.append(ids)
                batch_masks.append([True] * len(ids))  # All tokens are "completion"
                max_len = max(max_len, len(ids))

        # Build x and y on CPU first, then move to GPU once (avoids per-element kernel launches).
        pad_id = tokenizer.pad_token_id or 0
        x_np = np.full((batch_size, max_len), pad_id, dtype=np.int64)
        y_np = np.full((batch_size, max_len), -100, dtype=np.int64)  # -100 = ignore in CE
        for i, (ids, mask) in enumerate(zip(batch_ids, batch_masks)):
            n = len(ids)
            x_np[i, :n] = ids
            # Target = input shifted by 1, only where assistant_mask is True.
            mask_arr = np.array(mask, dtype=bool)
            # y[i, j] = ids[j+1] where mask[j+1] is True, for j in [0, n-2]
            if n > 1:
                targets = np.array(ids[1:], dtype=np.int64)        # ids[1..n-1]
                target_mask = mask_arr[1:]                          # mask[1..n-1]
                y_np[i, :n-1] = np.where(target_mask, targets, -100)

        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)

        # Forward + loss.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = student(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        lr_step = get_lr(step, steps, lr, lr * 0.1, max(1, steps // 10))
        for g in optimizer.param_groups:
            g["lr"] = lr_step
        optimizer.step()
        update_ema_local()

        # NaN/Inf watchdog: NaNs propagate silently and corrupt the model.
        if step % 50 == 0 and has_nan_params(student):
            print(f"  [SAFETY] NaN/Inf detected in parameters at step {step}")
            emergency_save(student, save_path, "nan", step,
                           optimizer=optimizer, ema_state=ema_state)
            break

        # Periodic resumable checkpoint (weights + optimizer/EMA/RNG sidecars).
        if save_every > 0 and step % save_every == 0:
            ckpt = step_checkpoint_path(save_path, step)
            save_training_checkpoint(student, ckpt, optimizer=optimizer,
                                     ema_state=ema_state, step=step)
            cleanup_step_checkpoints(save_path, keep_checkpoints)

        if step == start_step + 1:
            torch.cuda.synchronize()
            print(f"Step {step} ok | loss {loss.item():.4f} | {(time.perf_counter()-t0):.1f}s")
        if step % 20 == 0 or step == steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"Step {step:4d}/{steps} | loss {loss.item():.4f} | lr {lr_step:.2e} | {elapsed:.0f}s")
            if status_file:
                write_status_json(status_file, {"step": step, "steps": steps,
                                                "loss": loss.item(), "lr": lr_step,
                                                "elapsed_s": round(elapsed)})
            if heartbeat_file:
                write_heartbeat(heartbeat_file)

        # Quality check.
        if quality_check and step % 100 == 0:
            val = eval_loss(student, f"step {step}")
            if val > best_val + 0.5 and ema_state is not None:
                print("  [SAFETY] Quality regressed, restoring EMA weights")
                restore_ema(ema_state, student)
            elif val < best_val:
                best_val = val

    # Final save: resumable training state to save_path; EMA-merged
    # deliverable goes to a distinct ".ema" path so resume state stays intact.
    save_training_checkpoint(student, save_path, optimizer=optimizer,
                             ema_state=ema_state, step=current["step"])
    print(f"\nSaved resumable training checkpoint to {save_path} (weights + optimizer/EMA/RNG sidecars)")
    if ema_state is not None:
        stem, dot, ext = save_path.rpartition(".")
        ema_path = f"{stem}.ema.{ext}" if dot else save_path + ".ema"
        state = {k: ema_state.get(k, v) for k, v in student.state_dict().items()}
        save_checkpoint(state, ema_path)
        print(f"Saved EMA-merged deliverable to {ema_path}")
    else:
        print(f"Saved model to {save_path}")

    loss1 = eval_loss(student, "after synthetic distillation")
    print(f"\nQuality: {loss0:.4f} -> {loss1:.4f} ({(math.exp(loss1)/math.exp(loss0)-1)*100:+.1f}% ppl)")


def main():
    p = argparse.ArgumentParser(description="Sequence-level distillation from synthetic data")
    p.add_argument("--student-ckpt", default="research/checkpoints/distilled_llm.safetensors")
    p.add_argument("--data", nargs="+", required=True, help="JSONL file(s) with {prompt, completion} pairs")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0, help="Not used in sequence-level (kept for compat)")
    p.add_argument("--alpha", type=float, default=1.0, help="Not used in sequence-level (kept for compat)")
    p.add_argument("--save", default="research/checkpoints/synthetic_distilled.safetensors")
    p.add_argument("--ema-decay", type=float, default=None, help="Enable EMA with this decay (e.g. 0.999)")
    p.add_argument("--quality-check", action="store_true", help="Monitor val loss and restore EMA if regression")
    p.add_argument("--mix-pretrain", type=float, default=0.0,
                   help="Fraction of each batch from pretraining data (0.0-1.0). Prevents catastrophic forgetting.")
    p.add_argument("--pretrain-seq-len", type=int, default=512,
                   help="Sequence length for pretraining data mixing (shorter = faster)")
    p.add_argument("--use-chat-template", action="store_true", default=True,
                   help="Apply Qwen chat template (<|im_start|>user/assistant) with assistant-only loss masking (default: ON)")
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false",
                   help="Disable chat template (use plain prompt+completion concatenation)")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path to resume full training state from")
    add_safeguard_args(p)
    args = p.parse_args()

    # Expand globs.
    data_files = []
    for pattern in args.data:
        matched = glob.glob(pattern)
        if matched:
            data_files.extend(matched)
        elif Path(pattern).exists():
            data_files.append(pattern)
        else:
            print(f"Warning: no files match '{pattern}'")

    if not data_files:
        print("No data files found. Exiting.")
        return

    print(f"Data files: {data_files}")

    patch_triton_cache_for_windows()
    device = torch.device("cuda")

    train_on_synthetic(
        args.student_ckpt, data_files, args.steps, args.lr,
        args.max_length, args.temperature, args.alpha, args.save,
        args.ema_decay, args.quality_check, device,
        mix_pretrain_ratio=args.mix_pretrain,
        pretrain_seq_len=args.pretrain_seq_len,
        use_chat_template=args.use_chat_template,
        resume=args.resume,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        status_file=args.status_file,
        heartbeat_file=args.heartbeat_file,
        vram_limit_gb=args.vram_limit_gb,
    )


if __name__ == "__main__":
    main()
