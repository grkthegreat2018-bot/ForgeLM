"""Supervised Fine-Tuning (SFT) for tool-calling and conversational English."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Sequence, Value
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

load_dotenv()
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from research.config import get_config
from research.model_loader import ModelLoader
from research.task_logger import task_scope
from research.training_utils import (
    configure_optimizer,
    get_lr,
    patch_triton_cache_for_windows,
    add_safeguard_args,
    has_nan_params,
    vram_exceeded,
    write_status_json,
    write_heartbeat,
)
from research.checkpoint_io import (
    save_training_checkpoint,
    load_training_state,
    step_checkpoint_path,
    cleanup_step_checkpoints,
    emergency_save,
)


def format_chat(tokenizer, sample, max_length):
    """Convert a conversation or prompt/completion sample into token ids with assistant-only labels."""
    # Prompt/completion format (used for local tool-calling dataset).
    if "prompt" in sample and "completion" in sample:
        prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(sample["completion"], add_special_tokens=False)["input_ids"]
        ids = prompt_ids + comp_ids
        mask = [False] * len(prompt_ids) + [True] * len(comp_ids)
        if len(ids) > max_length + 1:
            ids = ids[: max_length + 1]
            mask = mask[: max_length + 1]
        return {"input_ids": ids, "assistant_mask": mask}

    messages = sample.get("messages") or sample.get("conversations") or []
    if not messages:
        # Plain-text fallback.
        text = sample.get("text", "")
        encoded = tokenizer(text, max_length=max_length + 1, truncation=True, add_special_tokens=False)
        ids = encoded["input_ids"]
        return {"input_ids": ids, "assistant_mask": [True] * len(ids)}

    # Most chat templates (including Qwen's) do not include the {% generation %}
    # marker, so assistant-token masking is not available. In that case we train
    # on the full conversation, which is still a valid SFT signal.
    chat_template = tokenizer.chat_template or ""
    supports_mask = "{% generation %}" in chat_template or "{%generation%}" in chat_template

    try:
        kwargs = dict(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            max_length=max_length + 1,
            truncation=True,
            add_special_tokens=False,
        )
        if supports_mask:
            kwargs["return_assistant_tokens_mask"] = True
        encoded = tokenizer.apply_chat_template(**kwargs)
        ids = list(encoded["input_ids"])
        mask = encoded.get("assistant_tokens_mask") if supports_mask else None
        if mask is None:
            mask = [True] * len(ids)
        else:
            mask = list(mask)
    except Exception:
        # Fallback: tokenize without assistant mask.
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length + 1,
            truncation=True,
            add_special_tokens=False,
        )
        ids = list(ids)
        mask = [True] * len(ids)

    # Avoid empty sequences that break dataset type inference.
    if not ids:
        ids = [tokenizer.eos_token_id]
        mask = [True]

    return {"input_ids": ids, "assistant_mask": mask}


def collate_fn(batch, pad_token_id, max_length):
    """Pad to max length and produce (x, labels) for causal LM training."""
    max_len = min(max(len(b["input_ids"]) for b in batch), max_length + 1)

    input_ids = []
    labels = []
    for b in batch:
        ids = list(b["input_ids"])[:max_length + 1]
        mask = list(b["assistant_mask"])[:max_length + 1]
        pad = [pad_token_id] * (max_len - len(ids))
        ids_padded = ids + pad
        mask_padded = mask + [False] * len(pad)

        # Shift by one for causal targets.
        x = ids_padded[:-1]
        y = ids_padded[1:]
        label_mask = mask_padded[1:]
        label = [yt if m else -100 for yt, m in zip(y, label_mask)]

        input_ids.append(x)
        labels.append(label)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def prepare_sft_dataset(ds, tokenizer, max_length):
    """Map to chat format, drop empty rows, and cast columns to aligned types."""
    if ds is None:
        return None
    ds = ds.map(lambda s: format_chat(tokenizer, s, max_length), remove_columns=ds.column_names)
    ds = ds.filter(lambda s: len(s["input_ids"]) > 1)
    ds = ds.cast_column("input_ids", Sequence(Value("int64")))
    ds = ds.cast_column("assistant_mask", Sequence(Value("bool")))
    return ds


def main():
    parser = argparse.ArgumentParser(description="SFT-align a ForgeAI research model.")
    parser.add_argument("--config", type=str, default="360m_mla")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint to load")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--optimizer", type=str, default="bnb", choices=["adamw", "bnb", "lion"])
    parser.add_argument("--checkpoint-dir", type=str, default="research/checkpoints")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile. On Windows this needs triton-windows 3.4.0 and a small cache.py path-length patch; see AGENTS.md.")
    parser.add_argument("--no-compile", action="store_false", dest="compile", help="Disable torch.compile")
    parser.add_argument("--tool-only", action="store_true", default=False, help="Use only the local tool_sft.jsonl dataset for focused tool-call training.")
    parser.add_argument("--chunked-ce", action="store_true", default=False, help="Use chunked fused linear cross-entropy (saves ~2.8 GB VRAM at batch 2). Not compatible with --compile.")
    parser.add_argument("--ce-chunk-size", type=int, default=256, help="Token chunk size for chunked CE")
    parser.add_argument("--lisa", action="store_true", default=False, help="Use LISA: train only top-k most important layers per step (importance = grad L2 norm).")
    parser.add_argument("--lisa-k", type=int, default=8, help="Number of top layers to update per LISA step.")
    parser.add_argument("--lisa-interval", type=int, default=20, help="Recompute layer importances every N steps.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume full training state from")
    add_safeguard_args(parser)
    args = parser.parse_args()

    cfg = get_config(args.config)
    cfg.max_seq_len = args.max_seq_length
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.batch_size = args.batch_size
    cfg.max_steps = args.max_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.max_lr = args.lr
    cfg.min_lr = args.min_lr
    if args.chunked_ce:
        cfg.use_chunked_ce = True
        cfg.ce_chunk_size = args.ce_chunk_size
        if args.compile:
            print("WARNING: --chunked-ce + --compile may cause graph breaks. Disabling compile.")
            args.compile = False

    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    with task_scope("sft") as task:
        task.log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

        task.log("Building model...")
        # --resume takes priority: weights are loaded from the resume checkpoint.
        checkpoint = args.resume or args.checkpoint or str(Path(cfg.checkpoint_dir) / "pretrained_llm.pt")
        raw_model = ModelLoader.build_model(cfg, checkpoint_path=checkpoint, compile=False).to(device)

        train_model = raw_model
        if args.compile:
            try:
                patch_triton_cache_for_windows()
                train_model = torch.compile(raw_model)
            except Exception as e:
                task.log(f"torch.compile failed ({e}); continuing eager.")

        optimizer = configure_optimizer(train_model, args.lr, cfg.weight_decay, args.optimizer)

        # Resume full training state (optimizer + step) if requested.
        start_step = 0
        if args.resume:
            ts = load_training_state(args.resume, optimizer=optimizer)
            if ts["step"] is not None:
                start_step = ts["step"]
            task.log(f"Resumed from {args.resume}: start_step={start_step}, "
                     f"optimizer={'loaded' if ts['has_optimizer'] else 'fresh'}")

        out_path = Path(cfg.checkpoint_dir) / "sft_llm.safetensors"

        task.log("Loading SFT datasets...")
        ds_tools = None
        ds_hermes = None
        ds_chat = None
        ds_local = None

        if not args.tool_only:
            try:
                ds_tools = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=False)
                ds_tools = ds_tools.select(range(min(args.max_samples, len(ds_tools))))
                ds_tools = prepare_sft_dataset(ds_tools, tokenizer, args.max_seq_length)
            except Exception as e:
                task.log(f"Could not load glaive-function-calling-v2: {e}")

            try:
                ds_hermes = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=False)
                ds_hermes = ds_hermes.select(range(min(args.max_samples // 2, len(ds_hermes))))
                ds_hermes = prepare_sft_dataset(ds_hermes, tokenizer, args.max_seq_length)
            except Exception as e:
                task.log(f"Could not load hermes-function-calling-v1: {e}")

            try:
                ds_chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:{}]".format(args.max_samples))
                ds_chat = prepare_sft_dataset(ds_chat, tokenizer, args.max_seq_length)
            except Exception as e:
                task.log(f"Could not load ultrachat_200k: {e}")

        try:
            local_path = Path(__file__).resolve().parent / "data" / "tool_sft.jsonl"
            ds_local = load_dataset("json", data_files=str(local_path), split="train", streaming=False)
            ds_local = ds_local.select(range(min(args.max_samples, len(ds_local))))
            ds_local = prepare_sft_dataset(ds_local, tokenizer, args.max_seq_length)
        except Exception as e:
            task.log(f"Could not load local tool_sft.jsonl: {e}")

        ds_list = [d for d in [ds_tools, ds_hermes, ds_chat, ds_local] if d is not None]
        if not ds_list:
            raise RuntimeError("No SFT datasets could be loaded.")
        sft_dataset = concatenate_datasets(ds_list).shuffle(seed=42)

        loader = DataLoader(
            sft_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id or tokenizer.eos_token_id, args.max_seq_length),
        )

        train_model.train()
        global_step = start_step
        accumulation_counter = 0
        t0 = time.time()

        # Ctrl-C emergency save: dump full training state, then re-raise.
        current = {"step": global_step}

        def _sigint_handler(signum, frame):
            emergency_save(raw_model, str(out_path), "interrupt", current["step"],
                           optimizer=optimizer)
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _sigint_handler)

        # LISA: collect transformer blocks (nn.Module children that look like a layer).
        lisa_blocks = None
        lisa_active = set()  # indices of layers currently unfrozen
        if args.lisa:
            # Identify layer modules: try .blocks, .layers, .h, else top-level children with params.
            blocks = []
            for attr in ("blocks", "layers", "h", "transformer_blocks"):
                if hasattr(raw_model, attr):
                    blocks = list(getattr(raw_model, attr))
                    break
            if not blocks:
                # Fallback: top-level children that have parameters.
                blocks = [m for m in raw_model.children() if any(p.requires_grad for p in m.parameters())]
            lisa_blocks = blocks
            task.log(f"LISA: detected {len(blocks)} candidate layers, training top-{args.lisa_k} per interval {args.lisa_interval}.")
            # Initial importance probe: keep all unfrozen for the first interval,
            # then the periodic recompute will select the top-k.
            # (No freeze here — the first _lisa_recompute() call happens at step lisa_interval.)

        def _lisa_recompute():
            """Compute weight-norm importance per block, unfreeze top-k.

            Uses weight L2 norms (no gradient probe pass needed — avoids the
            memory spike of the old approach). Layer importance correlates
            strongly with weight magnitude per the LISA paper.
            """
            importances = []
            for i, blk in enumerate(lisa_blocks):
                w2 = sum((p.data.norm() ** 2).item() for p in blk.parameters())
                importances.append((w2 ** 0.5, i))
            importances.sort(reverse=True)
            new_active = {i for _, i in importances[: args.lisa_k]}
            # Freeze old active that are no longer selected.
            for i in lisa_active - new_active:
                for p_ in lisa_blocks[i].parameters():
                    p_.requires_grad_(False)
            # Unfreeze newly selected.
            for i in new_active - lisa_active:
                for p_ in lisa_blocks[i].parameters():
                    p_.requires_grad_(True)
            lisa_active.clear()
            lisa_active.update(new_active)
            task.log(f"LISA step {global_step}: active layers {sorted(new_active)}")

        for batch in loader:
            global_step += 1
            if global_step > args.max_steps:
                break
            current["step"] = global_step

            # VRAM watchdog: abort with an emergency checkpoint before OOM.
            if vram_exceeded(args.vram_limit_gb, device):
                task.log(f"[SAFETY] VRAM limit {args.vram_limit_gb} GB exceeded at step {global_step}")
                emergency_save(raw_model, str(out_path), "emergency", global_step,
                               optimizer=optimizer)
                break

            lr = get_lr(global_step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = train_model(x, targets=y)
                loss = loss / args.gradient_accumulation

            loss.backward()
            accumulation_counter += 1

            if accumulation_counter % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # LISA: recompute importances and freeze/unfreeze layers.
                # Use weight-norm-based importance (no extra forward/backward pass).
                # This avoids the memory spike of the old probe-pass approach.
                if args.lisa and lisa_blocks is not None and global_step % args.lisa_interval == 0:
                    _lisa_recompute()

            # NaN/Inf watchdog: NaNs propagate silently and corrupt the model.
            if global_step % 50 == 0 and has_nan_params(train_model):
                task.log(f"[SAFETY] NaN/Inf detected in parameters at step {global_step}")
                emergency_save(raw_model, str(out_path), "nan", global_step,
                               optimizer=optimizer)
                break

            # Periodic resumable checkpoint (weights + optimizer/RNG sidecars).
            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt = step_checkpoint_path(str(out_path), global_step)
                save_training_checkpoint(raw_model, ckpt, optimizer=optimizer, step=global_step)
                cleanup_step_checkpoints(str(out_path), args.keep_checkpoints)

            if global_step % 20 == 0:
                torch.cuda.synchronize() if device.type == "cuda" else None
                t1 = time.time()
                tok_per_sec = (20 * args.batch_size * args.max_seq_length) / (t1 - t0)
                log_line = f"Step {global_step:4d}/{args.max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
                print(log_line)
                task.log(log_line)
                task.update(progress={"step": global_step, "max_steps": args.max_steps}, metrics={"loss": round(loss.item(), 4)})
                if args.status_file:
                    write_status_json(args.status_file, {"step": global_step, "max_steps": args.max_steps,
                                                         "loss": loss.item(), "lr": lr,
                                                         "tok_per_sec": round(tok_per_sec)})
                if args.heartbeat_file:
                    write_heartbeat(args.heartbeat_file)
                t0 = time.time()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_training_checkpoint(raw_model, str(out_path), optimizer=optimizer, step=global_step)
        task.log(f"SFT complete. Saved {out_path} (resumable: weights + optimizer/RNG sidecars)")
        print(f"SFT complete. Saved {out_path}")


if __name__ == "__main__":
    main()
