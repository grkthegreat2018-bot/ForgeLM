"""Distillation run — every dozen epochs, filter bloat & outdated knowledge.

Inspired by BIRD (2026, bootstrapped on-policy self-distillation with a
brevity instruction) and CheckpointKD (intermediate checkpoints as teachers).

The problem this solves: over many fine-tune epochs the model accumulates
bloat — every edit, every wrong version, every verbose exploration trace
gets baked into the weights. The distill run rebuilds a clean model that
remembers ONLY what's curated in the database, not every edit/ver.

Process:
  1. Load the current best epoch as the teacher.
  2. Build a clean, deduplicated knowledge corpus from the DB:
       - discoveries (confirmed findings)
       - theories with status 'supported' or 'refuted' (with conclusion)
       - scripts that ran successfully (returncode=0)
       - research summaries
     Drop anything abandoned / failed / empty -> that's the "bloat removed".
  3. For each knowledge item, the teacher generates a concise canonical
     response under a brevity instruction (BIRD's insight: brevity-instructed
     generation produces cleaner targets than the raw DB text).
  4. Fine-tune a FRESH copy of the base ForgeLM V10 checkpoint on these
     (prompt -> concise response) pairs. The fresh start means no prior
     bloat survives — only DB-curated knowledge is reinjected.
  5. The distilled model becomes a new epoch (kind='distill') and goes
     through the same best-vs-loser comparison as fine-tune epochs.

Reuses: model_loader, checkpoint_io, training_utils, quality_eval.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from research.paths import DATA_DIR, LFM25_CHECKPOINT
from research.self_play.discovery.discovery_db import DiscoveryDB


_EPOCHS_DIR = DATA_DIR / "discovery" / "epochs"

_BREVITY = ("Answer in one or two clear sentences. Be precise and concise. "
            "Do not hedge or repeat the question.")


@dataclass
class DistillConfig:
    max_new_tokens: int = 96
    temperature: float = 0.3
    # SFT hyperparams (lighter than finetune — fresh model, small clean set)
    epochs: int = 1
    batch_size: int = 1           # batch 1 + grad_accum 4 = effective batch 4
    grad_accum: int = 4
    max_lr: float = 3e-5
    min_lr: float = 3e-6
    warmup_steps: int = 10
    max_seq_len: int = 384
    weight_decay: float = 0.01
    vram_limit_gb: float = 11.0
    # VRAM-saving options (match finetune.py / sft_train.py)
    optimizer: str = "bnb"        # 8-bit AdamW (saves ~7GB)
    use_lora: bool = True         # LoRA: train ~1M params instead of 1.17B
    lora_r: int = 16
    lora_alpha: int = 32
    grad_checkpoint: bool = True  # activation checkpointing
    use_chunked_ce: bool = True   # avoids [B,T,V] logits materialization


def _build_corpus(db: DiscoveryDB) -> tuple[list[tuple[str, str]], int]:
    """Return (clean_corpus, bloat_removed_count).

    clean_corpus is (prompt, source_text) pairs; the teacher rewrites
    source_text concisely. bloat_removed is the count of items dropped
    (abandoned theories, failed scripts, empty rows).
    """
    corpus: list[tuple[str, str]] = []
    bloat = 0

    for r in db.query("SELECT summary FROM discoveries WHERE summary IS NOT NULL AND length(summary) > 10"):
        corpus.append(("Record a confirmed discovery.", r["summary"]))
    for r in db.query("SELECT statement, status, notes FROM theories WHERE status IN ('supported','refuted') AND length(statement) > 10"):
        verdict = f" ({r['status']})" if r["status"] == "refuted" else ""
        text = r["statement"] + verdict + (f" {r['notes']}" if r["notes"] else "")
        corpus.append(("State a verified hypothesis.", text))
    # abandoned theories = bloat
    bloat += db.query("SELECT COUNT(*) AS n FROM theories WHERE status='abandoned'")[0]["n"]
    for r in db.query("SELECT code FROM scripts WHERE returncode=0 AND length(code) > 20 AND length(code) < 800 LIMIT 150"):
        corpus.append(("Write useful Python code.", r["code"]))
    # failed scripts = bloat
    bloat += db.query("SELECT COUNT(*) AS n FROM scripts WHERE returncode != 0")[0]["n"]
    for r in db.query("SELECT query, summary FROM research WHERE summary IS NOT NULL AND length(summary) > 10 LIMIT 80"):
        corpus.append((f"Summarize findings on: {r['query']}", r["summary"]))

    # Deduplicate by source_text (bloat = repeats).
    seen: set[str] = set()
    deduped = []
    for p, s in corpus:
        if s in seen:
            bloat += 1
            continue
        seen.add(s)
        deduped.append((p, s))
    return deduped, bloat


def _teacher_rewrite(teacher, tokenizer, prompt: str, source: str,
                     cfg: DistillConfig, device: str) -> str:
    """Teacher generates a concise canonical response from the source text."""
    from research.model_loader import ModelLoader
    full = (f"{prompt}\nSource: {source}\n{_BREVITY}\nResponse:")
    out = ModelLoader.generate_text(teacher, tokenizer, full,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature)
    return out.strip() or source  # fall back to source if teacher empty


def distill_run(db: DiscoveryDB, teacher_checkpoint: str | None = None,
                config: DistillConfig | None = None,
                device: str = "cuda") -> str:
    """Run a distillation pass and return the new distilled checkpoint path.

    Args:
        db: discovery database (corpus source + epoch bookkeeping).
        teacher_checkpoint: path to the current best epoch (teacher). If None,
            uses the base ForgeLM V10 checkpoint.
        config: DistillConfig.
    Returns:
        Path to the new distilled epoch checkpoint.
    """
    from research.config import get_config
    from research.model_loader import ModelLoader, load_default_model
    from research.tokenizer_cache import get_tokenizer
    from research.training.training_utils import (
        configure_optimizer, get_lr, compute_ce_loss, has_nan_params,
        vram_exceeded)
    from research.checkpoint_io import save_checkpoint
    from research.self_play.discovery.finetune import _collate

    cfg = config or DistillConfig()

    # 1. Load teacher (current best or base).
    teacher, tokenizer = load_default_model("forgelm_v2_light")
    if teacher_checkpoint:
        from research.checkpoint_io import load_checkpoint
        sd = load_checkpoint(teacher_checkpoint)
        teacher.load_state_dict({k: v.to(teacher.device) for k, v in sd.items()})
    teacher.to(device).eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # 2. Build clean corpus.
    corpus, bloat = _build_corpus(db)
    if len(corpus) < 8:
        raise ValueError(f"DB too small to distill: only {len(corpus)} clean items")

    # 3. Teacher rewrites each item concisely (on-policy targets).
    pairs = []
    for prompt, source in corpus:
        target = _teacher_rewrite(teacher, tokenizer, prompt, source, cfg, device)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            c_ids = c_ids + [tokenizer.eos_token_id]
        ids = (p_ids + c_ids)[:cfg.max_seq_len]
        if len(ids) < 8:
            continue
        pairs.append({"ids": ids, "comp_start": min(len(p_ids), len(ids) - 1)})
    del teacher
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 4. Fresh student from the BASE checkpoint (no prior bloat).
    model_cfg = get_config("forgelm_v2_light", device=device)
    model_cfg.use_gradient_checkpointing = cfg.grad_checkpoint
    model_cfg.use_chunked_ce = cfg.use_chunked_ce
    model_cfg.ce_chunk_size = 128
    student = ModelLoader.build_model_fast(
        model_cfg, checkpoint_path=str(LFM25_CHECKPOINT),
        moe_top_k=0, dtype=torch.bfloat16)
    student.to(device).train()

    # ── LoRA (PEFT) ──
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                            "w_gate", "w_up", "w_down"],
            lora_dropout=0.0,
            bias="none",
        )
        student = get_peft_model(student, lora_cfg)
        student.print_trainable_parameters()
    else:
        for param in student.parameters():
            param.requires_grad_(True)

    opt = configure_optimizer(student, cfg.max_lr, cfg.weight_decay,
                              optimizer_name=cfg.optimizer)
    steps_per_epoch = max(1, len(pairs) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    step = 0
    for _ep in range(cfg.epochs):
        for sp in range(steps_per_epoch):
            if vram_exceeded(cfg.vram_limit_gb, device):
                raise RuntimeError("VRAM exceeded during distill")
            opt.zero_grad(set_to_none=True)
            for g in range(cfg.grad_accum):
                idx = (sp * cfg.grad_accum + g) % len(pairs)
                batch = [pairs[idx]] if cfg.batch_size == 1 else \
                    [pairs[(sp * cfg.grad_accum + j) % len(pairs)]
                     for j in range(cfg.batch_size)]
                ids, tgt, _ = _collate(batch, pad_id, device)
                # Chunked CE fast path
                if cfg.use_chunked_ce:
                    shift_tgt = tgt[:, 1:].contiguous()
                    pad_col = torch.full((shift_tgt.size(0), 1), -100,
                                         dtype=shift_tgt.dtype, device=device)
                    shift_tgt = torch.cat([shift_tgt, pad_col], dim=1)
                    out = student(ids, targets=shift_tgt)
                    loss = out[1] if isinstance(out, tuple) else out
                    if loss is None:
                        out = student(ids)
                        logits = out[0] if isinstance(out, tuple) else out
                        loss = compute_ce_loss(logits, tgt)
                else:
                    out = student(ids)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = compute_ce_loss(logits, tgt)
                (loss / cfg.grad_accum).backward()
            lr = get_lr(step, total_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            if has_nan_params(student):
                raise RuntimeError("NaN params during distill — aborting")

    # ── Merge LoRA adapters into base model for standalone save ──
    if cfg.use_lora and hasattr(student, "merge_and_unload"):
        student = student.merge_and_unload()
        print("  Merged LoRA adapters into base model for standalone save.")

    # 5. Save as a new distill epoch.
    _EPOCHS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_num = db.last_epoch_num() + 1
    path = str(_EPOCHS_DIR / f"epoch{epoch_num}_distill.safetensors")
    state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    save_checkpoint(state, path)
    from_epoch = (db.best_epoch() or {}).get("epoch_num")
    db.add_epoch(epoch_num, path, parent_epoch=from_epoch, kind="distill",
                 status="candidate")
    db.add_distill_run(epoch_num, from_epoch or 0, epoch_num,
                       filtered_items=len(pairs), bloat_removed=bloat)
    db.emit("distill_done", {"epoch": epoch_num, "checkpoint": path,
                             "clean_items": len(pairs), "bloat_removed": bloat})
    return path
