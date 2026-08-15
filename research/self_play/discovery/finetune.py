"""Fine-tune the discovery model on its own curated DB content.

When the LLM deems its DB high-quality and large enough, this module builds
an SFT dataset from the DB's best content and fine-tunes a copy of the
current best model. The result is saved as a new epoch checkpoint, then
handed to the epoch manager for best-vs-loser comparison.

Training data sources (only curated, verified content — bloat is excluded):
  - discoveries  -> "Summarize what you discovered." -> summary
  - theories     (status='supported') -> "State the hypothesis." -> statement
  - scripts      (returncode=0, non-empty stdout) -> prompt-stub -> code
  - research     -> "Summarize findings on {query}." -> summary

Reuses existing ForgeAI infra:
  - research.config.get_config
  - research.model_loader.ModelLoader.build_model_fast, load_default_model
  - research.tokenizer_cache.get_tokenizer
  - research.training.training_utils (configure_optimizer, get_lr,
    compute_ce_loss, has_nan_params, vram_exceeded, vram_gb, init_ema,
    update_ema)
  - research.checkpoint_io.save_checkpoint
  - research.json_compat.dumps
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from research.paths import DATA_DIR
from research.self_play.discovery.discovery_db import DiscoveryDB


_EPOCHS_DIR = DATA_DIR / "discovery" / "epochs"


@dataclass
class FinetuneConfig:
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 4
    max_lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 20
    max_seq_len: int = 512
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    vram_limit_gb: float = 11.0


def build_sft_dataset(db: DiscoveryDB, tokenizer, max_seq_len: int) -> list[dict]:
    """Build (prompt, completion) pairs from curated DB content.

    Now includes tool-use trajectories from the tool_trajectories table.
    Trajectories are rendered in Qwen chat format and split into per-turn
    examples (same approach as sft_train.py).
    """
    pairs: list[tuple[str, str]] = []

    for r in db.query("SELECT summary FROM discoveries WHERE summary IS NOT NULL AND length(summary) > 10"):
        pairs.append(("Record a confirmed discovery.", r["summary"]))

    for r in db.query("SELECT statement, notes FROM theories WHERE status='supported' AND length(statement) > 10"):
        comp = r["statement"] + (f" Evidence: {r['notes']}" if r["notes"] else "")
        pairs.append(("State a supported hypothesis.", comp))

    for r in db.query("SELECT code, stdout FROM scripts WHERE returncode=0 AND length(code) > 20 AND length(stdout) > 0 LIMIT 200"):
        pairs.append(("Write useful Python code.", r["code"][:800]))

    for r in db.query("SELECT query, summary FROM research WHERE summary IS NOT NULL AND length(summary) > 10 LIMIT 100"):
        pairs.append((f"Summarize findings on: {r['query']}", r["summary"]))

    # Tool-use trajectories: render as Qwen-format per-turn examples.
    # These teach the model to call tools and produce final answers.
    from research.self_play.discovery.qwen_adapter import qwen_render_messages
    traj_pairs = _build_trajectory_pairs(db)
    pairs.extend(traj_pairs)

    # Tokenize into input_ids + completion_start for CE-on-completion.
    out = []
    eos = tokenizer.eos_token_id
    for prompt, completion in pairs:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if eos is not None:
            c_ids = c_ids + [eos]
        ids = (p_ids + c_ids)[:max_seq_len]
        if len(ids) < 8:
            continue
        comp_start = min(len(p_ids), len(ids) - 1)
        out.append({"ids": ids, "comp_start": comp_start})
    return out


def _build_trajectory_pairs(db: DiscoveryDB,
                            min_reward: float = 0.5,
                            max_pairs: int = 200) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs from tool-use trajectories.

    Each trajectory is a multi-turn conversation. We split it into per-turn
    examples: each assistant turn becomes a completion, with everything before
    it as the prompt.

    This mirrors the per-turn splitting in sft_train.py.
    """
    from research.self_play.discovery.qwen_adapter import (
        qwen_render_messages, IM_START, IM_END)

    trajectories = db.get_trajectories(min_reward=min_reward, limit=max_pairs)
    pairs = []

    for traj in trajectories:
        messages = traj["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            continue

        # Split into per-turn examples
        for i in range(1, len(messages)):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue

            # Prompt = everything up to and including the generation prompt
            prompt_msgs = messages[:i]
            prompt = qwen_render_messages(
                prompt_msgs, tools=None, add_generation_prompt=True)

            # Completion = this assistant turn (with <|im_end|>)
            if msg.get("tool_calls"):
                import json as _json
                body = "\n".join(
                    _json.dumps(tc, ensure_ascii=False) for tc in msg["tool_calls"]
                )
            else:
                body = msg.get("content", "")

            completion = f"{body}{IM_END}\n"

            if len(prompt) > 10 and len(body) > 2:
                pairs.append((prompt, completion))

    return pairs


def _collate(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to max length in batch. Returns (input_ids, targets, comp_mask)."""
    maxlen = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
        cs = b["comp_start"]
        tgt[i, cs:L] = ids[i, cs:L]
        mask[i, cs:L] = True
    return ids.to(device), tgt.to(device), mask.to(device)


def finetune_from_db(db: DiscoveryDB, base_checkpoint: str | None = None,
                     config: FinetuneConfig | None = None,
                     device: str = "cuda") -> str:
    """Fine-tune on the DB and return the new checkpoint path.

    Args:
        db: the discovery database (training source + epoch bookkeeping).
        base_checkpoint: path to the parent epoch checkpoint (None = base LFM2.5).
        config: FinetuneConfig.
    Returns:
        Path to the newly saved epoch checkpoint.
    """
    from research.config import get_config
    from research.model_loader import ModelLoader, load_default_model
    from research.tokenizer_cache import get_tokenizer
    from research.training.training_utils import (
        configure_optimizer, get_lr, compute_ce_loss, has_nan_params,
        vram_exceeded, init_ema, update_ema)
    from research.checkpoint_io import save_checkpoint

    cfg = config or FinetuneConfig()

    # Load base model + tokenizer (reuse existing loaders).
    model_cfg = get_config("lfm25_1.2b", device=device)
    model = ModelLoader.build_model_fast(
        model_cfg, checkpoint_path=base_checkpoint,
        moe_top_k=0, dtype=torch.bfloat16)
    model.to(device).train()
    tokenizer = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    dataset = build_sft_dataset(db, tokenizer, cfg.max_seq_len)
    if len(dataset) < 8:
        raise ValueError(f"DB too small to fine-tune: only {len(dataset)} usable samples")

    opt = configure_optimizer(model, cfg.max_lr, cfg.weight_decay)
    ema = init_ema(model)
    steps_per_epoch = max(1, len(dataset) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs

    step = 0
    for ep in range(cfg.epochs):
        # Shuffle-ish: reverse for variety without importing random overhead.
        order = list(range(len(dataset)))[::-1]
        for sp in range(steps_per_epoch):
            if vram_exceeded(cfg.vram_limit_gb, device):
                raise RuntimeError(f"VRAM limit {cfg.vram_limit_gb}GB exceeded during fine-tune")
            opt.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                idx = (sp * cfg.grad_accum + _) % len(order)
                batch = [dataset[order[idx]]] if cfg.batch_size == 1 else \
                    [dataset[order[(sp * cfg.grad_accum + j) % len(order)]]
                     for j in range(cfg.batch_size)]
                ids, tgt, mask = _collate(batch, pad_id, device)
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else out
                loss = compute_ce_loss(logits, tgt) / cfg.grad_accum
                loss.backward()
            lr = get_lr(step, total_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model, cfg.ema_decay)
            step += 1
            if has_nan_params(model):
                raise RuntimeError("NaN params during fine-tune — aborting")

    # Save the new epoch checkpoint.
    _EPOCHS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_num = db.last_epoch_num() + 1
    path = str(_EPOCHS_DIR / f"epoch{epoch_num}.safetensors")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_checkpoint(state, path)
    db.add_epoch(epoch_num, path, parent_epoch=None, kind="finetune",
                 status="candidate")
    db.emit("finetune_done", {"epoch": epoch_num, "checkpoint": path,
                              "samples": len(dataset), "steps": step})
    return path
