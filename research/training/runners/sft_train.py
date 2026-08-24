"""Supervised fine-tuning (SFT) for ForgeLM V2 (LFM2.5-1.2B).

Trains the model on teacher-generated data (function-calling trajectories,
chain-of-thought, code) using completion-only cross-entropy loss — only
assistant tokens contribute to the loss, prompt/tool/user tokens are masked.

Data format:
  - tool_use_fc: JSONL with {"messages": [...]} — multi-turn OpenAI-style
    function-calling conversations (user / assistant w/ tool_calls / tool /
    final assistant). Rendered into Qwen chat format with tool-call markers.
  - short_cot / code / tool_use: JSONL with {"prompt": ..., "response": ...}
    — single-turn Q&A. Rendered as user→assistant chat.

The LFM2.5 tokenizer uses Qwen-style markers (<|im_start|>/<|im_end|>, ids 6/7)
but ships without a chat_template, so we apply the Qwen chat format manually:

    <|im_start|>user\n{content}<|im_end|>\n
    <|im_start|>assistant\n{content}<|im_end|>\n
    <|im_start|>tool\n{content}<|im_end|>\n

For assistant tool_calls messages, we serialize them using LFM2.5's native
special tokens (ids 10/11) as tool-call markers:

    <|im_start|>assistant
    {start_token}
    {"name": "...", "arguments": {...}}
    {end_token}
    <|im_end|>

Usage:
    python -m research.training.runners.sft_train \\
        --data research/data/finetune/tool_use_fc_70.jsonl \\
        --data research/data/finetune/short_cot_70.jsonl \\
        --data research/data/finetune/code_70.jsonl \\
        --config lfm25_1.2b \\
        --checkpoint research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors \\
        --save research/checkpoints/ForgeLM_V2_LFM25-1.2B.sft.safetensors \\
        --max-steps 500 --lr 5e-5 --batch-size 2 --seq-len 1024
"""
import argparse
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from research.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    save_training_checkpoint,
    step_checkpoint_path,
)
from research.config import get_config
from research.model_loader import ModelLoader
from research.runtime.task_logger import task_scope
from research.tokenizer_cache import get_tokenizer
from research.training.data.efficient_pipeline import (
    DiskTokenCache,
    AsyncPrefetcher,
    PackedSequenceDataset,
    ModelDataCache,
    get_disk_cache,
    get_model_cache,
)
from research.training.training_utils import (
    add_safeguard_args,
    configure_optimizer,
    get_lr,
    grad_accum_for_effective_batch,
    has_nan_params,
    init_ema,
    oom_guard,
    update_ema,
    patch_triton_cache_for_windows,
    ram_exceeded,
    ram_usage,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
    load_anchor_cached,
)

# Special token ids from the LFM2.5 tokenizer (Qwen-style).
IM_START = 6
IM_END = 7

# Tool call markers — LFM2.5's native special tokens (single-token ids 10/11).
# Built from hex to avoid IDE tool-call parsing confusion.
_TOOL_CALL_START = bytes.fromhex("3c7c746f6f6c5f63616c6c5f73746172747c3e").decode("ascii")
_TOOL_CALL_END = bytes.fromhex("3c7c746f6f6c5f63616c6c5f656e647c3e").decode("ascii")


# ── Chat-format rendering ────────────────────────────────────────────────────

def _render_tool_call(tc: dict) -> str:
    """Serialize one tool call using the native special token markers."""
    return _TOOL_CALL_START + '\n' + json.dumps(tc, ensure_ascii=False) + '\n' + _TOOL_CALL_END


def render_messages(messages: list[dict]) -> tuple[str, int]:
    """Render a multi-turn message list into Qwen chat format.

    Returns (text, completion_start_char). completion_start_char is the
    character offset where the FIRST assistant response begins — everything
    before it is the prompt (loss-masked), everything from there on is the
    completion (loss-bearing).

    For multi-turn conversations (tool_use_fc), the entire conversation after
    the first user message is the "completion" — the model learns to produce
    the full assistant + tool + final-assistant sequence given the user prompt.
    """
    parts = []
    completion_start = None
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "user":
            parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n")
        elif role == "assistant":
            if completion_start is None:
                # Mark where the first assistant turn begins (start of completion).
                completion_start = sum(len(p) for p in parts)
            if m.get("tool_calls"):
                body = "\n".join(_render_tool_call(tc) for tc in m["tool_calls"])
            else:
                body = m.get("content", "")
            parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        elif role == "tool":
            name = m.get("name", "tool")
            content = m.get("content", "")
            parts.append(f"<|im_start|>tool\n{name}\n{content}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{m.get('content','')}<|im_end|>\n")
    # Add a final generation prompt so the model knows the assistant turn is next
    # (only if the last message isn't already an assistant message).
    if messages and messages[-1]["role"] != "assistant":
        if completion_start is None:
            completion_start = sum(len(p) for p in parts)
        parts.append("<|im_start|>assistant\n")
    return "".join(parts), (completion_start or 0)


def render_single_turn(prompt: str, response: str) -> tuple[str, int]:
    """Render a single-turn Q&A into Qwen chat format.

    Returns (text, completion_start_char).
    """
    prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    completion_start = len(prompt_text)
    full = prompt_text + response + "<|im_end|>\n"
    return full, completion_start


def _render_message(m: dict) -> str:
    """Render a single message into Qwen chat format (no generation prompt)."""
    role = m["role"]
    if role == "user":
        return f"<|im_start|>user\n{m['content']}<|im_end|>\n"
    elif role == "assistant":
        if m.get("tool_calls"):
            body = "\n".join(_render_tool_call(tc) for tc in m["tool_calls"])
        else:
            body = m.get("content", "")
        return f"<|im_start|>assistant\n{body}<|im_end|>\n"
    elif role == "tool":
        name = m.get("name", "tool")
        content = m.get("content", "")
        return f"<|im_start|>tool\n{name}\n{content}<|im_end|>\n"
    return f"<|im_start|>{role}\n{m.get('content','')}<|im_end|>\n"


def split_multi_turn(messages: list[dict]) -> list[tuple[str, str]]:
    """Split a multi-turn conversation into per-turn training examples.

    Each assistant turn becomes a separate (prompt, completion) pair where:
      - prompt = all messages up to (but not including) this assistant turn,
        rendered in Qwen chat format + a generation prompt
        ("<|im_start|>assistant\n")
      - completion = the assistant turn body + "<|im_end|>\n"

    This teaches the model to STOP after each assistant turn (emit tool calls
    and stop, or emit a final answer and stop) — critical for agent loops
    where the orchestrator intercepts tool calls, executes them, and feeds
    results back.

    For a conversation:
      user → assistant(tool_calls) → tool → tool → assistant(final)
    We produce 2 examples:
      1. prompt=user, completion=tool_calls+<|im_end|>
      2. prompt=user+tool_calls+tool+tool, completion=final+<|im_end|>
    """
    examples = []
    prefix_parts = []
    for m in messages:
        if m["role"] == "assistant":
            # This is a turn boundary — create a training example.
            prompt_text = "".join(prefix_parts) + "<|im_start|>assistant\n"
            if m.get("tool_calls"):
                body = "\n".join(_render_tool_call(tc) for tc in m["tool_calls"])
            else:
                body = m.get("content", "")
            completion_text = body + "<|im_end|>\n"
            examples.append((prompt_text, completion_text))
            # Add this assistant turn to the prefix for subsequent examples.
            prefix_parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        else:
            prefix_parts.append(_render_message(m))
    return examples


# ── Dataset loading + tokenization ───────────────────────────────────────────

def load_examples(paths: list[str]) -> list[dict]:
    """Load and deduplicate examples from one or more JSONL files.

    Each line should be a JSON object. For tool_use_fc, expects {"messages": [...]}.
    For short_cot/code/tool_use, expects {"prompt": ..., "response": ...}.
    """
    examples = []
    seen = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        n_loaded = 0
        n_skipped = 0
        with open(p, encoding="utf-8") as f:
            content = f.read()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                n_skipped += 1
                continue
            # Normalize: either messages list or prompt/response.
            if "messages" in obj and isinstance(obj["messages"], list):
                key = json.dumps(obj["messages"], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                ex = {"type": "multi_turn", "messages": obj["messages"]}
                # Carry reward for reward-weighted SFT (if present)
                if "reward" in obj:
                    ex["reward"] = float(obj["reward"])
                examples.append(ex)
                n_loaded += 1
            elif "prompt" in obj and "response" in obj:
                key = obj["prompt"][:200]
                if key in seen:
                    continue
                seen.add(key)
                examples.append({"type": "single_turn",
                                 "prompt": obj["prompt"],
                                 "response": obj["response"]})
                n_loaded += 1
            else:
                n_skipped += 1
        print(f"  {path}: {n_loaded} loaded, {n_skipped} skipped")
    return examples


def load_examples_parquet(paths: list[str]) -> tuple[list[dict], list[dict] | None]:
    """Load examples from one or more Parquet files.

    Returns ``(examples, pre_tokenized)`` where:
      - If the parquet contains pre-tokenized columns (``input_ids`` +
        ``labels``), ``examples`` is empty and ``pre_tokenized`` is a list of
        ``{"input_ids": [...], "labels": [...]}`` dicts ready for training.
      - If the parquet contains raw data (``messages`` or ``prompt``/
        ``response``), ``examples`` is a list of normalised dicts (same format
        as :func:`load_examples`) and ``pre_tokenized`` is ``None``.
    """
    from research.training.data.parquet_dataset import ParquetDataset

    examples: list[dict] = []
    seen: set = set()
    pre_tokenized: list[dict] | None = None

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        ds = ParquetDataset(str(p))
        col_names = set(ds.schema.names)

        # ── Pre-tokenized path: input_ids + labels columns ──
        if "input_ids" in col_names and "labels" in col_names:
            if pre_tokenized is None:
                pre_tokenized = []
            n_loaded = 0
            for i in range(len(ds)):
                row = ds[i]
                ids = row.get("input_ids")
                labs = row.get("labels")
                if ids is None or labs is None:
                    continue
                pre_tokenized.append({
                    "input_ids": list(ids),
                    "labels": list(labs),
                    "n_comp": sum(1 for x in labs if x != -100),
                })
                n_loaded += 1
            print(f"  {path}: {n_loaded} pre-tokenized rows loaded")
            continue

        # ── Raw data path: messages or prompt/response ──
        n_loaded = 0
        n_skipped = 0
        for i in range(len(ds)):
            row = ds[i]
            if "messages" in row and isinstance(row["messages"], list):
                key = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                examples.append({"type": "multi_turn", "messages": row["messages"]})
                n_loaded += 1
            elif "prompt" in row and "response" in row:
                key = str(row["prompt"])[:200]
                if key in seen:
                    continue
                seen.add(key)
                examples.append({"type": "single_turn",
                                 "prompt": row["prompt"],
                                 "response": row["response"]})
                n_loaded += 1
            else:
                n_skipped += 1
        print(f"  {path}: {n_loaded} loaded, {n_skipped} skipped")

    return examples, pre_tokenized


# Tokenization cache — avoids re-tokenizing repeated prompts (2-3x speedup)
# Uses a tiered cache: in-memory LRU + disk-backed cache (DiskTokenCache).
# The disk cache persists across runs, eliminating re-tokenization on restart.
#
# NOTE: This cache is distinct from research/tokenizer_cache.py.
#   - tokenizer_cache.py caches the *tokenizer object* (singleton, lru_cache
#     maxsize=8) to avoid the 4.3s AutoTokenizer load overhead across call sites.
#   - This cache (_tokenize_cache) caches *tokenization results* (text→token IDs)
#     to avoid re-encoding repeated training texts, with disk persistence.
_tokenize_cache: OrderedDict[str, list[int]] = OrderedDict()
_CACHE_MAX = 10000  # max 10K entries in memory (~50MB typical)
_disk_tok_cache: DiskTokenCache | None = None


def _get_disk_tok_cache() -> DiskTokenCache:
    """Get the global disk tokenization cache (lazy init)."""
    global _disk_tok_cache
    if _disk_tok_cache is None:
        _disk_tok_cache = DiskTokenCache()
    return _disk_tok_cache


def _tokenize_cached(tokenizer, text: str, max_seq_len: int) -> list[int] | None:
    """Tokenize text with tiered caching (memory LRU + disk). Returns None on error.

    Cache hierarchy:
      1. In-memory LRU (fastest, bounded to _CACHE_MAX entries)
      2. Disk-backed cache (persists across runs, memory-mapped reads)
      3. Tokenizer (slowest, result cached in both layers)
    """
    # Layer 1: in-memory LRU
    if text in _tokenize_cache:
        _tokenize_cache.move_to_end(text)
        return _tokenize_cache[text]

    # Layer 2: disk cache
    disk = _get_disk_tok_cache()
    cached = disk.get(text, getattr(tokenizer, "_tokenizer_hash", "default"))
    if cached is not None:
        # Promote to in-memory cache
        if len(text) < 10000:
            while len(_tokenize_cache) >= _CACHE_MAX:
                _tokenize_cache.popitem(last=False)
            _tokenize_cache[text] = cached
        return cached

    # Layer 3: tokenize and cache in both layers
    try:
        enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
        if not isinstance(ids, list):
            ids = list(ids)
        if len(text) < 50000:
            # In-memory cache
            if len(text) < 10000:
                while len(_tokenize_cache) >= _CACHE_MAX:
                    _tokenize_cache.popitem(last=False)
                _tokenize_cache[text] = ids
            # Disk cache (persists across runs)
            disk.put(text, ids, getattr(tokenizer, "_tokenizer_hash", "default"))
        return ids
    except Exception:
        return None


def tokenize_example(ex: dict, tokenizer, max_seq_len: int) -> list[dict]:
    """Tokenize an example into one or more (input_ids, labels) pairs with
    completion-only masking.

    For single_turn: returns one example.
    For multi_turn: splits into per-assistant-turn examples (see split_multi_turn),
    each ending at <|im_end|> so the model learns to STOP after each turn.

    Returns a list (possibly empty if all splits are too long/short).
    """
    if ex["type"] == "multi_turn":
        pairs = split_multi_turn(ex["messages"])
    else:
        pairs = [(f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n",
                  ex["response"] + "<|im_end|>\n")]

    results = []
    for prompt_text, completion_text in pairs:
        full_text = prompt_text + completion_text
        # Tokenize with cache — repeated prompts skip re-tokenization (2-3x faster)
        input_ids = _tokenize_cached(tokenizer, full_text, max_seq_len)
        if input_ids is None:
            continue
        if len(input_ids) < 4 or len(input_ids) > max_seq_len:
            continue

        # Find completion start in token space (also cached).
        prompt_ids = _tokenize_cached(tokenizer, prompt_text, max_seq_len)
        if prompt_ids is None:
            continue
        comp_start_tok = len(prompt_ids)

        # Build labels: -100 for prompt, real id for completion.
        labels = [-100] * comp_start_tok + input_ids[comp_start_tok:]
        while len(labels) < len(input_ids):
            labels.append(-100)

        results.append({"input_ids": input_ids, "labels": labels,
                        "n_comp": len(input_ids) - comp_start_tok,
                        "reward": ex.get("reward", 1.0)})
    return results


def collate_batch(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad a batch of variable-length examples.

    Returns (input_ids, labels, attention_mask, reward_weights) all [B, T] or [B].
    Left-padding keeps the completion at the end so causal LM predicts correctly.
    Uses pin_memory when transferring to GPU for faster H2D copy.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    b = len(batch)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((b, max_len), dtype=torch.long)
    reward_weights = torch.ones(b, dtype=torch.float32)
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        offset = max_len - n  # left-pad
        input_ids[i, offset:] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, offset:] = torch.tensor(ex["labels"], dtype=torch.long)
        attn_mask[i, offset:] = 1
        reward_weights[i] = ex.get("reward", 1.0)
    # Use pin_memory + non_blocking for faster CPU→GPU transfer
    use_pin = "cuda" in device and torch.cuda.is_available()
    if use_pin:
        input_ids = input_ids.pin_memory()
        labels = labels.pin_memory()
        attn_mask = attn_mask.pin_memory()
    return (input_ids.to(device, non_blocking=use_pin),
            labels.to(device, non_blocking=use_pin),
            attn_mask.to(device, non_blocking=use_pin),
            reward_weights.to(device, non_blocking=use_pin))


# ── Training loop ────────────────────────────────────────────────────────────

def compute_loss(model, input_ids, labels, attn_mask,
                 entropy_alpha: float = 0.0,
                 sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Completion-only cross-entropy loss with optional anti-regression techniques.

    The model's forward(targets=...) computes mean CE over ALL non-ignored
    positions. We pass labels as targets with -100 for masked positions, so
    only completion tokens contribute.

    Memory-optimized fast paths (no full logits materialization):
      - entropy_alpha=0, no sample_weights → chunked_linear_cross_entropy
      - entropy_alpha>0, no sample_weights  → chunked_entropy_weighted_ce
      Both avoid materializing the [B, T, V] logits tensor — 20x+ faster on
      large vocabularies (vocab=65536).

    Anti-regression extensions (require the manual logits path when
    sample_weights is provided):
      - Token Entropy Weighting (WeFT/VCORE 2025): when entropy_alpha > 0,
        per-token CE is scaled by (1 + alpha * normalized_entropy), giving
        MORE weight to high-entropy (reasoning/uncertain) tokens and LESS
        to low-entropy (boilerplate) tokens.
      - Easy Sample Upweighting (ICML 2025): when sample_weights is provided
        (pre-computed from base model loss), per-example loss is scaled by
        weight = 1/(1+base_loss), so easy samples get higher weight.
    """
    # ── Fast path: chunked CE (no logits materialization) ──
    # Used when no per-example sample weighting is needed. The model's
    # forward(targets=labels) computes the loss internally via
    # chunked_linear_cross_entropy or chunked_entropy_weighted_ce,
    # never materializing [B, T, V] logits.
    if sample_weights is None:
        # Shift labels for causal LM: hidden[i] predicts token i+1.
        # Pad with -100 to match hidden [B, T] (last position has no target).
        shift_labels = F.pad(labels[:, 1:], (0, 1), value=-100)  # [B, T]
        # Set entropy_alpha on the model config so the forward uses the
        # correct chunked CE variant.
        if hasattr(model, 'config') and hasattr(model.config, 'entropy_alpha'):
            model.config.entropy_alpha = entropy_alpha
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=input_ids.is_cuda,
        ):
            out = model(input_ids, attention_mask=attn_mask, targets=shift_labels)
        if isinstance(out, tuple):
            loss = out[1] if out[1] is not None else out[0]
        else:
            loss = out
        if loss is None:
            # Model didn't use chunked CE (use_chunked_ce=False); fall through
            # to the manual path below.
            pass
        else:
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss ({loss.item()}). "
                    f"Check for NaN in inputs, exploding gradients, or lr too high."
                )
            return loss

    # ── Manual path: full logits for sample weighting ──
    # Only used when sample_weights is provided (requires per-example control).
    # Forward — compute logits manually for per-position control.
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=input_ids.is_cuda,
    ):
        out = model(input_ids, attention_mask=attn_mask)
    logits = out[0] if isinstance(out, tuple) else out  # [B, T, V]
    if logits is None:
        raise RuntimeError("model returned None logits; pass targets manually")
    # Shift: predict token t+1 from position t.
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()       # [B, T-1]

    # Per-token CE (reduction="none") for weighting control.
    ce_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())  # [B, T-1]

    # Mask for non-ignored (completion) tokens.
    mask = (shift_labels != -100).float()  # [B, T-1]

    if entropy_alpha > 0.0:
        # ── Token Entropy Weighting (WeFT/VCORE 2025) ──
        # High-entropy tokens (uncertain/reasoning) get MORE weight;
        # low-entropy tokens (boilerplate) get LESS.
        with torch.no_grad():
            probs = F.softmax(shift_logits.float(), dim=-1)  # [B, T-1, V]
            # Per-token entropy: H = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, T-1]
            # Normalize to [0, 2] range. Max entropy = ln(vocab_size).
            # For vocab=65536, ln(65536) ≈ 11.09.
            vocab_size = float(shift_logits.size(-1))
            max_entropy = torch.log(torch.tensor(vocab_size,
                                                 device=shift_logits.device,
                                                 dtype=torch.float32))
            normalized_entropy = (entropy / max_entropy.clamp(min=1e-8)) * 2.0  # [0, 2]
            # Weight: w = 1.0 + alpha * normalized_entropy
            entropy_weights = 1.0 + entropy_alpha * normalized_entropy  # [B, T-1]
        # Apply entropy weights to per-token CE.
        ce_per_token = ce_per_token * entropy_weights

    # Per-example loss: mean over non-ignored tokens.
    per_example_loss = (ce_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)  # [B]

    if sample_weights is not None:
        # ── Easy Sample Upweighting (ICML 2025) ──
        # weight = 1/(1+base_loss); easy samples (low base loss) get higher weight.
        per_example_loss = per_example_loss * sample_weights

    loss = per_example_loss.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite training loss ({loss.item()}). "
            f"Check for NaN in inputs, exploding gradients, or lr too high."
        )
    return loss


def compute_l2_sp_loss(model, anchor_named_params: dict, base_lambda: float) -> torch.Tensor:
    """L2-SP Anchor Regularization (NeurIPS 2024).

    Penalizes L2 distance between trainable parameters and anchor (pre-SFT)
    weights to prevent catastrophic forgetting / regression on base capabilities.

    Layer-wise lambda: lower layers (0-5) get 10x higher lambda than upper
    layers (6-15) to preserve foundational representations (embedding, early
    conv/attention) while allowing higher layers to adapt to the new task.

    Returns the weighted L2-SP loss (already scaled by per-layer lambda).
    The caller adds this to CE loss: total = ce_loss + l2_sp_loss.
    """
    import re
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name not in anchor_named_params:
            continue
        anchor_p = anchor_named_params[name]
        # Determine layer-wise lambda from parameter name.
        lam = base_lambda
        m = re.search(r'layers?\.?(\d+)', name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx <= 5:
                lam = base_lambda * 10.0  # Lower layers: 10x higher penalty
            # else: upper layers use base_lambda
        l2_loss = l2_loss + lam * (p.float() - anchor_p.float()).pow(2).sum()
    return l2_loss


def compute_sample_weights(model, dataset, pad_id, device, batch_size=1):
    """Pre-compute per-example sample weights from base model loss (ICML 2025).

    Runs one forward pass per example (in eval mode, no grad) to get the base
    model's completion-only CE loss. Then weight = 1/(1+base_loss) — easy
    samples (low loss) get higher weight, hard samples get lower weight.

    Returns a tensor of shape [len(dataset)] with per-example weights.
    """
    weights = torch.zeros(len(dataset), dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, labels, attn_mask, _rw = collate_batch(batch, pad_id, device)
            loss = compute_loss(model, input_ids, labels, attn_mask)
            # Per-example loss is the mean CE for this batch (batch_size=1 typically).
            w = 1.0 / (1.0 + loss.item())
            for j in range(len(batch)):
                weights[i + j] = w
    model.train()
    # Normalize weights so mean = 1.0 (keeps loss scale stable).
    weights = weights / weights.mean().clamp(min=1e-8)
    return weights


def main():
    p = argparse.ArgumentParser(description="SFT for ForgeLM V2 (LFM2.5-1.2B)")
    p.add_argument("--data", nargs="+", required=True,
                   help="Training data file(s) (JSONL or Parquet)")
    p.add_argument("--data-format", default="jsonl", choices=["jsonl", "parquet"],
                   help="Data format: 'jsonl' (default) or 'parquet'. "
                        "When 'parquet', files are read via ParquetDataset "
                        "(memory-mapped, ZSTD-compressed).")
    # ── Data pipeline: disk cache, packed sequences, async prefetch ──
    p.add_argument("--disk-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Use disk-backed tokenization cache (persists across runs, "
                        "eliminates re-tokenization on restart). Default: True.")
    p.add_argument("--pack-sequences", action=argparse.BooleanOptionalAction, default=True,
                   help="Pack variable-length examples into fixed-length sequences "
                        "(Llama-3 style, eliminates 30-50% padding waste). Default: True.")
    p.add_argument("--async-prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="Use async background prefetcher for data loading (2-3x throughput "
                        "on I/O-bound workloads). Default: True.")
    p.add_argument("--prefetch-count", type=int, default=4,
                   help="Number of batches to prefetch ahead (default 4).")
    p.add_argument("--config", default="forgelm_v7_moe",
                   help="Model config name (default: forgelm_v7_moe)")
    p.add_argument("--checkpoint", default="research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors",
                   help="Base checkpoint to fine-tune from")
    p.add_argument("--save", default="research/checkpoints/ForgeLM_V2_LFM25-1.2B.sft.safetensors",
                   help="Output checkpoint path")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (NeurIPS 2025: batch size 1 is stable, "
                        "no grad_accum needed for single-GPU)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="muon_sf", choices=["fused", "bnb", "lion", "muon", "muon_sf", "muon_sf_plain", "flash_adamw", "flash_lion", "forge", "sf_normuon", "amuse", "mona", "cpu_offload"])
    # ── FreeToken-inspired training enhancements (R&D round 14) ──
    p.add_argument("--freetoken", action="store_true",
                   help="Enable FreeToken-inspired training pipeline (arXiv:2608.16157): "
                        "double-buffered gradient pipeline, bandwidth-adaptive chunked "
                        "transfers, predictive offload. Only applies with --optimizer cpu_offload.")
    p.add_argument("--double-buffer", action="store_true",
                   help="Enable ping-pong grad buffers for transfer/compute overlap. "
                        "Subset of --freetoken. Only applies with --optimizer cpu_offload.")
    p.add_argument("--bandwidth-adaptive", action="store_true",
                   help="Profile PCIe bandwidth and auto-set chunk size for transfers. "
                        "Subset of --freetoken. Only applies with --optimizer cpu_offload.")
    p.add_argument("--chunk-size-mb", type=int, default=None,
                   help="Override auto-computed chunk size for chunked grad transfers. "
                        "Only applies with --bandwidth-adaptive or --freetoken.")
    p.add_argument("--elastic-grad-accum", action="store_true",
                   help="Dynamically increase grad_accum when VRAM pressure is high, "
                        "decrease when low. Prevents OOM without manual tuning.")
    p.add_argument("--grad-mixup", type=int, default=1,
                   help="Grad mixup: average gradients from N batches before optimizer step "
                        "(1=disabled, 2=two-batch, 3=three-batch). "
                        "Tested: 3-way mixup + muon_sf = 1.25x better convergence vs AdamW. "
                        "Cost: N× forward+backward per step, but fewer steps needed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable gradient checkpointing to save VRAM (default: True, "
                        "use --no-grad-checkpoint to disable)")
    p.add_argument("--checkpoint-strategy", default="all",
                   choices=["all", "ffn", "attn", "none", "lazy", "optimal"],
                   help="Selective gradient checkpointing: 'ffn' recomputes only "
                        "FFN (biggest activation, minimal compute penalty), 'attn' "
                        "recomputes only attention, 'all' recomputes full block, "
                        "'lazy' uses LazyTrain mixed-integer scheduling, "
                        "'optimal' uses Hirschberg knapsack for optimal selection. "
                        "Only applies when --grad-checkpoint is active.")
    p.add_argument("--lazy-train", action="store_true",
                   help="Enable LazyTrain scheduler: mixed-integer checkpoint "
                        "selection + Hybrid 8-bit operator (fused clip + 8-bit state). "
                        "1.24x sustained TFLOPS improvement on limited VRAM.")
    p.add_argument("--oomb", action="store_true",
                   help="Enable OOMB chunk-recurrent training for long contexts. "
                        "Processes sequences in chunks with on-the-fly recomputation. "
                        "O(1) activation memory regardless of sequence length. "
                        "Enables 128K+ context training on 12GB GPU.")
    p.add_argument("--hybrid-clip", action="store_true",
                   help="Use Hybrid8Bit fast gradient clipping (running EMA norm, "
                        "exact every 50 steps). Avoids global norm computation when "
                        "optimizer states are offloaded to CPU.")
    p.add_argument("--curriculum", default="none",
                   choices=["none", "vanilla", "pacing", "interleaved", "warmup"],
                   help="Curriculum learning strategy: order training data easy→hard. "
                        "18-45% fewer steps to reach baseline performance. "
                        "'pacing' gradually increases difficulty, 'warmup' does easy "
                        "warmup then random, 'interleaved' mixes with increasing hard ratio.")
    p.add_argument("--augment", action="store_true",
                   help="Enable training-time data augmentation (token noise, FIM, "
                        "target offset). Regularizes against overfitting in "
                        "multi-epoch training (data-constrained regime).")
    p.add_argument("--synpro", action="store_true",
                   help="Enable SYNPRO synthetic data generation: rephrase + reformat "
                        "training data for 3.7-5.2x effective tokens. "
                        "Useful in data-bound regime.")
    p.add_argument("--norm-type", default="rmsnorm",
                   choices=["rmsnorm", "seednorm", "dyt"],
                   help="Normalization type: 'seednorm' = SeeDNorm (self-rescaled "
                        "dynamic, better zero-shot), 'dyt' = Dynamic Tanh (bounded, "
                        "compatible with Muon optimizers). Default 'rmsnorm'.")
    p.add_argument("--distill", action="store_true",
                   help="Enable knowledge distillation from a teacher model. "
                        "Uses offline top-K logits caching + chunked KL loss.")
    p.add_argument("--teacher-checkpoint", type=str, default=None,
                   help="Teacher model checkpoint for distillation.")
    p.add_argument("--distill-topk", type=int, default=50,
                   help="Top-K logits to cache from teacher (distillation).")
    p.add_argument("--distill-truncate", type=float, default=1.0,
                   help="Sequence truncation ratio for distillation (0.5 = first 50%% "
                        "of tokens, 91%% performance at 50%% FLOPs).")
    p.add_argument("--distill-prefix", action="store_true",
                   help="On-policy prefix distillation: distill only reasoning "
                        "prefixes (2-40x FLOP reduction).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="Model dtype (bf16 recommended for 12GB GPU)")
    # NeurIPS 2025: batch size 1 is stable, no grad_accum needed for single-GPU
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--ema", action="store_true",
                   help="Enable EMA (Exponential Moving Average) shadow weights")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay rate (default 0.999)")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True,
                   help="Use LoRA (PEFT) instead of full fine-tuning. Faster, less VRAM. "
                        "(default: True, use --no-lora to disable)")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default 32)")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default 64)")
    p.add_argument("--bitnet-everywhere", action=argparse.BooleanOptionalAction, default=True,
                   help="Convert ALL nn.Linear to BitNetLinear (ternary b1.58 QAT). "
                        "No NF4/bnb needed — BitNet IS the quantization (1.58 bits). "
                        "Validated on V3 1.2B: 2.39x vs AdamW, 6.32GB VRAM. "
                        "(default: True, use --no-bitnet-everywhere to disable)")
    p.add_argument("--manual-lora", action="store_true",
                   help="Use manual LoRA adapters (BitNet-compatible, unlike PEFT). "
                        "Works with BitNetLinear. Auto-enabled with --bitnet-everywhere.")
    p.add_argument("--sequential-freeze", type=int, default=0,
                   help="Sequential freeze/unfreeze: train N layers at a time in phases. "
                        "0=disabled. 4=4 phases (4 layers each for 16-layer model). "
                        "Full forward pass preserves MHC/AttnRes; only gradients are scoped. "
                        "Best with 100+ steps/phase.")
    p.add_argument("--final-finetune-steps", type=int, default=0,
                   help="When using --sequential-freeze, reserve N steps at the end "
                        "to fine-tune ALL layers together. 0=disabled.")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile on the model (experimental on Windows)")
    p.add_argument("--use-forge-engine", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ForgeEngine for model loading (provides unified inference engine "
                        "with all activation features, crash recovery, KeyStack auto-detection). "
                        "The engine.model is used for training; engine is used for validation. "
                        "Default: True (use --no-use-forge-engine for legacy ModelLoader path).")
    # ── Anti-regression techniques ──
    p.add_argument("--entropy-alpha", type=float, default=0.5,
                   help="Token entropy weighting alpha (WeFT/VCORE 2025). "
                        "High-entropy tokens get more weight. 0 disables. Default 0.5")
    p.add_argument("--loss-function", default="ce",
                   choices=["ce", "focal", "label_smoothing", "lovasz", "dynamic_focal", "mixture"],
                   help="Loss function: 'ce' (standard), 'focal' (hard token focus), "
                        "'label_smoothing' (anti-overconfidence), 'lovasz' (exact-match), "
                        "'dynamic_focal' (curriculum focal), 'mixture' (combined). "
                        "Focal/Lovász give +36% exact match on math/QA (EMNLP 2024).")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (focusing parameter). 0=CE, 2=typical focal.")
    p.add_argument("--label-smoothing-eps", type=float, default=0.1,
                   help="Label smoothing epsilon (0=CE, 0.1=typical).")
    p.add_argument("--anchor", default=None,
                   help="Path to anchor checkpoint for L2-SP regularization "
                        "(NeurIPS 2024). If provided, penalizes drift from anchor weights.")
    p.add_argument("--l2-lambda", type=float, default=0.01,
                   help="L2-SP regularization lambda (NeurIPS 2024). "
                        "Lower layers get 10x this value. Default 0.01")
    p.add_argument("--sample-weighting", action="store_true", default=False,
                   help="Enable easy sample upweighting (ICML 2025). "
                        "Pre-computes base model loss per example; easy samples get higher weight.")
    # ── MTP auxiliary loss (Nemotron Lightning) ──
    p.add_argument("--mtp-weight", type=float, default=0.0,
                   help="Multi-Token Prediction auxiliary loss weight (Nemotron Lightning). "
                        "When > 0, enables MTP heads for auxiliary next-next-token prediction. "
                        "Improves representation quality and enables speculative decoding at inference. "
                        "0 = disabled (default).")
    # ── Validation during training ──
    p.add_argument("--val-every", type=int, default=0,
                   help="Run validation every N steps (0=disabled). "
                        "Uses a held-out portion of the dataset to compute eval loss. "
                        "Detects regression during training.")
    p.add_argument("--val-size", type=float, default=0.05,
                   help="Fraction of dataset to hold out for validation (default 0.05 = 5%%). "
                        "Only used when --val-every > 0.")
    add_safeguard_args(p)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    patch_triton_cache_for_windows()

    # Reduce CUDA memory fragmentation (critical for 12GB VRAM)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" and "cuda" in device else torch.float32

    # ── Load data ──
    print(f"Loading data from {len(args.data)} file(s) "
          f"[format={args.data_format}]...")
    pre_tokenized = None
    if args.data_format == "parquet":
        examples, pre_tokenized = load_examples_parquet(args.data)
    else:
        examples = load_examples(args.data)

    # ── Curriculum learning: order examples easy→hard ──
    if args.curriculum != "none" and examples:
        from research.training.data.curriculum_augment import CurriculumScheduler
        scheduler = CurriculumScheduler(strategy=args.curriculum)
        texts = [ex.get('prompt', '') + ' ' + ex.get('response', '') for ex in examples]
        order = scheduler.build_curriculum(texts)
        examples = [examples[i] for i in order]
        print(f"Curriculum ({args.curriculum}): {scheduler.stats()}")

    # ── SYNPRO synthetic data augmentation ──
    if args.synpro and examples:
        from research.training.data.curriculum_augment import SYNPROGenerator
        gen = SYNPROGenerator()
        texts = [ex.get('prompt', '') + ' ' + ex.get('response', '') for ex in examples[:100]]
        synthetic = gen.generate_batch(texts, n_per_text=2)
        print(f"SYNPRO: generated {len(synthetic)} synthetic variants from {len(texts)} source texts")

    # ── Tokenize (or use pre-tokenized parquet data directly) ──
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── Disk tokenization cache info ──
    if args.disk_cache:
        cache = _get_disk_tok_cache()
        print(f"Disk tokenization cache: {cache.stats()}")

    if pre_tokenized is not None:
        # Pre-tokenized parquet: skip tokenisation entirely.
        dataset = pre_tokenized
        print(f"Loaded {len(dataset)} pre-tokenized examples from parquet")
        if not dataset:
            print("No usable examples. Exiting.")
            return
    else:
        if not examples:
            print("No usable examples. Exiting.")
            return
        print(f"Total examples: {len(examples)}")
        print("Tokenizing (with disk cache)...")
        dataset = []
        n_dropped = 0
        for ex in examples:
            toks = tokenize_example(ex, tokenizer, args.seq_len)
            if not toks:
                n_dropped += 1
                continue
            dataset.extend(toks)
        print(f"Tokenized: {len(dataset)} usable examples "
              f"(from {len(examples)} conversations, {n_dropped} dropped)")
        if args.disk_cache:
            print(f"  Disk cache after tokenization: {_get_disk_tok_cache().stats()}")
    if not dataset:
        print("No examples after tokenization. Exiting.")
        return

    # ── Packed sequences (Llama-3 style, eliminates 30-50% padding) ──
    if args.pack_sequences:
        packed_ds = PackedSequenceDataset(dataset, seq_len=args.seq_len,
                                          pack_examples=True, pad_id=pad_id)
        print(f"Packed sequences: {packed_ds.stats()}")
        dataset = [packed_ds[i] for i in range(len(packed_ds))]
        # Convert packed dicts to the format expected by collate_batch
        dataset = [{"input_ids": ex["input_ids"].tolist(),
                     "labels": ex["labels"].tolist(),
                     "n_comp": ex["n_comp"],
                     "reward": 1.0} for ex in dataset]

    # ── Validation split ──
    val_dataset = None
    if args.val_every > 0:
        n_val = max(1, int(len(dataset) * args.val_size))
        random.shuffle(dataset)
        val_dataset = dataset[:n_val]
        dataset = dataset[n_val:]
        print(f"Validation: {len(val_dataset)} examples held out "
              f"(train: {len(dataset)}, val_every={args.val_every})")

    # ── Build model ──
    cfg = get_config(args.config, device=device)
    cfg.grad_clip = args.grad_clip
    if args.grad_checkpoint:
        cfg.use_gradient_checkpointing = True
        cfg.selective_gradient_checkpointing = args.checkpoint_strategy
        if args.checkpoint_strategy != "all":
            print(f"Selective gradient checkpointing: '{args.checkpoint_strategy}' strategy")
    elif args.batch_size > 8 or args.seq_len > 4096:
        # Auto-enable gradient checkpointing only for very large configs
        cfg.use_gradient_checkpointing = True
        print("Auto-enabled gradient checkpointing (batch>8 or seq>4096)")
    # Use chunked CE to save VRAM on the 65K vocab.
    cfg.use_chunked_ce = True
    cfg.ce_chunk_size = 128
    # MTP auxiliary loss (Nemotron Lightning)
    if args.mtp_weight > 0.0:
        cfg.use_mtp = True
        cfg.mtp_loss_weight = args.mtp_weight
        print(f"MTP auxiliary loss enabled (weight={args.mtp_weight})")

    print(f"Building model ({args.config}) from {args.checkpoint}...")
    # Use ForgeEngine for model loading (unified inference engine with all
    # activation features, crash recovery, KeyStack auto-detection).
    # The engine.model is used for training; engine is used for validation/eval.
    forge_engine = None
    if args.use_forge_engine:
        try:
            from research.inference.forge_engine import ForgeEngine
            print("  Loading via ForgeEngine (auto_activate=False for training)...")
            forge_engine = ForgeEngine.from_checkpoint(
                args.checkpoint, config_name=args.config,
                device=device, auto_activate=False)
            model = forge_engine.model
            print(f"  ForgeEngine loaded: {type(model).__name__}, "
                  f"KeyStack features: {forge_engine.keystack_features}")
        except Exception as e:
            print(f"  ForgeEngine load failed ({e}), falling back to ModelLoader")
            forge_engine = None
            model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.checkpoint, dtype=dtype)
    else:
        model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.checkpoint, dtype=dtype)
    model.to(device).train()

    # ── Advanced normalization: SeeDNorm or DyT ──
    if args.norm_type == "seednorm":
        from research.training.optim.advanced_norm import replace_rmsnorm_with_seednorm
        replace_rmsnorm_with_seednorm(model)
    elif args.norm_type == "dyt":
        from research.training.optim.advanced_norm import replace_rmsnorm_with_dyt
        replace_rmsnorm_with_dyt(model)

    # ── BitNet-everywhere: convert all Linear → BitNetLinear ──
    if args.bitnet_everywhere:
        from research.training.bitnet_lora import convert_to_bitnet_everywhere
        n_conv, n_already = convert_to_bitnet_everywhere(model)
        print(f"BitNet-everywhere: {n_conv} Linear → BitNetLinear, {n_already} already BitNet")

    # ── LoRA ──
    # Use manual LoRA if --bitnet-everywhere or --manual-lora (PEFT can't handle BitNetLinear).
    # Use PEFT LoRA only for --lora on non-BitNet models.
    use_manual_lora = args.manual_lora or (args.bitnet_everywhere and args.lora)

    if use_manual_lora:
        from research.training.bitnet_lora import add_lora_adapters
        target_mods = ["q_proj", "k_proj", "v_proj", "out_proj",
                       "w_gate", "w_up", "w_down"] if not args.bitnet_everywhere else None
        n_adapters, lora_params = add_lora_adapters(
            model, rank=args.lora_r, alpha=args.lora_alpha,
            target_modules=target_mods,
        )
        print(f"Manual LoRA: {n_adapters} adapters (rank={args.lora_r}), "
              f"{sum(p.numel() for p in lora_params)/1e6:.2f}M trainable params")
    elif args.lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # LFM2.5 module names: attn uses out_proj (not o_proj),
            # ffn uses w_gate/w_up/w_down (not gate_proj/up_proj/down_proj).
            # No task_type — ConfigurableResearchLLM is not an HF model;
            # we only train + merge, no generation through PEFT wrapper.
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                            "w_gate", "w_up", "w_down"],
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    if args.grad_checkpoint:
        # ConfigurableResearchLLM uses enable_gradient_checkpointing (not HF's
        # gradient_checkpointing_enable). The config flag path above already
        # enabled it via cfg.use_gradient_checkpointing, but call the method
        # explicitly too in case the model was built before the flag was set.
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

        # LazyTrain: mixed-integer checkpoint scheduling (1.24x TFLOPS)
        if args.lazy_train or args.checkpoint_strategy == "lazy":
            try:
                from research.training.runners.lazy_train import LazyTrainScheduler, TrainingBudget
                budget = TrainingBudget(gpu_memory_bytes=int(12 * 1024**3))
                scheduler = LazyTrainScheduler(model, budget)
                schedules = scheduler.analyze()
                scheduler.apply()
                print(f"  [LazyTrain] {scheduler.stats()}")
            except Exception as e:
                print(f"  [LazyTrain] unavailable ({e}), using standard checkpointing")

        # Optimal checkpointing: Hirschberg knapsack (25-28% faster than dp_knapsack)
        if args.checkpoint_strategy == "optimal":
            try:
                from research.training.runners.optimal_checkpoint import OptimalCheckpointPlanner
                planner = OptimalCheckpointPlanner(
                    model, memory_budget_bytes=int(6 * 1024**3))
                plan = planner.plan()
                planner.apply(plan)
                print(f"  [OptimalCheckpoint] {planner.memory_estimate()}")
            except Exception as e:
                print(f"  [OptimalCheckpoint] unavailable ({e})")

    # ── OOMB chunk-recurrent training (long context, O(1) activation memory) ──
    oomb_trainer = None
    if args.oomb:
        try:
            from research.training.runners.oomb_trainer import ChunkRecurrentTrainer
            oomb_trainer = ChunkRecurrentTrainer(
                model, chunk_size=512, max_seq_len=args.seq_len * 4,
                device=str(device), dtype=dtype)
            print(f"  [OOMB] Chunk-recurrent training: active "
                  f"(chunk=512, max_seq={args.seq_len * 4})")
        except Exception as e:
            print(f"  [OOMB] unavailable ({e})")

    # ── Hybrid 8-bit fast gradient clipping ──
    hybrid_clipper = None
    if args.hybrid_clip:
        try:
            from research.training.runners.lazy_train import Hybrid8BitOperator
            hybrid_clipper = Hybrid8BitOperator(max_norm=args.grad_clip)
            print("  [Hybrid8Bit] Fast gradient clipping: active (EMA norm, exact/50)")
        except Exception as e:
            print(f"  [Hybrid8Bit] unavailable ({e})")

    # ── torch.compile (experimental) ──
    if args.compile:
        try:
            from research.paths import TORCH_CACHE_DIR
            model = torch.compile(model, mode="reduce-overhead", dynamic=True,
                                  cache_dir=str(TORCH_CACHE_DIR / "inductor"))
            print("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without")

    # Auto-default to CPUAdamW on small GPUs (<15GB) if user didn't override
    if args.optimizer == "adamw" and torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_vram_gb < 15.0:
            print(f"  [Auto] GPU has {total_vram_gb:.1f}GB VRAM — switching to CPUAdamW optimizer")
            args.optimizer = "cpu_offload"

    optimizer = configure_optimizer(model, args.lr, args.weight_decay,
                                     optimizer_name=args.optimizer,
                                     freetoken=args.freetoken,
                                     double_buffer=args.double_buffer,
                                     bandwidth_adaptive=args.bandwidth_adaptive,
                                     chunk_size_mb=args.chunk_size_mb)

    # ── FORGE optimizer: register gradient hooks (must run AFTER optimizer
    # creation — the optimizer object didn't exist at the previous site) ──
    if args.optimizer == "forge":
        if hasattr(optimizer, 'register_hooks'):
            optimizer.register_hooks(model)

    # ── L2-SP Anchor Regularization (NeurIPS 2024) ──
    # Load anchor checkpoint and build a name→tensor map for L2-SP loss.
    # Only activates when --anchor is provided AND l2_lambda > 0.
    anchor_named_params = None
    if args.anchor and getattr(args, 'l2_lambda', 0) > 0:
        print(f"Loading anchor checkpoint for L2-SP: {args.anchor}")
        anchor_sd = load_anchor_cached(args.anchor)
        # Map anchor tensors to model device; match by parameter name.
        anchor_named_params = {}
        model_named = dict(model.named_parameters())
        for name, p in model_named.items():
            if name in anchor_sd:
                anchor_named_params[name] = anchor_sd[name].to(device).to(p.dtype)
        print(f"  L2-SP active: {len(anchor_named_params)} matched params, "
              f"lambda={args.l2_lambda} (lower layers 10x)")

    # ── Easy Sample Upweighting (ICML 2025) ──
    # Pre-compute per-example weights from base model loss.
    # Only activates when --sample-weighting flag is set.
    sample_weights = None
    if args.sample_weighting:
        print("Computing sample weights from base model loss (ICML 2025)...")
        sample_weights = compute_sample_weights(
            model, dataset, pad_id, device, batch_size=1)
        print(f"  Sample weights: min={sample_weights.min().item():.4f}, "
              f"max={sample_weights.max().item():.4f}, "
              f"mean={sample_weights.mean().item():.4f}")

    # ── EMA ──
    ema_state = None
    if args.ema:
        ema_state = init_ema(model)
        print(f"EMA enabled (decay={args.ema_decay})")

    # ── Grad accumulation ──
    grad_accum = max(1, args.grad_accum)
    eff_batch = args.batch_size * grad_accum
    print(f"\nTraining {args.max_steps} steps | batch {args.batch_size} | "
          f"grad_accum {grad_accum} | eff_batch {eff_batch} | "
          f"lr {args.lr} | seq_len {args.seq_len} | {len(dataset)} examples")
    if args.grad_mixup > 1:
        print(f"  Grad mixup: {args.grad_mixup}-way (averaging {args.grad_mixup} batches' gradients per step)")
    if args.entropy_alpha > 0:
        print(f"  Token entropy weighting: alpha={args.entropy_alpha} (WeFT/VCORE 2025)")

    # ── Sequential freeze/unfreeze schedule ──
    phase_schedule = None
    if args.sequential_freeze > 0:
        from research.training.bitnet_lora import compute_phase_schedule
        n_layers = len(model.blocks) if hasattr(model, 'blocks') else 16
        phase_schedule = compute_phase_schedule(
            n_layers=n_layers,
            n_phases=args.sequential_freeze,
            total_steps=args.max_steps,
            final_finetune_steps=args.final_finetune_steps,
        )
        print(f"  Sequential freeze: {args.sequential_freeze} phases, "
              f"{n_layers} layers, "
              f"{len(phase_schedule)} schedule entries")
        for i, (s, e, active) in enumerate(phase_schedule):
            layer_str = "ALL" if active is None else f"{active[0]}-{active[-1]}"
            print(f"    Phase {i+1}: steps {s}-{e}, layers {layer_str}")

    print(f"Save: {args.save}")

    # ── Async prefetcher info ──
    if args.async_prefetch:
        print(f"Async prefetcher: enabled (prefetch_count={args.prefetch_count})")

    aborted = False
    with task_scope("sft") as log:
        step = 0
        accum_count = 0
        indices = list(range(len(dataset)))

        # ── Async prefetcher: pre-collate batches in background ──
        # The prefetcher runs a background thread that pre-loads and collates
        # the next N batches while the GPU processes the current one.
        # We use it to pre-collate batches, then consume them in the main loop.
        prefetcher = None
        if args.async_prefetch and not args.augment:
            prefetcher = AsyncPrefetcher(
                dataset, batch_size=args.batch_size, device=device,
                collate_fn=collate_batch, prefetch_count=args.prefetch_count,
                shuffle=True, pad_id=pad_id)

        for epoch in range(100):
            random.shuffle(indices)
            # If using prefetcher, get batches from it; otherwise use indices
            if prefetcher is not None:
                batch_iter = iter(prefetcher)
            for batch_start in range(0, len(indices), args.batch_size):
                if step >= args.max_steps:
                    break
                if vram_exceeded(args.vram_limit_gb, device):
                    print("VRAM limit exceeded; emergency save + abort.")
                    emergency_save(model, args.save, "emergency", step, optimizer=optimizer)
                    aborted = True
                    break

                # ── RAM safeguard (psutil) ──
                # Prevents OOMKilled / freezing / stuttering from host memory pressure.
                # Throttle at threshold%, emergency save at threshold+10%.
                if ram_exceeded(args.ram_limit_percent):
                    ru = ram_usage()
                    emergency_pct = args.ram_limit_percent + 10
                    if ru.get("percent", 0) > emergency_pct:
                        print(f"RAM critical ({ru.get('percent', 0):.0f}%); emergency save + abort.")
                        emergency_save(model, args.save, "emergency", step, optimizer=optimizer)
                        aborted = True
                        break
                    else:
                        print(f"RAM high ({ru.get('percent', 0):.0f}%); gc.collect() + skip batch.")
                        import gc
                        gc.collect()
                        continue  # skip this batch, try next

                # ── Periodic gc.collect() to prevent memory fragmentation ──
                if step > 0 and step % 100 == 0:
                    import gc
                    gc.collect()

                batch_idx = indices[batch_start:batch_start + args.batch_size]
                if prefetcher is not None:
                    # Async prefetcher: batch is already collated and on device
                    try:
                        input_ids, labels, attn_mask, reward_weights = next(batch_iter)
                    except StopIteration:
                        break  # epoch exhausted
                else:
                    # Manual batching fallback (no DataLoader).
                    # collate_batch already uses pin_memory for GPU transfers.
                    # Future improvement: wrap dataset in torch.utils.data.DataLoader
                    # with num_workers=4, pin_memory=True, prefetch_factor=4 for
                    # multi-process data loading. The AsyncPrefetcher (enabled by
                    # default via --async-prefetch) provides similar throughput
                    # using a single background thread.
                    batch = [dataset[i] for i in batch_idx]
                    with log.time_phase("data_load"):
                        input_ids, labels, attn_mask, reward_weights = collate_batch(batch, pad_id, device)

                # ── Training-time data augmentation ──
                if args.augment:
                    from research.training.data.curriculum_augment import DataAugmentor
                    if not hasattr(main, '_augmentor'):
                        main._augmentor = DataAugmentor()
                    aug_input, aug_target = main._augmentor.augment(input_ids, vocab_size=cfg.vocab_size)
                    input_ids = aug_input
                    labels = aug_target

                # ── Sequential freeze/unfreeze ──
                # At phase boundaries, freeze/unfreeze LoRA params and rebuild optimizer.
                if phase_schedule is not None:
                    from research.training.bitnet_lora import (
                        get_active_layers_for_step, freeze_unfreeze_lora,
                        get_active_lora_params,
                    )
                    active_layers = get_active_layers_for_step(step, phase_schedule)
                    # Check if we just entered a new phase (compare to previous step)
                    if step == 0 or get_active_layers_for_step(step - 1, phase_schedule) != active_layers:
                        freeze_unfreeze_lora(model, active_layers=active_layers)
                        active_p = get_active_lora_params(model)
                        layer_str = "ALL" if active_layers is None else f"{active_layers[0]}-{active_layers[-1]}"
                        print(f"  [Phase] step {step}: training layers {layer_str} ({len(active_p)} params)")
                        # Rebuild optimizer with only active params
                        if active_p:
                            if args.optimizer in ("muon_sf", "muon_sf_plain"):
                                from research.training.bitnet_lora import build_muon_sf_lora_opt
                                optimizer = build_muon_sf_lora_opt(active_p, lr_muon=5e-3, lr_adam=args.lr)
                            else:
                                optimizer = configure_optimizer(model, args.lr, args.weight_decay,
                                                                optimizer_name=args.optimizer)
                            if hasattr(optimizer, 'train'):
                                optimizer.train()

                # ── Easy Sample Upweighting (ICML 2025) ──
                # Get pre-computed sample weights for this batch (if enabled).
                sw = None
                if sample_weights is not None:
                    sw = sample_weights[batch_idx].to(device)

                # ── Reward-Weighted SFT ──
                # Weight per-example loss by trajectory reward (if present).
                # High-reward trajectories get more weight, low-reward get less.
                # reward_weights is [B] with values in [0, 1]; normalize to mean=1.
                rw = reward_weights
                if rw.max() > 0:
                    rw = rw / rw.mean().clamp(min=1e-8)  # normalize mean=1
                    if sw is not None:
                        sw = sw * rw  # combine with easy-sample weights
                    else:
                        sw = rw

                # ── FreeToken 3-stage pipeline (R&D round 14) ──
                # With overlap optimizer (cpu_offload + double_buffer/freetoken):
                # The previous step's CPU AdamW runs in a background thread,
                # overlapping with this forward pass. We must wait() before
                # backward to ensure params are synced and grads are consumed.
                # This creates the 3-stage pipeline:
                #   Stage 1: GPU forward+backward (current batch)
                #   Stage 2: GPU→CPU grad transfer (overlapped with stage 3)
                #   Stage 3: CPU AdamW update (overlapped with stage 1 of next batch)
                _has_overlap = hasattr(optimizer, 'overlap') and optimizer.overlap
                if _has_overlap and hasattr(optimizer, 'wait'):
                    optimizer.wait()

                # Forward + backward (accumulate gradients).
                # Token entropy weighting (WeFT/VCORE 2025) applied inside compute_loss.
                with oom_guard(str(device), label="sft_fwd") as safe:
                    with log.time_phase("forward"):
                        ce_loss = compute_loss(model, input_ids, labels, attn_mask,
                                               entropy_alpha=args.entropy_alpha,
                                               sample_weights=sw)

                    # ── L2-SP Anchor Regularization (NeurIPS 2024) ──
                    # Total loss = CE_loss + l2_lambda * l2_sp_loss (layer-wise lambda inside).
                    if anchor_named_params is not None:
                        l2_sp = compute_l2_sp_loss(model, anchor_named_params, args.l2_lambda)
                        loss = ce_loss + l2_sp
                    else:
                        loss = ce_loss

                    with log.time_phase("backward"):
                        (loss / grad_accum).backward()
                if safe.skipped:
                    optimizer.zero_grad()
                    accum_count = 0
                    continue
                accum_count += 1
                last_loss = ce_loss.item()  # log CE only for comparability

                # ── Grad Mixup (novel: average gradients from N batches) ──
                # Tested in .devin/test_stack_winners.py: 3-way mixup + muon_sf
                # = 1.25x better convergence vs AdamW. Averages gradients from
                # N-1 additional batches before the optimizer step.
                if args.grad_mixup > 1 and accum_count >= grad_accum:
                    # Save current grads from first batch
                    saved_grads = {}
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            saved_grads[n] = p.grad.clone()

                    # Fetch N-1 more batches and accumulate their grads
                    mixup_batch_start = batch_start + args.batch_size
                    for mixup_i in range(args.grad_mixup - 1):
                        mixup_bs = mixup_batch_start + mixup_i * args.batch_size
                        if mixup_bs + args.batch_size > len(indices):
                            break  # not enough data for full mixup, use what we have
                        mixup_idx = indices[mixup_bs:mixup_bs + args.batch_size]
                        mixup_batch = [dataset[i] for i in mixup_idx]
                        mi, ml, mm, _ = collate_batch(mixup_batch, pad_id, device)
                        msw = sample_weights[mixup_idx].to(device) if sample_weights is not None else None
                        mce = compute_loss(model, mi, ml, mm,
                                           entropy_alpha=args.entropy_alpha,
                                           sample_weights=msw)
                        if anchor_named_params is not None:
                            mloss = mce + compute_l2_sp_loss(model, anchor_named_params, args.l2_lambda)
                        else:
                            mloss = mce
                        optimizer.zero_grad()
                        (mloss / grad_accum).backward()
                        # Accumulate: running average
                        for n, p in model.named_parameters():
                            if p.grad is not None and n in saved_grads:
                                saved_grads[n] = (saved_grads[n] * (mixup_i + 1) + p.grad) / (mixup_i + 2)

                    # Restore averaged grads
                    optimizer.zero_grad()
                    for n, p in model.named_parameters():
                        if n in saved_grads:
                            p.grad = saved_grads[n]

                # Only step optimizer when we've accumulated enough gradients.
                if accum_count < grad_accum:
                    continue

                # Optimizer step.
                with log.time_phase("optimizer"):
                    if hybrid_clipper is not None:
                        hybrid_clipper.clip(model.parameters())
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                    # DeepSeek-V3 aux-loss-free: update expert bias after step.
                    try:
                        from research.moe.moe import update_moe_biases, disable_dense_bypass
                        update_moe_biases(model)
                        # Disable dense_bypass after warmup so router activates.
                        warmup_steps = getattr(cfg, 'moe_dense_bypass_warmup_steps', 0)
                        if warmup_steps > 0 and step + 1 == warmup_steps:
                            disable_dense_bypass(model)
                    except Exception:
                        pass  # no-op for dense models
                accum_count = 0

                # ── FreeToken bandwidth recording + predictive offload (R&D 14) ──
                # Record bandwidth/VRAM samples for the predictive offload policy.
                # Every 10 steps, check if we should pre-emptively increase offload.
                if hasattr(optimizer, 'record_bandwidth_sample'):
                    vram_gb = torch.cuda.memory_allocated() / 1e9 if "cuda" in device else 0.0
                    optimizer.record_bandwidth_sample(vram_gb=vram_gb)
                    if step > 0 and step % 10 == 0 and hasattr(optimizer, 'should_preempt_offload'):
                        if optimizer.should_preempt_offload():
                            print(f"  [FreeToken] Predictive offload: VRAM trend suggests "
                                  f"approaching limit, increasing offload")

                # ── Elastic grad_accum (R&D 14) ──
                # Dynamically adjust grad_accum based on VRAM pressure.
                # When VRAM is high, increase grad_accum (smaller effective batch
                # per step, but avoids OOM). When low, decrease for faster convergence.
                if args.elastic_grad_accum and "cuda" in device:
                    vram_gb = torch.cuda.memory_allocated() / 1e9
                    vram_limit = args.vram_limit_gb
                    vram_ratio = vram_gb / vram_limit if vram_limit > 0 else 0
                    if vram_ratio > 0.85 and grad_accum < args.grad_accum * 4:
                        grad_accum = min(grad_accum + 1, args.grad_accum * 4)
                        print(f"  [Elastic] VRAM {vram_gb:.2f}/{vram_limit}GB — "
                              f"increased grad_accum to {grad_accum}")
                    elif vram_ratio < 0.60 and grad_accum > args.grad_accum:
                        grad_accum = max(grad_accum - 1, args.grad_accum)
                        print(f"  [Elastic] VRAM {vram_gb:.2f}/{vram_limit}GB — "
                              f"decreased grad_accum to {grad_accum}")

                # EMA update.
                if ema_state is not None:
                    update_ema(ema_state, model, args.ema_decay)

                # Periodic save.
                if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                    ckpt = step_checkpoint_path(args.save, step)
                    save_training_checkpoint(model, ckpt, optimizer=optimizer,
                                             ema_state=ema_state, step=step)
                    cleanup_step_checkpoints(args.save, args.keep_checkpoints)

                # NaN check.
                if step > 0 and step % 50 == 0 and has_nan_params(model):
                    print("NaN/Inf in params; emergency save + abort.")
                    emergency_save(model, args.save, "nan", step, optimizer=optimizer)
                    aborted = True
                    break

                # Logging.
                if step % 5 == 0 or step == args.max_steps - 1:
                    vram_gb = torch.cuda.memory_allocated() / 1e9 if "cuda" in device else 0.0
                    msg = (f"Step {step+1}/{args.max_steps} | epoch {epoch} | "
                           f"loss {last_loss:.4f} | lr {lr:.2e} | "
                           f"vram {vram_gb:.2f} GB")
                    # FreeToken bandwidth stats
                    if hasattr(optimizer, 'bandwidth_stats') and optimizer.bandwidth_adaptive:
                        bw = optimizer.bandwidth_stats()
                        if 'b_p' in bw:
                            msg += f" | B_P={bw['b_p']}"
                    print(msg)
                    log.log(msg)
                    write_status_json(args.status_file, {
                        "step": step + 1, "max_steps": args.max_steps,
                        "loss": last_loss, "lr": lr, "epoch": epoch,
                    })
                    write_heartbeat(args.heartbeat_file)

                # ── Validation ──
                if val_dataset is not None and step > 0 and step % args.val_every == 0:
                    # Note: don't call model.eval() — BitNet eval mode uses fp32
                    # master weights which causes dtype mismatch with bf16 inputs.
                    # torch.inference_mode() is sufficient for validation.
                    val_losses = []
                    with torch.inference_mode():
                        n_val_batches = min(10, len(val_dataset) // args.batch_size)
                        for vi in range(n_val_batches):
                            vbatch = val_dataset[vi * args.batch_size:(vi + 1) * args.batch_size]
                            if not vbatch:
                                continue
                            vi_ids, vi_labels, vi_mask, _ = collate_batch(vbatch, pad_id, device)
                            vloss = compute_loss(model, vi_ids, vi_labels, vi_mask,
                                                 entropy_alpha=0.0)  # plain CE for eval
                            val_losses.append(vloss.item())
                    val_loss = sum(val_losses) / max(len(val_losses), 1)
                    print(f"  [Val] step {step+1}: val_loss={val_loss:.4f} "
                          f"(train_loss={last_loss:.4f})")
                    write_status_json(args.status_file, {
                        "step": step + 1, "max_steps": args.max_steps,
                        "loss": last_loss, "val_loss": val_loss,
                        "lr": lr, "epoch": epoch,
                    })
                step += 1
            if aborted or step >= args.max_steps:
                break

        if not aborted:
            # If EMA enabled, restore EMA weights before saving (smoother model).
            if ema_state is not None:
                from research.training.training_utils import restore_ema
                restore_ema(ema_state, model)
                print("Restored EMA weights for final save.")
            # If LoRA enabled, merge adapter weights into base model so the
            # saved checkpoint is a standalone full model (no LoRA dependency
            # needed for inference). This is the standard approach for
            # self-play loops where the next epoch loads from a plain checkpoint.
            if use_manual_lora:
                from research.training.bitnet_lora import merge_lora_adapters
                n_merged = merge_lora_adapters(model)
                print(f"Merged {n_merged} manual LoRA adapters into base model for standalone save.")
            elif args.lora and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
                print("Merged LoRA adapters into base model for standalone save.")
            save_training_checkpoint(model, args.save, optimizer=optimizer,
                                     ema_state=ema_state, step=step,
                                     meta={"config": cfg.__dict__, "sft": True,
                                           "n_examples": len(dataset)})
            print(f"\nSaved SFT model to {args.save}")
            print(f"  Trained {step} steps on {len(dataset)} examples")


if __name__ == "__main__":
    main()
