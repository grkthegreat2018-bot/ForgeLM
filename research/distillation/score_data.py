"""Perplexity-based data scorer for distillation training data.

Runs each (prompt, solution) pair through the local ForgeLM model and computes
perplexity = exp(loss) of the solution tokens given the prompt. This is the
same technique used by Phi-3/Phi-4 for data curation.

## What perplexity tells us

- **Very high perplexity (>100)**: model finds the solution confusing — likely
  poorly formatted, wrong style, or garbage. FILTER OUT.
- **Very low perplexity (<5)**: model already knows this — no learning signal.
  CAN DEPRIORITIZE (but don't remove — easy examples help with format learning).
- **Medium perplexity (5-50)**: novel but learnable — Goldilocks zone.
  PRIORITIZE for training.

## Usage

    python -m research.distillation.score_data \
        --input research/distillation/distill_data.jsonl \
        --checkpoint research/checkpoints/ForgeLM_V2_BSP.safetensors

    # Score and filter to Goldilocks zone
    python -m research.distillation.score_data \
        --input research/distillation/distill_data.jsonl \
        --filter --min-ppl 5 --max-ppl 50 \
        --output research/distillation/distill_data_scored.jsonl

    # Score agentic trajectories
    python -m research.distillation.score_data \
        --input research/distillation/agentic_distill_data.jsonl \
        --agentic --checkpoint research/checkpoints/ForgeLM_V2_BSP.safetensors
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Load .env
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

os.environ.setdefault("PYTHONUTF8", "1")


def load_model_and_tokenizer(checkpoint: str | None = None,
                              config_name: str = "forgelm_v3",
                              device: str = "cuda"):
    """Load ForgeLM model + tokenizer for perplexity computation."""
    import torch
    from research.model_loader import load_default_model
    from research.tokenizer_cache import get_tokenizer

    model, _ = load_default_model(
        config_name, checkpoint_path=checkpoint,
        device=device, dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = get_tokenizer()
    return model, tokenizer


def compute_perplexity(model, tokenizer, prompt: str, completion: str,
                       device: str = "cuda", max_len: int = 2048) -> dict:
    """Compute perplexity of completion given prompt.

    Tokenizes prompt + completion, runs a forward pass, and computes
    cross-entropy loss over ONLY the completion tokens (prompt tokens are
    masked with -100).

    Returns:
        {"perplexity": float, "loss": float, "n_tokens": int, "error": str}
    """
    import torch

    # Tokenize prompt and completion separately to find the boundary
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    # Truncate if too long
    total_len = len(prompt_ids) + len(completion_ids)
    if total_len > max_len:
        # Truncate prompt from the left (keep completion intact)
        excess = total_len - max_len
        if excess < len(prompt_ids):
            prompt_ids = prompt_ids[excess:]
        else:
            # Completion itself is too long — truncate from right
            prompt_ids = []
            completion_ids = completion_ids[:max_len]

    # Build input_ids and labels (mask prompt with -100)
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(completion_ids) == 0:
        return {"perplexity": float("inf"), "loss": float("inf"),
                "n_tokens": 0, "error": "empty completion"}

    # Convert to tensors
    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids_t)

    try:
        with torch.no_grad():
            # Forward pass — get logits
            out = model(input_ids_t, attention_mask=attn_mask)
            logits = out[0] if isinstance(out, tuple) else out

            if logits is None:
                return {"perplexity": float("inf"), "loss": float("inf"),
                        "n_tokens": len(completion_ids),
                        "error": "model returned None logits"}

            # Shift for causal LM: predict token t+1 from position t
            shift_logits = logits[:, :-1, :].contiguous()  # [1, T-1, V]
            shift_labels = labels_t[:, 1:].contiguous()     # [1, T-1]

            # Compute per-token CE (only over completion tokens)
            import torch.nn.functional as F
            ce_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="none",
            )

            # Mask: only count completion tokens (where label != -100)
            mask = (shift_labels.view(-1) != -100).float()
            n_tokens = int(mask.sum().item())

            if n_tokens == 0:
                return {"perplexity": float("inf"), "loss": float("inf"),
                        "n_tokens": 0, "error": "no completion tokens"}

            loss = (ce_per_token * mask).sum() / mask.sum()
            perplexity = math.exp(min(loss.item(), 20))  # cap to avoid overflow

            return {
                "perplexity": round(perplexity, 4),
                "loss": round(loss.item(), 4),
                "n_tokens": n_tokens,
                "error": "",
            }
    except Exception as e:
        return {"perplexity": float("inf"), "loss": float("inf"),
                "n_tokens": len(completion_ids), "error": str(e)[:200]}


def score_file(input_path: str, output_path: str | None = None,
               checkpoint: str | None = None,
               filter_data: bool = False,
               min_ppl: float = 5.0, max_ppl: float = 100.0,
               agentic: bool = False,
               device: str = "cuda") -> dict:
    """Score all pairs in a JSONL file and optionally filter by perplexity.

    Args:
        input_path: input JSONL file
        output_path: if given, write scored/filtered data here
        checkpoint: model checkpoint path
        filter_data: if True, only keep pairs in [min_ppl, max_ppl] range
        min_ppl: minimum perplexity to keep (filter out too-easy)
        max_ppl: maximum perplexity to keep (filter out too-hard/garbage)
        agentic: if True, score agentic trajectories (use messages format)
        device: cuda or cpu

    Returns:
        statistics dict
    """
    print(f"Loading ForgeLM model...")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device=device)
    print(f"Model loaded. Scoring {input_path}...")

    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs")

    scored = []
    stats = {
        "total": len(pairs),
        "scored": 0,
        "errors": 0,
        "filtered_out": 0,
        "perplexities": [],
    }

    for i, pair in enumerate(pairs):
        if agentic:
            # Agentic trajectory: use task + final_answer
            prompt = pair.get("task", "")
            completion = pair.get("final_answer", "")
            if not completion and pair.get("messages"):
                # Use last assistant message as completion
                for msg in reversed(pair["messages"]):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        completion = msg["content"]
                        break
        else:
            # Regular distill pair: prompt + solution
            prompt = pair.get("prompt", "")
            completion = pair.get("solution", "")

        if not prompt or not completion:
            stats["errors"] += 1
            continue

        result = compute_perplexity(model, tokenizer, prompt, completion, device)

        if result["error"]:
            stats["errors"] += 1
            if i < 5:
                print(f"  [{i+1}] Error: {result['error'][:80]}")
            continue

        stats["scored"] += 1
        stats["perplexities"].append(result["perplexity"])

        pair["perplexity"] = result["perplexity"]
        pair["ppl_loss"] = result["loss"]
        pair["ppl_tokens"] = result["n_tokens"]

        # Filter check
        keep = True
        if filter_data:
            if result["perplexity"] < min_ppl or result["perplexity"] > max_ppl:
                keep = False
                stats["filtered_out"] += 1

        if keep:
            scored.append(pair)

        if (i + 1) % 50 == 0 or i == len(pairs) - 1:
            avg_ppl = sum(stats["perplexities"]) / max(len(stats["perplexities"]), 1)
            print(f"  [{i+1}/{len(pairs)}] scored={stats['scored']} "
                  f"errors={stats['errors']} "
                  f"avg_ppl={avg_ppl:.2f} "
                  f"last_ppl={result['perplexity']:.2f}")

    # Compute statistics
    ppls = stats["perplexities"]
    if ppls:
        ppls_sorted = sorted(ppls)
        n = len(ppls_sorted)
        stats["ppl_min"] = round(ppls_sorted[0], 2)
        stats["ppl_p25"] = round(ppls_sorted[n // 4], 2)
        stats["ppl_median"] = round(ppls_sorted[n // 2], 2)
        stats["ppl_p75"] = round(ppls_sorted[3 * n // 4], 2)
        stats["ppl_p90"] = round(ppls_sorted[int(n * 0.9)], 2)
        stats["ppl_max"] = round(ppls_sorted[-1], 2)
        stats["ppl_mean"] = round(sum(ppls) / len(ppls), 2)

        # Goldilocks zone count
        goldilocks = sum(1 for p in ppls if min_ppl <= p <= max_ppl)
        stats["goldilocks_count"] = goldilocks
        stats["goldilocks_pct"] = round(goldilocks / len(ppls) * 100, 1)

    # Write output
    if output_path and scored:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for pair in scored:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        stats["output_path"] = str(out)
        stats["output_count"] = len(scored)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Score distillation data by perplexity using local ForgeLM model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (scored/filtered)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (auto-detects latest if None)")
    parser.add_argument("--filter", action="store_true",
                        help="Filter to Goldilocks perplexity zone")
    parser.add_argument("--min-ppl", type=float, default=5.0,
                        help="Min perplexity to keep (default: 5)")
    parser.add_argument("--max-ppl", type=float, default=100.0,
                        help="Max perplexity to keep (default: 100)")
    parser.add_argument("--agentic", action="store_true",
                        help="Score agentic trajectories (task + final_answer)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ForgeAI Perplexity Scorer")
    print("=" * 70)
    print(f"  Input:    {args.input}")
    print(f"  Output:   {args.output or 'none (score only)'}")
    print(f"  Filter:   {args.filter} (range: {args.min_ppl}-{args.max_ppl})")
    print(f"  Agentic:  {args.agentic}")
    print(f"  Device:   {args.device}")
    print()

    stats = score_file(
        input_path=args.input,
        output_path=args.output,
        checkpoint=args.checkpoint,
        filter_data=args.filter,
        min_ppl=args.min_ppl,
        max_ppl=args.max_ppl,
        agentic=args.agentic,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("  PERPLEXITY STATISTICS")
    print("=" * 70)
    print(f"  Total pairs:       {stats['total']}")
    print(f"  Scored:            {stats['scored']}")
    print(f"  Errors:            {stats['errors']}")
    if "ppl_mean" in stats:
        print(f"\n  Perplexity distribution:")
        print(f"    min:    {stats['ppl_min']}")
        print(f"    p25:    {stats['ppl_p25']}")
        print(f"    median: {stats['ppl_median']}")
        print(f"    p75:    {stats['ppl_p75']}")
        print(f"    p90:    {stats['ppl_p90']}")
        print(f"    max:    {stats['ppl_max']}")
        print(f"    mean:   {stats['ppl_mean']}")
        print(f"\n  Goldilocks zone ({args.min_ppl}-{args.max_ppl}):")
        print(f"    count:  {stats['goldilocks_count']}/{stats['scored']}")
        print(f"    pct:    {stats['goldilocks_pct']}%")
    if args.filter:
        print(f"\n  Filtered out:      {stats['filtered_out']}")
        print(f"  Kept:              {stats.get('output_count', 0)}")
    if "output_path" in stats:
        print(f"\n  Output saved to:   {stats['output_path']}")


if __name__ == "__main__":
    main()
