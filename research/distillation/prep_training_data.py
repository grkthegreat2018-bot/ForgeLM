"""Prepare all downloaded HF datasets + distillation data for SFT training.

Converts all data to the unified format expected by sft_train.py:
  {"prompt": "...", "response": "..."}

Or for multi-turn (agentic):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Then shuffles, deduplicates, and splits into train/val sets.

Usage:
    python -m research.distillation.prep_training_data
    python -m research.distillation.prep_training_data --max-per-category 30000
    python -m research.distillation.prep_training_data --val-frac 0.02
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("PYTHONUTF8", "1")

# ── Config ──

HF_DATASETS_DIR = Path("research/distillation/hf_datasets")
DISTILL_DIR = Path("research/distillation")
OUTPUT_DIR = Path("research/data/finetune")

# Category weights — how much of each category to include
# Code-heavy mix: the model needs to write code, not call tools
# tool_use EXCLUDED — Hermes FC data caused tool-call hallucinations
CATEGORY_WEIGHTS = {
    "code": 1.0,       # Core — code generation (highest priority)
    "reasoning": 0.6,  # Useful but don't dominate
    "math": 0.5,       # Some math, but not the focus
    "logic": 0.6,      # Logical reasoning
    "planning": 0.4,   # Planning (smaller dataset anyway)
    "tool_use": 0.0,   # EXCLUDED — caused tool-call hallucinations
    "general": 0.2,    # Minimal general instruction following
    "distill": 2.0,    # Our own verified data — highest value
}

# Datasets to completely exclude (by filename in hf_datasets/)
EXCLUDE_DATASETS = {
    "hermes_fc.jsonl",       # Tool calls contaminate code generation
    "xlam_fc.jsonl",         # Same issue (if present)
}

# Min/max solution lengths (characters) — filter trivial and overly long
MIN_SOLUTION_LEN = 20
MAX_SOLUTION_LEN = 8000
MIN_PROMPT_LEN = 10
MAX_PROMPT_LEN = 4000


def load_hf_datasets(max_per_category: int = 50000) -> dict[str, list[dict]]:
    """Load all HF dataset JSONL files, grouped by category."""
    by_category: dict[str, list[dict]] = defaultdict(list)

    if not HF_DATASETS_DIR.exists():
        print(f"  Warning: {HF_DATASETS_DIR} not found")
        return by_category

    for f in sorted(HF_DATASETS_DIR.glob("*.jsonl")):
        if f.name == "download_stats.json":
            continue
        if f.name in EXCLUDE_DATASETS:
            print(f"  {f.name:35s} → SKIPPED (excluded)")
            continue
        count = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                category = obj.get("category", "general")
                prompt = obj.get("prompt", "")
                solution = obj.get("solution", "")

                # Length filtering
                if len(prompt) < MIN_PROMPT_LEN or len(prompt) > MAX_PROMPT_LEN:
                    continue
                if len(solution) < MIN_SOLUTION_LEN or len(solution) > MAX_SOLUTION_LEN:
                    continue

                by_category[category].append({
                    "prompt": prompt,
                    "response": solution,
                    "source": obj.get("source", f.name),
                })
                count += 1
        print(f"  {f.name:35s} → {count:6d} examples [{category}]")

    return by_category


def load_distill_data() -> list[dict]:
    """Load our own distillation data (highest priority)."""
    examples = []
    for fname in ["distill_data.jsonl", "distill_data_run2.jsonl"]:
        path = DISTILL_DIR / fname
        if path.exists():
            count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    prompt = obj.get("prompt", "")
                    solution = obj.get("solution", "")
                    if len(prompt) < MIN_PROMPT_LEN or len(solution) < MIN_SOLUTION_LEN:
                        continue
                    examples.append({
                        "prompt": prompt,
                        "response": solution,
                        "source": "distill",
                    })
                    count += 1
            print(f"  {fname:35s} → {count:6d} examples [distill]")
    return examples


def load_agentic_data() -> list[dict]:
    """Load agentic trajectories as multi-turn messages.

    NOTE: Agentic data contains tool-call trajectories which caused
    tool-call hallucinations in V3 SFT. Disabled by default.
    Set INCLUDE_AGENTIC=True to re-enable.
    """
    INCLUDE_AGENTIC = False  # Tool-call contamination fix
    if not INCLUDE_AGENTIC:
        print(f"  agentic_distill_data.jsonl            → SKIPPED (tool-call contamination)")
        return []

    examples = []
    path = DISTILL_DIR / "agentic_distill_data.jsonl"
    if path.exists():
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                task = obj.get("task", "")
                final_answer = obj.get("final_answer", "")
                messages = obj.get("messages", [])

                if messages and isinstance(messages, list):
                    # Use the full multi-turn trajectory
                    examples.append({"messages": messages, "source": "agentic"})
                    count += 1
                elif task and final_answer:
                    examples.append({
                        "prompt": task,
                        "response": final_answer,
                        "source": "agentic",
                    })
                    count += 1
        print(f"  agentic_distill_data.jsonl            → {count:6d} examples [agentic]")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove exact duplicates based on prompt content."""
    seen = set()
    deduped = []
    for ex in examples:
        if "messages" in ex:
            key = json.dumps(ex["messages"], ensure_ascii=False, sort_keys=True)[:500]
        else:
            key = ex.get("prompt", "")[:500]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def balance_and_sample(by_category: dict[str, list[dict]],
                       distill_examples: list[dict],
                       agentic_examples: list[dict],
                       max_per_category: int = 50000) -> list[dict]:
    """Balance categories according to weights and cap per-category."""
    all_examples = []

    # Add distill data (highest weight — include all)
    distill_capped = distill_examples[:max_per_category]
    all_examples.extend(distill_capped)
    print(f"\n  distill:    {len(distill_capped):6d} (weight=2.0, all included)")

    # Add agentic data (all included)
    agentic_capped = agentic_examples[:max_per_category]
    all_examples.extend(agentic_capped)
    print(f"  agentic:    {len(agentic_capped):6d} (all included)")

    # Add HF datasets by category with weights
    for category, examples in sorted(by_category.items()):
        weight = CATEGORY_WEIGHTS.get(category, 0.5)
        cap = int(max_per_category * weight)
        # Shuffle and cap
        random.shuffle(examples)
        capped = examples[:cap]
        all_examples.extend(capped)
        print(f"  {category:10s}  {len(capped):6d} (weight={weight}, cap={cap})")

    return all_examples


def split_train_val(examples: list[dict], val_frac: float = 0.02
                    ) -> tuple[list[dict], list[dict]]:
    """Split into train and validation sets."""
    random.shuffle(examples)
    n_val = max(100, int(len(examples) * val_frac))
    val = examples[:n_val]
    train = examples[n_val:]
    return train, val


def save_jsonl(examples: list[dict], path: Path) -> None:
    """Save examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(examples):,} examples to {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for ForgeLM V3 SFT")
    parser.add_argument("--max-per-category", type=int, default=50000,
                        help="Max examples per category (default: 50000)")
    parser.add_argument("--val-frac", type=float, default=0.02,
                        help="Validation set fraction (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("  ForgeLM V3 — Training Data Preparation")
    print("=" * 70)
    print()

    # ── Load all data ──
    print("Loading HF datasets:")
    by_category = load_hf_datasets(max_per_category=args.max_per_category)

    print("\nLoading distillation data:")
    distill_examples = load_distill_data()

    print("\nLoading agentic trajectories:")
    agentic_examples = load_agentic_data()

    # ── Balance and sample ──
    print("\n" + "-" * 70)
    print("  Balancing categories")
    print("-" * 70)
    all_examples = balance_and_sample(
        by_category, distill_examples, agentic_examples,
        max_per_category=args.max_per_category)

    # ── Deduplicate ──
    print(f"\n  Total before dedup: {len(all_examples):,}")
    all_examples = deduplicate(all_examples)
    print(f"  Total after dedup:  {len(all_examples):,}")

    # ── Split ──
    train, val = split_train_val(all_examples, val_frac=args.val_frac)

    # ── Save ──
    print("\n" + "-" * 70)
    print("  Saving")
    print("-" * 70)
    train_path = OUTPUT_DIR / "forgelm_v7_train.jsonl"
    val_path = OUTPUT_DIR / "forgelm_v7_val.jsonl"
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)

    # ── Stats ──
    print("\n" + "=" * 70)
    print("  FINAL STATISTICS")
    print("=" * 70)
    print(f"  Total examples:  {len(all_examples):,}")
    print(f"  Train:           {len(train):,}")
    print(f"  Validation:      {len(val):,}")

    # Count by source
    source_counts = defaultdict(int)
    for ex in all_examples:
        source = ex.get("source", "unknown")
        source_counts[source] += 1
    print(f"\n  By source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {src:30s} {count:6d}")

    # Count types
    n_single = sum(1 for ex in all_examples if "prompt" in ex)
    n_multi = sum(1 for ex in all_examples if "messages" in ex)
    print(f"\n  Single-turn:     {n_single:,}")
    print(f"  Multi-turn:      {n_multi:,}")

    # Estimate tokens
    total_chars = 0
    for ex in all_examples:
        if "messages" in ex:
            for msg in ex["messages"]:
                total_chars += len(msg.get("content", ""))
        else:
            total_chars += len(ex.get("prompt", ""))
            total_chars += len(ex.get("response", ""))
    est_tokens = total_chars // 4
    print(f"\n  Est. tokens:     {est_tokens:,}")
    print(f"  Est. epochs at batch=1, seq=1024:  {est_tokens // 1024:,} steps")


if __name__ == "__main__":
    main()
