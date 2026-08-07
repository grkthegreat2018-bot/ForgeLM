"""Reformat synthetic JSONL data into chat-template format with CoT augmentation.

Converts {prompt, completion} pairs into {messages: [{role:user,...}, {role:assistant,...}]}
format, and optionally prepends "Let's think step by step." to reasoning/math completions
to enable chain-of-thought distillation (SCoTD, ACL 2023).

The chat-template format is required for proper instruction tuning:
  - Without role markers, the model treats user+assistant text as one blob and
    learns to continue rather than answer (rambling behavior).
  - With Qwen chat template (<|im_start|>user ... <|im_end|> / <|im_start|>assistant ...),
    the model learns the boundary between "being asked" and "answering".

Usage:
    python -m research.reformat_chat --input research/data/all_teachers_v2_scored.jsonl \
        --output research/data/all_teachers_v2_chat.jsonl --cot-domains reasoning math
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


# CoT prefixes by domain. SCoTD paper: diverse reasoning chains from teacher
# dramatically help small models (125M-1.3B) learn step-by-step reasoning.
COT_PREFIXES = {
    "reasoning": "Let's think step by step.\n",
    "math": "Let's solve this step by step.\n",
    "coding": "Let's work through this step by step.\n",
}


def reformat_sample(sample, cot_domains=None):
    """Convert one {prompt, completion, domain?} sample to chat format.

    Returns: {"messages": [...], "domain": ..., "teacher": ...} or None if invalid.
    """
    cot_domains = cot_domains or set()
    prompt = sample.get("prompt", "")
    completion = sample.get("completion", "")
    domain = sample.get("domain", "")
    teacher = sample.get("teacher", "")

    if not prompt or not completion:
        return None

    # CoT augmentation: prepend step-by-step reasoning prefix for hard domains.
    # This teaches the model to verbalize its reasoning before answering
    # (Wei et al. 2022, SCoTD Mukherjee et al. 2023).
    if domain in cot_domains and domain in COT_PREFIXES:
        prefix = COT_PREFIXES[domain]
        # Don't double-prefix if completion already starts with similar phrasing.
        if not completion.lower().startswith(("let's", "step by step", "first,", "to solve")):
            completion = prefix + completion

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "domain": domain,
        "teacher": teacher,
        # Preserve quality score if present (for curriculum ordering).
        "_quality_score": sample.get("_quality_score"),
    }


def main():
    p = argparse.ArgumentParser(description="Reformat synthetic data to chat-template format with CoT")
    p.add_argument("--input", required=True, help="Input JSONL file(s)", nargs="+")
    p.add_argument("--output", required=True, help="Output JSONL file (chat format)")
    p.add_argument("--cot-domains", nargs="*", default=["reasoning", "math"],
                   help="Domains to augment with CoT prefix (default: reasoning math)")
    p.add_argument("--min-completion-len", type=int, default=20,
                   help="Skip samples with completions shorter than this (chars)")
    args = p.parse_args()

    cot_set = set(args.cot_domains)
    total_in = 0
    total_out = 0
    cot_count = 0
    domain_counts = {}

    with open(args.output, "w", encoding="utf-8") as out_f:
        for fp in args.input:
            path = Path(fp)
            if not path.exists():
                print(f"  Warning: {fp} not found, skipping")
                continue

            file_in = 0
            file_out = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_in += 1
                    file_in += 1
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Skip if already in messages format (idempotent: re-wrap).
                    if "messages" in sample and "prompt" not in sample:
                        # Already chat format — just pass through.
                        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        total_out += 1
                        file_out += 1
                        d = sample.get("domain", "unknown")
                        domain_counts[d] = domain_counts.get(d, 0) + 1
                        continue

                    if len(sample.get("completion", "")) < args.min_completion_len:
                        continue

                    reformatted = reformat_sample(sample, cot_set)
                    if reformatted is None:
                        continue

                    # Track CoT augmentation.
                    d = reformatted["domain"]
                    if d in cot_set and not sample.get("completion", "").lower().startswith(
                        ("let's", "step by step", "first,", "to solve")
                    ):
                        cot_count += 1

                    out_f.write(json.dumps(reformatted, ensure_ascii=False) + "\n")
                    total_out += 1
                    file_out += 1
                    domain_counts[d] = domain_counts.get(d, 0) + 1

            print(f"  {path.name}: {file_in} in -> {file_out} out")

    print(f"\nTotal: {total_in} in -> {total_out} out")
    print(f"CoT augmentations: {cot_count} samples (domains: {cot_set})")
    print(f"By domain: {domain_counts}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
