"""Merge all multi-teacher synthetic data into a single deduplicated file.

Combines lmstudio_*.jsonl files from all teachers into one training-ready
file, with optional deduplication and domain balancing.

Usage:
    python -m research.merge_synthetic --output research/data/all_teachers.jsonl
    python -m research.merge_synthetic --output research/data/all_teachers.jsonl --dedup --balance
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def merge_files(input_files, output_file, dedup=False, balance=False,
                max_per_domain=None):
    """Merge multiple JSONL files into one.

    Args:
        input_files: list of input JSONL paths
        output_file: output JSONL path
        dedup: if True, remove exact duplicates
        balance: if True, balance domains (cap per-domain count)
        max_per_domain: max samples per domain if balancing
    """
    all_samples = []
    seen_hashes = set()
    domain_counts = Counter()
    teacher_counts = Counter()
    skipped = 0

    for fpath in input_files:
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue

        teacher_name = fpath.stem.replace("lmstudio_", "")
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # Add teacher metadata.
                sample["teacher"] = teacher_name

                # Infer domain if missing (for older data without domain field).
                if "domain" not in sample:
                    prompt_lower = sample.get("prompt", "").lower()
                    if any(kw in prompt_lower for kw in ["code", "function", "implement", "python", "algorithm", "program", "debug"]):
                        sample["domain"] = "coding"
                    elif any(kw in prompt_lower for kw in ["math", "calculate", "solve", "equation", "prove", "integral", "derivative"]):
                        sample["domain"] = "math"
                    elif any(kw in prompt_lower for kw in ["write", "poem", "story", "essay", "creative", "describe", "narrative"]):
                        sample["domain"] = "writing"
                    elif any(kw in prompt_lower for kw in ["explain", "why", "how", "what", "reason", "analyze", "logic"]):
                        sample["domain"] = "reasoning"
                    else:
                        sample["domain"] = "knowledge"

                # Dedup check.
                if dedup:
                    text = sample.get("prompt", "") + " " + sample.get("completion", "")
                    h = hash(text)
                    if h in seen_hashes:
                        skipped += 1
                        continue
                    seen_hashes.add(h)

                domain = sample.get("domain", "unknown")
                domain_counts[domain] += 1
                teacher_counts[teacher_name] += 1
                all_samples.append(sample)
                count += 1

        print(f"  {fpath.name}: {count} samples")

    # Domain balancing: cap samples per domain.
    if balance and max_per_domain:
        from collections import defaultdict
        import random
        random.seed(42)
        by_domain = defaultdict(list)
        for s in all_samples:
            by_domain[s.get("domain", "unknown")].append(s)
        balanced = []
        for domain, samples in by_domain.items():
            if len(samples) > max_per_domain:
                random.shuffle(samples)
                samples = samples[:max_per_domain]
            balanced.extend(samples)
        all_samples = balanced
        print(f"  Balanced to {max_per_domain}/domain: {len(all_samples)} total")

    # Write merged file.
    with open(output_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nMerged: {len(all_samples)} samples → {output_file}")
    print(f"Skipped: {skipped}")
    print(f"\nBy teacher:")
    for t, c in teacher_counts.most_common():
        print(f"  {t}: {c}")
    print(f"\nBy domain:")
    for d, c in domain_counts.most_common():
        print(f"  {d}: {c}")

    return all_samples


def main():
    p = argparse.ArgumentParser(description="Merge multi-teacher synthetic data")
    p.add_argument("--data-dir", default="research/data")
    p.add_argument("--pattern", default="lmstudio_*.jsonl")
    p.add_argument("--output", default="research/data/all_teachers.jsonl")
    p.add_argument("--dedup", action="store_true", help="Remove exact duplicates")
    p.add_argument("--balance", action="store_true", help="Balance domains")
    p.add_argument("--max-per-domain", type=int, default=None,
                   help="Max samples per domain (for --balance)")
    args = p.parse_args()

    input_files = sorted(Path(args.data_dir).glob(args.pattern))
    if not input_files:
        print(f"No files found in {args.data_dir} matching {args.pattern}")
        return

    print(f"Merging {len(input_files)} files:")
    merge_files(input_files, args.output, dedup=args.dedup,
                balance=args.balance, max_per_domain=args.max_per_domain)


if __name__ == "__main__":
    main()
