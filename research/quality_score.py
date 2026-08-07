"""Data quality scoring for synthetic data filtering.

Scores synthetic data samples on multiple dimensions and filters out
low-quality ones. Critical for small models: bad data hurts more when
you have less of it.

Scoring dimensions:
1. Length: too short (<50 chars) or too long (>5000 chars) = low quality
2. Diversity: lexical diversity (unique words / total words)
3. Coherence: sentence length variance (very uniform = robotic)
4. Repetition: n-gram repetition penalty
5. Code validity: for coding samples, check if code blocks are present
6. Prompt-response alignment: check if response addresses the prompt

Usage:
    from research.quality_score import score_file, score_sample

    # Score and filter a file
    stats = score_file("research/data/all_teachers.jsonl",
                       output="research/data/all_teachers_scored.jsonl",
                       min_score=0.5)
"""
import json
import re
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def lexical_diversity(text: str) -> float:
    """Type-token ratio (unique words / total words). Higher = more diverse."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def sentence_length_variance(text: str) -> float:
    """Variance in sentence lengths. Low variance = robotic, high = natural."""
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(variance) / (mean + 1)  # normalized CV


def ngram_repetition(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams. Lower = better."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def has_code_blocks(text: str) -> bool:
    """Check if text contains code blocks (``` or indented code)."""
    return bool(re.search(r'```|^\s{4,}\w', text, re.MULTILINE))


def prompt_response_overlap(prompt: str, response: str) -> float:
    """Check if response addresses prompt keywords. Higher = more relevant."""
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    if not prompt_words:
        return 0.5  # neutral
    overlap = len(prompt_words & response_words) / len(prompt_words)
    return min(overlap, 1.0)


def score_sample(sample: Dict) -> Dict:
    """Score a single synthetic data sample.

    Args:
        sample: dict with 'prompt' and 'completion' keys

    Returns:
        dict with per-dimension scores and overall score (0-1)
    """
    prompt = sample.get("prompt", "")
    completion = sample.get("completion", "")
    full_text = prompt + " " + completion
    domain = sample.get("domain", "unknown")

    # 1. Length score (optimal range: 100-3000 chars).
    comp_len = len(completion)
    if comp_len < 50:
        length_score = 0.1
    elif comp_len < 100:
        length_score = 0.3
    elif comp_len <= 3000:
        length_score = 1.0
    elif comp_len <= 5000:
        length_score = 0.7
    else:
        length_score = 0.3

    # 2. Lexical diversity (TTR).
    ttr = lexical_diversity(completion)
    diversity_score = min(ttr * 2, 1.0)  # TTR ~0.5 is good for long text

    # 3. Coherence (sentence length variance).
    slv = sentence_length_variance(completion)
    coherence_score = min(slv * 3, 1.0)  # CV ~0.3+ is natural

    # 4. Repetition penalty.
    rep3 = ngram_repetition(completion, n=3)
    repetition_score = 1.0 - min(rep3 * 5, 1.0)  # 20% repetition = 0 score

    # 5. Code validity (for coding domain).
    if domain == "coding":
        code_score = 1.0 if has_code_blocks(completion) else 0.3
    else:
        code_score = 0.8  # neutral for non-code

    # 6. Prompt-response alignment.
    alignment = prompt_response_overlap(prompt, completion)

    # Weighted overall score.
    weights = {
        "length": 0.15,
        "diversity": 0.20,
        "coherence": 0.15,
        "repetition": 0.25,
        "code": 0.10,
        "alignment": 0.15,
    }
    scores = {
        "length": length_score,
        "diversity": diversity_score,
        "coherence": coherence_score,
        "repetition": repetition_score,
        "code": code_score,
        "alignment": alignment,
    }
    overall = sum(scores[k] * weights[k] for k in weights)

    return {
        "scores": scores,
        "overall": overall,
        "length": comp_len,
    }


def score_file(input_path: str, output_path: str = None,
               min_score: float = 0.4, dry_run: bool = False) -> Dict:
    """Score and optionally filter a JSONL file.

    Args:
        input_path: input JSONL file
        output_path: output JSONL file (filtered). If None, only reports stats.
        min_score: minimum overall score to keep (default 0.4)
        dry_run: if True, don't write output file

    Returns:
        dict with statistics
    """
    input_path = Path(input_path)
    print(f"\n[Quality] Scoring {input_path.name}...")

    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"  Read {len(samples)} samples")

    scored = []
    score_distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                          "0.6-0.8": 0, "0.8-1.0": 0}
    domain_scores = {}
    kept = 0
    filtered = 0

    for s in samples:
        result = score_sample(s)
        overall = result["overall"]
        s["_quality_score"] = overall
        s["_quality_scores"] = result["scores"]

        # Track distribution.
        bucket = f"{overall:.1f}"[:3]
        if overall < 0.2:
            score_distribution["0.0-0.2"] += 1
        elif overall < 0.4:
            score_distribution["0.2-0.4"] += 1
        elif overall < 0.6:
            score_distribution["0.4-0.6"] += 1
        elif overall < 0.8:
            score_distribution["0.6-0.8"] += 1
        else:
            score_distribution["0.8-1.0"] += 1

        # Track by domain.
        domain = s.get("domain", "unknown")
        if domain not in domain_scores:
            domain_scores[domain] = []
        domain_scores[domain].append(overall)

        if overall >= min_score:
            scored.append(s)
            kept += 1
        else:
            filtered += 1

    # Print stats.
    print(f"  Score distribution: {score_distribution}")
    print(f"  Kept: {kept} | Filtered: {filtered} (min_score={min_score})")
    print(f"  Average scores by domain:")
    for domain, scores in sorted(domain_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"    {domain}: avg={avg:.3f} (n={len(scores)})")

    # Write filtered output.
    if output_path and not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for s in scored:
                # Remove score fields before writing (or keep them for analysis).
                # Keep them — useful for downstream analysis.
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Written {output_path}")

    return {
        "total": len(samples),
        "kept": kept,
        "filtered": filtered,
        "distribution": score_distribution,
        "domain_scores": {d: sum(v) / len(v) for d, v in domain_scores.items()},
    }


def curriculum_order(input_path: str, output_path: str,
                     strategy: str = "length_asc") -> int:
    """Reorder samples for curriculum learning (easy → hard).

    Args:
        input_path: input JSONL
        output_path: output JSONL (reordered)
        strategy: ordering strategy
            - "length_asc": shortest first (easiest)
            - "length_desc": longest first (hardest)
            - "score_asc": lowest quality first
            - "score_desc": highest quality first
            - "difficulty": combined difficulty score (long + complex = hard)

    Returns:
        number of samples written
    """
    input_path = Path(input_path)
    print(f"\n[Curriculum] Ordering {input_path.name} by {strategy}...")

    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Compute ordering key for each sample.
    def difficulty_key(s):
        comp = s.get("completion", "")
        if strategy == "length_asc":
            return len(comp)
        elif strategy == "length_desc":
            return -len(comp)
        elif strategy == "score_asc":
            return s.get("_quality_score", score_sample(s)["overall"])
        elif strategy == "score_desc":
            return -s.get("_quality_score", score_sample(s)["overall"])
        elif strategy == "difficulty":
            # Combined: long + low diversity + high repetition = hard
            result = score_sample(s)
            length = len(comp)
            diversity = result["scores"]["diversity"]
            repetition = 1 - result["scores"]["repetition"]
            return length * (1 - diversity) * (1 + repetition)
        return 0

    samples.sort(key=difficulty_key)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Ordered {len(samples)} samples → {output_path}")
    return len(samples)
