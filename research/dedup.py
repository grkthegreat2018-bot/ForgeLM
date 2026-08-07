"""Data deduplication for synthetic data using MinHash.

Removes near-duplicate samples from synthetic data files. Critical for
small model training: "Internal Data Repetition Destroys Language Models"
(2025) shows 100% duplication causes -40% accuracy, while <25% is benign.

Uses MinHash + LSH for near-duplicate detection (Jaccard similarity).
Exact duplicates are removed via hash comparison first (fast path).

Usage:
    from research.dedup import dedup_file, dedup_directory

    # Deduplicate a single file
    n_before, n_after = dedup_file("research/data/lmstudio_glm5.2.jsonl")

    # Deduplicate all lmstudio files
    dedup_directory("research/data/", pattern="lmstudio_*.jsonl")
"""
import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Set, Dict


def normalize_text(text: str) -> str:
    """Normalize text for deduplication: lowercase, strip whitespace, remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash(text: str) -> str:
    """Exact hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def shingle(text: str, k: int = 5) -> Set[str]:
    """Create k-shingles from normalized text.

    Args:
        text: input text
        k: shingle size (default 5 words)

    Returns:
        set of shingle strings
    """
    words = normalize_text(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


class MinHash:
    """MinHash for estimating Jaccard similarity between sets.

    Args:
        n_perm: number of permutations (hash functions)
    """

    def __init__(self, n_perm: int = 128):
        self.n_perm = n_perm
        # Generate random hash function parameters.
        import random
        random.seed(42)  # reproducible
        self.a = [random.randint(1, 2**32 - 1) for _ in range(n_perm)]
        self.b = [random.randint(0, 2**32 - 1) for _ in range(n_perm)]
        self.max_val = 2**32 - 1

    def hash(self, s: str, seed_idx: int) -> int:
        """Hash a string with the seed_idx-th hash function."""
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        return ((self.a[seed_idx] * h + self.b[seed_idx]) % self.max_val)

    def signature(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature for a set of shingles."""
        if not shingles:
            return [self.max_val] * self.n_perm
        sig = []
        for i in range(self.n_perm):
            sig.append(min(self.hash(s, i) for s in shingles))
        return sig

    def jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.n_perm


class LSHIndex:
    """Locality-Sensitive Hashing index for fast near-duplicate detection.

    Args:
        n_perm: number of MinHash permutations
        n_bands: number of LSH bands (more bands = more candidates, less precision)
        threshold: Jaccard similarity threshold for duplicates
    """

    def __init__(self, n_perm: int = 128, n_bands: int = 32, threshold: float = 0.8):
        self.minhash = MinHash(n_perm)
        self.n_perm = n_perm
        self.n_bands = n_bands
        self.rows_per_band = n_perm // n_bands
        self.threshold = threshold
        self.bands: List[Dict[Tuple, List[int]]] = [defaultdict(list) for _ in range(n_bands)]
        self.signatures: List[List[int]] = []

    def add(self, idx: int, shingles: Set[str]):
        """Add a document to the LSH index."""
        sig = self.minhash.signature(shingles)
        self.signatures.append(sig)
        for b in range(self.n_bands):
            band = tuple(sig[b * self.rows_per_band:(b + 1) * self.rows_per_band])
            self.bands[b][band].append(idx)

    def query(self, shingles: Set[str]) -> List[int]:
        """Find candidate duplicates for a document."""
        sig = self.minhash.signature(shingles)
        candidates = set()
        for b in range(self.n_bands):
            band = tuple(sig[b * self.rows_per_band:(b + 1) * self.rows_per_band])
            if band in self.bands[b]:
                candidates.update(self.bands[b][band])
        # Verify with actual Jaccard estimation.
        confirmed = []
        for idx in candidates:
            if self.minhash.jaccard(sig, self.signatures[idx]) >= self.threshold:
                confirmed.append(idx)
        return confirmed


def dedup_file(filepath: str, threshold: float = 0.8,
               keep_first: bool = True, dry_run: bool = False) -> Tuple[int, int]:
    """Deduplicate a JSONL file.

    Args:
        filepath: path to JSONL file
        threshold: Jaccard similarity threshold (0.8 = 80% similar = duplicate)
        keep_first: if True, keep the first occurrence; else keep the longest
        dry_run: if True, only report stats without modifying the file

    Returns:
        (n_before, n_after) sample counts
    """
    filepath = Path(filepath)
    print(f"\n[Dedup] Processing {filepath.name}...")

    # Read all samples.
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    n_before = len(samples)
    print(f"  Read {n_before} samples")

    # Phase 1: exact dedup via hash.
    seen_hashes: Set[str] = set()
    exact_dup_indices = set()
    for i, s in enumerate(samples):
        text = s.get("prompt", "") + " " + s.get("completion", "")
        h = text_hash(text)
        if h in seen_hashes:
            exact_dup_indices.add(i)
        else:
            seen_hashes.add(h)

    print(f"  Exact duplicates: {len(exact_dup_indices)}")

    # Phase 2: near-duplicate dedup via MinHash + LSH.
    lsh = LSHIndex(threshold=threshold)
    near_dup_indices = set()

    for i, s in enumerate(samples):
        if i in exact_dup_indices:
            continue
        text = s.get("prompt", "") + " " + s.get("completion", "")
        shingles = shingle(text)
        dups = lsh.query(shingles)
        if dups:
            # Found near-duplicates. Keep the first (or longest).
            if keep_first:
                near_dup_indices.add(i)
            else:
                # Compare lengths, keep the longer one.
                dup_text = samples[dups[0]].get("prompt", "") + samples[dups[0]].get("completion", "")
                cur_text = s.get("prompt", "") + s.get("completion", "")
                if len(cur_text) > len(dup_text):
                    near_dup_indices.add(dups[0])  # mark old as dup
                else:
                    near_dup_indices.add(i)  # mark current as dup
        lsh.add(i, shingles)

    print(f"  Near-duplicates (Jaccard >= {threshold}): {len(near_dup_indices)}")

    # Remove duplicates.
    all_dups = exact_dup_indices | near_dup_indices
    kept = [s for i, s in enumerate(samples) if i not in all_dups]
    n_after = len(kept)
    print(f"  Kept: {n_after}/{n_before} ({n_before - n_after} removed)")

    if not dry_run and n_after < n_before:
        # Write back.
        backup = filepath.with_suffix(".jsonl.bak")
        filepath.rename(backup)
        with open(filepath, "w", encoding="utf-8") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Written {filepath} (backup: {backup.name})")

    return n_before, n_after


def dedup_directory(dirpath: str, pattern: str = "*.jsonl",
                    threshold: float = 0.8, dry_run: bool = False):
    """Deduplicate all matching files in a directory.

    Also performs cross-file deduplication (samples appearing in multiple files).

    Args:
        dirpath: directory containing JSONL files
        pattern: glob pattern for files
        threshold: Jaccard similarity threshold
        dry_run: if True, only report stats
    """
    dirpath = Path(dirpath)
    files = list(dirpath.glob(pattern))
    print(f"[Dedup] Found {len(files)} files matching {pattern}")

    # Cross-file dedup: build a global LSH index.
    # Use fewer permutations for speed (32 is sufficient for 0.8 threshold).
    lsh = LSHIndex(n_perm=32, n_bands=8, threshold=threshold)
    global_seen: Set[str] = set()
    cross_file_dups = 0

    # First pass: index all samples.
    file_samples = {}
    for f in files:
        samples = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines (e.g., bad escape sequences).
                        continue
        file_samples[f] = samples

    # Process each file.
    total_before = 0
    total_after = 0
    for f in files:
        samples = file_samples[f]
        kept = []
        for s in samples:
            text = s.get("prompt", "") + " " + s.get("completion", "")
            h = text_hash(text)
            if h in global_seen:
                cross_file_dups += 1
                continue
            global_seen.add(h)

            shingles = shingle(text)
            dups = lsh.query(shingles)
            if dups:
                cross_file_dups += 1
                continue
            lsh.add(len(global_seen) - 1, shingles)
            kept.append(s)

        total_before += len(samples)
        total_after += len(kept)

        if not dry_run and len(kept) < len(samples):
            with open(f, "w", encoding="utf-8") as fh:
                for s in kept:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"  {f.name}: {len(samples)} → {len(kept)}")

    print(f"\n[Dedup] Total: {total_before} → {total_after} "
          f"({total_before - total_after} removed, {cross_file_dups} cross-file dups)")
