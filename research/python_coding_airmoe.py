"""Download Python coding training data, process into AirMoE knowledge module.

Pipeline:
  1. Download flytech/python-codes-25k from HuggingFace (25K Python tasks)
  2. Parse into structured (instruction, code, output) tuples
  3. Classify by topic (math, strings, algorithms, data structures, etc.)
  4. Build knowledge packs per topic (text chunks for KV cache injection)
  5. Save as AirMoE module on D: drive

The module can be loaded at inference:
  - Router: classify user query → pick relevant topic expert
  - Expert: load that topic's knowledge pack from disk
  - Inject: KV cache injection gives the model the knowledge at zero token cost

All output on D: drive.
"""
import os
import sys
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# All paths on D:
OUTPUT_DIR = "D:/windsurf/ForgeAI/research/checkpoints/python_coding_airmoe"
CACHE_DIR = "D:/windsurf/ForgeAI/.devin/tmp/hf_cache"


def classify_topic(instruction: str, code: str) -> str:
    """Classify a Python task into a topic based on keywords."""
    text = (instruction + " " + code).lower()

    # Math / numbers
    if any(k in text for k in ["fibonacci", "factorial", "prime", "gcd", "lcm",
                                 "sqrt", "power", "exponent", "logarithm",
                                 "trigon", "sin", "cos", "tan", "pi", "euler",
                                 "matrix", "determinant", "eigenvalue",
                                 "integral", "derivative", "calculus",
                                 "sum of", "product of", "average", "mean",
                                 "median", "variance", "standard deviation",
                                 "quadratic", "linear equation", "polynomial"]):
        return "math"

    # String manipulation
    if any(k in text for k in ["string", "reverse", "palindrome", "anagram",
                                 "substring", "concat", "split", "join",
                                 "replace", "uppercase", "lowercase", "strip",
                                 "char", "vowel", "consonant", "word count",
                                 "character", "regex", "pattern"]):
        return "strings"

    # Data structures
    if any(k in text for k in ["list", "array", "dict", "dictionary", "set",
                                 "tuple", "stack", "queue", "deque",
                                 "linked list", "tree", "graph", "heap",
                                 "hashmap", "hash table", "node", "binary"]):
        return "data_structures"

    # Algorithms
    if any(k in text for k in ["sort", "search", "binary search", "bubble",
                                 "quicksort", "merge sort", "insertion",
                                 "recursion", "recursive", "dynamic programming",
                                 "backtrack", "greedy", "divide and conquer",
                                 "complexity", "big o", "traverse", "bfs", "dfs"]):
        return "algorithms"

    # File I/O
    if any(k in text for k in ["file", "open", "read", "write", "csv", "json",
                                 "pickle", "path", "directory", "os.",
                                 "pathlib", "filesystem"]):
        return "file_io"

    # OOP / classes
    if any(k in text for k in ["class", "object", "inheritance", "polymorphism",
                                 "encapsulation", "method", "constructor",
                                 "self.", "__init__", "abstract", "interface"]):
        return "oop"

    # Web / network
    if any(k in text for k in ["http", "request", "url", "api", "flask",
                                 "django", "fastapi", "socket", "server",
                                 "client", "endpoint", "rest"]):
        return "web"

    # Data science
    if any(k in text for k in ["numpy", "pandas", "matplotlib", "scipy",
                                 "dataframe", "series", "plot", "chart",
                                 "machine learning", "model", "train",
                                 "sklearn", "tensorflow", "pytorch"]):
        return "data_science"

    # Date / time
    if any(k in text for k in ["date", "time", "datetime", "calendar",
                                 "timezone", "timestamp", "schedule"]):
        return "datetime"

    # Error handling
    if any(k in text for k in ["try", "except", "error", "exception",
                                 "raise", "assert", "debug", "traceback"]):
        return "error_handling"

    # Default
    return "general"


def download_dataset() -> List[Dict]:
    """Download flytech/python-codes-25k from HuggingFace."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = CACHE_DIR

    print("  Downloading flytech/python-codes-25k...")
    from datasets import load_dataset

    ds = load_dataset("flytech/python-codes-25k", split="train",
                      cache_dir=CACHE_DIR)

    samples = []
    for row in ds:
        instruction = row.get("instruction", "")
        code = row.get("output", "")
        text = row.get("text", "")

        if not instruction or not code:
            continue

        # Clean up
        instruction = instruction.strip()
        code = code.strip()

        if len(instruction) < 10 or len(code) < 20:
            continue

        topic = classify_topic(instruction, code)

        samples.append({
            "instruction": instruction,
            "code": code,
            "topic": topic,
            "text": text[:500] if text else "",
        })

    print(f"  Downloaded {len(samples)} samples")
    return samples


def process_into_packs(samples: List[Dict],
                       max_per_topic: int = 500,
                       chunk_size: int = 400) -> Dict[str, List[Dict]]:
    """Group samples by topic and build knowledge chunks.

    Each chunk is a self-contained knowledge unit that can be:
      - Fed to KnowledgePack.from_text() for KV cache injection
      - Used as context for ContextPatchKey
      - Converted to fact vectors for FactInjectionKey
    """
    # Group by topic
    by_topic: Dict[str, List[Dict]] = {}
    for s in samples:
        topic = s["topic"]
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(s)

    # Build chunks per topic
    topic_chunks: Dict[str, List[Dict]] = {}
    for topic, items in by_topic.items():
        # Sort by code length (longer = more detailed)
        items.sort(key=lambda x: len(x["code"]), reverse=True)
        # Take top N
        items = items[:max_per_topic]

        chunks = []
        for i, item in enumerate(items):
            # Build knowledge text: instruction + code
            knowledge = (
                f"# Task: {item['instruction']}\n"
                f"# Solution:\n{item['code']}\n"
            )

            chunk_hash = hashlib.md5(knowledge.encode()).hexdigest()[:8]

            chunks.append({
                "text": knowledge,
                "topic": topic,
                "source": f"python-codes-25k",
                "hash": chunk_hash,
                "instruction": item["instruction"][:100],
                "code_length": len(item["code"]),
            })

        topic_chunks[topic] = chunks
        print(f"    {topic}: {len(chunks)} chunks")

    return topic_chunks


def build_airmoe_module(topic_chunks: Dict[str, List[Dict]]) -> str:
    """Build the AirMoE knowledge module on D: drive.

    File layout:
      python_coding_airmoe/
        module.json           # config + metadata
        all_knowledge.json    # all topics in one file
        topic_<name>.json     # per-topic chunks
        knowledge_<name>.txt  # per-topic concatenated knowledge text
    """
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Build knowledge texts (concatenated chunks per topic)
    knowledge_texts = {}
    for topic, chunks in topic_chunks.items():
        # Take top 50 chunks and concatenate
        combined = "\n\n---\n\n".join(c["text"] for c in chunks[:50])
        knowledge_texts[topic] = combined

    # Save module config
    total_chunks = sum(len(c) for c in topic_chunks.values())
    module_config = {
        "name": "python_coding_25k",
        "source": "flytech/python-codes-25k",
        "license": "MIT",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topics": list(topic_chunks.keys()),
        "total_chunks": total_chunks,
        "chunk_size": 400,
        "max_per_topic": 500,
        "description": "25K Python coding tasks grouped by topic for AirMoE",
    }

    config_path = out / "module.json"
    config_path.write_text(json.dumps(module_config, indent=2), encoding="utf-8")

    # Save per-topic chunks
    for topic, chunks in topic_chunks.items():
        chunk_path = out / f"topic_{topic}.json"
        chunk_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # Save knowledge texts
    for topic, text in knowledge_texts.items():
        text_path = out / f"knowledge_{topic}.txt"
        text_path.write_text(text, encoding="utf-8")

    # Save combined knowledge
    combined_path = out / "all_knowledge.json"
    combined_path.write_text(
        json.dumps(knowledge_texts, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Save all samples as JSONL (for self-play task generation)
    all_samples_path = out / "all_samples.jsonl"
    with open(all_samples_path, "w", encoding="utf-8") as f:
        for topic, chunks in topic_chunks.items():
            for chunk in chunks:
                f.write(json.dumps({
                    "topic": topic,
                    "instruction": chunk["instruction"],
                    "text": chunk["text"],
                }, ensure_ascii=False) + "\n")

    total_size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"\n  Module saved to {out}")
    print(f"  Total size: {total_size/1e6:.1f} MB")
    print(f"  Files: {len(list(out.iterdir()))}")

    return str(config_path)


def main():
    print("=" * 70)
    print("Python Coding Training Data → AirMoE Knowledge Module")
    print("=" * 70)

    # Phase 1: Download
    print("\n[1] Downloading Python coding dataset...")
    samples = download_dataset()

    # Phase 2: Process into topic packs
    print("\n[2] Processing into topic knowledge packs...")
    topic_chunks = process_into_packs(samples, max_per_topic=500)

    # Phase 3: Build AirMoE module
    print("\n[3] Building AirMoE module...")
    config_path = build_airmoe_module(topic_chunks)

    # Summary
    print(f"\n{'='*70}")
    print(f"Python Coding AirMoE Module Complete")
    print(f"{'='*70}")
    print(f"  Source: flytech/python-codes-25k (MIT license)")
    print(f"  Config: {config_path}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Topics: {len(topic_chunks)}")
    print(f"  Total chunks: {sum(len(c) for c in topic_chunks.values())}")
    print(f"\n  Topic breakdown:")
    for topic, chunks in sorted(topic_chunks.items()):
        print(f"    {topic}: {len(chunks)} chunks")
    print(f"\n  To use at inference:")
    print(f"    knowledge = json.load(open('{OUTPUT_DIR}/all_knowledge.json'))")
    print(f"    # Create KV cache packs per topic")
    print(f"    # Router classifies query → load relevant topic expert from disk")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
