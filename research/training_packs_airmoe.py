"""Download multiple training data packs and build ONE AirMoE module PER TOPIC.

Each topic gets its own module directory for hotswap-based loading:
  airmoe_modules/
    python_math/
      module.json
      knowledge.json
      knowledge.txt
      samples.jsonl
    python_strings/
      module.json
      ...
    math_arithmetic/
      ...
    reasoning_general/
      ...
    science_method/
      ...

At inference, the router classifies the user query → loads ONLY the relevant
topic module from disk → injects its knowledge pack. Other topics stay on disk.

Downloads:
  1. flytech/python-codes-25k       — 25K Python coding tasks (MIT)
  2. openai/gsm8k                   — 8.5K grade-school math word problems (MIT)
  3. DuoNeural/cot-reasoning-2k     — 2K chain-of-thought reasoning
  4. Raymond-dev-546730/Open-CoT-Reasoning-Mini — 10K CoT across domains
  5. mattwesney/CoT_Reasoning_Scientific_Discovery_and_Research — science

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

OUTPUT_BASE = "D:/windsurf/ForgeAI/research/checkpoints/airmoe_modules"
CACHE_DIR = "D:/windsurf/ForgeAI/.devin/tmp/hf_cache"

DATASETS = [
    {
        "name": "python_codes_25k",
        "hf_id": "flytech/python-codes-25k",
        "split": "train",
        "license": "MIT",
        "topic_hint": "python",
        "fields": {"instruction": "instruction", "output": "output"},
    },
    {
        "name": "gsm8k_math",
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "license": "MIT",
        "topic_hint": "math",
        "fields": {"instruction": "question", "output": "answer"},
    },
    {
        "name": "cot_reasoning_2k",
        "hf_id": "DuoNeural/cot-reasoning-2k",
        "split": "train",
        "license": "MIT",
        "topic_hint": "reasoning",
        "fields": {"instruction": "messages", "output": "messages"},
    },
    {
        "name": "open_cot_reasoning_mini",
        "hf_id": "Raymond-dev-546730/Open-CoT-Reasoning-Mini",
        "split": "train",
        "license": "MIT",
        "topic_hint": "reasoning",
        "fields": {"instruction": "input", "output": "output"},
    },
    {
        "name": "science_cot",
        "hf_id": "mattwesney/CoT_Reasoning_Scientific_Discovery_and_Research",
        "split": "train",
        "license": "MIT",
        "topic_hint": "science",
        "fields": {"instruction": "question", "output": "answer"},
    },
]


def classify_topic(instruction: str, output: str, hint: str = "") -> str:
    """Classify a sample into a topic."""
    text = (instruction + " " + output).lower()

    if hint == "python":
        if any(k in text for k in ["fibonacci", "factorial", "prime", "gcd",
                                     "sqrt", "matrix", "sum of", "average",
                                     "quadratic", "polynomial"]):
            return "python_math"
        if any(k in text for k in ["string", "reverse", "palindrome", "regex",
                                     "substring", "concat", "split"]):
            return "python_strings"
        if any(k in text for k in ["sort", "search", "binary", "recursive",
                                     "dynamic programming", "tree", "graph"]):
            return "python_algorithms"
        if any(k in text for k in ["class ", "object", "inherit", "method",
                                     "__init__", "self."]):
            return "python_oop"
        if any(k in text for k in ["file", "open(", "read(", "write(",
                                     "csv", "json", "path"]):
            return "python_file_io"
        return "python_general"

    if hint == "math":
        if any(k in text for k in ["proof", "theorem", "lemma", "corollary"]):
            return "math_theory"
        if any(k in text for k in ["algebra", "equation", "solve for"]):
            return "math_algebra"
        if any(k in text for k in ["geometry", "triangle", "circle", "angle"]):
            return "math_geometry"
        if any(k in text for k in ["probability", "combinatoric", "permutation"]):
            return "math_probability"
        return "math_arithmetic"

    if hint == "reasoning":
        if any(k in text for k in ["logic", "syllogism", "deductive", "premise"]):
            return "logic"
        if any(k in text for k in ["proof", "theorem", "mathematical proof"]):
            return "math_theory"
        return "reasoning_general"

    if hint == "science":
        if any(k in text for k in ["hypothesis", "experiment", "variable"]):
            return "science_method"
        if any(k in text for k in ["biology", "cell", "organism", "evolution"]):
            return "science_biology"
        if any(k in text for k in ["chemistry", "molecule", "reaction", "bond"]):
            return "science_chemistry"
        if any(k in text for k in ["physics", "force", "energy", "quantum"]):
            return "science_physics"
        return "science_general"

    return "general"


def extract_samples(row: Dict, fields: Dict, hint: str) -> Optional[Tuple[str, str]]:
    """Extract (instruction, output) from a dataset row."""
    inst_field = fields["instruction"]
    out_field = fields["output"]

    if inst_field == "messages" and out_field == "messages":
        messages = row.get("messages", [])
        if not messages or len(messages) < 2:
            return None
        instruction = ""
        output = ""
        for msg in messages:
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
            elif msg.get("role") == "assistant":
                output = msg.get("content", "")
        if instruction and output:
            return instruction, output
        return None

    instruction = str(row.get(inst_field, "")).strip()
    output = str(row.get(out_field, "")).strip()

    if not instruction or not output:
        return None
    if len(instruction) < 10 or len(output) < 20:
        return None

    return instruction, output


def download_dataset(ds_config: Dict) -> List[Dict]:
    """Download a single dataset and return processed samples."""
    os.environ["HF_HOME"] = CACHE_DIR
    from datasets import load_dataset

    name = ds_config["name"]
    hf_id = ds_config["hf_id"]
    split = ds_config["split"]
    hint = ds_config["topic_hint"]
    fields = ds_config["fields"]
    config_name = ds_config.get("config", None)

    print(f"\n  [{name}] Downloading {hf_id}...")
    try:
        if config_name:
            ds = load_dataset(hf_id, config_name, split=split, cache_dir=CACHE_DIR)
        else:
            ds = load_dataset(hf_id, split=split, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"    FAILED: {e}")
        return []

    # Check available fields
    if len(ds) > 0:
        available_fields = list(ds[0].keys())
        print(f"    Available fields: {available_fields}")

        # Auto-detect field names if configured ones don't exist
        inst_field = fields["instruction"]
        out_field = fields["output"]
        if inst_field != "messages" and inst_field not in available_fields:
            # Try common alternatives
            for alt in ["instruction", "question", "input", "prompt", "problem"]:
                if alt in available_fields:
                    fields = {"instruction": alt, "output": fields["output"]}
                    print(f"    Auto-detected instruction field: {alt}")
                    break
        if out_field != "messages" and out_field not in available_fields:
            for alt in ["output", "answer", "response", "solution", "completion"]:
                if alt in available_fields:
                    fields = {"instruction": fields["instruction"], "output": alt}
                    print(f"    Auto-detected output field: {alt}")
                    break

    samples = []
    for row in ds:
        result = extract_samples(row, fields, hint)
        if result is None:
            continue
        instruction, output = result
        topic = classify_topic(instruction, output, hint)

        samples.append({
            "instruction": instruction[:500],
            "output": output[:2000],
            "topic": topic,
            "source": name,
        })

    print(f"    Extracted {len(samples)} samples")
    return samples


def build_per_topic_modules(all_samples: List[Dict],
                             max_per_topic: int = 300) -> Dict[str, str]:
    """Build ONE AirMoE module PER TOPIC — each in its own directory.

    Each module directory contains:
      module.json     — config + metadata for this topic
      knowledge.json  — all knowledge chunks for this topic
      knowledge.txt   — concatenated knowledge text (for KV cache injection)
      samples.jsonl   — individual samples (for self-play task generation)
    """
    # Group by topic
    by_topic: Dict[str, List[Dict]] = {}
    for s in all_samples:
        topic = s["topic"]
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(s)

    base = Path(OUTPUT_BASE)
    base.mkdir(parents=True, exist_ok=True)

    # Save global index (router uses this to pick which module to load)
    global_index = {
        "description": "Per-topic AirMoE modules for hotswap loading",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topics": {},
    }

    module_paths = {}

    for topic, items in by_topic.items():
        # Sort by output length (longer = more detailed)
        items.sort(key=lambda x: len(x["output"]), reverse=True)
        items = items[:max_per_topic]

        # Build knowledge chunks
        chunks = []
        for item in items:
            knowledge = (
                f"# Task: {item['instruction']}\n"
                f"# Solution:\n{item['output']}\n"
            )
            chunk_hash = hashlib.md5(knowledge.encode()).hexdigest()[:8]
            chunks.append({
                "text": knowledge,
                "topic": topic,
                "source": item["source"],
                "hash": chunk_hash,
                "instruction": item["instruction"][:100],
                "output_length": len(item["output"]),
            })

        # Create topic module directory
        topic_dir = base / topic
        topic_dir.mkdir(parents=True, exist_ok=True)

        # module.json — config for this topic module
        sources_in_topic = list(set(c["source"] for c in chunks))
        module_config = {
            "name": f"airmoe_{topic}",
            "topic": topic,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_chunks": len(chunks),
            "sources": sources_in_topic,
            "description": f"AirMoE knowledge module for topic: {topic}",
            "hotswap": True,
        }
        (topic_dir / "module.json").write_text(
            json.dumps(module_config, indent=2), encoding="utf-8")

        # knowledge.json — all chunks
        (topic_dir / "knowledge.json").write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        # knowledge.txt — concatenated (top 30 chunks) for KV cache injection
        combined = "\n\n---\n\n".join(c["text"] for c in chunks[:30])
        (topic_dir / "knowledge.txt").write_text(combined, encoding="utf-8")

        # samples.jsonl — for self-play task generation
        with open(topic_dir / "samples.jsonl", "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({
                    "topic": topic,
                    "instruction": chunk["instruction"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                }, ensure_ascii=False) + "\n")

        # Compute module size
        module_size = sum(f.stat().st_size for f in topic_dir.iterdir() if f.is_file())

        # Add to global index
        global_index["topics"][topic] = {
            "path": str(topic_dir),
            "n_chunks": len(chunks),
            "size_mb": round(module_size / 1e6, 2),
            "sources": sources_in_topic,
            "knowledge_file": "knowledge.txt",
            "samples_file": "samples.jsonl",
        }

        module_paths[topic] = str(topic_dir)
        print(f"    {topic}/: {len(chunks)} chunks, {module_size/1e6:.1f} MB")

    # Save global index (router reads this to know which modules exist)
    index_path = base / "index.json"
    index_path.write_text(json.dumps(global_index, indent=2), encoding="utf-8")

    total_size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
    print(f"\n  All modules saved to {base}")
    print(f"  Total size: {total_size/1e6:.1f} MB")
    print(f"  Modules: {len(module_paths)}")

    return module_paths


def main():
    print("=" * 70)
    print("Per-Topic AirMoE Module Builder")
    print("=" * 70)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Phase 1: Download all datasets
    print(f"\n[1] Downloading {len(DATASETS)} datasets...")
    all_samples = []
    for ds_config in DATASETS:
        samples = download_dataset(ds_config)
        all_samples.extend(samples)

    print(f"\n  Total samples: {len(all_samples)}")

    # Phase 2: Build per-topic modules
    print(f"\n[2] Building per-topic AirMoE modules...")
    module_paths = build_per_topic_modules(all_samples, max_per_topic=300)

    # Summary
    print(f"\n{'='*70}")
    print(f"Per-Topic AirMoE Modules Complete")
    print(f"{'='*70}")
    print(f"  Base: {OUTPUT_BASE}")
    print(f"  Index: {OUTPUT_BASE}/index.json")
    print(f"  Modules: {len(module_paths)}")
    print(f"\n  Module layout:")
    print(f"    airmoe_modules/")
    for topic in sorted(module_paths.keys()):
        print(f"      {topic}/  (module.json, knowledge.json, knowledge.txt, samples.jsonl)")
    print(f"\n  Hotswap usage:")
    print(f"    router = TopicRouter('{OUTPUT_BASE}/index.json')")
    print(f"    topic = router.classify(query)")
    print(f"    module = AirMoEModule.load('{OUTPUT_BASE}/' + topic)")
    print(f"    module.inject_knowledge(model)  # KV cache injection")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
