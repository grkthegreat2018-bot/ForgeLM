"""Resume research trajectory generation from 48/100.

Loads existing v3_distill.jsonl, generates remaining 52 research trajectories,
appends them, and saves.
"""
import os, sys, json, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.distillation.distill_tool_calls import ToolCallDistiller

OUTPUT = "research/data/finetune/v3_distill.jsonl"
TARGET_RESEARCH = 100

# Load existing examples
with open(OUTPUT, encoding="utf-8") as f:
    existing = [json.loads(line) for line in f if line.strip()]

n_research = sum(1 for e in existing if e.get("type") == "research_trajectory")
remaining = TARGET_RESEARCH - n_research
print(f"Existing: {len(existing)} examples ({n_research} research trajectories)")
print(f"Need: {remaining} more research trajectories to reach {TARGET_RESEARCH}")

if remaining <= 0:
    print("Already at target. Nothing to do.")
    sys.exit(0)

# Generate only research trajectories
distiller = ToolCallDistiller()
print(f"\n[Resume] Generating {remaining} research trajectories...")
new_research = distiller.generate_research_trajectories(remaining)

# Append to existing
all_examples = existing + new_research
with open(OUTPUT, "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# Stats
from collections import Counter
types = Counter(e.get("type", "?") for e in all_examples)
print(f"\nDone! Total: {len(all_examples)} examples")
print(f"  Types: {dict(types)}")
print(f"  Research: {types['research_trajectory']}/{TARGET_RESEARCH}")
