"""Quick test: can the distill client generate tasks via APIs?"""
import os
import sys

# Load .env
from pathlib import Path
env_path = Path("D:/windsurf/ForgeAI/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from research.distillation.agentic_distill import AgenticDistillClient

client = AgenticDistillClient()
print(f"Available models: {len(client.models)}")
for m in client.models[:5]:
    print(f"  {m.model_id} ({m.provider})")

tasks = client.generate_tasks(n_tasks=5)
print(f"\nGot {len(tasks)} tasks")
for t in tasks[:3]:
    desc = t.get("task", "")[:80]
    tests = t.get("test_cases", [])
    print(f"  task: {desc}")
    print(f"  tests: {len(tests)}")
    if tests:
        print(f"  first test: input={tests[0].get('input', '')!r} output={tests[0].get('output', '')!r}")
