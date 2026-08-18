"""Debug: see what API tasks look like and why most are filtered."""
import os
from pathlib import Path
env_path = Path("D:/windsurf/ForgeAI/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from research.distillation.agentic_distill import AgenticDistillClient, _is_filler_task
import ast

client = AgenticDistillClient()
tasks = client.generate_tasks(n_tasks=20)
print(f"Got {len(tasks)} raw tasks\n")

for i, t in enumerate(tasks):
    desc = t.get("task", "")
    tests = t.get("test_cases", [])
    filler = _is_filler_task(desc)
    print(f"--- Task {i+1} (filler={filler}) ---")
    print(f"  desc: {desc[:100]}")
    print(f"  tests: {len(tests)}")
    for tc in tests[:2]:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        # Try parsing
        try:
            inp_parsed = ast.literal_eval(inp.strip())
            out_parsed = ast.literal_eval(out.strip())
            print(f"    input={inp!r} -> {inp_parsed!r}  output={out!r} -> {out_parsed!r}  OK")
        except Exception as e:
            print(f"    input={inp!r}  output={out!r}  PARSE_FAIL: {e}")
