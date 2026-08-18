"""Debug: see what the model generates when trying to solve an API task."""
import os, sys, torch
from pathlib import Path
env_path = Path("D:/windsurf/ForgeAI/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from research.model_loader import load_default_model
from research.tokenizer_cache import get_tokenizer
from research.self_play.infinite_curriculum import InfiniteCurriculum

model, tokenizer = load_default_model("forgelm_v3",
    checkpoint_path="research/checkpoints/ForgeLM_V3_SFT.safetensors",
    device="cuda", dtype=torch.bfloat16)
model.eval()

curr = InfiniteCurriculum(model=model, tokenizer=tokenizer, device="cuda",
                          max_gen_tokens=256, temperature=0.7, top_k=80, top_p=0.95)

# Get a few API tasks
tasks = curr.api_propose_tasks(n=30, domain="algorithms", difficulty="easy")
print(f"Got {len(tasks)} tasks\n")

if tasks:
    task = tasks[0]
    print(f"Task: {task.description[:100]}")
    print(f"Signature: {task.signature}")
    print(f"Test cases: {task.test_cases[:2]}")
    print()

    # Solve it
    result = curr.solve_task(task)
    print(f"Success: {result.get('final_success', False)}")
    print(f"Attempts: {len(result.get('attempts', []))}")
    if result.get("attempts"):
        att = result["attempts"][0]
        code = att.get("code", "")
        print(f"\n--- Generated code (first 500 chars) ---")
        print(code[:500])
        print(f"\n--- Error ---")
        print(att.get("error", "")[:500])
