import json
with open("research/data/finetune/v3_distill_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        t = ex["type"]
        task = ex["task"][:40]
        prompt = ex["prompt"][:80]
        resp = ex["response"][:200]
        print(f"=== {t}: {task}... ===")
        print(f"Prompt: {prompt}...")
        print(f"Response: {resp}...")
        print()
