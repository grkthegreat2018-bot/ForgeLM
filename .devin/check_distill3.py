import json
with open("research/data/finetune/v3_distill_test3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        if ex["type"] != "tool_call":
            continue
        task = ex["task"][:50]
        turns = ex["turns"]
        print(f"=== {task} ({len(turns)} turns) ===")
        for j, turn in enumerate(turns):
            role = turn["role"]
            content = turn["content"][:200].replace("\n", "\\n")
            print(f"  [{j}] {role}: {content}")
        print()
