import json
with open("research/data/finetune/v3_distill_test5.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        t = ex["type"]
        task = ex["task"][:50]
        turns = ex["turns"]
        has_thinking = any("IMD" in turn.get("content", "") for turn in turns)
        print(f"=== {t}: {task} ({len(turns)} turns, thinking={has_thinking}) ===")
        for j, turn in enumerate(turns):
            role = turn["role"]
            content = turn["content"][:300].replace("\n", "\\n")
            print(f"  [{j}] {role}: {content}")
        print()
