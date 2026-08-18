import json
with open("research/data/finetune/v3_research_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        print(f"=== {ex['type']}: {ex['task'][:60]} ===")
        print(f"Turns: {ex['n_turns']}, Scripts: {ex['n_scripts']}, Searches: {ex['n_searches']}")
        for j, turn in enumerate(ex["turns"]):
            role = turn["role"]
            content = turn["content"][:300].replace("\n", "\\n")
            print(f"  [{j}] {role}: {content}")
        print()
