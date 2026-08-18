import json
with open("research/data/finetune/v3_distill_test2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        if ex["type"] == "tool_call":
            for turn in ex["turns"]:
                if turn["role"] == "assistant":
                    has_end = "[end]" in turn["content"]
                    has_code = "python" in turn["content"]
                    task = ex["task"][:40]
                    print(f"{task}: [end]={has_end}, code={has_code}, len={len(turn['content'])}")
