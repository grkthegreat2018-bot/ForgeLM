import json
d = json.load(open(".devin/llm_eval_results.json", encoding="utf-8"))
for i, (rb, rs) in enumerate(zip(d["results_base"][:10], d["results_sft"][:10])):
    print(f"Q{i+1} [{rb['category']}]: {rb['question'][:60]}")
    print(f"  BASE: {rb['output'][:120]}")
    print(f"  SFT:  {rs['output'][:120]}")
    print()
