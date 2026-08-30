"""Check SFT data sources for content type — identify non-coding/reasoning data."""
import json, glob, os, sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

ROOT = r"D:\windsurf\ForgeAI\research\data"

sources_to_check = [
    "sft/tool_use/nemotron-agentic-v2",
    "sft/tool_use/glaive-fc-v2",
    "sft/tool_use/hermes-reasoning-tool",
    "sft/tool_use/toolace",
    "sft/reasoning/metamath",
    "sft/reasoning/openthoughts-114k",
    "sft/reasoning/numina-math-1.5",
    "sft/reasoning/mixture-of-thoughts",
    "sft/coding/evol-codealpaca",
    "sft/coding/magicoder-evol",
    "sft/coding/magicoder-oss",
    "sft/agentic/agent-instruct",
    "sft/agentic/agent-trek",
    "sft/agentic/opencode-agentic-mini",
]

for src in sources_to_check:
    base = os.path.join(ROOT, src)
    if not os.path.exists(base):
        print(f"{src}: DIR NOT FOUND")
        continue
    # Find first readable file
    files = sorted(glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(base, "**", "*.json"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True))
    if not files:
        print(f"{src}: NO DATA FILES")
        continue
    path = files[0]
    try:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            table = pq.read_table(path, columns=None)
            row = table.slice(0, 1).to_pylist()[0]
            # Show first column content
            for k, v in row.items():
                if isinstance(v, str) and len(v) > 20:
                    print(f"{src}: {k} = {v[:150]}")
                    break
            else:
                print(f"{src}: parquet cols={list(row.keys())}")
        else:
            with open(path, encoding="utf-8") as f:
                line = f.readline()
                obj = json.loads(line)
                if "messages" in obj:
                    m = obj["messages"][0]
                    role = m.get("role", "")
                    content = m.get("content", "")[:150]
                    print(f"{src}: {role} - {content}")
                elif "text" in obj:
                    print(f"{src}: text - {obj['text'][:150]}")
                elif "prompt" in obj:
                    print(f"{src}: prompt - {obj['prompt'][:150]}")
                else:
                    print(f"{src}: keys={list(obj.keys())}")
    except Exception as e:
        print(f"{src}: ERROR - {e}")
