import json, sys
sys.path.insert(0, '.')
with open('research/data/self_play/self_play_20260730_175946.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        p = json.loads(line)
        print(f"--- Packet {i+1}: {p['task']} ---")
        print(f"Prompt: {p['prompt']}")
        print(f"Generated code (first 300 chars):")
        print(repr(p['generated_code'][:300]))
        print(f"Error: {p['execution']['stderr'][:300]}")
        print()
