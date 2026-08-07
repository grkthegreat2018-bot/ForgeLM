"""Generate a synthetic tool-calling SFT dataset matching eval_suite prompts."""
import argparse
import json
import random
from pathlib import Path


TOOLS = [
    {"name": "get_weather", "action": "fetch weather", "params": {"city": "London"}},
    {"name": "get_stock", "action": "check stock price", "params": {"symbol": "NVDA"}},
    {"name": "list_directory", "action": "list directory", "params": {"path": "project"}},
    {"name": "search_files", "action": "search files", "params": {"pattern": "train.py"}},
    {"name": "calculate", "action": "calculate", "params": {"expression": "(25 + 17) * 3"}},
    {"name": "send_email", "action": "send email", "params": {"to": "alice@example.com", "subject": "Hello"}},
    {"name": "set_alarm", "action": "set alarm", "params": {"time": "08:00"}},
    {"name": "get_exchange_rate", "action": "get exchange rate", "params": {"base_currency": "USD", "target_currency": "EUR"}},
]

CITIES = ["London", "New York", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Dubai"]
SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
DIRECTORIES = ["project", "src", "research", "data", "docs"]
PATTERNS = ["train.py", "*.md", "config.*", "*.json"]
EXPRESSIONS = ["25 + 17", "(25 + 17) * 3", "100 / 4", "2 ** 10", "sqrt(16)"]
EMAILS = ["alice@example.com", "bob@example.com"]
SUBJECTS = ["Update", "Hello", "Meeting"]
TIMES = ["08:00", "14:30", "23:15"]
CURRENCIES = [("USD", "EUR"), ("GBP", "USD"), ("JPY", "USD"), ("EUR", "JPY")]


def build_examples(n=10000, seed=42):
    random.seed(seed)
    examples = []
    for _ in range(n):
        tool = random.choice(TOOLS)
        name = tool["name"]

        if name == "get_weather":
            params = {"city": random.choice(CITIES)}
        elif name == "get_stock":
            params = {"symbol": random.choice(SYMBOLS)}
        elif name == "list_directory":
            params = {"path": random.choice(DIRECTORIES)}
        elif name == "search_files":
            params = {"pattern": random.choice(PATTERNS)}
        elif name == "calculate":
            params = {"expression": random.choice(EXPRESSIONS)}
        elif name == "send_email":
            params = {"to": random.choice(EMAILS), "subject": random.choice(SUBJECTS)}
        elif name == "set_alarm":
            params = {"time": random.choice(TIMES)}
        elif name == "get_exchange_rate":
            base, target = random.choice(CURRENCIES)
            params = {"base_currency": base, "target_currency": target}

        func_spec = json.dumps({"name": name, "parameters": params})
        prompt = f"Call tool to {tool['action']}: <functions>[{func_spec}]</functions>"
        completion = f'<functioncall>{{"name": "{name}", "arguments": {json.dumps(params, ensure_ascii=False)}}}</functioncall>'
        examples.append({"prompt": prompt, "completion": completion})

    # Negative examples: tool provided but not needed.
    for _ in range(n // 10):
        tool = random.choice(TOOLS)
        func_spec = json.dumps({"name": tool["name"], "parameters": tool["params"]})
        prompt = f"Call tool to tell a joke: <functions>[{func_spec}]</functions>"
        completion = "You don't have a tool for jokes, but here's one: Why did the scarecrow win an award? He was outstanding in his field."
        examples.append({"prompt": prompt, "completion": completion})

    random.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="research/data/tool_sft.jsonl")
    parser.add_argument("--examples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_examples(args.examples, args.seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
