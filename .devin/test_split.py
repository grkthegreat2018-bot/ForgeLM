"""Test the per-turn splitting logic for multi-turn conversations."""
import json
from research.training.sft_train import split_multi_turn, tokenize_example
from research.tokenizer_cache import get_tokenizer

tok = get_tokenizer()

# A typical function-calling conversation
msgs = [
    {"role": "user", "content": "What is the weather in Tokyo and Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Paris"}},
    ]},
    {"role": "tool", "name": "get_weather", "content": '{"temp": 20, "summary": "sunny"}'},
    {"role": "tool", "name": "get_weather", "content": '{"temp": 18, "summary": "cloudy"}'},
    {"role": "assistant", "content": "Tokyo is 20C and sunny, Paris is 18C and cloudy."},
]

pairs = split_multi_turn(msgs)
print(f"Split into {len(pairs)} training examples:\n")

for i, (prompt, completion) in enumerate(pairs):
    print(f"--- Example {i+1} ---")
    print(f"PROMPT ({len(prompt)} chars):")
    print(repr(prompt[:200]))
    print(f"COMPLETION ({len(completion)} chars):")
    print(repr(completion[:200]))
    # Check that completion ends with <|im_end|>
    ends_with_eos = "<|im_end|>" in completion
    print(f"Ends with <|im_end|>: {ends_with_eos}")
    print()

# Now test tokenization
print("=" * 60)
print("TOKENIZATION TEST")
print("=" * 60)
ex = {"type": "multi_turn", "messages": msgs}
toks = tokenize_example(ex, tok, 1024)
print(f"Produced {len(toks)} tokenized examples")
for i, t in enumerate(toks):
    print(f"  Example {i+1}: {len(t['input_ids'])} tokens, {t['n_comp']} completion tokens")
    # Check the last few labels are not -100 (should be the <|im_end|> token)
    last_real = [l for l in t["labels"] if l != -100]
    print(f"    Last 5 completion labels: {last_real[-5:]}")
    # Token 7 = <|im_end|>, should be the last completion token
    print(f"    Last label is <|im_end|> (7): {last_real[-1] == 7}")
