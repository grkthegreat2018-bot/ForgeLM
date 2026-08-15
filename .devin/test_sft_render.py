"""Quick test of SFT rendering + tokenization."""
import json
from research.training.sft_train import render_messages, render_single_turn, tokenize_example
from research.tokenizer_cache import get_tokenizer

tok = get_tokenizer()

# Test multi-turn (function-calling)
msgs = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
    {"role": "assistant", "content": None, "tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]},
    {"role": "tool", "name": "get_weather", "content": '{"temp": 20, "summary": "sunny"}'},
    {"role": "assistant", "content": "Tokyo is 20C and sunny."},
]
text, comp_start = render_messages(msgs)
print("=== MULTI-TURN RENDER ===")
print(repr(text[:300]))
print("comp_start_char:", comp_start)
print("prompt prefix:", repr(text[:comp_start]))
print()

ex = {"type": "multi_turn", "messages": msgs}
tok_ex = tokenize_example(ex, tok, 1024)
if tok_ex:
    print("TOKENIZED OK")
    print("  total tokens:", len(tok_ex["input_ids"]))
    print("  completion tokens:", tok_ex["n_comp"])
    print("  first 10 labels:", tok_ex["labels"][:10])
    print("  last 10 labels:", tok_ex["labels"][-10:])
else:
    print("TOKENIZED FAILED")

print()
print("=== SINGLE-TURN RENDER ===")
text2, comp_start2 = render_single_turn("What is 2+2?", "4")
print(repr(text2))
print("comp_start_char:", comp_start2)
ex2 = {"type": "single_turn", "prompt": "What is 2+2?", "response": "4"}
tok_ex2 = tokenize_example(ex2, tok, 1024)
if tok_ex2:
    print("TOKENIZED OK")
    print("  total tokens:", len(tok_ex2["input_ids"]))
    print("  completion tokens:", tok_ex2["n_comp"])
