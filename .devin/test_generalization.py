"""Test generalization: can the SFT model use tools and rules it wasn't trained on?

Tests:
1. Novel tool — define a tool not in the training catalog, see if the model
   calls it with correct argument names.
2. Novel rule — give a constraint (e.g. "respond in exactly 10 words"), see
   if the model follows it.
3. Multi-tool novel — two new tools in one prompt.
4. Tool with unusual argument types (nested object, array).
"""
import json
import re
import sys
import torch

sys.stdout.reconfigure(encoding="utf-8")

from research.model_loader import load_default_model
from research.tokenizer_cache import get_tokenizer

EOS_ID = 7


def generate(model, tokenizer, prompt_text, max_new_tokens=256, temperature=0.0, device="cuda"):
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids = input_ids.clone()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            next_token = next_logits.argmax(-1, keepdim=True)
            if next_token.item() == EOS_ID:
                break
            ids = torch.cat([ids, next_token], dim=-1)
    new_ids = ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def parse_tool_calls(text):
    """Parse tool calls using balanced-brace JSON parsing (handles nested args)."""
    calls = []
    consumed = []
    for hint in re.finditer(r'\{"name"\s*:\s*"([^"]+)"', text):
        brace_start = hint.start()  # The regex starts at the { character
        if any(s <= brace_start < e for s, e in consumed):
            continue
        # Find balanced JSON
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(brace_start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            continue
        raw = text[brace_start:end]
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import json_repair
                obj = json_repair.loads(raw)
            except Exception:
                continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
            consumed.append((brace_start, end))
    return calls


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="research/checkpoints/ForgeLM_V2_BSP.safetensors")
    args = p.parse_args()

    device = "cuda"
    tokenizer = get_tokenizer()

    ckpt_name = args.checkpoint.split("/")[-1].replace(".safetensors", "")
    print("=" * 70)
    print(f"Generalization Test: {ckpt_name} with novel tools and rules")
    print("=" * 70)

    model, _ = load_default_model(
        config_name="lfm25_1.2b",
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()

    # ── Test 1: Novel tool (not in training catalog) ──
    print("\n--- Test 1: Novel tool 'get_stock_price' ---")
    prompt1 = (
        "<|im_start|>user\n"
        "You have access to this tool:\n"
        "- get_stock_price: Get the current price of a stock.\n"
        "  arguments: {\"ticker\": \"string (e.g. AAPL)\", \"exchange\": \"string (default NASDAQ)\"}\n\n"
        "What is the current price of NVDA stock?\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    out1 = generate(model, tokenizer, prompt1, max_new_tokens=128, device=device)
    tc1 = parse_tool_calls(out1)
    print(f"Output: {out1[:300]}")
    print(f"Tool calls: {tc1}")
    t1_pass = len(tc1) == 1 and tc1[0]["name"] == "get_stock_price" and "ticker" in tc1[0]["arguments"]
    print(f"PASS: {t1_pass}")

    # ── Test 2: Novel tool with unusual name ──
    print("\n--- Test 2: Novel tool 'fetch_sensor_data' ---")
    prompt2 = (
        "<|im_start|>user\n"
        "You have access to this tool:\n"
        "- fetch_sensor_data: Read data from an IoT sensor.\n"
        "  arguments: {\"sensor_id\": \"string\", \"metric\": \"string (temperature, humidity, pressure)\", \"since_hours\": \"integer\"}\n\n"
        "Get the temperature from sensor 'farm_03' for the last 24 hours.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    out2 = generate(model, tokenizer, prompt2, max_new_tokens=128, device=device)
    tc2 = parse_tool_calls(out2)
    print(f"Output: {out2[:300]}")
    print(f"Tool calls: {tc2}")
    t2_pass = len(tc2) == 1 and tc2[0]["name"] == "fetch_sensor_data" and tc2[0]["arguments"].get("sensor_id") == "farm_03"
    print(f"PASS: {t2_pass}")

    # ── Test 3: Novel rule (respond in JSON) ──
    print("\n--- Test 3: Novel rule (respond in exactly 10 words) ---")
    prompt3 = (
        "<|im_start|>user\n"
        "What is the capital of France? Answer in exactly 10 words.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    out3 = generate(model, tokenizer, prompt3, max_new_tokens=64, device=device)
    word_count = len(out3.strip().rstrip(".").split())
    print(f"Output: {out3[:200]}")
    print(f"Word count: {word_count}")
    t3_pass = 8 <= word_count <= 12  # allow some slack
    print(f"PASS (8-12 words): {t3_pass}")

    # ── Test 4a: Parallel tool calls (two independent tools, one generation) ──
    print("\n--- Test 4a: Parallel tool calls (two independent tools) ---")
    prompt4a = (
        "<|im_start|>user\n"
        "You have access to these tools:\n"
        "- get_weather: Get weather for a city.\n"
        "  arguments: {\"city\": \"string\"}\n"
        "- get_time: Get current time in a timezone.\n"
        "  arguments: {\"timezone\": \"string\"}\n\n"
        "Get the weather in Tokyo and the time in JST.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    out4a = generate(model, tokenizer, prompt4a, max_new_tokens=256, device=device)
    tc4a = parse_tool_calls(out4a)
    print(f"Output: {out4a[:400]}")
    print(f"Tool calls: {tc4a}")
    t4a_pass = (len(tc4a) >= 2
                and any(tc["name"] == "get_weather" for tc in tc4a)
                and any(tc["name"] == "get_time" for tc in tc4a))
    print(f"PASS (both tools in one generation): {t4a_pass}")

    # ── Test 4b: Sequential tool chaining (agent loop, 2 turns) ──
    print("\n--- Test 4b: Sequential tool chaining (agent loop) ---")
    prompt4b_turn1 = (
        "<|im_start|>user\n"
        "You have access to these tools:\n"
        "- create_ticket: Create a support ticket. Returns {\"ticket_id\": \"...\"}.\n"
        "  arguments: {\"subject\": \"string\", \"priority\": \"string (low, medium, high)\"}\n"
        "- assign_ticket: Assign a ticket to an agent.\n"
        "  arguments: {\"ticket_id\": \"string\", \"agent\": \"string (email)\"}\n\n"
        "Create a high-priority ticket about 'server down' and assign it to admin@company.com.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    out4b_t1 = generate(model, tokenizer, prompt4b_turn1, max_new_tokens=128, device=device)
    tc4b_t1 = parse_tool_calls(out4b_t1)
    print(f"Turn 1 output: {out4b_t1[:300]}")
    print(f"Turn 1 tool calls: {tc4b_t1}")
    t4b_turn1_ok = any(tc["name"] == "create_ticket" for tc in tc4b_t1)
    print(f"Turn 1 (create_ticket called): {t4b_turn1_ok}")

    t4b_pass = False
    if t4b_turn1_ok:
        # Simulate tool result and feed back for turn 2
        tool_result = '{"ticket_id": "TKT-4821"}'
        prompt4b_turn2 = (
            prompt4b_turn1
            + out4b_t1.split("<|im_end|>")[0] + "<|im_end|>\n"
            + f"<|im_start|>tool\ncreate_ticket\n{tool_result}<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
        out4b_t2 = generate(model, tokenizer, prompt4b_turn2, max_new_tokens=128, device=device)
        tc4b_t2 = parse_tool_calls(out4b_t2)
        print(f"Turn 2 output: {out4b_t2[:300]}")
        print(f"Turn 2 tool calls: {tc4b_t2}")
        t4b_pass = any(tc["name"] == "assign_ticket" for tc in tc4b_t2)
        print(f"Turn 2 (assign_ticket called): {t4b_pass}")
    else:
        print("Turn 2 skipped (turn 1 failed)")

    # ── Test 5: Tool with nested object argument ──
    print("\n--- Test 5: Tool with nested object argument ---")
    prompt5 = (
        "<|im_start|>user\n"
        "You have access to this tool:\n"
        "- update_profile: Update a user profile.\n"
        "  arguments: {\"user_id\": \"string\", \"fields\": \"object (e.g. {\\\"name\\\": \\\"...\\\", \\\"age\\\": 30})\"}\n\n"
        "Update user 'u123' to set name to 'Alice' and age to 30.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    out5 = generate(model, tokenizer, prompt5, max_new_tokens=128, device=device)
    tc5 = parse_tool_calls(out5)
    print(f"Output: {out5[:300]}")
    print(f"Tool calls: {tc5}")
    t5_pass = len(tc5) == 1 and tc5[0]["name"] == "update_profile" and tc5[0]["arguments"].get("user_id") == "u123"
    print(f"PASS: {t5_pass}")

    del model
    torch.cuda.empty_cache()

    # Summary
    results = [t1_pass, t2_pass, t3_pass, t4a_pass, t4b_pass, t5_pass]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {sum(results)}/{len(results)} passed")
    print(f"  Novel tool (get_stock_price):     {'PASS' if t1_pass else 'FAIL'}")
    print(f"  Novel tool (fetch_sensor_data):   {'PASS' if t2_pass else 'FAIL'}")
    print(f"  Novel rule (10 words):            {'PASS' if t3_pass else 'FAIL'}")
    print(f"  Parallel tool calls:              {'PASS' if t4a_pass else 'FAIL'}")
    print(f"  Sequential tool chaining:         {'PASS' if t4b_pass else 'FAIL'}")
    print(f"  Nested object argument:           {'PASS' if t5_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
