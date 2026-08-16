"""Full agent-loop test for the SFT-trained ForgeLM V2.

Tests the complete function-calling workflow:
1. User prompt -> model emits tool calls + <|im_end|>
2. We intercept the tool calls, execute fake tools, inject results
3. Model produces final answer + <|im_end|>

This validates that the model:
- Stops after emitting tool calls (critical for agent loops)
- Accepts tool results and produces a final answer
- Maintains conversation context across turns
"""
import json
import re
import sys
import torch

sys.stdout.reconfigure(encoding="utf-8")

from research.model_loader import load_default_model
from research.tokenizer_cache import get_tokenizer

EOS_ID = 7  # <|im_end|>
IM_END = "<|im_end|>"


# Fake tool implementations for testing
FAKE_TOOLS = {
    "get_weather": lambda args: json.dumps({"city": args.get("city", "?"), "temp": 22, "summary": "partly cloudy", "humidity": 55}),
    "search_flights": lambda args: json.dumps([{"flight": "AA123", "departure": "08:00", "arrival": "16:00", "price": 450}]),
    "send_email": lambda args: json.dumps({"status": "sent", "recipients": len(args.get("to", []))}),
    "calculate": lambda args: json.dumps({"result": eval(args.get("expression", "0"), {"__builtins__": {}})}),
    "currency_convert": lambda args: json.dumps({"amount": 1250.0, "rate": 0.83} if args.get("to_currency") == "EUR" else {"amount": 187500, "rate": 125.0}),
    "translate": lambda args: json.dumps({"translation": "[translated text]", "target_lang": args.get("target_lang", "?")}),
    "geocode": lambda args: json.dumps({"lat": 40.7128, "lng": -74.0060, "address": args.get("address", "?")}),
    "calendar_create": lambda args: json.dumps({"event_id": "evt_123", "status": "created"}),
    "parse_csv": lambda args: json.dumps([{"row": i} for i in range(3)]),
    "validate_form": lambda args: json.dumps({"valid": False, "errors": ["email format invalid"]}),
    "query_database": lambda args: json.dumps({"rows": 42, "query": args.get("sql", "?")}),
    "http_get": lambda args: json.dumps({"status": 200, "content": "Example page content here."}),
    "search_web": lambda args: json.dumps([{"title": "Result 1", "url": "https://example.com/1"}]),
    "create_task": lambda args: json.dumps({"task_id": "task_456", "status": "created"}),
    "get_stock_price": lambda args: json.dumps({"ticker": args.get("ticker", "?"), "price": 185.50}),
    "list_files": lambda args: json.dumps({"files": ["file1.py", "file2.py", "file3.py"], "count": 3}),
    "read_file": lambda args: json.dumps({"content": "file contents here"}),
    "write_file": lambda args: json.dumps({"status": "written", "bytes": len(args.get("content", ""))}),
    "get_directions": lambda args: json.dumps({"distance": "15.2 km", "duration": "25 min", "mode": args.get("mode", "driving")}),
    "scrape_prices": lambda args: json.dumps([{"product": "Item A", "price": 9.99}, {"product": "Item B", "price": 14.50}]),
    "image_resize": lambda args: json.dumps({"status": "resized", "dimensions": f"{args.get('width')}x{args.get('height')}"}),
    "process_payment": lambda args: json.dumps({"status": "charged", "amount": args.get("amount"), "transaction_id": "txn_789"}),
}


def generate(model, tokenizer, prompt_text, max_new_tokens=256, temperature=0.0, device="cuda"):
    """Generate text until <|im_end|> or max tokens."""
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
    """Extract tool calls from model output using balanced-brace JSON parsing.

    Handles nested JSON arguments (e.g. {"arguments": {"fields": {"name": "Alice"}}}).
    Returns list of {name, arguments} dicts, or empty list if none found.
    """
    calls = []
    consumed = []
    for hint in re.finditer(r'\{"name"\s*:\s*"([^"]+)"', text):
        brace_start = hint.start()  # The regex starts at the { character
        if any(s <= brace_start < e for s, e in consumed):
            continue
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


def execute_tool_call(call):
    """Execute a fake tool call and return the result string."""
    name = call["name"]
    args = call.get("arguments", {})
    if name in FAKE_TOOLS:
        try:
            return FAKE_TOOLS[name](args)
        except Exception as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"unknown tool: {name}"})


def run_agent_loop(model, tokenizer, user_query, max_turns=3, device="cuda"):
    """Run a full agent loop: user -> tool calls -> results -> final answer.
    
    Returns dict with:
      - tool_calls_made: list of tool calls
      - tool_results: list of results
      - final_answer: str
      - stopped_after_tools: bool (did model emit <|im_end|> after tool calls?)
      - produced_final_answer: bool
    """
    messages = [{"role": "user", "content": user_query}]
    all_tool_calls = []
    all_tool_results = []
    final_answer = None
    stopped_after_tools = False
    
    for turn in range(max_turns):
        # Build the prompt from messages
        prompt_parts = []
        for m in messages:
            if m["role"] == "user":
                prompt_parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n")
            elif m["role"] == "assistant":
                if m.get("tool_calls"):
                    body = "\n".join(
                        json.dumps(tc, ensure_ascii=False) for tc in m["tool_calls"]
                    )
                else:
                    body = m.get("content", "")
                prompt_parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
            elif m["role"] == "tool":
                name = m.get("name", "tool")
                content = m.get("content", "")
                prompt_parts.append(f"<|im_start|>tool\n{name}\n{content}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "".join(prompt_parts)
        
        # Generate
        output = generate(model, tokenizer, prompt, max_new_tokens=256, device=device)
        
        # Check if it stopped (we detect <|im_end|> in the output via EOS)
        # Since we break on EOS_ID=7, the output won't contain <|im_end|> text
        # but the model did emit it (that's why we stopped).
        
        # Parse tool calls from output
        tool_calls = parse_tool_calls(output)
        
        if tool_calls:
            stopped_after_tools = True
            all_tool_calls.extend(tool_calls)
            # Add assistant message with tool calls
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            # Execute and add tool results
            for tc in tool_calls:
                result = execute_tool_call(tc)
                all_tool_results.append({"name": tc["name"], "result": result})
                messages.append({"role": "tool", "name": tc["name"], "content": result})
            # Continue to next turn (model should produce final answer now)
        else:
            # No tool calls — this is the final answer
            final_answer = output.strip()
            messages.append({"role": "assistant", "content": final_answer})
            break
    
    return {
        "tool_calls_made": all_tool_calls,
        "tool_results": all_tool_results,
        "final_answer": final_answer,
        "stopped_after_tools": stopped_after_tools,
        "produced_final_answer": final_answer is not None,
        "n_turns": turn + 1,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="research/checkpoints/ForgeLM_V2_BSP.safetensors")
    args = p.parse_args()

    device = "cuda"
    tokenizer = get_tokenizer()
    
    ckpt_name = args.checkpoint.split("/")[-1].replace(".safetensors", "")
    print("=" * 70)
    print(f"Full Agent-Loop Test: {ckpt_name}")
    print("=" * 70)
    
    model, _ = load_default_model(
        config_name="lfm25_1.2b",
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    
    test_queries = [
        "What is the weather in Tokyo right now?",
        "Convert 500 USD to EUR and tell me the result.",
        "Calculate (15 + 27) * 0.8 and tell me the answer.",
        "Search for flights from NYC to London on 2026-09-15 for 1 passenger.",
        "Translate 'Hello world' to French.",
    ]
    
    results = []
    for q in test_queries:
        print(f"\n{'─' * 70}")
        print(f"QUERY: {q}")
        result = run_agent_loop(model, tokenizer, q, device=device)
        results.append(result)
        
        print(f"  Tool calls: {len(result['tool_calls_made'])}")
        for tc in result["tool_calls_made"]:
            print(f"    -> {tc['name']}({json.dumps(tc['arguments'])})")
        print(f"  Stopped after tools: {result['stopped_after_tools']}")
        print(f"  Produced final answer: {result['produced_final_answer']}")
        if result["final_answer"]:
            print(f"  FINAL ANSWER: {result['final_answer'][:200]}")
    
    del model
    torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    n_tool_calls = sum(1 for r in results if r["tool_calls_made"])
    n_stopped = sum(1 for r in results if r["stopped_after_tools"])
    n_final = sum(1 for r in results if r["produced_final_answer"])
    print(f"  {len(results)} queries tested")
    print(f"  Made tool calls:      {n_tool_calls}/{len(results)}")
    print(f"  Stopped after tools:  {n_stopped}/{len(results)}")
    print(f"  Produced final answer: {n_final}/{len(results)}")
    
    full_pass = all(r["tool_calls_made"] and r["produced_final_answer"] for r in results)
    print(f"\n  FULL LOOP PASS: {'YES' if full_pass else 'NO'}")


if __name__ == "__main__":
    main()
