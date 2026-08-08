"""OpenAI-compatible chat server for the custom ForgeAI research model."""
import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader
from research.runtime.signal_capture import SignalLogger


def build_prompt(tokenizer, messages, tools=None, add_generation_prompt=True):
    """Use the Qwen chat template to turn messages (and optional tools) into a prompt string."""
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools or [],
        )
    except Exception:
        # Fallback for older tokenizer versions.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return text


def parse_tool_calls(text):
    """Parse <functioncall>...</functioncall> tags into OpenAI tool_calls format."""
    tool_calls = []
    if "<functioncall>" not in text:
        return tool_calls, text

    remaining = text
    while "<functioncall>" in remaining and "</functioncall>" in remaining:
        start = remaining.index("<functioncall>")
        end = remaining.index("</functioncall>", start) + len("</functioncall>")
        inner = remaining[start + len("<functioncall>") : end - len("</functioncall>")].strip()
        remaining = remaining[:start] + remaining[end:]
        try:
            obj = json.loads(inner)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": obj.get("name", ""),
                    "arguments": json.dumps(obj.get("arguments", obj.get("parameters", {}))),
                },
            })
        except Exception:
            pass
    return tool_calls, remaining.strip()


class OpenAICompatHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    config = None
    signal_logger = None  # SignalLogger instance (None = capture disabled)
    default_temperature = 0.7
    default_max_tokens = 512

    def _send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunk_dict):
        line = f"data: {json.dumps(chunk_dict)}\n\n".encode("utf-8")
        self.wfile.write(line)
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/v1/models", "/v1/models/"):
            self._send_json(200, {
                "object": "list",
                "data": [{"id": "custom-research-llm", "object": "model"}],
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._handle_chat_completions()
        elif parsed.path in ("/v1/feedback", "/v1/feedback/"):
            self._handle_feedback()
        elif parsed.path in ("/v1/code_result", "/v1/code_result/"):
            self._handle_code_result()
        else:
            self._send_json(404, {"error": "Not found"})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _generate_tokens(self, idx, max_new_tokens, temperature, eos_id):
        """Generator yielding token ids using KV cache."""
        model = self.model
        device = next(model.parameters()).device
        past_key_values = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    if past_key_values is None:
                        idx_cond = idx[:, -self.config.max_seq_len :]
                        logits, _, past_key_values = model(idx_cond, use_cache=True)
                    else:
                        logits, _, past_key_values = model(idx[:, -1:], past_key_values=past_key_values, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            if token_id == eos_id:
                break
            idx = torch.cat((idx, next_token), dim=1)
            yield token_id

    def _handle_chat_completions(self):
        body = self._read_body()
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        temperature = float(body.get("temperature", self.default_temperature))
        max_tokens = int(body.get("max_tokens", self.default_max_tokens))

        if not messages:
            self._send_json(400, {"error": "messages required"})
            return

        prompt = build_prompt(self.tokenizer, messages, tools=tools, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(next(self.model.parameters()).device)
        eos_id = self.tokenizer.eos_token_id

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            for token_id in self._generate_tokens(input_ids, max_tokens, temperature, eos_id):
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "custom-research-llm",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                self._send_sse(chunk)

            self._send_sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": "custom-research-llm", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            generated_ids = []
            for token_id in self._generate_tokens(input_ids, max_tokens, temperature, eos_id):
                generated_ids.append(token_id)

            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
            tool_calls, content = parse_tool_calls(generated_text)
            message = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            finish_reason = "stop" if not tool_calls else "tool_calls"

            # Signal capture: log the interaction for live training (Phase 1).
            interaction_id = None
            if self.signal_logger is not None:
                interaction_id = self.signal_logger.log_interaction(
                    messages, content, temperature=temperature, max_tokens=max_tokens
                )

            response_body = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "custom-research-llm",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            }
            if interaction_id:
                response_body["interaction_id"] = interaction_id
            self._send_json(200, response_body)


    def _handle_feedback(self):
        """POST /v1/feedback — log user feedback on a previous interaction.

        Body: {"interaction_id": "...", "action": "accept|edit|reject", "edited_content": "..."}
        """
        if self.signal_logger is None:
            self._send_json(200, {"status": "capture_disabled"})
            return
        body = self._read_body()
        iid = body.get("interaction_id")
        action = body.get("action")
        edited = body.get("edited_content")
        if not iid or action not in ("accept", "edit", "reject"):
            self._send_json(400, {"error": "interaction_id and action (accept/edit/reject) required"})
            return
        self.signal_logger.log_feedback(iid, action, edited_content=edited)
        self._send_json(200, {"status": "logged", "interaction_id": iid, "action": action})

    def _handle_code_result(self):
        """POST /v1/code_result — log code execution outcome for a previous interaction.

        Body: {"interaction_id": "...", "success": true/false, "output": "...", "error": "...", "language": "python"}
        """
        if self.signal_logger is None:
            self._send_json(200, {"status": "capture_disabled"})
            return
        body = self._read_body()
        iid = body.get("interaction_id")
        if not iid:
            self._send_json(400, {"error": "interaction_id required"})
            return
        self.signal_logger.log_code_result(
            iid,
            success=body.get("success", False),
            output=body.get("output"),
            error=body.get("error"),
            language=body.get("language"),
        )
        self._send_json(200, {"status": "logged", "interaction_id": iid})


def main():
    parser = argparse.ArgumentParser(description="Serve the custom ForgeAI model via an OpenAI-compatible API.")
    parser.add_argument("--config", type=str, default="360m_mla")
    parser.add_argument("--checkpoint", type=str, default="research/checkpoints/sft_llm.pt")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile for inference (experimental; dynamic KV shapes may recompile).")
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="int8",
                        help="Inference weight quantization. int8 gives 4-8x speedup (default). "
                             "int4 gives 3x speedup with ~1-2%% quality loss. none disables.")
    parser.add_argument("--quant-group-size", type=int, default=128, help="INT4 group size (smaller=more accurate).")
    parser.add_argument("--signal-log", type=str, default=None,
                        help="Path to JSONL file for live training signal capture. Enables /v1/feedback and /v1/code_result endpoints.")
    args = parser.parse_args()

    cfg = get_config(args.config)
    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None

    print(f"Loading model {args.config}...")
    model = ModelLoader.build_model(cfg, checkpoint_path=checkpoint, compile=args.compile)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    # Apply inference quantization (INT8 = 4-8x speedup, default).
    if args.quantize == "int8":
        from research.quantization.inference_quant import quantize_model_int8
        n = quantize_model_int8(model)
        print(f"INT8 quantization applied to {n} layers (4-8x inference speedup)")
    elif args.quantize == "int4":
        from research.quantization.inference_quant import quantize_model_int4
        n = quantize_model_int4(model, group_size=args.quant_group_size)
        print(f"INT4 quantization applied to {n} layers (group={args.quant_group_size}, 3x speedup)")

    OpenAICompatHandler.model = model
    OpenAICompatHandler.tokenizer = tokenizer
    OpenAICompatHandler.config = cfg
    OpenAICompatHandler.default_temperature = args.temperature
    OpenAICompatHandler.default_max_tokens = args.max_tokens
    OpenAICompatHandler.signal_logger = (
        SignalLogger(args.signal_log, enabled=True) if args.signal_log else None
    )
    if OpenAICompatHandler.signal_logger:
        print(f"Signal capture enabled → {args.signal_log}")
        print("  POST /v1/feedback        {interaction_id, action, edited_content?}")
        print("  POST /v1/code_result     {interaction_id, success, output?, error?, language?}")

    server = ThreadingHTTPServer((args.host, args.port), OpenAICompatHandler)
    print(f"OpenAI-compatible server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
