"""OpenAI-compatible FastAPI server for ForgeAI.

Endpoints:
  POST /v1/chat/completions  — chat with SSE streaming and tool calling
  POST /v1/completions       — text completions
  GET  /v1/models            — list available models
  GET  /health               — health check

Usage:
  $env:PYTHONPATH="D:\\windsurf\\ForgeAI"; python -m research.serving.server --port 8000
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from research.inference.forge_engine import ForgeEngine
from research.model_loader import unpack_output_with_kv
from research.self_play.discovery.qwen_adapter import (
    IM_END,
    IM_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen_parse_tool_calls,
    qwen_render_messages,
)

# ── Config ─────────────────────────────────────────────────────────────────

MODEL_ID = "ForgeLM_V2_LFM25-1.2B"
DEFAULT_CHECKPOINT = "research/checkpoints/ForgeLM_V2_LFM25-1.2B.sft10.safetensors"

# Markers to strip from content shown to the client
_SPECIAL_MARKERS = (TOOL_CALL_START, TOOL_CALL_END, IM_START, IM_END)

# ── Engine singleton ───────────────────────────────────────────────────────

_engine: ForgeEngine | None = None
_checkpoint_path: str = DEFAULT_CHECKPOINT


def get_engine() -> ForgeEngine:
    """Lazily initialise the ForgeEngine singleton."""
    global _engine
    if _engine is None:
        print(f"[server] Loading checkpoint: {_checkpoint_path}")
        _engine = ForgeEngine.from_checkpoint(_checkpoint_path)
        _engine.activate(kv_cache="standard", decoding="standard")
        print("[server] Engine ready")
    return _engine


# ── Pydantic models ────────────────────────────────────────────────────────


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 80
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    repetition_penalty: Optional[float] = 1.05


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 80
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    repetition_penalty: Optional[float] = 1.05


# ── Generation helpers ─────────────────────────────────────────────────────


def _get_eos_set(tokenizer) -> set[int]:
    """Collect all possible EOS token IDs."""
    eos_set = {7, 151643, 151645}  # LFM2.5 <|im_end|> + Qwen2.5 defaults
    eos_attr = getattr(tokenizer, "eos_token_id", None)
    if eos_attr is not None:
        eos_set.add(eos_attr)
    return eos_set


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling filter."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -1] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    generated_ids: list[int],
) -> torch.Tensor:
    """Sample the next token from logits with temperature/top-k/top-p/rep-penalty."""
    next_logits = logits[:, -1:, :] / max(temperature, 1e-5)
    if temperature <= 0:
        return next_logits.argmax(-1, keepdim=True)
    # Repetition penalty (last 64 tokens)
    if generated_ids:
        for tid in set(generated_ids[-64:]):
            next_logits[:, :, tid] /= rep_penalty
    # Top-k filtering
    if top_k > 0:
        k = min(top_k, next_logits.shape[-1])
        thresh = torch.topk(next_logits, k)[0][..., -1:, None]
        next_logits = next_logits.masked_fill(next_logits < thresh, float("-inf"))
    # Top-p filtering
    if top_p < 1.0:
        next_logits = _top_p_filter(next_logits, top_p)
    probs = F.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_full(
    engine: ForgeEngine,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 80,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
) -> str:
    """Generate full text with special tokens preserved for tool-call parsing."""
    model = engine.model
    tokenizer = engine.tokenizer
    device = engine.device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    eos_tensor = torch.tensor(list(_get_eos_set(tokenizer)), device=device)
    generated_ids: list[int] = []
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        logits, past_kv = unpack_output_with_kv(out)
        for _ in range(max_new_tokens):
            next_token = _sample_next(
                logits, temperature, top_k, top_p, repetition_penalty, generated_ids
            )
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if (next_token == eos_tensor).any().item():
                break
            out = model(next_token, past_key_values=past_kv, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def stream_tokens(
    engine: ForgeEngine,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 80,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
):
    """Yield decoded text deltas for each generated token (special tokens preserved)."""
    model = engine.model
    tokenizer = engine.tokenizer
    device = engine.device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    eos_tensor = torch.tensor(list(_get_eos_set(tokenizer)), device=device)
    generated_ids: list[int] = []
    prev_decoded = ""
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        logits, past_kv = unpack_output_with_kv(out)
        for _ in range(max_new_tokens):
            next_token = _sample_next(
                logits, temperature, top_k, top_p, repetition_penalty, generated_ids
            )
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if (next_token == eos_tensor).any().item():
                break
            # Incremental decode: full sequence diff handles multi-byte tokens
            current = tokenizer.decode(generated_ids, skip_special_tokens=False)
            delta = current[len(prev_decoded):]
            prev_decoded = current
            if delta:
                yield delta
            out = model(next_token, past_key_values=past_kv, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)


def _build_prompt(request: ChatCompletionRequest) -> str:
    """Convert OpenAI messages + tools to Qwen chat format."""
    messages = [m.model_dump() for m in request.messages]
    tools = None
    if request.tools:
        tools = [
            {
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": t.function.parameters or {},
            }
            for t in request.tools
        ]
    return qwen_render_messages(messages, tools=tools, add_generation_prompt=True)


def _parse_tool_calls_openai(raw_text: str) -> tuple[list[dict], str]:
    """Parse tool calls from raw model output. Returns (openai_tool_calls, content)."""
    calls, musing = qwen_parse_tool_calls(raw_text)
    if not calls:
        return [], raw_text.strip()
    openai_calls = []
    for call in calls:
        openai_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", call.get("args", {}))),
            },
        })
    return openai_calls, musing


def _strip_markers(text: str) -> str:
    """Remove special-token markers from text."""
    for marker in _SPECIAL_MARKERS:
        text = text.replace(marker, "")
    return text


# ── SSE helpers ────────────────────────────────────────────────────────────


def _chat_sse_chunk(
    completion_id: str, model: str,
    delta: dict | None = None, finish_reason: str | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _completion_sse_chunk(
    completion_id: str, model: str,
    text: str = "", finish_reason: str | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(title="ForgeAI OpenAI-Compatible Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "forgeai",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = get_engine()
    prompt = _build_prompt(request)
    temperature = request.temperature or 0.0
    top_k = request.top_k or 80
    top_p = request.top_p or 1.0
    max_tokens = request.max_tokens or 512
    rep_penalty = request.repetition_penalty or 1.05
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if request.stream:
        return StreamingResponse(
            _stream_chat(
                engine, prompt, completion_id, request.model,
                max_tokens, temperature, top_k, top_p, rep_penalty,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    raw_text = generate_full(
        engine, prompt, max_new_tokens=max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=rep_penalty,
    )
    tool_calls, content = _parse_tool_calls_openai(raw_text)
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else "stop"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_chat(engine, prompt, completion_id, model, max_tokens,
                 temperature, top_k, top_p, rep_penalty):
    """Generator yielding SSE events for streaming chat completions."""
    accumulated = ""
    in_tool_call = False

    # Initial role chunk
    yield _chat_sse_chunk(completion_id, model, delta={"role": "assistant", "content": ""})

    for delta in stream_tokens(
        engine, prompt, max_new_tokens=max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=rep_penalty,
    ):
        accumulated += delta
        # Detect tool-call regions: between TOOL_CALL_START and TOOL_CALL_END
        starts = accumulated.count(TOOL_CALL_START)
        ends = accumulated.count(TOOL_CALL_END)
        currently_in_tc = starts > ends
        if currently_in_tc:
            in_tool_call = True
            continue
        if in_tool_call and not currently_in_tc:
            in_tool_call = False
            continue
        # Stream content delta (strip special markers)
        clean = _strip_markers(delta)
        if clean:
            yield _chat_sse_chunk(completion_id, model, delta={"content": clean})

    # Parse tool calls from full accumulated text
    tool_calls, _ = _parse_tool_calls_openai(accumulated)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield _chat_sse_chunk(completion_id, model, delta={
                "tool_calls": [{
                    "index": i, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": ""},
                }]
            })
            yield _chat_sse_chunk(completion_id, model, delta={
                "tool_calls": [{"index": i, "function": {"arguments": tc["function"]["arguments"]}}]
            })
        yield _chat_sse_chunk(completion_id, model, finish_reason="tool_calls")
    else:
        yield _chat_sse_chunk(completion_id, model, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    engine = get_engine()
    temperature = request.temperature or 0.0
    top_k = request.top_k or 80
    top_p = request.top_p or 1.0
    max_tokens = request.max_tokens or 512
    rep_penalty = request.repetition_penalty or 1.05
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if request.stream:
        return StreamingResponse(
            _stream_completion(
                engine, request.prompt, completion_id, request.model,
                max_tokens, temperature, top_k, top_p, rep_penalty,
            ),
            media_type="text/event-stream",
        )

    raw_text = generate_full(
        engine, request.prompt, max_new_tokens=max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=rep_penalty,
    )
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "text": _strip_markers(raw_text), "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_completion(engine, prompt, completion_id, model, max_tokens,
                       temperature, top_k, top_p, rep_penalty):
    """Generator yielding SSE events for streaming text completions."""
    for delta in stream_tokens(
        engine, prompt, max_new_tokens=max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=rep_penalty,
    ):
        clean = _strip_markers(delta)
        if clean:
            yield _completion_sse_chunk(completion_id, model, text=clean)
    yield _completion_sse_chunk(completion_id, model, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── Entry point ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ForgeAI OpenAI-compatible server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    global _checkpoint_path
    _checkpoint_path = args.checkpoint

    # Pre-load engine so the first request is fast
    get_engine()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
