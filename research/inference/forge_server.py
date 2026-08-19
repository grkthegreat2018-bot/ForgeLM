"""ForgeAI OpenAI-compatible HTTP server.

FastAPI server that wraps ModelRegistry with OpenAI-compatible endpoints.
Supports SSE streaming, model routing, sleep/wake management, tool calling,
Prometheus-compatible health checks, **concurrent task-based generation**
with session management, and **request batching** for multi-task throughput.

Surpasses LM Studio's proxy-only approach by serving models directly
with all ForgeAI optimizations (MTP, QuaRot, ProgressiveKV, batched decode,
session-based task continuity, etc.).

Usage:
    python research/inference/forge_server.py --models lfm2.5,qwen2.5

    # Or programmatically:
    from research.inference.forge_server import ForgeServer
    server = ForgeServer()
    server.register("forgelm-v3", checkpoint="...", config="forgelm_v3")
    server.serve(port=8000)

Task-based concurrent generation:
    POST /v1/tasks                    — create a new task (returns task_id)
    POST /v1/tasks/{task_id}/messages — append a user message + generate
    GET  /v1/tasks/{task_id}          — get task status + history
    GET  /v1/tasks                    — list all active tasks
    DELETE /v1/tasks/{task_id}        — delete a task
"""
import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research.inference.model_registry import ModelRegistry, VRAMBudgetExceeded
from research.inference.session_manager import (
    SessionManager, BatchQueue, TaskBootConfig,
)
from research.paths import LFM25_CHECKPOINT, LFM25_HF_DIR
from research.self_play.discovery.qwen_adapter import (
    IM_END,
    IM_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen_parse_tool_calls,
    qwen_render_messages,
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None

class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "lfm2.5-1.2b"
    messages: list[ChatMessage]
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Any] = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[list[str]] = None
    repetition_penalty: float = 1.05

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict | None = None
    finish_reason: str | None = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)

class CompletionRequest(BaseModel):
    model: str = "lfm2.5-1.2b"
    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[list[str]] = None
    repetition_penalty: float = 1.05

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "forgeai"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

class SleepRequest(BaseModel):
    level: int = Field(default=1, ge=1, le=2)


# ── Task API models ──────────────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    model: str = "lfm2.5-1.2b"
    system_prompt: str = ""
    seed: Optional[int] = None  # default seed for this task's generations
    boot_params: Optional[dict] = None  # per-task engine boot configuration
    metadata: Optional[dict] = None

class TaskMessageRequest(BaseModel):
    content: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    repetition_penalty: float = 1.05
    seed: Optional[int] = None  # override task's default seed for this message
    stream: bool = False
    stop: Optional[list[str]] = None
    tools: Optional[list[ToolDefinition]] = None

class UpdateBootConfigRequest(BaseModel):
    """Update a task's boot-time engine configuration."""
    kv_cache: Optional[str] = None  # standard, paged, hadamard_int4, snapkv, snapkv_4bit
    decoding: Optional[str] = None  # standard, speculative, batched, mtp_selfspec
    quantize: Optional[str] = None  # None, int8, int4, fp8
    acceleration: Optional[str] = None  # None, cuda_graph, airllm_streaming
    kv_cache_tokens: Optional[int] = None  # Limit KV cache allocation
    kv_bits: Optional[int] = None  # 4 or 8
    mrl_keep_ratio: Optional[float] = None
    use_v0_warm: Optional[bool] = None
    use_progressive_kv: Optional[bool] = None
    use_compile: Optional[bool] = None
    use_triton_conv: Optional[bool] = None
    use_prefix_cache: Optional[bool] = None
    use_spec_attn: Optional[bool] = None
    warmup: Optional[bool] = None

class TaskResponse(BaseModel):
    task_id: str
    model: str
    status: str = "active"
    created_at: float
    messages: int
    total_tokens: int

class TaskListResponse(BaseModel):
    object: str = "list"
    data: list[dict]

class TaskMessageResponse(BaseModel):
    task_id: str
    role: str = "assistant"
    content: str
    created_at: float


# ── Tool-call helpers ────────────────────────────────────────────────────────

_SPECIAL_MARKERS = (TOOL_CALL_START, TOOL_CALL_END, IM_START, IM_END)


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


# ── Server ───────────────────────────────────────────────────────────────────

class ForgeServer:
    """FastAPI server wrapping ModelRegistry with tool-calling support.

    Features:
      - OpenAI-compatible /v1/chat/completions and /v1/completions
      - Task-based concurrent generation via /v1/tasks endpoints
      - Request batching via BatchQueue (multiple requests → single forward pass)
      - Session management with LRU eviction
      - SSE streaming for both chat and task endpoints
      - Model sleep/wake for VRAM management
    """

    def __init__(self, registry: ModelRegistry | None = None,
                 batch_window_ms: int = 50, max_batch_size: int = 8,
                 max_sessions: int = 64):
        self.registry = registry or ModelRegistry()
        self.session_manager = SessionManager(max_sessions=max_sessions)
        self.batch_queue = BatchQueue(
            self.registry, self.session_manager,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
        )
        self.app = FastAPI(title="ForgeAI Inference Server", version="3.0.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._setup_routes()

    def register(self, model_id: str, checkpoint: str, config_name: str,
                 tokenizer_path: str | None = None,
                 vram_budget_gb: float = 0, **kwargs):
        """Register a model with the registry."""
        tok_path = tokenizer_path or str(LFM25_HF_DIR)
        self.registry.register(
            model_id, checkpoint, config_name,
            tokenizer_path=tok_path, vram_budget_gb=vram_budget_gb, **kwargs)

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the HTTP server (blocking)."""
        self.batch_queue.start()
        print(f"\n  {'='*60}")
        print(f"  ForgeAI Inference Server v3.0.0")
        print(f"  Listening on http://{host}:{port}")
        print(f"  Models: {[m['id'] for m in self.registry.list_models()]}")
        print(f"  Batch queue: window={self.batch_queue.batch_window*1000:.0f}ms, "
              f"max_batch={self.batch_queue.max_batch_size}")
        print(f"  Session manager: max={self.session_manager.max_sessions} tasks")
        print(f"  Endpoints: /v1/chat/completions, /v1/completions, /v1/models,")
        print(f"             /v1/tasks, /v1/tasks/{{id}}/messages,")
        print(f"             /v1/tasks/{{id}}/config (PATCH), /health")
        print(f"  {'='*60}\n")
        try:
            uvicorn.run(self.app, host=host, port=port, log_level="warning")
        finally:
            self.batch_queue.stop()

    def _build_prompt(self, req: ChatCompletionRequest) -> str:
        """Build prompt from chat messages, using qwen_adapter when tools are present."""
        messages = [m.model_dump() for m in req.messages]
        if req.tools:
            tools = [
                {
                    "name": t.function.name,
                    "description": t.function.description or "",
                    "parameters": t.function.parameters or {},
                }
                for t in req.tools
            ]
            return qwen_render_messages(messages, tools=tools, add_generation_prompt=True)
        # No tools: simple chat template
        return qwen_render_messages(messages, add_generation_prompt=True)

    def _setup_routes(self):
        app = self.app
        registry = self.registry

        @app.get("/health")
        async def health():
            """Health check with Prometheus-compatible metrics."""
            stats = registry.stats()
            return {
                "status": "ok",
                "version": "3.0.0",
                "models_loaded": stats["total_models"],
                "models_awake": stats["awake"],
                "vram_total_gb": round(stats["total_vram_gb"], 2),
                "vram_free_gb": round(stats["free_vram_gb"], 2),
                "active_tasks": len(self.session_manager.list_tasks()),
                "batch_queue_pending": len(self.batch_queue._pending),
            }

        @app.get("/v1/models")
        async def list_models():
            """OpenAI-compatible model list."""
            models = registry.list_models()
            now = int(time.time())
            return ModelListResponse(
                data=[
                    ModelInfo(id=m["id"], created=now, owned_by="forgeai")
                    for m in models
                ],
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(req: ChatCompletionRequest):
            """OpenAI-compatible chat completions with SSE streaming and tool calling."""
            prompt = self._build_prompt(req)

            if req.stream:
                return StreamingResponse(
                    self._stream_chat(req, prompt),
                    media_type="text/event-stream",
                )

            # Non-streaming
            raw_text = registry.generate(
                req.model, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            tool_calls, content = _parse_tool_calls_openai(raw_text)
            message: dict[str, Any] = {"role": "assistant", "content": content or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            finish = "tool_calls" if tool_calls else "stop"
            resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            return ChatCompletionResponse(
                id=resp_id,
                created=int(time.time()),
                model=req.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content or None,
                                        tool_calls=tool_calls or None),
                    finish_reason=finish,
                )],
            )

        @app.post("/v1/completions")
        async def completions(req: CompletionRequest):
            """OpenAI-compatible text completions."""
            if req.stream:
                return StreamingResponse(
                    self._stream_completion(req),
                    media_type="text/event-stream",
                )
            raw_text = registry.generate(
                req.model, req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
            return {
                "id": resp_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "text": _strip_markers(raw_text),
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        @app.post("/v1/models/{model_id}/sleep")
        async def sleep_model(model_id: str, req: SleepRequest = SleepRequest()):
            """Put a model to sleep to free VRAM."""
            if model_id not in [m["id"] for m in registry.list_models()]:
                raise HTTPException(404, f"Model '{model_id}' not found")
            registry.sleep(model_id, level=req.level)
            return {"status": "ok", "model": model_id, "level": req.level}

        @app.post("/v1/models/{model_id}/wake")
        async def wake_model(model_id: str):
            """Wake a sleeping model."""
            if model_id not in [m["id"] for m in registry.list_models()]:
                raise HTTPException(404, f"Model '{model_id}' not found")
            try:
                registry.wake(model_id)
            except VRAMBudgetExceeded as e:
                raise HTTPException(507, str(e))
            return {"status": "ok", "model": model_id}

        @app.get("/v1/models/{model_id}/stats")
        async def model_stats(model_id: str):
            """Get detailed stats for a specific model."""
            models = registry.list_models()
            for m in models:
                if m["id"] == model_id:
                    return m
            raise HTTPException(404, f"Model '{model_id}' not found")

        # ── Task API: concurrent task-based generation ───────────────────

        @app.post("/v1/tasks")
        async def create_task(req: CreateTaskRequest):
            """Create a new task with its own conversation context.

            Returns a task_id that can be used to send messages and
            maintain conversation continuity across multiple requests.
            Multiple tasks can run concurrently — the batch queue will
            group them into batched forward passes for throughput.
            """
            # Verify model exists
            if req.model not in [m["id"] for m in registry.list_models()]:
                raise HTTPException(404, f"Model '{req.model}' not registered")
            task_id = self.session_manager.create_task(
                model_id=req.model,
                system_prompt=req.system_prompt,
                seed=req.seed,
                boot_config=req.boot_params,
            )
            return {
                "task_id": task_id,
                "model": req.model,
                "status": "active",
                "seed": req.seed,
                "boot_params": req.boot_params or {},
                "created_at": time.time(),
            }

        @app.post("/v1/tasks/{task_id}/messages")
        async def send_task_message(task_id: str, req: TaskMessageRequest):
            """Send a message to a task and get a response.

            The task's conversation history is maintained server-side,
            so each continuation request only needs the new user message.
            Multiple tasks can send messages concurrently — they will be
            batched into a single forward pass when possible.
            """
            session = self.session_manager.get_task(task_id)
            if session is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            # Build prompt from session history + new message
            messages = []
            if session.system_prompt:
                messages.append({"role": "system", "content": session.system_prompt})
            messages.extend(session.history)
            messages.append({"role": "user", "content": req.content})

            # Render with tools if present
            if req.tools:
                tools = [
                    {
                        "name": t.function.name,
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {},
                    }
                    for t in req.tools
                ]
                prompt = qwen_render_messages(
                    messages, tools=tools, add_generation_prompt=True)
            else:
                prompt = qwen_render_messages(
                    messages, add_generation_prompt=True)

            # Record user message in session
            self.session_manager.append_message(task_id, "user", req.content)

            # Resolve seed: message-level seed overrides task-level seed
            effective_seed = req.seed if req.seed is not None else session.seed

            if req.stream:
                # Streaming via SSE
                stream_queue = asyncio.Queue()
                future = self.batch_queue.submit(
                    task_id=task_id,
                    model_id=session.model_id,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    seed=effective_seed,
                    stop=req.stop,
                    stream=True,
                    stream_queue=stream_queue,
                )
                return StreamingResponse(
                    self._stream_task(task_id, stream_queue, future),
                    media_type="text/event-stream",
                )

            # Non-streaming: submit to batch queue and wait
            future = self.batch_queue.submit(
                task_id=task_id,
                model_id=session.model_id,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                seed=effective_seed,
                stop=req.stop,
                stream=False,
            )
            try:
                # Wait for result (with timeout)
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, future.result),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                raise HTTPException(504, "Generation timed out")
            except Exception as e:
                raise HTTPException(500, str(e))

            # Parse tool calls if present
            tool_calls, content = _parse_tool_calls_openai(result)

            return {
                "task_id": task_id,
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or None,
                "created_at": time.time(),
            }

        @app.get("/v1/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get task status and conversation history."""
            session = self.session_manager.get_task(task_id)
            if session is None:
                raise HTTPException(404, f"Task '{task_id}' not found")
            return {
                "task_id": task_id,
                "model": session.model_id,
                "status": "active",
                "created_at": session.created_at,
                "last_used": session.last_used,
                "messages": session.history,
                "total_tokens": session.total_tokens,
                "seed": session.seed,
                "boot_config": session.boot_config.to_dict(),
            }

        @app.patch("/v1/tasks/{task_id}/config")
        async def update_task_config(task_id: str, req: UpdateBootConfigRequest):
            """Update a task's boot-time engine configuration.

            The new config will be applied on the next generation request.
            This allows live reconfiguration of KV cache strategy, decoding
            strategy, quantization, acceleration, and other engine params
            without recreating the task or losing conversation history.

            Only fields that are set (non-None) will be updated; others
            retain their current values.
            """
            session = self.session_manager.get_task(task_id)
            if session is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            # Merge: only update fields that are set in the request
            current = session.boot_config.to_dict()
            updates = {k: v for k, v in req.model_dump().items()
                       if v is not None}
            current.update(updates)
            new_config = TaskBootConfig.from_dict(current)

            self.session_manager.update_boot_config(task_id, new_config)
            return {
                "status": "updated",
                "task_id": task_id,
                "boot_config": new_config.to_dict(),
            }

        @app.get("/v1/tasks")
        async def list_tasks():
            """List all active tasks."""
            tasks = self.session_manager.list_tasks()
            return {"object": "list", "data": tasks}

        @app.delete("/v1/tasks/{task_id}")
        async def delete_task(task_id: str):
            """Delete a task and free its session context."""
            deleted = self.session_manager.delete_task(task_id)
            if not deleted:
                raise HTTPException(404, f"Task '{task_id}' not found")
            return {"status": "deleted", "task_id": task_id}

    async def _stream_chat(self, req: ChatCompletionRequest, prompt: str):
        """SSE streaming generator for chat completions with tool-call detection."""
        resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def _sse_chunk(delta: dict, finish_reason: str | None = None) -> str:
            chunk_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            # Initial role chunk
            yield _sse_chunk({"role": "assistant", "content": ""})

            accumulated = ""
            in_tool_call = False
            for text_chunk in self.registry.generate_stream(
                req.model, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                accumulated += text_chunk
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
                clean = _strip_markers(text_chunk)
                if clean:
                    yield _sse_chunk({"content": clean})

            # Parse tool calls from full accumulated text
            tool_calls, _ = _parse_tool_calls_openai(accumulated)
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    yield _sse_chunk({"tool_calls": [{
                        "index": i, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": ""},
                    }]})
                    yield _sse_chunk({"tool_calls": [{
                        "index": i,
                        "function": {"arguments": tc["function"]["arguments"]},
                    }]})
                yield _sse_chunk({}, finish_reason="tool_calls")
            else:
                yield _sse_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "error"}],
                "error": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_task(self, task_id: str, stream_queue: asyncio.Queue,
                           future):
        """SSE streaming generator for task-based generation."""
        resp_id = f"task-{task_id}"
        created = int(time.time())

        def _sse_chunk(delta: dict, finish_reason: str | None = None) -> str:
            chunk_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "task_id": task_id,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            yield _sse_chunk({"role": "assistant", "content": ""})

            while True:
                chunk = await stream_queue.get()
                if chunk is None:  # sentinel = end of stream
                    break
                clean = _strip_markers(chunk)
                if clean:
                    yield _sse_chunk({"content": clean})

            yield _sse_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "task_id": task_id,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "error"}],
                "error": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_completion(self, req: CompletionRequest):
        """SSE streaming generator for text completions."""
        resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def _sse_chunk(text: str = "", finish_reason: str | None = None) -> str:
            chunk_data = {
                "id": resp_id,
                "object": "text_completion",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "text": text,
                             "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            for text_chunk in self.registry.generate_stream(
                req.model, req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                clean = _strip_markers(text_chunk)
                if clean:
                    yield _sse_chunk(text=clean)
            yield _sse_chunk(finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "text_completion",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "text": "",
                             "finish_reason": "error"}],
                "error": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"


# ── CLI ──────────────────────────────────────────────────────────────────────

# Default model registrations
DEFAULT_MODELS = {
    "lfm2.5-1.2b": {
        "checkpoint": str(LFM25_CHECKPOINT),
        "config": "forgelm_v3",
        "tokenizer": str(LFM25_HF_DIR),
        "vram_gb": 2.5,
    },
}


def main():
    parser = argparse.ArgumentParser(description="ForgeAI Inference Server v3.0")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--models", type=str, default="lfm2.5-1.2b",
                        help="Comma-separated model IDs to load")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--config", type=str, default="forgelm_v3",
                        help="Model config preset")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Override tokenizer path")
    parser.add_argument("--vram-gb", type=float, default=0,
                        help="VRAM budget in GB (0=auto)")
    parser.add_argument("--batch-window-ms", type=int, default=50,
                        help="Batch collection window in ms (default 50)")
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Maximum requests per batch (default 8)")
    parser.add_argument("--max-sessions", type=int, default=64,
                        help="Maximum concurrent task sessions (default 64)")
    args = parser.parse_args()

    server = ForgeServer(
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.max_batch_size,
        max_sessions=args.max_sessions,
    )

    # Register models
    model_ids = [m.strip() for m in args.models.split(",")]
    for mid in model_ids:
        if mid in DEFAULT_MODELS and not args.checkpoint:
            spec = DEFAULT_MODELS[mid]
            server.register(
                mid, spec["checkpoint"], spec["config"],
                tokenizer_path=spec.get("tokenizer"),
                vram_budget_gb=args.vram_gb or spec.get("vram_gb", 0),
            )
        elif args.checkpoint:
            server.register(
                mid, args.checkpoint, args.config,
                tokenizer_path=args.tokenizer,
                vram_budget_gb=args.vram_gb,
            )
        else:
            print(f"  [WARN] No checkpoint for '{mid}', skipping. "
                  f"Use --checkpoint to specify.")

    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
