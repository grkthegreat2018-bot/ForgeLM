"""ForgeAI OpenAI-compatible HTTP server.

FastAPI server that wraps ModelRegistry with OpenAI-compatible endpoints.
Supports SSE streaming, model routing, sleep/wake management, and
Prometheus-compatible health checks.

Surpasses LM Studio's proxy-only approach by serving models directly
with all ForgeAI optimizations (MTP, QuaRot, ProgressiveKV, etc.).

Usage:
    python research/inference/forge_server.py --models lfm2.5,qwen2.5

    # Or programmatically:
    from research.inference.forge_server import ForgeServer
    server = ForgeServer()
    server.register("forgelm-v3", checkpoint="...", config="forgelm_v3")
    server.serve(port=8000)
"""
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research.inference.model_registry import ModelRegistry, VRAMBudgetExceeded
from research.paths import LFM25_CHECKPOINT, LFM25_HF_DIR


# ── Pydantic models ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "lfm2.5-1.2b"
    messages: list[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False

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


# ── Server ───────────────────────────────────────────────────────────────────

class ForgeServer:
    """FastAPI server wrapping ModelRegistry."""

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.app = FastAPI(title="ForgeAI Inference Server", version="2.0.0")
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
        print(f"\n  {'='*60}")
        print(f"  ForgeAI Inference Server v2.0.0")
        print(f"  Listening on http://{host}:{port}")
        print(f"  Models: {[m['id'] for m in self.registry.list_models()]}")
        print(f"  Endpoints: /v1/chat/completions, /v1/models, /health")
        print(f"  {'='*60}\n")
        uvicorn.run(self.app, host=host, port=port, log_level="warning")

    def _setup_routes(self):
        app = self.app
        registry = self.registry

        @app.get("/health")
        async def health():
            """Health check with Prometheus-compatible metrics."""
            stats = registry.stats()
            return {
                "status": "ok",
                "version": "2.0.0",
                "models_loaded": stats["total_models"],
                "models_awake": stats["awake"],
                "vram_total_gb": round(stats["total_vram_gb"], 2),
                "vram_free_gb": round(stats["free_vram_gb"], 2),
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
            """OpenAI-compatible chat completions with SSE streaming."""
            # Build prompt from messages
            prompt = self._build_prompt(req.messages)

            if req.stream:
                return StreamingResponse(
                    self._stream_response(req.model, prompt, req),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming response
                output = registry.generate(
                    req.model, prompt,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                return ChatCompletionResponse(
                    id=resp_id,
                    created=int(time.time()),
                    model=req.model,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=output),
                        finish_reason="stop",
                    )],
                )

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

    @staticmethod
    def _build_prompt(messages: list[ChatMessage]) -> str:
        """Build a single prompt string from chat messages."""
        parts = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            else:
                parts.append(content)
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    async def _stream_response(self, model_id: str, prompt: str,
                               req: ChatCompletionRequest):
        """SSE streaming generator."""
        resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # For streaming, we generate in chunks by calling generate()
        # with small max_new_tokens repeatedly. A real implementation
        # would use token-level streaming from the engine.
        try:
            # Generate all tokens at once (future: token-level streaming)
            full_output = self.registry.generate(
                model_id, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                finish_sentence=False,
            )

            # Stream token by token (split on spaces for word-level chunks)
            words = full_output.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                delta = {"content": chunk}
                chunk_data = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)  # Simulate token-level timing

            # Final chunk with finish_reason
            final_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {},
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
    parser = argparse.ArgumentParser(description="ForgeAI Inference Server")
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
    args = parser.parse_args()

    server = ForgeServer()

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
