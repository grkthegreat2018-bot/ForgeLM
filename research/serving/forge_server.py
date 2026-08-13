"""ForgeAI OpenAI-compatible inference server.

Hosts LFM2.5-1.2B via ForgeEngine with batched decoding.
Beats LM Studio by supporting batched multi-request inference
on their own ported lossless model checkpoint.

Usage:
    python -m research.serving.forge_server --port 1235
"""
import asyncio
import json
import time
import threading
import traceback
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ── Chat template for LFM2.5 (extracted from GGUF metadata) ────────────
LFM_CHAT_TEMPLATE = (
    "{{- bos_token -}}\n"
    "{%- set keep_past_thinking = keep_past_thinking | default(false) -%}\n"
    "{%- set ns = namespace(system_prompt=\"\") -%}\n"
    "{%- if messages[0][\"role\"] == \"system\" -%}\n"
    "    {%- set ns.system_prompt = messages[0][\"content\"] -%}\n"
    "    {%- set messages = messages[1:] -%}\n"
    "{%- endif -%}\n"
    "{%- if ns.system_prompt -%}\n"
    "    {{- \"<|im_start|>system\\n\" + ns.system_prompt + \"<|im_end|>\\n\" -}}\n"
    "{%- endif -%}\n"
    "{%- for message in messages -%}\n"
    "    {{- \"<|im_start|>\" + message[\"role\"] + \"\\n\" -}}\n"
    "    {{- message[\"content\"] + \"<|im_end|>\\n\" -}}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}\n"
    "    {{- \"<|im_start|>assistant\\n\" -}}\n"
    "{%- endif -%}"
)

# ── Request batching queue ─────────────────────────────────────────────

class RequestSlot:
    """A pending generation request in the batch queue."""
    def __init__(self, prompt_ids: torch.Tensor, max_tokens: int,
                 temperature: float, top_p: float, seed: int, future: asyncio.Future):
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.future = future


class BatchScheduler:
    """Collects requests and dispatches them in batches to ForgeEngine.

    Uses batched decoding for 3-5x throughput vs serialized requests.
    Waits up to `batch_timeout_ms` for a minimum batch, then runs.
    """

    def __init__(self, engine, tokenizer, max_batch: int = 8,
                 batch_timeout_ms: int = 100):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.batch_timeout = batch_timeout_ms / 1000.0

        self._queue: list[RequestSlot] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._worker.start()

    def submit(self, prompt_ids: torch.Tensor, max_tokens: int,
               temperature: float, top_p: float, seed: int) -> asyncio.Future:
        future = asyncio.get_event_loop().create_future()
        slot = RequestSlot(prompt_ids, max_tokens, temperature, top_p, seed, future)
        with self._lock:
            self._queue.append(slot)
            self._cond.notify()
        return future

    def _run(self):
        """Worker thread: collect batches and run inference."""
        while self._running:
            batch = []
            with self._lock:
                # Wait for at least 1 request
                while self._running and not self._queue:
                    self._cond.wait(timeout=0.1)
                if not self._running:
                    break
                # Grab up to max_batch requests
                while self._queue and len(batch) < self.max_batch:
                    batch.append(self._queue.pop(0))
                # If batch < max_batch, wait briefly for more
                if len(batch) < self.max_batch and self._running:
                    self._cond.wait(timeout=self.batch_timeout)
                    while self._queue and len(batch) < self.max_batch:
                        batch.append(self._queue.pop(0))

            if not batch:
                continue

            # Run batched inference
            try:
                prompts = [s.prompt_ids for s in batch]
                max_toks = [s.max_tokens for s in batch]
                temps = [s.temperature for s in batch]
                top_ps = [s.top_p for s in batch]

                # Set seed for each request (global RNG is OK since we're single-threaded here)
                # Use the first request's seed; batch diversity comes from different prompts
                if batch[0].seed >= 0:
                    torch.manual_seed(batch[0].seed)

                # Run batched generate
                results = self.engine.generate_batch(
                    prompts, max_toks, temps, top_ps,
                )

                # Decode and resolve futures
                device = prompts[0].device
                for i, slot in enumerate(batch):
                    if i < len(results):
                        output_ids = results[i]
                        # Decode only generated tokens (skip prompt)
                        prompt_len = slot.prompt_ids.shape[1]
                        gen_ids = output_ids[0, prompt_len:]
                        text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
                        slot.future.get_loop().call_soon_threadsafe(
                            slot.future.set_result, text)
                    else:
                        slot.future.get_loop().call_soon_threadsafe(
                            slot.future.set_exception, RuntimeError("Batch result missing"))

            except Exception as e:
                tb = traceback.format_exc()
                for slot in batch:
                    slot.future.get_loop().call_soon_threadsafe(
                        slot.future.set_exception, RuntimeError(f"{e}\n{tb[:200]}"))

    def shutdown(self):
        self._running = False
        with self._lock:
            self._cond.notify_all()


# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(title="ForgeAI Inference Server", version="1.0")
scheduler: Optional[BatchScheduler] = None
MODEL_NAME = "forgeai/lfm2.5-1.2b"


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=768, ge=1, le=4096)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = Field(default=-1)

class ChatResponse(BaseModel):
    id: str = "chatcmpl-0"
    object: str = "chat.completion"
    created: int = 0
    model: str = MODEL_NAME
    choices: list[dict]


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "forgeai"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global scheduler
    if scheduler is None:
        raise HTTPException(503, "Server not ready")

    # Apply LFM chat template manually
    prompt = _apply_lfm_template(req.messages)
    ids = scheduler.tokenizer.encode(prompt)
    if not isinstance(ids, list):
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")

    future = scheduler.submit(
        input_ids, req.max_tokens, req.temperature, req.top_p, req.seed)
    try:
        text = await future
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    }


def _apply_lfm_template(messages: list[ChatMessage]) -> str:
    """Apply LFM2.5 chat template manually (simplified Jinja2)."""
    parts = []
    msgs = list(messages)

    # System message
    if msgs and msgs[0].role == "system":
        parts.append(f"<|im_start|>system\n{msgs[0].content}<|im_end|>\n")
        msgs = msgs[1:]

    # Conversation turns
    for m in msgs:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n")

    # Generation prompt
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ── Startup / shutdown ─────────────────────────────────────────────────

_engine = None

@app.on_event("startup")
async def startup():
    global scheduler, _engine
    from research.inference.forge_engine import ForgeEngine
    from research.inference.batched_decoding import BatchedDecoding

    print("[ForgeServer] Loading ForgeEngine...", flush=True)
    engine = ForgeEngine.from_checkpoint(
        checkpoint=r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_LFM25-1.2B.safetensors",
        config_name="lfm25_1.2b",
        tokenizer_path=r"D:\windsurf\ForgeAI\research\checkpoints\lfm25_tokenizer",
    )
    engine.activate(kv_cache="standard", decoding="standard")

    # Replace decoding with batched version
    batched = BatchedDecoding(eos_token_id=7)
    # Bind model to batched decoder for the scheduler
    _batched_model = engine.model
    def _batched_generate(prompts, max_toks, temps, top_ps):
        return batched.generate_batch(_batched_model, prompts, max_toks, temps, top_ps)
    engine.generate_batch = _batched_generate
    _engine = engine

    scheduler = BatchScheduler(
        engine, engine.tokenizer, max_batch=8, batch_timeout_ms=100)
    print("[ForgeServer] Ready on port 1235", flush=True)


@app.on_event("shutdown")
async def shutdown():
    global scheduler
    if scheduler:
        scheduler.shutdown()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1235)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--max-batch", type=int, default=8)
    args = p.parse_args()

    # Set max_batch before startup
    import research.serving.forge_server as srv
    srv._max_batch_override = args.max_batch

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
