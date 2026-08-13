# ForgeEngine + Agent Swarm Revamp — Definitive Plan (v2, 2026-08-12)
## Goal: Surpass LM Studio + llama.cpp as the most optimal & moddable LLM loader/manager

---

## 0. Competitive Analysis — What They Have, What We Can Do Better

### 0.1 LM Studio architecture (reference)
- **model.yaml** open standard: single portable file describing model + all variants (GGUF, MLX, safetensors) with metadata, load options, custom logic (enable/disable thinking)
- **Plugin SDK** (TypeScript/Node.js v22): 4 component types — Tools Provider, Prompt Preprocessor, Generator (replaces local LLM entirely), Custom Config UI
- **Runtime backends** decoupled as independent extension packs (CUDA 12 llama.cpp, Vulkan, CPU) — user downloads only what they need
- **HuggingFace bridge**: app is a filtered catalog viewer, downloads GGUF from HF CDN, resumable
- **Weakness**: loads ONE model at a time, no hot-swap, no speculative decoding, no custom kernels, TypeScript plugins are slow for compute-heavy work

### 0.2 llama.cpp architecture (reference)
- **mmap() zero-copy loading**: GGUF file IS the tensor storage. No protobuf, no JSON-in-zip, no copy. Model loads in milliseconds via demand-paged virtual memory. Page cache does the rest.
- **GGUF format**: Magic "GGUF" (4 bytes) → version → tensor_count → kv_count → KV metadata → tensor info (name, dims, dtype, offset) → page-aligned tensor data blob. Under 200 lines to parse.
- **GGML compute graph**: 3-phase pipeline — build graph → dispatch to backend → execute. Hand-written SIMD kernels on quantized data. 46.7x faster than scalar FP32.
- **CUDA Graphs**: 1.2x speedup on H100, indirect copy pointers to avoid frequent graph updates
- **EAGLE-3** now in PR #18039 (2-3x speedup, SOTA speculative decoding)
- **Weakness**: single-stream by default, no native multi-model, C++ codebase hard to extend, no plugin system, no structured outputs, no agent framework

### 0.3 What ForgeAI UNIQUELY has (neither LM Studio nor llama.cpp has these)
| Capability | Description | Competitive Edge |
|---|---|---|
| Hybrid conv+attention native | LFM2.5 architecture: 10 double-gated conv + 6 GQA layers | llama.cpp treats all models as Transformers — no conv optimization |
| MTP speculative decoding | Multi-Token Prediction heads trained into the model | llama.cpp EAGLE-3 is add-on; MTP is baked into the architecture |
| QuaRot-KV | Hadamard-rotated KV cache for lossless 4-bit quantization | Unique — no other engine has this |
| RotorQuant | Weight rotation for improved quantization accuracy | Unique |
| V0-WarmStart | Value residual warm-start for faster KV cache setup | Unique |
| ProgressiveKV | Anchor + residual KV streams for efficient long-context | Unique |
| MRL-AdaptiveContext | Dynamic dimensionality reduction at inference time | Unique |
| L1 Speculative Attention | 57% attention compute cut, mathematically lossless | Unique |
| DSpark speculative decoding | Alternative speculative decoding variant | Unique (vs EAGLE/MTP only) |
| SnapKV | Smart KV cache eviction for long contexts | Competitive with vLLM |
| PreAllocated KVCache | Pre-allocated contiguous KV cache for zero-fragmentation | llama.cpp uses dynamic allocation |

### 0.4 The Gap We Fill — What ForgeAI will have that surpasses both
| Feature | LM Studio | llama.cpp | **ForgeAI v2** |
|---|---|---|---|
| Multi-format loader | GGUF only | GGUF only | **GGUF + safetensors + HF Hub direct** |
| mmap zero-copy | Via llama.cpp | Yes | **Yes (for GGUF) + safetensors fast path** |
| Hot-swap models | No (one at a time) | No | **Sleep/wake + concurrent multi-model with VRAM partitioning** |
| Plugin SDK | TypeScript/Node (4 types) | None | **Python native (8 types) + C extension API for kernel plugins** |
| Speculative decoding | No | EAGLE-3 (PR) | **MTP (native) + EAGLE-3 + DSpark + adaptive combo** |
| Structured outputs | No | No | **JSON-schema constrained decoding (outlines)** |
| Agent framework | No | No | **Orchestrator-worker with tool-calling schema** |
| OpenAI-compatible server | Yes (proxy) | Yes | **Yes + model routing + sleep/wake endpoints** |
| Observability | No | No | **Prometheus metrics + token-level tracing** |
| Architecture support | Transformer | Transformer | **Hybrid (conv+attn), Transformer, SSM** |
| KV cache innovations | Standard | Standard | **QuaRot + ProgressiveKV + SnapKV + V0** |

---

## 1. Architecture — The ForgeAI v2 Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    forge_server.py                          │
│  FastAPI: /v1/chat/completions, /v1/models, /v1/sleep, ... │
│  SSE streaming, model routing, rate limiting, auth          │
├─────────────────────────────────────────────────────────────┤
│                  model_registry.py                          │
│  Multi-engine VRAM manager, sleep/wake scheduler,           │
│  lazy model loading, concurrent model support               │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ ForgeEng │ ForgeEng │ ForgeEng │ ForgeEng │  Plugin Host   │
│ LFM2.5   │ Qwen2.5  │ (idle)   │ (idle)   │  Python + C    │
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│                    forge_loader.py                          │
│  GGUF mmap | safetensors | HuggingFace Hub | model.yaml    │
├─────────────────────────────────────────────────────────────┤
│                    forge_engine.py (fixed)                  │
│  KV cache: QuaRot, ProgressiveKV, SnapKV, Standard, Paged  │
│  Decoding: MTP, EAGLE-3, DSpark, Standard, Medusa          │
│  Quant: INT4, INT8, FP8, RotorQuant                        │
│  Accel: CUDA graphs, AirLLM streaming, torch.compile       │
├─────────────────────────────────────────────────────────────┤
│                    model_loader.py                          │
│  ConfigurableResearchLLM, PreAllocatedKVCache               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Implementation Plan

### Phase A — Engine Foundation (fixes + core capabilities)

#### A1. Fix critical bugs in forge_engine.py
- **A1.1** `_detect_keystack_features()` indentation: move feature detection OUT of the `else:` block so it works for both single-file AND sharded checkpoints. Currently silent no-op for sharded dirs.
- **A1.2** `_generate_streaming()`: change from per-token layer reload to per-forward-pass. Maintain KV cache across decode steps. Load all layer shards once per step, compute all layers, free. Currently O(tokens × layers) disk I/O.
- **A1.3** `_finish_to_stop()`: don't re-run full sequence for KV recovery. Use the existing KV cache from the generation step.
- **A1.4** Prefix cache: actually store the KV tensor, not `None`. Use hash of input_ids as key.

#### A2. Add sleep/wake to ForgeEngine
```python
def sleep(self, level: int = 1):
    """Level 1: offload weights to CPU, keep CUDA context.
       Level 2: discard weights entirely, keep tokenizer/config."""
    if level == 1:
        self.model.to('cpu', non_blocking=True)
        torch.cuda.empty_cache()
        self._awake = False
    elif level == 2:
        # Discard model, keep tokenizer + config + CUDA ctx
        self._stored_state = {
            'config': getattr(self.model, 'config', None),
            'dtype': next(self.model.parameters()).dtype,
        }
        del self.model
        torch.cuda.empty_cache()
        self._awake = False

def wake(self, checkpoint_path: str = None):
    """Restore model to GPU. Level 1: CPU→GPU copy. Level 2: reload from disk."""
    if hasattr(self, '_stored_state'):
        # Level 2 wake: rebuild model
        self.model = ModelLoader.build_model_fast(self._stored_state['config'], ...)
        del self._stored_state
    self.model.to(self.device)
    self._awake = True
```

#### A3. Build forge_loader.py — universal model loader
GGUF format parser (under 200 lines):
```
Header: "GGUF" (4B) | version (u32) | tensor_count (u64) | kv_count (u64)
KV metadata: key (string) | value_type (u32) | value (variable)
Tensor info: name (string) | n_dims (u32) | dims[] (u64) | dtype (u32) | offset (u64)
Tensor data: page-aligned binary blob at offsets
```

Support matrix:
- **GGUF**: mmap the file, parse header, create tensor views at offsets. No copy. <100ms load.
- **Safetensors**: existing fast path via `ModelLoader.build_model_fast()`.
- **HuggingFace Hub**: `from_pretrained` → convert to safetensors → cache locally.
- **model.yaml**: parse metadata, resolve sources, delegate to appropriate loader.

Auto-detection: check magic bytes at file start ("GGUF" → GGUF, first 8 bytes → check for safetensors JSON header → safetensors, else try HF).

#### A4. Build model_registry.py — multi-engine VRAM manager
```python
class ModelRegistry:
    engines: dict[str, ForgeEngine]      # model_id → engine
    budgets: dict[str, int]              # model_id → VRAM budget bytes
    sleeping: dict[str, bool]            # model_id → is asleep
    total_vram: int                      # from torch.cuda.get_device_properties

    def register(self, model_id, checkpoint, config, vram_budget_bytes):
        """Reserve VRAM budget and load model."""
        used = sum(b for mid, b in self.budgets.items() if not self.sleeping.get(mid))
        free = torch.cuda.mem_get_info()[0]
        if used + vram_budget_bytes > free * 0.9:
            # Try sleeping idle models first
            for mid, eng in self.engines.items():
                if not self.sleeping.get(mid) and mid != model_id:
                    eng.sleep(level=1)
                    if self._check_fit(model_id, vram_budget_bytes):
                        break
            else:
                raise VRAMBudgetExceeded(...)

    def switch(self, from_model, to_model):
        """Sleep one, wake another. <3s round-trip target."""
        self.engines[from_model].sleep(level=1)
        self.engines[to_model].wake()

    def load_concurrent(self, *model_ids):
        """Load all models at once, partitioning VRAM."""
        # For models small enough to fit together (e.g., LFM2.5 2.3GB + Qwen2.5-1.5B 3GB = 5.3GB on 12GB GPU)
```

Key differentiator: LM Studio loads one at a time. We support BOTH sleep/wake fast swap AND concurrent multi-model when VRAM allows.

#### A5. Build forge_server.py — OpenAI-compatible HTTP server
Endpoints:
- `POST /v1/chat/completions` — OpenAI-compatible, routes by `model` field. SSE streaming with `stream: true`.
- `GET /v1/models` — list registered models with status (awake/asleep/loading).
- `POST /v1/models/{id}/sleep` — put model to sleep (level=1 or 2).
- `POST /v1/models/{id}/wake` — wake model.
- `GET /v1/models/{id}/stats` — VRAM usage, tokens generated, uptime.
- `GET /health` — Prometheus metrics endpoint.

Streaming implementation:
```python
async def chat_completions_stream(engine, messages, **params):
    async def token_generator():
        for token in engine.generate_stream(messages, **params):
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

---

### Phase B — Speculative Decoding Supremacy

#### B1. Add EAGLE-3 draft head (surpass llama.cpp's PR #18039)
EAGLE-3 architecture for LFM2.5:
- Extract features from 3 layers (low: layer 1, mid: layer 7, high: layer 15)
- Concatenate → project to d_model → single decoder layer → predict k=4 tokens
- Training-time test: autoregressive simulation during training
- Dynamic draft tree (from EAGLE-2): prune low-confidence branches

Why this beats llama.cpp's EAGLE-3: we can fuse it with our existing MTP heads. LFM2.5 already has MTP trained in. EAGLE-3 draft head + MTP verification = combined speedup.

#### B2. Adaptive speculative decoding router
Auto-select best strategy per prompt:
- Code generation → n-gram + EAGLE (4.9x theoretical on InstructCoder workload)
- General chat → EAGLE-3 (1.96x at batch=1)
- Short prompts (<32 tokens) → MTP (lower overhead, no feature extraction)
- Long prompts (>512 tokens) → standard (draft overhead > benefit)

#### B3. L1 Speculative Attention integration
Already implemented. At batch=1, 57% attention compute cut compounds with speculative decoding for multiplicative speedup.

---

### Phase C — Plugin SDK (surpass LM Studio's 4-type system)

#### C1. Plugin architecture — 8 component types
| # | Type | Purpose | LM Studio Equivalent |
|---|---|---|---|
| 1 | **Backend** | Register custom model format loader (GGUF, safetensors, ONNX, custom) | (built-in only) |
| 2 | **Quantizer** | Custom quantization strategy (RotorQuant, BitNet, WANDA, custom) | (built-in only) |
| 3 | **KVCache** | Custom KV cache strategy (SnapKV, H2O, StreamingLLM, custom) | (not available) |
| 4 | **Decoder** | Custom decoding strategy (MTP, EAGLE, DSpark, Medusa, custom) | (not available) |
| 5 | **Attention** | Custom attention kernel (FlashAttention, FlexAttention, custom) | (not available) |
| 6 | **Preprocessor** | Modify input before model (RAG injection, prompt templates, safety filter) | Prompt Preprocessor |
| 7 | **Postprocessor** | Modify output after model (format enforcement, content filter, citation) | (not available) |
| 8 | **Generator** | Replace local model entirely (remote API adapter, ensemble, cascade) | Generator |

#### C2. Plugin SDK (Python)
```python
# research/plugins/base.py
class ForgePlugin:
    """Base class for all ForgeAI plugins."""
    name: str
    version: str
    author: str

class BackendPlugin(ForgePlugin):
    def can_load(self, source: str | Path) -> bool: ...
    def load(self, source, config: ModelConfig) -> nn.Module: ...

class QuantizerPlugin(ForgePlugin):
    def quantize(self, model: nn.Module, bits: int, **kwargs) -> nn.Module: ...
    def dequantize(self, model: nn.Module) -> nn.Module: ...

class KVCachePlugin(ForgePlugin):
    def create(self, n_heads, head_dim, n_kv, max_seq, device, dtype) -> KVCacheStrategy: ...

class DecoderPlugin(ForgePlugin):
    def create(self, model, **kwargs) -> DecodingStrategy: ...

class AttentionPlugin(ForgePlugin):
    def replace_attention(self, model: nn.Module) -> nn.Module: ...

class PreprocessorPlugin(ForgePlugin):
    async def process(self, messages: list[dict], context: dict) -> list[dict]: ...

class PostprocessorPlugin(ForgePlugin):
    async def process(self, text: str, context: dict) -> str: ...

class GeneratorPlugin(ForgePlugin):
    async def generate(self, messages: list[dict], **params) -> AsyncIterator[str]: ...
```

Plugin discovery: scan `research/plugins/installed/` for Python files with `@register_plugin` decorator. Entry points via `pyproject.toml` `[project.entry-points."forgeai.plugins"]`.

#### C3. C extension API (for kernel plugins)
Expose a minimal C ABI for custom CUDA/Vulkan kernels:
```c
// forge_plugin.h
typedef struct ForgeTensor {
    void* data;
    int64_t shape[8];
    int ndim;
    int dtype;  // FORGE_FP16, FORGE_BF16, FORGE_INT4, etc.
    int device; // FORGE_CUDA, FORGE_CPU
} ForgeTensor;

typedef void (*forge_kernel_fn)(ForgeTensor* inputs, int n_inputs,
                                 ForgeTensor* outputs, int n_outputs,
                                 void* params);
```

This is how we surpass LM Studio: their TypeScript plugins can't touch GPU kernels. Our Python SDK handles orchestration, C API handles hot-path compute.

---

### Phase D — Agent Swarm v2 (production-grade)

#### D1. Cut LM Studio dependency
Replace `LMSTUDIO_API = "http://localhost:1234/v1"` with `FORGE_API = "http://localhost:8000/v1"`.
Agents use `model` field to route: `"model": "lfm2.5-1.2b"` for LFM2.5, `"model": "qwen2.5-1.5b"` for Qwen.

#### D2. Structured outputs via outlines
```python
from outlines import generate

class DraftOutput(BaseModel):
    topic: str
    question: str
    answer: str
    sources: list[str]

# Constrained generation — model CANNOT produce invalid JSON
draft = generate.json(model, DraftOutput)(prompt)
```
Eliminates "Reply with ONLY the search query" drift. The model is literally constrained to produce valid JSON matching the schema.

#### D3. OpenAI function-calling schema
Expose tools as JSON schema in the API payload:
```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "web_search",
      "description": "Search the web for information",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
      }
    }
  }]
}
```
Model chooses when to call tools — not hardcoded phases. This is the key to "agents following orders correctly."

#### D4. Orchestrator-worker pattern
Replace fixed 5-phase pipeline with:
1. **Orchestrator agent** receives task, decomposes into subtasks with typed contracts
2. **Dispatcher** assigns subtasks to specialist worker agents (parallel where independent)
3. **Worker agents** execute with tool access, return typed results
4. **Aggregator** combines results, checks consistency
5. **Orchestrator** validates final output against task spec

Each agent has:
- `max_turns`: 10 (hard cap, prevents loops)
- `token_budget`: 4096 (total per agent per run)
- `output_schema`: Pydantic model (structured output enforced)

#### D5. Fix specific bugs
- Critique phase (line 382): exclude by `agent.name`, not index
- Drop `sudo-think` `imd` tags for LFM2.5 (use standard system/user/assistant format)
- Add retry with backoff on LM API failures
- Add timeout per agent call (30s default)

---

### Phase E — Observability & Developer Experience

#### E1. Prometheus metrics
```
forge_requests_total{model, status}
forge_tokens_generated_total{model}
forge_vram_usage_bytes{model, type}  # type=weights|kv_cache|overhead
forge_latency_seconds{model, quantile}
forge_sleep_wake_duration_seconds{model, direction}
```

#### E2. Token-level tracing
Log every token with: timestamp, model, agent_id, prompt_hash, latency_us.
Store in SQLite for offline analysis. Export as Chrome trace format.

#### E3. CLI tool
```bash
forge list                    # list registered models + status
forge load lfm2.5-1.2b        # load model
forge chat lfm2.5-1.2b        # interactive chat
forge serve                   # start server
forge plugin install <name>   # install plugin
forge benchmark               # run benchmark suite
```

---

### Phase F — Verification & Benchmarks

#### F1. Correctness
- `test_detect_features_sharded`: verify `_detect_keystack_features` works on sharded dirs
- `test_streaming_kv_cache`: verify streaming maintains correct KV state across tokens
- `test_multi_model_load`: load LFM2.5 + Qwen2.5-1.5B simultaneously, assert both respond with correct output
- `test_sleep_wake_roundtrip`: sleep level 1, wake, generate — output must match pre-sleep output
- `test_structured_output`: verify outlines-constrained generation never produces invalid JSON
- `test_tool_calling`: verify model calls `web_search` tool when asked to research

#### F2. Performance targets
| Metric | Current | Target | How |
|---|---|---|---|
| Model load time (GGUF mmap) | N/A | <100ms | mmap zero-copy |
| Model load time (safetensors) | ~2s | <500ms | PreAllocated + pinned memory |
| Sleep/wake round-trip (level 1) | N/A | <3s | non_blocking CPU offload |
| Token generation (LFM2.5 bf16, batch=1) | ~30 tok/s | ~60 tok/s | CUDA graphs + compile + MTP |
| Token generation (LFM2.5 bf16, MTP k=4) | N/A | ~80 tok/s | MTP speculative decoding |
| Token generation (LFM2.5 bf16, EAGLE-3) | N/A | ~90 tok/s | EAGLE-3 draft head |
| Swarm 4-agent run (2 topics) | ~60s | ~30s | Structured outputs eliminate retries |

#### F3. Comparison benchmarks
Run identical prompts on ForgeAI v2, LM Studio (llama.cpp backend), and raw llama.cpp server:
- Single-stream tok/s (LFM2.5 bf16)
- Model load time
- Multi-model switch time
- Agent task completion rate (structured vs unstructured)

---

## 3. Implementation Order (dependency-driven)

```
Week 1-2: Phase A (foundation)
  A1 → A2 → A3 → A4 → A5
  (Fixes → sleep/wake → loader → registry → server)
  GATE: server serves LFM2.5 via OpenAI-compatible API

Week 3: Phase B (speculative decoding)
  B1 → B2 → B3
  (EAGLE-3 → adaptive router → L1 integration)
  GATE: MTP+EAGLE-3 achieves >2x speedup at batch=1

Week 4: Phase D (agent swarm v2)
  D1 → D2 → D3 → D4 → D5
  (Cut LM Studio → structured outputs → tool calling → orchestrator → fixes)
  GATE: 4-agent swarm completes 2 topics with structured outputs

Week 5: Phase C + E (plugins + observability)
  C1 → C2 → E1 → E2 → E3
  GATE: plugin loads and runs; metrics dashboard shows VRAM usage

Week 6: Phase F (verification)
  F1 → F2 → F3
  GATE: all benchmarks pass, comparison shows ForgeAI > LM Studio
```

---

## 4. Key Architectural Decisions

1. **Python-first plugin SDK** (not TypeScript): our entire ecosystem is Python. C extension API for kernel plugins handles the perf-critical path.
2. **mmap GGUF as primary fast-load format**: llama.cpp's killer feature. We adopt it natively. 100ms loads.
3. **EAGLE-3 + MTP fusion**: we have MTP baked into LFM2.5. Adding EAGLE-3 draft head gives us two complementary speculative paths.
4. **VRAM budget model, not utilization fraction**: vLLM's `gpu_memory_utilization` is buggy for multi-instance. We use absolute byte budgets per engine.
5. **Structured outputs as first-class**: outlines/guidance integration at the engine level, not the agent level. Any model can be constrained.
6. **Sleep/wake with CUDA context preservation**: vLLM Sleep Mode's key insight. We keep tokenizer, config, CUDA context, and (level 1) CPU-backed weights.
