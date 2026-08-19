# ForgeAI Project Overview

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    ForgeEngine (Inference)                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ KV Cache │  │ Decoding  │  │ Quantize │  │ Innovations│  │
│  │ (7 strats)│  │ (5 strats)│  │ (INT4/8/ │  │ (MRL/V0/  │  │
│  │           │  │           │  │  FP8)    │  │  QuaRot)  │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ConfigurableResearchLLM (LFM2.5-1.2B)              │   │
│  │  10 conv + 6 GQA | 1.17B params | 2.34GB bf16      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↕                    ↕                    ↕
┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐
│  Self-Play Loop │  │  SFT Training│  │   Evaluation      │
│                 │  │              │  │                   │
│ Task Curriculum │  │ JSONL data   │  │ Benchmark compare │
│ → Tool calls    │  │ → Render     │  │ (quality/code/    │
│ → Execute tools │  │ → Tokenize   │  │  reasoning/tool)  │
│ → Reward        │  │ → CE loss    │  │                   │
│ → Save to DB    │  │ → Save .sft  │  │ Promote/Demote    │
│ → Export traj   │  │              │  │                   │
│ → Finetune/GRPO │  │              │  │                   │
└─────────────────┘  └──────────────┘  └───────────────────┘
```

---

## 1. ForgeLM Model

**File**: `research/model_loader.py`, `research/config.py`

| Property | Value |
|----------|-------|
| Base | Liquid AI LFM2.5-1.2B-Instruct (100% lossless port) |
| Params | 1.17B |
| Size | 2.34 GB (bf16) |
| Layers | 16 (10 double-gated conv + 6 GQA attention) |
| d_model | 2048 |
| Heads | 32 query, 8 KV (GQA 4x) |
| head_dim | 64 |
| FFN | SwiGLU (intermediate=8192) |
| Norm | RMSNorm + QK-layernorm |
| Position | RoPE theta=1M, 128K context (32K VRAM budget) |
| Vocab | 65536, tied embeddings |
| Configs | `lfm25_1.2b` (full), `lfm25_tiny` (4-layer test) |

**Key classes**:
- `ConfigurableResearchLLM` — full model with chunked CE, MTP, MoE aux loss
- `PreAllocatedKVCache` — O(1) append, INT8 quant support
- `ModelLoader` — architecture caching, memory-mapped loading, HF key remapping, hybrid offload

**Checkpoint**: `research/checkpoints/ForgeLM_V2_LFM25-1.2B.sft10.safetensors` (current production)
**Tokenizer**: `research/checkpoints/lfm25_tokenizer/` (gigatoken HFCompat wrapper)

---

## 2. ForgeEngine

**File**: `research/inference/forge_engine.py`

**Purpose**: Unified inference engine — orchestrates all runtime strategies.

**API**:
```python
engine = ForgeEngine.from_checkpoint("checkpoint.safetensors")
engine.activate(kv_cache="paged", decoding="standard", quantize="int4", warmup=True)
output = engine.generate("Hello", max_new_tokens=100, temperature=0.7)
```

**Features**:
- Auto-detects VRAM capacity → enables AirLLM streaming if model doesn't fit
- Auto-detects KeyStack features from checkpoint metadata
- CUDA graph capture for decode acceleration
- Warmup pass to initialize CUDA kernels
- Prefix caching for repeated prompts
- Sleep/wake for model switching (Level 1: CPU offload, Level 2: discard weights)

**Activate options**: kv_cache, decoding, quantize, acceleration, mrl_keep_ratio, kv_bits, use_v0_warm, use_progressive_kv, use_compile, use_prefix_cache, use_spec_attn, kv_cache_tokens, warmup

---

## 3. KV Cache Backends

**File**: `research/inference/kv_backend.py`

| Strategy | Description | Status |
|----------|-------------|--------|
| standard | Pre-allocated tensor, O(1) append | ✅ Working |
| paged | vLLM-style paged memory + prefix caching | ✅ Working |
| rotorquant | Givens rotation + Lloyd-Max quant (3-4 bit) | ✅ Working |
| hadamard_int4 | Block-diagonal Hadamard + INT4 | ✅ Working |
| compressed | H2O heavy-hitter eviction + KV quant | ✅ Working |
| streaming | StreamingLLM attention sinks + sliding window | ✅ Working |
| snapkv | Observation-window eviction | ✅ Working |

---

## 4. Decoding Strategies

**File**: `research/inference/decoding.py`, `research/decoding/`

| Strategy | Description | Speedup | Status |
|----------|-------------|---------|--------|
| standard | Autoregressive with KV cache | 1x | ✅ Working |
| speculative | Draft model + verify | 2-3x | ✅ Working (needs draft model) |
| medusa | Parallel prediction heads | 2.2-3.6x | ✅ Working |
| dspark | Semi-AR + confidence scheduling | ~3x | ✅ Working (experimental) |
| mtp_selfspec | MTP heads self-speculative | 2-4x | ✅ Working |
| eagle | Feature extrapolation | 5.6x | ❌ Stub/missing |

**Key files**:
- `research/decoding/medusa.py` (367 lines) — MedusaHeads, MedusaTrainer, medusa_generate
- `research/decoding/dspark.py` (460+ lines) — DSparkHead, DSparkTrainer
- `research/decoding/mtp.py` (242 lines) — MTPHead, MTPTrainer

---

## 5. Quantization

**Files**: `research/inference/int4_quant.py`, `research/quantization/`

| Method | Type | Compression | Status |
|--------|------|-------------|--------|
| INT4 weight-only | Per-group symmetric | ~3.8x | ✅ Working |
| INT8 weight-only | Per-channel | ~2x | ✅ Working |
| FP8 (Blackwell) | torch._scaled_mm native | ~2x | ✅ Working |
| RotorQuant KV | Givens + Lloyd-Max | 3-4 bit | ✅ Working |
| Paged KV | vLLM-style | N/A | ✅ Working |
| KV Compress | H2O + KVQuant | Variable | ✅ Working |
| BitNet | 1.58-bit | ~8x | ❌ Not implemented |
| SpinQuant | Hadamard rotation | N/A | ⚠️ Key only |
| WANDA | Pruning | Variable | ❌ Not implemented |

---

## 6. Innovations (Runtime)

**File**: `research/inference/innovations.py`

| Innovation | Description | Status |
|------------|-------------|--------|
| MRL Adaptive Context | Truncate to d_keep dimensions, reduce compute O(d²)→O(d'²) | ✅ Working |
| QuaRot KV | Hadamard rotation for better KV quantization | ✅ Working |
| V0 Warm Start | ValueResidual V_0 pre-populates KV cache | ✅ Working |
| Progressive KV | Anchor (MSB) + residual (LSB) split streams | ✅ Working |

---

## 7. Self-Play System

### 7a. Infinite Tool Loop

**File**: `research/self_play/discovery/infinite_tool_loop.py`

**Flow**: Self-Play → Export → Finetune/GRPO → Evaluate → Promote/Demote → Repeat

**Config** (`LoopConfig`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| tasks_per_epoch | 50 | Tasks per self-play session |
| max_turns | 5 | Max tool-call turns per task |
| max_gen_tokens | 256 | Max tokens per generation |
| temperature | 0.7 | Sampling temperature |
| min_reward | 0.4 | Min reward to save trajectory |
| ft_max_steps | 200 | Finetune steps |
| ft_lr | 1e-5 | Finetune learning rate |
| ft_batch_size | 2 | Finetune batch size |
| ft_seq_len | 1536 | Finetune sequence length |
| use_grpo | False | Use GRPO instead of SFT |
| grpo_group_size | 4 | GRPO group size |
| max_epochs | 10 | Max loop iterations |
| eval_threshold | 0.7 | Min quality to promote |

### 7b. Tool Registry

**File**: `research/self_play/discovery/discovery_tools.py`

| Tool | Purpose |
|------|---------|
| think | Record train-of-thought |
| sudo_think | Meta-reason about process |
| run_script | Sandboxed Python execution (8s timeout, no network) |
| web_search | DuckDuckGo HTML search (no API key) |
| save_research | Persist web research finding |
| propose_theory | Log hypothesis |
| update_theory | Update theory status/evidence |
| record_discovery | Record confirmed finding |
| query_db | Read-only SQL against memory DB |
| migrate_schema | Audited DDL (CREATE/ALTER/INDEX only) |
| summarize_context | Model summarizes its own context |
| finish_session | End discovery session |

### 7c. Qwen Adapter (Tool Call Format)

**File**: `research/self_play/discovery/qwen_adapter.py`

- Renders tool calls with literal text markers (built from hex to avoid IDE issues)
- Parses with 3-phase fallback: marker-wrapped JSON → bare JSON → LFM2.5 Pythonic format
- Uses `json_repair` for malformed JSON
- **xGrammar integration**: Constrained decoding — free text until `{` emitted, then grammar enforces valid JSON with tool name enum
- Loads fresh HF tokenizer for xgrammar (gigatoken wrapper not supported)

### 7d. Context Manager

**File**: `research/self_play/discovery/context_manager.py`

- Token budget: 32K max, 4K reserved for generation
- Compresses at 75% capacity
- Keeps system + tool defs + last 6 turns intact
- Never splits tool_call + tool_result pairs
- Heuristic or model-generated summaries

### 7e. Discovery DB

**File**: `research/self_play/discovery/discovery_db.py`

SQLite database with 11 tables: sessions, thoughts, scripts, research, theories, discoveries, events, schema_migrations, epochs, distill_runs, tool_trajectories.

Key: `tool_trajectories` stores full multi-turn conversations with reward scores for SFT/RL training.

### 7f. Epoch Manager

**File**: `research/self_play/discovery/epoch_manager.py`

- DB quality score: coverage (40%) + resolution (20%) + script quality (20%) + trajectory quality (20%)
- Distill every 12 epochs, finetune otherwise
- Compare candidate vs best on quality/skill/compute → promote or archive

---

## 8. Training / Finetune

### 8a. SFT Training

**File**: `research/training/sft_train.py`

**Data formats**:
- `tool_use_fc`: JSONL `{"messages": [...]}` — multi-turn function calling
- `short_cot/code/tool_use`: JSONL `{"prompt": ..., "response": ...}` — single-turn

**Training features**:
- Completion-only loss (prompt masked with -100)
- Per-turn splitting for multi-turn (each assistant turn = separate example)
- Chunked CE (memory-efficient, avoids materializing full logits)
- LoRA support
- Gradient checkpointing
- EMA (exponential moving average)
- torch.compile support
- Cosine LR schedule with warmup
- Gradient clipping (1.0)

**CLI**: `python -m research.training.sft_train --data ... --checkpoint ... --save ... --max-steps 30 --lr 1e-5`

### 8b. Chunked CE

**File**: `research/training/chunked_ce.py`

Pure-PyTorch fused linear+CE — avoids materializing full [B*T, V] logits. 2-3x slower than Liger Triton kernel but works on all GPUs.

### 8c. GRPO Trainer

**File**: `research/self_play/grpo_trainer.py`

**Fully implemented** with advanced features:
- MC-GRPO: Median baseline for small groups
- Turn-level advantages for multi-turn tool use
- KL penalty: KL(π_current || π_ref)
- PPO-style clipping
- ACR (Advantage Collapse Rate) monitoring
- Extensions: SC-GRPO, OM-GRPO, GVPO
- Tool-use rewards: continuous 0..1

### 8d. Training Data

**Directory**: `research/data/finetune/`

| Dataset | Count | Content |
|---------|-------|---------|
| tool_use_fc_300.jsonl | 300 | Function calling (14 tools) |
| short_cot_300.jsonl | 300 | Chain-of-thought reasoning |
| code_300.jsonl | 300 | Python programming |
| reasoning_500.jsonl | 500 | General reasoning |
| concise_86.jsonl | 86 | Concise answers |
| codebase_74.jsonl | 74 | Codebase Q&A |
| self_correction.jsonl | — | Self-correction data |
| self_play_epoch1.jsonl | — | Self-play generated |

**Expert data**: `research/data/expert_training/hf_datasets/` — algorithms, coding, creativity, general, math, python, theory, token_efficiency, tool_use

---

## 9. Evaluation

**Files**: `research/evaluation/`, `.devin/test_model_compare.py`

| Benchmark | Description | Status |
|-----------|-------------|--------|
| test_model_compare.py | Quality/code/reasoning/tool-use comparison (25 tests) | ✅ Working |
| reasoning_benchmarks.py | ARC-AGI-2, NeoCoder, FineReason, ThinkBench | ✅ Working |
| livecodebench_eval.py | LiveCodeBench (ICLR 2025, contamination-free) | ✅ Working |
| goal_scorer.py | Multi-dimensional: minimalism, efficiency, diversity, consistency, confidence | ✅ Working |
| prompt_tests*.py | Prompt-based evaluation | ✅ Working |

**Benchmark categories** (test_model_compare.py):
- Tool use (3 tests): novel tool, parallel tools, nested args
- Reasoning (5 tests): capitals, math, Fibonacci, syllogism, primes
- Code (5 tests): BST, LRU cache, binary search, etc.
- Conciseness (5 tests): word limits, lowercase constraints
- Instruction following (5 tests): format, style constraints
- Self-correction (2 tests): false premise, sycophancy

---

## 10. Model Merging & Injection

**Files**: `research/merge_models.py`, `research/inject_and_merge.py`

**Merge methods**: SLERP, TIES, DARE, SVD, Task Arithmetic, Linear (model soup)

**Injection types**: facts, test_gated, context_patch, selfplay_patch, spectral

**CLI**:
```bash
python -m research.merge_models --method slerp --model-a a.safetensors --model-b b.safetensors --out merged.safetensors
python -m research.inject_and_merge --target ckpt.safetensors --inject-type facts --merge-method task_arith
```

---

## 11. Runtime

**Files**: `research/runtime/`

| Component | Description | Status |
|-----------|-------------|--------|
| cuda_graph.py | CudaGraphRunner + CudaGraphGenerator (capture/replay) | ✅ Working |
| vram_manager.py | Boot profiling, dynamic KV sizing, compile cache | ✅ Working |
| forward_cache.py | LRU cache for repeated forward passes (20-40% savings) | ✅ Working |
| self_model.py | Self-model implementation | ✅ Working |
| task_logger.py | Task logging | ✅ Working |

---

## 12. Current Production State

| Component | Checkpoint | Notes |
|-----------|------------|-------|
| Best model | ForgeLM_V2_LFM25-1.2B.sft10.safetensors | SFT7 base + reasoning/code/self-correction data |
| Self-play | Working end-to-end | xgrammar constrained decoding, 7/8 tool use |
| Tool format | Literal text markers | Model outputs messy but parser is lenient |
| Training | SFT with completion-only loss | No anti-regression yet, no token reweighting |
| Evaluation | 25-test benchmark | Quality/code/reasoning/tool/conciseness/instruction |

**Known issues**:
- Model outputs `<tool_call:` with colon instead of proper marker format (parser compensates)
- EAGLE decoding not implemented (stub)
- BitNet/WANDA quantization not implemented
- No anti-regression techniques in SFT (no L2-SP, no token reweighting)
- No FP4 quantization for Blackwell (FP8 exists)
- fastsafetensors falls back to standard (missing DirectStorage DLLs on Windows)
