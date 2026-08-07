# Boot-Time Audit — Load, Warmup, First-Token Pipeline

> Covers the boot-time (cold-start → first-token) pipeline, which the key audits did NOT cover. Maps each boot-time stage to existing ForgeAI assets (DSpark, Gigatoken, torch.compile) and to published techniques we could adopt. Compiled 2026-08-06.
>
> **Bottom line:** ForgeAI already has 3 boot-time assets (Gigatoken tokenizer, DSpark spec-decode, torch.compile). The audit found 8 more published boot-time techniques we could adopt, and 2 novel boot-time keys. The boot pipeline is: **Tokenize → Load weights → Compile → Profile KV → Capture CUDA graphs → Warm prefix cache → Serve first token.**

---

## The Boot-Time Pipeline (7 stages)

| # | Stage | Time (typical 8B model) | ForgeAI Asset | Published Alternative |
|---|---|---|---|---|
| 1 | **Tokenization** (text → token IDs) | 10-500ms (CPU-bound, long prompts) | Gigatoken (30-40x faster) | TokTier, tuetoken, IREE, GPUTOK, Basetenkenizer |
| 2 | **Weight loading** (disk → VRAM) | 5-150s (I/O bound) | safetensors mmap | fastsafetensors (4.8-7.5x), Foundry, prefetch, pread backend |
| 3 | **Compilation** (torch.compile/Inductor) | 10-40s (JIT) | torch.compile (patched triton) | Persistent compile cache, eager compilation, -O0/-O3 tiers |
| 4 | **KV cache profiling** (measure peak memory) | 1-5s | Manual VRAM budget | vLLM growable KV cache, --kv-cache-memory-bytes |
| 5 | **CUDA graph capture** (eliminate kernel launch overhead) | 10-180s (shape buckets) | Not implemented | Foundry (template-based, 99% reduction), hybrid JIT-CUDA graph |
| 6 | **Prefix cache warming** (precompute system prompt KV) | 1-10s per prompt | Knowledge Pack (related) | RadixAttention, prompt-cache-warmer, off-peak pre-warming |
| 7 | **First token** (prefill + first decode) | 50-500ms | DSpark (60-85% faster) | EAGLE-3 (6.5x), Medusa, Lookahead, hybrid JIT-CUDA graph |

---

## Stage-by-Stage Audit

### Stage 1: Tokenization

**ForgeAI:** Gigatoken (compat + native modes, 30-40x faster than HF, exact parity)

**Published alternatives (all faster than Gigatoken in some regime):**

| Tool | Speed | Notes | Worth adopting? |
|---|---|---|---|
| **TokTier** (arxiv 2607.29678) | 437x faster than HF, 2.1x faster than Gigatoken (fully prewarmed) | Exact stateful CPU+GPU tokenization, 17 tokenizer families, zero divergence. **Beats Gigatoken.** | YES — drop-in replacement |
| **tuetoken** (github) | 30x faster than HF, faster than tiktoken by 10x on long inputs | Rust core, drop-in AutoTokenizer replacement, O(n) even on adversarial inputs | YES — simpler than TokTier |
| **IREE tokenizer** (github) | 3-12x faster than tiktoken, 10-20x faster than HF | Pure C, zero allocations, streaming encode/decode, 317KiB | YES — for embedded/edge |
| **GPUTOK** (arxiv 2603.02597) | 7.6x faster than HF, 1.7x faster than tiktoken | GPU BPE, but 70-80% time is memory allocation | Maybe — needs memory pooling |
| **Basetenkenizer** (Baseten blog) | 18x faster than tiktoken (million-token) | Rust, exact token ID parity, for Kimi K3 | Maybe — Kimi-specific |

**Recommendation:** Replace Gigatoken with **TokTier** (2.1x faster, exact parity, handles stateful tokenization for agentic workloads). Keep Gigatoken as fallback.

### Stage 2: Weight Loading

**ForgeAI:** safetensors mmap (standard)

**Published alternatives:**

| Tool | Speedup | Notes | Worth adopting? |
|---|---|---|---|
| **fastsafetensors** (github) | 4.8-7.5x faster, 26.4 GB/s NVMe | Used by vLLM and SGLang (`--load-format fastsafetensors`), supports Windows | YES — drop-in |
| **Prefetch strategy** (vLLM PR #36012) | 1500s → 435s (3.5x) | Background thread preloads to OS page cache, overlaps I/O with parsing | YES — simple |
| **pread backend** (safetensors PR #760) | Fixes OOM on unified memory | Alternative to mmap, reads into temp buffers | For DGX Spark/Apple Silicon |
| **Foundry** (arxiv 2604.06664) | 99% cold-start reduction | Template-based CUDA graph context materialization (also covers Stage 5) | YES — for autoscaling |

**Recommendation:** Adopt **fastsafetensors** (4.8-7.5x faster, Windows support, used by vLLM/SGLang). Add **prefetch** for network storage.

### Stage 3: Compilation (torch.compile)

**ForgeAI:** torch.compile with patched triton-windows (10K → 13.3K tok/s, 7.5GB VRAM)

**Published optimizations:**

| Technique | Effect | Notes | Worth adopting? |
|---|---|---|---|
| **Persistent compile cache** (vLLM) | Warm start skips compilation entirely | Cache FX graphs + Triton kernels to disk, reuse across restarts | YES — already in vLLM, we should cache |
| **Persistent autotune cache** (PyTorch PR #180529) | Skips GPU autotuning on restart | `TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR` | YES — env var, free |
| **Eager compilation** (vLLM PR #35472) | All ranges compiled upfront | No lazy on-demand compilation, predictable startup | YES — cleaner lifecycle |
| **-O0/-O3 tiers** (vLLM planned) | Trade startup for performance | -O0 = no optimization (fast boot), -O3 = max optimization | When available |

**Recommendation:** Enable **persistent compile cache** + **persistent autotune cache**. These are env vars / cache dirs — zero code change, massive warm-start improvement.

### Stage 4: KV Cache Profiling

**ForgeAI:** Manual VRAM budget (`--vram-budget`, `--fp16` flags in self_play_expert_training.py)

**Published alternatives:**

| Technique | Effect | Notes | Worth adopting? |
|---|---|---|---|
| **Growable KV cache** (vLLM PR #50779) | Reserve VA, commit physical pages incrementally | Profile → warmup → measure actual free → extend. No OOM from guessing. | YES — eliminates VRAM trial-and-error |
| **--kv-cache-memory-bytes** (vLLM PR #21489) | Pin KV cache size directly | Skip profiling on subsequent runs, use suggested value from first run | YES — deterministic |
| **fp8 KV cache** (vLLM) | ~2x more KV tokens | `--kv-cache-dtype fp8` | We have KV4Bit already |

**Recommendation:** Implement **growable KV cache** pattern (reserve virtual, commit physical after warmup). This eliminates the OOM crashes the user has been hitting.

### Stage 5: CUDA Graph Capture

**ForgeAI:** Not implemented (using torch.compile only)

**Published techniques:**

| Technique | Speedup | Notes | Worth adopting? |
|---|---|---|---|
| **CUDA graphs for decode** (solana.garden guide) | 22ms → 14ms per token, +38% throughput | Capture shape buckets, replay with single launch. Prefill stays eager. | YES — biggest single win for decode |
| **Foundry** (arxiv 2604.06664) | 10 min → 3.9s cold start (99% reduction) | Template-based: persist graph + context offline, reconstruct online. Multi-GPU from single capture. | YES — for autoscaling |
| **Hybrid JIT-CUDA graph** (arxiv 2604.23467) | 66% TTFT reduction | Static subgraphs → CUDA graph, dynamic → JIT. Asynchronous capture. | YES — best for short sequences |

**Recommendation:** Implement **CUDA graph capture for decode buckets** (biggest win). For autoscaling, adopt **Foundry** (template persistence).

### Stage 6: Prefix Cache Warming

**ForgeAI:** Knowledge Pack (precompute KV for knowledge domains, inject at inference)

**Published techniques:**

| Technique | Effect | Notes | Worth adopting? |
|---|---|---|---|
| **Off-peak pre-warming** (gingerlabs) | O(n²) GPU → O(n) storage I/O | Precompute KV for static system prompts at 3AM, inject at peak | YES — we have Knowledge Pack already |
| **prompt-cache-warmer** (crate) | Synthetic warmup call at deploy | Fire 1-token completion with cache_control, verify cache hit | For Anthropic API (not self-hosted) |
| **RadixAttention** (SGLang) | Automatic prefix reuse across requests | Radix tree over KV blocks, handles branching overlap | YES — for multi-turn/agent workloads |
| **Session-start preload** (crewAI RFC) | Warm all agent system prompts at kickoff | Fire probe calls per agent before first task | YES — for multi-agent |

**Recommendation:** We already have Knowledge Pack. Add **RadixAttention-style prefix sharing** for multi-request workloads. Use Knowledge Pack for *portable* prefix warming (cross-session).

### Stage 7: First Token (Speculative Decoding)

**ForgeAI:** DSpark (semi-autoregressive + confidence-scheduled verification, 60-85% speedup)

**Published alternatives:**

| Method | Speedup | Notes | Worth adopting? |
|---|---|---|---|
| **EAGLE-3** (NeurIPS 2025) | 6.5x | Multi-layer feature fusion, training-time test. EAGLE-3.1 adds FC norm + post-norm feedback. | YES — complementary to DSpark |
| **Medusa** | 2-3x | Multiple decoding heads on target, no separate draft model | If we have MTP heads already |
| **Lookahead decoding** | 1.2-1.5x | Target model drafts via n-gram lookahead, zero new infrastructure | YES — simplest, no training |
| **LayerSkip** (ACL 2024) | 2.16x | Early exit + self-speculative decoding, layer dropout during training | Needs training-time support |
| **DSpark** (arxiv 2607.05147) | 60-85% | **Already in ForgeAI.** Semi-autoregressive + confidence-scheduled verification. | Have it |

**Recommendation:** DSpark is already strong. Add **Lookahead decoding** (zero infrastructure, 1.2-1.5x on top of DSpark). Consider **EAGLE-3** for maximum speedup (needs draft model training).

---

## Novel Boot-Time Keys (2)

### `boot_pipeline_key.py` — Boot Pipeline Orchestrator (TRIVIAL)
Orchestrates the 7-stage boot pipeline with optimal ordering and parallelization:
```
forward(data):
  # data = {"model_config": cfg, "system_prompts": [str], "vram_budget": GB}
  # Stage 1: Tokenize system prompts (TokTier, parallel with Stage 2)
  # Stage 2: Load weights (fastsafetensors + prefetch)
  # Stage 3: Compile (torch.compile with persistent cache)
  # Stage 4: Profile KV (growable: reserve VA, commit 1 block)
  # Stage 5: Capture CUDA graphs (decode buckets, Foundry templates if available)
  # Stage 6: Warm prefix cache (Knowledge Pack from system prompts)
  # Stage 7: Ready — first token uses DSpark
  # Parallelize: Stage 1 || Stage 2, then 3→4→5, then 6
  return {"ready": True, "boot_time_ms": ...}
```
**Class:** TRIVIAL (orchestration, no weights). **Novelty:** The *unified orchestration* of all 7 stages with parallelization is not found as a single system. vLLM/SGLang do parts but not the full pipeline with Knowledge Pack integration.

### `graph_template_key.py` — CUDA Graph Template Pack (TRIVIAL)
Persists CUDA graph + execution context offline, reconstructs online:
```
forward(data):
  # data = {"model": M, "batch_shapes": [shapes]}
  # Offline: capture graphs for each shape, serialize topology + kernel binaries + memory layout
  # Store as template pack
  return {"graph_templates": {shape: (topology, kernels, mem_layout)}}

reverse(weights):
  # Online: reconstruct executable graphs from templates
  # Patch rank-dependent communication state for multi-GPU
  return {"cuda_graphs": {shape: executable_graph}}
```
**Class:** TRIVIAL (runtime, no model weights). **Novelty:** Foundry does this but as a system, not as a portable pack. The *pack* framing (capture on one machine, deploy on another) is a ForgeAI twist.

---

## Updated Boot-Time Asset Map for ForgeAI

| Stage | Current | Recommended Upgrade | Effort |
|---|---|---|---|
| 1 Tokenize | Gigatoken | **TokTier** (2.1x faster) | 1 day (drop-in) |
| 2 Load | safetensors mmap | **fastsafetensors** (4.8-7.5x) + prefetch | 1 day (drop-in) |
| 3 Compile | torch.compile (patched) | + **persistent cache** + **autotune cache** | 1 hour (env vars) |
| 4 KV profile | Manual VRAM budget | **Growable KV cache** | 2 days |
| 5 CUDA graphs | Not implemented | **CUDA graph capture** + **Foundry templates** | 3-5 days |
| 6 Prefix warm | Knowledge Pack | + **RadixAttention** for multi-request | 2 days |
| 7 First token | DSpark | + **Lookahead decoding** (zero infra) | 1 day |

**Total estimated boot-time reduction:** From ~317s (vLLM 8B baseline) to ~30-50s with all optimizations. ForgeAI's current boot time is unknown but likely faster (smaller model, no CUDA graphs yet).

---

## Summary

| Category | Count |
|---|---|
| Boot-time stages | 7 |
| ForgeAI existing assets | 3 (Gigatoken, DSpark, torch.compile) |
| Published techniques to adopt | 8 |
| Novel boot-time keys | 2 (boot_pipeline, graph_template) |
| NOT A KEY (system-level, not weight transform) | 0 (all boot-time ops are runtime) |

**Key insight:** Boot-time is a *systems* problem, not a *weights* problem. None of the 7 stages involve weight transforms — they're all I/O, compilation, memory management, and caching. This is why the key framework (FULL/BI/PARTIAL/TRIVIAL) maps them all as TRIVIAL (runtime orchestration). The novel contributions are in *orchestration* (boot_pipeline) and *portability* (graph_template pack).

**The "pack" pattern extends to boot-time:** Just as 6 of 17 novel weight-keys are pack keys, the boot-time novel keys also use the pack pattern (graph_template = portable CUDA graph pack). This reinforces "extract → store → inject without training" as ForgeAI's signature approach.

---

*Compiled 2026-08-06. Companion to `KEY_NOVELTY_AUDIT.md` and `KEY_NOVELTY_AUDIT_PART2.md`. Boot-time is systems-level, not weight-level — all keys are TRIVIAL (runtime).*
