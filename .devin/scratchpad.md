# ForgeAI Gap Analysis â€” Engine, Trainer, Server Compatibility
## Consolidated findings (2026-09-05)

Sources: local codebase audit + online research (vLLM V1, SGLang, llama.cpp,
TGI, Axolotl, unsloth, LLaMA-Factory, OpenAI API spec 2026).

---

## BENCHMARK RESULTS â€” ForgeEngine Backend Bottlenecks (2026-09-05)

Hardware: RTX 5070 (12GB VRAM), 32GB RAM, Python 3.13, PyTorch (CUDA).
Model: ForgeLM V2 Light config (1.2B, 2048 d_model, 16 layers, conv+attn hybrid).
NOTE: Checkpoint file contained Qwen 2.5 0.5B weights (mismatched shapes);
model ran with random weights but correct architecture shapes, so throughput
numbers are valid for backend performance profiling.

### KV Cache Strategies (50 token generation, 3 runs)
| Strategy | tok/s | Latency | VRAM alloc | vs Paged |
|---|---|---|---|---|
| paged | 91.5 | 76ms | 4876MB | 1.00x (baseline) |
| cpu_offload | 91.8 | 76ms | 4876MB | 1.00x (offload not engaging) |
| snapkv | 85.2 | 82ms | 4868MB | 0.93x |
| s4r | 78.2 | 89ms | 4876MB | 0.85x |
| standard | 71.6 | 98ms | 4868MB | 0.78x |
| rotorquant | 62.3 | 112ms | 4868MB | 0.68x (32% slower) |
| hadamard_int4 | 53.4 | 131ms | 4868MB | 0.58x (42% slower) |

**Finding**: Paged KV is fastest. RotorQuant and Hadamard INT4 are 32-42% slower
due to per-step quantization overhead. Standard KV is 22% slower than paged
(suggests paged has better memory access patterns).

### Quantization (50 token generation, 3 runs)
| Mode | tok/s | Latency | VRAM alloc | vs None |
|---|---|---|---|---|
| none | 71.1 | 98ms | 4868MB | 1.00x |
| int8 | ERROR | - | 1770MB | dtype mismatch (BFloat16 vs Float) |
| int4 | 25.2 | 278ms | 2828MB | 0.35x (3.5x SLOWER) |
| w8a8 | 25.4 | 276ms | 2828MB | 0.36x (3.5x SLOWER) |
| nvfp4 | 25.3 | 277ms | 2828MB | 0.36x (3.5x SLOWER) |
| fp8 | 25.4 | 276ms | 2828MB | 0.36x (3.5x SLOWER) |

**CRITICAL**: ALL quantization modes are 3.5x SLOWER than unquantized!
Root cause: naive dequantize-then-matmul, no fused quantized GEMM kernels.
On RTX 5070 (Blackwell), FP8/NVFP4 should use `scaled_mm` or cutlass kernels.
int8 is completely broken (dtype mismatch).

### Decoding Strategies
| Mode | tok/s | Error |
|---|---|---|
| standard | 25.4 | (degraded â€” runs after quantization tests) |
| speculative | ERROR | "missing required argument 'draft_model'" |
| mtp_selfspec | ERROR | "got unexpected keyword argument 'min_p'" |
| eagle3 | ERROR | "got unexpected keyword argument 'min_p'" |

**CRITICAL**: ALL advanced decoding modes are broken.
- Speculative requires external draft_model (not auto-configured)
- MTP/EAGLE3 have API mismatch â€” `generate()` passes `min_p` but their
  `generate()` signatures don't accept it

### Acceleration
| Feature | tok/s | Error |
|---|---|---|
| prefix_cache | ERROR | "too many values to unpack (expected 2)" |
| chunked_prefill | 25.2 | (works but degraded by prior quantization) |
| cuda_graph | ERROR | "Cannot copy between CPU and CUDA tensors during CUDA graph capture" |
| compile | ERROR | CUDAGraph tree overwrite on conv state clone |
| optimal (auto) | ERROR | Same CUDAGraph tree overwrite |

**CRITICAL**: torch.compile + CUDA graphs crash on conv layers.
Root cause: `self._conv_state = Bx[:, -(k-1):, :].transpose(1,2).clone()`
in conv forward triggers CUDAGraph tree overwrite detection.
Fix: call `torch.compiler.cudagraph_mark_step_begin()` before each model
invocation, or clone outside compile region.

### Batch Generation
ERROR: "BatchedDecoding.generate_batch() got an unexpected keyword
argument 'max_new_tokens'" â€” API mismatch between engine and batch decoder.

### Bottleneck Profiler
All per-layer timings reported 0.0ms with 0 calls â€” profiler hooks not
attaching correctly to `ModularBlock` layers. Total time 178ms for 16 tokens
(89.6 tok/s) but 100% classified as "non_layer" time.

### State Degradation (STICKY QUANTIZATION)
After running quantization tests, subsequent standard decoding runs at 25 tok/s
instead of 71 tok/s. Quantization is not properly undone when re-activating
without quantize=. This means `activate(quantize=None)` does not restore
original bf16 weights â€” the quantized weights persist.

### VRAM Leak Between Engine Loads
Retest with fresh engine per quantization mode caused OOM on 3rd load:
w8a8 allocated 25.9GB (2x VRAM!), fp8 load failed. `del engine; gc.collect();
cuda.empty_cache()` is insufficient â€” quantization creates weight copies that
aren't tracked by the allocator.

### Checkpoint Mismatch
`ForgeLM_V2_Light.sft.safetensors` contains Qwen 2.5 0.5B weights
(151936 vocab, 896 d_model) but config `forgelm_v2_light` expects
(65536 vocab, 2048 d_model). Load fails silently â†’ AirLLM streaming fallback
â†’ random weights. This is a data integrity issue.

---

## BOTTLENECK SEVERITY RANKING

### P0 â€” Broken features (block production use)
1. **Quantization is 3.5x COUNTERPRODUCTIVE** â€” all modes (int4/w8a8/nvfp4/fp8)
   run at 25 tok/s vs 71 tok/s unquantized. Naive dequant, no fused kernels.
2. **int8 quantization BROKEN** â€” dtype mismatch (BFloat16 vs Float)
3. **torch.compile + CUDA graphs CRASH on conv layers** â€” CUDAGraph tree
   overwrite on `_conv_state.clone()`. Breaks `activate_optimal()`.
4. **ALL speculative decoding BROKEN** â€” speculative needs draft_model,
   MTP/EAGLE3 reject `min_p` kwarg
5. **Prefix cache BROKEN** â€” unpacking error
6. **Batch generation API mismatch** â€” wrong kwarg name
7. **Sticky quantization** â€” `activate(quantize=None)` doesn't restore weights
8. **VRAM leak between engine loads** â€” quantization copies not freed

### P1 â€” Performance bottlenecks
9. **RotorQuant KV 32% slower** than paged â€” quantization overhead
10. **Hadamard INT4 KV 42% slower** than paged
11. **Standard KV 22% slower** than paged â€” memory access patterns
12. **Bottleneck profiler not working** â€” 0 calls per layer, hooks not attaching

### P2 â€” Data issues
13. **Checkpoint mismatch** â€” ForgeLM_V2_Light.sft contains Qwen 2.5 0.5B weights
14. **Python 3.13 compat** â€” missing `Optional`/`nn` imports (fixed in this session)

Priority key:
- P0 = breaks standard clients / production use
- P1 = important ecosystem compatibility
- P2 = advanced / optional features

---

## A. Inference Engine â€” Missing vs. vLLM/SGLang/llama.cpp

### Scheduler / KV cache
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| Continuous batching (per-step admission/preemption) | ABSENT | P0 | `session_manager.BatchQueue` uses fixed 52ms window + static batch dispatch, not iteration-level |
| PagedAttention / paged KV with block tables | PARTIAL | P0 | Engine has "paged" KV modes but no vLLM-style block-table paged attention with <5% waste |
| Chunked prefill mixed with decode | PARTIAL | P1 | `chunked_prefill` exists in feature_registry; not unified-scheduler style |
| Automatic prefix caching (block/radix) | PARTIAL | P1 | Prefix caching + chunked prefix caching exist; not radix-tree cross-request |
| Disaggregated prefill/decode | ABSENT | P2 | No PD split, no NIXL/Mooncake/LMCache transfer |
| FP8 KV cache | PARTIAL | P2 | Some KV quant modes exist; not standard FP8 KV |
| KV offloading | PRESENT | - | `cpu_kv_offload.py` wired |

### Speculative decoding
| Feature | Status | Priority |
|---|---|---|
| EAGLE-3 / MTP / Medusa / self-spec | PRESENT | - |
| External draft model | ABSENT | P1 |
| N-gram / suffix speculation | ABSENT | P2 |
| DFlash / DSpark | PARTIAL (DSPark referenced) | P2 |

### Attention / kernels
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| FlashAttention-2/3/4 backend | UNCLEAR | P1 | `flash_attention`/`varlen_attention` exist; FA3/4 unverified |
| FlashInfer / MLA / Flex Attention | ABSENT | P2 | |
| Fused MoE kernels | UNCLEAR | P2 | MoE config exists; fused kernel unverified |
| CUDA Graphs | PRESENT | - | |
| torch.compile | PRESENT | - | |

### Quantization
| Format | Status | Priority |
|---|---|---|
| INT8/INT4/FP8/W8A8/NVFP4/BitNet | PRESENT | - |
| GPTQ / AWQ | ABSENT | P1 |
| GGUF quantized load | ABSENT (P0 for GGUF users) | P0 | `forge_loader.py` returns quantized GGUF as raw uint8, no dequant; `ForgeEngine.from_checkpoint()` never calls `ForgeLoader` |
| compressed-tensors / TorchAO | ABSENT | P2 |

### Output / API features
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| `logprobs` / `top_logprobs` | ABSENT | P0 | `generate()` returns text only |
| `n` (multiple completions) | ABSENT | P1 | Only `generate_batch` (different prompts) |
| `presence_penalty` | ABSENT | P1 | Only `repetition_penalty`/`frequency_penalty` |
| `seed` in /v1/chat/completions | ABSENT | P1 | Task API has it; main chat handler doesn't forward |
| `stop` not forwarded in chat handler | ABSENT | P0 | Declared in request model, not passed to `registry.generate()` |
| `top_k`/`repetition_penalty` not forwarded | ABSENT | P0 | Same â€” declared but dropped in streaming + non-streaming |
| `tool_choice` enforcement | ABSENT | P0 | In request model, not enforced |
| Streaming tool-call incremental deltas | PARTIAL | P1 | Tool calls parsed after accumulation, not streamed as deltas |
| `stream_options.include_usage` | ABSENT | P1 | |
| `response_format` (json_schema) | PARTIAL | P1 | Simplified char-level FSM, not full XGrammar CFG |
| Regex / EBNF / GBNF grammars | ABSENT | P1 | `xgrammar.py` is simplified first-char FSM |
| Reasoning parsers / `reasoning_effort` | ABSENT | P2 | |
| `system_fingerprint` | ABSENT | P2 | |

### Endpoints
| Endpoint | Status | Priority |
|---|---|---|
| /v1/chat/completions, /v1/completions | PRESENT | - |
| /v1/embeddings | ABSENT | P0 |
| /v1/responses (agentic) | ABSENT | P1 |
| /v1/audio/* | ABSENT | P2 |
| /v1/images/* | ABSENT | P2 |
| /v1/files, /v1/fine_tuning | ABSENT | P2 |
| /v1/batches | ABSENT | P2 |
| /v1/moderations | ABSENT | P2 |

### Multimodal
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| Vision input (image URL/tensor/parts) | ABSENT | P1 | V12 config has vision fields; no inference path |
| Audio input | ABSENT | P2 | |
| Embedding models / pooling | ABSENT | P1 | |
| Reranking endpoint | ABSENT | P2 | |

### Multi-LoRA serving
| Feature | Status | Priority |
|---|---|---|
| Per-request adapter selection | ABSENT | P1 |
| Batched multi-LoRA | ABSENT | P1 |

### Distributed inference
| Feature | Status | Priority |
|---|---|---|
| Tensor parallelism | ABSENT | P1 |
| Pipeline parallelism | ABSENT | P2 |
| Expert parallelism | ABSENT | P2 |

### Model loading / interchange
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| safetensors load/save | PRESENT | - | `checkpoint_io.py`, `model_loader.py` |
| PyTorch .pt | PRESENT | - | |
| GGUF usable load | ABSENT | P0 | Quantized types raw uint8 |
| HF Hub auto-config from config.json | ABSENT | P1 | `from_checkpoint()` needs preset name |
| HF â†’ Forge conversion | PARTIAL | P1 | Only Qwen3/Gemma3/Llama4 + generic Qwen/Llama remap |
| Export to GGUF/HF/ONNX/TFLite | ABSENT | P1 | Only internal NLRQ int8 export |
| ProgressiveLoader wired | ABSENT | P2 | Class exists, not used by engine |
| Tokenizer training | ABSENT | P2 | LFM2.5 tokenizer only |

---

## B. Trainer â€” Missing vs. Axolotl/unsloth/LLaMA-Factory/TRL

### Distributed training
| Feature | Status | Priority |
|---|---|---|
| DDP | ABSENT | P0 |
| FSDP / FSDP2 | ABSENT | P0 |
| DeepSpeed ZeRO | ABSENT | P0 |
| Accelerate / torchrun launcher | ABSENT | P0 |
| Multi-node | ABSENT | P1 |
| Tensor/expert/context parallelism | ABSENT | P2 |

### Training methods
| Method | Status | Priority |
|---|---|---|
| SFT / CPT / LoRA / QLoRA | PRESENT | - |
| DPO / ORPO / KTO | PRESENT | - |
| GRPO / RLVR | PRESENT | - |
| PPO (learned reward model) | ABSENT | P1 |
| SimPO / IPO / NCA / R-DPO / cDPO | ABSENT | P2 |
| DoRA | ABSENT | P2 |
| Reward modeling / PRM | ABSENT | P1 |

### Data
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| JSONL / Parquet | PARTIAL | P1 | Parquet module imported but **missing from repo** |
| HF `datasets` loader for SFT | ABSENT | P0 | Only DPO uses `load_dataset` |
| Packed .bin pretrain streams | PRESENT | - | `train_8b_all.py` |
| Dataset mixing / weighted domains | PARTIAL | P1 | CPT has reasoning-ratio; no general mixing |
| Padding-free / multipacking | ABSENT | P1 | |
| `forge.training.data.*` modules | MISSING | P0 | `efficient_pipeline`, `parquet_dataset`, `curriculum_augment` imported by `sft_train.py` but not in repo â†’ SFT may fail at import |
| `DataLoader` workers/pin_memory | ABSENT | P1 | Manual shuffling + AsyncPrefetcher (missing module) |

### Resume / early stopping / schedulers
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| `sft_train --resume` | ABSENT | P0 | Saves state, no resume flag |
| Universal `resume_from_checkpoint` | ABSENT | P1 | Per-runner only |
| Early stopping / patience | ABSENT | P1 | `best_val` tracked but never stops |
| LR scheduler choices | PARTIAL | P1 | `train_8b_all` has linear/wsd/cosine; SFT has internal cosine only |
| Warmup config | PARTIAL | P2 | |

### Loss functions
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| CE / chunked CE / entropy-weighted | PRESENT | - | |
| Focal / label_smoothing / lovasz / mixture | DECLARED BUT UNIMPLEMENTED | P1 | CLI flags exist, `compute_loss()` only uses `F.cross_entropy` |
| DPO/ORPO/KTO losses | PRESENT | - | In `dpo_align.py` |

### Evaluation
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| MMLU / HumanEval / GSM8K / BBH harness | ABSENT | P0 | Only `CheckpointTester` 50-question probe |
| lm-evaluation-harness integration | ABSENT | P1 | |
| Loss-only validation | PRESENT | - | |

### Experiment tracking
| Feature | Status | Priority |
|---|---|---|
| W&B | ABSENT | P1 |
| TensorBoard | ABSENT | P1 |
| MLflow / SwanLab / ClearML | ABSENT | P2 |
| Status JSON / heartbeat files | PRESENT | - |

### Optimizers
| Feature | Status | Priority |
|---|---|---|
| AdamW / 8bit / BAdam / Muon / GaLore / APOLLO / Forge | PRESENT | - |
| Adafactor / SOAP / Prodigy / Sophia | ABSENT | P2 |
| FP8 training wired to runners | ABSENT | P1 | `fp8_training.py` exists, not wired |

### Kernels
| Feature | Status | Priority |
|---|---|---|
| FlashAttention 2/3 training wrapper | UNCLEAR | P1 | `use_varlen` exists; FA3 unverified |
| Liger Kernel / Cut Cross Entropy / ScatterMoE | ABSENT | P2 |
| Gradient checkpointing | PRESENT | - | With optimal planner |

### Other
| Feature | Status | Priority | Evidence |
|---|---|---|---|
| `train_v8.py` real data | BROKEN | P1 | Uses synthetic random token IDs, not real data |
| MoE CLI controls (z-loss, load balance, expert drop) | ABSENT | P2 | Config exists, no CLI |
| Multimodal training (VLM/audio) | ABSENT | P2 | |
| YAML declarative config | ABSENT | P2 | CLI args only |

---

## C. Production / Ops
| Feature | Status | Priority |
|---|---|---|
| Prometheus metrics | ABSENT | P1 |
| OpenTelemetry tracing | ABSENT | P2 |
| Watermarking | ABSENT | P2 |
| Router mode (multi-instance) | ABSENT | P2 |

---

## Top P0 (breaks standard use)
1. `stop`/`top_k`/`repetition_penalty` declared but not forwarded in chat handler
2. `tool_choice` not enforced
3. No `logprobs`
4. No `/v1/embeddings`
5. GGUF quantized load broken (raw uint8, not dequantized)
6. No continuous batching (fixed-window only)
7. `forge.training.data.*` modules missing â†’ SFT may fail at import
8. No HF `datasets` loader for SFT
9. No distributed training (DDP/FSDP/DeepSpeed)
10. No `sft_train --resume`
11. No standard eval harness (MMLU/HumanEval/GSM8K)
12. `train_v8.py` runs on synthetic data

## Top P1 (ecosystem compatibility)
- GPTQ/AWQ, external draft model, multi-LoRA serving, tensor parallelism
- HF auto-config from config.json, export to GGUF/HF
- Regex/EBNF grammars, `response_format` strict, `n`, `presence_penalty`, `seed`
- Streaming tool deltas, `stream_options.include_usage`
- PPO/reward modeling, padding-free packing, early stopping, LR scheduler choices
- W&B/TensorBoard, FP8 training wired, FA2/3 training wrapper
- Prometheus metrics, vision input

## Sub-BitNet Quantization R&D (2026-09-06)

### Goal
Training-free ~1-bit quantization key for LLMs, tested on Qwen 2.5 0.5B (CUDA, RTX 5070 12GB).

### Best Configuration Found
- **Method**: BiLLM (salient + concentrated/sparse split) + NF4-quantized SVD residual
- **Params**: salient_frac=0.10, salient_order=2, svd_rank=48, svd_block_size=32
- **Result**: PPL 1096 at 1.55 bpw (FP16 baseline PPL 22 on same corpus)
- **VRAM**: 3.21 GB for quantized model (vs ~1 GB FP16)

### Key Findings
1. **Pure ternary (1.58 bpw)**: PPL 3M+ — completely unusable without residuals.
2. **BiLLM alone (no SVD)**: PPL 5568-7616 at ~1.10 bpw — too much distortion.
3. **Mean-centered residuals**: Dramatically worse than absmean (zero-mean) binarization. PPL in hundreds of thousands. Mean-centering is NOT a good fit for this model.
4. **Hadamard rotation**: Helps attention layers (896?1024, 12.5% padding) but HURTS down_proj (4864?8192, 40.6% padding). Auto-disable Hadamard when padding > 15%.
5. **GPTQ compensation**: Inconsistent — improved single-layer SQNR by +2dB but worsened average output SQNR by -0.69dB across layers. Not enabled by default.
6. **SVD residual is essential**: The gap between BiLLM-only (PPL ~5000+) and BiLLM+SVD (PPL ~1096) is enormous. SVD captures the structured low-rank component of the quantization error.
7. **Block-wise NF4 (bs=32)** for SVD factors is the sweet spot. Per-tensor NF4 is too coarse (PPL 18304). Per-row NF4 adds too much scale overhead. bs=16 wastes bits on scales, bs=128+ loses precision.

### Storage Accounting (1.55 bpw)
Per layer (out_f, in_f):
- Salient binary: salient_order × out_f × n_salient bits
- Non-salient binary: 1 × out_f × n_nonsalient bits
- Split mask: 1 × out_f × n_nonsalient bits (stored as bool)
- SVD U: 4 × rank × out_f + 16 × ceil(rank×out_f / bs) bits (NF4 + scales)
- SVD V: 4 × rank × h_size + 16 × ceil(rank×h_size / bs) bits (NF4 + scales)
- Total ÷ (out_f × in_f) = effective bpw

### Failed Approaches (documented dead ends)
- Ternary alone: PPL 3M+
- BiLLM mean-centered: PPL 100K-1M
- Per-tensor NF4 SVD: PPL 18K (too coarse)
- 8-bit SVD: always exceeds 1.58 bpw budget
- IRB 1-round + SVD in Hadamard space: PPL 1.5M (Hadamard padding inflates SVD cost)
- IRB 1-round + SVD in original space: PPL 1.5M (residual not well-aligned)

### Remaining Limitations
- PPL 1096 is still far from FP16 baseline (22). This is NOT "~1-bit working" in a usable sense.
- The 1.55 bpw target is met but quality is insufficient for practical use.
- Longer evaluation corpus needed for stable PPL measurement.
- Sub-1.58 bpw with usable quality likely requires either: (a) calibration data (GPTQ/AWQ), (b) higher rank SVD (exceeds budget), or (c) training-based refinement (QAT).
