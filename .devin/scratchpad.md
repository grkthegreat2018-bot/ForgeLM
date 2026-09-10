# ForgeAI Gap Analysis — Engine, Trainer, Server Compatibility
## Consolidated findings (2026-09-05)

Sources: local codebase audit + online research (vLLM V1, SGLang, llama.cpp,
TGI, Axolotl, unsloth, LLaMA-Factory, OpenAI API spec 2026).

---

## BENCHMARK RESULTS — ForgeEngine Backend Bottlenecks (2026-09-05)

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

**UPDATE 2026-09-09 (critique F4 verification)**:
- int8 dtype mismatch: FIXED. `quantize_model_int8(fast=True)` uses
  `FastINT8Linear` with `torch._scaled_mm` (FP8 path). Integration test
  `test_quant_parity.py::test_quant_mode_applies_and_generates[int8]` passes.
- min_p API mismatch in MTP/EAGLE3: FIXED. All decoding classes now accept
  `**kwargs` in their `generate()` signatures (base class has `**kwargs`).
- 3.5x slowdown: STILL OPEN. Requires fused quantized GEMM kernels
  (torch._scaled_mm for FP8, cutlass for INT4). This is deep R&D work.
- compile/cuda_graph crash: STILL OPEN. CUDAGraph tree overwrite on conv
  state clone. Fix: `torch.compiler.cudagraph_mark_step_begin()` before
  each model invocation, or clone outside compile region.

### Decoding Strategies
| Mode | tok/s | Error |
|---|---|---|
| standard | 25.4 | (degraded — runs after quantization tests) |
| speculative | ERROR | "missing required argument 'draft_model'" |
| mtp_selfspec | ERROR | "got unexpected keyword argument 'min_p'" |
| eagle3 | ERROR | "got unexpected keyword argument 'min_p'" |

**CRITICAL**: ALL advanced decoding modes are broken.
- Speculative requires external draft_model (not auto-configured)
- MTP/EAGLE3 have API mismatch — `generate()` passes `min_p` but their
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
argument 'max_new_tokens'" — API mismatch between engine and batch decoder.

### Bottleneck Profiler
All per-layer timings reported 0.0ms with 0 calls — profiler hooks not
attaching correctly to `ModularBlock` layers. Total time 178ms for 16 tokens
(89.6 tok/s) but 100% classified as "non_layer" time.

### State Degradation (STICKY QUANTIZATION)
After running quantization tests, subsequent standard decoding runs at 25 tok/s
instead of 71 tok/s. Quantization is not properly undone when re-activating
without quantize=. This means `activate(quantize=None)` does not restore
original bf16 weights — the quantized weights persist.

### VRAM Leak Between Engine Loads
Retest with fresh engine per quantization mode caused OOM on 3rd load:
w8a8 allocated 25.9GB (2x VRAM!), fp8 load failed. `del engine; gc.collect();
cuda.empty_cache()` is insufficient — quantization creates weight copies that
aren't tracked by the allocator.

### Checkpoint Mismatch
`ForgeLM_V2_Light.sft.safetensors` contains Qwen 2.5 0.5B weights
(151936 vocab, 896 d_model) but config `forgelm_v2_light` expects
(65536 vocab, 2048 d_model). Load fails silently → AirLLM streaming fallback
→ random weights. This is a data integrity issue.

---

## BOTTLENECK SEVERITY RANKING

### P0 — Broken features (block production use)
1. **Quantization is 3.5x COUNTERPRODUCTIVE** — all modes (int4/w8a8/nvfp4/fp8)
   run at 25 tok/s vs 71 tok/s unquantized. Naive dequant, no fused kernels.
2. **int8 quantization BROKEN** — dtype mismatch (BFloat16 vs Float)
3. **torch.compile + CUDA graphs CRASH on conv layers** — CUDAGraph tree
   overwrite on `_conv_state.clone()`. Breaks `activate_optimal()`.
4. **ALL speculative decoding BROKEN** — speculative needs draft_model,
   MTP/EAGLE3 reject `min_p` kwarg
5. **Prefix cache BROKEN** — unpacking error
6. **Batch generation API mismatch** — wrong kwarg name
7. **Sticky quantization** — `activate(quantize=None)` doesn't restore weights
8. **VRAM leak between engine loads** — quantization copies not freed

### P1 — Performance bottlenecks
9. **RotorQuant KV 32% slower** than paged — quantization overhead
10. **Hadamard INT4 KV 42% slower** than paged
11. **Standard KV 22% slower** than paged — memory access patterns
12. **Bottleneck profiler not working** — 0 calls per layer, hooks not attaching

### P2 — Data issues
13. **Checkpoint mismatch** — ForgeLM_V2_Light.sft contains Qwen 2.5 0.5B weights
14. **Python 3.13 compat** — missing `Optional`/`nn` imports (fixed in this session)

Priority key:
- P0 = breaks standard clients / production use
- P1 = important ecosystem compatibility
- P2 = advanced / optional features

---

## A. Inference Engine — Missing vs. vLLM/SGLang/llama.cpp

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
| `top_k`/`repetition_penalty` not forwarded | ABSENT | P0 | Same — declared but dropped in streaming + non-streaming |
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
| HF → Forge conversion | PARTIAL | P1 | Only Qwen3/Gemma3/Llama4 + generic Qwen/Llama remap |
| Export to GGUF/HF/ONNX/TFLite | ABSENT | P1 | Only internal NLRQ int8 export |
| ProgressiveLoader wired | ABSENT | P2 | Class exists, not used by engine |
| Tokenizer training | ABSENT | P2 | LFM2.5 tokenizer only |

---

## B. Trainer — Missing vs. Axolotl/unsloth/LLaMA-Factory/TRL

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
| `forge.training.data.*` modules | MISSING | P0 | `efficient_pipeline`, `parquet_dataset`, `curriculum_augment` imported by `sft_train.py` but not in repo → SFT may fail at import |
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
7. `forge.training.data.*` modules missing → SFT may fail at import
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
1. **Pure ternary (1.58 bpw)**: PPL 3M+ � completely unusable without residuals.
2. **BiLLM alone (no SVD)**: PPL 5568-7616 at ~1.10 bpw � too much distortion.
3. **Mean-centered residuals**: Dramatically worse than absmean (zero-mean) binarization. PPL in hundreds of thousands. Mean-centering is NOT a good fit for this model.
4. **Hadamard rotation**: Helps attention layers (896?1024, 12.5% padding) but HURTS down_proj (4864?8192, 40.6% padding). Auto-disable Hadamard when padding > 15%.
5. **GPTQ compensation**: Inconsistent � improved single-layer SQNR by +2dB but worsened average output SQNR by -0.69dB across layers. Not enabled by default.
6. **SVD residual is essential**: The gap between BiLLM-only (PPL ~5000+) and BiLLM+SVD (PPL ~1096) is enormous. SVD captures the structured low-rank component of the quantization error.
7. **Block-wise NF4 (bs=32)** for SVD factors is the sweet spot. Per-tensor NF4 is too coarse (PPL 18304). Per-row NF4 adds too much scale overhead. bs=16 wastes bits on scales, bs=128+ loses precision.

### Storage Accounting (1.55 bpw)
Per layer (out_f, in_f):
- Salient binary: salient_order � out_f � n_salient bits
- Non-salient binary: 1 � out_f � n_nonsalient bits
- Split mask: 1 � out_f � n_nonsalient bits (stored as bool)
- SVD U: 4 � rank � out_f + 16 � ceil(rank�out_f / bs) bits (NF4 + scales)
- SVD V: 4 � rank � h_size + 16 � ceil(rank�h_size / bs) bits (NF4 + scales)
- Total � (out_f � in_f) = effective bpw

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

---

# R38 Candidate Survey — 6-Topic Research Agenda (2026-09-07)
Sources: 3 parallel codebase audits (optimizers, BitNet/latent, KV/MoE) + AGENTS.md R1-R37 notes.

## 1. GaLore / low-rank gradient projection
- EXISTS: APOLLO (native, SVD-free, orge/training/optim/apollo.py) - wired in configure_optimizer but NOT in sft_train CLI choices. GaLore = thin galore_torch wrapper (training_utils.py:311), orphaned from CLI. FiraNLRQ (low-rank projection inactive).
- GAPS: no native GaLore (project_grad/weight-recovery), no per-layer adaptive rank, no rank annealing, no GaLore-x-NLRQ/LoRA.
- NOVEL: GaLore projected in NLRQ factor subspace (S-vectors already low-rank); rank annealing on loss plateau; GaLore x BitNet int8 masters.

## 2. BitNet b1.58 ternary training
- EXISTS (rich): BitNetLinear STE + learned qscale + int8@int8 GEMM + Triton b1.58 kernel (bitnet_b158_key.py); BitNetEmbedding/Conv1d; --bitnet-everywhere (sft_train default ON); enable_int8_training (int8 GPU + bf16 CPU master); NanoQuantQAT sub-1-bit; TernaryOptimizer 2-bit states (R20, ORPHANED); MuonBitNet4Bit (ORPHANED); BitNetResidualLinear (R24, Key exists, NOT wired into ModelLoader).
- GAPS: no from-scratch ternary pretraining (train_8b_all disables QAT at init); lm_head/embed skipped in HF QAT path; residual key unwired.
- NOVEL: full-ternary training stack = TernaryOptimizer + BitNetResidual + from-scratch ternary init; ternary-native ForgeLM pretrain.

## 3. State-free / schedule-free optimizers
- EXISTS: MuonScheduleFree, MuonSFBlockwise (sft default), SFNorMuon, AMUSE, MONA, FlashAdamW/Lion 8-bit, CPUAdamW (ZeRO offload, wired), BAdam, ForgeOptimizer. ORPHANED: flashoptim, AdamW4Bit, NVMeStreamedBAdam, TernaryOptimizer.
- GAPS: Sophia, SOAP/Shampoo, CAME, AdEMAMix, Adafactor, true state-free class.
- BASELINE RULE: must beat muon_sf_plain (V3) or cpu_offload (V4).
- NOVEL: AdEMAMix w/ slow momentum on CPU + fast on GPU (mixed, directive D); Sophia-H diag-Hessian on ternary; state-free ternary flip-direction optimizer (revive TernaryOptimizer).

## 4. AdaCoM / trainable context management
- EXISTS: 26 KV strategies wired via build_kv_cache (SnapKV, H2O, StreamingLLM, S4R, SpectralKV, HQE, auto_context meta-manager, Matryoshka, VToken...). Learned bits: KVpopScorer MLP (skeleton, NOT wired), ContinuationPredictor (wired), MoSA router.
- GAPS: NO AdaCoM-style trainable/decoupled context manager; no token merging/pruning at runtime; PyramidKV absent; keys/cache + keys/compression dirs EMPTY.
- NOVEL: STE hard-decision context gate (reuse ModRouter machinery from mod_router_key.py) deciding per-token keep/compress/evict, trained end-to-end; wire train_kvpop_scorer as first step.

## 5. Latent representation prediction
- EXISTS: NOTHING. Closest: MTPModule (token-level aux CE), SIGRegLoss (spectral reg), EAGLE (inference-only hidden use).
- GAPS: fully greenfield - no BYOL/VICReg/latent-matching loss anywhere.
- NOVEL: inter-layer latent prediction head (predict h_{L+k} from h_L, continuous MSE/cosine target - no vocab softmax, tiny head, ~0 VRAM); stabilizes ternary QAT (cross-domain w/ #2); boosts self-play sample efficiency; AirMoE latent supervision.

## 6. 30B efficiency frontier / local MoE
- EXISTS: MoELayer (top-k noisy, switch aux, DeepSeek-V3 aux-free bias, shared expert, dense bypass warm-start) - trainable via sft_train; BitNet experts; ExpertTyingKey; ElbowRouter/AllocMoE/LDACalibrator (inference, wired); AirMoE hotswap/infinite (ORPHANED from engine); LASER/METRO routers (ORPHANED).
- GAPS: no fine-grained experts (many-small DeepSeek-style), no fused Triton MoE kernel, no expert parallelism, AirMoE not in forge_engine.
- NOVEL: fine-grained experts (n_experts up, d_ff down, constant active params) + BitNet ternary experts + shared expert + CPU-RAM expert hotload (ExpertHotload domain exists in evolution sim).

## QUICK WINS (do regardless)
1. Expose apollo/galore/flashoptim/nvme_muon_4bit in sft_train --optimizer choices (code exists, unreachable).
2. Wire BitNetResidualLinear into ModelLoader (use_bitnet_residual flag exists, no loader wiring).
3. Wire train_kvpop_scorer skeleton.

## PRIORITY MATRIX (impact x effort x novelty on RTX 5070)
- A. Latent-prediction aux loss: greenfield, ~0 VRAM, helps #2+#5+self-play. HIGH novelty/effort ratio.
- B. Trainable context manager (AdaCoM-style): biggest KV gap, ModRouter STE reusable. HIGH.
- C. Full-ternary training stack: completes BitNet story, TernaryOptimizer exists. MEDIUM-HIGH.
- D. CLI unblock orphans: trivial. DO FIRST.
- E. Fine-grained BitNet MoE: MEDIUM (big test infra needed).
- F. Native GaLore + adaptive rank: MEDIUM (APOLLO already covers much of it).

---

# Web Research Sweep — 2025-2026 LLM Techniques (2026-09-07)
3 parallel agents, web_search only (no webfetch; numbers from abstracts/snippets - verify before implementing). Organized by the 5 focus areas.

## FOCUS 1: Low-mem full-parameter training
- **SubTrack-Grad / SubTrack++** (arXiv:2502.01586, NeurIPS 2025): rank-1 Grassmannian gradient-subspace tracking replaces periodic SVD; projection-aware Adam realigns momenta on subspace shift + recovery scaling. 65% wall-time cut vs GaLore; 3B overhead 31% vs GaLore 157%. FIT HIGH - drop-in upgrade for GaLore wrapper.
- **GaLore 2** (arXiv:2504.20437): randomized SVD + FSDP integration; Llama-7B pretrain 500B tokens. FIT HIGH.
- **Q-GaLore** (arXiv:2407.08296, CPAL 25): INT4 projections + INT8 weights + lazy per-layer subspace updates; LLaMA-7B on RTX 4060 Ti 16GB. FIT HIGH - stack on NLRQ.
- **Adam-mini** (arXiv:2406.16793, ICLR 25): Hessian-block shared LR kills dense v tensor; -45-50% opt memory, +49.6% throughput. FIT HIGH (2-3GB saved at 1.2B).
- **Q-Adam-mini** (ICML 25): Adam-mini + INT8 first momentum; 8x total GPU mem reduction, 60M-8B validated. FIT HIGH.
- **SOLO** (arXiv:2505.00347): fixes signal-swamping in ultra-low-bit EMAs; 2-3 bit Adam states. FIT HIGH - pairs with TernaryOptimizer (ternary weights + 2-bit momentum).
- **GradLite** (arXiv:2510.22467): low-rank Jacobian approx + error feedback; stable with dropped/compressed activations; -50% opt+activation mem. FIT HIGH.
- **ZenFlow** (arXiv:2505.12242): importance-aware offload (top-k grads stay GPU, rest async CPU); 5x vs ZeRO-Offload, -50% PCIe, >85% stall cut. FIT HIGH - CPUAdamW successor.
- **Batch-1 training** (arXiv:2507.07101): Adam beta scaled by token half-life (not steps); batch-size-1 SGD stable, equal/better per-FLOP; kills grad accumulation. FIT HIGH for long-seq on 12GB.
- **BCD** (arXiv:2506.12037): block coordinate descent, only active block's optimizer state on GPU; 7B on RTX 4090. FIT MEDIUM (3-8B later).
- **LLMQ** (arXiv:2512.15306): consumer-GPU CUDA training framework; 7B @ 70% MFU on 16GB 5060Ti. FIT HIGH as reference impl.
- **Sparse MeZO** (arXiv:2402.15751, NeurIPS 25): ZO on param subset; fine-tuning only. FIT LOW for pretrain.

## FOCUS 2: Small-model behavior & potential
- **Overtraining scaling laws** (arXiv:2403.08540, ICLR 25): 104 models 0.011-6.9B; 1.4B @ 32x overtrain predictable from 300x cheaper runs. ACTION: 0.1-0.4B wind-tunnel runs to set 1.2B token budget.
- **Overtraining knee**: 0.9M model @ 222k tok/param degraded after peak (INT 4.55 -> 3.31). Overtraining has a peak - find the knee with proxies.
- **IMU-1 / Qwen3-0.6B repro** (HF 2026): NorMuon on 2D + AdamW on 1D/embed + WSD + z-loss(1e-4) = PPL 28.66->23.52 (-18%). NOTE: NorMuon already in repo (sf_normuon) - recipe validates it.
- **L20-Edu-135M** (arXiv:2606.22189): single-GPU recipe: 10B FineWeb-Edu + 3B math/code/reasoning curated; deep-thin, GQA, tied embed, MinHash/LSH dedup. ACTION: mirror data gate.
- **Data repetition destroys small LMs** (arXiv:2606.24998): 10% FLOPs on repeats = 67% FLOP-equivalent loss @ 344M. ACTION: dedup across epochs, monitor eval loss.
- **Dispersion loss** (arXiv:2602.00217): small LMs condense embeddings into narrow cone; cosine-dispersion aux loss fixes geometry, +10 benchmarks, zero params. ACTION: cheap second aux loss alongside R41 latent head.
- **EOPD** (arXiv:2603.07079): entropy-aware on-policy distillation, reverse+forward KL; Qwen3-0.6B +1.37 / 1.7B +2.39 / 4B +5.05 math.
- **Prefix OPD** (ACL 26): train only on reasoning-trace prefixes; matches full OPD @ 2-40x less FLOP.
- **OPD recipe** (arXiv:2604.13016): teacher must add new capability + compatible reasoning patterns; off-policy cold-start rescues failing OPD.
- **Looped transformers / latent thoughts** (arXiv:2502.17416, ICLR 25): k layers looped L times ~ kL layers on reasoning; zero extra params. FIT HIGH.
- **LOTUS** (arXiv:2606.31779): latent-CoT matches explicit CoT @ 3B; thought latency -2.5-6.9x.
- **Coconut** (arXiv:2412.06769): continuous thought (hidden state as next input); beats CoT on backtracking tasks; needs curriculum.
- **Matryoshka LM suites** (arXiv:2608.09703): nested 500M/1.5B/3B; -36% total train compute, +14-26% spec-decode throughput.
- **MoEsturizer** (ICLR 26): dense->MoE upcycling @ sub-1B with 150k SFT samples; 4-2/8-2 top-k beats dense base. ACTION: replaces from-scratch MoE plan (R43-B) - upcycle instead.
- **MoR - Mixture of Recursions** (arXiv:2507.10524): layer sharing + token-level dynamic recursion depth + recursion-wise KV; 135M-1.7B, 2.18x throughput. FIT HIGH.
- **MoDA** (arXiv:2603.15619): depth-attention (heads read prior-layer KV); 1.5B: PPL -0.2, downstream +2.11%, +3.7% FLOPs only.
- **Sherry** (ACL 26): 3:4 sparse ternary 1.25 bpw packing; 1B Llama-3.2 zero accuracy loss, 25% bit savings.
- **Cloe** (arXiv:2608.28809): ternary QAT degrades MMLU/factual most; task FT recovers 89.8%/79.4%. ACTION: ship dual checkpoints (fp general + ternary specialized).
- **T1** (arXiv:2504.04718, ICLR 26): tool-filtered test-time verification; 1B beats 8B on MATH. Pairs with ForgeAI tool harness.
- **PA-Tool** (arXiv:2510.07248): schema renaming to pretraining-aligned patterns; +17% tool use, -80% schema errors, training-free.
- **Manthan-1.5B** (arXiv:2507.05065): tool-mediated reasoning via GRPO; 65% GSM8K @ 1.5B on T4.
- **Falcon-H1** (arXiv:2507.22448): parallel attn+Mamba2; 0.5B ~ 7B-2024 quality, 256K ctx.
- **Index-1.9B-32K**: 32K ctx via 10B-token long-PT + doc packing w/ reset attn/position IDs.
- **Jet-Long** (arXiv:2607.07740): bifocal RoPE (local faithful + dynamic long window); +2-4.8 RULER zero-shot, <=4% overhead, no retrain.

## FOCUS 3: Context mem cost + token gen speed
- **KV-Direct** (arXiv:2603.19664): K/V are deterministic projections of residual stream; cache 5KB residual/token vs 136KB KV (Gemma3-4B); 42MB vs 103MB peak over 20 turns; recompute up to 5x FASTER than reading cached KV. Validates engine's residual_stream/capture strategies - promote with these numbers.
- **ShadowKV** (arXiv:2410.21465, ICML 25 spotlight): low-rank keys on GPU + values offloaded + sparse retrieval; 6x batch, 3.04x throughput, no accuracy loss. Pairs with cpu_offload tier.
- **STAR-KV** (arXiv:2606.08382, ICML 26): differentiable soft-threshold per-head/per-block rank; 75% KV compression (20x w/ quant), 6.9x attn speedup, 3.1x e2e. Triton kernels - fits stack.
- **MoBA / FlashMoBA** (arXiv:2502.13189, 2511.11571): block-sparse routing; FlashMoBA 14.7x over FA2 @ small blocks. Add to auto_context pool.
- **SelKV** (arXiv:2607.16213): per-token merge-or-drop cosine gate + attention compensation; 25% retention near-lossless, 3.3x @ 100k.
- **SemantiCache** (arXiv:2603.14303): semantic chunking + clustered merging; 2.61x decode.
- **FreeKV** (arXiv:2505.13109): speculative KV retrieval + double-buffered recall; 13x over SOTA retrieval. Layers onto cpu/disk offload tiers.
- **HeteroSpec** (arXiv:2505.13254): entropy-adaptive speculation depth; 4.24x over EAGLE-3, training-free, exact distribution. Direct upgrade to EAGLE/MTP path.
- **HiSpec** (arXiv:2510.01336): early-exit hierarchical verification; 1.28-2.01x.
- **SpecPV** (arXiv:2512.02337): partial-KV self-speculation; 6x long-context decode.
- **MHA2MLA** (ACL 25) / **TransMLA** (arXiv:2502.07864, NeurIPS 25): retrofit GQA->MLA; 92-93% KV cut, ~1% quality drop, 6B tokens FT.
- **xKV** (arXiv:2503.18893): cross-layer aligned SVD of KV; 8x, training-free.
- **Value Residual / SVFormer** (arXiv:2410.17897, ACL 25): value residuals + shared first-layer V; 2x KV cut, -16% params, -20% data.
- **Star Attention** (arXiv:2411.17116): two-phase blockwise context parallel; 11x mem, 97-100% accuracy.
- **Fast-dLLM** (ICLR 26): diffusion LM KV + parallel unmask; 27.6x - arch mismatch, note for future dLLM track.
- **NeuroPrefetcher** (arXiv:2608.22643, ICPP 26): NVMe delta prefetch of sparse rows; 7.9-12x over demand paging. LOW @ 1.2B, HIGH for expert hotload later.

## FOCUS 4: Quants beyond BitNet b1.58
- **BTC-LLM** (arXiv:2506.12040, ACL 26): binary codebook + learnable transform; 0.7-1.11 bpw; 13B @ 0.8bpw = -3.1% zero-shot, 1.6x vs FP16. FIT HIGH @ 1.2B (~200MB resident).
- **LittleBit** (arXiv:2506.13771): latent factorization + binarized factors + residual comp; 0.1 bpw, 31x mem cut, 11.6x kernel speedup. FIT MEDIUM (risky).
- **AQLM 1-bit** (arXiv:2401.06118): additive VQ codebooks; Llama-2-7b 1bit PPL 7.85. MEDIUM (VQ lookup latency @ small models).
- **OptRot** (arXiv:2512.24124): data-free learned rotations minimizing kurtosis; beats SpinQuant/QuaRot/OSTQuant weight-only; fuses into weights. FIT HIGH - upgrade for hadamard_int4/rotorquant paths.
- **QoQ/QServe W4A8KV4** (arXiv:2405.04532, MLSys 25): 4w/8a/4kv co-design; 1.2-3.5x serving. FIT HIGH - exact slot missing in quant stack.
- **KVTuner** (arXiv:2502.04420, ICML 25): layer-wise mixed-precision KV MOO search; 3.25-bit effective lossless, +21% throughput. Plug into HQE/2bit selector.
- **PatternKV** (arXiv:2510.05176): pattern-aligned residual KV quant; 2-bit-equiv gains, 0.08% drop @ 4-bit, 1.5x throughput.
- **AQUA-KV** (arXiv:2501.19392): predictor-based KV residual quant. MEDIUM.
- **QJL** (AAAI 25): 1-bit JL-transform KV, no scale/zero-point metadata; 3-bit effective, >5x mem cut. HIGH - sub-2-bit KV alternative.
- **NVFP4 KV** (NVIDIA Dec 25): FP4 KV w/ E4M3 per-16 scales; -50% vs FP8 KV, <1% loss. Blackwell-native.
- **Sparse-BitNet** (arXiv:2603.05168): 1.58-bit + N:M sparsity joint training; 1.30x, ternary more sparsity-tolerant than BF16.
- Sub-BitNet landscape confirms repo's R25/26 findings: ternary near-optimal at low rates; corrections (low-rank/codebook) beat pure ternary - TernLC approach validated by BTC-LLM/LittleBit line.

## FOCUS 5: Training-process-time minimization
- **NVFP4 pretraining** (arXiv:2509.25149): Hadamard + 2D block quant + stochastic rounding + selective HP layers; 12B/10T tokens, MMLU-pro parity w/ FP8. HIGH on Blackwell.
- **TetraJet-v2** (arXiv:2510.27527): unbiased double-block FP4 + OsciReset; 1.67x vs FP8 e2e; tested to 370M. MEDIUM.
- **u-µP** (arXiv:2407.17465): muP + unit scaling; FP8-stable defaults, proxy->target HP transfer. HIGH - tune 100M proxy, transfer to 1.2B/8B.
- **ScheduleFree+** (arXiv:2605.19095): fixes SF averaging/large-batch issues; beats WSD, +31% @ 1000 tok/param, anytime checkpoints. HIGH - direct muon_sf upgrade.
- **WSM** (arXiv:2507.17634): decay phase -> checkpoint merging; beats WSD, no pre-defined length. MEDIUM.
- **AdEMAMix** (arXiv:2409.03137, ICLR 25): fast+slow EMA; 1.3B @ 101B tokens = AdamW @ 197B (+95% token efficiency). Already on R43 backlog.
- **SOAP / KL-SOAP** (arXiv:2409.11321 ICLR 25; 2509.03378): Adam in Shampoo eigenbasis; >40% fewer iters; KL-SOAP cuts memory. MEDIUM.
- **FA4** (arXiv:2603.05451, MLSys 26): Blackwell-native attention; 1.3x vs cuDNN, 2.7x vs Triton, 1613 TFLOP/s B200. HIGH - watch SM120/consumer support.
- **u-S FP8** (arXiv:2502.05967): scaling rules -> FP8 w/o dynamic scaling; 1B-13B parity +33% faster. HIGH.
- **MuToR** (arXiv:2505.10518): register-token MTP, no extra heads, NTP-compatible. MEDIUM - compose with existing MTP.
- **Influence Distillation** (arXiv:2505.19051): 2nd-order data selection, 3.5x faster selection. MEDIUM (data pipeline).
- **SUS backprop** (arXiv:2505.15080): sparse unbiased attention backward O(n^2)->O(nc), c~25-30, +1% grad variance. MEDIUM for long-ctx training.

## CROSS-CUTTING NOTES
- sf_normuon (NorMuon) already wired - IMU-1 validates at 0.6B scale; adopt full recipe (WSD+z-loss+optimizer split).
- residual_stream + capture KV strategies already in engine - KV-Direct gives them published numbers + recompute-is-faster result; promote + benchmark.
- ExpertHotload sim domain + NeuroPrefetcher = same idea; relevant when MoE upcycling lands.
- MoE upcycling (MoEsturizer) replaces from-scratch fine-grained MoE in R43-B - far cheaper on 12GB.
- Ternary dual-checkpoint strategy (Cloe) fits existing bitnet_everywhere + LoRA pipeline.

---

# 0.5B Time-to-Model Research (2026-09-07, 2 agents)

## A. Measured throughput datapoints (single consumer GPU)
- modded-nanogpt 124M on 1x RTX 4090: 130-163k tok/s, val loss 3.25 in ~90-115 min. Techniques: Muon (77.5% staged gain, largest single), FlexAttention (54.5%), arch modernizations RoPE/QK-norm/ReLU2/untied-emb (45.2%), value emb + U-Net skips (36.5%), FP8, fused CE, packing.
- 2x4090: 1.88B tokens to 3.28 target (vs 6.44B baseline) = 3.4x TOKEN EFFICIENCY from arch+opt tricks.
- 1x RTX 5090 community: 124M to 3.28 in ~42 min. 8x5090: ~5.5 min.
- TinyLlama 1.1B: 17k tok/s on 1x4090 (56% MFU on A100). TinyStories 19M: 90k eager / 127k compiled on 2060 Super (compile = 1.4-1.5x).
- Qwen2.5-0.5B full FT on A100: 41k tok/s @ 39.6% MFU (Chronicals, fused CE 5GB->135MB logits).
- llm.c GPT-2 124M: 62k tok/s on A10G (48h for 10B tokens); ~200k on A100.
- RTX 5070 0.5B estimate: 8-15k tok/s BF16 well-tuned (25-45% MFU on 61.7 TFLOPS peak); best case ~17-19k. FP8 on SM120 EXPERIMENTAL (nanochat: lm_head FP8 only ~1% e2e, +2GB; MXFP8 broken on SM120 in some torchao versions). Do NOT assume FP8 speedup.
- WINDOWS: WDDM ~2x slower host<->GPU transfers (GeForce cannot enable TCC); sysmem fallback ~3x slowdown (set 'Prefer No Sysmem Fallback'); pinned-memory leaks; prefer Linux/WSL2. Native Windows OK only for fully-resident compute-bound loops.

## B. 0.5B from-scratch existence proofs (quality ceiling is LOW)
- NPC Nano 0.5B: 8.93B tokens (FineWeb-Edu mix) -> HellaSwag 36.8, ARC-E 50.0, PIQA 65.0, GSM8K 1.67%. +15B math tokens on top: GSM8K still ~2% (capacity wall).
- ZeroShot-500M: 7.9B tokens on RTX 5090, ~51h, loss 2.75.
- Talon-D1-0.5B: ~10B tokens -> HS 39.1, ARC-C 27.8.
- SparseLM0.5B (Sakana): 10B tokens -> 40.4% mean task acc.
- Qwen2-0.5B (massively more tokens): HS 49.3, ARC-C 31.5, GSM8K 36.5, MMLU 45.4. Gap is huge.

## C. Sample-efficient paths (proven)
- Warm-start Qwen2.5-0.5B-Instruct + 25k-200k examples SFT/distill: 0.5-6h on 5070. DistilQwen2.5-0.5B: AlpacaEval 2.46->4.89, IFEval 42.8->52.6 (100k ex, 3 ep). Sepolian 0.5B-math 25k ex: GSM8K 21.6->36.2.
- DistiLLM-2 (ICML25 oral): 0.5B Qwen student, 50k prompts, win 66-77% vs base vs 1.8-14B teachers. Student cost 1-5h; teacher logit gen = real cost.
- Agent distillation (2k trajectories): 0.5B student ~= 1.5B CoT tier. T1: 1B + tools > 8B on MATH. PA-Tool: 0 training, +17% tool use.
- Low-Rank Clone (LRC): 20B tokens beats 36T-trained Qwen3-1.7B (claims >1000x token efficiency) - plausible for 0.5B.
- Continued pretrain from strong 0.5B ckpt: 0.1B tokens ~4h, 1B ~1.5d, 2B ~3d (with 10-20% replay).
- MoE upcycling at 0.5B on 12GB: NOT feasible/helpful (public demo bigger+slower+worse).
- Phi-1 lesson: textbook synthetic data works (1.3B, ~50B tokens seen incl. 8 passes -> HumanEval 50.6) but 0.5B hits capacity wall on broad tasks.

## D. Recommended recipe (5070, hours-to-1-day)
1. Base = Qwen2.5-0.5B-Instruct (or Coder-0.5B). 2. 25k-200k curated examples (DistilQwen/Magpie/OpenHermes/SmolTalk; DeepMath-103K/NuminaMath; xLAM 60k for tools). 3. Optional teacher pass (local 7B quant or API). 4. Full FT or LoRA 1-3 ep, lr 1e-5..2e-5, seq 2048, bf16, FA, grad-accum to fit 12GB. 5. lm-eval-harness check. 6. Deploy w/ test-time compensation (structured output, retrieval, calculator, self-consistency, PA-Tool schemas).
- Custom tokenizer/arch needed -> from-scratch 7-10B curated tokens = ~5-10 days, base-level quality only.

---

# SOTA Research Sweep for R49+ (2026-09-08)

Sources: K2 Horizon (IFM, 2026-09-06), Uno paper arXiv:2609.04010, Qwen3-Next/Qwen3.5,
Kimi Linear (arXiv:2510.26692), DeepSeek-V3.2 (arXiv:2512.02556), Nemotron 3 Nano/Super,
MiniMax M2/M2.5/M2.7 (arXiv:2605.26494), Muon ecosystem (Dion/Dion2/Dion3, SOAP-at-scale,
MuonClip/QK-Clip, Muon Split), MoDA family (MoD-Attention arXiv:2603.15619, MixDA ACL23,
MoA arXiv:2506.05928, MoDE arXiv:2410.10181).

## T1 — Uno: diffusion-augmented decoding (HIGHEST VALUE)
- Two weight sets: frozen AR weights (NTP) + lightweight diffusion weights trained via
  "Diffusion Distillation" to emit token blocks in parallel. Psi-Spec samplers = LOSSLESS
  (samples from the AR distribution itself). No draft model. Up to 3x speedup; Pareto-dominates
  EAGLE-3/DFlash at every batch size; lowest added params + memory of any spec method.
- Shipped in K2 Horizon as conditional-LoRA adapters (392 tensors, 7B + 0.9B sizes).
  Code: github.com/ifm-ai/uno. HF: s-sahoo/uno collection.
- ForgeAI fit: new `forge/decoding/uno.py`; adapter via existing LoRA stack; phase mgmt in
  faser.py; combine with adaptive_speculative (diffusion for large batch, EAGLE for small).
- Novel twist: self-distillation Uno on ForgeLM 1.2B (base distills its own AR distribution,
  no external teacher); Psi-Spec + Vegas verification-guided KV selection share verify pass.
- VRAM: adapter ~2-5% of base params; block-parallel decode reuses existing KV. Fits 12GB.

## T1 — KDA / Kimi Linear (GatedDeltaNet successor)
- KDA = GDN + fine-grained channel-wise decay Diag(alpha_t) (per-dim forget gate vs GDN's
  head-scalar gate); efficient via specialized Diagonal-Plus-Low-Rank transition; chunkwise.
- Hybrid 3:1 KDA:MLA BEATS full MLA at equal recipe; -75% KV cache; up to 6x decode TPOT @1M ctx.
- Kernels open-sourced in FLA (fla/ops/kda) + vLLM impl. 48B-A3B checkpoints public.
- ForgeAI fit: new linear-attention key (forge/keys/attention/); hybrid ratio knob already
  exists via layer_types (ForgeHybrid). Pairs with mamba3_key lineage; port-first rule applies
  (identity warm start from GDN-style init).
- Novel twist: KDA decay rates driven by LeRoPE learnable frequencies (merge two existing keys).

## T1 — MoVA: Mixture-of-Value Attention (K2 Horizon 36B-A4B)
- Sparsity moved INTO attention: 64 value experts, 4 active/token; 45/48 layers MoVA;
  MoE FFN 100 experts/8 active; GQA 32/8 heads; compatible w/ FlashAttention + sparse attn.
- 36B-A4B ~= dense 32B trained identically (controlled comparison).
- ForgeAI fit: new attention key; router machinery reusable from moe/moe.py; synergy with
  GTA (tied V=K) and GLA (latent KV). Value experts can be BitNet/IRI-FP4 quantized for VRAM.
- Novel twist: value-expert hotswap via AirMoE infra (disk-backed value experts on 12GB).

## T1 — DSA lightning indexer (DeepSeek-V3.2)
- Per-layer tiny low-head scorer (FP8, Hadamard/rotate_activation orthogonal transform) +
  top-k (2048) token selection -> additive mask into main attention. Own 1-head index-K cache.
- Continued-train from dense: aligns distribution, quality parity on long context.
- ForgeAI fit: upgrade QSA/CSA keys (qsa_key.py/csa_key.py) with trainable indexer; synergy
  with Vegas (verification-guided selection) + compact_attention (block-union).
- Novel twist: distill indexer from base model attention maps (training-free warm start);
  shared indexer across layer groups to cut index-K cache.

## T2 — Gated Attention + zero-centered RMSNorm (Qwen3-Next/3.5)
- Attention output gate (attn_output_gate) + weight-decayed zero-centered RMSNorm gamma.
  Cheap stability wins at small scale. ForgeAI has QK-norm; add output gate + zc-RMSNorm flags.

## T2 — Muon ecosystem (K2/GLM-5/DeepSeek-V4 all use Muon now)
- QK-Clip: clip Wq/Wk update rows when attention logits explode -> zero loss spikes @15.5T tokens.
- Muon Split: per-head orthogonalization for MLA up-projections (GLM-5).
- Dion2/Dion3: sample fraction of rows/cols before orthogonalization -> up to 6x cheaper step,
  matches Muon loss. SOAP: per-step QR fixes large-batch instability.
- ForgeAI fit: extend muon_sf_blockwise.py / sf_spectral_optimizers.py (SFNorMuon) with
  Dion2-style sampled orthogonalization (single-GPU: NS-iteration cost is the bottleneck);
  QK-Clip as opt-in guard in sft_train.py.

## T2 — LatentMoE (Nemotron 3 Super) + router variants
- Compress tokens to latent dim (1024) BEFORE experts -> 4x more experts (512/top-22) same cost.
  ForgeAI: moe.py + AirMoE (latent compression also cuts expert disk I/O for hotswap).
- MiniMax M2: sigmoid gating (not softmax) + 256 fine-grained experts top-8; QK-Norm + partial RoPE.
- Qwen3-Next: 1:50 sparsity (512 experts, 10+1 active). Add all as Router variants + presets.

## T2 — MTP scaling (Qwen3.5 multi-step, Nemotron 2 shared-weight MTP, MiniMax 3 modules)
- MTP modules double as speculative draft paths. ForgeAI mtp.py + mtp_key exist; extend to
  multi-step/multi-module configs; MTP depth as evolution domain knob.

## T3 — MoDA: Mixture-of-Depths Attention (arXiv:2603.15619)
- Heads attend to current-layer sequence KV + depth KV from ALL preceding layers; fused kernel
  97.3% of FA2 eff @64K; +0.2 ppl, +2.11% downstream, +3.7% FLOPs; better with post-norm.
- ForgeAI fit: new key bridging mod_router_key + attn_residual_key + residual_cache infra.

## T3 — MoDA/MixDA/MoA/MoDE adapter family
- MoA (2506.05928): HETEROGENEOUS adapter experts (LoRA+DoRA+PiSSA mixed) + token-level routing
  beats homogeneous MoE-LoRA. ForgeAI has all 3 adapter types + forge_adapter entropy fusion ->
  heterogeneous fusion is a novel combo for forge_adapter.py.
- MixDA/MoDE: domain adapters parallel to FFN / layer-level domain experts -> adapter library
  with AirMoE hotswap; two-stage (domain-unlabeled then task-labeled) fits SFT runner.

## T3 — RL/systems (MiniMax M2.5/M2.7, DeepSeek-V3.2)
- CISPO for MoE RL stability (grpo_trainer.py); prefix-tree merging ~40x RL speedup;
  windowed-FIFO scheduling; agentic task synthesis (1800 envs / 85k prompts).
- MiniMax kept FULL attention deliberately (hybrid rejected: eval bottleneck, RL scale,
  low-precision traps) — counterpoint datapoint for ForgeEvolve hybrid-vs-full scoring.
- Kimi K2 Thinking: INT4 QAT as production path (ForgeAI int4/sub-bitnet lineage).
- Nemotron 3 Super: NVFP4 PRETRAINING (not just inference quant) — gap in ForgeAI quant stack.

## K2 Horizon release notes (openness angle)
- 6 models 0.9B-375B-A23B, Apache 2.0, intermediate checkpoints + data recipes + logs public.
- 512K native context (flagship); reasoning_effort + k2_horizon parsers in vLLM/SGLang.
- Directly usable: warm-start/distillation source for ForgeLM (per 0.5B section above).

---

# SOTA Research Sweep — Round 2 (2026-09-08, 2 agents)

## Linear attention / SSM frontier
- **RWKV-7 Goose**: per-head matrix state, generalized delta rule w/ vector decay w_t + per-channel
  ICL rate a_t; S_t = S_{t-1}(diag(w_t) − κ̂_t^T(a_t⊙κ̂_t)) + v_t^T k̃_t. 2.9B = 3B SOTA.
  FLA kernels (fla/ops/rwkv7). 12GB-feasible naive PyTorch; state fp32/bf16 (drift when gate≈1).
- **KDA**: per-key-dim forget gate α_t∈[0,1]^{d_k}; S_t = Diag(α_t)S_{t-1} + β_t k_t(v_t − (Diag(α_t)S_{t-1})^T k_t)^T;
  DPLR form D=Diag(α), a=βk, b=k⊙α. FLA fla/ops/kda + vLLM impl. Stability: gate lower-bound,
  L2-normalized q/k. Generic DPLR kernels overkill → bespoke chunk kernel needed for speed.
- **Gated DeltaNet exact inits** (Qwen3-Next): g_t = exp(−exp(A_log)·softplus(a_t+dt_bias)),
  β=sigmoid(b); A_log init log(uniform(0.01,16)); dt_bias softplus-inverse [1e-3,0.1]; conv kernel 4;
  q/k L2-norm required; chunkwise via WY/Householder product. β>1 (negative eigenvalues) unlocks
  state tracking but destabilizes.
- **Mamba-3 details** (validates/extends our mamba3_key): 3-term recurrence (exponential-trapezoidal
  discretization) h_t = α h_{t-1} + β B_{t-1}x_{t-1} + γ B_t x_t; complex SSM = real SSM + data-dep
  RoPE on B/C; MIMO rank-R=4 → state update is matmul not outer product. +0.6pp SISO / +1.2pp MIMO
  vs GDN at 1.5B; HALF Mamba-2 state size. state-spaces/mamba mamba3.py. Angles need fp32.
- **Hybrid ratios**: Qwen3-Next/Kimi 3:1; Nemotron ~4:1; Zamba 6:1; MiniMax-01 7:1. 1B-scale ablation
  (Bae et al., 60B tok): best quality 1:1, best quality/efficiency ~1:5; attention anchors mid-stack.

## Diffusion decoding (beyond Uno)
- **BD3-LM**: block-AR + masked diffusion inside block; exact likelihood; blocks 4/8/16.
- **LLaDA 2.0**: AR→dLLM via 3-phase block-WSD (warmup-block→full-seq→decay-block); block=32,
  confidence-aware parallel (CAP) decoding; 2.1x accel.
- **DiffusionGemma** (Gemma 4 26B-A4B): encoder prefills KV, decoder denoises 256-token canvas;
  entropy-bounded denoising (bound 0.1), renoise non-selected, stop at avg entropy<0.005 + 2 identical
  consecutive predictions; 700+ tok/s on RTX 5090. Canvas sampler easy to prototype in PyTorch.
- **DFlash**: block-diffusion drafter, >6x lossless, +2.5x over EAGLE-3 — needs draft model (heavy).
- **SparseSpec** (already in ForgeAI R39): self-speculative, most 12GB-friendly of the family.
- Verdict: Uno (adapter, no draft model) is the right entry; canvas sampler = research side-quest.

## Optimizers / training
- **QK-Clip exact**: track per-head S_max = max|q·k|/sqrt(d); if > tau (30 or 100): γ=tau/S_max,
  W_q^h←sqrt(γ)W_q^h, W_k^h←sqrt(γ)W_k^h; MLA: W_uq/W_uk get sqrt(γ), shared rotary W_qr gets γ.
  Only violating heads capped. Cost ≈ one L×H max-reduce. Refs: Megatron core/optimizer/qk_clip.py.
- **Dion2**: row/col subsample before orthogonalization, rank_fraction=0.25 → 1B/100B tok loss
  2.635 vs Muon 2.623; Dion3 = Gram Newton-Schulz + CuteDSL + megabatch, up to 6x step time.
  Sampling wins when rank_fraction ≤ 0.25 at ≥1B. github.com/microsoft/dion.
- **NorMuon**: +21.7% vs Adam / +11.3% vs Muon at 1.1B, same memory (per-row 2nd-moment normalize).
  ForgeAI already has SFNorMuon — validate full recipe (WSD + z-loss + optimizer split).
- **ALF-LB** (DeepSeek): expert bias b_k += -u(load_k - target) per batch, no aux loss.
  **Sigmoid gating** (MiniMax M2): sigmoid(logits)+e_score_correction_bias, top-k, renormalize.
  → both are cheap Router variants for moe.py.
- **NVFP4 pretraining recipe** (Nemotron 3 Super): E2M1 + 16-elem microblocks + FP8 scales + RHT +
  stochastic rounding; BF16 kept: final 15% layers, latent proj, MTP, QKV/attn proj, embeddings;
  Mamba output MXFP8. SM120: official TE fused NVFP4 FAILS (232KB smem, no .rs instr);
  torch._scaled_mm via torchao _addmm_nvfp4_dispatch works w/ separate quant kernels.
- **INT4 QAT**: BF16 master + fake QDQ + STE; W4A16 serving; keep lm_head/embed/router high-prec;
  ~5k steps vs original FP distribution (Gemma recipe). Trivial on 12GB for 1.2B.
- **CISPO** (MiniMax): L = -A·sg(clip(r,1-ε_lo,1+ε_high))·logπ — clipped IS weight DETACHED so every
  token keeps gradient; matches DAPO in ~half the time, beats GRPO under high off-policy reuse.
  torchrl CISPOLoss exists. Prefix-tree merging: trie over multi-turn rollouts, reuse shared
  prefix KV — big win for agentic GRPO in self_play.

## Speculative decoding 2026 (no draft model)
- DFlash (block-diffusion drafter, >6x, +2.5x over EAGLE-3; needs drafter weights — heavy for 12GB),
  DiffuSpec (causal-consistency path search, 3x), Spiffy (draft graphs, 6.3x token rate),
  trajectory-level diffusion speculation. Best 12GB fit remains self-speculative sparse attention
  (SparseSpec — already wired R39) and Uno (adapter-only).

## R49 prioritization inputs
- Uno: no arch change, lossless fallback, adapter-sized cost → highest value/effort.
- KDA: biggest arch win (−75% KV, ≥full-attn quality at 3:1) — port via ForgeHybrid zero-init gate.
- MoVA: novel attention-sparsity axis; zero-init router = lossless; composes w/ BitNet experts.
- DSA indexer: lossless at k=∞; distill from attention maps = training-free warm start.
- NVFP4 pretraining: high value, SM120 kernel work required (torchao TE path broken on SM120).
- RWKV-7: optional 4th linear-attention family; FLA kernels exist; lower priority than KDA
  (KDA strictly extends GDN which Qwen3-Next/Kimi validated at scale).

---

# SOTA Research Sweep — Round 3 (2026-09-08, 3 agents)

## A. Reasoning / test-time compute
- **Thinking budgets**: Qwen3 `thinking_budget` (accuracy ~log-linear in budget; splice stop-think
  transition); GPT-5.x `reasoning_effort` (23x token spread), Gemini `thinking_level`, Claude 4.7+
  adaptive-only; s1 budget forcing (AIME24 50->57 on 32B). Interleaved thinking: Qwen3-2507 style,
  0.6B artifact exists (Jarrodbarnes HF).
- **Entropy-guided early exit**: EAT (stop-think + entropy monitor; 12-22% token cut, no acc loss;
  github.com/xidulu/EAT), EntroCut (prefix entropy, 40% cut, arXiv:2601.22617), ASAG (attention-state
  entropy, arXiv:2606.15070), ETR (entropy-trend REWARD: +9.9% acc, -67% CoT len on 7B;
  github.com/Xuan1030/ETR), SPREG (+20% AIME25), LZ Penalty (LZ77-codelength repetition penalty,
  TMLR 2026 — drop-in logit processor).
- **Latent reasoning**: Coconut (hidden-state feedback), Soft Thinking (prob-weighted embedding mix,
  +2.48% pass@1 / -22.4% tokens, TRAINING-FREE — lowest-risk 1B experiment), SoftCoT (projector),
  NoisyCoconut (noise+consensus -> selective abstention). Verdict: latent reasoning NOT proven <3B;
  Soft Thinking is the only cheap 1B experiment.
- **Rubric/generative rewards**: Kimi K2 self-critique rubric (core/prescriptive/human rubrics);
  GenPRM 1.5B > GPT-4o on ProcessBench, RM-R1 +13.8% vs 405B; reward-hacking audits: IFM K2 Horizon
  70.2%->66.9% after exploit removal (harbor analyze); TRACE/ARA/HackProbe detectors.
- **TTS consensus**: RL internalizes search; exploitation (selection) is the bottleneck on open-ended
  tasks (judge-reward corr rho~0.12); Hybrid TTS +28.6%; BG-MCTS (budget-guided) beats budget-agnostic.
  Play: GRPO trains dense PRM -> PRM guides MCTS/BoN at inference.
- **Concise-CoT training family**: DSS-GRPO (think/answer segment masks, separate returns), ETR,
  CRT, Extra-CoT (73% reduction on Qwen3-1.7B), TH2T (-70% easy tokens), Budget Guidance (63% tokens,
  acc held). GLM-5 uses DSA + async RL (slime) + segment-wise GRPO. DeepSeek-V4: CSA/HCA hybrid attn,
  1M ctx, 10% KV vs V3.2.

## B. Agentic data / RL environments / memory
- **Reproducible synthesis**: NeMo Gym (21 RLVR envs + 37 datasets, Apache-2.0, CPU envs, documented
  single-GPU 1B GRPO configs) = best fit for ForgeEvolve; ToolACE (26.5k APIs/390 domains, Apache-2.0
  dataset); xLAM/APIGen (3.7k executable APIs, 60k verified samples, code Apache); Magpie (4M+ samples);
  ToolGrad (answer-first tool chains, ~100% pass). DeepSeek/Kimi/MiniMax pipelines NOT open (concept only).
- **Single-GPU RL**: NeMo-RL has 1B GRPO recipes; mini-grpo (~500 LOC, no critic, LoRA, 3090-class);
  OverlapRL (async staleness-aware single-GPU); 4-bit+LoRA GRPO on 8GB. verifiers (willccbb) env lib MIT.
- **Prefix sharing for GRPO**: Prefix Grouper (MIT; shared-prefix forward, mathematically equivalent,
  kills redundant O(P^2) prefill); Tree-GRPO (tree rollouts at ReAct step nodes -> step-level supervision,
  1/4 rollout budget); vLLM --enable-prefix-caching; psRL 5.2x (closed).
- **PRM/ORM**: RLVR rule rewards dominant; learned PRMs hackable (DeepSeek-R1 skipped them); generative
  verifiers (TANGO co-trained, STV, pairwise Swiss-tournament self-verification); PRM-free step credit:
  SPRO, VeriGate, SC-GRPO, lambda-GRPO.
- **Memory systems (GUI chat/lorebook upgrades)**: Mem0 (add-only extraction + entity linking + hybrid
  retrieval; LoCoMo 92.5 / LongMemEval 94.4, Apache-2.0), Letta/MemGPT (in-context blocks + archival,
  git-backed MemFS), HippoRAG 2 (personalized PageRank over KG), A-Mem (Zettelkasten auto-linking),
  Zep/Graphiti (bi-temporal KG + MCP server).
- **Self-evolving agents**: M2.7 (own scaffold edits, +30%), DGM (SWE-bench 20->50% via self-mutation +
  archive), SE-Agent (trajectory revision/recombination). Safety: sandbox + A/B gates + kill-switch;
  scaffold/prompt evolution only, never autonomous weight/infra edits.

## C. Edge / on-device
- **Gemma 3n**: MatFormer nested elastic sub-models (Mix-n-Match runtime slicing), PLE per-layer
  embedding offload to disk, AltUp predict-correct width widening, LAuReL rank-64 augmented residual +
  15-layer KV sharing. PLE-style offload + runtime spec = portable; rest needs retrain.
- **LFM2/2.5**: NAS-chosen gated short-conv + sparse GQA hybrid (1.2B = 10 conv + 6 attn layers);
  LFM2.5: 128K vocab/ctx + native tool calling; LFM2-VL = SigLIP2 NaFlex + dynamic image-token budget
  (directly relevant to V11/V12 SigLIP2 tower). 2x CPU prefill vs same-size transformers.
- **SmolLM3**: NoPE (drop RoPE every 4th layer) + YaRN + 3-stage data mix (web 85->75%, code 12->15%,
  math 3->10%); dual-mode /think //no_think. Qwen3-0.6/1.7B = best local distillation teachers.
- **llama.cpp 2026**: NVFP4 merged upstream (Blackwell TC), MXFP4 in ik fork; TurboQuant KV
  (turbo4_0 4.5bpw 3.6x / turbo2_0 2.5bpw 6.4x, mixed K/V policies, fused FA); IQK CPU +150-350%
  prompt processing; DSpark/DFlash/MTP in llama.cpp.
- **Windows/Blackwell**: WDDM mandatory on GeForce (no TCC); disable sysmem fallback (NVCPL) to avoid
  paging; Triton >=3.5.1/3.6 fixes sm_120 segfaults (triton-lang/triton-windows wheels; torch.compile
  works); --enforce-eager costs ~12x on affected workloads. CUDA >=12.8, PyTorch >=2.7.
- **KV edge frontier**: CLA (2x), YOCO, FusedKV; EG-MLA >91% KV cut; SelKV (25% KV -> 3.3x decode @100K,
  near-lossless GQA); ZSMerge (5% retention, 3x @54K); KVSlimmer (Hessian-exact); **OasisKV: hot KV in
  HBM + full cache in host RAM + spec-drafted LOOKAHEAD prefetch** (composes with Uno draft blocks +
  our cpu_kv_offload); TurboQuant in llama.cpp (fastest win for existing weights).
- **Audio**: Kyutai STT (streaming Moshi, 1B/2.6B, word timestamps) = only practical open streaming STT;
  separate pipeline, not a V10 graft.

## D. Cross-connections found (novel combos for ForgeAI)
- OasisKV lookahead prefetch x Uno block drafts x cpu_kv_offload = spec-guided tiered KV (new).
- Prefix Grouper x ForgeEvolve GRPO (shared system prompts across group) = free rollout speedup.
- ETR/DSS-GRPO rewards x grpo_trainer.py = concise-CoT + segment masks (highest-yield 1B win).
- GenPRM-style 1.2B judge x ForgeEvolve scoring (replaces/augments simulators for open-ended domains).
- MrRoPE mixed-radix theory (arXiv:2601.22181) unifies PI/NTK/YaRN as radix conversions -> upgrade
  path for LeRoPE (learnable radix schedule). LaMPE: sigmoid length-adaptive mapping, training-free.
- Jet-Long (bifocal dynamic RoPE) already wired as `use_jet_long` — MrRoPE/LaMPE are its successors.
- NoPE-every-4th-layer x hybrid layer_types: free config knob for V12 retrain.
- BLT patches (entropy-based dynamic patching) + Fast-BLT (BLT-D diffusion / BLT-S self-spec, >50%
  bandwidth cut) — research-only for ForgeAI (tokenization/ was deleted), but BLT-S self-speculation
  pairs conceptually with Uno.
- LFM2-VL dynamic image-token budget x V11 SigLIP2 tower = vision token budgeting for V12 VL.

---

# R49 Implementation Log (2026-09-08)

## Implemented (all tests green)
- **Uno decoding** (`forge/decoding/uno.py` NEW): UnoDecoding strategy w/ Psi-Spec
  lossless verify (greedy = bit-exact vs AR; sampling = rejection sampling when
  proposer gives logprobs), NgramProposer (training-free suffix drafting),
  DiffusionGemma-style entropy-bounded drafting stop, lossless fallback
  (proposer=None → StandardDecoding). Wired: engine/decoding.py build_decoding("uno")
  (lazy import — circular), forge_engine._activate_decoding("uno"),
  forge_gui/api/activation_catalog.py DECODING_OPTIONS += uno.
- **MoE sigmoid gating** (moe.py): Router/MoELayer/replace_ffn_with_moe `gating=`
  param; sigmoid scores for top-k, renormalized selected weights; aux_free path
  uses full sigmoid probs. Default softmax = bit-exact unchanged.
- **Dion2 row-sampling** (muon_sf_blockwise.py): `rank_fraction` param (default 1.0
  = unchanged); vendored `_newton_schulz`/`_muon_update_fallback`/`_dion2_update`;
  muon + schedulefree imports now OPTIONAL (module importable/testable without
  them — fallback base = torch.optim.Optimizer, plain AdamW for SF side).
  NOTE: real `muon`/`schedulefree` pkgs NOT installed in the Python313 env —
  module previously unimportable there; now works with fallbacks.
- **QK-Clip** (`forge/training/optim/qk_clip.py` NEW): QKClipMonitor.attach(model, tau)
  → per-head max|logit| observation via GQA forward hook (model_loader.py, training-only,
  zero overhead when off) + clip() rescales Wq/Wk rows by sqrt(gamma) (GQA: group-max
  gamma for shared KV heads). sft_train `--qk-clip-tau` (0=off) + GUI finetune spinbox.
- **CISPO** (grpo_trainer.py): rl_algorithm="cispo" — L = -A·sg(clip(r,1-ε_lo,1+ε_hi))·logπ,
  detached IS weight (no PPO dead zones), no KL. ε_lo=0.2/ε_hi=0.28 defaults.
- **ETR entropy-trend reward**: use_etr_reward + etr_coeff — bonus for downward
  entropy trend across completion, added to advantage pre-loss.
- **FIXED pre-existing bug**: grpo_trainer train_step `total_kl += kl.detach()`
  referenced undefined `kl` in GTPO branch (would crash any GTPO end-to-end run);
  both GTPO + CISPO branches now define `kl = zeros` alongside kl_loss.
- Tests: test_r49_uno.py (7) + test_r49_phase0.py (18). Related suites re-run:
  moe/grpo/model_loader/engine_fixes 84 ✓, gui+catalog 276 ✓, r37+r49 153 ✓,
  r38/r39 228 ✓. Full suite (minus crash file): **2515 passed, 15 skipped**.

## PRE-EXISTING BUG (confirmed, not mine — needs a fix session)
- Full-suite run hard-crashes (native access violation, no pytest summary) at
  `test_r36_empty_states.py::test_maybe_show_onboarding_shows_when_not_onboarded`
  WHEN test_r36_empty_states runs BEFORE test_r36_gui in one session.
  Bisect: empty_states alone PASS (14), gui alone PASS (17), empty_states→gui CRASH,
  download→empty_states→gui CRASH. Reproduces with my changes excluded.
  Suspect: module-scoped `qapp` fixture + OnboardingDialog construction after
  another module created/destroyed Qt state (native Qt lifecycle bug on PySide6).
  Next session: make qapp fixture session-scoped shared helper or defer dialog
  construction; verify with `pytest tests/unit/test_r36_empty_states.py tests/unit/test_r36_gui.py`.

## R49 remaining (next session)
- KDA key (port-first via ForgeHybrid zero-init gate pattern) — biggest arch win.
- MoVA key (zero-init router = lossless; BitNet value experts twist).
- DSA lightning indexer (lossless at k=∞; distill from attention maps).
- zc-RMSNorm + attention output-gate flags (lossless identity at init).
- Uno diffusion-distillation trainer (self-distillation from own AR distribution).
- MTP multi-step heads; sigmoid/ALF router preset wiring in config.py.




