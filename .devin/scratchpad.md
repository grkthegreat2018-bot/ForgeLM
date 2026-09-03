# ForgeEngine R&D Research — Frontier Gap Analysis
Date: 2026-09-03
Source: 16 web searches across NeurIPS/ACL/arXiv 2025-2026 + ForgeEngine inventory

## Current ForgeEngine capabilities (inventory)
- Loading: from_checkpoint (prequant/standard/fallback/streaming), sleep/wake
- Generation: generate, generate_stream, generate_raw, generate_adaptive, generate_batch, generate_with_tools
- KV cache: 14 strategies (standard, paged, rotorquant, hadamard_int4, compressed, streaming, snapkv, snapkv_4bit, paged_eviction, xquant, cpu_offload, s4r, hqe_kv, 2bit)
- Decoding: standard, speculative, medusa, dspark, eagle3, mtp_selfspec
- Quantization: none, int8, int4, fp8, w8a8, nvfp4, BitNet ternary
- Acceleration: none, cuda_graph, airllm_streaming, megakernel, flex_decoding
- Innovations: MRL-AdaptiveContext, QuaRotKV, V0WarmStart, ProgressiveKV
- Diagnostics: benchmark, bottleneck, stats, vram_usage, diagnose, read_log, read_output
- LoRA: load_lora, unload_lora, has_lora, lora_info
- Session: begin/continue/pin/unpin/end_session, session_stats
- Library: save/set_enabled/set_budget/lookup/search/optimize/stats
- Merge/Evolve: merge_checkpoints, evolve_merge (just added)
- Schedulers: fastserve, jet_long, kairos, lampe, libra, triroute, unified_radix, feather, hotprefix, corun, async_d2h, moe_optim
- Attention: atflash, compact_attention, cosa, fa4_attention, faser, pod_attention, flex_decoding, fused_qk_norm_rope_cache, seq_aware_split

## GAP ANALYSIS — Prioritized by value × fit

### TIER 1: High-value, directly fits existing architecture

#### 1. Self-Speculative Decoding with Sparse Attention (SparseSpec/Vegas)
- **What**: Use the same model as both draft AND target model. Draft phase uses sparse attention (only critical KV entries), verify phase uses full attention. Verification attention scores are reused to identify critical KV entries → zero-overhead KV selection.
- **Papers**: SparseSpec (MLSys 2026, 2.13x throughput), Vegas (arXiv 2602.07223, 1.25-2.81x over vLLM)
- **Why for ForgeAI**: 12GB VRAM can't fit a separate draft model. Self-speculation needs zero extra memory. The existing snapkv/s4r KV selection can be enhanced by reusing verification scores. Lossless.
- **Implementation**: New decoding strategy "self_speculative_sparse" that uses existing KV cache strategies in sparse mode for drafting, full mode for verification.

#### 2. Test-Time Scaling / Search Methods (FFS, DORA, VG-Search, AB-MCTS)
- **What**: Inference-time compute scaling via parallel sampling, tree search, and adaptive resource allocation.
- **Papers**: FFS (arXiv 2505.18149, +15% AIME, training-free), DORA (NeurIPS 2025, optimal allocation), VG-Search (NeurIPS 2025, 52% FLOP reduction), AB-MCTS (arXiv 2503.04412, adaptive wider-vs-deeper)
- **Why for ForgeAI**: Pure inference-time, no training needed. FFS is trivially simple (launch N samples, return first to finish). Composes with generate_batch. The self-play subsystem already has evaluation infrastructure.
- **Implementation**: New methods: generate_first_finish(), generate_beam_search(), generate_mcts(), generate_dora()

#### 3. Constrained/Structured Decoding (XGrammar, DCCD)
- **What**: Guarantee output conforms to JSON schema / context-free grammar via token masking at each step.
- **Papers**: XGrammar (arXiv 2411.15100, 100x speedup), DCCD (arXiv 2603.03305, +24pp accuracy), CFGzip (arXiv 2605.29986, 7.5x speedup)
- **Why for ForgeAI**: The agent harness (generate_with_tools) needs reliable JSON tool calls. Currently relies on parsing which can fail. Constrained decoding eliminates parsing failures entirely. XGrammar co-designs with inference engine to overlap grammar computation with GPU.
- **Implementation**: New method generate_structured(prompt, schema) with grammar-based token masking in the logits processor.

#### 4. Verification-Guided KV Selection (Vegas co-design)
- **What**: Instead of standalone KV selection algorithms, reuse attention scores from the verification/full-attention phase to identify critical KV entries for the next draft step.
- **Papers**: Vegas (arXiv 2602.07223)
- **Why for ForgeAI**: ForgeEngine has 14 KV cache strategies, all using standalone selection. Vegas shows this is suboptimal — the criticality information is already computed during full attention but thrown away. Zero-overhead improvement to snapkv/s4r.
- **Implementation**: Modify the KV cache strategies to accept "attention hints" from the previous forward pass.

### TIER 2: High-value, moderate implementation effort

#### 5. Streaming Safety Guardrails (SIREN, StreamGuard, HiddenGuard)
- **What**: Real-time harmful content detection during generation using internal model representations (not a separate guard model).
- **Papers**: SIREN (ACL 2026, 250x fewer params than guard models), StreamGuard (arXiv 2604.03962, forecasting-based), HiddenGuard (ACL 2026, token-level redaction via hidden states)
- **Why for ForgeAI**: ForgeEngine has tool_security.py but no output safety. SIREN uses internal layers (lightweight probes) — fits the 12GB constraint (no separate guard model). StreamGuard does forecasting on partial generations.
- **Implementation**: New SafetyGuardrail class with streaming detection hooks in generate_stream.

#### 6. Learned Prefix Cache Eviction (LPC) + Workflow-Aware Prefetching (KVFlow)
- **What**: Replace LRU eviction with learned policy that predicts which conversations will continue. For agent workflows, prefetch KV from CPU→GPU based on agent step graph.
- **Papers**: LPC (NeurIPS 2025, 18-47% cache size reduction), KVFlow (NeurIPS 2025, 1.83-2.19x speedup for agent workflows)
- **Why for ForgeAI**: ForgeEngine has LRUPrefixCache + ChunkedPrefixCache + session_cache. The agent harness (generate_with_tools) creates multi-step workflows where KV prefetching would help. 12GB VRAM means cache eviction policy matters a lot.
- **Implementation**: Upgrade prefix_cache.py with learned eviction. Add workflow-aware prefetching to session_cache.

#### 7. MoE Inference-Time Load Balancing (LASER, LPR, METRO)
- **What**: Plug-and-play routing algorithms that balance expert load at inference time without retraining.
- **Papers**: LASER (arXiv 2510.03293, plug-and-play), LPR (arXiv 2506.21328, Gini 0.70→0.035), METRO (arXiv 2512.09277, balances activated experts not tokens, 11-22% decode latency reduction)
- **Why for ForgeAI**: ForgeEngine has MoE support (moe/ keys, moe_optim scheduler). LASER is plug-and-play (no retraining). METRO's insight about memory-bound regime is directly relevant to 12GB VRAM.
- **Implementation**: New inference-time router in moe/routers.py that adapts based on gate score distributions.

#### 8. Graft: Prune-then-Retrieve Tree Construction for Speculative Decoding
- **What**: Training-free, lossless. Couples dynamic-depth pruning with retrieval to fill topological gaps in draft trees.
- **Papers**: Graft (arXiv 2605.20104, 5.41x speedup, +21.8% over EAGLE-3)
- **Why for ForgeAI**: ForgeEngine has eagle3 decoding. Graft is a training-free enhancement. The retrieval component could use the existing prefix_cache infrastructure.
- **Implementation**: Extend the eagle3 decoding strategy with prune-then-graft logic.

### TIER 3: Novel but higher effort / longer-term

#### 9. In-Context RL (ICRL) Prompting
- **What**: LLMs can perform RL during inference via multi-round prompting with scalar reward feedback.
- **Paper**: "Reward Is Enough" (arXiv 2506.06303)
- **Why for ForgeAI**: Novel inference-time self-improvement. Could enhance agent loops. The self-play subsystem already has reward computation.
- **Implementation**: New generate_icrl() method that does multi-round prompting with reward feedback.

#### 10. Self-Correction / Rollback for Agent Loops (SPOC, GA-Rollback, MIRROR)
- **What**: Structured reflection and rollback for agent tool-use errors.
- **Papers**: SPOC (arXiv 2506.06923, spontaneous self-correction in single pass), GA-Rollback (EMNLP 2025, generator+assistant rollback), MIRROR (arXiv 2505.20670, intra+inter reflection)
- **Why for ForgeAI**: ForgeEngine has generate_with_tools but no built-in error recovery. The agent harness would benefit from rollback capability.
- **Implementation**: Enhance generate_with_tools with reflection/rollback phases.

#### 11. Mamba-3 / Hybrid Attention Tree Speculation (Bole)
- **What**: Mamba-3 improves SSM with complex-valued states and MIMO. Bole enables tree speculation for hybrid-attention models (3.4-7.7x on linear-attention verification).
- **Papers**: Mamba-3 (arXiv 2603.15569), Bole (arXiv 2608.01651, 4.72x decode throughput)
- **Why for ForgeAI**: ForgeEngine has mamba_key.py and hybrid architecture support. Bole's tree speculation for hybrid models is novel — no existing system does this well.
- **Implementation**: New Mamba-3 key. Extend decoding strategies for hybrid-attention tree speculation.

#### 12. OOMB Training System (chunk-recurrent, O(1) activation memory)
- **What**: Training with constant activation memory via chunk-recurrent forward + on-the-fly recomputation.
- **Paper**: OOMB (arXiv 2602.02108, 4M-token context on single H200)
- **Why for ForgeAI**: Directly addresses the 12GB VRAM training constraint. 10MB overhead per 10K context tokens.
- **Implementation**: Enhance sft_train.py with chunk-recurrent training mode.

### TIER 4: Novel tests to add

#### T1. KV Cache Compression Instruction-Following Test
- **Paper**: "Pitfalls of KV Cache Compression" (ACL 2026)
- **What**: Test that KV compression doesn't break multi-instruction following / cause system prompt leakage.
- **Implementation**: Test that generates with compressed KV cache and verifies all instructions in a multi-instruction prompt are followed.

#### T2. Bit-exact Forward Pass Tests for Evolutionary Merges
- **What**: Per AGENTS.md preset lineage check, verify that crossover/mutation produces models with expected forward-pass behavior.
- **Implementation**: Test that loads parent + offspring, runs forward pass on same input, compares logits.

#### T3. Generation-Focused Evaluation (not perplexity)
- **Paper**: GenDistill (arXiv 2603.26556, perplexity underestimates gap by 20.8pp)
- **What**: Tests should evaluate generation quality, not just perplexity.
- **Implementation**: Add generation-based evaluation to the test suite for distilled/merged models.

#### T4. InferenceBench-style Agent Self-Optimization Test
- **Paper**: InferenceBench (arXiv, agent benchmark for inference optimization)
- **What**: Test that the agent harness can optimize its own inference configuration (quantization, KV cache, decoding strategy).
- **Implementation**: Test that gives the agent a model + benchmark objective and verifies it explores the config space.

#### T5. FA4 SM120 Correctness Test
- **What**: Verify that fa4_attention.py produces correct results on RTX 5070 (SM120).
- **Paper**: blackwell-geforce-nvfp4-gemm (SM120 uses SM80-era mma.sync, not SM100's tcgen05)
- **Implementation**: Bit-exact comparison of FA4 output vs reference attention on SM120.

---

## Round 2 Research Sweep — 2026-09-03 (24 additional web searches)

### TIER 1 additions (high value, fits existing architecture)

#### R2-1. Dynamic Sparse Attention via Evolving Token Importance (EvoSparse)
- **Paper**: ACL 2026 long.530 — "Evolving Sparsity" (iLearn-Lab/ACL26-EvoSparse)
- **What**: Model token importance as a *dynamic process* across decoding steps + layers. Two mechanisms: (1) Cross-Step Accumulation (decayed accumulation of sparse attention scores, query-agnostic, no recompute); (2) Cross-Layer Propagation (use Retrieval Heads to compute query-aware indices, propagate across layers).
- **Numbers**: Up to 5.36× attention latency speedup, 2.33× end-to-end decoding. Approaches full attention on PG-19/RULER/LongBench/math.
- **Why for ForgeAI**: ForgeEngine has 14 KV strategies but all use *static* importance. EvoSparse's cross-step accumulation is essentially free (decay-weighted running average of attention scores already computed). Cross-layer propagation via Retrieval Heads integrates with the DuoAttention-style head classification. Fits the existing snapkv/s4r infrastructure — add an "evolving" mode.
- **Implementation**: New `EvolvingSparseKVCache(KVCacheStrategy)` in `kv/sparse_kv.py` (new file). Maintains per-token importance vector that decays across steps. Verify-phase attention scores update the vector.

#### R2-2. ProxyAttn — Training-Free Sparse Attention via Representative Heads
- **Paper**: ICLR 2026 (wyxstriker/ProxyAttn), arXiv 2509.24745
- **What**: Compress attention head dimension to estimate block importance via "pooled representative heads". Block-aware dynamic budget per head. 10.3× attention acceleration, 2.4× prefill acceleration, no quality loss.
- **Why for ForgeAI**: ForgeEngine has 9 attention variants but none use head-pooling for importance estimation. ProxyAttn is training-free and works with MHA/GQA. The "representative heads" insight (heads are similar) means we can compute importance on 1/N heads and broadcast — direct VRAM savings on 12GB.
- **Implementation**: New `ProxyAttnWrapper` in `attention/proxy_attn.py` (new file). Pools K attention heads into 1 representative, computes block scores, broadcasts dynamic budget.

#### R2-3. RRAttention — Per-Head Round-Robin Block Sparse
- **Paper**: ACL 2026 long.1199
- **What**: Rotates query sampling positions across heads within each stride → maintains query independence + global pattern discovery. O(L²/S²) complexity. 99% of full attention recovered at 50% blocks, 2.4× speedup at 128K.
- **Why for ForgeAI**: ForgeEngine has no round-robin-style sparse attention. RRAttention is training-free, query-independent (no preprocessing), and works at long context. The stride-based aggregation composes with chunked prefill.
- **Implementation**: New `RRAttentionWrapper` in `attention/rr_attention.py` (new file).

#### R2-4. AB-Sparse — Adaptive Block Size per Head
- **Paper**: arXiv 2605.12110
- **What**: Different attention heads need different block granularities. Adaptive per-head block size + lossless block centroid quantization. Custom CUDA kernels.
- **Why for ForgeAI**: All ForgeEngine KV strategies use uniform block sizes. AB-Sparse's insight (heads have heterogeneous block sensitivity) is novel and fits the existing paged/snapkv infrastructure. Block centroid quantization is *lossless* and saves memory — critical for 12GB.
- **Implementation**: Extend `paged_eviction.py` with per-head adaptive block sizing.

#### R2-5. DirectKV — Zero-Copy KV Offloading (NVLink-C2C)
- **Paper**: OSDI 2026 (Luo et al.)
- **What**: GPU kernels directly access CPU-resident KV cache via NVLink-C2C. No GPU staging buffer. Fused KV generation + attention in one CUDA kernel. Warp-level pipelining overlaps fetch/compute/writeback. 50% transfer reduction, 43% GPU memory reduction, 1.2× end-to-end.
- **Why for ForgeAI**: ForgeEngine has `cpu_kv_offload.py` but uses staging buffers. On RTX 5070 (PCIe, no NVLink-C2C), the *fused kernel + warp pipelining* ideas still apply — they hide PCIe latency. The "no staging buffer" insight is directly applicable.
- **Implementation**: Upgrade `cpu_kv_offload.py` with fused attention+fetch kernel and warp-level pipelining. Even without NVLink-C2C, the kernel fusion helps on PCIe.

#### R2-6. HybridGen — CPU-GPU Parallel Attention
- **Paper**: arXiv 2604.18529
- **What**: Parallelize attention across CPU and GPU. GPU processes recent tokens, CPU handles older offloaded tokens. Decouples attention logits from the pipeline so CPU can proactively compute next-layer attention using current-layer input (exploiting inter-layer similarity). Feedback scheduler balances load. Semantic-aware KV mapping for CXL/NUMA.
- **Why for ForgeAI**: ForgeEngine's `hybrid_offload.py` puts conv on CPU, attention on GPU. HybridGen is the *opposite* and complementary — attention itself is split. On 12GB VRAM + 32GB RAM, splitting attention across CPU+GPU doubles effective KV capacity. The "next-layer proactive attention" trick is novel and fits the existing layer-pipelining infrastructure.
- **Implementation**: New `HybridAttentionSplitter` in `kv/hybrid_attention.py` (new file). Integrates with `cpu_kv_offload.py`.

#### R2-7. NEO — Asymmetric GPU-CPU Pipelining for Online Inference
- **Paper**: MLSys 2025 (NEO-MLSys25/NEO)
- **What**: Offload attention compute + KV states to CPU to increase GPU batch size. Asymmetric pipelining + load-aware scheduling. Up to 7.5× throughput on T4, 79.3% on A10G with better CPUs.
- **Why for ForgeAI**: Directly relevant to 12GB VRAM. NEO's insight: GPU compute is wasted because batch size is memory-limited. Offloading *attention compute* (not just KV) to CPU increases batch size. The T4 numbers (similar tier to RTX 5070) are compelling.
- **Implementation**: New `NEOOffloader` class in `kv/neo_offload.py` (new file). Asymmetric pipeline scheduler.

### TIER 2 additions

#### R2-8. TriLens — Per-Layer Logit-Lens Entropy for Hallucination Detection
- **Paper**: arXiv 2606.01033 (TriLens)
- **What**: At every layer, read multi-head self-attention output + FFN output + residual stream through the model's logit lens, record only the entropy. 3L-dimensional trajectory. Single forward pass, no multiple samples. Strong detector across instruction-tuned LLMs.
- **Why for ForgeAI**: ForgeEngine has no hallucination detection. TriLens is *single-pass* (no extra generation) and stores only 3L floats per token — negligible memory. Fits the existing `generate_stream` infrastructure as a per-token signal.
- **Implementation**: New `HallucinationDetector` in `inference/safety/trilens.py` (new file). Hooks into forward pass to record per-layer entropy. Exposed via `generate_stream` metadata.

#### R2-9. PoP — Prediction-of-Prediction Inter-Layer Activation Fusion
- **Paper**: arXiv 2608.27165
- **What**: Capture layer-transition uncertainty by fusing intermediate hidden representations across depth during a single forward pass. 75.5% AUROC on TruthfulQA. <1.2% runtime latency, zero extra generation passes.
- **Why for ForgeAI**: Even cheaper than TriLens (no logit lens, just hidden state fusion). Single forward pass. Could be combined with TriLens for ensemble.
- **Implementation**: Add to `safety/trilens.py` as a complementary signal.

#### R2-10. HSRM — Hidden-State Reward Models for Test-Time Verification
- **Paper**: arXiv 2608.30841 (JXL884/HSRM)
- **What**: Lightweight (~2M param) Transformer encoder that reads generator's internal hidden states at reasoning-step boundaries to rank candidate solutions. No text re-processing. Trained from self-generated trajectories with outcome labels.
- **Why for ForgeAI**: ForgeEngine has `generate_with_tools` (agent loops) and the self-play subsystem. HSRM enables verification of reasoning steps *without* a separate verifier LLM — critical for 12GB VRAM (can't fit 2 models). The 2M params fit in <10MB.
- **Implementation**: New `ReasoningVerifier` in `inference/safety/hsrm.py` (new file). Hooks into `generate_with_tools` to score reasoning steps.

#### R2-11. CLUE — Training-Free Hidden-State Clustering Verifier
- **Paper**: ACL 2026 long.788
- **What**: Summarize each reasoning trace by activation delta (hidden state start - end of reasoning span). Predict correctness by comparing to two class centroids from labeled experience. Training-free, non-parametric.
- **Why for ForgeAI**: Even simpler than HSRM — no training, just centroid comparison. Could be the "fast path" verifier with HSRM as the "accurate path".
- **Implementation**: Add to `safety/hsrm.py` as `CLUEVerifier` class.

#### R2-12. River-LLM — Seamless Early Exit via KV Share
- **Paper**: ACL 2026 long.1746
- **What**: Training-free token-level early exit. Lightweight "KV-Shared Exit River" generates missing KV cache for skipped layers naturally during exit. State transition similarity predicts cumulative KV errors → guides exit decisions. 1.71-2.16× speedup on math/code.
- **Why for ForgeAI**: ForgeEngine has no early-exit. River-LLM solves the "KV cache absence" problem that killed decoder early-exit. Training-free. The "KV-Shared Exit River" is a small module that generates KV for skipped layers — fits the existing layer-skip infrastructure in `faser.py`.
- **Implementation**: New `RiverEarlyExit` class in `attention/river_exit.py` (new file). Integrates with `FASERController`.

#### R2-13. TIDE — Token-Informed Depth Execution
- **Paper**: arXiv 2603.21365 (NeurIPS)
- **What**: Post-training system with tiny learned routers at checkpoint layers. Selects earliest converged layer per token. Works with any HuggingFace causal LM. Fused CUDA kernels for fp32/fp16/bf16. 98-99% early exit rate during decoding. 4MB router checkpoint, 3-min calibration on 2000 samples.
- **Why for ForgeAI**: ForgeEngine has `FASEREarlyExit` but TIDE is more practical — auto-detects GPU, works with any HF model, tiny router. The 4MB router fits trivially in 12GB. Calibration is fast.
- **Implementation**: New `TIDERouter` in `attention/tide.py` (new file). Calibration utility.

#### R2-14. BLADE — Boundary-Expanded Layer-Adaptive Dynamic Exit
- **Paper**: arXiv 2607.28966
- **What**: Multi-granular checkpoints (sentence, self-doubt, paragraph boundaries). Learns compact subset of informative probe layers. Calibrated predictions + checkpoint-specific confirmation rules. 24.8% token reduction on Qwen3-8B, 15.8% on Qwen3-4B.
- **Why for ForgeAI**: ForgeEngine uses Qwen3-family models. BLADE is specifically validated on Qwen3. The "self-doubt boundary" detection is novel and fits reasoning models.
- **Implementation**: Extend `tide.py` with boundary-expanded checkpoints.

#### R2-15. BUDDY — Budget-Driven Dynamic Depth Routing
- **Paper**: arXiv 2606.09514
- **What**: Lightweight Decision Module scores layers, executes top-k to satisfy budget. Reuses first-layer KV cache as global context source. Optional Budget Predictor for input-dependent compute level. Supports *multiple budgets* in one trained model.
- **Why for ForgeAI**: The "multiple budgets in one model" feature is unique — ForgeEngine could expose `generate(budget="low"|"medium"|"high")` without loading different models. Critical for 12GB (can't hold multiple model variants).
- **Implementation**: New `BuddyRouter` in `attention/buddy.py` (new file).

#### R2-16. L2A — End-to-End Dynamic Sparsity (Resource-Adaptive)
- **Paper**: arXiv 2606.27743
- **What**: Budget-conditioned + input-aware gating networks. Jointly optimizes layer skipping + head pruning + reasoning-token reduction. Single model traces entire Pareto frontier. 34% layer sparsity, within 0.6% of dense on GSM8K.
- **Why for ForgeAI**: The "single model, entire Pareto frontier" is exactly what 12GB VRAM needs. No need to load different models for different quality/speed tradeoffs. The three-axis optimization (layers + heads + tokens) is more comprehensive than ForgeEngine's current single-axis approaches.
- **Implementation**: New `L2AGate` in `attention/l2a.py` (new file). Integrates with all three: layer skip, head prune, token budget.

#### R2-17. POLAR — Program-of-Layers (Skip OR Loop)
- **Paper**: arXiv 2606.06574
- **What**: Pretrained layers can be *skipped or looped* to form customized programs per input. MCTS reveals better programs almost always exist. Lightweight POLAR prediction network generates execution programs.
- **Why for ForgeAI**: ForgeEngine has layer-skip (FASER) but no layer-*loop*. POLAR shows looping can *correct* incorrect predictions — novel for reasoning. The "program-of-layers" abstraction is a clean extension of the existing layer-skip infrastructure.
- **Implementation**: Extend `faser.py` with loop capability. New `POLARRouter` class.

#### R2-18. WIDE — Token-Level Dynamic Width Pruning
- **Paper**: arXiv 2607.28418
- **What**: First end-to-end differentiable token-level dynamic width pruning. Each token dynamically selects attention-head groups + FFN-channel groups. Pruning-kernel co-design. 50% sparsity → 55.1% over SOTA dynamic depth. 1.98× prefill, 4.95× decode kernel speedup.
- **Why for ForgeAI**: ForgeEngine prunes at layer level (FASER) and head level (MoSA). WIDE adds *FFN-channel* pruning — a new axis. The 4.95× decode speedup is huge for 12GB VRAM.
- **Implementation**: New `WIDEPruner` in `attention/wide.py` (new file). Token-level head + channel selection.

#### R2-19. DART — Dynamic Attention-Guided Runtime Tracing for FFN Pruning
- **Paper**: arXiv 2601.22632
- **What**: Training-free on-the-fly context-based FFN pruning. Monitors attention score distribution shifts → dynamically updates neuron masks. <10MB memory, 0.1% FLOPs overhead. 14.5% accuracy gain over prior dynamic at 70% FFN sparsity.
- **Why for ForgeAI**: ForgeEngine has no FFN pruning. DART is training-free, <10MB, and uses attention scores *already computed*. The "knowledge neurons shift during generation" insight is novel.
- **Implementation**: New `DARTPruner` in `quant/dart_prune.py` (new file). Hooks into attention to update FFN masks.

#### R2-20. Prox — Training-Free FFN Activation Sparsity via Approximate Salience
- **Paper**: arXiv 2607.27591
- **What**: Two-stage training-free sparse SwiGLU FFNs. Stage 1: input sparsity + quantized proxy weights → construct mask. Stage 2: compute selected channels exactly. 1.99× decode speedup at 70% FFN sparsity. Compatible with quantization + sparse attention.
- **Why for ForgeAI**: ForgeEngine uses SwiGLU models (Qwen3, Llama). Prox is training-free and *composes* with existing quantization + sparse attention — multiplicative gains.
- **Implementation**: New `ProxFFN` in `quant/prox_ffn.py` (new file).

#### R2-21. SVD Contextual Sparsity Predictors
- **Paper**: arXiv 2603.14110
- **What**: Truncation-aware SVD of gate projection matrix → fast training-free sparse pattern predictor. Theoretical guarantees on prediction accuracy. 1.8× decode speedup, <1% degradation on math/code.
- **Why for ForgeAI**: ForgeEngine has SVD-based compression in `merge_models.py`. SVD contextual sparsity reuses the same math for *inference-time* prediction — clean synergy.
- **Implementation**: Extend `quant/prox_ffn.py` with SVD-based predictor variant.

### TIER 3 additions

#### R2-22. MVR-cache — Multi-Vector Retrieval Semantic Cache
- **Paper**: arXiv 2605.24914 (PKU-SDS-lab/MVR-Cache)
- **What**: Learnable prompt segmentation + ColBERT-style MaxSim multi-vector retrieval for semantic caching. RL-trained segmentation model. 37% higher cache hit rates, same correctness guarantees.
- **Why for ForgeAI**: ForgeEngine has `learned_prefix_cache.py` but it's exact-match. MVR-cache adds *semantic* matching — prompts that are semantically equivalent but syntactically different hit the cache. Huge for agent workflows where prompts vary slightly.
- **Implementation**: Extend `learned_prefix_cache.py` with MVR mode.

#### R2-23. CanonCache — Semantic KV-Cache Canonicalization
- **Paper**: Zenodo 20299052
- **What**: Deterministic semantic canonicalization rewrites semantically equivalent prompts to canonical form before inference → increases exact-prefix overlap. 5%→37.5% cache hit rate, 47.5% token reduction, 7.5× SCAR.
- **Why for ForgeAI**: Simpler than MVR-cache — just canonicalize prompts. Could be a pre-processing step before any KV cache strategy. Fits the existing `prefix_cache.py`.
- **Implementation**: New `PromptCanonicalizer` in `inference/cache/canon.py` (new file).

#### R2-24. LazyAttention — Deferred Positional Encoding for Zero-Copy KV Reuse
- **Paper**: arXiv 2606.04302
- **What**: Kernelize deferred positional encoding → position-agnostic KV reuse. Single physical KV copy serves multiple logical requests at arbitrary positions. 1.37× TTFT, 1.40× throughput.
- **Why for ForgeAI**: ForgeEngine's prefix cache requires *exact prefix match*. LazyAttention enables reuse at *arbitrary positions* — much more flexible. Critical for RAG where documents appear at different positions.
- **Implementation**: New `LazyPosEncoder` in `attention/lazy_pos.py` (new file). Modifies attention kernel to apply RoPE on-the-fly.

#### R2-25. ProphetKV — User-Query-Driven Selective Recomputation for RAG
- **Paper**: arXiv 2602.02579
- **What**: For RAG, dynamically prioritize tokens based on semantic relevance to user query. Dual-stage recomputation pipeline fuses layer-wise attention metrics. 96-101% of full-prefill accuracy at 20% recomputation.
- **Why for ForgeAI**: ForgeEngine has `cacheblend.py` for KV reuse. ProphetKV's query-driven selection is a direct upgrade — instead of generic importance, use *user query relevance*.
- **Implementation**: Extend `cacheblend.py` with query-driven selection mode.

#### R2-26. Native Retrieval Embeddings from LLM Hidden States
- **Paper**: arXiv 2603.08429
- **What**: Lightweight projection head maps LLM hidden states → embedding space. Eliminates separate embedding model. 97% of baseline retrieval quality. Trained with alignment + contrastive + rank distillation losses.
- **Why for ForgeAI**: ForgeEngine's agent harness generates search queries then needs a separate embedding model. Native retrieval = no second model = saves VRAM. The projection head is tiny.
- **Implementation**: New `NativeRetriever` in `inference/retrieval/native.py` (new file). Projection head trained via self-play.

#### R2-27. MrRoPE — Mixed-radix Rotary Position Embedding
- **Paper**: arXiv 2601.22181
- **What**: Generalized RoPE-extension via radix system conversion. Unifies YaRN, Self-Extend, DCA as special cases. MrRoPE-Pro: training-free, 85% recall at 128K, 2× YaRN accuracy on InfiniteBench.
- **Why for ForgeAI**: ForgeEngine has `LeRoPE` and `RoPEID` in `position/__init__.py`. MrRoPE is a *generalization* — could subsume existing methods. Training-free long-context extension is critical for 12GB (can't afford long-context fine-tuning).
- **Implementation**: New `MrRoPE` class in `position/mr_rope.py` (new file).

#### R2-28. Jet-Long — Dynamic Bifocal RoPE
- **Paper**: arXiv 2607.07740
- **What**: Tuning-free zero-shot. Local RoPE-faithful window + long-range window with *dynamically adapting* rescaling factor. Recovers base model exactly at short inputs. 1.39× FA2 throughput on H100. +4.79pp RULER over baseline at 128K on Qwen3-1.7B.
- **Why for ForgeAI**: ForgeEngine uses Qwen3 models. Jet-Long is specifically validated on Qwen3-1.7B/4B/8B. The dynamic rescaling (vs fixed YaRN factor) is the novel twist. Fused into single CuTe kernel — but we can do a Triton version for SM120.
- **Implementation**: Already partially exists as `jet_long.py` in scheduler — verify it's the bifocal RoPE version, upgrade if needed.

#### R2-29. LaMPE — Length-aware Multi-grained Positional Encoding
- **Paper**: ACL 2026 findings.1608
- **What**: Training-free. Dynamic relationship between mapping length and input length via parametric scaled sigmoid. Multi-grained attention allocates positional resolution across regions. Significant improvements over YaRN/Self-Extend on 5 benchmarks.
- **Why for ForgeAI**: Already partially exists as `lampe.py` in scheduler — verify implementation matches paper. The "multi-grained" attention (fine locality + long-range) is novel.
- **Implementation**: Verify `scheduler/lampe.py` is the positional encoding version (not just scheduler).

#### R2-30. EAGLE 3.1 + SpecForge + P-EAGLE + ARC-Decode (Speculative Decoding Suite)
- **Papers**: EAGLE 3.1 (vLLM blog 2026-05-26), SpecForge (arXiv 2603.18567), P-EAGLE (arXiv 2602.01469), ARC-Decode (ICML 2026, NeuraLiying/ARC-Decode)
- **What**:
  - EAGLE 3.1: Config-driven extension in vLLM, backward-compatible with EAGLE-3 checkpoints. 2.03× per-user throughput on Kimi K2.6.
  - SpecForge: Open-source training framework for EAGLE-3 draft models. 9.9× faster training. SpecBundle draft models achieve 4.48× end-to-end speedup.
  - P-EAGLE: Parallel drafting (multiple tokens per forward pass) via learnable shared hidden state. 1.10-1.36× over autoregressive EAGLE-3.
  - ARC-Decode: Training-free, plug-and-play on EAGLE-3. Risk-bounded acceptance for *sampling* (EAGLE-3 degrades under sampling). Entropy-guided pruning + local shift estimation + risk-bounded acceptance.
- **Why for ForgeAI**: ForgeEngine has `Eagle3Decoding` but not the 3.1 config-driven version, parallel drafting, or risk-bounded acceptance. ARC-Decode is critical — ForgeEngine uses temperature/top_p sampling, where EAGLE-3 degrades. ARC-Decode fixes this for free.
- **Implementation**: Upgrade `decoding.py` Eagle3Decoding → EAGLE 3.1. Add `PEagleDecoding` (parallel). Add `ARCDecodeWrapper` for sampling mode.

#### R2-31. ST-MoE / SpecPrefetch — Expert Prefetching for MoE
- **Papers**: ST-MoE (arXiv 2606.15453), SpecPrefetch (arXiv 2607.24787), "Speculating Experts" (arXiv 2603.19289)
- **What**: Predict next-layer experts from current internal representations. Proactively stage experts before use. Overlap loading with computation. ST-MoE: spatio-temporal prediction across adjacent layers + consecutive tokens. SpecPrefetch: lightweight adapters, router-preserving, window-aware budget. "Speculating Experts": 14% TPOT reduction.
- **Why for ForgeAI**: ForgeEngine has `moe_optim.py` with `ElbowRouter`/`AllocMoE`/`LDACalibrator` but no *expert prefetching*. On 12GB VRAM, MoE experts must be offloaded to CPU — prefetching hides the transfer latency. The prediction-from-internal-representations insight is training-free.
- **Implementation**: New `ExpertPrefetcher` in `moe/prefetch.py` (new file). Integrates with `moe_optim.py`.

#### R2-32. DSE / L³ — Conditional Memory via Sparse Embedding Lookup
- **Papers**: DSE (ACL 2026 long.226), L³ (arXiv 2601.21461)
- **What**: New axis of sparsity — massive embedding table indexed by local N-grams for O(1) retrieval. Offloads static recall to memory, freeing attention for reasoning. DSE: deterministic hashing → offload to host memory with negligible overhead. L³: static token-based routing, CPU-offloaded inference with no overhead.
- **Why for ForgeAI**: ForgeEngine has MoE but no *lookup-based* sparsity. DSE/L³ are orthogonal to MoE — can combine. The "CPU-offloaded with no overhead" property is perfect for 12GB VRAM. Static knowledge → CPU embedding table, dynamic reasoning → GPU.
- **Implementation**: New `ConditionalMemory` in `moe/conditional_memory.py` (new file). N-gram indexed embedding table on CPU.

#### R2-33. TokTier / Incremental BPE / GPUTOK / LoPT / BlockBPE — Tokenization Acceleration
- **Papers**: TokTier (arXiv 2607.29678), Incremental BPE (arXiv 2605.30813), GPUTOK (arXiv 2603.02597), LoPT (ACL 2026 long.1529), BlockBPE (arXiv 2507.11941)
- **What**: GPU-accelerated, incremental, parallel BPE tokenization. TokTier: stateful CPU+GPU, exact reference matching. Incremental BPE: O(n log²t), 3× over HF tokenizers. GPUTOK: 1.7× over tiktoken, 7.6× over HF. LoPT: lossless parallel, character-position matching. BlockBPE: O(nd), 2× over tiktoken.
- **Why for ForgeAI**: ForgeEngine's `_tokenize` is CPU-bound. For long-context (1M tokens), tokenization becomes a bottleneck. GPU tokenization overlaps with model loading.
- **Implementation**: New `FastTokenizer` wrapper in `inference/tokenize.py` (new file). GPU BPE for long contexts.

#### R2-34. Vector-Index Output Embeddings (HNSW Logits)
- **Paper**: arXiv 2608.27460
- **What**: Replace dense vocabulary projection with HNSW-based vector index. Retrieve top-k tokens via MIPS. 82% decoding throughput improvement for Gemma 3 270M. Drop-in replacement.
- **Why for ForgeAI**: ForgeEngine's `_sample_next_token` does dense vocabulary projection. For models with 100k+ vocab (Qwen3), this is a bottleneck. HNSW retrieval is faster for batch-size-1 (the 12GB regime).
- **Implementation**: New `HNSWLogitsHead` in `inference/output_head.py` (new file).

#### R2-35. VQ-Logits — Vector Quantized Output Layer
- **Paper**: arXiv 2505.10202
- **What**: Replace V×d output matrix with K-vector codebook (K≪V). Predict logits over codebook, scatter to full vocab. 99% parameter reduction in output layer, 6× logit speedup, 4% perplexity increase.
- **Why for ForgeAI**: Complementary to R2-34. VQ-Logits *compresses* the output matrix (saves VRAM), HNSW *accelerates* it. Could combine: VQ-Logits for compression + HNSW over codebook for speed.
- **Implementation**: Add `VQLogitsHead` to `output_head.py`.

#### R2-36. TurboQuant-H — Hadamard Rotation 2-Bit Embedding Quantization
- **Paper**: Cactus blog (TurboQuant ICLR 2026 variant)
- **What**: Hadamard rotation + per-group Lloyd-Max codebooks for 4× embedding compression. 2.125 effective bits/dim. 40% total model storage reduction, +0.06 perplexity.
- **Why for ForgeAI**: For models with per-layer embeddings (Gemma E-series), embeddings dominate. Even for Qwen3, the embedding table is large. 4× compression with negligible quality loss is huge for 12GB.
- **Implementation**: New `TurboQuantEmbedding` in `quant/turboquant_emb.py` (new file).

#### R2-37. FlashSVD v1.5 — Low-Rank Transformer Inference Runtime
- **Paper**: arXiv 2605.08314
- **What**: Unified runtime for SVD-compressed transformers. Phase-specific kernels, dense-KV decode, packed MLP execution, per-layer CUDA-graph replay. 2.55× decode, 2.39× end-to-end speedup.
- **Why for ForgeAI**: ForgeEngine has `merge_models.py` with SVD-based model merging. FlashSVD shows how to *serve* SVD-compressed models efficiently. The "phase-specific kernels" (prefill vs decode) insight is novel.
- **Implementation**: New `SVDRuntime` in `quant/svd_runtime.py` (new file). Integrates with merge_models.py.

#### R2-38. Swift-SVD / AA-SVD / ERC-SVD — Training-Free SVD Compression
- **Papers**: Swift-SVD (arXiv 2604.01609), AA-SVD (arXiv 2604.02119), ERC-SVD (PMLR 328)
- **What**: Training-free, closed-form SVD compression. Swift-SVD: activation-aware, single eigenvalue decomposition, 3-70× faster than prior. AA-SVD: anchors to original outputs while modeling input shifts. ERC-SVD: uses residual matrix to reduce truncation loss, selectively compresses last layers.
- **Why for ForgeAI**: ForgeEngine's `merge_models.py` does SVD but not these advanced variants. Swift-SVD's closed-form solution is O(n) — could compress models in seconds, not hours.
- **Implementation**: Add Swift-SVD/AA-SVD/ERC-SVD as methods in `merge_models.py`.

#### R2-39. QUASAR / LoRDS / LAQuant — QAT Improvements
- **Papers**: QUASAR (arXiv 2608.13966), LoRDS (arXiv 2601.22716), LAQuant (arXiv 2605.08755)
- **What**: QUASAR: continuous loss-aware reconstruction in QAT loop, 29% KL reduction at 2-bit. LoRDS: unified PTQ+QAT+PEFT via low-rank decomposed scaling, zero inference overhead, 9.6% over QLoRA. LAQuant: layer-wise lookahead loss for reasoning models, 15.11pp AIME improvement, 3.42× decode speedup.
- **Why for ForgeAI**: ForgeEngine has quantization but ForgeAI's `sft_train.py` could benefit from QAT. LAQuant is specifically for *reasoning models* (long decoding) — directly relevant. LoRDS's "zero inference overhead" PEFT is critical for 12GB.
- **Implementation**: Add QAT modes to `sft_train.py`. New `LoRDSAdapter` in `training/`.

#### R2-40. Budget Guidance / SelfBudgeter / Elastic Reasoning / ACTS / TAB — Reasoning Length Control
- **Papers**: Budget Guidance (ACL 2026 findings.1866), SelfBudgeter (ACL 2026 findings.1063), Elastic Reasoning (ICLR 2026), ACTS (arXiv 2606.03965), TAB (arXiv 2604.05164)
- **What**: Inference-time reasoning length control. Budget Guidance: Gamma distribution predictor, soft token-level guidance, 26% accuracy gain under tight budgets. SelfBudgeter: self-estimate budget, budget-guided GRPO, 61% length compression. Elastic Reasoning: separate thinking/solution phases with independent budgets. ACTS: MDP controller agent steers frozen reasoner. TAB: turn-adaptive budgets for multi-turn.
- **Why for ForgeAI**: ForgeEngine has `generate_adaptive` but no reasoning-length control. These are all *inference-time* (no retraining for Budget Guidance, ACTS). Critical for reasoning models that over-think. The self-play subsystem could train the budget predictors.
- **Implementation**: New `ReasoningBudgetController` in `inference/reasoning/budget.py` (new file). Integrates with `generate_adaptive` and `generate_with_tools`.

#### R2-41. Predictive Scheduling for Reasoning Compute Allocation
- **Paper**: arXiv 2602.01237
- **What**: Pre-run lightweight predictors (MLP on hidden states or LoRA classifier on question text) to estimate optimal reasoning length before generation. Greedy batch allocator distributes fixed budget. 7.9pp accuracy gain over uniform budgeting on GSM8K. Middle layers (12-17) carry richest signal.
- **Why for ForgeAI**: ForgeEngine has `generate_batch` — predictive scheduling can allocate different token budgets per request in the batch. The "middle layers carry richest signal" insight tells us *where* to hook the predictor.
- **Implementation**: Add `PredictiveScheduler` to `reasoning/budget.py`.

#### R2-42. ATLAS-RTC — Token-Level Runtime Control for Constrained Generation
- **Paper**: arXiv 2603.27905
- **What**: Intercepts generation at logit distribution before each token. Runtime controller scores structural drift against formal output contract. Graduated interventions: logit biasing → temperature modulation → token masking → mid-step rollback with re-steering. Stateful, stage-aware contracts.
- **Why for ForgeAI**: ForgeEngine has `generate_with_tools` which needs JSON conformance. ATLAS-RTC is more sophisticated than simple grammar masking — it has *graduated* interventions and *mid-step rollback*. The "closed-loop" design fits agent workflows.
- **Implementation**: New `ATLASRuntimeController` in `inference/constrained/atlas.py` (new file). Integrates with `generate_with_tools`.

#### R2-43. SHIM — Lightweight Bias Correction for Constrained Decoding
- **Paper**: arXiv 2608.10137
- **What**: Rigid grammar masking distorts probability distribution. SHIM: offline-trained logit correction conditioned on parser/lexer state + candidate next tokens. Negligible overhead (parser state already computed). Leaves base LM weights untouched.
- **Why for ForgeAI**: ForgeEngine's constrained decoding (if implemented) would have the bias problem. SHIM fixes it cheaply. The "parser state already encodes future validity" insight is elegant.
- **Implementation**: Add `SHIMCorrector` to `constrained/atlas.py`.

#### R2-44. SWAI — Training-Free Logit-Space Steering
- **Paper**: arXiv 2601.10960
- **What**: Precompute z-normalized one-vs-rest log-odds scores from labeled corpora. Bias high-scoring tokens within top-K candidate set. Training-free, no internal activations, no auxiliary model. Controls readability, politeness, toxicity.
- **Why for ForgeAI**: ForgeEngine has no output style control. SWAI is a simple lookup table — negligible memory. Could enable `generate(style="polite"|"technical"|"simple")`.
- **Implementation**: New `LogitSteerer` in `inference/steering/swai.py` (new file).

#### R2-45. ClusterFusion++ — Full Transformer-Block Fusion
- **Paper**: arXiv 2604.23553
- **What**: Cluster-level fusion of entire decoder block: LayerNorm→QKV→RoPE→attention→output projection→Post-LN→MLP→residual. CUDA-Graph-compatible with persistent TMA descriptors. 1.34× throughput on RTX 5090-class.
- **Why for ForgeAI**: ForgeEngine has `block_fusion.py` with `BlockFusionRunner`/`CompiledBlockFusion`. ClusterFusion++ is the next step — fuse the *entire* block, not just attention-side ops. RTX 5090-class = SM120 family = our hardware.
- **Implementation**: Upgrade `block_fusion.py` with full-block fusion mode.

#### R2-46. Fused QK Norm + RoPE + Cache + Quant (Single Kernel)
- **Paper**: vLLM PR #38621
- **What**: Single CUDA kernel: QK RMSNorm + RoPE + KV cache write + optional FP8 quant. Warp-per-head design. Vectorized loads, warp-shuffle RMSNorm. V written first (fire-and-forget) while K normalizes — hides store latency.
- **Why for ForgeAI**: ForgeEngine has `fused_qk_norm_rope_cache.py` — verify it includes the quant fusion. If not, upgrade. The "V first, K second" latency hiding is a novel trick.
- **Implementation**: Verify/upgrade `attention/fused_qk_norm_rope_cache.py`.

#### R2-47. PackInfer — Heterogeneous Batched Attention
- **Paper**: arXiv 2602.06072
- **What**: Pack multiple requests into unified kernel launches. Compute- and I/O-aware execution for heterogeneous batched inference. Load-balanced execution groups. I/O-aware grouping co-locates shared-prefix requests. 13-20.1% latency reduction, 20% throughput over FlashAttention.
- **Why for ForgeAI**: ForgeEngine has `generate_batch` but no heterogeneous batching. PackInfer's "pack into unified kernel" is critical for 12GB — can't afford separate kernel launches per request.
- **Implementation**: New `PackedBatchAttention` in `attention/pack_infer.py` (new file).

#### R2-48. BatchGen — Sequence Coroutine Compute Model
- **Paper**: OSDI 2026 (arXiv 2606.21712)
- **What**: Each sequence = fine-grained event-driven coroutine. Runtime can reorganize work dynamically: larger expert-level batches, mitigate stragglers, reallocate across devices. 2.3× batch completion reduction on 128-GPU cluster, 9.6× over offloading baseline on memory-constrained GPUs.
- **Why for ForgeAI**: The "9.6× on memory-constrained GPUs" is directly relevant to 12GB. The coroutine model enables mid-layer pausing — novel for ForgeEngine.
- **Implementation**: New `CoroutineScheduler` in `scheduler/coroutine.py` (new file).

#### R2-49. FASER — Fine-Grained Speculative Decoding Phase Management
- **Paper**: arXiv 2604.20503
- **What**: Dynamically adjust speculative length per request within continuous batch. Early pruning of rejected tokens in verification. Break verification into frontiers/chunks, overlap with draft phase via spatial multiplexing. 53% throughput, 1.92× latency.
- **Why for ForgeAI**: ForgeEngine has `FASERController`/`FASEREarlyExit`/`FASERFrontier` — verify these implement the full FASER paper. The "spatial multiplexing of draft+verify" is the key novel trick.
- **Implementation**: Verify/upgrade `attention/faser.py`.

#### R2-50. Continuous Depth Batching for Looped LMs
- **Paper**: arXiv 2608.09444
- **What**: Schedule at granularity of individual loop iterations. Boundary stages (embedding, LM head) and loop steps in separate priority queues. Exit decisions one step ahead. 99% of theoretical max speedup, 1.5-1.9× throughput.
- **Why for ForgeAI**: If ForgeEngine adopts POLAR (R2-17) layer looping, CDB enables efficient batching of different loop counts per token. Without CDB, looped models can't be batched efficiently.
- **Implementation**: Add CDB mode to `scheduler/coroutine.py`.

#### R2-51. Medha — Multi-Million Context Serving
- **Paper**: haoran-qiu.com/pdf/medha-preprint.pdf
- **What**: Adaptive prefill chunking with preemption. Sequence Pipeline Parallelism (SPP) for TTFT. KV-Cache Parallelism (KVP) for TPOT. Input-length aware least-remaining-slack scheduling. 10M token scaling, 30× median latency reduction, 5× throughput.
- **Why for ForgeAI**: ForgeEngine has `chunked_prefill.py` but not adaptive chunking or KV-cache parallelism. For 12GB VRAM, KVP (distributing decode across CPU+GPU) is critical for long context.
- **Implementation**: Upgrade `prefill/__init__.py` with adaptive chunking. New `KVPDecoder` in `kv/kvp.py` (new file).

#### R2-52. SparseServe — Hierarchical HBM-DRAM for Dynamic Sparse Attention
- **Paper**: arXiv 2509.24626
- **What**: Offload underutilized KV caches to DRAM for larger batch sizes. Fragmentation-aware KV transfer (FlashH2D/FlashD2H). Working-set-aware batch size control. Layer-segmented prefill. 9.26× TTFT, 3.14× throughput.
- **Why for ForgeAI**: ForgeEngine has `cpu_kv_offload.py` but no *hierarchical* management. SparseServe's "working-set-aware batch size control" is novel — dynamically adjust batch size based on real KV working set.
- **Implementation**: Upgrade `cpu_kv_offload.py` with hierarchical HBM-DRAM management.

#### R2-53. DualDecoder — Predictive KV Prefetch for Long Context
- **Paper**: arXiv 2607.26475
- **What**: Critical KV entries for next token can be predicted from preceding speculated token. Proactive prefetch overlapped with decoding. Layer-aware transfer schedule. Layer-scoped memory manager. 2.62× decoding throughput.
- **Why for ForgeAI**: ForgeEngine has `async_d2h.py` with `PinnedTokenBuffer`/`AsyncTokenReader`/`StreamedGenerator` — verify it implements predictive prefetch. DualDecoder's "predict from speculated token" is a novel signal source.
- **Implementation**: Verify/upgrade `scheduler/async_d2h.py`.

#### R2-54. KernelFlume — Elastic Core-Attention Scaling
- **Paper**: arXiv 2606.29207
- **What**: Disaggregate stable projection/FFN path from core-attention. Weight nodes execute dense kernels, weightless attention nodes store KV partitions. Routing table maps token ranges to attention endpoints. Inter-layer kernel pipelining. 32-61% cost reduction.
- **Why for ForgeAI**: ForgeEngine has no attention-compute disaggregation. KernelFlume's "attention nodes scale with request-state demand" is novel for 12GB — could put attention on CPU, projection/FFN on GPU.
- **Implementation**: New `KernelFlumeScheduler` in `scheduler/kernel_flume.py` (new file).

#### R2-55. VLCache — Vision Token KV Cache Reuse
- **Paper**: arXiv 2512.12977
- **What**: For VLMs, store vision encoder output + KV cache. When same image recurs, bypass vision encoder entirely. Layer-aware recomputation (2-5% of tokens). 1.2-16× TTFT speedup.
- **Why for ForgeAI**: ForgeEngine has no VLM support but the architecture (multi-modal keys) exists. VLCache's "compute 2%, reuse 98%" is critical for VLM inference on 12GB.
- **Implementation**: New `VLCache` in `kv/vl_cache.py` (new file). For future VLM support.

#### R2-56. HybridKV / MM-ShiftKV / Q-Cache — Multimodal KV Compression
- **Papers**: HybridKV (ACL 2026 long.594), MM-ShiftKV (ACL 2026 findings.1447), Q-Cache (AAAI v40i16)
- **What**: HybridKV: classify heads as static/dynamic, hybrid compression. 7.9× KV memory reduction, 1.52× decode. MM-ShiftKV: decode-aware prefill-stage KV selection (variance-expanded query proxies). Q-Cache: cross-layer attention sharing for VLMs, 35% KV reduction, 1.5× throughput.
- **Why for ForgeAI**: For future VLM support. Q-Cache's "cross-layer attention sharing" is a novel axis — reuse queries across adjacent layers.
- **Implementation**: Add multimodal modes to KV strategies.

#### R2-57. FairyFuse / Litespark — Multiplication-Free Ternary CPU Inference
- **Papers**: FairyFuse (arXiv 2604.20913), Litespark (arXiv 2605.06485)
- **What**: Ternary weights {−1,0,+1} → replace multiply with add/sub/no-op. FairyFuse: AVX-512 masked add/sub, 29.6× kernel speedup, 32.4 tok/s on Xeon. Litespark: SIMD integer dot product, 9.2× TTFT, 52× throughput, 14× memory reduction on Apple Silicon.
- **Why for ForgeAI**: ForgeEngine has BitNet ternary support. These show how to *actually exploit* ternary on CPU — currently ForgeEngine likely treats ternary as dense float. For CPU offload mode, this is a huge win.
- **Implementation**: New `TernaryCPUKernel` in `quant/ternary_cpu.py` (new file). AVX-512/NEON masked add/sub kernels.

#### R2-58. cpubrrr / Arm NEON INT4 — Hand-Optimized CPU MoE Kernels
- **Papers**: cpubrrr (GitHub arizqi/cpubrrr), Arm vLLM blog (2026-07-29)
- **What**: cpubrrr: gpt-oss:20b at ~110 tok/s on M4 Max CPU, 7.5× llama.cpp. Hand-written NEON/SME kernels, quad-interleaved weight layout, integer-accumulation Q8_K. Arm vLLM: INT8 W8A8 and W4A8 on Neoverse, chunked prefill + prefix caching.
- **Why for ForgeAI**: ForgeEngine's CPU offload could benefit from these layout/kernel tricks. The "quad-interleaved weight layout" for sequential stream reading is a novel layout insight.
- **Implementation**: Add NEON/AVX-512 kernel variants to `ternary_cpu.py` and `cpu_kv_offload.py`.

#### R2-59. TileLens — Tile-Major Memory Layout for Large-Granularity Memory
- **Paper**: arXiv 2607.04031
- **What**: Reshape contiguous memory blocks into 2D rectangles aligning with tile boundaries. Reduces read amplification on HBF/RoMe. Near-HBM performance on HBF-augmented GPUs.
- **Why for ForgeAI**: ForgeEngine's weight layouts are row-major. Tile-major is a novel layout for future HBF/flash-augmented hardware. Forward-looking.
- **Implementation**: Add tile-major layout option to weight packing in `novel_quant.py`.

#### R2-60. DOPS — Dynamic Operator Scheduling + Weight Layout Arbiter
- **Paper**: arXiv 2607.25498
- **What**: Stage-aware DAG + Bifocal scheduler (dynamic operator-to-device placement) + Weight Layout Arbiter (hardware-efficient weight layouts under memory constraints). 1.20-2.23× over PD baseline + 1.28-1.33× from WLA.
- **Why for ForgeAI**: ForgeEngine's `hybrid_offload.py` does static placement. DOPS adds *dynamic* placement + *layout co-optimization*. The WLA is critical for 12GB — different weight layouts for different devices.
- **Implementation**: New `DOPSScheduler` in `scheduler/dops.py` (new file).

#### R2-61. cflow — CPU-First Streaming Engine with Pipeline-Native Architectures
- **Paper**: arXiv 2608.23841
- **What**: Co-design model architecture + runtime. L2-sized tiles in compute-consumption order. Top-k experts only. Fused projections. Delay-aware schedule. 2× critical-path weight bandwidth reduction. 5.94 tok/s on 30.9B MoE.
- **Why for ForgeAI**: The "L2-sized tiles in compute-consumption order" is a novel layout for CPU inference. The "delay-aware schedule from per-model dependency parameters" is a novel scheduling approach.
- **Implementation**: Add cflow-style tiling to `cpu_kv_offload.py`.

#### R2-62. Breakable CUDA Graph (BCG) — SGLang
- **Paper**: LMSYS blog 2026-08-17
- **What**: Segmented CUDA Graph execution for prefill. 1.70× faster than eager, 3.8-5.2× faster graph builds than torch.compile. Full CUDA Graph for prefill with FA4/FlashInfer. Memory reuse across shapes and graph segments.
- **Why for ForgeAI**: ForgeEngine has `breakable_cuda_graph.py` — verify it implements BCG's segmented execution and memory reuse across shapes. If not, upgrade.
- **Implementation**: Verify/upgrade `graphs/breakable_cuda_graph.py`.

#### R2-63. GraCE — Compiler Framework for CUDA Graphs
- **Paper**: OSDI 2026 (Ghosh et al.)
- **What**: Automatic code transformations for CUDA Graph amenability. Eliminates parameter copy overhead. Selective deployment via cost-benefit analysis. 2× benefit over PyTorch2.
- **Why for ForgeAI**: ForgeEngine's CUDA Graph support is manual. GraCE automates it via compiler — could reduce the engineering effort for new model variants.
- **Implementation**: Add GraCE-style automatic graph capture to `graphs/foundry.py`.

#### R2-64. RouteNLP / CASCADIA / UCCI / RLCascadeRouter — Model Cascade Routing
- **Papers**: RouteNLP (arXiv 2604.23577), CASCADIA (ICLR 2026), UCCI (arXiv 2605.18796), RLCascadeRouter (arXiv 2608.15817)
- **What**: Route queries across tiered model portfolio. RouteNLP: difficulty-aware router + conformal cascading + distillation co-optimization, 58% cost reduction. CASCADIA: bi-level optimization (deployment + routing), 4× tighter SLOs. UCCI: calibrated uncertainty, 31% cost reduction. RLCascadeRouter: RL-based, quality-estimator-free.
- **Why for ForgeAI**: ForgeEngine has `generate_adaptive` but no model cascading. On 12GB, can't hold multiple models simultaneously — but could *swap* between a small (1.2B) and large (8B) model. Cascading reduces average cost.
- **Implementation**: New `ModelCascade` in `inference/cascade.py` (new file). Integrates with `sleep`/`wake` for model swapping.

#### R2-65. Unified Radix Cache for Hybrid Models
- **Paper**: LMSYS blog 2026-08-11
- **What**: One radix tree for full attention KV + sliding window KV + recurrent states. Component-specific reuse validity. HiCache (GPU L1, Host L2, distributed L3).
- **Why for ForgeAI**: ForgeEngine has `unified_radix.py` with `CacheNode`/`TreeComponent`/`FullAttentionComponent`/`RecurrentStateComponent`/`UnifiedRadixCache` — verify it matches the paper. If not, upgrade with HiCache tiering.
- **Implementation**: Verify/upgrade `scheduler/unified_radix.py`.

### Round 2 Novel Tests to Add

#### R2-T1. EvoSparse Cross-Step Accumulation Test
- Verify that importance vector decays correctly across decoding steps.
- Verify that retrieval heads propagate indices across layers.

#### R2-T2. ProxyAttn Representative Head Pooling Test
- Verify pooled head scores approximate full head scores within tolerance.
- Verify dynamic budget allocation per head.

#### R2-T3. River-LLM KV-Shared Exit Test
- Verify that skipped layers' KV cache is correctly generated by the Exit River.
- Verify state transition similarity predicts KV errors.

#### R2-T4. TIDE Router Calibration Test
- Verify 4MB router checkpoint loads correctly.
- Verify per-token exit layer selection matches convergence criterion.

#### R2-T5. HSRM Reasoning Step Verification Test
- Verify hidden states at reasoning-step boundaries correctly rank candidate solutions.
- Verify 2M-param encoder trains from self-generated trajectories.

#### R2-T6. TriLens Hallucination Detection Test
- Verify 3L-dimensional entropy trajectory is computed correctly.
- Verify detector AUROC on TruthfulQA-style prompts.

#### R2-T7. ARC-Decode Risk-Bounded Acceptance Test
- Verify that under sampling (temperature > 0), ARC-Decode maintains acceptance length.
- Verify risk-bounded relaxation doesn't violate output distribution.

#### R2-T8. Budget Guidance Gamma Predictor Test
- Verify Gamma distribution predictor models remaining thinking length.
- Verify soft token-level guidance adheres to budget.

#### R2-T9. ATLAS-RTC Graduated Intervention Test
- Verify logit biasing → temperature modulation → masking → rollback sequence.
- Verify JSON schema conformance under graduated control.

#### R2-T10. DirectKV Zero-Copy Offload Test
- Verify fused attention+fetch kernel correctness on PCIe (no NVLink-C2C).
- Verify warp-level pipelining hides latency.

#### R2-T11. HybridGen CPU-GPU Parallel Attention Test
- Verify CPU processes older tokens, GPU processes recent tokens.
- Verify next-layer proactive attention uses inter-layer similarity.

#### R2-T12. DART Dynamic FFN Pruning Test
- Verify attention score distribution shifts trigger mask updates.
- Verify <10MB memory, 0.1% FLOPs overhead.

#### R2-T13. Prox Two-Stage Sparse SwiGLU Test
- Verify Stage 1 mask construction from quantized proxy weights.
- Verify Stage 2 exact computation on selected channels.
- Verify compatibility with existing quantization.

#### R2-T14. VQ-Logits Codebook Scatter Test
- Verify K-vector codebook maps to full vocabulary correctly.
- Verify 99% parameter reduction, 6× logit speedup.

#### R2-T15. HNSW Logits Retrieval Test
- Verify top-k token retrieval matches dense projection.
- Verify batch-size-1 speedup on 100k+ vocab.

#### R2-T16. MrRoPE Radix Conversion Test
- Verify MrRoPE-Uni and MrRoPE-Pro as generalizations of YaRN/Self-Extend/DCA.
- Verify 85% recall at 128K on Needle-in-a-Haystack.

#### R2-T17. Jet-Long Dynamic Bifocal RoPE Test
- Verify local window is RoPE-faithful, long-range window adapts rescaling.
- Verify base model recovered exactly at short inputs.

#### R2-T18. ST-MoE Expert Prefetch Prediction Test
- Verify spatio-temporal prediction across adjacent layers + consecutive tokens.
- Verify prefetch overlap hides transfer latency.

#### R2-T19. DSE Conditional Memory Lookup Test
- Verify N-gram indexed embedding table retrieves in O(1).
- Verify CPU offload has negligible throughput overhead.

#### R2-T20. PackInfer Heterogeneous Batch Test
- Verify multiple requests packed into unified kernel launch.
- Verify load-balanced execution groups.

#### R2-T21. Continuous Depth Batching (CDB) Test
- Verify loop iterations scheduled at granularity of individual iterations.
- Verify boundary stages (embedding, LM head) in separate priority queues.

#### R2-T22. Model Cascade Routing Test
- Verify difficulty-aware router routes to correct model tier.
- Verify conformal calibration thresholds are distribution-free.

#### R2-T23. Swift-SVD Closed-Form Compression Test
- Verify single eigenvalue decomposition produces optimal rank allocation.
- Verify 3-70× faster than iterative SVD methods.

#### R2-T24. LAQuant Reasoning QAT Test
- Verify layer-wise lookahead loss preserves next-layer residual stream.
- Verify 15.11pp AIME improvement over ParoQuant.

#### R2-T25. Ternary CPU Kernel (FairyFuse/Litespark) Test
- Verify AVX-512/NEON masked add/sub produces correct ternary GEMV.
- Verify zero floating-point multiply instructions in inner loop.

---

## Summary: Round 2 Research Sweep

**Total new features identified**: 65 (R2-1 through R2-65)
**Total new tests identified**: 25 (R2-T1 through R2-T25)

**Combined with Round 1**: 
- Tier 1 features: 4 (R1) + 7 (R2) = 11
- Tier 2 features: 4 (R1) + 14 (R2) = 18
- Tier 3 features: 4 (R1) + 44 (R2) = 48
- Total features: 12 (R1) + 65 (R2) = 77
- Total tests: 5 (R1) + 25 (R2) = 30

**Highest-impact next targets for ForgeEngine on RTX 5070 12GB**:
1. **R2-1 EvoSparse** — 5.36× attention speedup, training-free, fits existing KV strategies
2. **R2-30 ARC-Decode** — fixes EAGLE-3 under sampling, training-free, plug-and-play
3. **R2-12 River-LLM** — 2.16× speedup, training-free early exit, solves KV absence
4. **R2-16 L2A** — single model traces entire Pareto frontier, perfect for 12GB
5. **R2-19/20 DART/Prox** — FFN pruning, training-free, composes with quantization
6. **R2-5/R2-6 DirectKV/HybridGen** — CPU-GPU split attention, doubles effective KV capacity
7. **R2-40 Budget Guidance** — 26% accuracy gain under tight budgets, training-free
8. **R2-42 ATLAS-RTC** — graduated constrained decoding for agent JSON conformance
9. **R2-34/R2-35 HNSW/VQ-Logits** — output head acceleration for large vocab models
10. **R2-57 FairyFuse/Litespark** — multiplication-free ternary CPU inference for BitNet models

---

## Round 3 Research Sweep — 2026-09-03 (16 additional web searches)

### R3-1. Capture / HybridServe — Activation Cache (store input, recompute KV)
- **Papers**: Capture (casys-kaist), HybridServe (ICCD 2025)
- **What**: Store *input activations* instead of K,V. K,V regenerated via linear projection on demand. 50% memory reduction per cached block. Hybrid KV/ACT caching balances PCIe bandwidth vs GPU compute. 2.19-2.5× throughput.
- **Why for ForgeAI**: ForgeEngine's `cpu_kv_offload.py` stores full K,V. Storing activations halves the offload bandwidth. The "regenerate K,V via linear projection" is essentially free compute on RTX 5070.
- **Implementation**: Add ACT block type to `cpu_kv_offload.py`. Hybrid KV/ACT block table.

### R3-2. KV-Direct — Residual Stream as Sole State (KV Cache is Redundant)
- **Paper**: arXiv 2603.19664
- **What**: Proven bit-identically that K,V at every layer are deterministic projections of the residual stream. Store residual vectors (5KB/token on Gemma 3-4B) instead of KV. Recompute from scratch → token-identical output under greedy decoding.
- **Why for ForgeAI**: ForgeEngine has 14 KV strategies — all store K,V. KV-Direct proves this is redundant. 5KB/token vs ~20-40KB/token for KV. 4-8× memory reduction. Bit-exact.
- **Implementation**: New `ResidualStreamCache(KVCacheStrategy)` in `kv/residual_cache.py` (new file). Stores residual, regenerates KV via W_K, W_V projections.

### R3-3. KV Packet — Recomputation-Free Context-Independent KV Reuse
- **Paper**: arXiv 2604.13226
- **What**: Wrap cached documents in lightweight trainable soft-token adapters. Trained via self-supervised distillation to bridge context discontinuities. Zero FLOPs at reuse time.
- **Why for ForgeAI**: ForgeEngine has `cacheblend.py` which does selective recomputation (non-zero FLOPs). KV Packet eliminates recomputation entirely. The soft-token adapters are tiny.
- **Implementation**: Extend `cacheblend.py` with packet mode. Train adapters via self-play.

### R3-4. RAC — Reference-Aware Activation Compression for Split Inference
- **Paper**: arXiv 2608.04991
- **What**: For edge-cloud split inference, compress boundary hidden states. Retrieve exact-token historical spans for prefill uplinks. Grouped affine alignment + calibrated residual quantization. 1.24-2.72× TTFT.
- **Why for ForgeAI**: ForgeEngine has no split inference. RAC enables edge-cloud split for 12GB VRAM + cloud GPU. The "retrieve historical spans" reuses prefix cache infrastructure.
- **Implementation**: New `SplitInferenceCodec` in `inference/split/rac.py` (new file).

### R3-5. SinkRouter — Sink-Aware Head Skipping
- **Paper**: arXiv 2604.16883
- **What**: Attention sink = stable, reachable, error-controllable fixed point. Detect sink signal, skip computations producing near-zero output. Triton kernel with block-level branching + Split-K. 2.03× at 512K.
- **Why for ForgeAI**: ForgeEngine has no sink-aware optimization. SinkRouter is training-free, head-level (not token-level). Composes with existing sparse attention.
- **Implementation**: New `SinkRouter` in `attention/sink_router.py` (new file).

### R3-6. P0-Sink Circuit — Mechanistic Understanding of Attention Sinks
- **Paper**: arXiv 2603.06591
- **What**: BOS/P0 token hidden states develop amplified ℓ2 norm through certain layers → triggers P0 sink. Three-stage formation during pretraining. Even without BOS, P0 sink re-emerges via deeper MLP sublayers.
- **Why for ForgeAI**: Understanding sinks enables *controlled* sink exploitation. Could amplify/suppress sinks for specific tasks. The "norm inflation" signal is detectable at inference.
- **Implementation**: Add sink detection to `SinkRouter`. Use norm inflation as early signal.

### R3-7. OutRo — Sink Token for Enhanced Contextual Representations
- **Paper**: arXiv 2603.14337
- **What**: Align non-sink token representations with sink representation in feature space. Allow sink token to attend beyond causal constraint. 1.1× overhead, improves video QA.
- **Why for ForgeAI**: Novel use of sinks for *quality improvement* (not just efficiency). The "sink attends beyond causal" is a novel attention modification.
- **Implementation**: Add OutRo mode to `SinkRouter`.

### R3-8. LongCat Sparse Attention (LSA) — Streaming-Aware + Cross-Layer + Hierarchical
- **Paper**: arXiv 2608.01662
- **What**: Three orthogonal strategies: (1) Streaming-Aware Indexing (scattered→contiguous for coalesced HBM); (2) Cross-Layer Indexing (amortize indexing across consecutive layers via distillation); (3) Hierarchical Indexing (coarse-to-fine scoring). Scales to 1M tokens, 1.6T-A48B.
- **Why for ForgeAI**: ForgeEngine has sparse attention but not cross-layer index reuse. The "streaming-aware contiguous layout" is a novel memory layout insight. Cross-layer indexing amortizes the O(L²) scoring cost.
- **Implementation**: New `LSAAttention` in `attention/longcat.py` (new file).

### R3-9. MISA — Mixture of Indexer Sparse Attention
- **Paper**: arXiv 2605.07363
- **What**: Treat DSA indexer heads as MoE pool. Lightweight router picks query-dependent subset of h≪H_I active heads. O(H_I L) → O(hL + H_I M). 3.82× over DSA indexer kernel.
- **Why for ForgeAI**: ForgeEngine has MoE infrastructure. MISA applies MoE *to attention indexers* — novel cross-domain combination. The "indexer-head-axis routing" is a new axis of efficiency.
- **Implementation**: New `MISAIndexer` in `attention/misa.py` (new file).

### R3-10. LRQK — Low Rank Query and Key Attention
- **Paper**: NeurIPS 2025
- **What**: Jointly decompose Q,K into rank-r factors during prefill. Proxy attention scores in O(lr) at decode. Top-k selection + mixed GPU-CPU cache with hit/miss. Exact attention outputs preserved.
- **Why for ForgeAI**: ForgeEngine has no low-rank attention. LRQK is *exact* (not approximate) and reduces CPU-GPU data movement. The "proxy attention via low-rank factors" is novel for KV selection.
- **Implementation**: New `LRQKAttention` in `attention/lrqk.py` (new file).

### R3-11. SparDA — Decoupled Forecast Projection for Lookahead Selection
- **Paper**: arXiv 2606.04511 (NVlabs)
- **What**: Fourth per-layer projection (Forecast) alongside Q,K,V. Predicts KV blocks needed by next layer → lookahead selection overlaps CPU→GPU prefetch with current-layer execution. <0.5% params. 1.7× decode, 5.3× throughput.
- **Why for ForgeAI**: ForgeEngine has `async_d2h.py` for prefetch but no *learned* predictor. SparDA's Forecast projection is a tiny learned module. The "predict next-layer needs from current hidden state" is the DualDecoder idea generalized.
- **Implementation**: New `SparDAProjection` in `attention/sparda.py` (new file).

### R3-12. MLRA — Multi-Head Low-Rank Attention (TP-friendly MLA)
- **Paper**: arXiv 2603.02188
- **What**: Partitionable latent states for efficient 4-way TP decoding. Solves MLA's TP sharding bottleneck. 2.8× decode speedup over MLA.
- **Why for ForgeAI**: ForgeEngine has MLA support. MLRA enables TP for MLA — relevant if ForgeAI ever uses multi-GPU. Forward-looking.
- **Implementation**: Add MLRA mode to existing MLA implementation.

### R3-13. HiSparse — Hierarchical KV Cache for Sparse Attention Serving
- **Paper**: arXiv 2608.07009 (merged into SGLang)
- **What**: Full KV history in host memory, fixed-size GPU cache. Fused CUDA kernel: hit detection + LRU replacement + H2D fetches inside decode CUDA graph. Layer-wise prefetching hides ~50% of miss overhead. Exact (outputs unchanged).
- **Why for ForgeAI**: ForgeEngine has `cpu_kv_offload.py` but no *hierarchical* management with fused kernel. HiSparse is exact and merged into SGLang — production-tested. The "inside the decode CUDA graph" insight is novel.
- **Implementation**: Upgrade `cpu_kv_offload.py` with HiSparse-style hierarchical management.

### R3-14. Min-k Sampling — Semantic Cliff Detection
- **Paper**: ACL 2026 long.681
- **What**: Analyze local shape of sorted logit distribution to identify "semantic cliffs" (sharp transitions from high-confidence to long-tail). Position-weighted relative decay rate. Strict temperature invariance. Low sensitivity to hyperparameters.
- **Why for ForgeAI**: ForgeEngine's `_sample_next_token` uses top-k/top-p/min-p. Min-k is *temperature-invariant* — solves the "temperature sensitivity" problem. Better quality at extreme temperatures.
- **Implementation**: Add `min_k` sampling mode to `_sample_next_token` in `forge_engine.py`.

### R3-15. Shortcut Decoding — Dual-Signal Convergence Detection for CoT
- **Paper**: ACL 2026 long.1330
- **What**: Lightweight MLP probe on hidden states + step-level entropy. Detects reasoning convergence → switches to final answer generation. 35% token reduction, maintains accuracy. No base model update.
- **Why for ForgeAI**: ForgeEngine has `generate_adaptive` but no CoT-specific early exit. Shortcut decoding is training-free (just a probe). The "convergence before text completion" insight is novel.
- **Implementation**: New `ShortcutDecoder` in `inference/reasoning/shortcut.py` (new file).

### R3-16. SyncThink — Training-Free Reasoning Saturation Detection
- **Paper**: OpenReview Hc9jAnIB3f
- **What**: Answer tokens attend weakly to early reasoning, focus on `<|im_start|>`. Monitor reasoning-transition signal → terminate. 62% accuracy with 656 tokens vs 61.22% with 2141 tokens. +8.1 on GPQA by preventing over-thinking.
- **Why for ForgeAI**: Even simpler than Shortcut Decoding — no probe, just attention monitoring. The "answer tokens focus on boundary" insight is observable in existing attention.
- **Implementation**: Add `SyncThinkTerminator` to `reasoning/shortcut.py`.

### R3-17. CoT-Flow — Probabilistic Flow Reasoning
- **Paper**: ACL 2026 long.1215
- **What**: Reconceptualize reasoning steps as continuous probabilistic flow. Quantify each step's contribution to ground-truth. Flow-guided decoding (greedy flow-based) + flow-based RL (verifier-free dense reward).
- **Why for ForgeAI**: ForgeEngine's self-play has reward computation. CoT-Flow enables *dense* per-step rewards without a verifier — critical for 12GB (can't fit verifier model).
- **Implementation**: New `CoTFlowScorer` in `inference/reasoning/cot_flow.py` (new file).

### R3-18. Memory-Augmented CoT Compression
- **Paper**: arXiv 2608.21265
- **What**: Context-Generation Substitution Law: explicit reasoning context substitutes for decode-time generation. Construct reusable reasoning memories from historical traces, retrieve as prefill-side scaffolds. +21-29pp over CoD, 1.14-1.49× latency speedup.
- **Why for ForgeAI**: ForgeEngine has `library_save`/`library_search` for memory. Memory-Augmented CoT reuses *reasoning patterns* (not just facts). The "substitution law" is a novel theoretical insight.
- **Implementation**: Extend library system with reasoning memory mode.

### R3-19. SCoT — Speculative Chain-of-Thought (Large+Small Model Collaboration)
- **Paper**: ACL 2026 findings.76
- **What**: Thought-level drafting with lightweight draft model. Select best CoT draft, correct errors with target model. Thinking behavior alignment. 48-66% latency reduction.
- **Why for ForgeAI**: ForgeEngine has speculative decoding at *token* level. SCoT does it at *thought* level — coarser granularity, bigger speedup. Could use a small 0.5B model as thought-drafter.
- **Implementation**: New `SCoTDecoder` in `decoding.py` or new file.

### R3-20. LBR — Local Branch Routing (Token-Level Test-Time Scaling)
- **Paper**: arXiv 2606.25354
- **What**: Expand small local lookahead tree, forward all branches, lightweight router selects depth-1 subtree. Prune-shift-grow decoding. End-to-end RLVR trainable. Improves Pass@1 and Pass@32.
- **Why for ForgeAI**: ForgeEngine has `generate_beam_search` (planned). LBR is more efficient — only local lookahead, not full beam. The "hidden states of candidate futures" routing is novel.
- **Implementation**: New `LocalBranchRouter` in `inference/reasoning/lbr.py` (new file).

### R3-21. EGB — Entropy-Gated Branching
- **Paper**: EACL 2026 long.235
- **What**: Branch only at high-uncertainty steps. Prune with lightweight verifier. 22.6% accuracy improvement, 31-75% faster than test-time beam search.
- **Why for ForgeAI**: ForgeEngine has entropy computation in `_sample_next_token`. EGB uses entropy as *branching signal* — novel use of existing signal. The "39/50 high-entropy spikes lead to flawed steps" insight is actionable.
- **Implementation**: Add `EntropyGatedBrancher` to `reasoning/lbr.py`.

### R3-22. DREAM — Dual-Phase Reward-Guided Adaptive Reasoning
- **Paper**: ACL 2026 findings.511
- **What**: Separate planning and execution phases. Search over each independently. Dynamic budget allocation — early stop on confident steps, reallocate to challenging ones.
- **Why for ForgeAI**: ForgeEngine's `generate_with_tools` is single-phase. DREAM's planning/execution split is novel for agent workflows. The "execution errors propagate more than planning errors" insight is actionable.
- **Implementation**: Add DREAM mode to `generate_with_tools`.

### R3-23. Gambit — Thought-Level Beam Search
- **Paper**: arXiv 2608.08020
- **What**: Periodically prune unpromising trajectories, branch from high-quality prefixes. Lightweight scorer probes hidden states. Continuous high hardware utilization. Strictly dominates existing baselines.
- **Why for ForgeAI**: ForgeEngine has no thought-level search. Gambit's "scorer probes hidden states" is cheaper than separate verifier. The "continuous high hardware utilization" is critical for 12GB.
- **Implementation**: Add `GambitSearch` to `reasoning/lbr.py`.

### R3-24. Intra-Expert Activation Sparsity
- **Paper**: arXiv 2605.08575
- **What**: 90% intra-expert sparsity available in pretrained MoE models without modification. Skip inactive neurons. 2.5× MoE layer speedup, 1.2× end-to-end in vLLM.
- **Why for ForgeAI**: ForgeEngine has MoE support. Intra-expert sparsity is *free* — no training, no modification. The 90% number is surprisingly high.
- **Implementation**: Add intra-expert sparse execution path to `moe/` modules.

### R3-25. SERE — Similarity-Based Expert Re-Routing for Batch Decoding
- **Paper**: arXiv 2602.07616
- **What**: Re-route tokens from secondary experts to most similar primary counterparts. Preserve critical experts. Custom CUDA kernel, single-line vLLM integration. 2.0× speedup.
- **Why for ForgeAI**: ForgeEngine has `moe_optim.py` with `ElbowRouter`. SERE adds *similarity-based* re-routing — novel for batch decoding. The "single-line integration" suggests low effort.
- **Implementation**: Add `SERERouter` to `moe/routers.py`.

### R3-26. PCoMoE — Path-Compositional MoE Execution
- **Paper**: arXiv 2609.01024
- **What**: Shift from monolithic expert selection to fine-grained path composition. Compatibility-aware layer-wise pruning. Reusable sub-expert structures. 1.31× speedup, +10% accuracy.
- **Why for ForgeAI**: ForgeEngine treats experts as atomic. PCoMoE's "sub-expert structures" is a novel granularity. The +10% accuracy is unusual for a speedup method.
- **Implementation**: Add `PCoMoEExecutor` to `moe/` modules.

### R3-27. MoE-Infinity — Sparsity-Aware Expert Cache for Personal Machines
- **Paper**: arXiv 2401.14361
- **What**: For batch-size-1 on personal machines, experts exhibit high reuse. Sparsity-aware expert cache traces activation patterns, guides replacement/prefetching. 3.1-16.7× per-token latency improvement.
- **Why for ForgeAI**: ForgeEngine runs on RTX 5070 — personal machine, batch-size-1. MoE-Infinity is *specifically designed* for this scenario. The 16.7× number is for exactly our use case.
- **Implementation**: New `MoEInfinityCache` in `moe/expert_cache.py` (new file).

### R3-28. SlidingServe — SLO-Aware Sliding-Window Scheduling
- **Paper**: arXiv 2606.05933
- **What**: Lightweight batch latency predictor. SlidingChunker combines current + next iteration for dynamic chunking. Multi-Level Priority Sorter. BatchConstructor via dynamic programming. 30% service capacity improvement, 16-53% SLO violation reduction.
- **Why for ForgeAI**: ForgeEngine has `generate_batch` but no SLO-aware scheduling. SlidingServe's "combine current + next iteration" is a novel chunking insight.
- **Implementation**: New `SlidingServeScheduler` in `scheduler/sliding_serve.py` (new file).

### R3-29. LPRS + APC — Fairness-Aware Chunked-Prefill Scheduling
- **Paper**: arXiv 2606.09061
- **What**: Aging-based scheduling (accumulated waiting time + remaining prefill work). Latency-Prediction-Based Request Scheduling (LPRS). Active Prefill Control (APC). 10% mean latency reduction, P99 tail latency reduction.
- **Why for ForgeAI**: ForgeEngine has `chunked_prefill.py` but no fairness-aware scheduling. The "aging-based priority" is a classic OS technique applied to LLM serving.
- **Implementation**: Add aging-based priority to `ChunkedPrefiller`.

### R3-30. Kairos — SLO-Aware Disaggregated Scheduling
- **Paper**: arXiv 2605.02329
- **What**: Urgency-based priority scheduling for prefill (predict completion times). Slack-guided adaptive batching for decode. 23.9% TTFT SLO, 27.1% TPOT SLO, 33.8% e2e SLO improvement.
- **Why for ForgeAI**: ForgeEngine has `kairos.py` in scheduler — verify it implements the full paper. The "slack-guided adaptive batching" for decode is novel.
- **Implementation**: Verify/upgrade `scheduler/kairos.py`.

### R3-31. QUARTZ — Quantile-Aware Routing for TTFT SLOs
- **Paper**: ACL 2026 findings.1888
- **What**: Predict service-time *quantiles* (not point estimates) using prompt length, cache-hit signals, decoding params. Route to worker minimizing predicted tail completion. SGLang router upgrade.
- **Why for ForgeAI**: ForgeEngine has no multi-worker routing. QUARTZ's "quantile not point" insight is novel — tail latency matters more than mean for SLOs.
- **Implementation**: New `QuantileRouter` in `scheduler/quartz.py` (new file).

### R3-32. ProServe — Multi-Priority Request Scheduling
- **Paper**: arXiv 2512.12928
- **What**: Two-tier: SlideBatching (engine-level, sliding boundary deadline-first vs density-first) + GoRouting (service-level, gain-oriented capability-aware dispatch). Formalizes multi-priority as service gain maximization.
- **Why for ForgeAI**: ForgeEngine has no priority system. ProServe's "service gain" formalization is novel — different priorities contribute different gains.
- **Implementation**: Add priority queue to `generate_batch`.

### R3-33. Prism — GPU Memory Ballooning for Multi-LLM Co-Serving
- **Paper**: OSDI 2026
- **What**: Elastic memory allocation unifies spatial + temporal sharing. kvcached balloon driver. Fast memory reallocation between model weights and KV cache. Deployed on 10K+ GPUs.
- **Why for ForgeAI**: ForgeEngine has `sleep`/`wake` for model swapping. Prism's "memory ballooning" is a more principled approach. The kvcached driver is open-source.
- **Implementation**: Upgrade `sleep`/`wake` with balloon-style memory management.

### R3-34. eLLM — Elastic Memory Management (Virtual Tensor Abstraction)
- **Paper**: arXiv 2506.15155
- **What**: Virtual Tensor Abstraction decouples virtual address space from physical GPU memory. Elastic inflation/deflation using CPU as extensible buffer. 2.32× decode throughput, 3× larger batch for 128K.
- **Why for ForgeAI**: ForgeEngine's memory management is static. Virtual tensors enable dynamic allocation. The "CPU as extensible buffer" fits the 12GB+32GB split.
- **Implementation**: New `VirtualTensorPool` in `inference/memory/virtual_tensor.py` (new file).

### R3-35. vToken — Token-Level Virtualization for Reclaimable KV Caches
- **Paper**: arXiv 2608.13263
- **What**: Decouple logical token liveness from physical block placement. Token-table indirection + async repacking of live tokens. Preserves PagedAttention + CUDA Graph. 27-72% retained KV block reduction, 1.37× throughput.
- **Why for ForgeAI**: ForgeEngine has `paged.py` with PagedAttention. vToken adds *token-level* (not block-level) reclamation — solves intra-block fragmentation. The "preserves CUDA Graph" is critical.
- **Implementation**: Add vToken layer to `paged.py`.

### R3-36. AVMP — Asymmetric Virtual Memory Paging for Hybrid Mamba-Transformer
- **Paper**: arXiv 2605.22416
- **What**: Separate KV caches (linear growth) and SSM states (fixed) into physically distinct pools behind unified virtual address space. Migrate capacity between pools on allocation failure. 7.6% OOM reduction, 1.83-13.3× throughput. Tested on RTX 3060 12GB.
- **Why for ForgeAI**: ForgeEngine has `mamba_key.py` for hybrid models. AVMP is *tested on RTX 3060 12GB* — exactly our hardware class. The asymmetric pool design is novel for hybrid models.
- **Implementation**: New `AVMPAllocator` in `inference/memory/avmp.py` (new file).

### R3-37. Leviathan — Decoupled Input/Output Representations
- **Paper**: arXiv 2601.22040
- **What**: Replace input embedding matrix with Learned Embedding Vectorization (LEV) — compact continuous mapping. Untied output head. 0.2% param increase, 9% perplexity reduction, 2.1× fewer training tokens. Gains concentrated in rare tokens (81% perplexity reduction).
- **Why for ForgeAI**: ForgeEngine uses tied embeddings. Leviathan is a training-time change but the insight (rare tokens benefit most) could guide inference-time embedding optimization.
- **Implementation**: Forward-looking. Note for next model version.

### R3-38. PIT — Pseudo-Inverse Tying for Stable Token Interface
- **Paper**: arXiv 2602.04556
- **What**: Synchronize embedding and unembedding as coupled projections of shared latent token memory. Orthonormal shared memory via thin polar decomposition. Learned SPD transform via Cholesky factor. Stable triangular solves. Improved training stability + adaptation.
- **Why for ForgeAI**: ForgeEngine's model editing/patching (merge_models.py) assumes stable token interface. PIT guarantees this stability. Forward-looking for next model version.
- **Implementation**: Forward-looking. Note for next model version.

### R3-39. Kronecker Embeddings — Byte-Level Structured Token Representations
- **Paper**: arXiv 2605.29459
- **What**: Replace |V|×d embedding table with fixed byte-level character-position factorization + single learned projection. 91-94% input-side param reduction. Drop-in nn.Embedding replacement.
- **Why for ForgeAI**: For models with large vocab (Qwen3 150k+), embeddings dominate. Kronecker Embeddings eliminate 91-94% of input-side params. Huge for 12GB VRAM.
- **Implementation**: Forward-looking. Note for next model version.

### R3-40. MixLLM — Global Mixed-Precision Between Output Features
- **Paper**: MLSys 2026
- **What**: Identify important output features globally (not per-layer). Two-step dequantization for Tensor Core. Fast data type conversion. Overlap memory access + dequant + MatMul. 10% more bits → perplexity increase <0.2 (vs ~0.5 SOTA).
- **Why for ForgeAI**: ForgeEngine has per-layer quantization. MixLLM's *global* output-feature importance is a novel granularity. The "overlap dequant with MatMul" is a system-level insight.
- **Implementation**: Add global mixed-precision mode to `novel_quant.py`.

### R3-41. SharQ — Bridging Activation Sparsity and FP4 Quantization
- **Paper**: arXiv 2606.26587
- **What**: Online sparse-dense decomposition. N:M mask extracts outlier-dominated sparse backbone → FP4. Dense residual relative to quantized sparse backbone. Two paths share single FP4 weight payload. Fused preparation kernel. Training-free. 2.2-2.4× over FP16 on RTX 5090.
- **Why for ForgeAI**: ForgeEngine has NVFP4 quantization. SharQ *bridges* sparsity and FP4 — multiplicative gains. Tested on RTX 5090 (SM120 family). The "single FP4 weight payload, path-specific scale views" is elegant.
- **Implementation**: New `SharQLinear` in `quant/sharq.py` (new file).

### R3-42. ACBQ — Adaptive Cross-Block Quantization
- **Paper**: ACL 2026 long.1971
- **What**: Treat self-attention and FFN as separate quantization units with module-specific objectives. Adaptive cross-block strategy accounts for cross-layer dependencies. Works for W4A4 and W2.
- **Why for ForgeAI**: ForgeEngine quantizes uniformly. ACBQ's "attention vs FFN as separate units" is novel. The cross-block dependency modeling reduces error propagation.
- **Implementation**: Add ACBQ mode to `novel_quant.py`.

### R3-43. GRINQH — Graded Input-Based Quantization Hierarchy
- **Paper**: arXiv 2606.23419
- **What**: Weight-only PTQ that unifies quantization and sparsification. Activation magnitudes as proxy for importance → dynamic precision levels. Hierarchical nested memory layout. Effective 2-bit generation. New Pareto frontier.
- **Why for ForgeAI**: ForgeEngine has int4/int8. GRINQH enables *effective 2-bit* — critical for 12GB. The "unified quantization + sparsification" is novel. Dynamic precision per channel.
- **Implementation**: New `GRINQHQuantizer` in `quant/grinqh.py` (new file).

### R3-44. HyQuant — Hybrid-Precision Quantization for Attention
- **Paper**: arXiv 2608.27875
- **What**: Quantize most attention states to low-bit, retain vertical-line tokens + local sliding window in high precision. Vertical-line-aware attention-pattern signals. Fused KV dequant + attention. Works for prefill + decode.
- **Why for ForgeAI**: ForgeEngine quantizes KV cache uniformly. HyQuant's "vertical-line tokens in high precision" is a novel attention-specific quantization. The pattern-aware selection is training-free.
- **Implementation**: Add HyQuant mode to KV cache strategies.

### R3-45. Tempo — SVLM as Query-Aware Temporal Compressor for Long Video
- **Paper**: arXiv 2604.08120
- **What**: Small VLM (6B) as local temporal compressor. Adaptive Token Allocation (ATA) — training-free O(1) dynamic router. 0.5-16 tokens/frame. 52.3 on LVBench (4101s) under 8K token budget, outperforming GPT-4o.
- **Why for ForgeAI**: For future VLM support. The "SVLM as compressor" is a novel cascade. ATA's "semantic front-loading" is a novel allocation strategy.
- **Implementation**: Forward-looking. Note for VLM support.

### R3-46. StreamingTOM — Streaming Token Compression for Video
- **Paper**: CVPR 2026
- **What**: Causal Temporal Reduction (fixed per-frame budget, adjacent-frame changes + token saliency). Online Quantized Memory (4-bit, on-demand retrieval). 15.7× KV cache compression. 1.2× lower peak memory, 2× faster TTFT.
- **Why for ForgeAI**: For future streaming VLM. The "causal + fixed-budget" is novel for streaming. 4-bit online quantization with retrieval is a novel memory design.
- **Implementation**: Forward-looking. Note for VLM support.

### R3-47. OmniZip — Audio-Guided Dynamic Token Compression
- **Paper**: CVPR 2026
- **What**: "Listen-to-prune" paradigm — audio guides video token pruning. Audio retention score per time group. Interleaved spatio-temporal compression. 2.51-3.42× speedup.
- **Why for ForgeAI**: For future omni-modal support. The cross-modal guidance (audio→video pruning) is novel.
- **Implementation**: Forward-looking. Note for omni-modal support.

### R3-48. TTF — Temporal Token Fusion for Video
- **Paper**: arXiv 2605.07355
- **What**: Training-free, plug-and-play. Anchor frame selection + local window similarity search + token fusion. 67% visual token removal, 99.5% accuracy. ~0.16 GFLOPs overhead.
- **Why for ForgeAI**: For future VLM support. Simplest video token compression. The "anchor frame + local similarity" is easy to implement.
- **Implementation**: Forward-looking. Note for VLM support.

### R3-49. Unified Spatiotemporal Token Compression
- **Paper**: CVPR 2026
- **What**: Global token retention pool. Unified selection (attention weights + semantic similarity). Clustering + refilling. Text-aware merging inside LLM. 2% retention → 90.1% performance, 2.6% FLOPs.
- **Why for ForgeAI**: For future VLM support. The "text-aware merging inside LLM" is a novel secondary compression.
- **Implementation**: Forward-looking. Note for VLM support.

### R3-50. DynaSplit — Latency-Aware Dynamic Model Partitioning
- **Paper**: WCCST 2026
- **What**: Dynamic partition points + per-layer quantization + cloud TP degree at 100ms intervals. Sub-ms performance model. Hybrid greedy-beam search (>99.8% search space reduction). Async execution overlaps GPU+RDMA (91% pipeline utilization). <300ms p95 for 8-14B on commodity edge.
- **Why for ForgeAI**: ForgeEngine has `hybrid_offload.py` (static). DynaSplit adds *dynamic* partitioning + quantization + TP co-optimization. The sub-ms performance model enables real-time decisions.
- **Implementation**: New `DynaSplitScheduler` in `scheduler/dynasplit.py` (new file).

### R3-51. NetKV — Network-Aware Decode Instance Selection
- **Paper**: arXiv 2606.03910
- **What**: Network cost oracle for disaggregated serving. O(|D|) greedy per-request. Tier rankings robust to stale telemetry. 21.2% mean TTFT reduction, 20.1pp SLO attainment.
- **Why for ForgeAI**: Forward-looking for multi-GPU. The "ignoring network term is arbitrarily suboptimal as context grows" is a proven result.
- **Implementation**: Forward-looking. Note for multi-GPU.

### R3-52. Privacy-Aware Split Inference with Lookahead Decoding
- **Paper**: arXiv 2602.16760
- **What**: Asymmetric layer split (embedding+unembedding local). Lookahead decoding amortizes WAN latency. N-gram speculation accepts 1.2-1.3 tokens/step. 8.7-9.3 tok/s on Mistral 7B over 80ms WAN. 4.9GB local VRAM for 12B.
- **Why for ForgeAI**: ForgeEngine has `hybrid_offload.py`. Privacy-aware split with lookahead is novel. The 4.9GB local VRAM for 12B is relevant to 12GB.
- **Implementation**: New `PrivacySplitInference` in `inference/split/privacy.py` (new file).

### R3-53. Splitwise — Lyapunov-Assisted DRL for Edge-Cloud Partitioning
- **Paper**: arXiv 2512.23310
- **What**: Decompose transformer layers into attention heads + FFN sub-blocks. Hierarchical DRL + Lyapunov optimization. Jointly minimize latency, energy, accuracy. 1.4-2.8× latency reduction, 41% energy reduction.
- **Why for ForgeAI**: The "head + FFN sub-block granularity" is finer than layer-level. DRL + Lyapunov is a novel optimization approach. Forward-looking.
- **Implementation**: Forward-looking. Note for edge-cloud.

### R3-54. Stateful Inference for Multi-Agent Tool Calling
- **Paper**: arXiv 2605.26289
- **What**: Stateful KV cache across persistent sessions — O(Δ_t) delta-only cost vs O(n_t) per-turn. Radix prefix cache with metadata-only sequence aliasing. Prompt-deterministic response cache. Prompt-lookup speculative decoder. 2.1× per-turn, 4.2× on median turn of 35-turn workflow.
- **Why for ForgeAI**: ForgeEngine has `session_cache` and `generate_with_tools`. Stateful inference is a *direct upgrade* — O(Δ) vs O(n) per turn. The "metadata-only sequence aliasing" is novel for multi-agent.
- **Implementation**: Upgrade `session_cache.py` with stateful KV + metadata aliasing.

### R3-55. CachedAttention — Hierarchical KV for Multi-Turn
- **Paper**: arXiv 2403.19708
- **What**: Hierarchical KV caching (GPU/CPU/SSD). Layer-wise pre-loading + async saving. Scheduler-aware fetching/eviction. Decoupled positional encoding for context window overflow. 87% TTFT reduction, 7.8× prefill throughput, 70% cost reduction.
- **Why for ForgeAI**: ForgeEngine has `session_cache` but no hierarchical storage. The "decoupled positional encoding for overflow" is novel — enables longer conversations than context window.
- **Implementation**: Add hierarchical storage to `session_cache.py`.

### R3-56. SwiftCache — Heterogeneous KV Cache Sharing
- **Paper**: arXiv 2606.16135
- **What**: Models with low KV demand donate idle GPU memory to high-demand models. Cross-model KV sharing over NVLink. Only active layer's KV in local GPU. 69% P99 TTFT reduction, 3.98× max context length.
- **Why for ForgeAI**: ForgeEngine runs one model at a time. SwiftCache's "only active layer in GPU" is a novel memory design for single-GPU. The cross-model sharing is forward-looking.
- **Implementation**: Add "active layer only" mode to `cpu_kv_offload.py`.

### R3-57. LPC — Learned Prefix Cache Eviction (already in R1-Tier2)
- **Paper**: NeurIPS 2025
- **What**: Conversational content analysis predicts which conversations will continue. Combined with last access timestamps. 18-47% cache size reduction, 11% throughput.
- **Why for ForgeAI**: Already noted in R1. Confirmed relevant.
- **Implementation**: Upgrade `learned_prefix_cache.py`.

### R3-58. Text-Level Prompt Cache (MNN)
- **Paper**: GitHub alibaba/MNN#4330
- **What**: Text-level comparison avoids BPE round-trip ambiguity. Token-boundary splitting ensures proper suffix. Self-updating cache. History trim safety (full KV clear + re-prefill on divergence).
- **Why for ForgeAI**: ForgeEngine's prefix cache is token-level. Text-level comparison avoids re-encoding issues. The "BPE round-trip ambiguity" is a real problem.
- **Implementation**: Add text-level comparison mode to `prefix_cache.py`.

### R3-59. LongRoPE2 — Near-Lossless Context Window Scaling
- **Paper**: PMLR v267
- **What**: Hypothesis: insufficient training in higher RoPE dimensions causes OOD. Evolutionary search guided by "needle-driven" perplexity. Mixed context window training. 128K extension with 98.5% short-context performance, 80× fewer tokens than Meta.
- **Why for ForgeAI**: ForgeEngine has `LeRoPE`. LongRoPE2's "needle-driven perplexity" is a novel search objective. The 80× token efficiency is significant.
- **Implementation**: Add LongRoPE2 mode to `position/__init__.py`.

### R3-60. Weight Tying Bias — Output Gradients Dominate
- **Paper**: ACL 2026 findings.2027
- **What**: Tied embedding matrices align more with output (unembedding) than input. Output gradients dominate early training. Scaling input gradients reduces bias. Explains why weight tying harms at scale.
- **Why for ForgeAI**: Understanding for next model version. The "scale input gradients" fix is simple.
- **Implementation**: Forward-looking. Note for next model version training.

---

## Round 3 Novel Tests to Add

#### R3-T1. Activation Cache (Capture) Test
- Verify K,V regenerated from stored activations are bit-identical to original.
- Verify 50% memory reduction per cached block.
- Verify hybrid KV/ACT ratio optimization.

#### R3-T2. Residual Stream Cache (KV-Direct) Test
- Verify K,V recomputed from residual are bit-identical (D_KL = 0).
- Verify 5KB/token memory footprint.
- Verify token-identical output under greedy decoding.

#### R3-T3. KV Packet Adapter Test
- Verify soft-token adapters bridge context discontinuities.
- Verify zero FLOPs at reuse time.
- Verify F1 comparable to full recomputation.

#### R3-T4. SinkRouter Head Skipping Test
- Verify sink signal detection identifies correct heads.
- Verify skipped computations produce near-zero output.
- Verify 2.03× speedup at long context.

#### R3-T5. LongCat Sparse Attention Test
- Verify streaming-aware indexing produces contiguous layouts.
- Verify cross-layer indexing amortizes cost (with distillation).
- Verify hierarchical indexing coarse-to-fine scoring.

#### R3-T6. LRQK Low-Rank Attention Test
- Verify proxy attention scores approximate full attention.
- Verify top-k selection matches full attention selection.
- Verify mixed GPU-CPU cache hit/miss mechanism.

#### R3-T7. SparDA Forecast Projection Test
- Verify Forecast predicts next-layer KV block needs.
- Verify prefetch overlaps with current-layer execution.
- Verify <0.5% parameter overhead.

#### R3-T8. HiSparse Hierarchical KV Test
- Verify full KV in host memory, fixed-size GPU cache.
- Verify fused kernel: hit detection + LRU + H2D inside CUDA graph.
- Verify layer-wise prefetching hides ~50% miss overhead.

#### R3-T9. Min-k Sampling Test
- Verify semantic cliff detection identifies correct truncation boundary.
- Verify strict temperature invariance.
- Verify quality maintained at extreme temperatures.

#### R3-T10. Shortcut Decoding Convergence Test
- Verify MLP probe detects reasoning convergence.
- Verify 35% token reduction with maintained accuracy.
- Verify no base model update required.

#### R3-T11. SyncThink Termination Test
- Verify answer token attention focuses on boundary tokens.
- Verify reasoning-transition signal correctly triggers termination.
- Verify +8.1 on GPQA by preventing over-thinking.

#### R3-T12. Intra-Expert Sparsity Test
- Verify 90% intra-expert sparsity in pretrained MoE.
- Verify inactive neuron skipping produces correct output.
- Verify 2.5× MoE layer speedup.

#### R3-T13. SERE Re-Routing Test
- Verify secondary experts re-routed to similar primary counterparts.
- Verify critical experts preserved.
- Verify 2.0× speedup with minimal quality loss.

#### R3-T14. MoE-Infinity Expert Cache Test
- Verify sparsity-aware cache traces activation patterns.
- Verify 3.1-16.7× per-token latency improvement on personal machine.
- Verify batch-size-1 expert reuse.

#### R3-T15. vToken Token-Level Virtualization Test
- Verify token-table indirection decouples liveness from placement.
- Verify async repacking of live tokens.
- Verify CUDA Graph compatibility preserved.

#### R3-T16. AVMP Asymmetric Paging Test
- Verify KV and SSM pools are physically distinct.
- Verify capacity migration on allocation failure.
- Verify 7.6% OOM reduction on 12GB GPU.

#### R3-T17. SharQ Sparse-Dense Decomposition Test
- Verify N:M mask extracts outlier-dominated sparse backbone.
- Verify dense residual relative to quantized sparse backbone.
- Verify single FP4 weight payload with path-specific scales.
- Verify 2.2-2.4× over FP16 on SM120.

#### R3-T18. GRINQH 2-Bit Generation Test
- Verify activation magnitude proxy assigns correct precision levels.
- Verify hierarchical nested memory layout.
- Verify effective 2-bit generation quality.

#### R3-T19. Stateful Multi-Agent Inference Test
- Verify O(Δ_t) delta-only cost per turn.
- Verify metadata-only sequence aliasing for multi-agent.
- Verify prompt-deterministic response cache.

#### R3-T20. DynaSplit Dynamic Partitioning Test
- Verify sub-ms performance model predicts latency.
- Verify dynamic partition points adapt to bandwidth.
- Verify <300ms p95 for 8B on commodity edge.

---

## Summary: Round 3 Research Sweep

**Total new features identified**: 60 (R3-1 through R3-60)
**Total new tests identified**: 20 (R3-T1 through R3-T20)

**Combined with Rounds 1+2**:
- Tier 1 features: 11 (R1+R2) + 12 (R3) = 23
- Tier 2 features: 18 (R1+R2) + 20 (R3) = 38
- Tier 3 features: 48 (R1+R2) + 28 (R3) = 76
- Total features: 77 (R1+R2) + 60 (R3) = 137
- Total tests: 30 (R1+R2) + 20 (R3) = 50

**Highest-impact next targets from Round 3 for ForgeEngine on RTX 5070 12GB**:
1. **R3-2 KV-Direct** — bit-exact 4-8× KV memory reduction, proven redundant
2. **R3-1 Capture** — 50% KV offload bandwidth reduction via activation cache
3. **R3-13 HiSparse** — exact hierarchical KV, merged into SGLang, fused kernel
4. **R3-14 Min-k Sampling** — temperature-invariant sampling, solves temp sensitivity
5. **R3-15/R3-16 Shortcut/SyncThink** — 35% CoT token reduction, training-free
6. **R3-24 Intra-Expert Sparsity** — 90% free sparsity in pretrained MoE, 2.5× speedup
7. **R3-27 MoE-Infinity** — 16.7× for batch-size-1 on personal machines
8. **R3-41 SharQ** — bridges sparsity+FP4, 2.2× on RTX 5090 (SM120)
9. **R3-54 Stateful Multi-Agent** — O(Δ) per-turn, 4.2× on 35-turn workflow
10. **R3-36 AVMP** — tested on RTX 3060 12GB, 13.3× throughput for hybrid models

**GRAND TOTAL across 3 rounds**: 137 features, 50 tests identified for ForgeEngine integration.
