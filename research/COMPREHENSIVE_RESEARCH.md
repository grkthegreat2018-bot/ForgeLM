# Comprehensive LLM Research Document

**Scope**: General research on LLM training, inference, reasoning, quantization, and optimization (2023-2026)
**Focus**: Small-to-mid scale models (1B-13B), consumer/edge deployment, reasoning capabilities
**Date**: August 2026

---

## Table of Contents

1. [Hardware Profile & Consumer GPU Benchmarks](#1-hardware-profile--consumer-gpu-benchmarks)
2. [Weight Quantization](#2-weight-quantization)
3. [KV Cache Optimization](#3-kv-cache-optimization)
4. [Speculative Decoding](#4-speculative-decoding)
5. [Inference Optimizations](#5-inference-optimizations)
6. [Attention Mechanism Innovations](#6-attention-mechanism-innovations)
7. [Training & Alignment](#7-training--alignment)
8. [Synthetic Data Generation](#8-synthetic-data-generation)
9. [Architecture Components Deep Dive](#9-architecture-components-deep-dive)
10. [Python Libraries Ecosystem](#10-python-libraries-ecosystem)
11. [Evaluation & Benchmarks](#11-evaluation--benchmarks)
12. [Memory & Training Optimization](#12-memory--training-optimization)
13. [Verified Benchmark Results](#13-verified-benchmark-results)
14. [Recommendations Summary](#14-recommendations-summary)
15. [MoE for Small Models & Upcycling](#15-moe-for-small-models--upcycling)
16. [Test-Time Compute & Inference-Time Scaling](#16-test-time-compute--inference-time-scaling)
17. [Kernel-Level Optimizations](#17-kernel-level-optimizations)
18. [Safety, Red-Teaming & Jailbreak Defense](#18-safety-red-teaming--jailbreak-defense)
19. [Agent Frameworks & Tool Use for Small Models](#19-agent-frameworks--tool-use-for-small-models)
20. [Tokenizer Deep Dive](#20-tokenizer-deep-dive)
21. [Perplexity vs Downstream Performance](#21-perplexity-vs-downstream-performance)
22. [Model Calibration](#22-model-calibration)
23. [SGLang vs vLLM Deep Comparison](#23-sglang-vs-vllm-deep-comparison)
24. [Additional 2025-2026 Research Highlights](#24-additional-2025-2026-research-highlights)
25. [Chain-of-Thought Training & Self-Taught Reasoning](#25-chain-of-thought-training--self-taught-reasoning)
26. [Reasoning Distillation to Small Models](#26-reasoning-distillation-to-small-models)
27. [Process Reward Models & Step Verification](#27-process-reward-models--step-verification)
28. [Self-Reflection, Self-Correction & Metacognition](#28-self-reflection-self-correction--metacognition)
29. [System 1 / System 2 Dual-Process Reasoning](#29-system-1--system-2-dual-process-reasoning)
30. [Abstract Reasoning & Fluid Intelligence](#30-abstract-reasoning--fluid-intelligence)
31. [Tree-of-Thought, Graph-of-Thought & Planning](#31-tree-of-thought-graph-of-thought--planning)
32. [Reasoning Training Roadmap](#32-reasoning-training-roadmap)
33. [Neurosymbolic & Hybrid Reasoning](#33-neurosymbolic--hybrid-reasoning)
34. [Causal, Counterfactual & Abductive Reasoning](#34-causal-counterfactual--abductive-reasoning)
35. [Analogical Reasoning & Transfer](#35-analogical-reasoning--transfer)
36. [World Models & Model-Based Reasoning](#36-world-models--model-based-reasoning)

---

## 1. Hardware Profile & Consumer GPU Benchmarks

### Consumer GPU Specifications (RTX 5070 / 4070-class)
- **Architecture**: Blackwell (GB205 die)
- **VRAM**: 12GB GDDR7
- **Memory Bandwidth**: 672 GB/s (192-bit bus, PAM3 signaling)
- **Tensor Cores**: 192 (5th gen) — native FP4/F8 support
- **FP8 Tensor TFLOPS**: ~247 dense / ~494 sparse
- **TDP**: 250W
- **Key advantage over RTX 4070**: 33% more bandwidth (672 vs 504 GB/s), 2.1x FP8 TFLOPS

### Measured LLM Performance (Third-Party Benchmarks)

| Model | Quant | Context | Token Gen (tok/s) | Prompt Processing (tok/s) | Source |
|-------|-------|---------|-------------------|--------------------------|--------|
| Qwen3-4B | Q4_K_M | 4K | 180.2 | — | gpubattle.com |
| Llama 3.1 8B | Q4_K_M | 4K | 119.9 | — | gpubattle.com |
| Qwen3 8B | Q4_K | 4K | 85.8 | 3,487 | hardware-corner.net |
| Qwen3 8B | Q4_K | 16K | 59.1 | 1,600 | hardware-corner.net |
| Qwen3 14B | Q4_K | 16K | 40.6 | 1,315 | hardware-corner.net |
| Phi-3 Mini | Q4_K_M | 4K | 165.0 | — | myaihardware.com |
| Mistral 7B | Q4_K_M | 4K | 88.0 | — | myaihardware.com |

### Blackwell-Specific Features
- **FP8 (E4M3/E5M2)**: Native 5th-gen Tensor Core support. Halves VRAM vs FP16, retains 99% quality. `<1%` accuracy loss on standard benchmarks.
- **NVFP4**: Hardware-native 4-bit floating point. 1.6x throughput vs BF16 with only 2-4% quality loss. Effectively doubles model capacity — some 20B models that OOM at FP8 can fit at NVFP4. Currently requires vLLM + TensorRT-LLM pipeline.
- **FlashAttention-3**: Blackwell-native, significant speedup over FA2.

### Model-Specific Estimate (1-2B scale)
- **BF16 weights**: 2.34 GB → fits easily in 12GB
- **INT4 weights**: ~0.6 GB → massive headroom for KV cache
- **Expected throughput**: 300-500+ tok/s (extrapolating from Phi-3 Mini at 165 tok/s — hybrid conv-attention models is smaller and hybrid conv is faster)
- **Context budget at INT4**: Could support 64K+ context with INT4 weights + INT8 KV cache

---

## 2. Weight Quantization

### 2.1 INT4 Weight-Only Quantization (GPTQ, AWQ, ExLlamaV2)

**What**: Post-training quantization compressing weights to 4-bit integers while keeping activations in FP16/BF16.

**GPTQ (arXiv:2210.17323)**:
- Uses approximate second-order information (Hessian) for one-shot quantization
- Quantizes each row of weight matrix independently to minimize error
- **Accuracy**: 0.8-2.7% average drop (varies by model)
- **Speedup**: 3.25x on A100, 4.5x on A6000
- **Memory**: 4x reduction
- WikiText PPL LLaMA-7B: FP16=5.68, GPTQ-4bit=6.09 (+7.2%)

**AWQ (Activation-aware Weight Quantization)**:
- Preserves ~1% of important weights at higher precision based on activation magnitudes
- **Accuracy**: 1.8% average drop (slightly better than GPTQ on multilingual)
- **Speedup**: 3x at small batch, degrades at high batch
- Qwen2-1.5B: BF16=48.4%, AWQ=46.5% (-1.9% avg) — relevant for our small (1-2B) scale

**ExLlamaV2 (EXL2 format)**:
- Mixed precision (2-8 bits) within a single model
- Custom CUDA kernels optimized for consumer GPUs
- **Accuracy**: <1% PPL increase
- **Speed**: 8-10x faster than HF Transformers
- TinyLlama 1.1B @ 3.0 bpw: 656 t/s (3090Ti), 770 t/s (4090)
- **Best Windows support** of all INT4 methods

**Key Finding (ACL 2025, "Give Me BF16 or Give Me Death")**:
- FP8 (W8A8-FP) is **effectively lossless** across all model scales
- Well-tuned INT8 (W8A8-INT): only 1-3% accuracy degradation
- INT4 weight-only (W4A16-INT): **rivals 8-bit quantization** in accuracy
- Simple GPTQ variant **outperforms AWQ** on real-world tasks (challenges prior assumptions)

**Long-Context Warning (EMNLP 2025)**:
- 8-bit quantization preserves accuracy (~0.8% drop) on long-context tasks
- 4-bit methods show substantial losses on tasks with ≥64K token inputs (drops up to 59%)
- AWQ: 1.8% avg drop, GPTQ-int4: 2.7%, BNB-nf4: 6.9%
- **Implication**: Use INT8 for long-context, INT4 acceptable for short-context

**Practical Implications**: HIGHLY RELEVANT. small (1-2B) in FP16 = 2.4GB. INT4 = 0.6GB, leaving 11.4GB for KV cache. ExLlamaV2 is the best fit for consumer GPU (e.g. RTX 5070/4070) Windows.

### 2.2 INT8 Quantization (LLM.int8(), SmoothQuant)

**LLM.int8() (NeurIPS 2022)**:
- Mixed-precision: >99.9% of values in 8-bit, outlier features in 16-bit
- **Accuracy**: Negligible (near-zero) loss
- **Speedup**: 1.5-1.56x
- **Memory**: 2x reduction

**SmoothQuant (arXiv:2211.10438)**:
- Migrates quantization difficulty from activations to weights
- **Accuracy**: 1-3% drop
- **Speedup**: 1.56x, 2x memory reduction
- WikiText PPL LLaMA-2-70B: <0.47 loss vs FP16

**When INT8 beats INT4**: Large-batch serving (batch ≥16), models with significant outliers, training scenarios (QAT), long-context tasks.

**Practical Implications**: MODERATE. INT8 = 1.2GB weights. Less critical since INT4 fits easily. Prefer INT8 if long-context or training planned.

### 2.3 FP8 (E4M3, E5M2) — Blackwell Native

**E4M3**: Max value 448, ~18 binades dynamic range. For weights/activations (forward pass).
**E5M2**: Max value 57,344, ~32 binades. For gradients (backward pass).

**Accuracy (ACL 2025 study, 500K+ evaluations)**:
- FP8 W8A8-FP: **0.2% average accuracy drop** — effectively lossless
- INT8 W8A8-INT: 0.8% average drop
- INT4 W4A16-INT: competitive with 8-bit

**consumer GPU (e.g. RTX 5070/4070) Blackwell Support**:
- Native 5th-gen Tensor Core FP8 computation
- No performance penalty vs INT8
- FP8 superior to INT8 for inference quality (maintains floating-point exponent)
- Llama 3.1 8B FP8: ~8GB VRAM (vs 16GB FP16) — fits 12GB with 4GB KV budget
- **For small (1-2B) model**: FP8 weights = ~1.2GB, near-lossless quality

**Practical Implications**: MODERATE-HIGH. consumer GPU (e.g. RTX 5070/4070) supports natively. Near-lossless quality. Best choice when quality is paramount and memory is sufficient. Limited Windows ecosystem support vs INT4.

### 2.4 BitNet b1.58 (1.58-bit Ternary)

**What**: Native 1-bit LLM architecture with ternary weights {-1, 0, +1}, trained from scratch.

**Performance (arXiv:2402.17764)**:
- Matches FP16 Transformer in perplexity and end-task performance
- 2B parameter model trained on 4T tokens
- bitnet.cpp: up to **6.25x speedup** over full-precision, **2.32x** over low-bit baselines
- CPU speedups: 2.37x-6.17x (x86), 1.37x-5.07x (ARM)
- Energy reduction: 55-82%

**GPU W2A8 kernel (Microsoft)**:
- Custom CUDA kernels for 2-bit weight × 8-bit activation
- Up to 3.63x speedup over BF16 on A100
- BitNet-b1.58-2B vs Gemma-2-2B BF16: 3.27x faster (64 input, 16 output)

**Practical Implications**: NOT APPLICABLE to existing hybrid conv-attention models (requires training from scratch). If training a new model from scratch, BitNet-style training could be considered for extreme efficiency.

### 2.5 QuaRot & SpinQuant (Rotation-Based)

**QuaRot (arXiv:2404.00456)**:
- End-to-end W4A4KV4 quantization via Hadamard rotations
- Removes outliers by spreading them across dimensions
- LLaMA-2-70B: retains 99% zero-shot performance, 3.33x prefill speedup, 3.89x memory savings
- 6-bit and 8-bit: lossless with round-to-nearest (no calibration needed)

**SpinQuant (arXiv:2405.16406)**:
- Learned rotation matrices via Cayley SGD (vs QuaRot's random rotations)
- LLaMA-2 7B W4A4KV4: 2.9 point gap to FP16
- Surpasses LLM-QAT by 19.1 points, SmoothQuant by 25.0 points
- For LLaMA-3 8B: reduces gap to FP16 by 45.1% vs QuaRot
- Random rotation variance can cause up to 13 points difference — learning rotations eliminates this

**Practical Implications**: LOW-MODERATE. Complex to implement. Benefits most at large scale (7B+). For small (1-2B), simpler methods (GPTQ, AWQ, bitsandbytes) sufficient. Consider if planning W4A4KV4 end-to-end quantization.

### 2.6 WANDA (Pruning)

**What**: Pruning by Weights and Activations — removes weights based on |W| × ||X||₂.

**Results (arXiv:2306.11695)**:
- 50% unstructured sparsity: WANDA PPL 6.42 vs magnitude pruning 14.89 (LLaMA2-7B)
- Competitive with SparseGPT without retraining
- Supports 2:4 and 4:8 structured sparsity

**Practical Implications**: LOW. Pruning less critical when memory isn't the bottleneck. Quantization provides better savings with less complexity.

### 2.7 QServe (W4A4KV4 Serving)

**What**: System co-design for W4A8KV4 quantization optimized for large-batch cloud serving.

**Results (arXiv:2405.04532)**:
- Llama-3-8B: 1.2x on A100, 1.4x on L40S vs TensorRT-LLM
- Qwen1.5-72B: 2.4x on A100, 3.5x on L40S
- Reduces serving cost by 3x

**Practical Implications**: LOW. Designed for large-batch cloud serving. Overkill for single-GPU deployment.

---

## 3. KV Cache Optimization

### 3.1 KV Cache Quantization (INT4/INT8)

**INT8 KV**: Near-lossless (<1% PPL degradation), 2x memory reduction, +30% RPS throughput.
**INT4 KV**: 1-7% PPL degradation (model-dependent), 4x memory reduction, +40% RPS.

**Model-specific INT4 KV tolerance**:
| Model | GQA Ratio | INT4 Tolerance | INT2 Tolerance |
|-------|-----------|----------------|----------------|
| Mistral-7B-v0.1 | 4:1 | Excellent (-1.0% PPL) | Good (+2.2% PPL) |
| Qwen3-8B | 4:1 | Good (+7.0% PPL) | Degraded (+96% PPL) |
| Yi-1.5-9B | 8:1 | Excellent (-0.3% PPL) | Usable (+15% PPL) |

**hybrid conv-attention models GQA**: 32 query heads, 8 KV heads = 4:1 GQA ratio → likely good INT4 KV tolerance.

**Practical Implications**: MODERATE for long contexts. For short contexts (<4K), KV cache is small. For 16K+ context, INT8 KV recommended (near-lossless). INT4 KV viable given 4:1 GQA.

### 3.2 SnapKV (arXiv:2404.14469)

**What**: Training-free KV cache compression selecting important KV positions per attention head based on observation window patterns.

**Results**:
- **3.6x generation speedup**, **8.2x memory efficiency** at 16K tokens
- Comparable accuracy across 16 long-sequence datasets
- 380K context on single A100-80GB with negligible accuracy drop

**Caveat (ACL 2026, "Pitfalls of KV Cache Compression")**:
- Certain instructions degrade much more rapidly with compression
- System prompt leakage as case study — compression can cause instructions to be "ignored"
- Compression method, instruction order, and KV eviction bias all contribute

**Reasoning Task Warning (arXiv:2512.12008)**:
- H2O and decoding-enabled SnapKV are dominant strategies for reasoning models
- No singular strategy fits all — performance heavily influenced by dataset type
- Low-budget eviction can produce longer reasoning traces (tradeoff: cache size vs inference cost)

**Practical Implications**: LOW-MODERATE. Most beneficial for >16K context. For moderate contexts, simpler quantization sufficient. Consider for extreme long context (>32K).

### 3.3 H2O (Heavy Hitter Oracle, arXiv:2306.14048)

**What**: Dynamic KV cache eviction retaining balance of recent tokens and "Heavy Hitters" (tokens contributing most to attention scores).

**Results**:
- Up to **29x throughput** improvement vs DeepSpeed Zero-Inference
- Up to **1.9x latency reduction**
- Heavy Hitter ratio: ~20% typical
- H2 emergence correlates with frequent token co-occurrence

**Practical Implications**: LOW. Designed for large-scale multi-tenant serving. Overkill for single-GPU small (1-2B).

### 3.4 StreamingLLM (arXiv:2309.17453)

**What**: Uses "attention sinks" (initial tokens) + sliding window for infinite-length generation.

**Results**:
- Stable language modeling up to **4M+ tokens**
- **22.2x speedup** over sliding window recomputation
- Just 4 initial tokens sufficient as attention sinks

**Practical Implications**: MODERATE for streaming/multi-turn dialogue. Enables infinite context without memory growth. Simple to implement.

### 3.5 PagedAttention (vLLM, arXiv:2309.06180)

**What**: OS-style virtual memory paging for KV cache. Blocks of 16 tokens allocated on demand from global pool.

**Results**:
- **24x throughput** vs HuggingFace Transformers
- **3.5x** vs HuggingFace TGI
- **2-4x** vs FasterTransformer/Orca
- Near-zero memory waste (<4% vs 60-80% traditional)
- Block sharing via copy-on-write: 55% overhead reduction for beam search

**Tuning**:
- `--enable-prefix-caching`: 30-60% throughput gain on shared system prompts
- `--enable-chunked-prefill`: smooths P99 latency
- `--gpu-memory-utilization 0.92-0.95`: more KV blocks, more concurrency

**Practical Implications**: HIGHLY RELEVANT for production serving. Essential for multi-request serving on limited VRAM (8-16GB). Less critical for single-user inference.

### 3.6 Hadamard/RotorQuant KV Cache Rotation

**Hadamard (WHT)**: Butterfly network, O(d log d) complexity. Spreads outliers before quantization.
- Block-diagonal Hadamard recovers nearly all accuracy lost by naive INT4
- Within 1-3 points of BF16 on fragile models where naive INT4 scores zero
- Fused rotation-quantization kernel: zero measurable end-to-end overhead

**RotorQuant**: Block-diagonal rotations using Clifford algebra Cl(3,0) rotors. O(d) complexity.
- 28% faster decode, 5.3x faster prefill, 44x fewer params vs TurboQuant
- Better PPL: 6.91 vs 7.07

**IsoQuant (arXiv:2603.28430)**: Quaternion 4D rotations. Production-ready in llama.cpp.
- 4.5x-4.7x speedup over RotorQuant
- Peak speedups above 6x

**PlanarQuant**: 2D Givens rotations. Fastest, good accuracy. Also in llama.cpp.

**Practical Implications**: MODERATE for INT4 KV cache. Essential if using INT4 KV for accuracy. IsoQuant/PlanarQuant available in llama.cpp with good Windows support.

### 3.7 Advanced KV Compression (2025-2026)

**VecInfer (ACL 2026)**: Vector quantization with outlier-suppressed codebook.
- 2-bit KV cache with performance comparable to full precision
- 2.7x speedup in large-batch self-attention
- 8.3x reduction in single-batch end-to-end latency on Llama-3.1-8B at 196K context

**C²KV (arXiv:2607.17715)**: Compressed and composable KV cache reuse.
- 17x inference speedup under long contexts
- Position-agnostic compressed KV manifold

**SparDA (arXiv:2606.04511)**: Sparse decoupled attention with Forecast projection.
- <0.5% additional parameters
- 1.25x prefill speedup, 1.7x decode speedup
- 5.3x higher decode throughput via larger batch sizes

**Practical Implications**: LOW. These are cutting-edge methods for large models with extreme context. Overkill for small (1-2B) on 12GB.

---

## 4. Speculative Decoding

### 4.1 Vanilla Speculative Decoding (Leviathan 2023, arXiv:2305.09781)

**What**: Draft model proposes K tokens, target model verifies in parallel. **Zero quality loss** — mathematically proven to preserve target distribution.

**Results**: 2-3x speedup on T5-XXL. Example: 38-token sentence with only 9 serial runs using 6M draft model for 97M target.

**Draft Model Sizing for small (1-2B) Target**:
- Optimal ratio: 1:8 to 1:12 (draft:target)
- Hard floor: ~500M parameters (below this, acceptance drops to 30-50%)
- For small (1-2B) target: optimal draft = 100-150M parameters
- **Problem**: Very few quality 100-150M models exist. Draft cost becomes 20-30% of target (vs 5% for large targets)
- **Recommendation**: AVOID separate draft model for small (1-2B). Use EAGLE or self-speculative instead.

### 4.2 EAGLE-1 / EAGLE-2 / EAGLE-3

**EAGLE-1 (ICML 2024, arXiv:2401.15077)**:
- Feature-level drafting: predicts target's second-to-top-layer features
- Draft head is tiny (~50M params, single transformer layer)
- **2.7-3.5x latency speedup** on LLaMA2-Chat 70B
- 2x throughput, provably maintains target distribution

**EAGLE-2 (EMNLP 2024, arXiv:2406.16858)**:
- Dynamic draft tree structure based on confidence scores
- ~1.8x faster than EAGLE-1

**EAGLE-3 (NeurIPS 2025, arXiv:2503.01840)**:
- Abandons feature prediction for direct token prediction
- Multi-layer feature fusion (early, middle, late layers)
- **Up to 6.5x speedup**, 1.4x improvement over EAGLE-2
- Acceptance rates: 75-85% on chat workloads
- Predicts tokens directly instead of features → benefits from increased training data
- **vLLM support** since v0.8.5, CUDA graphs in v0.9.1
- Up to **2.5x speedup** across diverse scenarios in production vLLM

**Practical Implications**: HIGHLY RECOMMENDED. EAGLE-3 draft head is tiny (~50M params). Feature-level approach works with hybrid architecture. Multi-layer fusion could leverage both conv and attention layers. Memory overhead: 10-15% extra KV cache.

### 4.3 Medusa (arXiv:2401.10774)

**What**: Multiple decoding heads added to target model, each predicting tokens at different future positions.

**Results**:
- Medusa-1: **2.2x speedup** (backbone frozen, lossless)
- Medusa-2: **2.3-2.8x speedup** (full fine-tune)
- Acceptance rates: 55-70% (lower than EAGLE-3)
- Simpler than EAGLE — no separate draft model

**Practical Implications**: MODERATE. Simpler than EAGLE but lower acceptance rates. Good fallback if EAGLE training proves difficult. Heads are parameter-efficient.

### 4.4 Multi-Token Prediction (MTP) — DeepSeek-V3 Style

**What**: Auxiliary training module predicting multiple future tokens simultaneously during pre-training.

**DeepSeek-V3 Results (arXiv:2412.19437)**:
- MTP acceptance rate: **85-90%** for second token prediction
- 1.8x throughput improvement out of box
- MTP module: 14B params for 671B base model (~2% overhead)

**vLLM MTP Implementation (DeepSeek-R1)**:
- Acceptance rate: **81-82.3%** for k=1
- Speedup: **1.63x** at QPS=1, 1.18x at QPS=2
- At higher QPS (>6), speedup diminishes to <1.0x (verification dominates)
- Best for low-QPS (single-user) scenarios

**Practical Implications**: LOW for existing model (requires pre-training from scratch). If training hybrid conv-attention models from scratch, MTP could be incorporated. For inference-only, not applicable unless model already trained with MTP.

### 4.5 Lookahead Decoding (Fu 2024, arXiv:2402.02057)

**What**: Jacobi iteration-based parallel decoding. No draft model, no auxiliary data stores.

**Results**:
- Up to **1.8x speedup** on MT-Bench
- Up to **4x speedup** with strong scaling on multiple GPUs for code completion
- Exact decoding — no quality loss
- Compatible with FlashAttention

**Practical Implications**: HIGHLY RELEVANT. No draft model needed. Works with any architecture. Excellent for single-GPU deployment. Good option when draft model unavailable.

### 4.6 Self-Speculative Decoding (Early Exit)

**LayerSkip (ACL 2024, arXiv:2404.18911)**:
- Layer dropout during training + early exit loss
- **2.16x** on summarization, **1.82x** on coding, **2.0x** on semantic parsing

**Kangaroo (2024)**:
- Fixed shallow sub-network as self-draft
- **1.68x** on Spec-Bench with 88.7% fewer parameters than Medusa-1

**DEL (Dynamic Exit Layer, 2025, arXiv:2504.05598)**:
- Adaptively selects exit layer and speculation length
- **2.16-2.62x** over vanilla decoding

**Practical Implications**: HIGHLY RELEVANT. No extra model needed. Memory-efficient for limited VRAM (8-16GB). LayerSkip's layer dropout could work with hybrid architecture. DEL's dynamic selection could optimize for conv vs attention blocks.

### 4.7 Comparison & Recommendation

| Method | Speedup | Acceptance | Memory Overhead | Training Required |
|--------|---------|------------|-----------------|-------------------|
| EAGLE-3 | 3.0-6.5x | 75-85% | 10-15% KV | Yes (draft head) |
| Medusa-2 | 2.3-2.8x | 55-70% | ~8% params | Yes (heads) |
| MTP (k=1) | 1.63x | 81-82% | Built-in | Pre-training only |
| Lookahead | 1.8x | N/A | None | None |
| LayerSkip/DEL | 2.16-2.62x | N/A | None | Layer dropout |

**For small (1-2B) hybrid conv-attention models**:
1. **EAGLE-3** — highest speedup, small overhead, feature-level works with hybrid
2. **LayerSkip/DEL** — if EAGLE training difficult (self-speculative, no extra model)
3. **Lookahead** — if zero additional infrastructure desired
4. **Medusa** — if simplicity prioritized over maximum speedup
5. **AVOID**: Vanilla speculative (no suitable draft model), MTP (requires retraining)

### 4.8 Acceptance Rate Optimization

**LK Losses (2025)**: Direct acceptance rate optimization. 8-10% gain in average acceptance length.
**EDSD (2026)**: Entropy-driven. 24.8% training efficiency improvement, 4.0% longer acceptance.
**DropMatch (2025)**: Training-free dropout sampling. 1.09-1.33x over baseline, combinable with EAGLE-3.

---

## 5. Inference Optimizations

### 5.1 FlashAttention 2 / 3 / 4

**FlashAttention-2**: 2-4x wall-clock speedup over standard attention. Linear memory. 35% H100 utilization.

**FlashAttention-3 (arXiv:2407.08608)**:
- Hopper GPU optimizations (warp-specialization, TMA, async)
- **1.5-2.0x faster** than FA2 with FP16 (740 TFLOPs/s, 75% H100 utilization)
- **FP8**: close to **1.2 PFLOPs/s** (1.3 PFLOPs/s BF16 in NeurIPS version)
- **2.6x lower numerical error** than baseline FP8 attention
- Forward pass: 1.5-2.0x over FA2. Backward: 1.5-1.75x.

**FlashAttention-4**: Written in CuTeDSL. Optimized for Hopper AND Blackwell (H100, B200).
- **consumer GPU (e.g. RTX 5070/4070) (Blackwell)**: FA-4 should provide significant benefits

**Memory savings**: 10x at seq=2K, 20x at seq=4K. Linear vs quadratic.

**Practical Implications**: ESSENTIAL. Should be baseline for all attention layers. consumer GPU (e.g. RTX 5070/4070) (Blackwell) supports FA-3/FA-4. Critical for limited VRAM (8-16GB) constraint. Conv blocks need separate optimization.

### 5.2 FlexAttention (PyTorch 2.5+)

**What**: Flexible API for custom attention patterns via score modification functions, compiled to optimized kernels.

**Use cases**: Custom attention variants (sliding window, sparse, CSA). Research and experimentation.
**Performance**: Competitive with handwritten kernels. Sparsity provides up to 2x for causal masking (50% sparsity).

**Practical Implications**: MODERATE. Useful if customizing attention patterns or researching conv/attention interaction. Not needed for standard deployment.

### 5.3 CUDA Graphs

**What**: Capture and replay GPU operation graphs to reduce CPU launch overhead.

**Performance**:
- CPU launch overhead: from 2μs + 200ns/node to **2.5μs + 1ns/node** (CUDA 12.6)
- 25-40% speedup in graph instantiation
- Up to 15% better repeat launch performance

**When it helps**: CPU-bound workloads, repeated operations, small batch sizes. NOT effective if already GPU-bound.

**Practical Implications**: HIGHLY RELEVANT for inference with batch size 1. Single-GPU deployment often CPU-bound on launch overhead. PyTorch `reduce-overhead` mode uses CUDA Graphs automatically. Particularly useful for decode phase.

### 5.4 torch.compile

**Modes**:
| Mode | Compile Time | Extra Memory | Best For |
|------|-------------|-------------|----------|
| `default` | Low-medium | None | General use, 20-30% speedup |
| `reduce-overhead` | Medium | Yes | Inference, small batches, 30-50% speedup |
| `max-autotune` | High | Maybe | Training, static shapes |
| `max-autotune-no-cudagraphs` | High | None | Dynamic shapes, debugging |

**Results**: 20-40% faster on transformer inference. 1.8-2x geomean on TorchBench (80+ models). vLLM uses torch.compile as core component.

**Dynamic shapes**: `reduce-overhead` auto-skips graphs if dynamic shapes detected. Use `dynamic=False` to specialize on first shape.

**Practical Implications**: ESSENTIAL. Easy single-line optimization. Start with `default`, use `reduce-overhead` for inference (batch=1). Hybrid architecture may cause graph breaks at conv/attention boundaries — may need `max-autotune-no-cudagraphs`.

### 5.5 Continuous Batching (vLLM)

**What**: Dynamic batching where sequences enter/leave mid-generation. Combined with PagedAttention.

**Results**:
- **24x throughput** vs HuggingFace Transformers
- **36.9x** over FasterTransformer (ORCA paper, OSDI 2022)
- **8x** over naive HF serving (Anyscale benchmarks)
- Memory waste drops to <4% (only last partially-filled block)

**Practical Implications**: HIGHLY RELEVANT for serving multiple requests. Less critical for single-user inference.

### 5.6 Chunked Prefill

**What**: Split large prefill into smaller chunks interleaved with decode operations.

**Benefits**: Prevents long prefill from blocking decode latency. Better GPU utilization. Mathematically equivalent to full prefill.

**Practical Implications**: HIGHLY RELEVANT for long prompts. vLLM V1 enables by default. Important for serving workloads with variable prompt lengths.

### 5.7 Prefix Caching / Radix Tree

**What**: Cache KV blocks for common prompt prefixes (system prompts, few-shot examples).

**Results**: 30-60% throughput gain on shared system prompts. Single highest-value flag in vLLM (`--enable-prefix-caching`).

**Practical Implications**: HIGHLY RELEVANT for chat applications with system prompts. Essential for RAG pipelines with shared context.

### 5.8 KV Cache Pre-allocation Strategies

**PagedAttention**: Fixed blocks (16 tokens), on-demand allocation, <4% waste.
**vAttention (2024)**: Virtual memory, 64KB page granularity, no contiguous allocation.
**LayerKV (2024)**: Layer-wise offloading, reduces TTFT without sacrificing TPOT.

**Practical Implications**: PagedAttention essential for serving. Layer-wise offloading not needed (model fits in VRAM).

---

## 6. Attention Mechanism Innovations

### 6.1 GQA (Grouped Query Attention)

**hybrid conv-attention models already uses GQA**: 32 query heads, 8 KV heads = **4x KV cache reduction**.

| Variant | KV Heads | KV Cache | Quality | Speed |
|---------|----------|----------|---------|-------|
| MHA | H (one per query) | 1.0x | Highest | Slowest |
| GQA (G=H/4) | H/4 | 0.25x | Close to MHA | Near MQA |
| MQA | 1 | 1/H x | Noticeable degradation | Fastest |

Uptrained GQA achieves quality close to MHA with 5% of original compute for uptraining.

**Practical Implications**: Already optimal. No changes needed.

### 6.2 MLA (Multi-head Latent Attention, DeepSeek-V2/V3)

**What**: Low-rank compression of QKV projections.

**Results**: **57x compression** vs MHA. KV cache reduced 93.3%. 5.76x generation throughput. 32K context: 120GB (MHA) → 2.2GB (MLA).

**Practical Implications**: NOT directly applicable (hybrid conv-attention models uses GQA). Could be considered for future variants. GQA already sufficient for small (1-2B).

### 6.3 Sliding Window Attention (Mistral)

**What**: Each token attends only to W previous tokens. O(n·W) vs O(n²).

**Properties**: Fixed cache size. After L layers, information propagates L×W tokens. Example: 4K window, 4 layers → 16K effective context.

**Practical Implications**: MODERATE. hybrid conv-attention models has 32K context. Conv blocks may already provide local context. Could reduce memory for very long contexts.

### 6.4 Linear Attention / Mamba / Mamba-2

**Mamba (2023)**: Selective SSMs. 5x higher throughput than Transformers. Mamba-3B outperforms Transformers 2x its size.
**Mamba-2 (2024, arXiv:2406.07887)**: 8B Mamba-2-Hybrid exceeds 8B Transformer on all 12 tasks (+2.65 points average). Predicted 8x faster inference.

**When hybrid wins**: Pure SSMs lag on copying, in-context learning, long-context reasoning. Hybrid models (43% Mamba-2, 7% attention, 50% MLP) exceed pure Transformer.

**Practical Implications**: hybrid conv-attention models is ALREADY hybrid (conv + attention). Conv blocks similar in spirit to SSMs. Current design already validated by architecture search.

### 6.5 Hybrid Conv-Attention (hybrid conv-attention models, Jamba)

**hybrid conv-attention models Architecture (arXiv:2511.23404)**:
- 16 blocks: 10 double-gated short conv + 6 GQA
- **2x faster** prefill/decode on CPU vs similarly sized models
- Hardware-in-the-loop architecture search (STAR)
- hybrid conv-attention models-2.6B: 79.56% IFEval, 82.41% GSM8K

**Why conv layers help**:
- Efficient local processing (linear complexity)
- No KV cache for conv blocks → memory savings
- Hardware-friendly (optimized kernels)
- Complement attention's global context with local patterns
- **Key insight (Liquid AI)**: Short convolutions DON'T need linear attention. hybrid conv-attention models proves softmax attention + gated local convs suffices — no SSM/Mamba/linear attention needed.

**hybrid conv-attention models Conv block formula**:
```python
def lfm2_conv(x):
    x = linear(x)       # input projection
    x = B * x           # gating (input-dependent)
    x = conv(x)          # short conv
    x = C * x            # gating
    x = linear(x)
    return x
```

**Practical Implications**: This IS the architecture. Already optimized. Further optimization possible within this paradigm.

### 6.6 NSA (Native Sparse Attention, DeepSeek 2025)

**What**: Hardware-aligned, natively trainable sparse attention with dynamic hierarchical strategy.

**Results**: Matches or exceeds Full Attention. Substantial speedups on 64K sequences.

**Practical Implications**: LOW. Designed for 27B+ models. Overkill for small (1-2B) scale.

### 6.7 QK-LayerNorm

**What**: Normalize Q and K before attention scores. Controls variance of attention logits, prevents entropy collapse.

**Benefits**: Reduces learning rate sensitivity. Enables stable training across 3 orders of magnitude LR variation. Improves performance in some cases (OLMoE).

**Placement**: Before RoPE rotations (Qwen3, Gemma3, hybrid conv-attention models style).

**Practical Implications**: Already implemented in hybrid conv-attention models (`use_qk_norm=True` in config). Standard practice.

### 6.8 RoPE Variants

**Standard RoPE (arXiv:2104.09864)**: Encodes relative positions through rotation matrices. No learned parameters. Good extrapolation.
- hybrid conv-attention models uses base=1,000,000 for 128K context (32K for VRAM budget)

**YaRN (arXiv:2309.00071)**: Combines NTK-aware interpolation with scaling. Better extrapolation.

**LeRoPE (2025)**: Learnable RoPE frequencies. 3.4% less compute to match RoPE at 2.5B scale.

**AdaRoPE (2025)**: Head-specific rotation frequencies. Better context extension.

**Practical Implications**: Standard RoPE with base=1M sufficient for 32K context. YaRN/NTK if extending to 128K+. LeRoPE/AdaRoPE if training from scratch.

---

## 7. Training & Alignment

### 7.1 DPO (Direct Preference Optimization, arXiv:2305.18290)

**Formula**:
```
L_DPO = -E[log σ(β log πθ(yw|x)/πref(yw|x) - β log πθ(yl|x)/πref(yl|x))]
```

**Results**: Comparable to PPO-based RLHF. Zephyr-7B outperformed Llama-2-70B RLHF on MT-Bench. ~50% faster than PPO. Only 2 models needed (policy + reference) vs 4 in PPO.

**Practical Implications**: HIGHLY RECOMMENDED. Lower compute, stable, good with limited preference data. Avoids PPO's 4-model memory overhead.

### 7.2 SimPO / KTO / IPO

**SimPO (arXiv:2405.14734)**:
- Eliminates reference model — uses average log probability as implicit reward
- **Outperforms DPO** on AlpacaEval 2 with 20% less compute
- Only 1 model in memory (vs 2 for DPO)
- **Best for limited VRAM (8-16GB) constraint**

**KTO (arXiv:2402.01306)**:
- Works with binary feedback (good/bad) instead of preference pairs
- 85-90% of DPO performance
- Excellent if only binary labels available

**IPO (arXiv:2310.12036)**:
- L2 regularization to prevent overfitting
- Better calibration, 2-3% lower win rates
- Useful if calibration critical

**Ranking for small (1-2B)**: **SimPO > DPO > KTO > IPO**
- SimPO: Best performance, lowest memory (no reference model)
- DPO: Proven, stable, good default
- KTO: Only if binary data
- IPO: Only if calibration critical

### 7.3 GRPO (Group Relative Policy Optimization, DeepSeek-R1)

**Formula**:
```
A_i = (r_i - mean({r_1, ..., r_G})) / std({r_1, ..., r_G})

J_GRPO(θ) = E[(1/G) Σ min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i) - β * KL(πθ || πref)]
```

**Results (DeepSeek-R1, arXiv:2501.12948)**:
- 71% pass@1 on AIME 2024 (up from 15.6%)
- 147K H800 GPU-hours (10x less than comparable)
- ~50% memory reduction vs PPO (no critic model)
- Group size 16: best trade-off

**Variants**:
- **SC-GRPO**: Token-level credit assignment. 8.1% improvement over vanilla GRPO.
- **OM-GRPO**: Masks gradients on answer span to prevent "answer hacking". 4.24 point improvement.
- **DAPO**: Higher clip range for exploration.

**Practical note**: If all samples in a group are 100% correct or 0% correct, std=0 → advantage=0 → zero gradients. Skip these batches for efficiency.

**Practical Implications**: HIGHLY RELEVANT for reasoning tasks (math, code) with verifiable rewards. Eliminates critic model memory. Group size 8-16 for limited VRAM (8-16GB).

### 7.4 RLHF vs RLAIF vs DPO vs GRPO Comparison

| Method | Data Needs | Compute | Memory | Stability | Best For | small (1-2B) Fit |
|--------|-----------|---------|--------|-----------|----------|----------|
| RLHF (PPO) | Preferences + RM | Very High | 4 models | Low | Frontier alignment | Low — too memory intensive |
| DPO | Preference pairs | Low-Med | 2 models | High | Most teams | HIGH — recommended default |
| SimPO | Preference pairs | Low | 1 model | High | Memory-constrained | VERY HIGH — best for 12GB |
| GRPO | Verifiable rewards | High | 2 models | Medium | Reasoning (math/code) | HIGH — for reasoning |
| RLAIF/CAI | AI-generated prefs | Medium | 2-3 models | Medium | Scaling labels | MEDIUM — if labels scarce |

### 7.5 Constitutional AI / RLAIF

**CAI (arXiv:2212.08073)**: Train harmless AI using principles (constitution), no human harm labels. Two phases: supervised (self-critique + revision) + RL (AI feedback).

**RLAIF (arXiv:2309.00267)**: Replace human labelers with AI model. Comparable to RLHF. Direct-RLAIF bypasses RM training.

**Critical finding (EMNLP 2024)**: Improvements from RL step largely due to using weaker teacher for SFT vs stronger critic. Simple SFT with GPT-4 as teacher can outperform full LAIF pipeline.

**Practical Implications**: Use RLAIF with strong frontier model (GPT-4/Claude) as judge. Cost-effective. Validate with human spot-checks.

### 7.6 Curriculum Learning

**Depth of Thought (DoT, arXiv:2508.18279)**: Difficulty = reasoning steps in CoT. DoT curricula outperform length/judge-score curricula by 5-8%.

**Goldilocks RL (arXiv:2602.14868)**: Teacher selects "neither too easy nor too hard" questions. 12% improvement over random curriculum.

**TACLer (arXiv:2601.21711)**: Two-phase Thinking/NoThinking hybrid. 50% compute reduction, 9% accuracy improvement.

**Practical Implications**: HIGHLY RECOMMENDED. Small models benefit more from curriculum. Start with simple instructions, progress to reasoning. Reduces compute needed.

### 7.7 Self-Play / Recursive Self-Improvement

**SPIN (arXiv:2401.01335)**: Model generates own training data, distinguishes self-generated from human data. Outperforms DPO with extra GPT-4 data.

**Self-Rewarding LM (arXiv:2401.10020)**: Model provides own rewards via LLM-as-a-Judge. Llama-2-70B after 3 iterations outperforms Claude 2, Gemini Pro on AlpacaEval 2.0. **Requires strong base model** — may not apply to small (1-2B).

**Practical Implications**: SPIN most practical for data-constrained scenarios. Self-rewarding requires sufficiently capable base model (challenging at small (1-2B)). Use conservative iteration (1-2 rounds) with human validation.

### 7.8 Reward Hacking

**Common modes**: Length hacking, keyword stuffing, format manipulation, mode collapse, likelihood hacking.

**Detection**: Monitor reward vs external judge divergence. Track KL divergence. Measure response length shifts.

**Mitigation**: Stronger KL penalty (β=0.1-0.2). Length normalization. Diverse evaluation. Use DPO/SimPO instead of PPO to reduce hacking risk. Small models MORE susceptible — implement strong monitoring.

---

## 8. Synthetic Data Generation

### 8.1 Evol-Instruct (WizardLM, arXiv:2304.12244)

**Two directions**:
- **In-depth**: Add constraints, deepen reasoning, concretize, increase reasoning steps, complicate input
- **In-breadth**: Generate completely new instructions based on existing ones
- **Filtering**: Elimination evolving for failed generations

**Results**: WizardLM-7B on 70K evolved instructions. On difficulty ≥8: 42.9% vs ChatGPT 35.0% win rate. In-depth:in-breadth ratio of 3:1 works well.

**Practical Implications**: HIGHLY RECOMMENDED. Start with 1K-5K seed instructions. Generate 50K-100K diverse instructions. Filter with quality classifier.

### 8.2 Self-Instruct (arXiv:2212.10560)

**Pipeline**: Seed (175 examples) → Generate instructions → Generate I/O pairs → Filter → Fine-tune → Iterate.

**Results**: 33% absolute improvement on Super-NaturalInstructions. 52K instructions from 175 seeds.

**Practical Implications**: Good starting point for initial instruction tuning. Combine with Evol-Instruct for full spectrum.

### 8.3 Magpie (arXiv:2406.08464)

**Key insight**: Aligned LLMs generate user queries when prompted with only pre-query templates.

**Results**: 4M instructions from Llama-3-Instruct. Fine-tuning Llama-3-8B-Base with Magpie surpasses previous public datasets. Sometimes comparable to official Llama-3-8B-Instruct.

**Practical Implications**: Excellent for data generation. Use strong frontier model as teacher. Generate 100K-500K pairs. Particularly valuable for multi-turn conversation data.

### 8.4 Synthetic Data Diversity (2025)

**Key finding (arXiv:2505.24768)**: **Microscopic diversity** (token-level distribution in responses) shows STRONGEST correlation with model performance. Maximum diversity = superior performance across all strategies.

**Multi-source synthetic data** (arXiv:2511.01490): Mitigates distribution collapse. Preserves output distribution breadth. Human data most effective for reducing self-preference bias.

**BARE (arXiv:2502.01697)**: Base models (non-instruction-tuned) offer greater output diversity. Two-stage: base generates diverse data → instruction-tuned refines for quality. 1000 BARE samples can match best similarly-sized models on LiveCodeBench. 101% improvement over instruct-only on GSM8K.

**Practical Implications**: CRITICAL. Focus on microscopic diversity in responses. Use multiple synthetic data sources. Consider base model for diversity generation. Measure diversity at multiple levels.

### 8.5 Data Filtering

**MinHash LSH**: k=128-256 hash functions, threshold 0.7-0.8. Standard error ~0.09 (k=128), ~0.06 (k=256). Linear time and memory.

**Perplexity filtering**: Filter too simple (PPL too low) or too complex (PPL too high). Typical: PPL 100-1000 for web text.

**Practical Implications**: ESSENTIAL for pretraining data. Budget 10-20% of compute for filtering. For synthetic data: additional dedup against training set.

### 8.6 Textbook Quality Data (Phi Series)

**Phi-2 (2.7B)**: 1.4T tokens of "textbook-quality" data. Outperforms models 10x larger on some benchmarks. 50% compute reduction for target performance.

**Phi-4 (14B, arXiv:2504.21318)**: Large innovative synthetic datasets for reasoning. Surpasses GPT-4o on MATH, GPQA.

**What makes "textbook quality"**: Clear explanations (step-by-step), educational structure, accuracy, diversity, coherence.

**Practical Implications**: CRITICAL. Prioritize textbook-quality synthetic data over raw web scrape. Target: 50-100B high-quality tokens (vs 1T+ for larger models). Use frontier models to generate high-quality explanations.

### 8.7 Distillation from Frontier Models

**Law of Capacity Gap (ACL 2025)**: Optimal teacher scales linearly with student. For 1.9B student, optimal teacher ~4-6B. For small (1-2B): optimal teacher ~2.6-4B.

**Generalization vs Fidelity (ACL 2025)**: KD improves small models by up to 10% (peak 22% task-specific). Teacher task expertise matters more than raw performance.

**Agent Distillation (NeurIPS 2025)**: Distill full task-solving behavior with tools. 0.5B-3B students competitive with 1.5B-7B with CoT.

**Practical Implications**: Use hybrid conv-attention models-2.6B or 8.3B as teacher (capacity gap appropriate). Focus on teacher task expertise. Agent distillation if adding tool use.

### 8.8 Multi-Agent Data Generation

**MetaSynth (ACL 2025)**: Meta-prompting with expert agents. 25M tokens for domain adaptation. Mistral-7B: 4.08% Finance, 13.75% Biomedicine improvement.

**GRA (ACL 2025)**: Generator → Reviewer → Adjudicator with small LMs. Matches Qwen-2.5-72B-Instruct quality.

**Logical DA (ACL 2025)**: 4 agents, 2 phases. 7.61% average improvement for fine-tuning. **Reflection mechanism** for continuous quality improvement.

**Practical Implications**: MetaSynth most practical for domain adaptation. Logical DA for reasoning data. Multi-agent approaches improve diversity but add complexity.

### 8.9 Data Mixing Ratios

**Data Mixing Laws (arXiv:2403.16952)**: Optimized mixture achieves performance comparable to 48% more steps on default mixture.

**Two-Phase Pretraining (arXiv:2412.15285)**: Phase 1: high-quality sources. Phase 2: upweight target-ability data. Outperforms random ordering by 3.4%, natural distribution by 17%.

**Practical recommendations for small (1-2B)**:
- **Pretraining**: 70-80% general web, 20-30% domain-specific. Focus on quality over quantity.
- **SFT**: 50% instruction following, 30% reasoning, 20% domain-specific
- **Continued training**: Start pretraining-like, transition to target-domain heavy
- **Target**: 50-100B high-quality tokens total

---

## 9. Architecture Components Deep Dive

### 9.1 SwiGLU FFN (arXiv:2002.05202)

**Formula**: `FFN_SwiGLU(x) = (SiLU(xW) ⊗ (xV))W2` where SiLU(x) = x * σ(x)

**Comparison**:
- SwiGLU: Best performance, ~33% more params than standard FFN. Standard in Llama, Qwen, hybrid conv-attention models.
- GeGLU: Similar to SwiGLU, GELU activation instead of SiLU
- ReLU: Best for sparse inference, worse performance
- ReLU²: Sparse activations, limited adoption

**Results**: 0.1-0.3 PPL improvement over GELU. hybrid conv-attention models already uses SwiGLU (intermediate=8192).

### 9.2 RMSNorm (arXiv:1910.07467)

**Formula**: `RMSNorm(x) = γ * x / RMS(x)` where `RMS(x) = √((1/n) * Σ(x_i²))`

**vs LayerNorm**: Removes mean subtraction (re-centering). 7-64% speedup. Comparable performance. No bias term needed.

**Why no mean subtraction**: LLMs naturally operate orthogonal to uniform vector, making mean subtraction redundant (geometric perspective).

**Partial RMSNorm**: Estimate RMS from p% of inputs. p=10-20% for additional speed.

**hybrid conv-attention models already uses RMSNorm.** Standard in all modern decoder-only LLMs.

### 9.3 RoPE Math (arXiv:2104.09864)

**Rotation**: For each 2D pair (x_{2i}, x_{2i+1}):
```
R_m = [[cos(mθ), -sin(mθ)], [sin(mθ), cos(mθ)]]
```

**Why it works**: Relative position (n-m) emerges naturally from rotation. Preserves inner product structure. No learned parameters. Better extrapolation than absolute embeddings.

**Theta selection**: hybrid conv-attention models uses base=1,000,000 for 128K context (32K for VRAM budget). Theoretical bounds exist for base vs context length.

### 9.4 Weight Tying

**What**: Share parameters between input embedding and output projection: `W_input = W_output = W_shared`

**Recent findings (ACL 2026)**: Tied embeddings align more with output space than input space. Output gradients dominate early training. Can compromise input representation quality at scale.

**PIT (Pseudo-Inverse Tying, arXiv:2601.22040)**: Decouples input/output representations. Uses orthonormal shared memory + learned SPD transform. Identity init = lossless.

**Small hybrid models typically use tied embeddings.** At 1-2B scale, parameter savings matter. PIT provides orthonormal shared memory alternative (identity init = lossless).

### 9.5 Zero-Init Residual (arXiv:2003.04887)

**What**: Initialize residual gates to 0: `y = x + α * f(x)` where α starts at 0.

**Why it works**: At init, y = x (identity). Satisfies dynamical isometry. Prevents vanishing/exploding gradients. Enables 120-layer Transformers.

**Results**: 12-layer Transformers: 56% faster convergence. 1000-layer Wide-ResNet: 94.3% on CIFAR-10.

**Zero-init residual** is a common config option for hybrid models. Identity init = lossless at start.

### 9.6 Manifold Hyper-Connections (mHC, DeepSeek-V4)

**What**: Generalizes residual connections to multiple streams with doubly stochastic constraints (Birkhoff polytope).

**Properties**: Norm preservation (||P||₂ ≤ 1). Compositional closure. Permutation extremes.

**mHC-lite (arXiv:2601.05732)**: Simplified using convex combinations of permutation matrices. Guarantees exact doubly stochasticity.

**hybrid conv-attention models config has `use_mhc` option.** Gate init=0 = lossless. Better suited for very large models. For small (1-2B): standard residuals sufficient.

### 9.7 Attention Residuals (AttnRes, Kimi K3)

**What**: Replaces fixed residual accumulation with softmax attention over preceding layer outputs.

**Results (Kimi K3)**:
- MMLU: 73.5 → 74.6
- GPQA-Diamond: 36.9 → 44.4
- Math: 53.5 → 57.1
- HumanEval: 59.1 → 62.2

**Block AttnRes**: Partitions layers into blocks. O(Bd) memory vs O(Ld). With ~8 blocks, recovers most gains.

**hybrid conv-attention models config has `use_attn_residual` with `attn_res_k=4`.** Block AttnRes with 4-8 blocks feasible for limited VRAM (8-16GB).

### 9.8 Multi-Token Prediction Heads (DeepSeek-V3)

**What**: Auxiliary module for simultaneous multi-token prediction. 14B params for 671B base (~2% overhead). Enhances performance + enables speculative decoding.

**MTP config** typically uses `mtp_n_heads=2`, `mtp_loss_weight=0.3`. Parameter overhead is more significant at 1-2B scale. Skip for existing models unless retraining from scratch.

### 9.9 Conv Layers in LLMs (hybrid conv-attention models, Hyena, StripedHyena)

**hybrid conv-attention models**: 10 double-gated short conv + 6 GQA. 2x faster on CPU. Optimized for edge deployment.

**Hyena (arXiv:2302.10866)**: Long convolutions + data-controlled gating. 2x faster than attention at 8K, 100x at 64K.

**Why double-gated**: First gate (B) = input-dependent modulation. Second gate (C) = output-dependent modulation. Flexible control over information flow.

**Key insight**: hybrid conv-attention models proves short convolutions DON'T need linear attention or SSMs. Softmax attention + gated local convs suffices.

### 9.10 Embedding Centering (OEC, arXiv:2601.02031)

**Problem**: Anisotropic embeddings cause output logit divergence at end of training.

**μ-centering (deterministic)**: `E_centered = E - μ`
**μ-loss (regularization)**: `L_μ = λ * ||μ||²`

**Results**: Both outperform z-loss in training stability. μ-loss significantly less sensitive to hyperparameter tuning. Works with and without weight tying.

**hybrid conv-attention models config has `oec_mode` and `oec_lambda`.** Recommended for stability. Use μ-loss with λ=0.01-0.1. Simple implementation, low overhead.

---

## 10. Python Libraries Ecosystem

### 10.1 Inference Engines

| Library | What | Windows | small (1-2B) Fit | Notes |
|---------|------|---------|----------|-------|
| **vLLM** | High-throughput serving, PagedAttention | Improving | HIGH for serving | De facto standard, 24x vs HF |
| **SGLang** | Structured generation, RadixAttention | Limited | MODERATE | Agent workflows, prefix reuse |
| **llama.cpp** | Cross-platform, GGUF, CPU/GPU | Excellent | HIGH for local | Bedrock of local LLM inference |
| **ExLlamaV2** | Consumer GPU, EXL2 format | Good | VERY HIGH | Best single-stream on consumer GPU (e.g. RTX 5070/4070) |
| **TensorRT-LLM** | NVIDIA first-party, lowest latency | Limited | LOW | Overkill for small (1-2B) |
| **Ollama** | One-line model installs, OpenAI API | Excellent | HIGH for dev | Easiest to use |

### 10.2 Quantization Libraries

| Library | What | Windows | small (1-2B) Fit |
|---------|------|---------|----------|
| **bitsandbytes** | INT8/INT4 (NF4), QLoRA | Good | HIGH if fine-tuning |
| **autoawq** | AWQ 4-bit (deprecated → llm-compressor) | Good | LOW — use vLLM instead |
| **auto-gptq** | GPTQ 4-bit with calibration | Good | MODERATE — GPTQModel preferred |
| **hqq** | Half-Quadratic Quantization, no calibration | Good | MODERATE — quick quantization |
| **torchao** | PyTorch-native quantization, sparsity | Good | MODERATE — PyTorch native |

### 10.3 Attention Kernels

| Library | What | Windows | small (1-2B) Fit |
|---------|------|---------|----------|
| **flash-attn** | FA2/FA3 fused attention | Experimental | HIGH — consumer GPU (e.g. RTX 5070/4070) supports |
| **flash-attn-3** | Hopper-optimized, FP8 | Not tested | NOT RELEVANT — Blackwell, not Hopper |
| **xformers** | Memory-efficient attention | Good | MODERATE — alternative to FA |
| **FlexAttention** | Custom attention patterns (torch 2.5+) | Good | LOW unless custom attention |

### 10.4 Training Libraries

| Library | What | Windows | small (1-2B) Fit |
|---------|------|---------|----------|
| **liger-kernel** | Triton kernels (RMSNorm, RoPE, SwiGLU, CE) | Limited (Triton) | MODERATE if training |
| **unsloth** | 2x faster training, 80% less VRAM | Good | VERY HIGH for fine-tuning |
| **axolotl** | Fine-tuning config framework | Good | MODERATE |
| **trl** | SFT, DPO, GRPO | Good | HIGH for alignment |
| **peft** | LoRA, QLoRA, adapters | Excellent | VERY HIGH — essential for QLoRA |
| **transformers** | Model loading and training | Excellent | ESSENTIAL |

### 10.5 Optimization Libraries

| Library | What | Windows | small (1-2B) Fit |
|---------|------|---------|----------|
| **torch.compile** | JIT compile, kernel fusion | Good | HIGH — easy win |
| **CUDA graphs** | Capture/replay GPU ops | Good | MODERATE — via torch.compile |
| **torchao** | Low-bit dtypes, quantization | Good | MODERATE |
| **deepspeed** | ZeRO, CPU offloading | Limited | LOW — single GPU |
| **accelerate** | Mixed precision, device placement | Excellent | MODERATE |

### 10.6 INT4 GEMM Kernels

| Kernel | What | Windows | small (1-2B) Fit |
|--------|------|---------|----------|
| **marlin** | FP16xINT4 GEMM, near-ideal 4x | Limited | MODERATE |
| **Machete** | Mixed-input GEMM for Hopper | Limited | NOT RELEVANT — Hopper only |
| **Atom** | Various optimized kernels | Experimental | LOW |

### 10.7 Recommended Windows Stack

```
Inference:     ExLlamaV2 (EXL2 4-bit) or llama.cpp (GGUF)
Training:      Unsloth + PEFT + TRL (QLoRA fine-tuning)
Quantization:  bitsandbytes (NF4) or GPTQModel (GPTQ)
Serving:       Ollama (local) or vLLM (if production)
Attention:     FlashAttention (if Windows build available) or xformers
Optimization:  torch.compile (reduce-overhead mode)
```

---

## 11. Evaluation & Benchmarks

### 11.1 Common Benchmarks

| Benchmark | What | Size | Status | small (1-2B) Target |
|-----------|------|------|--------|-------------|
| MMLU | 57-subject knowledge | 14K | Saturated (93%) | 55-65% |
| HellaSwag | Commonsense reasoning | 70K | Saturated (96.4%) | 70-80% |
| ARC | Visual reasoning | — | Saturated | 60-70% |
| GSM8K | Grade-school math | 8.5K | Saturated (99.7%) | 75-85% |
| HumanEval | Code generation | 164 | Saturated | 40-60% |
| LiveCodeBench | Contamination-free code | 500+ | Current | 15-30% |

### 11.2 Reasoning Benchmarks (2025)

| Benchmark | What | Difficulty | small (1-2B) Target |
|-----------|------|-----------|-------------|
| GPQA Diamond | PhD-level QA | PhD experts 65% | 20-30% |
| MATH-500 | Competition math | Largely solved | 40-50% |
| AIME 2025 | Olympiad math | Very hard | 5-10% |
| SuperGPQA | 285 disciplines | Best ~60% | 15-25% |

### 11.3 Small Model Benchmarks (1-2B)

**ThinkSLM (EMNLP 2025)**: 72 models, 6 families, 17 benchmarks. Key finding: reasoning ability strongly influenced by training methods and data quality, not just scale. Quantization preserves reasoning, pruning disrupts it.

**Standouts at 3-4B** (targets to beat):
- Phi-4-mini (3.8B): 67.3% MMLU, 88.6% GSM8K, 64.0% MATH
- Gemma 3 4B: 43.6% MMLU-Pro, 89.2% GSM8K, 71.3% HumanEval
- SmolLM3-3B: 41.7% GPQA Diamond, 36.7% AIME 2025

### 11.4 Eval Contamination

**Taxonomy**: T1 (exact), T2 (syntactic), T3 (semantic), T4 (task-level).

**Impact**: 6-40% inflation across benchmarks. Small models MORE affected (less capacity to generalize).

**Mitigation**: Use n-gram matching (k=13) for detection. Filter training data against test sets. Use newer benchmarks (LiveCodeBench, AIME 2025).

### 11.5 Speed Benchmarks

**Key metrics**: TTFT (Time To First Token), TPOT (Time Per Output Token), tokens/sec.

**For small (1-2B) on consumer GPU (e.g. RTX 5070/4070)**: Target 300-500+ tok/s (BF16), 400-600+ (INT4).

---

## 12. Memory & Training Optimization

### 12.1 Gradient Checkpointing

**What**: Recompute forward during backward to save VRAM. Trade compute for memory.

**Results**: ~60% memory reduction at ~33% compute overhead. Standard for training large models.

**hybrid conv-attention models config has `use_gradient_checkpointing`.** Recommended for training on limited VRAM (8-16GB).

### 12.2 Liger-Kernel (arXiv:2410.10989)

**What**: Triton kernels for RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy.

**Results**:
- **20% throughput increase**, **60% memory reduction**
- Post-training losses (DPO, ORPO, SimPO): up to **80% memory savings**
- Fused Linear CE: eliminates logits materialization — critical for large vocabularies
- 47% peak memory reduction at batch size 256 with Llama-3.2-1B + torch.compile

**hybrid conv-attention models config has `use_liger_ce` and `use_chunked_ce`.** Recommended if training. Particularly valuable for 65K vocabulary.

### 12.3 Chunked Cross-Entropy

**What**: Fuses head projection + CE without materializing full logits.

**hybrid conv-attention models config has `use_chunked_ce` with `ce_chunk_size=256`.** Eliminates need to materialize 65536-dim logits. Critical for memory efficiency.

### 12.4 Mixed Precision Training

- **bf16**: Default for training. Same dynamic range as FP32, lower precision. consumer GPU (e.g. RTX 5070/4070) supports natively.
- **fp16**: Higher precision but limited range (overflow risk). Use loss scaling.
- **fp8 training**: Possible on Blackwell. Halves memory. Requires careful gradient scaling.

### 12.5 LoRA / QLoRA

**LoRA**: Low-rank adaptation. r=16 recovers within 11.6% of full fine-tuning while training <1% of params and using 31% less peak memory.

**QLoRA**: LoRA on quantized base model. NF4 quantization + LoRA adapters.
- r=16 is sweet spot — diminishing returns beyond
- Paged optimizers (PagedAdamW): 25% throughput improvement
- Enables fine-tuning 7B models on 8GB VRAM

**For small (1-2B) on 12GB**: Full fine-tuning likely possible. QLoRA for maximum memory efficiency or multi-task adaptation.

### 12.6 8-bit Adam / Paged Optimizers

**8-bit Adam**: Halves optimizer memory. Minimal accuracy impact.
**PagedAdamW (bitsandbytes)**: Uses CUDA unified memory for optimizer states. Automatic offloading to CPU when GPU memory pressured. 25% throughput improvement on RTX 4060.

### 12.7 Memory Budget for small (1-2B) Training on 12GB

| Component | BF16 Full FT | QLoRA (NF4) | With Liger |
|-----------|-------------|-------------|------------|
| Model weights | 2.4 GB | 0.6 GB (NF4) | 2.4 GB |
| Gradients | 2.4 GB | 0.1 GB (LoRA) | 2.4 GB |
| Optimizer (AdamW) | 9.6 GB (2x model) | 0.4 GB (LoRA) | 4.8 GB (8-bit) |
| Activations | 2-4 GB | 2-4 GB | 1-2 GB (fused) |
| **Total** | **~16-18 GB** ❌ | **~3-5 GB** ✅ | **~10-12 GB** ✅ |

**Conclusion**: Full BF16 fine-tuning won't fit 12GB. Use QLoRA or 8-bit optimizer + gradient checkpointing + Liger-Kernel.

---

## 13. Verified Benchmark Results (INT4 Quantization)

### Real Measurements (INT4 Quantization on Consumer GPU)

| Config | VRAM (MB) | Speed (tok/s) | Output Match | Notes |
|--------|-----------|---------------|--------------|-------|
| Baseline (BF16) | 4,832 | 55.0 | YES | Reference |
| Custom INT4 | 1,231 | 20.4 | **NO** | Quality degradation! |
| bitsandbytes INT4 (NF4) | 1,232 | 44.7 | YES | Near-lossless |

### Analysis

**Custom INT4**: 75% VRAM savings but **output mismatch** and 0.4x speed (slower than baseline!). The custom INT4 implementation has quality issues — output diverges from BF16. Speed degradation suggests unoptimized dequantization kernels.

**bitsandbytes NF4**: Same 75% VRAM savings, **output matches BF16**, and 0.8x speed (much closer to baseline). NF4 (NormalFloat 4-bit) is better calibrated for transformer weight distributions.

**Key Takeaways**:
1. bitsandbytes NF4 is the clear winner for INT4 on this hardware
2. Custom INT4 implementation needs debugging (quality + speed issues)
3. 75% VRAM reduction (4832 → 1231 MB) leaves massive headroom for KV cache
4. At 1231 MB weights, could support 64K+ context with INT8 KV cache
5. Speed gap (55 → 44.7 tok/s) is due to dequantization overhead — Marlin/ExLlamaV2 kernels would close this gap

### Memory Budget at INT4 (bitsandbytes NF4)

| Component | Memory | Notes |
|-----------|--------|-------|
| Model weights (NF4) | 1,231 MB | Verified |
| KV cache (4K context, FP16) | ~500 MB | 6 attention layers × 8 KV heads |
| KV cache (16K context, FP16) | ~2 GB | Scales linearly |
| KV cache (16K, INT8) | ~1 GB | 2x reduction |
| Activations | ~1 GB | Dynamic |
| **Total (16K context)** | **~4-5 GB** | **7+ GB free!** |

**Conclusion**: limited VRAM (8-16GB) is MORE than sufficient. Even at 32K context with INT4 weights + INT8 KV, total is ~6-7GB. Could potentially run 64K+ context.

---

## 14. Recommendations Summary

### Tier 1: Immediate High-Impact (Do First)

1. **bitsandbytes NF4 quantization** — Verified working, 75% VRAM savings, output-preserving
2. **torch.compile (reduce-overhead)** — Single-line 30-50% speedup for inference
3. **FlashAttention** — Essential for attention layers (if Windows build available)
4. **EAGLE-3 draft head** — 3-6.5x inference speedup, small overhead
5. **PagedAttention + prefix caching** — For any serving scenario

### Tier 2: Training & Alignment (When Training)

1. **SimPO** — Best alignment method for limited VRAM (8-16GB) (1 model, no reference needed)
2. **GRPO** — For reasoning tasks with verifiable rewards (math, code)
3. **QLoRA (r=16) + bitsandbytes** — Fine-tuning on limited VRAM (8-16GB)
4. **Liger-Kernel** — 20% throughput, 60% memory reduction for training
5. **Chunked CE / Fused Linear CE** — Critical for 65K vocabulary
6. **Gradient checkpointing** — Enable for training on limited VRAM

### Tier 3: Data & Quality (When Building Datasets)

1. **Evol-Instruct** — Generate 50K-100K diverse instructions from 1K-5K seeds
2. **Textbook-quality synthetic data** — Quality > quantity for small (1-2B)
3. **Microscopic diversity** — Maximize token-level diversity in responses
4. **Multi-source synthetic data** — Prevent distribution collapse
5. **Curriculum learning** — DoT-based scheduling for reasoning data
6. **Distillation from 2-8B models** — Optimal capacity gap for 1-2B students

### Tier 4: Architecture Enhancements

All keys are zero/identity init = **lossless at start**:
1. **QK-Norm** — Already enabled, stabilizes training
2. **Zero-init residual** — Stable training, faster convergence
3. **PIT** — Better than weight tying, orthonormal shared memory
4. **OEC (μ-loss)** — Suppresses anisotropy, stabilizes late training
5. **MTP** — If training from scratch, enables speculative decoding
6. **mHC** — For deeper variants, manifold-constrained residuals
7. **AttnRes (Block)** — Cross-layer retrieval, 4-8 blocks feasible
8. **Safety layer** — Zero/identity init, lossless integration

### Tier 5: Advanced/Future (When Scaling)

1. **FP8 inference** — Blackwell-native, near-lossless
2. **NVFP4** — 4-bit floating point, 1.6x throughput vs BF16
3. **FlashAttention-4** — CuTeDSL, Blackwell-optimized
4. **MoE upcycling** — Convert dense small (1-2B) to MoE for sparse activation
5. **Lookahead decoding** — No draft model, 1.8x speedup
6. **Self-speculative (LayerSkip/DEL)** — 2.16-2.62x, no extra model

### What to AVOID for Small Models on Limited VRAM

- **Full PPO/RLHF** — 4 models in memory, won't fit consumer GPUs
- **Vanilla speculative decoding** — No suitable 100-150M draft model for 1-2B
- **MTP on existing model** — Requires retraining from scratch
- **BitNet b1.58** — Requires training from scratch
- **QuaRot/SpinQuant** — Complex, benefits most at 7B+
- **H2O** — Overkill for single-GPU, designed for multi-tenant
- **NSA** — Designed for 27B+ models
- **TensorRT-LLM** — Overkill for small (1-2B), limited Windows support
- **DeepSpeed** — Single GPU, Linux-focused

### Expected Performance Gains (Stacked)

| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Baseline (BF16) | 1.0x | 1.0x |
| + torch.compile | 1.3-1.5x | 1.3-1.5x |
| + FlashAttention | 1.5-2x | 2-3x |
| + CUDA Graphs | 1.1-1.2x | 2.2-3.6x |
| + EAGLE-3 | 3-6.5x | 6.6-23x |
| + INT4 (bitsandbytes) | 0.8x (speed) but 75% VRAM | — |
| + Continuous batching (serving) | 2-4x throughput | — |

**For single-user inference**: 6-23x potential speedup with EAGLE-3 + torch.compile + FlashAttention.
**For multi-user serving**: 30-50x throughput with vLLM + PagedAttention + continuous batching + prefix caching.

---

## Appendix: Key ArXiv References

### Quantization
- GPTQ: arXiv:2210.17323
- AWQ: arXiv:2306.00909
- LLM.int8(): NeurIPS 2022
- SmoothQuant: arXiv:2211.10438
- FP8: arXiv:2209.05433
- BitNet b1.58: arXiv:2402.17764
- QuaRot: arXiv:2404.00456
- SpinQuant: arXiv:2405.16406
- WANDA: arXiv:2306.11695
- QServe: arXiv:2405.04532
- BitDistiller: arXiv:2402.10631
- "Give Me BF16 or Give Me Death": ACL 2025

### KV Cache
- SnapKV: arXiv:2404.14469
- H2O: arXiv:2306.14048
- StreamingLLM: arXiv:2309.17453
- PagedAttention/vLLM: arXiv:2309.06180
- VecInfer: ACL 2026
- C²KV: arXiv:2607.17715
- SparDA: arXiv:2606.04511
- IsoQuant: arXiv:2603.28430

### Speculative Decoding
- Vanilla: arXiv:2305.09781
- EAGLE-1: arXiv:2401.15077
- EAGLE-2: arXiv:2406.16858
- EAGLE-3: arXiv:2503.01840
- Medusa: arXiv:2401.10774
- MTP/DeepSeek-V3: arXiv:2412.19437
- Lookahead: arXiv:2402.02057
- LayerSkip: arXiv:2404.18911
- DEL: arXiv:2504.05598

### Training & Alignment
- DPO: arXiv:2305.18290
- IPO: arXiv:2310.12036
- KTO: arXiv:2402.01306
- SimPO: arXiv:2405.14734
- GRPO/DeepSeek-R1: arXiv:2501.12948
- Constitutional AI: arXiv:2212.08073
- RLAIF: arXiv:2309.00267
- SPIN: arXiv:2401.01335
- Self-Rewarding LM: arXiv:2401.10020

### Architecture
- SwiGLU: arXiv:2002.05202
- RMSNorm: arXiv:1910.07467
- RoPE: arXiv:2104.09864
- GQA: arXiv:2305.13245
- MLA/DeepSeek-V2: arXiv:2405.04434
- FlashAttention-3: arXiv:2407.08608
- hybrid conv-attention models: arXiv:2511.23404
- Mamba-2: arXiv:2406.07887
- NSA: arXiv:2502.11089
- OEC: arXiv:2601.02031
- mHC/DeepSeek-V4: arXiv:2512.24880
- AttnRes/Kimi K3: arXiv:2603.15031
- PIT: arXiv:2601.22040

### Data
- Evol-Instruct/WizardLM: arXiv:2304.12244
- Self-Instruct: arXiv:2212.10560
- Magpie: arXiv:2406.08464
- BARE: arXiv:2502.01697
- Phi-4: arXiv:2504.21318
- Data Mixing Laws: arXiv:2403.16952

### Inference
- vLLM: arXiv:2309.06180
- Liger-Kernel: arXiv:2410.10989
- torch.compile: PyTorch 2.0+
- FlexAttention: PyTorch 2.5+

### Safety
- GUARD-SLM: arXiv:2603.28817
- CHASE: arXiv:2606.05523
- RLShield: ACL 2026 findings.1182
- DSICL: ACL 2026 findings.2123

### Test-Time Compute
- s1: arXiv:2501.19393
- Compute-optimal scaling: ICLR 2025

### Tokenizers
- Tokenizer Choice: arXiv:2310.08754
- BPE Suboptimal: EMNLP 2020 findings.414

---

## 15. MoE for Small Models & Upcycling

### 15.1 MoEsturizer (ICLR 2026, OpenReview HDZ2GBwrWo)

**What**: Resource-efficient MoE upcycling for sub-billion parameter models. Converts dense pretrained models into sparse MoE variants using only a few hundred thousand samples of supervised fine-tuning.

**Results**:
- 150K samples, **one 96GB GPU** sufficient
- Upcycled models consistently **outperform dense base models** on 9 benchmarks
- Competitive with dense counterparts of equivalent total size, despite activating fewer parameters
- Experts-Top K configurations: 4-2/8-2 (4 experts top-2, 8 experts top-2)
- **Key finding**: Depth scaling or higher top-k adds little — upcycling is lightweight

**Practical Implications**: HIGHLY RELEVANT. Could convert small models (1-2B) dense to sparse MoE. 150K samples is achievable on consumer GPU (e.g. RTX 5070/4070). Would increase capacity without proportional compute increase. hybrid conv-attention models-8B-A1B already demonstrates MoE works in hybrid architecture family.

### 15.2 SPRI (arXiv:2606.16456) — SVD-Partitioned Residual Init

**What**: Distributes SVD-partitioned residuals from pretrained FFN weights across routed experts, introducing controlled expert diversity.

**Results**: On CoVoST2 (15 En-to-XX directions):
- +2.58 BLEU, +3.32 COMET over fully fine-tuned dense
- +3.39 BLEU, +4.34 COMET over prior best MoE upcycling baseline

**Practical Implications**: Useful if upcycling hybrid conv-attention models to MoE. SVD-based initialization preserves pretrained knowledge while creating expert diversity. Good for multilingual or multi-domain adaptation.

### 15.3 Nexus (EMNLP 2025, findings.1323)

**What**: Adaptive upcycling with novel router using domain embeddings. Can integrate new experts into existing trained model without hurting previous domains.

**Results**:
- 2.1% relative gain for initial upcycling
- **18.8% relative gain** for extending MoE to new domain with new expert
- Enables ecosystem where users continuously assemble their own MoE-mix

**Practical Implications**: Interesting for incremental domain adaptation. Add new experts for new domains without retraining. Router uses domain embeddings for faster specialization.

### 15.4 Drop-Upcycling (ICLR 2025)

**What**: Combines pretrained knowledge reuse with statistical re-initialization of some weights. Promotes expert specialization.

**Results**: 5.9B active param MoE matches 13B dense model performance with **1/4 of training FLOPs**. Outperforms previous MoE construction methods for long-term training (hundreds of billions of tokens).

**Practical Implications**: Less relevant (designed for large-scale continued training). MoEsturizer better for resource-constrained small model upcycling.

### 15.5 Expert Upcycling (arXiv:2604.19835)

**What**: Progressively expand MoE capacity by increasing expert count during continued pre-training. Expert duplication + router extension, holding top-K fixed.

**Results**: 7B→13B upcycled model matches fixed-size baseline on validation loss while saving **~32% GPU hours**. Utility-based expert selection (gradient-based importance) more than triples gap closure.

**Practical Implications**: Useful if progressively scaling up. Start with small MoE, add experts over time. Saves compute vs training large MoE from scratch.

### 15.6 FLAME-MoE (arXiv:2505.20225)

**What**: 38M to 1.7B active parameters. 64 experts with top-8 gating.

**Results**: **3.4 points** accuracy improvement over dense baselines at matched active parameters.

### 15.7 SmallThinker (arXiv:2507.20984)

**What**: Two-level sparse structure — fine-grained MoE + sparse FFN. Pre-attention router for parameter prefetching.

**Results**: SmallThinker-4B-A0.6B: **20+ tokens/s on CPU** with Q4_0. Outperforms larger LLMs. Demonstrates MoE viable for edge deployment.

**Practical Implications**: If converting to MoE, SmallThinker's pre-attention routing + sparse FFN pattern could work. CPU performance relevant for hybrid conv-attention models's edge-first design.

### 15.8 MoE Recommendation for small (1-2B)

**Best approach**: MoEsturizer upcycling
- 150K samples, single GPU feasible
- Converts dense small (1-2B) to sparse MoE (e.g., 4-2 or 8-2)
- Increases total capacity while keeping active params low
- hybrid conv-attention models conv blocks could remain dense (they're already efficient), MoE only for FFN layers
- **Expected**: 2-4 point improvement on benchmarks at similar inference cost

---

## 16. Test-Time Compute & Inference-Time Scaling

### 16.1 s1: Simple Test-Time Scaling (EMNLP 2025, arXiv:2501.19393)

**What**: Minimal recipe for test-time scaling matching o1-preview with just **1,000 examples** + budget forcing.

**Budget Forcing mechanism**:
- Control test-time compute by forcefully terminating thinking OR lengthening it
- Append "Wait" multiple times when model tries to end → model double-checks answer
- Can fix incorrect reasoning steps

**Results**:
- s1-32B (Qwen2.5-32B-Instruct + s1K) exceeds o1-preview on competition math by **up to 27%** (MATH and AIME24)
- Scaling with budget forcing: AIME24 from 50% → **57%** (extrapolation beyond training)
- s1K dataset: 1,000 questions with reasoning traces, selected for difficulty, diversity, quality

**Critical Analysis (arXiv:2507.14419)**:
- Scaling behavior largely attributed to **scaling down** (enforcing maximum length), not scaling up
- Appending "Wait" leads to inconsistencies — model may oscillate between solutions
- Key distinction: o1-like models learn to naturally scale up during RL; simple test-time scaling just imposes lower upper limit
- **Goal should be unlocking higher performance, not reproducing appearance of scaling**

**Practical Implications**: MODERATE. Budget forcing implementable on any model. 1,000 examples is achievable. However, small (1-2B) may lack reasoning capacity to benefit significantly from extended thinking. More relevant for 7B+ models.

### 16.2 Compute-Optimal Test-Time Scaling (ICLR 2025)

**What**: Adaptive allocation of test-time compute based on question difficulty.

**Two mechanisms**:
1. Searching against dense, process-based verifier reward models (PRMs)
2. Updating model's distribution over response adaptively given prompt

**Results**:
- **4x more efficient** than best-of-N baseline for math reasoning
- Test-time compute can outperform **14x larger model** on problems where smaller base model has non-trivial success rate
- Effectiveness critically varies with prompt difficulty

**Practical Implications**: HIGHLY RELEVANT if reasoning is priority. Adaptive compute allocation means easy questions get less compute, hard questions get more. Can punch above weight class on problems where model has some baseline capability.

### 16.3 Implementation

**Budget Forcing (simplest)**:
```python
# Extend thinking by appending "Wait" when model tries to end
MAX_TOKENS_THINKING = 32000
NUM_IGNORE = 1  # How often to ignore end-of-thinking token

# In generation loop:
if model_generates_end_token and tokens_generated < MAX_TOKENS_THINKING:
    append "Wait" to generation
    continue
```

**Best-of-N with verifier (more sophisticated)**:
1. Generate N candidate solutions
2. Score each with verifier (process or outcome reward Model)
3. Select best or aggregate

**Self-consistency (simplest effective)**:
1. Generate N solutions at temperature > 0
2. Take majority vote on final answer
3. Works well for math/code with verifiable answers

---

## 17. Kernel-Level Optimizations

### 17.1 Triton Kernels for LLM Inference

**RMSNorm (custom Triton vs PyTorch)**:
- PyTorch RMSNorm: **11% of peak memory bandwidth**
- Custom Triton: **88% of peak memory bandwidth** — **8.1x speedup**
- Key: single-pass computation (read x once, accumulate sum of squares in registers, normalize, write)
- Fused RMSNorm + Residual: **6.0x speedup**

**SwiGLU (fused)**: **1.6x speedup** over PyTorch
**INT8 GEMM**: ~1.0x speed but 2x memory savings

**Why custom kernels win**:
1. **Kernel launch overhead**: PyTorch dispatches multiple small CUDA kernels (~5-10μs each)
2. **Intermediate tensors**: PyTorch materializes intermediates to GPU memory → more traffic
3. **Generic implementations**: PyTorch handles every edge case; custom kernels optimize for specific shapes/dtypes

### 17.2 Triton FP8 GEMM (PyTorch blog, accelerating-llama3)

**TK-GEMM**: Optimized Triton FP8 GEMM with SplitK parallelization.
- **1.94x** over base Triton matmul
- **1.87x** over cuBLAS FP8
- **1.71x** over cuBLAS FP16
- For Llama3-70B inference on H100

### 17.3 Triton GPTQ Kernel Acceleration (PyTorch blog, accelerating-triton)

**Results**: 3x speedup for core GPTQ kernel, 6x for AutoGPTQ. Example: 275μs → 47μs on typical Llama inference input.

**Techniques**: Coalesced memory access (shared/global memory throughput), reduced warp stalling.

### 17.4 SplitK Fused Dequant+GEMM (arXiv:2402.00025)

**What**: Fused W4A16 dequantization + GEMM with SplitK work decomposition for memory-bound inference.

**Results**:
- **65% average speedup** on A100
- **124% average speedup** on H100 (peak 295%)
- For skinny matmuls (small batch × large weight) common in LLM inference

### 17.5 CUTLASS Integration

**Key insight (maknee blog)**: Adding "cutlass" to CUDA/Triton kernel name triggers ptxas (NVIDIA compiler) to perform additional optimization pass. **100-150 TFLOPs improvement** just from name. ptxas makes optimization decisions based on kernel name.

**Implication**: Backend compiler optimizations matter as much as kernel code. CUTLASS-optimized kernels significantly outperform generic ones for FP8/Tensor Core operations.

### 17.6 Liger-Kernel (covered in Section 12.2)

Triton kernels for RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy. 20% throughput increase, 60% memory reduction. Drop-in replacement for HF transformers.

### 17.7 Kernel Optimization Priority

1. **RMSNorm fusion** (8x speedup) — highest single-op win
2. **Fused dequant + GEMM** for INT4 (124% on H100, likely similar on Blackwell)
3. **SwiGLU fusion** (1.6x)
4. **Fused Linear Cross-Entropy** (Liger, eliminates logits materialization for 65K vocab)
5. **RoPE fusion** (Liger)

**For consumer GPU (e.g. RTX 5070/4070) (Blackwell)**: Triton supports Blackwell. Custom kernels for conv blocks (hybrid conv-attention models-specific) would be unique contribution — no existing library optimizes double-gated short convolutions.

---

## 18. Safety, Red-Teaming & Jailbreak Defense

### 18.1 GUARD-SLM (arXiv:2603.28817)

**What**: Token activation-based defense against jailbreak attacks specifically for Small Language Models.

**Key findings**:
- SLMs remain **highly vulnerable** to malicious prompts that bypass safety alignment
- Different input types form distinguishable patterns in internal representation space
- Lightweight token activation method filters malicious prompts during inference while preserving benign ones
- Analyzes hidden-layer activations across different layers and architectures

**Practical Implications**: HIGHLY RELEVANT — specifically designed for SLMs. Could integrate as a safety layer. Operates in representation space during forward pass (intraprocess defense).

### 18.2 Draft Model Safeguard (arXiv:2605.19321)

**What**: Uses speculative inference with small draft models to pre-screen prompts for safety.

**Key insight**: Jailbreak attacks transfer from LLMs to SLMs. SLM draft responses reflect safety implications of large target model responses.

**Results**:
- Reduces false-negative rate of jailbreak prompts by **32.4%** vs pre-model guards
- Reduces prompt-to-response time by **97.07%** vs post-model guards
- Combines benefits of both approaches

**Practical Implications**: Could use small models (1-2B) as draft model for larger models, OR use even smaller model as draft for small (1-2B). Synergizes with speculative decoding infrastructure.

### 18.3 DSICL: Defensive Suffix + ICL (ACL 2026, findings.2123)

**What**: Template-based ICL with offline-optimized defensive suffix.

**Results**:
- Reduces attack success rate to **nearly 0%** against GCG and AutoDAN
- Maintains model utility on benign tasks
- **Negligible computational overhead**

**Practical Implications**: Simple, effective, no retraining needed. Defensive suffix can be prepended to any prompt. Works against both white-box and black-box attacks.

### 18.4 RLShield (ACL 2026, findings.1182)

**What**: Dynamic jailbreak detection via reinforced adaptive learning. SAC agent learns optimal sample-specific detection thresholds.

**Results**: F1 improvement up to **7.3%**, **3x inference efficiency** gain across multiple LLM backbones.

### 18.5 CHASE (arXiv:2606.05523)

**What**: Co-evolutionary red-blue teaming. Attacker trained via GRPO, defender hardened through GRPO + rejection-sampled SFT.

**Results**: Cuts mean StrongREJECT score by **43.2%** with 0% utility loss on benign tasks. Effective against 5 held-out attack families (PAIR, TAP, AutoDAN, PAP, Translation).

**Practical Implications**: GRPO-based safety training synergizes with reasoning GRPO. Could train safety and reasoning simultaneously. Closed-loop red-blue teaming generates adversarial examples automatically.

### 18.6 Safety Strategy for Small Models (1-2B)

**Tier 1 (Inference-time, no training)**:
1. DSICL defensive suffix — 0% attack success, negligible overhead
2. GUARD-SLM token activation filtering — SLM-specific
3. Input/output filtering

**Tier 2 (Training-time)**:
1. Constitutional AI / RLAIF for safety alignment
2. CHASE-style red-blue teaming with GRPO
3. Safety key — zero/identity init, lossless

**Tier 3 (System-level)**:
1. Draft model safeguard (if using speculative decoding)
2. RLShield dynamic threshold detection
3. Rate limiting + prompt logging

**hybrid conv-attention models specific**: Conv blocks may have different safety properties than attention blocks. GUARD-SLM's layer analysis could reveal which layers (conv vs attention) are most safety-relevant.

---

## 19. Agent Frameworks & Tool Use for Small Models

### 19.1 Agent Distillation (arXiv:2505.17612)

**What**: Distill large language agents into small models with retrieval and code tools.

**Results**:
- 0.5B-3B students competitive with 1.5B-7B with CoT
- First-thought prefix + self-consistent action generation
- Agent-distilled Qwen2.5-1.5B-Instruct available on HuggingFace
- Built on smolagents v1.13.0

**Key techniques**:
1. Log teacher agent trajectories (with tools)
2. SFT on trajectories using TRL
3. Benchmark on factual + mathematical reasoning

**Practical Implications**: HIGHLY RELEVANT. 1.5B distilled agent works — small (1-2B) is feasible. First-thought prefix technique helps small models plan. Retrieval + code tools augment limited parametric knowledge.

### 19.2 smolagents (HuggingFace)

**What**: Lightweight agent library. Core logic in ~1,000 lines of code. Code agents that write actions in Python.

**Features**:
- Model-agnostic (transformers, ollama, OpenAI, Anthropic via LiteLLM)
- Tool-agnostic (MCP servers, LangChain tools, Hub Spaces)
- CodeAgent: writes actions in code (vs JSON tool calls)
- Sandboxed execution (Docker, E2B, Modal, Blaxel)

**For local models**:
```python
from smolagents import TransformersModel
model = TransformersModel(model_id="local-model", max_new_tokens=4096, device_map="auto")
```

**Practical Implications**: Excellent framework for agent deployment. Code agents better than JSON for small models (less formatting overhead). Local transformers model support = direct integration.

### 19.3 2B-Agent (pypi)

**What**: Local-first coding agent for terminal. Designed specifically for small local models (Nemotron 3 Nano 4B, gpt-oss:20b, Qwen family).

**Key insight**: Generic OpenAI-compatible `/v1` shim **degrades small model tool selection**. Talking to Ollama's native `/api/chat` endpoint with fixed 5-tool set works much better.

**Design principle**: All complexity on host side. Model's world never changes — same 5 tools, same native wire format. TUI, plan checklist, task management all rendered around tool loop.

**Practical Implications**: Validates that small models CAN do agentic coding if harness is designed for them. Key lesson: don't use generic OpenAI shim — use native protocol. Keep tool set small and fixed.

### 19.4 AgentFloor (arXiv:2605.00334)

**What**: 30-task benchmark for tool use, 6-tier capability ladder.

**Findings**:
- Small/mid-sized models sufficient for **short-horizon structured tool use**
- Frontier models needed for long-horizon planning
- Strongest open-weight matches GPT-5 on benchmark

**Practical Implications**: Short-horizon tool use is achievable. Focus on structured, well-defined tool interactions. Avoid long multi-step planning tasks.

### 19.5 Agent Strategy for Small Models (1-2B)

**Recommended approach**:
1. **Distill agent behavior** from GPT-4/Claude teacher using agent-distillation framework
2. **Use smolagents** with local TransformersModel for hybrid conv-attention models
3. **Keep tool set small** (5-8 tools max, per 2B-Agent insight)
4. **Use native protocol** not generic OpenAI shim
5. **Code agents** over JSON tool calls (less formatting overhead for small models)
6. **First-thought prefix** for planning (from agent distillation paper)
7. **Focus on short-horizon tasks** (1-3 tool calls, not 10+)

**Realistic capabilities for small (1-2B) agent**:
- ✅ Single-tool file operations (read, write, search)
- ✅ Code execution with feedback
- ✅ Retrieval-augmented QA
- ✅ Simple multi-step workflows (2-3 steps)
- ❌ Long-horizon planning (10+ steps)
- ❌ Complex tool composition
- ❌ Autonomous multi-agent coordination

---

## 20. Tokenizer Deep Dive

### 20.1 BPE (Byte Pair Encoding)

**What**: Greedily merges most frequent byte pairs iteratively. Used by GPT-2/3/4, LLaMA, Qwen.

**Properties**:
- Greedy construction procedure
- Fast encoding
- Well-supported (tiktoken, HF tokenizers)
- **Issue**: Suboptimal for language model pretraining (EMNLP 2020) — recovers subword units that don't align with morphology

### 20.2 Unigram LM (SentencePiece)

**What**: Probabilistic model selecting subword units that maximize corpus likelihood. Used by T5, ALBERT, XLNet.

**Properties**:
- Recovers subword units aligning more closely with morphology
- **Matches or outperforms BPE** across downstream tasks (English and Japanese)
- Better for agglutinative languages (Japanese, Turkish, Korean)
- SentencePiece implementation handles pre-processing well

### 20.3 WordPiece

**What**: Similar to BPE but uses likelihood-based scoring for merges. Used by BERT, DistilBERT.

**Properties**:
- Good for English
- Less commonly used in modern LLMs

### 20.4 BBPE (Byte-level BPE)

**What**: BPE with 256-byte base vocabulary instead of characters. Used by GPT-2, GPT-4.

**Properties**:
- Handles all possible characters
- Reasonable vocabulary size
- Good for multilingual

### 20.5 tiktoken

**What**: OpenAI's fast BPE implementation. Used by GPT-3.5/4.

**Properties**: Extremely fast. Rust-based. Used via `tiktoken.get_encoding("cl100k_base")`.

### 20.6 Tokenizer Choice Impact (arXiv:2310.08754)

**Study**: 24 mono- and multilingual LLMs at 2.6B parameter scale.

**Key findings**:
- Tokenizer choice **significantly impacts** downstream performance and training costs
- Common metrics (fertility, parity) **not always predictive** of model performance
- Multilingual tokenizers (5 European languages) require **3x vocabulary size** vs English
- English-centric tokenizers for multilingual: **68% additional training cost** + performance degradation

### 20.7 Vocabulary Size vs Embedding Size Trade-off (TU Delft)

**Study**: Small transformers (~10M parameters), BPE vs WordPiece vs SentencePiece.

**Findings**:
- Different tokenization strategies have **minimal impact** on model performance
- **Vocabulary size vs embedding size trade-off significantly affects** language understanding and efficiency
- Increasing vocabulary beyond threshold does NOT enhance understanding
- Sweet spot exists — too small = high fertility, too large = wasted embedding parameters

### 20.8 hybrid conv-attention models Tokenizer

**Specs**: Vocab=65536, tied embeddings. Uses hybrid conv-attention models tokenizer from `research/checkpoints/lfm25_tokenizer/`.

**Analysis**:
- 65536 vocab is large for small (1-2B) model (Llama-3 uses 128K, Qwen2.5 uses 151K)
- Tied embeddings: 65536 × 2048 = 134M params for embeddings (11.5% of 1.17B total)
- If untied: 268M params (23%) — significant overhead
- **PIT alternative**: Could reduce embedding parameter overhead while preserving quality

**Recommendations**:
- Current tokenizer likely sufficient for English + common multilingual
- For code: verify tokenizer handles code syntax well (indentation, operators)
- For long context: 65536 vocab with base=1M RoPE is well-tuned
- **Chunked CE critical**: 65536 vocab means logits tensor is large — fused linear CE prevents OOM

---

## 21. Perplexity vs Downstream Performance

### 21.1 The Disconnect

**Traditional view**: Perplexity (PPL) disconnected from downstream task performance. PPL fell out of fashion as evaluation metric.

**Why PPL fails for long-context (arXiv:2410.23771)**:
- PPL **overlooks key tokens** by averaging across all tokens
- Key tokens (essential for long-context understanding) are obscured
- PPL may only reflect **local information modeling**, not long-range dependency

**LongPPL solution**: Focus on key tokens via long-short context contrastive method. Pearson correlation of **-0.96** with long-context benchmarks (vs poor correlation for standard PPL).

### 21.2 Train-Before-Test (arXiv:2507.05195)

**Finding**: Train-before-test restores connection between perplexity and downstream task performance.

- Post-fine-tuning PPL rankings align with post-fine-tuning downstream rankings
- **Even pre-fine-tuning PPL of base model predicts post-fine-tuning downstream performance**
- Model potential is dominated by one latent factor (rank-1 model-score matrix)
- Rankings transfer gracefully across benchmarks with train-before-test

### 21.3 Power Law Relationship (ICLR 2025)

**Finding**: Power law relationship between perplexity and average top-1 error on downstream tasks.

- Can predict downstream performance from PPL using 20x less compute
- Works for aggregate performance (not individual tasks)
- Among models trained on same data distribution

### 21.4 Practical Guidance

**Use PPL for**:
- Comparing models trained on same data distribution
- Tracking training progress (loss curve)
- Quick sanity checks during development

**Don't rely on PPL for**:
- Long-context capability (use LongPPL or needle-in-haystack)
- Cross-family model comparison
- Specific downstream task performance
- Reasoning capability assessment

**Recommended evaluation suite for small (1-2B)**:
1. PPL on held-out validation set (training progress)
2. MMLU (knowledge — target 55-65%)
3. GSM8K (math reasoning — target 75-85%)
4. HumanEval / LiveCodeBench (code — target 40-60%)
5. IFEval (instruction following — target 70-80%)
6. Needle-in-haystack (long context retrieval)
7. MT-Bench with LLM-as-judge (chat quality)

---

## 22. Model Calibration

### 22.1 The Overconfidence Problem

LLMs are often **overconfident** — generate incorrect answers with high confidence. Critical for reliable deployment.

### 22.2 ECE (Expected Calibration Error)

**What**: Measures difference between predicted confidence and actual accuracy. Standard calibration metric.

**Issue**: ECE may be unreliable due to model's poor calibration to human linguistic variation (Ilia & Aziz 2024).

### 22.3 Brier Score

**What**: Mean squared error between predicted probabilities and actual outcomes. Proper scoring rule.

**ConfTuner (NeurIPS 2025)**: Tokenized Brier score as loss function for training LLMs to express confidence verbally.
- Theoretically proven to be proper scoring rule
- Correctly incentivizes model to report true probability of being correct
- Improves calibration across diverse reasoning tasks
- Generalizes to black-box models (GPT-4o)
- Enables downstream gains in self-correction and model cascade

### 22.4 CCPS (EMNLP 2025)

**What**: Calibrating LLM Confidence by Probing Perturbed Representation Stability.

**Results**:
- Reduces ECE by **~55%**
- Reduces Brier score by **21%**
- Increases accuracy by **5 percentage points**
- Works across Llama, Qwen, Mistral (8B-32B)

### 22.5 Sample Consistency Calibration (AAAI 2025)

**What**: Derive confidence from distribution of multiple randomly sampled generations.

**Findings**:
- Consistency-based methods outperform post-hoc approaches
- Intermediate explanations, model scaling, larger sample sizes enhance calibration
- **Instruction-tuning makes calibration more difficult**
- Confidence scores from consistency can enhance model performance

### 22.6 GRACE Benchmark (arXiv:2502.19684)

**What**: Granular benchmark for evaluating model calibration against human calibration.

**Finding**: Although humans are less accurate than models, **humans are generally better calibrated**. Models struggle on GRACE. CalScore metric penalizes confidently-wrong predictions where humans don't know answer.

### 22.7 Calibration Strategy for small (1-2B)

**Simplest effective approach**: Self-consistency
1. Generate N solutions (N=5-10) at temperature 0.7
2. Measure agreement across solutions
3. High agreement = high confidence, low agreement = low confidence
4. No training needed, works immediately

**If training**: ConfTuner with tokenized Brier score. Proper scoring rule, minimal overhead.

**For deployment**: CCPS-style perturbation probing if representation-level calibration needed.

**Warning**: Instruction-tuning degrades calibration. Post-training (DPO/SimPO) may further degrade. Monitor ECE after alignment training.

---

## 23. SGLang vs vLLM Deep Comparison

### 23.1 Architecture Differences

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Cache structure | PagedAttention (block-level, 16 tokens) | RadixAttention (token-level radix tree) |
| Prefix caching | Automatic Prefix Caching (APC) | RadixAttention (automatic, more flexible) |
| Multi-turn | Good with APC | **Better** — auto-discovers partial overlaps |
| Structured output | Supported | **First-class** — compressed FSM for JSON/regex |
| Cold start | ~60s (70B) | ~60s (70B) |
| Compilation | None | None (vs TensorRT-LLM: 25-40 min) |

### 23.2 Performance (March 2026 benchmarks)

| Model | vLLM (tok/s) | SGLang (tok/s) | Delta |
|-------|-------------|---------------|-------|
| Llama 3.1 8B | ~12,500 | ~16,215 | +29% SGLang |
| Llama 3.3 70B (FP8) | ~1,850 | ~1,920 | +4% SGLang |

**Pattern**: SGLang leads on smaller models (29% at 8B). Gap narrows at 70B+ (4%). At 100+ concurrent requests, SGLang's p95 TTFT stays tighter.

### 23.3 When to Choose Which

**Choose SGLang for**:
- Multi-turn conversations with evolving context
- Structured output (JSON, regex, tool calls)
- Agent workflows with branching
- Smaller models (8B and below) where 29% throughput matters
- Unpredictable conversation patterns

**Choose vLLM for**:
- Batch inference on templated prompts
- Broadest model support
- More mature ecosystem and documentation
- Larger models (70B+) where gap is small
- Predictable request patterns

### 23.4 Relevance for Small Models (1-2B)

**SGLang is likely better** for small (1-2B) because:
- 29% throughput advantage at small model scale
- RadixAttention better for multi-turn chat (common use case)
- Structured output first-class (useful for agent/tool use)
- small (1-2B) is firmly in "small model" territory where SGLang wins

**However**: SGLang has limited Windows support. vLLM's Windows support is improving faster. For Windows deployment, vLLM may be more practical despite lower throughput.

**Recommendation**: Use vLLM for Windows development/deployment. Consider SGLang if deploying on Linux or if structured output is critical.

---

## 24. Additional 2025-2026 Research Highlights

### 24.1 Overtraining Scaling Laws (ICLR 2025)

**Study**: 104 models, 0.011B to 6.9B parameters, various training tokens.

**Findings**:
- Scaling laws extrapolate in both over-training amount and parameter count
- Can predict 1.4B/900B-token run (32x over-trained) from 300x less compute
- Power law relates perplexity to downstream task performance
- **For small (1-2B)**: Overtraining (more tokens than Chinchilla-optimal) reduces inference cost. SmolLM2 (1.7B) trained on 11T tokens = heavily overtrained. hybrid conv-attention models trained on 10-12T tokens.

### 24.2 Downstream Scaling Laws Reliability (EMNLP 2025, findings.877)

**Finding**: Predictable downstream scaling occurs only **39% of the time**. Seemingly benign changes to experimental setting can completely change scaling behavior.

**Implication**: Don't over-rely on scaling law predictions for specific tasks. Validate empirically.

### 24.3 S2O: Early Stopping for Sparse Attention (ACL 2026)

**What**: Online permutation + early stopping for sparse attention.

**Results**: On Llama-3.1-8B at 128K context:
- 3.82x MSE reduction at matched sparsity
- 3.31x prefill compute density reduction
- **7.51x attention speedup**, **3.81x end-to-end speedup**

### 24.4 SpenseGPT (arXiv:2606.10445)

**What**: Hybrid 2:4 sparse + dense format for practical pruning. First one-shot pruning to show real B200 end-to-end decoding speedup.

**Results**: 1.2x end-to-end decoding speedup on B200 with FP8, preserving accuracy. On Qwen3-32B and Seed-OSS-36B.

### 24.5 Key Takeaways

The field is moving toward:
1. **Hardware-software co-design** (Blackwell FP4/FP8, sparse tensor cores)
2. **Test-time compute scaling** (o1-style reasoning, budget forcing)
3. **MoE for all scales** (even sub-billion models benefit)
4. **Safety as first-class concern** (especially for SLMs)
5. **Agent capabilities for small models** (distillation + right harness)
6. **Overtraining small models** (SmolLM2: 11T tokens for 1.7B)
7. **Structured generation** (SGLang's RadixAttention + compressed FSMs)

### Reasoning & Cognition
- STaR: arXiv:2203.11321
- HS-STaR: EMNLP 2025 main.282
- Quiet-STaR / Fast Quiet-STaR: EMNLP 2025 findings.1020
- ReST-EM: arXiv:2312.10003
- DeepSeek-R1: arXiv:2501.12948
- Let's Verify Step by Step: OpenAI 2023
- R-PRM: EMNLP 2025 main.679
- PRMBench: ACL 2025 long.1230
- LLM2 (System 2): NAACL 2025 short.15
- Dualformer: ICLR 2025
- ThinkSLM: EMNLP 2025 main.1659
- CRV+CogPO: EMNLP 2025 main.377
- Small Model Learnability Gap: arXiv:2502.12143
- ARC-AGI-2: arXiv:2505.11831
- ARC Prize 2025: arXiv:2601.10904
- ReVISE: ICML 2025
- CRITIC: ICLR 2024
- Graph of Thoughts: AAAI 2024

---

## 25. Chain-of-Thought Training & Self-Taught Reasoning

### 25.1 STaR (Self-Taught Reasoner, arXiv:2203.11321)

**What**: Model generates CoT reasoning for problems, filters by correctness, fine-tunes on correct rationales. Iterates.

**Mechanism**:
1. Given problem with known answer, model generates CoT reasoning
2. If answer correct: add (problem, rationale) to training set
3. If wrong: provide hint (answer), regenerate with hint
4. Fine-tune on accumulated rationales
5. Repeat with improved model

**Results**: Significant improvement on arithmetic, commonsense reasoning. Iterative self-improvement.

### 25.2 HS-STaR (EMNLP 2025, main.282)

**What**: Hierarchical sampling with difficulty estimation and budget reallocation.

**Key insight**: Problems near the **boundary of LLM's reasoning capability** offer significantly greater learning utility than both easy and overly difficult ones.

**Mechanism**:
1. Pre-sampling with reward-guided difficulty estimation
2. Identify boundary-level problems (not too easy, not too hard)
3. Reallocate remaining budget toward high-utility problems
4. Re-sample on boundary problems

**Results**: Significantly outperforms uniform-budget baselines without additional sampling budget.

**Practical Implications**: HIGHLY RELEVANT. Boundary problems maximize learning. For small (1-2B) with limited compute, focusing on "Goldilocks zone" problems is critical. Don't waste samples on trivial or impossible problems.

### 25.3 Quiet-STaR & Fast Quiet-STaR (EMNLP 2025, findings.1020)

**Quiet-STaR**: Generates token-level thought traces (internal "thinking" between tokens). Improves reasoning but incurs substantial inference overhead.

**Fast Quiet-STaR**: Curriculum-learning-based training that gradually reduces thought tokens, enabling model to **internalize abstract and concise reasoning**.

**Fast Quiet-STaR NTP**: Eliminates explicit thought token generation entirely during inference via RL fine-tuning.

**Results**:
- Fast Quiet-STaR consistently outperforms Quiet-STaR under same inference time budget
- NTP variant: **+9% accuracy** on Mistral 7B, **+5.7%** on Qwen2.5 7B
- Maintains same inference latency (no thought tokens at inference)

**Practical Implications**: HIGHLY RELEVANT. "Thinking without thought tokens" = reasoning benefits without inference overhead. For small (1-2B) on consumer GPU (e.g. RTX 5070/4070), can't afford long CoT generation. Internalized reasoning via curriculum + RL is the path.

### 25.4 CARE-STaR (ACL 2025, findings.1116)

**What**: Constraint-aware STaR for instruction following with multiple constraints.

**Mechanism**: Classifies constraints by difficulty, generates different CoTs, sets positive rewards for CoTs beneficial to accuracy, iteratively optimizes.

**Results**: Substantially enhances capability to handle complex instructions, outperforms SFT.

### 25.5 CoT for Long-Context (EMNLP 2025, findings.170)

**What**: Process-supervised framework teaching models to generate high-quality reasoning paths for long-context tasks.

**Key findings**:
- CoT benefits **generalize across most long-context scenarios**
- Benefits **amplify with increasing context length**
- Quality assessment: answer correctness + process reliability (source faithfulness + intrinsic consistency)
- +13.6/+3.8 points on MuSiQue for LLaMA/Qwen over outcome supervision

**Practical Implications**: Important for long-context use cases. CoT + process supervision better than outcome-only for retrieval-heavy tasks.

### 25.6 Minimal Parameter Budget for Reasoning (arXiv:2504.03635)

**What**: Scaling law for minimal parameter budget required for implicit reasoning.

**Finding**: An optimally sized LM can reliably reason over approximately **0.008 bits of information per parameter** at most.

**Implication for small (1-2B)**: 1.17B params × 0.008 bits = ~9.36M bits = ~1.17MB of "reasoning capacity." This is the implicit reasoning ceiling without CoT. Explicit CoT (System 2) extends this by externalizing reasoning to token space.

---

## 26. Reasoning Distillation to Small Models

### 26.1 DeepSeek-R1 Distillation (arXiv:2501.12948)

**What**: Distill reasoning patterns from DeepSeek-R1 (671B MoE) into dense models (1.5B-70B).

**Key finding**: **Distilled reasoning patterns outperform RL-discovered reasoning on small models.** Reasoning patterns from larger models transfer better than what small models can discover through RL alone.

**Released models**: DeepSeek-R1-Distill-Qwen-1.5B, 7B, 14B, 32B; Distill-Llama-8B, 70B.

**DeepSeek-R1-Distill-Qwen-1.5B performance**:
- AIME 2024: 28.9% (vs o1-mini 63.6%)
- MATH-500: 83.9%
- GPQA Diamond: 33.8%
- LiveCodeBench: 17.4%

**Practical Implications**: DIRECTLY APPLICABLE. R1-Distill-Qwen-1.5B proves 1.5B can reason. small models (1-2B) is similar scale. Distill R1's CoT traces into hybrid conv-attention models via SFT. Expected: significant reasoning improvement on math/code.

### 26.2 Small Model Learnability Gap (arXiv:2502.12143)

**Critical finding**: Small models (≤3B) **do NOT consistently benefit** from long CoT reasoning or distillation from larger models. They perform better with **shorter, simpler reasoning chains** that align with their intrinsic learning capacity.

**Mix Distillation solution**: Combine long and short CoT examples, or reasoning from both larger and smaller models.

**Results**: Mix Distillation significantly improves small model reasoning vs training on either data alone.

**Practical Implications**: CRITICAL. Don't just distill R1's long CoTs — they may be too complex for small (1-2B). Use Mix Distillation: combine short CoTs (from smaller models) with long CoTs (from R1). Match reasoning complexity to model capacity.

### 26.3 CRV + CogPO (EMNLP 2025, main.377)

**What**: Critique-Rethink-Verify system for training small reasoning models with cognitive alignment.

**Problem**: Small models have different reasoning capacities and cognitive trajectories than large models. Direct CoT distillation can be ineffective.

**CRV System** (3 agents):
1. **Critique**: Assess CoT quality according to smaller model's cognitive capabilities
2. **Rethink**: Refine CoTs based on critiques
3. **Verify**: Check correctness of refined results

**CogPO**: Cognitive Preference Optimization — aligns reasoning processes with cognitive capacities.

**Results**: Outperforms other methods by large margin on reasoning benchmarks.

**Practical Implications**: HIGHLY RECOMMENDED. Don't blindly distill — adapt CoTs to small (1-2B)'s cognitive capacity first. CRV pipeline: critique R1 CoTs for small (1-2B) → rethink → verify → CogPO training.

### 26.4 Cognivolve Curriculum (arXiv:2505.11643)

**What**: Four-stage easy-to-hard curriculum on GPT-2 (124M) for reasoning emergence.

**Stages**: Lexical matching → multi-step symbolic inference

**Results**:
- Reaches target accuracy in **half the optimization steps** of single-phase baseline
- Activates order-of-magnitude more gradient-salient reasoning heads
- Shifts reasoning heads toward deeper layers
- Higher-entropy attention balancing local and long-range context
- **Order matters**: out-of-order curriculum or optimizer resets fail to reproduce gains

**Caveat**: Final-answer success still lags conventional run by ~30%.

**Practical Implications**: Curriculum works even at 124M. For small (1-2B), four-stage curriculum (simple → complex reasoning) should be even more effective. Progression, not extra compute, drives the effect.

### 26.5 Distillation Strategy for Small Models (1-2B)

**Recommended pipeline**:
1. **Collect R1 CoT traces** for target domains (math, code, reasoning)
2. **Apply CRV system**: Critique traces for small (1-2B) cognitive capacity → Rethink → Verify
3. **Mix Distillation**: Combine adapted long CoTs with short CoTs from smaller models
4. **Curriculum ordering**: Easy → hard reasoning problems
5. **CogPO training**: Align reasoning with cognitive capacity via preference optimization
6. **Optionally**: GRPO with verifiable rewards for further refinement

**Expected outcomes** (based on R1-Distill-Qwen-1.5B):
- MATH-500: ~80% (from ~40-50% baseline)
- GSM8K: ~85% (from ~75%)
- AIME: ~20-30% (from ~5%)
- LiveCodeBench: ~15-20% (from ~10%)

---

## 27. Process Reward Models & Step Verification

### 27.1 Let's Verify Step by Step (OpenAI, 2023)

**What**: Process-supervised reward models (PRMs) provide feedback for each intermediate reasoning step, vs outcome-supervised (ORMs) that only score final result.

**Results**:
- Process supervision **significantly outperforms** outcome supervision on MATH dataset
- Process-supervised model solves **78%** of MATH problems
- Active learning significantly improves efficacy
- Released PRM800K: 800,000 step-level human feedback labels

**Key insight**: Step-level feedback catches reasoning errors that outcome-only feedback misses. A wrong final answer might have a correct first 9 steps and one error in step 10 — ORM can't localize, PRM can.

### 27.2 R-PRM (EMNLP 2025, main.679)

**What**: Reasoning-Driven PRM that activates inherent reasoning to enhance process-level evaluation.

**Innovations**:
1. Use stronger LLMs to generate seed data from limited annotations
2. Self-improvement through preference optimization (no additional annotated data)
3. Inference-time scaling to harness reasoning potential

**Results**:
- +13.9 F1 on ProcessBench, +8.5 F1 on PRMBench over baselines
- +8.6 points accuracy across six challenging math datasets when guiding reasoning

### 27.3 PRMBench (ACL 2025, long.1230)

**What**: Fine-grained benchmark for PRMs. 6,216 problems, 83,456 step-level labels.

**Evaluation dimensions**: Simplicity, soundness, sensitivity.

**Finding**: Current PRMs have **significant weaknesses** in detecting various implicit error types. 25 models tested — all show gaps.

### 27.4 Lessons in PRM Development (ACL 2025, findings.547)

**Key findings**:
- Monte Carlo estimation-based data synthesis yields **inferior** performance vs LLM-as-a-judge and human annotation
- Best-of-N evaluation has biases — need combined response-level + step-level metrics
- **Consensus filtering**: integrate MC estimation with LLM-as-a-judge for best results
- Released new SOTA open-source PRM

### 27.5 Uncertainty-Aware PRMs (arXiv:2502.11250)

**What**: CoT Entropy — novel uncertainty quantification for step-wise verification.

**Problem**: PRMs are proxies for human judgment, susceptible to reward hacking.

**Solution**: Incorporate uncertainty estimates to improve robustness of judge-LM PRMs.

### 27.6 PRM Strategy for Small Models (1-2B)

**Training a PRM for small (1-2B)**:
1. **Data**: Use R1 to generate solutions, label steps via LLM-as-a-judge (not MC estimation)
2. **Consensus filtering**: Combine MC + LLM-as-judge for robust labels
3. **R-PRM approach**: Use reasoning-driven PRM with self-improvement
4. **Uncertainty**: Add CoT Entropy for robustness

**Using PRM at inference**:
1. Generate N candidate solutions with CoT
2. Score each step with PRM
3. Select solution with highest cumulative step scores
4. Or use PRM for tree search (best-first over reasoning steps)

**Memory consideration**: PRM is another model in memory. For limited VRAM (8-16GB):
- Use PRM as scoring function (forward-only, no gradients)
- Or use LLM-as-judge (GPT-4 API) instead of local PRM
- Or train lightweight PRM head on top of hybrid conv-attention models (similar to EAGLE draft head)

---

## 28. Self-Reflection, Self-Correction & Metacognition

### 28.1 Confidence vs Critique Decomposition (ACL 2025, long.203)

**What**: Decomposes self-correction into two capabilities:
- **Confidence**: Being confident about correct answers (not changing them)
- **Critique**: Turning wrong answers to correct

**Finding**: Different models exhibit distinct behaviors — some confident, others critical. **Trade-off exists**: improving one can decline the other.

**Improvement strategy**: Transform SFT data format to improve both capabilities simultaneously. Outperforms vanilla SFT.

**Practical Implications**: Self-correction is tricky for small models. Need to balance confidence (don't change correct answers) with critique (fix wrong ones). Data format matters.

### 28.2 CRITIC (ICLR 2024)

**What**: LLMs self-correct with tool-interactive critiquing. Uses external tools (search, code interpreter) to validate and amend outputs.

**Mechanism**:
1. Generate initial output
2. Use tools to evaluate aspects (search for facts, run code for verification)
3. Revise based on tool feedback
4. Iterate

**Results**: Consistently enhances performance on QA, math, toxicity reduction.

**Key insight**: **External feedback is crucial** for ongoing self-improvement. Intrinsic self-correction without external feedback often degrades performance.

**Practical Implications**: Use code interpreter as external verifier for math/code tasks. Search engine for factual QA. External tools compensate for limited internal knowledge.

### 28.3 Intrinsic Metacognition (EMNLP 2025, main.171)

**What**: LLMs have intrinsic meta-cognition (self-awareness of step errors) but need a good "lens" to access it.

**AutoMeco**: Automated Meta-cognition Evaluation framework for benchmarking lenses.

**MIRA**: Training-free Markovian Intrinsic Reward Adjustment to boost meta-cognition lenses.

**Finding**: Perplexity can reflect answer correctness and serve as lens of meta-cognition, but lacks step-level analysis.

### 28.4 IoRT: Instruct-of-Reflection (NAACL 2025, long.502)

**What**: Dynamic-meta instruction for iterative reflection. Addresses redundant, drift, and stubborn issues in static reflection.

**Mechanism**: Instructor generates dynamic instructions (refresh, stop, select) based on meta-thoughts and self-consistency classifier to guide next reflection iteration.

**Results**: +10.1% average improvement over baselines on math and commonsense reasoning.

**Practical Implications**: Static reflection can hurt small models (drift, stubbornness). Dynamic instruction (knowing when to stop, refresh, or select) is more effective.

### 28.5 ReVISE (ICML 2025)

**What**: Refine via Intrinsic Self-Verification — LLMs self-correct through self-verification without external verifiers.

**Mechanism**:
1. Structured curriculum based on preference learning
2. Tackle self-verification and reasoning correction sequentially
3. Collect failed and successful reasoning paths for preference pairs
4. Confidence-aware decoding at inference

**Results**: Achieves natural test-time scaling by integrating self-verification and correction. Effective across reasoning tasks.

**Practical Implications**: No external verifier needed — self-verification only. Preference learning curriculum is trainable on 12GB. Confidence-aware decoding is inference-time only.

### 28.6 Self-Correction Reality for Small Models

**Key findings from literature**:
1. **Intrinsic self-correction without external feedback often degrades performance** (multiple studies)
2. Small models are more susceptible to degradation — less capacity to identify own errors
3. **External tools** (CRITIC) or **external verifiers** (PRM) are more reliable
4. **ReVISE** shows self-verification can work with proper training curriculum
5. **Confidence-aware decoding** helps — don't correct if confident

**Practical strategy for small (1-2B)**:
- **Tier 1**: External verification (code interpreter for math/code, search for facts)
- **Tier 2**: ReVISE-style self-verification with confidence-aware decoding
- **Tier 3**: IoRT-style dynamic reflection (know when to stop)
- **AVOID**: Naive intrinsic self-correction (likely to degrade)

---

## 29. System 1 / System 2 Dual-Process Reasoning

### 29.1 Dual-Process Theory for LLMs

**System 1 (fast, intuitive)**: Direct token generation, no explicit reasoning. Fast, low compute, but error-prone on complex tasks.

**System 2 (slow, deliberative)**: Chain-of-thought, step-by-step reasoning, self-verification. Slow, high compute, more accurate.

**Key insight (arXiv:2502.12470)**: Human reasoning is a **spectrum**, not binary. LLMs should dynamically shift between modes based on task demands, balancing speed and accuracy.

### 29.2 LLM2 Framework (NAACL 2025, short.15)

**What**: Combines LLM (System 1) with process-based verifier (System 2).

**Mechanism**:
- LLM generates plausible candidates (System 1)
- Verifier provides process-based feedback (System 2)
- Verifier trained with pairwise comparison loss on synthetic process-supervision data

**Results**: Llama3-1B on GSM8K: **50.3 → 57.8** (+7.5). With self-consistency: major@20 **56.2 → 70.2** (+14.0).

**Practical Implications**: DIRECTLY APPLICABLE — tested on 1B model. +7.5 points on GSM8K with verifier. +14 with self-consistency. Lightweight verifier trainable on 12GB.

### 29.3 Dualformer (ICLR 2025)

**What**: Single Transformer with controllable fast/slow modes via randomized reasoning trace training.

**Training**: Different parts of reasoning traces strategically dropped during training.

**Inference modes**:
- **Fast mode**: Output only solution (no reasoning)
- **Slow mode**: Output reasoning + solution
- **Auto mode**: Model decides which mode to engage

**Results**:
- Slow mode: 97.6% optimal rate on maze tasks (vs Searchformer 93.3%), 45.5% fewer reasoning steps
- Fast mode: 80% optimal rate (vs Solutionformer 48%)
- Auto mode: dynamically switches based on task difficulty

**Practical Implications**: Train with randomized reasoning trace dropping. Get both fast and slow modes in one model. Auto mode = adaptive compute allocation. Particularly valuable for small (1-2B) where you can't always afford slow mode.

### 29.4 Snap-Think (REALM 2025)

**What**: Dual-mode mechanism combining System 1 (fast) and System 2 (slow) to break free from reasoning loops (overthinking).

**TVC (Think, Validate, Consensus)**: Multi-agent system detecting overthinking via recursive mental state modeling.

**Results**: GPT-4o on NYT Connections: **98% solve rate** vs CoT's 72%. Maintains semantic grounding and efficiency.

**Practical Implications**: Overthinking/analysis paralysis is a real problem. Need mechanism to detect and break out of reasoning loops. Confidence thresholds + dynamic mode switching.

### 29.5 Fast/Slow/Tool-Augmented Taxonomy (arXiv:2508.12265)

**Two knowledge boundaries**:
1. **Fast/Slow**: Intuitive vs deliberative
2. **Internal/External**: Model parameters vs external tools

**Four quadrants**:
- Fast + Internal: Direct generation (System 1)
- Slow + Internal: Chain-of-thought (System 2)
- Fast + External: Tool-augmented quick responses
- Slow + External: Deliberative reasoning with tool verification

**Key insight**: Effective reasoning requires **adapting strategy to problem demands**. Not all problems need System 2. Not all need tools.

### 29.6 Dual-Process Strategy for Small Models (1-2B)

**Implementation plan**:
1. **Dualformer-style training**: Randomized reasoning trace dropping → fast/slow/auto modes
2. **LLM2-style verifier**: Lightweight PRM as System 2 verifier
3. **Auto mode**: Confidence-based mode selection
   - High confidence → fast mode (direct answer)
   - Low confidence → slow mode (CoT + verification)
4. **Tool integration**: Code interpreter for math/code, search for facts
5. **Overthinking detection**: Confidence threshold + iteration limit

**Expected behavior**:
- Simple questions: Fast, direct answer (~50ms)
- Medium questions: Brief CoT, self-verify (~200ms)
- Hard questions: Full CoT, PRM verification, tool use (~2s)
- Impossible questions: Graceful degradation, abstention

---

## 30. Abstract Reasoning & Fluid Intelligence

### 30.1 ARC-AGI Benchmark Series

**ARC-AGI-1 (2019)**: Few-shot grid-transformation problems. Minimal prior knowledge required. Tests fluid intelligence.

**Performance progression**:
- 2020 winner: 20% (program synthesis)
- 2024 winner: 55.5% (MindsAI team)
- 2025: Opus 4.6 reaches **93.0%** on ARC-AGI-1
- Cost fell **390x** in one year (o3's $4,500/task → GPT-5.2's $12/task)

**ARC-AGI-2 (arXiv:2505.11831)**: Upgraded with more cognitively complex tasks.
- Frontier models: ~68.8%
- Humans: near-perfect
- ARC Prize 2025 top score: 24% (Kaggle, 1,455 teams)

**ARC-AGI-3 (preview)**: Interactive reasoning challenges requiring exploration, planning, memory, goal acquisition, alignment.
- Best model: 13%
- Humans: near-perfect

### 30.2 LLM Fluid Intelligence Deficiency (NAACL 2025, long.423)

**Three major limitations identified**:
1. **Limited ability for skill composition** — can't combine learned skills in novel ways
2. **Unfamiliarity with abstract input formats** — struggle with non-textual representations
3. **Intrinsic deficiency of left-to-right decoding** — autoregressive generation limits planning

**Implication**: These are **fundamental architectural limitations**, not just scale issues. Even trillion-scale models struggle with ARC.

### 30.3 Refinement Loops (ARC Prize 2025, arXiv:2601.10904)

**Defining theme of 2025**: Emergence of refinement loops — per-task iterative optimization guided by feedback signal.

**Types**:
- **Evolutionary program synthesis**: Iteratively improve programs
- **Application-layer refinements**: Commercial AI systems with feedback loops
- **Weight-space refinement**: Zero-pretraining deep learning with small networks (7M params competitive)

**Key finding**: ARC Prize 2025 winners needed **hundreds of thousands of synthetic examples** to reach 24% on ARC-AGI-2. **Reasoning remains knowledge-bound** — not pure fluid intelligence.

### 30.4 Knowledge-Augmented ARC (arXiv:2505.17482)

**KAAR (Knowledge Augmentation for Abstract Reasoning)**:
- Encodes core knowledge priors in ontology
- Three hierarchical levels based on dependencies
- Progressively augments priors at each level
- Stage-wise reasoning reduces interference from irrelevant priors

**RSPC (Repeated-Sampling Planning-aided Code Generation)**: Highest test accuracy among candidate solvers.

**Finding**: Knowledge augmentation + structured reasoning improves ARC performance. Pure reasoning without knowledge priors is insufficient.

### 30.5 Abstract Reasoning for small (1-2B)

**Reality check**: small (1-2B) models will score very low on ARC-AGI (likely <5% on ARC-AGI-2). This is expected — even 70B models struggle.

**What small (1-2B) CAN do**:
- **Pattern recognition**: Identify simple visual/textual patterns
- **Analogical reasoning**: Simple A:B::C:D analogies
- **Rule induction**: Infer simple rules from examples (with CoT)
- **Compositional reasoning**: Combine 2-3 simple rules (with scaffolding)

**What small (1-2B) CANNOT do**:
- Complex multi-step abstract reasoning (ARC-AGI-2+)
- Novel skill composition without prior knowledge
- Planning over 5+ steps without external support

**Strategy**: Don't target ARC-AGI. Focus on practical reasoning (math, code, logical deduction) where knowledge + CoT suffices. Use tools for verification.

---

## 31. Tree-of-Thought, Graph-of-Thought & Planning

### 31.1 Tree of Thoughts (ToT)

**What**: Tree search over LLM "thoughts" — each node is a partial solution, branches are reasoning steps.

**Mechanism**:
1. Decompose problem into steps
2. Generate multiple candidate thoughts at each step
3. Evaluate thoughts (LLM-as-judge or heuristic)
4. Expand most promising (beam search / best-first)
5. Backtrack if needed

**Policy-Guided ToT (TMLR 2025)**: Levin Tree Search adapted to ToT. LM probabilities as heuristic. Theoretical bound on states expanded. Consistently achieves higher accuracy under fixed LM query budget.

**Practical Implications**: ToT requires multiple LLM calls per step — expensive for small (1-2B). Use with small beam width (2-3) and shallow depth (3-4 levels). Best for problems with verifiable intermediate states.

### 31.2 Graph of Thoughts (GoT, AAAI 2024)

**What**: Generalizes ToT to arbitrary graphs. Thoughts are vertices, dependencies are edges.

**Advantages over ToT**:
- Combine arbitrary thoughts into synergistic outcomes
- Distill essence of whole networks
- Enhance thoughts via feedback loops (cycles)
- Closer to human thinking (recurrence, complex networks)

**Results**: Sorting quality +62% over ToT, costs -31%.

### 31.3 ARIES (arXiv:2502.21208)

**What**: Autonomous reasoning with LLMs on interactive thought graph environments. Policy LLM agent drives transformations.

**Results**: +29% accuracy on HumanEval vs static schedules. -35% inference costs. No search requirements.

**Key insight**: Another LLM as policy agent (not the reasoning LLM) can dynamically adapt problem-solving strategy.

### 31.4 Visual Thinking (arXiv:2503.11790)

**What**: LMMs reason through self-generated conceptual diagrams within Graph-of-Thought framework.

**Results**: GPT-4o on Blocksworld: **35.5% → 90.2%**. Outperforms o1-preview by 16 points on Floor Tiles.

**Practical Implications**: Not directly applicable (requires multimodal). But concept of externalizing reasoning to non-text representations is valuable. For small (1-2B): structured output formats (tables, pseudocode) as "diagrams."

### 31.5 Planning Strategy for small (1-2B)

**Realistic planning capabilities**:
- ✅ 2-3 step plans with CoT
- ✅ Best-of-N selection with simple verifier
- ✅ ToT with beam width 2-3, depth 3
- ⚠️ GoT with feedback loops (complex, may not converge)
- ❌ Deep search (10+ levels)
- ❌ Multi-agent policy coordination

**Recommended approach**:
1. **Simple problems**: Direct generation (System 1)
2. **Medium problems**: CoT + self-consistency (N=5, majority vote)
3. **Hard problems**: ToT with PRM-guided beam search (width 2-3, depth 3-4)
4. **Verifiable problems**: Generate → execute → feedback → refine (CRITIC-style)

---

## 32. Reasoning Training Roadmap for Small Models

### Phase 1: Foundation (SFT on adapted CoT data)
1. Collect R1 CoT traces for math, code, reasoning
2. Apply CRV system: Critique for small (1-2B) capacity → Rethink → Verify
3. Mix Distillation: Combine short + long CoTs
4. Curriculum ordering: Easy → hard
5. Train with SFT + chunked CE + Liger kernels

### Phase 2: Preference Optimization
1. Generate multiple solutions per problem
2. Score with PRM or LLM-as-judge
3. Train with SimPO (1 model, memory-efficient) or CogPO (cognitive alignment)
4. Focus on boundary-difficulty problems (HS-STaR principle)

### Phase 3: RL with Verifiable Rewards
1. GRPO with math/code verification rewards
2. Group size 8-16 for limited VRAM (8-16GB)
3. OM-GRPO masking to prevent answer hacking
4. Start with high-reward examples, progress to harder (curriculum)

### Phase 4: System 2 Integration
1. Train lightweight PRM verifier (LLM2-style)
2. Dualformer-style randomized trace training (fast/slow/auto modes)
3. Confidence-aware decoding for mode selection
4. Tool integration (code interpreter, search)

### Phase 5: Test-Time Scaling
1. Self-consistency (N=5-10, majority vote)
2. Budget forcing for extendable thinking
3. ToT with PRM-guided beam search (when needed)
4. Overthinking detection (Snap-Think style)

### Expected Performance Trajectory

| Phase | MATH-500 | GSM8K | AIME | HumanEval |
|-------|----------|-------|------|-----------|
| Baseline | ~40% | ~75% | ~5% | ~35% |
| Phase 1 (SFT) | ~70% | ~82% | ~15% | ~50% |
| Phase 2 (PrefOpt) | ~75% | ~85% | ~20% | ~55% |
| Phase 3 (GRPO) | ~80% | ~87% | ~25% | ~58% |
| Phase 4 (Sys2) | ~83% | ~88% | ~28% | ~60% |
| Phase 5 (TTC) | ~85% | ~89% | ~30% | ~62% |

*Estimates based on R1-Distill-Qwen-1.5B trajectory and literature.*

---

### Neurosymbolic & Causal Reasoning
- LoCo LLM: ICLR 2025
- Neurosymbolic Program Synthesis: UT Austin 2025 survey
- Neurosymbolic AI survey: IJCAI 2025.1195
- Sound & Complete Neurosymbolic: PMLR v284, 2025
- LINC: EMNLP 2023
- Logic-LM: EMNLP 2023
- CounterBench: arXiv:2502.11008
- GEAR: arXiv:2509.24096
- Causal Parrots: NLP4DH 2025.29
- CauSciBench: NeurIPS 2025 Workshop
- SimuRA: arXiv:2507.23773
- SWAP: ACL 2025 long.1540
- CoEx: EMNLP 2025 findings.1179

### Analogical & Inductive Reasoning
- KnowledgePrompts: COLING 2025 main.268
- Prototypical to Relational: INLG 2025 main.28
- Curious Case of Analogies: AAAI 2026 v40i37.40414
- SAL: arXiv:2502.00996
- AnaScore: NAACL 2025 long.54
- Patterns Over Principles: ACL 2025 findings.1006
- Induction Heads: NAACL 2025 findings.283
- CoT Hurts Induction: NeurIPS 2025
- Model Prior in Induction: EMNLP 2025 main.534

---

## 33. Neurosymbolic & Hybrid Reasoning

### 33.1 The Neurosymbolic Paradigm

**Core idea**: Combine neural networks (pattern recognition, learning from data) with symbolic AI (logic, rules, verifiable reasoning). Addresses LLM weaknesses in logical consistency, verifiability, and compositional generalization.

**Three integration patterns** (IJCAI 2025 survey):
1. **Symbolic → LLM**: Inject symbolic knowledge into LLM training (constraints, rules)
2. **LLM → Symbolic**: LLM generates symbolic representations for external solvers
3. **LLM + Symbolic**: Tight integration, LLM and solver co-process

### 33.2 LoCo LLM (ICLR 2025)

**What**: Logically Consistent LLMs via neuro-symbolic integration. Teaches LLM to be logically consistent with external facts and rules.

**Mechanism**:
- Compile logical constraints (propositional logic formulas) into circuits
- Use semantic loss to encourage model to allocate probability only to factual/consistent outputs
- Fine-tune base LLM according to knowledge base of facts and rules

**Results**:
- Improves self-consistency even with limited fine-tuning data
- Combines multiple logical constraints principledly
- **Extrapolates to unseen but semantically similar knowledge** — systematic generalization

**Practical Implications**: Semantic loss from compiled logic circuits can enforce consistency without external tools at inference. Useful for factual QA, knowledge graph reasoning.

### 33.3 LINC (EMNLP 2023)

**What**: Neurosymbolic approach combining LLMs with first-order logic provers.

**Mechanism**: LLM translates natural language reasoning into formal logic, external prover verifies.

### 33.4 Logic-LM (EMNLP 2023)

**What**: Empowers LLMs with symbolic solvers for faithful logical reasoning.

**Mechanism**: LLM generates symbolic representation → symbolic solver executes → result translated back.

**Key insight**: Delegate hard logical reasoning to external solvers. LLM handles translation, solver handles logic.

### 33.5 Sound & Complete Neurosymbolic Reasoning (PMLR 2025)

**What**: Directly integrates LLM into interpretation function of formal semantics for paraconsistent logic.

**Properties**: Preserves underlying logic's **soundness and completeness**. LLM provides broad-coverage parametric knowledge while logic guarantees correctness.

**Practical Implications**: Theoretical framework for neurosymbolic reasoning that leverages LLM knowledge while maintaining formal guarantees.

### 33.6 Neurosymbolic Program Synthesis (UT Austin 2025 Survey)

**What**: Survey of program synthesis combining symbolic primitives with neural components.

**Advantages over end-to-end deep learning**:
- **Reliability**: Programs can be verified
- **Interpretability**: Programs are readable
- **Verifiability**: Formal guarantees possible
- **Compositionality**: Programs compose naturally

**Methods**: Symbolic search + gradient-based optimization. Programs induced using combination of both.

**Practical Implications**: For ARC-style tasks, neurosymbolic program synthesis outperforms pure neural approaches. LLMs can guide search, symbolic methods verify.

### 33.7 Neurosymbolic Strategy

**For reasoning tasks requiring logical guarantees**:
1. **LLM → Solver pipeline** (Logic-LM style): LLM translates, external solver verifies
2. **Semantic loss training** (LoCo style): Compile constraints, train with semantic loss
3. **Program synthesis** (neurosymbolic): LLM proposes program, symbolic search refines
4. **Paraconsistent integration**: LLM as interpretation function in formal logic

**When to use neurosymbolic**:
- ✅ Logical reasoning (syllogisms, constraint satisfaction)
- ✅ Knowledge graph reasoning
- ✅ Mathematical proofs (with formal verifier)
- ✅ Rule-based systems (legal, medical protocols)
- ⚠️ Commonsense reasoning (harder to formalize)
- ❌ Creative generation (no logical constraints needed)

---

## 34. Causal, Counterfactual & Abductive Reasoning

### 34.1 Pearl's Ladder of Causation

Three levels of causal reasoning:
1. **Associational**: What is? (correlation, observation)
2. **Interventional**: What if I do? (do-calculus, experimentation)
3. **Counterfactual**: What if I had done differently? (retrospective)

**LLM performance**: Good at associational, poor at interventional, very poor at counterfactual.

### 34.2 CounterBench (arXiv:2502.11008)

**What**: Benchmark for counterfactual reasoning with formal rules (not just commonsense).

**Design**: 1K counterfactual questions with varying difficulty, diverse causal graph structures, multiple nonsensical name variants (to prevent memorization).

**Results**: Most LLMs perform at **random guessing level** on counterfactual reasoning.

**CoIn solution**: Guides LLMs through iterative reasoning and backtracking to explore counterfactual solutions. Significantly improves performance.

**Practical Implications**: Counterfactual reasoning is a fundamental weakness. Even large models struggle. CoIn-style iterative backtracking helps but doesn't solve it.

### 34.3 Causal Parrots to Prophets (NLP4DH 2025)

**What**: Evaluates LLMs across Pearl's three levels using CLadder dataset.

**Techniques tested**:
- CoT, Self-consistency (SC), CausalCoT
- **CausalToT** (new): Causal Tree of Thoughts
- **CausalPoT** (new): Causal Program of Thoughts

**Results**:
- Larger models more robust against perturbations
- All LLMs struggle with counterfactual reasoning
- CausalToT and CausalPoT **significantly improve** performance over existing techniques
- Hybrid approaches (LLM + formal reasoning) mitigate limitations

**Key insight**: "Causal parrots" — LLMs often reproduce causal patterns from training data without genuine causal understanding. Formal reasoning frameworks help.

### 34.4 CauSciBench (NeurIPS 2025 Workshop)

**What**: Comprehensive benchmark for causal inference in scientific research.

**Scope**: Complete causal analysis pipeline — problem formulation, variable selection, method choice, statistical model implementation, result interpretation.

**Results**:
- OpenAI-o3 (best): 53.0% MRE on real datasets
- Synthetic datasets: 6.2% MRE
- Textbook: 30.6% MRE
- **Huge gap between synthetic and real-world** causal inference

### 34.5 GEAR: Abductive Reasoning (arXiv:2509.24096)

**What**: General Evaluation for Abductive Reasoning — generating plausible hypotheses to explain observations.

**Three metrics**:
1. **Consistency**: Hypothesis correctly explains observations
2. **Generalizability**: Hypothesis makes meaningful predictions on unseen inputs
3. **Diversity**: Hypotheses cover many distinct predictions/patterns

**Results**: 9 LLMs tested on 4 benchmarks, 50,340 candidate hypotheses. GEAR reveals differences obscured by gold-answer evaluation.

**Momentum-based curriculum**: Dynamically adjusts training data by learning velocity. Improves all three objectives without gold-label supervision.

**Practical Implications**: Abductive reasoning (hypothesis generation) is key for scientific discovery. GEAR provides scalable, label-free evaluation.

### 34.6 RECV: Deductive vs Abductive (ACL 2025 findings.1059)

**What**: Benchmark for claim verification assessing deductive and abductive reasoning.

**Results**:
- LLMs can address **deductive reasoning** problems
- LLMs **consistently fail** at abductive reasoning
- Rationale generation not always beneficial
- Generated rationales semantically similar to humans, especially in deductive cases

### 34.7 Implications for Reasoning Architecture

**Causal reasoning hierarchy (difficulty)**:
1. Associational — LLMs handle well
2. Deductive — LLMs handle with CoT
3. Interventional — LLMs struggle, need formal frameworks
4. Abductive — LLMs fail, need hypothesis generation frameworks
5. Counterfactual — LLMs fail badly, need formal causal models

**Strategies**:
- **CausalPoT**: Translate causal reasoning to programs, execute formally
- **CausalToT**: Tree search over causal hypotheses
- **CoIn**: Iterative backtracking for counterfactuals
- **Neurosymbolic**: External causal solvers (do-calculus engines)
- **GEAR curriculum**: Train abductive reasoning with momentum-based curriculum

---

## 35. Analogical Reasoning & Transfer

### 35.1 Proportional Analogies (COLING 2025, main.268)

**What**: A:B::C:D analogy completion. 15K MCQA dataset.

**Results**: Best model achieves only **55% accuracy** despite extensive training data.

**Knowledge enhancement types**:
- **Exemplar**: Similar examples (least helpful)
- **Structured**: Knowledge graphs, ontologies
- **Targeted**: Specific relational knowledge (most helpful)

**Key finding**: Targeted knowledge about the specific relationship helps more than general examples or structured knowledge.

### 35.2 Relational vs Prototypical (INLG 2025, main.28)

**What**: Complex analogy benchmark with multiple plausible answers (not single ground truth).

**Three measures**:
1. Ranked relational overlap
2. Context embedding similarity
3. Prototypicality

**Results**: GPT-4 performs well on embedding-based and prototypicality measures but **consistently underperforms on fine-grained relational mappings**.

**Key insight**: Surface-level semantic fluency ≠ structured relational reasoning. LLMs can identify similar things but struggle with precise relational structure mapping.

### 35.3 Curious Case of Analogies (AAAI 2026)

**What**: Investigates analogical reasoning mechanisms in LLMs using proportional and story analogies.

**Three findings**:
1. **LLMs encode relationships** between analogous entities — attributive and relational info propagates through mid-upper layers in correct cases
2. **LLMs struggle to apply relational info to new entities** — unlike humans. Strategic patching of hidden representations at critical positions can help
3. **Successful analogy = strong structural alignment**. Failures show degraded/misplaced alignment

**Practical Implications**: Analogical reasoning is **emerging but limited**. Representation patching at critical layers could improve transfer. Mid-upper layers are key for relational processing.

### 35.4 SAL: Self-supervised Analogical Learning (arXiv:2502.00996)

**What**: Train models to transfer high-quality symbolic solutions from solved cases to rare/unfamiliar cases.

**Mechanism**:
1. Extract abstract/symbolic solutions from cases model can solve
2. Train model to apply same solutions to similar but unfamiliar cases
3. Self-supervised — no external teacher needed

**Results**: +2% to +20% on StrategyQA, GSM8K, HotpotQA. More generalizable and controllable.

**Practical Implications**: HIGHLY RELEVANT for small models. Self-supervised analogical transfer improves reasoning consistency without external data. Abstract solution transfer = better generalization.

### 35.5 AnaScore (NAACL 2025, long.54)

**What**: Automatic metric for semantic parallelism in proportional analogies.

**Finding**: Formally explainable examples are more beneficial for analogical reasoning. Ambiguous analogies hinder inference. Positive correlation between analogy quality and model performance.

### 35.6 Patterns Over Principles (ACL 2025, findings.1006)

**What**: Evaluates LLM inductive reasoning under noisy observations.

**Robust Rule Induction task**: Infer rules from data with noisy examples.

**Results**:
- LLMs exhibit **instability under noise** (0 accuracy change but only 70% consistency)
- **Reliance on memorized patterns over genuine abstraction**
- Susceptibility to hypothesis drift and pattern overfitting
- SRR (Sample-steered Rule Refinement) helps via observation diversification + execution-guided feedback

**Key insight**: LLMs are "pattern matchers" not "principle extractors." Noise reveals this — slight perturbation causes completely different rules despite similar accuracy.

### 35.7 Induction Heads (NAACL 2025, findings.283)

**What**: Induction heads as essential mechanism for in-context learning pattern matching.

**Results**: Ablating induction heads causes:
- **~32% performance decrease** on abstract pattern recognition
- Few-shot ICL drops to zero-shot levels on NLP tasks

**Practical Implications**: Induction heads are the mechanistic basis for ICL. Understanding which heads are induction heads enables targeted interventions.

### 35.8 CoT Can Hurt Inductive Reasoning (NeurIPS 2025)

**Critical finding**: Chain-of-thought reasoning can **degrade inductive performance**. Large Reasoning Models often **underperform non-reasoning counterparts** on inductive tasks.

**Three failure modes**:
1. Incorrect sub-task decomposition
2. Incorrect sub-task solving
3. Incorrect final answer summarization

**Solution**: Structured interventions that adapt CoT generation based on identified failure types. Improve accuracy without retraining.

**Key insight**: More reasoning steps ≠ better reasoning. Structure matters more than length. For inductive tasks, CoT can amplify errors.

### 35.9 Model Priors Dominate Induction (EMNLP 2025, main.534)

**Finding**: In real-world inductive reasoning, hypothesis generation is **primarily driven by model's inherent priors**, not in-context demonstrations. Removing demonstrations causes minimal loss.

**Implication**: LLMs don't truly "induce" from examples — they retrieve pre-existing hypotheses. The examples just activate existing knowledge.

---

## 36. World Models & Model-Based Reasoning

### 36.1 LLMs as World Models

**Concept**: A world model predicts how actions affect environments. LLMs can serve as world models by:
1. **Precondition prediction**: Determining if action is applicable in current state
2. **Effect prediction**: Predicting resulting state after action

**COLING 2025 (main.503)**: Fine-tuning separate LLMs for precondition and effect prediction. Generated knowledge aligns with human understanding. Supports creation of action chains for planning.

### 36.2 SimuRA (arXiv:2507.23773)

**What**: Simulative Reasoning Architecture — LLM-based world model for planning via simulation.

**Mechanism**:
1. Policy module proposes potential actions
2. World model **simulates outcomes** of proposed actions
3. Select action based on simulated results

**Results**:
- Web browsing: flight search success 0% → **32.2%**
- World-model planning: **+124%** over autoregressive planning
- Consistent advantage across tasks

**Practical Implications**: Mental simulation (world model) is a fundamentally different reasoning paradigm from autoregressive generation. LLMs can simulate outcomes before acting.

### 36.3 SWAP: Structure-Aware Planning (ACL 2025, long.1540)

**What**: Combines structured knowledge representation (entailment graphs) with learned planning.

**Components**:
- **Policy model**: Proposes candidate expansions
- **World model**: Predicts structural updates (multiple alternatives)
- **Discriminator**: Re-ranks based on plausibility
- **Diversity-based Modelling**: Samples from remaining probability mass
- **Contrastive Ranking**: Directly compares candidates within prompts

**Results**: Significantly improves upon base models on math, logical reasoning, and coding. Consistently outperforms existing reasoning methods.

**Practical Implications**: Entailment graphs + world model = structured, verifiable reasoning. Better than pure CoT for complex multi-step tasks.

### 36.4 External World Models (AgentScen 2025)

**What**: LLM constructs and refines explicit external world model (state transition function).

**Mechanism**:
1. LLM generates initial state transition function
2. Function refined using feedback from environment interactions
3. Test cases accumulated from past experiences
4. World model construction treated as program synthesis

**Results**: Perfect accuracy on Blocksworld. Significantly outperforms baselines on planning success and LLM query efficiency.

**Key insight**: External world model (as code) is more reliable than LLM's internal implicit world model. Can be debugged, verified, and reused.

### 36.5 CoEx: Co-evolving World Model (EMNLP 2025, findings.1179)

**What**: Hierarchical agent where world model co-evolves with exploration.

**Problem**: LLM's static internal world model becomes misaligned with true world state over time.

**Solution**: Hierarchical state abstraction + neurosymbolic belief state (textual inferences + code-based symbolic memory). Continuously incorporates subgoal experiences.

**Practical Implications**: Static world models degrade. Need dynamic updating. Neurosymbolic belief state (text + code) is more maintainable than pure neural.

### 36.6 World Model Architecture for LLMs

**Approaches**:
1. **Internal (implicit)**: LLM's parametric knowledge as world model (limited, static)
2. **Internal (explicit)**: Train LLM to output state predictions (SimuRA)
3. **External (symbolic)**: LLM generates code-based world model (External WM)
4. **Hybrid (neurosymbolic)**: Text inferences + code memory (CoEx)

**For small models**: External/hybrid approaches better. Small models lack parametric knowledge for reliable internal world models. Code-based state transitions are verifiable and debuggable.

### 36.7 Model-Based Reasoning Applications

**Planning**:
1. Generate candidate actions
2. Simulate each with world model
3. Select action with best predicted outcome
4. Execute, observe, update world model
5. Repeat

**Counterfactual reasoning**:
1. Current state + alternative action
2. World model simulates alternative trajectory
3. Compare outcomes

**Scientific reasoning**:
1. Hypothesis generation (abductive)
2. World model simulates experiment
3. Compare prediction with observation
4. Update hypothesis

---

## 37. Reasoning Type Taxonomy & Capability Assessment

### 37.1 Reasoning Types Ranked by LLM Difficulty

| Reasoning Type | LLM Capability | Best Technique | Small Model Feasibility |
|---------------|---------------|----------------|------------------------|
| Deductive | Good | CoT | ✅ Yes |
| Associational | Good | Direct | ✅ Yes |
| Inductive (clean data) | Moderate | ICL + hypothesis search | ⚠️ Limited |
| Inductive (noisy data) | Poor | SRR, structured CoT | ❌ No |
| Analogical (surface) | Moderate | Targeted knowledge | ⚠️ Limited |
| Analogical (relational) | Poor | Representation patching | ❌ No |
| Abductive | Poor | GEAR curriculum | ❌ No |
| Causal (interventional) | Poor | CausalPoT | ❌ No |
| Counterfactual | Very Poor | CoIn, formal solvers | ❌ No |
| Abstract (ARC-AGI) | Very Poor | Refinement loops | ❌ No |

### 37.2 What Small Models (1-2B) Can Realistically Do

**Strong capabilities**:
- Deductive reasoning with CoT (math, logic puzzles)
- Pattern matching / ICL
- Code generation and execution
- Retrieval-augmented QA
- Simple planning (2-3 steps with tools)

**Moderate capabilities** (with scaffolding):
- Inductive reasoning from clean examples
- Surface-level analogies
- Self-correction with external verification
- Best-of-N selection

**Weak capabilities** (even with techniques):
- Counterfactual reasoning
- Abductive hypothesis generation
- Relational analogical mapping
- Abstract reasoning (ARC-AGI)
- Causal intervention reasoning

### 37.3 Scaffolding Strategies

**To extend small model reasoning**:
1. **External tools**: Code interpreter (deductive), search (factual), formal solver (logic)
2. **Neurosymbolic**: LLM translates → external solver verifies
3. **World models**: External state simulation for planning
4. **Self-consistency**: Multiple samples + majority vote
5. **PRM verification**: Step-by-step correctness checking
6. **Curriculum training**: Easy → hard progression
7. **Analogical transfer (SAL)**: Transfer solutions from solved to unsolved cases
8. **Budget forcing**: Extend thinking time for hard problems

### 37.4 Key Insights from Reasoning Research

1. **Reasoning ≠ pattern matching**: LLMs often retrieve patterns rather than reason (Model Priors Dominate)
2. **CoT can hurt**: More steps can amplify errors, especially for inductive reasoning
3. **Structure matters more than length**: Well-structured short reasoning > long unstructured CoT
4. **External verification is critical**: Intrinsic self-correction often degrades performance
5. **Distillation > RL for small models**: R1 distillation outperforms RL-discovered reasoning
6. **Match complexity to capacity**: Small models need shorter, simpler reasoning chains
7. **Boundary problems maximize learning**: Not too easy, not too hard (HS-STaR)
8. **Internalized reasoning possible**: Fast Quiet-STaR shows thinking without thought tokens
9. **World models enable planning**: Mental simulation > autoregressive generation
10. **Neurosymbolic integration provides guarantees**: Logic + neural = verifiable reasoning

---

*Document compiled August 2026. Sources: 150+ research papers from 2023-2026, verified benchmark data, third-party consumer GPU benchmarks.*

---

## 38. Consumer Blackwell (sm_120) Specifics — Native Windows

### 38.1 Triton Consumer Blackwell Patch (sm_120a → sm_120)

**Problem**: `triton-windows==3.4.0.post21` generates `sm_120a` PTX targets for consumer Blackwell (RTX 5070/5080/5090), but consumer Blackwell has NO tensor memory (tcgen05). This causes `illegal memory access` in EVERY Triton kernel, rendering `torch.compile`, Liger Kernel, and custom Triton kernels completely unusable.

**Fix** (from upstream PR #9734, which was reverted): three changes to `triton/backends/nvidia/compiler.py`:
1. `sm_arch_from_capability` only adds "a" suffix for `90 <= capability < 120`
2. PTX `.target` regex handles the "a" suffix
3. `make_ttgir` pipeline routes sm_120 away from tensor memory passes

After patching, basic Triton kernels work and `torch.compile` delivers 13,240 tok/s (unchanged from pre-patch on working paths). Clear `~/.triton/cache` after patching so kernels recompile with correct `sm_120` target.

**Scope**: The bug only manifests on consumer Blackwell (sm_120). Datacenter Blackwell (sm_100a, B100/B200) and Hopper (sm_90a) are unaffected.

### 38.2 Gigatoken — Rust BPE Tokenizer

~1000x faster than HuggingFace tokenizers for bulk pre-tokenization. A Rust BPE tokenizer with Python bindings that encodes text at GB/s (24.53 GB/s on 144-core EPYC, 8.79 GB/s on M4 Max). Supports 23 tokenizer families including GPT-2, Llama 3/4, Qwen 2/2.5/3, DeepSeek V3/R1, GLM 4/5, Phi-4, and Gemma.

Speedup comes from SIMD-optimized pre-tokenization (replacing regex engines), minimized branching, and efficient caching of pre-token mappings. Critical for pre-training pipelines where 100M+ tokens must be tokenized offline before training begins.

**Win11**: Ships as a pip wheel (`gigatoken==0.9.0`) with pre-compiled Rust binaries for Windows. No WSL2 required. Verified on RTX 5070 box: ~35x speedup in compatibility mode (HFCompat wrapper), ~17x in native batched mode (`encode_batch_list`). Note: `encode_files` (disk-spill API) returns a flat token stream with no document boundaries, making per-document EOS insertion impossible — use `encode_batch_list` for streaming pipelines instead.

### 38.3 Native FP8 on Consumer Blackwell

RTX 5070 (sm_120, consumer Blackwell) supports FP8 tensor core matmul via `torch._scaled_mm`. Benchmarked at 2.02x faster than BF16 for a [2048, 1024] × [151680, 1024] matmul (4.65 ms vs 9.40 ms). FP8 elementwise ops are NOT supported (FP8 is matmul-only). Full FP8 training integration requires per-tensor scaling factors and vocab dimension padding to multiples of 16. FP8 is most valuable for inference (2x throughput, 50% weight memory) rather than training (stability concerns).

`torch.float8_e4m3fn` and `torch.float8_e5m2` dtypes are available in PyTorch 2.8.0+cu128. `torch._scaled_mm` works on sm_120 without any patches. Matmul dimensions must be divisible by 16 for FP8 (pad vocab from 151665 to 151680).

### 38.4 Liger Kernel v0.8.1 Blackwell Status

Liger v0.8.1 (released July 2026) added Blackwell-specific hardware gating for cross-entropy and SwiGLU tuning, plus an experimental CuTe DSL backend (`LIGER_KERNEL_IMPL=cutedsl`) for Blackwell / B200. The CuTe DSL cross-entropy scaffolding targets datacenter Blackwell (sm_100a) and requires the `cutlass` Python package.

On consumer Blackwell (sm_120, RTX 5070), the Triton CE kernel still crashes even after the sm_120a compiler patch — there are additional Triton issues beyond the "a" suffix. Liger is left installed (`pip install --no-deps liger-kernel`) for future use but is NOT wired into the ForgeAI training loop. The pure-PyTorch Chunked CE (section 12.3) serves as the working alternative.
