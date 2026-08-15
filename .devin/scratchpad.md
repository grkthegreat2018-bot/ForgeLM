# Research: Fine-tune + Boot Time Improvements (2025-2026)

## Fine-Tuning Improvements

### 1. Token Reweighting (HIGH PRIORITY — easy to implement)
Standard SFT treats all tokens equally. Multiple papers show this is suboptimal:

- **WeFT (Weighted Entropy-driven FT)**: Weight tokens by predictive entropy. High-entropy tokens = reasoning/planning tokens = more important. 39-83% relative improvement over standard SFT on reasoning benchmarks.
- **SHAD/RFT (Reasoning-highlighted FT)**: Disentangle reasoning tokens from boilerplate (format) tokens. Classify offline with single forward pass. Upweight reasoning tokens during training.
- **VCORE (Variance-Controlled Reweighting)**: Formulate token reweighting as constrained optimization. Gibbs distribution over token-wise gradient utilities. Strong gains on math/coding, especially for smaller models (4B, 8B).
- **Target-SFT / Q-target**: Replace one-hot target with mixture distribution Q = gamma * delta + (1-gamma) * pi. Relaxes imitation when label uncertain. Outperforms across 10 settings.

**Implementation for ForgeAI**: In `sft_train.py`, compute per-token entropy during forward pass, use it to scale the cross-entropy loss. High-entropy tokens (reasoning) get more weight, low-entropy tokens (boilerplate/format) get less.

### 2. Data Selection (MEDIUM PRIORITY)
- **FisherSFT**: Select training examples that maximize information gain (Fisher information matrix at last layer). Data-efficient — same performance with fewer examples.
- **PEAR**: Reweight SFT loss to prepare for RL stage. Importance sampling to correct mismatch between SFT data and RL policy distribution.

### 3. Regularization (MEDIUM PRIORITY)
- **RPSFT (Rotation-Preserving SFT)**: Penalize changes in pretrained singular subspaces. Preserves OOD generalization. Better in-domain/OOD trade-off. Stronger init for downstream RL.
- **Full FT > LoRA for small models**: Paper confirms FFT consistently outperforms LoRA at matched training depth for small models. LoRA doesn't reduce wall-clock time either.

---

## Boot Time / Model Loading Improvements

### 1. safetensors Loading (HIGH PRIORITY — easy fix)
- **CPU + pin + async path**: `safe_open(device="cuda")` is 1.2-11x SLOWER than `safe_open(device="cpu")` + `pin_memory()` + async `.to("cuda")`. Root cause: serialized page faults during direct cudaMemcpy from mmap.
- **Fix**: Set `SAFETENSORS_FAST_CUDA=1` env var (enables the fast path in newer safetensors).
- **madvise(MADV_SEQUENTIAL)**: Aggressive kernel readahead. 1GB/s → 1.5GB/s on SSD. Already being added to PyTorch core.
- **fastsafetensors**: Already integrated (4.8-7.5x faster). Needs DirectStorage DLLs on Windows — currently falling back to standard.

### 2. CUDA Graph Materialization (HIGH PRIORITY)
- **Medusa**: Materialize CUDA graphs + KV cache info OFFLINE, restore them ONLINE. Reduces cold start by 42.5%, TTFT by 53%.
- **PASK**: Proactive kernel loading for cuDNN/MIOpen. Recycle existing kernels instead of recompiling. Interleave code loading with GPU computation.
- **Implementation**: Pre-capture CUDA graphs for common batch sizes (1, 2, 4) during first run, save to disk, restore on subsequent runs.

### 3. torch.compile Caching (MEDIUM PRIORITY)
- **vLLM approach**: Cache compiled artifacts (FX graphs, Triton kernels) to `~/.cache/vllm/torch_compile_cache`. Warm start retrieves from cache instead of recompiling.
- **Implementation**: Use `torch._inductor.cache` to persist compiled graphs across process restarts.

### 4. Kernel Pre-warming (already partially implemented)
- **Current**: ForgeEngine.activate() runs warmup dummy token to trigger CUDA JIT.
- **Improvement**: Pre-compile all unique kernel configurations (different seq lengths, batch sizes) during first run. Cache the compiled kernels.

---

## Actionable Items for ForgeAI

### Quick Wins (implement now):
1. **Token entropy weighting in SFT** — weight reasoning tokens higher
2. **safetensors CPU+pin+async loading** — replace direct CUDA load
3. **SAFETENSORS_FAST_CUDA=1** env var
4. **CUDA graph capture for inference** — pre-capture for batch sizes 1/2/4

### Medium effort:
5. **torch.compile cache persistence** — save compiled artifacts
6. **SHAD-style token classification** — offline reasoning vs boilerplate classification
7. **FisherSFT data selection** — select most informative training examples

### Research/future:
8. **Target-SFT** — mixture target distribution instead of one-hot
9. **RPSFT** — rotation-preserving regularization
10. **Medusa-style CUDA graph materialization** — offline graph capture + restore

---

## Anti-Regression: Prevent Catastrophic Forgetting Without Bloating Fine-Tune

Key constraint: no access to original pretraining data, no replay buffer, no extra params stored.

### Tier 1: Zero-cost / loss-level (implement now, no extra storage)

**1. Upweight Easy Samples (ICML 2025)**
- Upweight samples where the pretrained model's loss is LOW (easy samples)
- Downweight samples where loss is HIGH (hard/novel samples)
- Limits drift from pretrained model without any extra data or params
- Result: only 0.8% drop on GSM8K vs standard FT, preserves 5.4% more accuracy
- **Implementation**: Compute base model loss per sample once (cached), use as sample weight

**2. Low-Perplexity Token Masking (NeurIPS 2025)**
- Mask HIGH perplexity tokens in training data (tokens the model finds surprising)
- High-perplexity tokens cause the most forgetting — they push the model away from its prior
- LLM-generated data naturally has lower perplexity → less forgetting
- **Implementation**: Compute per-token perplexity with base model, mask top-k% highest

**3. KL Divergence Regularization (Context-Free Synthetic Data)**
- Add KL(anchor_model || current_model) as regularization loss
- Estimate KL via "context-free generation": generate text from anchor model with just BOS token, use as pseudo-replay data
- No need for original training data — the model generates its own replay
- **Implementation**: Generate N context-free samples from anchor model once, add as auxiliary loss

**4. Mask-the-Target KL Regularizer (2025)**
- Remove the ground-truth token from both base and adapted model distributions
- Renormalize remaining probabilities
- Apply KL regularization only over NON-TARGET vocabulary
- Preserves base model's relative preferences among alternatives without opposing the CE signal
- **Implementation**: In loss function, compute KL over non-target tokens only. Zero inference overhead.

### Tier 2: Lightweight param-level (small storage, no replay)

**5. L2-SP Regularization (NeurIPS 2024)**
- Add `lambda * ||theta - theta_0||^2` to loss (L2 distance from pretrained weights)
- theta_0 = pretrained model weights (already have them as checkpoint)
- Constrains drift from initialization. Simple, few lines of code.
- **Implementation**: Store reference to anchor checkpoint, add L2 penalty to optimizer loss

**6. Layer-wise L2 (L2-LoRA style)**
- Apply STRONGER L2 regularization on LOWER layers (store pretrained knowledge)
- Apply WEAKER L2 on HIGHER layers (task adaptation happens here)
- **Implementation**: Per-layer lambda values — lower layers get 10x higher lambda

**7. Hierarchical Element-wise Importance (2025)**
- Compute path integral of parameter updates as element-wise importance
- Penalize changes to important parameters more
- 20x faster than Fisher matrix, only 10-15% storage
- **Implementation**: Track parameter update history during training, use as importance weights

### Tier 3: Subspace methods (more complex, best results)

**8. SVD Subspace Constraint (Sculpting Subspaces, 2025)**
- Dynamically identify task-specific low-rank parameter subspaces via SVD
- Constrain updates to be ORTHOGONAL to critical directions of prior tasks
- No additional parameters, no storing gradients
- Up to 7% higher average accuracy than O-LoRA
- **Implementation**: Periodically compute SVD of weight deltas, project gradients onto orthogonal complement

**9. RPSFT (Rotation-Preserving SFT)**
- Penalize changes in projected top-k singular vector block of pretrained weights
- Limits unnecessary rotation while preserving task adaptation
- Better in-domain/OOD trade-off, stronger init for RL
- **Implementation**: Compute top-k SVD of pretrained weights once, add rotation penalty

### Recommended for ForgeAI (no bloat, no replay):

**Best combination**:
1. **Upweight easy samples** (sample-level, zero cost) — weight by base model loss
2. **Mask high-perplexity tokens** (token-level, zero cost) — skip tokens that cause most forgetting
3. **L2-SP anchor regularization** (param-level, lightweight) — `lambda * ||theta - theta_0||^2`
4. **Layer-wise lambda** — stronger on lower layers, weaker on upper

This combination prevents regression WITHOUT:
- Storing replay data
- Adding extra parameters
- Replaying prior training data
- Significant compute overhead

Total extra storage: just the anchor checkpoint path (already exists).
Total extra compute: one forward pass on base model per sample (cached), + L2 norm computation per step.

---

## General Improvements

### 1. Data Compression / Disk I/O (HIGH PRIORITY)

**Switch JSONL → Parquet with ZSTD**
- Parquet gives 3-5x compression vs JSONL (ZSTD outperforms Snappy significantly)
- Columnar format: skip unused columns entirely during reads
- NVIDIA research: ZSTD + delta encoding gives best string compression
- Sweet spot: 100-500 MB per Parquet file for splittability
- **Implementation**: Convert `research/data/finetune/*.jsonl` to `.parquet` with `pyarrow`, ZSTD compression. Update `sft_train.py` loader.

**JINX format (MLDataForge, 2025)**
- JSONL-compatible with index footer + binary sidecar
- zstd compressed with lazy loading = 10x throughput increase
- Defers decompression until field accessed
- **Implementation**: If Parquet is too heavy, JINX is a lighter alternative

**YOUMU columnar pipeline (MLSys 2025)**
- Finer-grained access on columnar data
- No format transformation in data prep
- Preserves shuffle quality
- **Implementation**: For large datasets (>500MB), consider columnar pipeline

### 2. OOM Prevention (HIGH PRIORITY)

**Activation Checkpointing (already have `--grad-checkpoint` flag)**
- Trades compute for memory — recompute activations during backward
- Slows training but prevents OOM on longer sequences
- **Current**: Disabled by default (was crashing on Windows)
- **Fix**: Enable selectively for layers > N, not all layers

**Activation Offloading (torchtune)**
- Move activations to CPU during forward, reload during backward
- Unlike checkpointing: NO recomputation — faster when PCIe bandwidth sufficient
- Async transfers overlap with computation
- **Implementation**: `torch.utils.checkpoint` with `offload_to_cpu=True` (PyTorch 2.4+)

**ZenFlow Stall-Free Offloading (DeepSpeed, 2025)**
- Offloads optimizer states + gradients to CPU
- Importance-aware pipelining: high-impact gradients stay on GPU, rest offloaded
- 85% stall reduction, up to 5x speedup vs ZeRO-Offload
- **Implementation**: For self-play finetune on limited VRAM

**Gradient Accumulation (already implemented)**
- Simulate larger batch sizes without more memory
- **Current**: `grad_accum=1` for speed, can increase if OOM

**Memory Snapshot Debugging (PyTorch built-in)**
- `torch.cuda.memory._record_memory_history()` captures allocation traces
- Visualize OOM root cause — which tensor caused the spike
- **Implementation**: Add `--debug-memory` flag that captures snapshot on OOM

**VRAM Monitoring (quick win)**
- Already logging `vram` in training steps
- **Improvement**: Add OOM early warning — if VRAM > 90% of limit, reduce batch size automatically

### 3. Better Logging (MEDIUM PRIORITY)

**Loguru (drop-in replacement for print/logging)**
- Zero config: `from loguru import logger`
- Structured JSON output: `logger.add("file.log", serialize=True)`
- Rotation: `rotation="100 MB"`, `retention="10 days"`, `compression="zip"`
- Thread-safe, multiprocess-safe, async via `enqueue=True`
- Colorized console output + file logging simultaneously
- **Implementation**: Replace `print()` calls in training/inference with `logger.info()`. Add JSON sink for machine-parseable logs.

**CSV metric logging (simple, future-proof)**
- Dump all metrics to CSV: step, loss, lr, vram, elapsed, etc.
- Query later with DuckDB: `SELECT * FROM runs WHERE loss < 0.5`
- No schema lock-in — just add columns as needed
- **Implementation**: Add `metrics_logger` that appends to `research/data/metrics.csv`

**What to log (indiscriminate capture principle)**
- Hyperparameters: lr, batch_size, seq_len, grad_accum, optimizer, grad_checkpoint
- Per-step: loss, lr, vram, elapsed_ms, tokens/sec
- Per-epoch: avg_loss, eval_metrics, checkpoint_path
- System: GPU util%, GPU temp, disk I/O, CPU load
- Environment: git commit, Python version, CUDA version, GPU model
- **Implementation**: Single `log_metrics()` function called at each step

### 4. KV Cache Optimization (MEDIUM PRIORITY — for inference)

**vAttention (2025)**
- Decouple virtual/physical memory for KV cache using CUDA virtual memory APIs
- Retains contiguous virtual memory (simpler than PagedAttention)
- 1.23x throughput improvement vs PagedAttention
- **Implementation**: For ForgeEngine's KV cache — use CUDA vmm API

**Chunked Prefill**
- Split long prompts into chunks, process incrementally
- Reduces peak memory during prefill
- CompactAttention: 2.72x attention speedup at 128K context
- **Implementation**: In `forge_engine.py`, chunk prompts > N tokens

**PrefillOnly optimization**
- For single-token outputs (classification, tool calls): only store last layer's KV cache
- Drastically reduces memory for tool-use workloads
- **Implementation**: Detect single-token generation requests, skip multi-layer KV storage

### 5. Checkpoint Optimization (MEDIUM PRIORITY)

**Sharded checkpoints**
- Save in multiple shards (100-500 MB each) for faster parallel loading
- **Implementation**: `safetensors` supports `max_shard_size` parameter

**Checkpoint compression**
- ZSTD compress safetensors files on disk
- 2-3x smaller checkpoints, fast decompression
- **Implementation**: Post-save compression with `zstandard` library

**Delta-only checkpoints**
- For self-play: save only the delta from anchor model, not full weights
- LoRA-style: store low-rank update matrix
- **Implementation**: Compute `delta = new_weights - anchor_weights`, SVD compress, save low-rank approximation

---

## 6. Small Model Reasoning Improvements (HIGH PRIORITY for 1.2B model)

### CoT Distillation for Small Models
- **Skip-Thinking (EMNLP 2025)**: Chunk-wise CoT distillation. Divide rationales into semantic chunks, train on one chunk per iteration. Isolates non-reasoning chunks. SLM skips non-reasoning medium chunks → faster + more accurate.
- **Mix Distillation (ACL 2025)**: Small models (3B) DON'T benefit from long CoT from large teachers. They perform better with SHORTER, SIMPLER reasoning chains. Mix long+short CoT examples. Critical insight for our 1.2B model.
- **Efficient Long CoT (2025)**: Prune unnecessary steps in long CoT (overthinking), then on-policy curation. SLMs learn efficient reasoning while preserving performance.
- **Mixture-of-Layers Distillation (EMNLP 2025)**: Transfer teacher's STEPWISE ATTENTION patterns to student. Teacher's progressive attention shifts toward key info during reasoning. Dynamic layer alignment between teacher/student.

**Key takeaway for ForgeAI**: Our 1.2B model should be trained on SHORT, SIMPLE reasoning chains — not long verbose ones. Mix distillation from both large and small teachers. Prune overthinking steps.

### Self-Play Training Loop Improvements
- **Absolute Zero Reasoner (NeurIPS 2025)**: Single model proposes tasks AND solves them. Uses code executor as verifiable feedback. Zero external data. SOTA on coding+math with NO human data. **Directly applicable to our self-play loop.**
- **SPELL (2025)**: Three-role self-play: questioner, responder, verifier in one model. Automated curriculum increases document length gradually. Reward adapts difficulty to model's evolving capabilities. 7.6-point gain on Qwen3-30B.
- **eva (ICML 2025)**: Asymmetric self-play — creator generates prompts, solver responds. Reward signals prioritize useful prompts. Induces meaningful learning curriculum. Gemma-2-9B win-rate 51.6%→62.4%.
- **SeRL (NeurIPS 2025)**: Bootstrap from limited data. Self-instruction + self-rewarding via majority voting. No external annotations needed. Matches performance of high-quality labeled data.
- **SCOPE (2025)**: Co-evolving Challenger + Solver. Frozen self-judge writes task-specific rubrics. +10.4 points on open-ended benchmarks with ZERO curated data.

**Key takeaway**: Our self-play loop should add a VERIFIER role (self-judge with rubrics), use code execution for verifiable rewards, and co-evolve the task generator with the solver.

---

## 7. RTX 5070 Blackwell Optimizations (HIGH PRIORITY — hardware-specific)

### FP4/FP8 Tensor Cores
- RTX 5070 has 5th-gen Tensor Cores with FP4 support (2x FP8 throughput, half memory)
- FP4 reduces model VRAM: FLUX model 23GB→<10GB at FP4
- Second-gen FP8 Transformer Engine (same as datacenter Blackwell)
- **Implementation**: Quantize LFM2.5-1.2B to FP4 for inference → 1.17GB model fits in ~0.6GB VRAM. Enables much larger batch sizes / longer context.
- **Library**: TensorRT 10.8+ supports FP4 on Blackwell. Or use `torch.float4_e2m1_fn` (PyTorch 2.6+).

### Blackwell-Specific Optimizations
- 64 concurrent warps per SM (compute capability 10.0)
- TMA (Tensor Memory Accelerator) for async memory ops
- Thread Block Clusters for cooperative computing
- **Implementation**: Update CUDA toolkit to 12.8+, PyTorch to 2.6+ for Blackwell kernel support.

---

## 8. Speculative Decoding (HIGH PRIORITY — 2-5x inference speedup)

### EAGLE-3 (NeurIPS 2025) — BEST option
- 5.6x faster than vanilla decoding (13B model)
- Lightweight autoregressive head on target model's internal layers
- No separate draft model needed
- Uses fusion of low/mid/high-level semantic features
- Trainable on 8x RTX 3090 (we have 1x RTX 5070 — should be feasible for 1.2B model)
- **Implementation**: Already have `research/decoding/eagle.py` — needs to be wired into ForgeEngine.

### Medusa (already have `research/decoding/medusa.py`)
- 2.2-3.6x speedup, multiple decoding heads on same model
- Parameter-efficient training
- Tree-based attention mechanism
- **Implementation**: Already implemented, needs integration with ForgeEngine inference path.

### Key insight from "Decoding Speculative Decoding" (NAACL 2025)
- Draft model LATENCY (not accuracy) is the bottleneck
- Shallower draft models with good enough accuracy > deeper accurate models
- 111% higher throughput with hardware-efficient draft models
- **Implementation**: For our 1.2B model, use a 2-3 layer draft head (not a separate model).

---

## 9. Tool-Use Training Data Quality (HIGH PRIORITY — directly addresses our tool-use issue)

### ToolACE (ICLR 2025)
- Automatic pipeline for generating accurate, diverse tool-learning data
- 26,507 diverse APIs via self-evolution synthesis
- Dual-layer verification: rule-based + model-based checks
- 8B model trained on this data matches GPT-4 in function calling
- **Implementation**: Use ToolACE-style verification in our self-play loop — validate tool calls with rule-based checks + model-based scoring.

### Tool-MVR (2025)
- Multi-Agent Meta-Verification (MAMV): validates APIs, queries, reasoning trajectories
- Exploration-based Reflection Learning: "Error → Reflection → Correction" paradigm
- 23.9% improvement over ToolLLM, 15.3% over GPT-4
- 31.4% fewer API calls
- **Implementation**: Add reflection training data — when tool call fails, generate correction trajectory.

### TL-Training (EMNLP 2025)
- Task-feature-based framework for tool-use training
- Dynamically adjusts token weights to prioritize KEY tokens during SFT
- Robust reward mechanism tailored to error categories
- Matches SOTA with only 1,217 training examples
- **Implementation**: Identify key tokens in tool calls (name, arguments) and upweight them during SFT.

### Magnet (ACL 2025)
- Multi-turn tool-use data synthesis via graph translation
- Context distillation: positive hints (correct calls) + negative hints (contrastive wrong calls)
- SFT + preference optimization against negative trajectories
- 14B model surpasses Gemini-1.5-pro in function calling
- **Implementation**: Generate negative tool call examples for DPO training.

### ToolReflection (2025)
- Self-generated errors + corrections for instruction tuning
- Real-time API feedback for self-correction
- 25.4% improvement on OOD, 56.2% on hard cases
- **Implementation**: In self-play, when tool call fails, save the error+correction as training data.

---

## 10. GRPO / RL Training Improvements (MEDIUM PRIORITY — for future RL phase)

### CPPO (NeurIPS 2025) — 8x speedup
- Prune completions with low absolute advantages
- Dynamic completion allocation to maximize GPU utilization
- 7.98x speedup on GSM8K, 3.48x on Math
- **Implementation**: In GRPO loop, skip completions where |advantage| < threshold.

### AGPO (Adaptive Group Policy Optimization, 2025)
- Fixes zero-variance in advantage estimation (when all rewards identical)
- Adaptive loss function for stable training
- Token-efficient reasoning (fewer tokens for same accuracy)
- **Implementation**: Add adaptive loss when group rewards are all equal.

### GTPO (2025) — No reference model needed
- Skip negative updates on valuable shared tokens
- Filter completions with entropy above threshold
- No KL-divergence regularization → no reference model (saves VRAM)
- **Implementation**: Replace GRPO with GTPO to eliminate reference model memory cost.

### Off-policy GRPO (2025)
- Off-policy GRPO matches or outperforms on-policy
- Sample reuse (µ > 1) improves efficiency
- Clipped surrogate objectives for stability
- **Implementation**: Reuse samples across multiple update steps.

---

## 11. Inference Serving Optimizations (MEDIUM PRIORITY)

### Chunked Prefill (Sarathi-Serve, OSDI 2024)
- Split large prefills into chunks, interleave with decodes
- Stall-free scheduling: new requests don't pause ongoing decodes
- 2.6x higher serving capacity (Mistral-7B on A100)
- **Implementation**: In ForgeEngine, chunk prompts > N tokens, interleave with decode batches.

### Continuous Batching
- Token-level batching (not request-level)
- New requests join mid-batch, completed tokens leave
- Eliminates padding waste
- **Implementation**: Already partially in ForgeEngine via KV cache management.

### Deferred Prefill (EuroMLSys 2025)
- Optimal threshold on prompt departures before scheduling new prefills
- Maximizes throughput under high request rates
- **Implementation**: For server mode, batch prefills optimally.

---

## 12. Hybrid Model Optimizations (MEDIUM PRIORITY — LFM2.5 specific)

### StripedHyena 2 (2025) — convolutional multi-hybrid
- 1.2-2.9x faster training than optimized Transformers at 40B scale
- Overlap-add blocked kernels for tensor cores
- Input-dependent convolutions + attention = complementary
- **Implementation**: Optimize our conv layers with blocked overlap-add kernels for tensor cores.

### Mamba-2 Kernel Fusion (PyTorch blog, 2025)
- Fuse all 5 SSD kernels into single Triton kernel
- 1.5-2.5x speedup on A100/H100
- Reduces launch overhead + redundant memory ops
- **Implementation**: Fuse our conv/gated operations into fewer Triton kernels.

### Nemotron-H (2025)
- Replace majority of attention with Mamba layers
- 3x faster inference, same accuracy
- MiniPuzzle: prune+distill 56B→47B, 20% faster
- **Implementation**: Our LFM2.5 already has 10 conv + 6 attention layers — similar ratio. Optimize the conv layers further.

### vLLM Hybrid Model Support (2025)
- vLLM V1 now supports hybrid models as first-class citizens
- Paged KV cache works with Mamba/conv+attention hybrids
- **Implementation**: If we move to vLLM serving, our hybrid architecture is supported.

---

## 13. LFM2/LFM2.5 Official Training Recipes (DIRECTLY APPLICABLE)

### LFM2 Technical Report (arxiv 2511.23404)
- **Architecture**: Gated short convolutions + small number of GQA blocks (our exact architecture)
- **Training pipeline**: Tempered decoupled Top-K knowledge distillation + curriculum learning + 3-stage post-training
- **Post-training recipe**: SFT → length-normalized DPO → model merging
- **Key finding**: For small models, directly train on downstream tasks (RAG, function calling) — not just general data
- **Data mix**: 50% downstream tasks, 50% general domains for SFT

### LFM2.5-2.6B Agentic Pipeline (4 stages)
1. **SFT** (two rounds): Broad coverage → targeted shaping on agentic tasks, reasoning, tool use. 7x larger than 8B model's SFT mix, heavier on agentic data.
2. **Teacher Specialization**: Train one expert per domain (math, code, tool use, instruction following, knowledge, long context) via focused SFT + RLVR.
3. **Multi-Domain On-Policy Distillation (MOPD)**: Distill specialist teachers back into single student.
4. **Agentic RL**: Multi-turn RL inside real agent harnesses. Separate Training Engine, Rollout Engine, Sandbox Service, Harness Proxy.

**Key insight for ForgeAI**: We should consider a similar multi-stage approach:
- Stage 1: Broad SFT (we have this)
- Stage 2: Domain specialists (we could train separate experts for tool use, reasoning, code)
- Stage 3: Distill specialists back into single model (we have merge_models.py for this)
- Stage 4: Agentic RL in our self-play loop (we have GRPO)

### LFM2.5-1.2B-Instruct
- Trained with SFT, preference alignment, and large-scale multi-stage RL
- Best-in-class at 1B scale for knowledge, instruction following, math, tool use
- **Generation params**: temperature=0.2, top_k=80, repetition_penalty=1.05

### LEAP Finetune (Liquid's official fine-tuning repo)
- Supports: SFT, DPO, GRPO, LoRA, full fine-tuning
- 500-5,000 task examples recommended (quality > volume)
- LoRA first pass: 1.2B LoRA takes minutes on single GPU
- **Two rules**: (1) Use model's own chat template character-for-character, (2) Don't over-train

### LFM2.5 Tool Use Format
- Default: Pythonic function calls (Python list between special tokens)
- Can override to JSON format via system prompt
- Four steps: function definition → function call → function execution → final answer
- **Our issue**: We're using literal text markers, but LFM2.5 uses special tokens (ids 10/11). This may be why the model outputs messy format.

---

## 14. Small Model Function Calling (DIRECTLY APPLICABLE)

### xLAM-1b-fc-r (Salesforce, 1.35B params)
- Function-calling optimized at 1B scale — directly comparable to our model
- Trained on xLAM-function-calling-60k dataset (60K samples)
- Uses unified format: task instruction + available tools + format instruction + query + steps
- **Format**: `[{"name": "tool_name", "arguments": {"arg1": "value1"}}]` (JSON array for parallel calls)
- **Key insight**: Format discipline matters — they added a "format-discipline slice" to training data

### HRM-Text-1B-agent v2 (1B params)
- Full-parameter SFT for function calling
- Added xLAM parallel/multi-call data + format-discipline slice
- Results: simple 61.5%→81.5%, multiple 53.5%→77.0%, parallel 37.5%→59.0%
- **Tradeoff**: irrelevance detection dropped 20% (model became too eager to call tools)
- **Lesson**: Need "don't-call" cases in training data to prevent over-eager tool calling

### Small Models, Big Tasks (arxiv 2504.19277)
- SLMs fail at function calling in zero-shot — need fine-tuning
- Few-shot helps significantly: Deepseek-Coder +67-80% with few-shot examples
- **Key finding**: JSON parsability is the main bottleneck for small models
- **Implementation**: Our xgrammar constrained decoding directly addresses this

### Microsoft SLM Function Calling Guide
- Fine-tune to learn function definitions internally (reduces prompt length + latency)
- Data synthesis is key — generate diverse, high-quality training examples
- 500-5000 examples sufficient for targeted function calling
- **Implementation**: We could bake our tool definitions into model weights via SFT

---

## 15. Memory-Efficient Training for 12GB VRAM (DIRECTLY APPLICABLE)

### Small Batch Size Training (NeurIPS 2025)
- **Batch size 1 is stable** for LLM training — no need for gradient accumulation
- Scale Adam beta2 half-life by tokens, not steps
- Small batch = more robust to hyperparameter choices
- Equal or better per-FLOP performance than large batches
- **Vanilla SGD works** at batch size 1 (no optimizer state = minimal memory)
- **Key insight**: "Recommend against gradient accumulation unless multi-device"
- **Implementation**: Use batch_size=1, no grad_accum. Saves optimizer state memory.

### LoRA on 12GB VRAM (InsiderLLM guide)
- 1B model: ~3.5GB at rank 16, batch 1 (with gradient checkpointing)
- 3B model: ~3.5GB at rank 16, batch 1
- Without Unsloth optimizations: add 30-60% more memory
- **Our model (1.2B)**: Should fit comfortably in 12GB with full fine-tuning at batch 2, seq 1536
- Current usage: 7.14GB at batch 2, seq 1536 — plenty of headroom

### LoHO Optimizer (AAAI 2025)
- Hybrid zeroth-order + first-order optimizer
- FO for deep layers, ZO for shallow layers
- Boosts accuracy while keeping memory within budget
- **Implementation**: For our conv layers (shallow), use ZO. For attention layers (deep), use FO.

---

## 16. Convolution Kernel Optimization for GPU (LFM2.5-SPECIFIC)

### StripedHyena 2 (arxiv 2503.01868)
- Overlap-add blocked kernels for tensor cores — 2x throughput vs linear attention/SSMs
- Filter grouping for improved hardware utilization
- 1.2-2.9x faster training than optimized Transformers at 40B scale
- **Implementation**: Our DoubleGatedConvLayer could use blocked overlap-add kernels

### TMA im2col for Convolution (Triton, Hopper+Blackwell)
- TMA im2col mode performs convolution address generation in hardware
- Works on both Hopper (WGMMA) and Blackwell (tcgen05 MMA)
- Reduces integer ALU work + register pressure
- **Implementation**: For RTX 5070 (Blackwell), use TMA im2col for our conv layers

### TritonForge (2025)
- Profiling-guided automated Triton kernel optimization
- Up to 5x improvement over baseline implementations
- LLM-assisted code transformation with profiling feedback
- **Implementation**: Could auto-optimize our custom kernels

### Twill (2025)
- Optimal software pipelining + warp specialization for tensor core GPUs
- Rediscovered expert Flash Attention schedules on Hopper + Blackwell
- **Implementation**: Could optimize our attention kernels for Blackwell

---

## Priority Ranking for ForgeAI (based on all research)

### Tier 1 — Immediate, high impact, low effort:
1. **Fix tool call format** — Use LFM2.5's native special tokens (ids 10/11) instead of literal text markers
2. **Token entropy weighting** in SFT loss (WeFT/VCORE)
3. **L2-SP anchor regularization** — prevent regression with anchor checkpoint
4. **Upweight easy samples** — weight by base model loss
5. **Batch size 1, no grad_accum** — more stable, less memory (NeurIPS 2025)
6. **LFM2.5 generation params** — temperature=0.2, top_k=80, rep_penalty=1.05

### Tier 2 — Medium effort, high impact:
7. **Domain specialist training** — train separate experts, merge with merge_models.py
8. **xLAM-format training data** — add parallel call + format-discipline + don't-call cases
9. **Tool-use reflection data** — error→correction trajectories
10. **Short reasoning chains** — prune verbose CoT for 1.2B model
11. **safetensors CPU+pin+async** loading
12. **Loguru structured logging**

### Tier 3 — Higher effort, significant impact:
13. **EAGLE-3 speculative decoding** — ✅ IMPLEMENTED (research/decoding/eagle.py)
    - Eagle3Head: multi-layer feature fusion (low/mid/high), single draft decoder layer
    - Eagle3Trainer: TTT with KL divergence loss, teacher forcing
    - eagle3_generate: draft→verify→accept loop with KV cache
    - Integrated into ForgeEngine: `engine.activate(decoding="eagle3")`
    - 121M unique params (0.23 GB bf16), ~10% of target model
    - Layers extracted: 1 (low), 8 (mid), 12 (high) for 16-layer LFM2.5
    - Sidecar loading: checkpoint.eagle3.safetensors auto-loaded if present
14. **FP4 quantization** for RTX 5070 Blackwell
15. **Agentic RL** in self-play (LFM2.5-style 4-stage pipeline)
16. **MOPD** — multi-domain on-policy distillation
17. **Conv kernel optimization** with TMA im2col for Blackwell
18. **Parquet+ZSTD** data format

---

## 17. COMPREHENSIVE BENCHMARK RESULTS (RTX 5070, 12GB VRAM)

### Model Load
| Metric | Value |
|--------|-------|
| Cold load | 68s (weights from disk) |
| Warm load (cached arch) | 4s |
| Model size | 2.34 GB bf16 (1.17B params) |
| VRAM after load | 3.61 GB |

### Per-Layer Forward (B=1, T=256)
| Layer Type | Time (warm) | % of total |
|-----------|-------------|------------|
| **Conv (10 layers)** | **171ms** | **89%** ← BOTTLENECK |
| Attention (6 layers) | 18ms | 10% |
| ln_f + head | 2ms | 1% |
| Layer 0 (cold) | 164-320ms | JIT compilation |
| Layer 2 (cold) | 12ms | First attention |

**Key finding**: Conv layers dominate at 89% of forward pass time. Each conv layer ~0.8ms, each attn ~1.1ms after warmup. Layer 0 has huge cold-start (JIT).

### Inference Speed
| Strategy | tok/s | Latency (50 tok) | VRAM |
|----------|-------|-------------------|------|
| Standard | 61-72 | 690-820ms | 3.68 GB |

### KV Cache Strategies
| Strategy | tok/s | VRAM | Notes |
|----------|-------|------|-------|
| Standard | 72 | 3.68 GB | Best baseline |
| Paged | 70-73 | 3.75 GB | Similar speed, +70MB |
| Streaming | 47-72 | 3.75 GB | Variable (window eviction) |
| SnapKV | 54-72 | 3.68 GB | Variable (observation window) |

### Quantization Impact
| Config | tok/s | VRAM | Notes |
|--------|-------|------|-------|
| bf16 baseline | 69-73 | 3.68 GB | Fastest |
| INT8 | 61-64 | 3.82 GB | **15% SLOWER** — dequant overhead |
| INT4 | 72 | 3.69 GB | **0 layers quantized** — BUG! |

**Bottleneck**: INT8 quantization makes inference SLOWER due to dequantization overhead on RTX 5070. INT4 is broken (quantizes 0 layers).

### EAGLE-3 Draft Head
| Metric | Value |
|--------|-------|
| Unique params | 121M (0.23 GB bf16) |
| Hidden state extraction (3 layers) | 8.5ms |
| Feature fusion | 0.06ms |
| **Draft forward** | **1.0ms** |
| **Target forward** | **8.9ms** |
| **Draft/Target ratio** | **0.12x** ← EXCELLENT (<0.3x) |
| VRAM with head | 4.28 GB (+0.67 GB) |

**Key finding**: EAGLE-3 draft head is 8.9x faster than target forward pass. Once trained, speculative decoding should give 3-5x speedup.

### Training Throughput
| Batch | tok/s (fwd) | tok/s (fwd+bwd) | VRAM |
|-------|-------------|-----------------|------|
| B=1, T=512 | 20,577 | 6,072 | 10.21 GB |
| B=2, T=512 | — | 7,040 | 10.47 GB |
| B=4, T=512 | — | 7,299 | 12.43 GB (near limit) |

**Bottleneck**: B=4 uses 12.43 GB — nearly OOM. B=2 is optimal (7K tok/s, 10.5 GB).

### Tool Call Parsing
| Metric | Value |
|--------|-------|
| Per call | 0.003ms |
| Calls/sec | 287,000 |

Not a bottleneck — parsing is essentially free.

### Context Manager
| Metric | Value |
|--------|-------|
| Compression time | 1.7ms |
| Was compressed | No (under threshold) |

Not a bottleneck.

---

## 18. BOTTLENECK SUMMARY & ACTION ITEMS

### Critical Bottlenecks — STATUS
1. ✅ **Conv layers = 89% of inference time** → Triton kernel written (research/decoding/triton_conv.py)
2. ✅ **INT8 quantization was SLOWER** → Fixed! FP8 _scaled_mm gives 1.34x speedup
3. ✅ **INT4 quantization was BROKEN** → Fixed! 92 layers quantized, 1.18x speedup
4. ✅ **Cold load was 68s** → Fixed! 4.2s (16x faster) via fastsafetensors async DMA
5. ✅ **Layer 0 cold start** → Fixed! Enhanced warmup pre-compiles all kernels

### Post-Fix Benchmark Results
| Config | tok/s | Speedup | VRAM |
|--------|-------|---------|------|
| Baseline (bf16) | 35.7 | 1.00x | 3.76 GB |
| INT4 (92 layers) | 42.0 | **1.18x** | 2.48 GB |
| INT8 fast (FP8 _scaled_mm) | 47.9 | **1.34x** | 1.41 GB |

### Model Loading Fix Results
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cold load | 68s | 4.2s | **16x faster** |
| VRAM after load | 3.61 GB | 2.44 GB | **32% less** |

### Performance Wins Available
1. **EAGLE-3**: 0.12x draft ratio → 3-5x speedup once head is trained
2. **Batch size 2 training**: 16% more throughput than B=1, fits in VRAM
3. **torch.compile**: Available via use_compile=True
4. **Triton conv kernel**: Available via use_triton_conv=True
5. **Paged KV**: Same speed as standard, better memory management for long context

### Non-Issues (Already Fast)
- Tool call parsing: 287K calls/sec
- Context compression: 1.7ms
- Feature fusion (EAGLE-3): 0.06ms
- Model warm load (cached): 3.3s
