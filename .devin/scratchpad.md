# Self-Play Quality Scaling Analysis

## V7 Compute Cost Minimization R&D (2025-01)

### Problem
Full V7 (forgelm_v7): 2.8B params, 7.36 GB storage (NLRQ INT8), 8.47 GB on GPU.
Training needs: model (8.47GB) + gradients (5.6GB) + activations = 14+ GB > 12GB VRAM.
Inference: NLRQ dequantization materializes 268MB/projection × 3 × 32 layers = 25.9GB OOM.

### Fix Already Applied: NLRQ Factorized Forward
Changed `nlrq_ffn_key.py` forward from `W = U @ diag(S) @ V; y = x @ W^T` to
`y = ((x @ V^T) * S) @ U^T`. Avoids materializing (out, in) weight matrix.
Peak memory: O(batch * (rank + out)) vs O(out * in). Forward now works on 12GB.

### Research Findings: Training Memory Reduction

#### Tier 1: Gradient Offloading (implement now)
- **offload_adam** (tascj): Hook-driven grad D2H in backward, pinned CPU memory.
  Modes: stochastic_rounding (bf16 states, smallest), fp32_master (bit-exact AdamW).
  Trades ~10% throughput for 2-3x larger model training. NUMA-aware pinning.
  → **Action**: Replace CPUAdamW with OffloadAdam pattern. Hook fires during
  backward(), streams grads to CPU pinned buffer. Frees GPU grad memory immediately.
  Expected: V7 2.8B trainable on 12GB (8.47GB model + ~0GB grads + O(1) activations).

- **ZenFlow** (arXiv 2505.12242): Stall-free offloading. Keeps important gradients
  on GPU for immediate update, offloads rest to CPU async. 5x speedup, 85% less
  GPU stall. Gradient selection via spatial+temporal locality.
  → **Action**: Novel twist — use TITAN memory importance scores to select which
  gradients stay on GPU. TITAN already computes per-layer importance.

#### Tier 2: Block-wise / Layer-wise Training (implement next)
- **BAdam** (NeurIPS 2024): Block coordinate descent with Adam. Updates one
  transformer layer at a time. Memory: 2M + 16M/D GB (D=blocks). For V7 (M=2.8B,
  D=32 layers): 2*2.8 + 16*2.8/32 = 5.6 + 1.4 = 7.0 GB. Fits 12GB easily.
  Outperforms LoRA on MT-bench. Microsoft BlockOptimizers repo has impl.
  → **Action**: Implement BAdam wrapper. Only one layer's optimizer states live
  on GPU at a time. Cycle through layers. Full-parameter training at LoRA cost.

- **LISA** (NeurIPS 2024): Layerwise Importance Sampled AdamW. Freezes most
  middle layers, only updates top/bottom + randomly sampled layers. Memory = LoRA
  cost but outperforms LoRA by 10-35% on MT-bench. Key insight: weight norms are
  skewed across layers (embedding/output layers have 100x larger norms).
  → **Action**: Implement LISA scheduler. Use existing MoD router scores to
  determine which layers to update. Novel: MoD-aware LISA.

#### Tier 3: Low-Rank Gradient Projection (R&D, higher risk)
- **GaLore** (ICML 2024): Projects gradients to low-rank subspace via SVD.
  65.5% optimizer memory reduction. 7B model on 24GB GPU. Already in codebase
  (galore_torch import in training_utils.py line 235) but optional dep.
  → **Action**: Already available. Wire it for V7 with rank=768 (matches NLRQ).

- **Q-GaLore** (PMLR 2025): INT4 projection matrices + INT8 weights with
  stochastic rounding. Adaptive subspace updates based on convergence. 7B on
  24GB. Reduces SVD overhead vs GaLore.
  → **Action**: Implement if GaLore SVD overhead is bottleneck.

- **Fira** (NeurIPS 2025): Full-rank training under low-rank constraint.
  Uses norm-based scaling from low-rank optimizer as proxy for full-rank.
  8x smaller optimizer memory than GaLore, outperforms it. Plug-and-play.
  → **Action**: Highest-value R&D target. Fira + NLRQ = both weight AND
  optimizer in low-rank. Novel: "NLRQ-Fira" — use NLRQ's SVD as Fira's
  low-rank subspace (free, no extra SVD needed).

- **Weight Refactorization + Momentum Reset** (arXiv 2505.22922): Two techniques
  that beat GaLore/Fira with 25% less memory. Weight refactorization = rescale
  weights to unit norm. Momentum reset = clear optimizer momentum periodically.
  → **Action**: Easy wins, implement as flags on existing optimizers.

### Research Findings: Inference Optimization

#### NLRQ Factorized Matmul (already implemented)
- `y = ((x @ V^T) * S) @ U^T` — avoids materializing W. Works.
- Further: fuse into single Triton kernel with INT8 tensor cores.

#### INT8 Tensor Cores on Blackwell (RTX 5070, SM120)
- 5th-gen Tensor Cores: dedicated INT8 + FP8 paths. 2.18x speedup vs fp16 on 4090.
- `torch._int_mm` / `torch._scaled_mm`: native INT8 matmul via cuBLAS.
- Triton `tl.dot` with int8 operands + int32 accumulation (sm_80+).
- → **Action**: Write Triton kernel for NLRQ that does INT8 x INT8 matmul
  directly on tensor cores, fusing dequant+matmul+scale. Expected 2x inference.

#### FP4 on Blackwell
- Native FP4 tensor core support on RTX 5070. 2x memory + 2x compute vs INT8.
- Block-level scaling factors (8-bit). Two-level scaling.
- → **Action**: FP4 NLRQ factors (instead of INT8). 2x smaller storage.
  7.36GB → 3.68GB. Enables full 2048 seq_len training.

#### torch.compile Fusion
- `config.force_fuse_int_mm_with_mul = True` fuses dequant+matmul.
- torch.compile generates efficient Triton kernels for weight-only quant.
- → **Action**: `torch.compile(model, mode="max-autotune")` on NLRQ forward.

### Priority Implementation Order
1. **Gradient offloading** (offload_adam pattern) — enables V7 training on 12GB
2. **BAdam** — block-wise training, 7GB for V7, full-parameter
3. **Fira + NLRQ** — novel low-rank training using NLRQ's existing SVD
4. **Triton INT8 NLRQ kernel** — 2x inference speedup
5. **FP4 NLRQ factors** — 2x memory reduction for training+inference
6. **LISA with MoD-aware selection** — novel layer-wise training

---

## Current State
- **Model**: LFM2.5-1.2B (1.17B params, 2.34GB bf16)
- **Self-play**: AZR curriculum (propose→solve→verify), LoRA finetuning (r=16, ~1M trainable params)
- **Eval**: fast_eval with 7 categories (tool_use, knowledge, reasoning, code, instruction, concise, self_correction)
- **Promotion threshold**: candidate quality >= base quality * (1 - eval_threshold=0.5)
- **Training VRAM**: ~6-7GB GPU (CPUAdamW offload), 19GB CPU RAM — fits RTX 5070 12GB

## Research Findings

### 1. Parameter Threshold for Reasoning
- **Critical threshold: 1.6B params** (arXiv 2502.15120) — below this, CoT reasoning fails
- **Current model (1.2B) is BELOW the reasoning threshold**
- AZR scaling: 3B→+5.7pts, 7B→+10.2pts, 14B→+13.2pts — "bigger bases yield bigger gains"
- AZR used Qwen2.5-7B as primary base; 3B was minimum tested

### 2. Self-Play Quality Bottleneck
- 1.2B model can propose tasks but struggles to SOLVE complex ones
- Small Model Learnability Gap (arXiv 2502.12143): models ≤3B don't benefit from long CoT
- Mix Distillation needed: blend short+long CoT for small models
- SGS (Self-Guided Self-Play): 7B model after 200 rounds > 671B model pass@4

### 3. Distillation Infrastructure (already built)
- Multi-provider teacher pool: gpt-oss-120b, DeepSeek-R1, Qwen3-32B, GLM-4.7
- 7+ free providers with rate-limit rotation
- Agentic distillation: teachers call tools, generate trajectories
- Curriculum distillation: 2-stage (internal solver → externalized reasoning)

### 4. Parameter Scaling Options

#### Option A: MoE (V5 config — already built)
- 7.5B total params, 1.2B active (same inference cost)
- 8 experts top-2 + shared expert
- AirMoE: experts hotloaded from disk, LRU cached (~2MB each with SVD+int4)
- VRAM: ~300MB weights (BitNet) + 1GB KV = 1.3GB
- **Problem**: Training MoE is harder (router instability, load balancing)

#### Option B: Depth Scaling (16→24 or 32 layers)
- More layers = more reasoning depth
- 24 layers: ~1.7B params (above 1.6B threshold!)
- 32 layers: ~2.3B params
- VRAM: 3.4GB / 4.6GB bf16 (fits 12GB with CPUAdamW)
- **Advantage**: Simple, proven, no architecture changes

#### Option C: Width Scaling (d_model 2048→3072)
- 3072 dim: ~2.6B params
- Better per-layer capacity
- VRAM: 5.2GB bf16 (fits 12GB with CPUAdamW)
- **Advantage**: More capacity per layer

#### Option D: V6 Compressed + Scale Up
- Use Monarch/Kronecker FFN compression to fit a LARGER model in same VRAM
- 2.3B model with Monarch FFN ≈ 1.0B VRAM (same as current 1.2B dense)
- "Free" parameter expansion via compression
- **Advantage**: 2x params at same VRAM cost

## Recommended Path

### Phase 1: Cross the 1.6B threshold (immediate)
- Scale to 24 layers (1.7B params) — minimal change, crosses reasoning threshold
- Use V6 compression (Monarch FFN) to keep VRAM at ~1.7GB
- Distill from teacher pool (gpt-oss-120b, DeepSeek-R1) to bootstrap reasoning
- Mix Distillation: blend short+long CoT (arXiv 2502.12143)

### Phase 2: MoE expansion (medium term)
- V5 MoE config: 7.5B total, 1.2B active
- AirMoE disk offload for inference (already built)
- Train with dense_bypass=True (lossless start), gradually activate router
- Expert tying (g=2) to halve expert params

### Phase 3: Self-play at scale (long term)
- AZR self-play on 7B+ base (proven to work)
- SGS mode (already in codebase) to prevent Conjecturer collapse
- SAERL for diversity control + difficulty curriculum
- RLVR with verifiable rewards (math, code)

## VRAM Budget (RTX 5070, 12GB)

| Config | Params | VRAM (bf16) | VRAM (BitNet) | Train VRAM |
|--------|--------|-------------|---------------|------------|
| V4 (current) | 1.22B | 2.45 GB | 0.61 GB | 6-7 GB |
| V6 compressed | 491M | 0.98 GB | 0.25 GB | 4-5 GB |
| 24-layer | 1.7B | 3.4 GB | 0.85 GB | 8-9 GB |
| 24-layer + V6 | ~1.0B | 2.0 GB | 0.50 GB | 6-7 GB |
| 32-layer | 2.3B | 4.6 GB | 1.15 GB | 10-11 GB |
| 32-layer + V6 | ~1.4B | 2.8 GB | 0.70 GB | 7-8 GB |
| V5 MoE (active) | 1.2B | 2.4 GB | 0.60 GB | 6-7 GB |
| V5 MoE (total) | 7.5B | 15 GB | 3.75 GB | N/A (disk) |

## Key Insight
The #1 bottleneck is NOT VRAM — it's the 1.6B reasoning threshold.
Current 1.2B model is below it. V6 compression lets us scale to 2.3B+
at the SAME VRAM cost as the current 1.2B dense model.
