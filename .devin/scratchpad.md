# R&D Round 15: Param Memory Cost Minimization

## Date: 2026-08-24
## Target: ForgeLM V7, ForgeEngine, Trainer

## Current State (verified)

### V7 Weight Memory Breakdown
| Component | Dense | Compressed | Technique | Savings |
|-----------|-------|-----------|-----------|---------|
| Embedding | 268M | 35.7M | Factorized (rank=512) | 86.7% |
| Attention | 537M | ~67M | BitNet b1.58 ternary | 87.5% |
| FFN | 12.88 GB | 1.52 GB | NLRQ INT8 (rank=768) | 88.2% |
| Other | ~200M | ~200M | - | - |
| **Total** | ~18.3 GB | ~2.86 GB | - | 84.4% |

### NLRQ Compression Ratio Discrepancy
- AGENTS.md claims 12.8x CR
- Actual for V7 dims (4096×16384, rank=768): **8.47x**
- 12.8x was for smaller dims (2048×8192, rank=256)
- This is a documentation bug, not a code bug

### KV Cache (32K context, V7)
- Full bf16: 4.0 GB (32 layers × 128 MB)
- RotorQuant INT4 (default): 1.0 GB (4x)

### Trainer Memory (V7-8B-B, BAdam)
- Weights bf16: ~5.6-8.0 GB (GPU)
- Active block optimizer: ~0.7-1.0 GB (GPU)
- Inactive optimizer states: ~20-30 GB (CPU)
- Total GPU: ~8.5-10.8 GB

## Identified Gaps

1. **int4 gradient compression**: CONFIGURED but NOT IMPLEMENTED
   - `grad_compression="int4"` in hybrid_offload.py line 115
   - Only stored as attribute, never used for actual compression
   - Evolution promoted int4 (score 30.00/11.10) but feature is a no-op

2. **NLRQ only on FFN**: Attention Q/K/V/O only compressed by BitNet
   - NLRQ on attention would be WORSE than BitNet (5.33x vs 10.1x for square matrices)
   - Not a viable direction — NLRQ compression ratio depends on matrix aspect ratio

3. **PEAGLE draft head**: 958 MB for 7 separate output heads
   - 7 × Linear(1024, 65536) = 471M params
   - Could tie to 1 shared head + LoRA adapters: 67.5M params (6.2x reduction)

4. **NLRQ INT4 factors**: Currently INT8, INT4 would 2x FFN compression
   - Need Hadamard rotation to spread outliers before INT4 quantization
   - FFN: 1.52 GB → 0.76 GB

## R&D Round 15 Techniques (implementing)

### Technique 1: int4 Gradient Compression with EF21
- File: research/training/optim/hybrid_offload.py
- Novel twist: EF21 error feedback (residual buffer on GPU) to preserve convergence
- Impact: 4x grad transfer bandwidth cut, enables full grad_offload on V7

### Technique 2: HINT4-NLRQ (Hadamard-INT4 factors)
- File: research/keys/compression/nlrq_ffn_key.py
- Novel twist: block-diagonal Hadamard rotation on SVD factors before INT4 quantization
- Impact: FFN 1.52 GB → 0.76 GB (2x), total V7 weights ~2.86 → ~2.1 GB

### Technique 3: Tied PEAGLE Heads with Position LoRA
- File: research/decoding/peagle.py
- Novel twist: shared output head + per-position low-rank adapters (rank=32)
- Impact: 958 MB → 135 MB (6.2x), frees ~0.82 GB VRAM for longer context

## Projected V7 Memory After R&D Round 15

| Component | Current | After R&D 15 | Technique |
|-----------|---------|-------------|-----------|
| FFN weights | 1.52 GB | 0.76 GB | HINT4-NLRQ |
| PEAGLE draft | 958 MB | 135 MB | Tied heads + LoRA |
| Grad transfer | 2.34 GB | 0.59 GB | int4 + EF21 |
| **Total inference** | ~4.5 GB | ~3.7 GB | - |
| **Total training** | ~8.5 GB | ~7.8 GB | - |
