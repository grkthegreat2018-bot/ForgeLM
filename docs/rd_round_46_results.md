# R&D Round 46 — Novel Quantization Results

**Date:** 2026-09-05
**Model:** Qwen/Qwen2.5-0.5B (494M params, 942.3 MB FP16)
**Hardware:** RTX 5070 12GB VRAM, Intel Core Ultra 7 265F, 32GB RAM
**Calibration:** 128 samples from a single short text (77 tokens per layer)

## Summary

R46 implemented and benchmarked six novel FP4 quantization algorithms targeting
different sources of quantization error. **AWQFP4 is the clear winner**, achieving
PPL 35.25 (delta +3.625 vs FP16) — a **70% reduction in quality gap** compared to
R45's SchurAB-FP4 (PPL 43.75, delta +12.125), while also using less memory
(193.1 MB vs 203.8 MB).

## Algorithms Implemented

### 1. HadamardRotatedFP4 (HR-FP4) — no calibration
QuaRot/SpinQuant-style Hadamard rotation pre-pass. Rotates weight columns by a
deterministic orthonormal Hadamard matrix to reduce outliers, then quantizes
the rotated weights. The rotation is recomputed at inference time (zero storage
overhead). The inverse rotation is applied during dequantization.

### 2. GPTQFP4 — needs calibration
GPTQ-style column-wise error compensation. Quantizes columns left-to-right in
groups, pushing each group's quantization error into remaining unquantized
columns using the inverse Hessian (computed via Cholesky decomposition of the
calibration activation Gram matrix). The diagonal approximation of the inverse
Hessian is used for the batched group update.

### 3. AWQFP4 — needs calibration
AWQ-style activation-aware weighting. Computes a per-channel Hessian proxy
h_j = E[x_j^2] from calibration activations, then uses it to weight the
MSE-optimal FP4 scale search. Salient channels (high activation magnitude)
get more accurate quantization at the expense of quiet channels. Simpler than
GPTQ (no error compensation, just weighted scale selection).

### 4. OptimalGridFP4 (OG-FP4) — no calibration
Data-dependent 4-bit codebook search (OptIQ/AFQ-style). Instead of the fixed
FP4 E2M1 magnitudes, runs Lloyd-Max iteration on the weight distribution to
find the optimal 8-magnitude codebook per layer. The codebook is stored as
8 float values per layer (negligible overhead).

### 5. HadamardGPTQFP4 (HR-GPTQ-FP4) — needs calibration
Combined Hadamard rotation + GPTQ error compensation. The rotation is applied
first, then GPTQ operates on the rotated weights with the rotated Hessian.

### 6. HadamardAWQFP4 (HR-AWQ-FP4) — needs calibration
Combined Hadamard rotation + AWQ activation-aware weighting. The rotation is
applied first, then AWQ-style weighted scale search on the rotated weights.

## CUDA Benchmark Results (Qwen 2.5 0.5B)

| Method | PPL | dPPL | Speed (tok/s) | Weight (MB) | Eff bits | Recon err |
|---|---:|---:|---:|---:|---:|---:|
| FP16 (baseline) | 31.625 | 0.000 | 47.5 | 942.3 | 16.00 | 0.0000 |
| **AWQFP4** | **35.250** | **+3.625** | **10.1** | **193.1** | **4.53** | **0.1003** |
| HadamardAWQFP4 | 40.500 | +8.875 | 3.1 | 250.9 | 5.88 | 0.0850 |
| HadamardGPTQFP4 | 40.500 | +8.875 | 3.5 | 250.9 | 5.88 | 0.0856 |
| SchurAB-FP4 (R45) | 43.750 | +12.125 | 6.0 | 203.8 | 4.78 | 0.0929 |
| OptimalGridFP4 | 44.500 | +12.875 | 9.3 | 193.1 | 4.53 | 0.0847 |
| HadamardRotatedFP4 | 45.250 | +13.625 | 3.3 | 250.9 | 5.88 | 0.0838 |
| GPTQFP4 | 52.000 | +20.375 | 6.0 | 193.1 | 4.53 | 0.0963 |
| NVFP4 (baseline) | 43.250 | +11.625 | 47.0 | 259.7 | 16.00* | 0.1044 |

*NVFP4 eff_bits shows 16.00 due to memory estimation not accounting for its
internal packed format; actual is ~4.5 bits.

## Key Findings

### AWQFP4 is the best quality method
- **PPL 35.25** — only +3.625 above FP16, a 70% quality gap reduction vs R45
- **193.1 MB** — smallest weight memory, 79.5% savings vs FP16
- **4.53 effective bits** — best bits-per-quality tradeoff
- The AWQ insight (weight the MSE by activation importance) is the single most
  effective technique for FP4 quantization quality

### Hadamard rotation helps GPTQ but hurts AWQ
- HR-GPTQ-FP4 (PPL 40.5) >> GPTQ-FP4 (PPL 52.0): rotation makes the Hessian
  better-conditioned, helping GPTQ's error compensation
- HR-AWQ-FP4 (PPL 40.5) < AWQ-FP4 (PPL 35.25): rotation dilutes channel
  salience by mixing channels, hurting AWQ's activation-aware weighting

### GPTQ alone is disappointing with limited calibration
- GPTQ-FP4 (PPL 52.0) is worse than SchurAB-FP4 (PPL 43.75)
- Only 77 calibration tokens (from a single short text) produce a noisy Hessian
- GPTQ typically needs hundreds to thousands of calibration samples
- The diagonal inverse Hessian approximation may also be too coarse

### OptimalGridFP4 has best reconstruction error but not best PPL
- OG-FP4 recon err = 0.0847 (lowest among no-calibration methods)
- But PPL = 44.5 (worse than SchurAB-FP4's 43.75)
- The optimal codebook minimizes weight MSE but not output MSE
- This confirms that weight reconstruction error ≠ output quality

### Speed remains a challenge
- All Python-dequantization methods are 4-15x slower than NVFP4
- NVFP4 uses a fused CUDA kernel for dequantization
- AWQFP4 at 10.1 tok/s is the fastest of the novel methods
- A fused CUDA kernel for the packed FP4 format would be needed for
  production-speed inference

## Memory Comparison

| Method | Weight (MB) | Savings vs FP16 | Eff bits |
|---|---:|---:|---:|
| FP16 | 942.3 | 0% | 16.00 |
| AWQFP4 | 193.1 | 79.5% | 4.53 |
| OptimalGridFP4 | 193.1 | 79.5% | 4.53 |
| GPTQFP4 | 193.1 | 79.5% | 4.53 |
| SchurAB-FP4 (R45) | 203.8 | 78.4% | 4.78 |
| HadamardRotatedFP4 | 250.9 | 73.4% | 5.88 |
| HadamardGPTQFP4 | 250.9 | 73.4% | 5.88 |
| HadamardAWQFP4 | 250.9 | 73.4% | 5.88 |
| NVFP4 | 259.7 | 72.4% | ~4.5* |

The Hadamard methods use more memory because in_features is padded to the next
power of 2 (896 → 1024 for Qwen 0.5B), a 14% increase.

## Files Created

- `forge/engine/quant/novel_quant_r46.py` — all 6 algorithms + conversion functions
- `scripts/smoke_test_r46.py` — 9 CPU smoke tests (all pass)
- `scripts/test_novel_quant_r46.py` — CUDA benchmark script
- `tests/unit/test_novel_quant_r46.py` — 46 unit tests (all pass)

## Bugs Fixed During R46

1. **AWQFP4 Hessian expansion**: Initial `h.unsqueeze(0).expand(out_f, n_blocks, block_size)`
   failed because h was (in_features,) not (n_blocks, block_size). Fixed by
   reshaping to `(n_blocks, block_size)` first.

2. **GPTQ using Hessian instead of inverse Hessian**: Initial implementation used
   `H` (the Hessian) in the error compensation update where `H_inv` (the inverse
   Hessian) was required. This caused numerical explosion (PPL >100M). Fixed by
   computing the inverse Hessian via Cholesky decomposition.

3. **GPTQ Hessian regularization too weak**: Initial damping (0.01 * mean(diag))
   was insufficient for channels with near-zero activation, causing the diagonal
   inverse to explode. Fixed by clamping `H_inv_diag` to min=1e-8.

## Dead Ends

- **Hadamard + AWQ combination**: Despite both being strong individually, the
  combination is worse than AWQ alone. The rotation dilutes channel salience.
  Documented as a dead end for the AWQ use case.

- **GPTQ with limited calibration**: GPTQ requires substantially more calibration
  data than was available (77 tokens from one text). With proper calibration
  (512+ samples from diverse text), GPTQ may perform better. Left as future work.

- **OptimalGridFP4 for output quality**: While OG-FP4 achieves the lowest weight
  reconstruction error, it doesn't translate to better PPL because it optimizes
  weight MSE, not output MSE. An activation-aware grid search (combining OG-FP4
  with AWQ-style weighting) might be more effective.

## Next Steps (R47 candidates)

1. **AWQ + OptimalGrid**: Combine AWQ's activation-aware weighting with OG-FP4's
   data-dependent codebook. Use the Hessian proxy to weight the Lloyd-Max
   iteration, placing codebook levels where they reduce output error.

2. **GPTQ with proper calibration**: Use 512+ calibration samples from a diverse
   corpus (WikiText, C4) instead of a single text. This should significantly
   improve GPTQ's Hessian quality.

3. **Fused CUDA kernel for FP4 dequantization**: The current Python/PyTorch
   dequantization is 4-15x slower than NVFP4's fused kernel. A custom CUDA
   kernel for the packed FP4 format would close the speed gap.

4. **AWQ with per-channel scaling**: Current AWQ uses per-block scaling weighted
   by the Hessian. Per-channel scaling (finer granularity) might further improve
   quality at a small memory cost.

5. **Layer-adaptive bit allocation**: Use the calibration Hessian to determine
   which layers need more bits (FP4) vs fewer (INT3/ternary), similar to
   HAWQ-V3's sensitivity-aware bit allocation.
