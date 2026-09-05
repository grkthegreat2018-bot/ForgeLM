# R&D Round 44: Novel Quantization Algorithms

**Date**: 2026-09-05
**Model**: Qwen 2.5 0.5B (494M params, 942 MB FP16)
**Hardware**: NVIDIA RTX 5070 (12GB VRAM, Blackwell SM120)
**Test sequence**: 115 tokens (English text about ML/quantization)

## Summary

Designed and tested 4 novel post-training quantization (PTQ) algorithms on a real
Qwen 2.5 0.5B model, measuring perplexity, memory, and inference speed. Two
algorithms succeeded (AB-FP4, SR-INT4), two are documented dead ends
(HadamardLift, TernaryLift).

## Algorithms

### 1. AdaptiveBlockFP4 (AB-FP4) — SUCCESS

**Novel combination**: AS-FP4's MSE-optimal per-block scale search + NVFP4's
two-level block scaling + novel kurtosis-based per-block bit allocation.

**Key idea**: After computing per-block kurtosis (excess kurtosis), blocks are
classified into three tiers:
- Low kurtosis (<3, platykurtic): 3-bit FP4 (coarser, fewer levels)
- Medium kurtosis (3-7): 4-bit FP4 (standard)
- High kurtosis (>7, leptokurtic): 6-bit (FP4 + 2-bit residual)

This gives variable effective bit-width per block while keeping the average at
~4-bit. High-kurtosis blocks (with outliers) get more precision exactly where
it's needed, without a separate sparse path.

**Results**: PPL 9.19 (+0.69 over FP16), 569.7 MB (39.5% savings), recon err 0.093.
**Beats NVFP4 baseline** (10.25) by 1.06 PPL points.

**Limitation**: Speed is 8.7 tok/s (0.18x FP16) due to dequant overhead. Needs
fused kernel or cached dequantized weights.

### 2. SparseResidualINT4 (SR-INT4) — SUCCESS

**Novel combination**: SpQR's sparse outlier isolation + ForgeQuant's INT8 sparse
path + novel error-threshold-based outlier selection.

**Key idea**: After INT4 base quantization, compute per-element reconstruction
error. The outlier threshold is adaptive per-layer:
`threshold = mean(|error|) + n_std * std(|error|)`

With n_std=1.0 (aggressive), more outliers are captured, bringing reconstruction
error to 0.080 — the best of all novel methods.

**Results** (aggressive, n_std=1.0): PPL 9.19 (+0.69), 664.3 MB (29.5% savings),
27.4 tok/s (0.57x FP16), recon err 0.080.

**Advantage over AB-FP4**: 3.1x faster (27.4 vs 8.7 tok/s) at same PPL.

### 3. HadamardLift (HLQ) — DEAD END

**Novel combination**: QuaRot's Hadamard rotation + LiftQuant's dimensional
lifting + 1-bit/2-bit sign quantization in lifted space.

**Why it failed**: STE (straight-through estimator) optimization of the projection
matrix P does not converge. Reconstruction error stays at 0.54-0.72 even with 200
Adam steps at lr=0.005. The fundamental issue is that sign quantization in lifted
space loses too much information, and STE gradients through sign() are too noisy.

**What would be needed**: LiftQuant's actual algorithm uses much more
sophisticated optimization (iterative with proper gradient computation, not STE).
A simple STE approach is insufficient for projection matrix optimization.

### 4. TernaryLift (TL) — DEAD END

**Novel combination**: BitNet b1.58 ternary {-1,0,+1} + LiftQuant lifting.

**Why it failed**: Same convergence issue as HadamardLift. The ternary zero level
helps slightly (recon err 0.44 vs 0.54) but not enough for useful PPL.

## Benchmark Results

| Method | PPL | delta-PPL | Speed | Weight | Savings | Recon |
|--------|-----|-----------|-------|--------|---------|-------|
| FP16 baseline | 8.50 | 0.00 | 48.0 tok/s | 942.3 MB | 0% | 0.000 |
| **AB-FP4** | **9.19** | **+0.69** | 8.7 tok/s | 569.7 MB | 39.5% | 0.093 |
| **SR-INT4 aggressive** | **9.19** | **+0.69** | 27.4 tok/s | 664.3 MB | 29.5% | 0.080 |
| SR-INT4 (n_std=2.0) | 10.25 | +1.75 | 27.4 tok/s | 447.8 MB | 52.5% | 0.104 |
| SR-INT3 aggressive | 19.50 | +11.00 | 18.8 tok/s | 670.4 MB | 28.9% | 0.184 |
| SR-INT3 | 41.25 | +32.75 | 27.1 tok/s | 449.0 MB | 52.4% | 0.241 |
| INT4 baseline | 9.06 | +0.56 | 20.9 tok/s | 858.2 MB | 8.9% | 0.000 |
| NVFP4 baseline | 10.25 | +1.75 | 20.1 tok/s | 259.7 MB | 72.4% | 0.104 |

## Analysis

### Quality vs Memory tradeoff

AB-FP4 and SR-INT4 aggressive both achieve PPL 9.19, which is:
- Better than NVFP4 (10.25) by 1.06 PPL points
- Slightly worse than INT4 (9.06) by 0.13 PPL points
- Only +0.69 over FP16 baseline

The key advantage is memory: AB-FP4 uses 39.5% less memory than FP16, and
SR-INT4 aggressive uses 29.5% less. INT4 baseline only saves 8.9% because it
stores quantized weights as float16 (not truly packed nibbles).

### Speed considerations

On RTX 5070 (SM120), the dequantization overhead dominates for small models:
- FP16: 48 tok/s (baseline)
- SR-INT4: 27.4 tok/s (0.57x) — dequant + F.linear
- AB-FP4: 8.7 tok/s (0.18x) — complex dequant with bit allocation
- INT4: 20.9 tok/s (0.44x) — dequant + F.linear

For a 0.5B model, the weight matrices are small enough that dequant overhead
outweighs memory bandwidth savings. This would change for larger models (7B+)
where memory bandwidth is the bottleneck.

### Hardware notes (RTX 5070 / SM120)

- SM120 uses SM80-era `mma.sync`, not SM100 `tcgen05`
- No TMEM, 99 KB shared memory per SM
- FP4 native support requires validation (not assumed)
- All novel methods use dequant + F.linear (no native low-bit GEMM)
- A fused dequant+matmul kernel would significantly improve speed

## Files

- `forge/engine/quant/novel_quant_r44.py` — implementation (832 lines)
- `scripts/test_novel_quant_r44.py` — benchmark script (479 lines)
- `tests/unit/test_novel_quant_r44.py` — unit tests (23 tests, all pass)
- `scripts/r44_quant_results.json` — raw results

## Next Steps

1. **Speed optimization**: Implement cached dequantized weights for AB-FP4
   (like QuantizedLinear does for INT4/INT8)
2. **SR-INT4 tuning**: Try group_size=32 and per-layer adaptive n_std
3. **Integration**: Wire AB-FP4 and SR-INT4 into `forge_engine.py`
   `_apply_quantization` dispatch
4. **Larger model test**: Validate on Qwen 2.5 1.5B/3B where memory bandwidth
   matters more
5. **Lifting methods**: Would need proper LiftQuant-style optimization (not STE)
   to work — shelved as documented dead end
