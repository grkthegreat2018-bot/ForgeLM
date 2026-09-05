# R&D Round 48 — Extreme Low-Bit Quantization (Sub-1-bit & Ternary)

**Date:** 2026-09-05
**Model:** Qwen/Qwen2.5-0.5B (494M params, 942.3 MB FP16)
**Hardware:** RTX 5070 12GB VRAM, Intel Core Ultra 7 265F, 32GB RAM

## Summary

R48 implemented and benchmarked three extreme low-bit PTQ methods recommended
by another agent: NanoQuant (sub-1-bit binary factorization), BTC-LLM (binary
codebook clustering), and TernaryPTQ (1.58-bit ternary). All three achieve
extreme memory savings (92-99%) but with severe quality degradation on
non-BitNet-trained models, confirming that sub-2-bit PTQ requires either QAT
or block-level reconstruction refinement.

## Methods Implemented

### 1. NanoQuant — Low-rank binary factorization + ADMM
- **Source:** Samsung Research, ICML 2026 (arXiv 2602.06694)
- **Formula:** W ≈ s1 ⊙ (U_±1 @ V_±1^T) ⊙ s2^T
- **Storage:** r×(d_out+d_in) bits + 16×(d_out+d_in) bits for scales
- **Key innovation:** ADMM-based initialization of binary factors, then
  optional STE gradient refinement

### 2. BTC-LLM — Binary codebook clustering + Hadamard rotation
- **Source:** ACL 2026 (arXiv 2506.12040)
- **Formula:** Cluster binary ±1 weight rows into K codebook patterns
- **Storage:** K×d_in bits (codebook) + d_out×log2(K) bits (indices) + d_out×16 (scales)
- **Key innovation:** Hadamard rotation before binarization reduces outliers,
  making binary patterns more clustered. On-the-fly Hadamard reconstruction
  (no stored matrix) keeps memory minimal.

### 3. TernaryPTQ — Ternary {-1,0,+1} with Hessian-refined scales
- **Source:** BitNet b1.58 (arXiv 2402.17764) + ScaleQ-1.58 (arXiv 2608.01078)
- **Formula:** W_q = sign(W/scale) × (|W/scale| > 0.7), values in {-1, 0, +1}
- **Storage:** 1.6 bits/w (base-3 packed: 5 ternary values per byte) + 16 bits/channel
- **Key innovation:** Per-channel absmean scale (not per-tensor), optional
  Hessian-weighted gradient refinement of scales using calibration activations

## Benchmark Results (Qwen 2.5 0.5B, real weights)

| Method | PPL | dPPL | Speed | Weight | Eff bits | Savings |
|---|---:|---:|---:|---:|---:|---:|
| FP16 (baseline) | 31.625 | 0.000 | 20.3 | 942.3 | 16.00 | — |
| **AWQFP4 (R47)** | **31.625** | **+0.000** | 21.6 | 193.1 | 4.527 | 79.5% |
| NVFP4 (baseline) | 43.250 | +11.625 | 45.8 | 193.1 | 4.527 | 79.5% |
| TernaryPTQ (vanilla) | 284,672 | +284,640 | 48.3 | 68.8 | 1.614 | 92.7% |
| TernaryPTQ (Hessian) | 1,359,872 | +1,359,840 | 25.3 | 68.8 | 1.614 | 92.7% |
| NanoQuant (r=128) | 1,064,960 | +1,064,928 | 20.7 | 9.4 | 0.221 | 99.0% |
| NanoQuant (r=256) | 152,576 | +152,544 | 46.3 | 17.8 | 0.418 | 98.1% |
| BTC-LLM (K=256) | 8,355,840 | +8,355,808 | 20.5 | 11.5 | 0.269 | 98.8% |
| BTC-LLM (K=512) | 45,088,768 | +45,088,736 | 47.0 | 20.5 | 0.480 | 97.8% |

## Key Findings

### 1. Sub-2-bit PTQ quality collapses without QAT
All three extreme low-bit methods produce PPL >100K on Qwen 2.5 0.5B —
a model not trained with quantization awareness. This confirms the
fundamental finding from BitNet b1.58: **ternary/binary quantization
requires training (QAT) or extensive block-level reconstruction**.

### 2. NanoQuant achieves the most extreme compression
At rank=128, NanoQuant compresses 942.3 MB → 9.4 MB (99.0% savings,
0.221 bits/w). This is the most extreme compression ever achieved in
our R&D rounds. The quality is unusable (PPL 1M) but the compression
ratio is real — with block-level reconstruction (as in the paper),
this could become usable.

### 3. BTC-LLM rotation overhead eliminated
Initial implementation stored full Hadamard matrices (896×896 float16
per layer = 1.9MB × 168 = 319MB overhead). Fixed by reconstructing
Hadamard on-the-fly from just the order (log2(N) integers). Memory
dropped from 3371 MB → 11.5 MB.

### 4. TernaryPTQ: Hessian refinement made it worse
The Hessian-weighted scale refinement (lr=0.005, 20 iters) made PPL
worse (1.36M vs 284K for vanilla). The scale optimization diverged
even with clamping. This suggests that for ternary PTQ, the vanilla
absmean scale is already near-optimal — the quality bottleneck is the
ternary representation itself, not the scale.

### 5. AWQFP4 (R47) remains the undisputed champion
No extreme low-bit method comes close to AWQFP4's quality (PPL 31.625
= FP16). The 4.53-bit FP4 representation with activation-aware
weighting is the sweet spot for PTQ on non-BitNet-trained models.

## What Would Make Extreme Low-Bit Work?

Based on the NanoQuant paper and ScaleQ-1.58:

1. **Block-level reconstruction** (NanoQuant's key innovation):
   After ADMM initialization, optimize each transformer block's
   binary factors to minimize block-level output error, not just
   weight-level MSE. This is the missing piece in our implementation.

2. **Model-level KL calibration** (NanoQuant):
   After block-level reconstruction, calibrate global scaling factors
   to align model-level activations with the original.

3. **QAT or quantization-aware pre-training** (BitNet b1.58):
   The most reliable path to sub-2-bit quality. BitNet b1.58 trains
   from scratch with ternary weights and matches FP16 quality.

4. **Reasoning-aware calibration** (ScaleQ-1.58 / AYOT):
   Use the model's own chain-of-thought reasoning traces as
   calibration data, not random texts. This helps preserve reasoning
   ability during ternary quantization.

## ForgeEngine Integration

All three methods are integrated into the ForgeEngine quantize dispatch:
- `quantize="nanoquant"` — NanoQuant low-rank binary
- `quantize="btc"` — BTC-LLM binary codebook
- `quantize="ternary_ptq"` — Ternary PTQ with Hessian refinement

All have fallback chains to AWQFP4 → NVFP4 → int4 → None.

## Files

- `forge/engine/quant/novel_quant_r48.py` — implementation (660 lines)
- `scripts/smoke_test_r48.py` — smoke tests (6/6 pass)
- `tests/unit/test_novel_quant_r48.py` — unit tests (27/27 pass)
- `scripts/test_novel_quant_r48.py` — real-model benchmark
- `forge/engine/forge_engine.py` — dispatch integration

## Block-Level Reconstruction (Added)

A general-purpose `BlockReconstructor` infrastructure was added at
`forge/engine/quant/block_recon.py` (684 lines). It:

- Captures each transformer block's input, kwargs (position_embeddings,
  attention_mask, position_ids), and output from the original model
- Patches quantized layers with soft-forward closures that use STE for
  binary params and direct gradients for continuous scales
- Optimizes per-block parameters to minimize block-level output MSE
- Supports both independent and progressive (sequential) reconstruction
- Works with NanoQuant, BTC, TernaryPTQ, and R46 FP4 layers

### Block reconstruction results on Qwen 2.5 0.5B (NanoQuant r=128)

| Mode | PPL Before | PPL After | Block Loss Change |
|---|---:|---:|---|
| Independent (3 blocks) | 222K | 21.4M (worse) | Block 1: 0.032→0.021 (34% better) |
| Independent (24 blocks) | 152K | 10.1M (worse) | Most blocks improved 5-35% |
| Progressive (24 blocks) | 143K | 684K (worse) | Blocks 0-2 improved, 3+ saturated at ~248 |

### Why PPL got worse despite per-block loss improvement

1. **Scale-only optimization**: Binary factors at ±1 are insensitive to
   small gradient updates (sign() doesn't change). Only scales are
   optimized, but the binary factors are the dominant error source.

2. **Error accumulation**: Changing scales in early blocks shifts the
   input distribution for later blocks. Without global KL calibration
   (NanoQuant's Step 3.3), the scale changes amplify through the stack.

3. **Missing Hessian-preconditioned ADMM**: The NanoQuant paper uses 400
   ADMM iterations with Hessian-aware preconditioning (KFAC with shrinkage)
   and SVID initialization. Our 50-iteration plain ADMM produces much
   poorer binary factors that scale optimization cannot compensate for.

### What's needed to make block reconstruction effective

Per the NanoQuant paper (ICML 2026, Appendix A & B):
1. **Hessian-aware ADMM** (400 outer iters, KFAC preconditioners, SVID init)
2. **Latent matrix optimization** (optimize continuous pre-binarization
   matrices, not just scales)
3. **Model-level KL calibration** after block reconstruction
4. **Error propagation mitigation** (adjust FP weights before quantizing
   each block to compensate for accumulated errors)

The block reconstruction infrastructure is functional and ready for these
improvements. It can also be used with AWQFP4 and NVFP4 where the
initialization is better.

## Conclusion

R48 confirms the compression-quality Pareto frontier:
- **4-5 bits (AWQFP4):** PTQ works, matches FP16 quality
- **1.58 bits (Ternary):** PTQ collapses without QAT (PPL 284K)
- **<1 bit (NanoQuant/BTC):** PTQ collapses harder (PPL 152K-45M)

The extreme low-bit methods are implemented and integrated for future
experimentation with QAT or block-level reconstruction. AWQFP4 (R47)
remains the production-recommended quantizer for PTQ scenarios.

---

## R48b — NanoQuant Full Pipeline Implementation (2026-09-06)

### Implemented Features

All four missing NanoQuant pipeline features were implemented:

1. **Hessian-preconditioned ADMM with SVID init** (`novel_quant_r48.py`)
   - `factorize_admm_nanoquant()` function with:
     - Hessian preconditioning via per-channel activation norms
     - Stabilized Cholesky linear solves with symmetrization + diagonal stabilization
     - Cubic rho scheduler (paper default)
     - Configurable outer iterations (default 400, paper-aligned)
     - Sign-based Z-update for direct binary constraint enforcement
     - Magnitude balancing (Appendix A)
     - Alternating least-squares scale optimization (5 iterations)
   - `quantize_model_nanoquant()` now accepts `hessian_norms` dict for
     per-layer activation-aware preconditioning
   - Default `admm_iters` raised from 50 to 400 (paper default)

2. **Latent matrix optimization via STE** (`block_recon.py`)
   - `reconstruct_block()` now optimizes continuous latent matrices
     (`U_soft`, `V_soft`) via straight-through estimator
   - Separate learning rates: binary latent params get 10x LR
   - SGD with momentum (Adam produces NaN with tiny STE gradients)
   - `optimize_binary` flag to disable STE when scales-only is preferred
   - Soft-forward closures use `_ste_sign()` for gradient flow through
     `sign()` barrier

3. **Model-level KL calibration** (`block_recon.py`)
   - `calibrate_kl()` method on `BlockReconstructor`
   - Captures original model logits, optimizes all scale parameters
     across all blocks to minimize KL divergence
   - Uses `F.kl_div(log_softmax(quant), softmax(orig))` for stability
   - Best-finite-state tracking with rollback
   - Handles `CausalLMOutputWithPast` output format
   - Configurable iterations, learning rate

4. **Error propagation mitigation** (`block_recon.py`)
   - `reconstruct_with_error_mitigation()` method
   - Runs original block forward with the current (corrupted) input
     to produce an error-corrected target
   - Optimizes quantized block to match the corrected target instead
     of the stale original target
   - Propagates reconstructed output as input for next block
   - `optimize_binary` flag passes through to `reconstruct_block()`

### ForgeEngine Integration

`forge_engine.py` `block_reconstruct()` method updated:
- New modes: `"progressive"`, `"standard"`, `"error_mitigation"`, `"kl_calib"`
- `kl_calib` mode: progressive block recon + KL calibration
- `kl_iters` parameter for KL calibration iterations
- All modes are opt-in; existing dispatch behavior unchanged

### Validation Results (Qwen 2.5 0.5B, real weights)

#### ADMM Quality Improvement

| ADMM Version | Rank | Iters | Relative Error | Base PPL |
|---|---:|---:|---:|---:|
| Old (alternating binarization) | 128 | 50 | ~1.0 (random) | 7×10²⁰ |
| New (Hessian + sign Z-update + LS scales) | 128 | 200 | 0.78 | 366K |
| New (Hessian + sign Z-update + LS scales) | 256 | 200 | — | 2.88M |
| New (Hessian + sign Z-update + LS scales) | 512 | 200 | — | 1.98M |

The new ADMM produces dramatically better binary factors (0.78 vs ~1.0
relative error on synthetic weights). Base PPL improved from 7×10²⁰ to
366K at rank 128.

#### KL Calibration Results (Best Finding)

| Rank | Bits/w | Base PPL | After KL (lr=0.001, 50 iters) | Improvement |
|---:|---:|---:|---:|---:|
| 128 | 0.221 | 108,766 | 5,845 | 18.6× |
| 256 | 0.418 | 1,488,280 | 6,677 | 223× |
| 512 | 0.811 | 1,979,289 | 3,265 | 607× |

**KL calibration is the most effective technique** — it reduces PPL by
18-607× by optimizing only scale parameters globally. The optimal learning
rate is 0.001; higher rates (0.01, 0.1) diverge.

#### Block Reconstruction Results

| Mode | Base PPL | After Recon | After KL | Notes |
|---|---:|---:|---:|---|
| Standard (independent) | 1.98M | 57,900 | 15,408 | Block recon helps, KL helps more |
| Progressive | 1.98M | — | — | Later blocks diverge (loss ~248) |
| Error mitigation (scales only) | 319K | 6.4M (worse) | — | Corrected target is itself corrupted |
| Error mitigation (binary STE) | 366K | 995K (worse) | — | STE destabilizes binary factors |
| KL only (no block recon) | 1.98M | — | 3,265 | **Best result** |

#### Key R&D Findings

1. **KL calibration alone beats block reconstruction + KL**: Block
   reconstruction can hurt global PPL even when local block losses
   decrease. KL-only is simpler and more effective.

2. **STE latent optimization can destabilize binary factors**: Optimizing
   continuous latent matrices via STE reduces local MSE but can move
   binary factors away from their ADMM-optimal values, hurting global
   quality. The `optimize_binary=False` flag is safer.

3. **Error mitigation's corrected target is corrupted**: Running the
   original block with corrupted input produces a bad target. The
   quantized block then optimizes toward this bad target, making things
   worse.

4. **Higher LR in KL calibration diverges**: lr=0.001 is optimal.
   lr=0.01 gives 1.5× worse PPL. lr=0.1 diverges (PPL → 96M).

5. **Sub-1-bit remains fundamentally hard**: Even with all NanoQuant
   pipeline features, PPL at 0.221 bits/w is 5,845 (270× worse than
   FP16's 21.79). At 0.811 bits/w, PPL is 3,265 (150× worse). This
   confirms that sub-1-bit PTQ on non-BitNet-trained models requires QAT.

6. **AWQFP4 (R47) remains production champion**: PPL 31.625 at 4.53
   bits/w. No sub-2-bit method comes close.

### Production Recommendation

- **For production dispatch**: Use AWQFP4 (R47) at 4.53 bits/w
- **For experimental sub-1-bit**: Use NanoQuant + KL calibration
  (`mode="kl_calib"`, `lr=0.001`, `kl_iters=50`). Be aware PPL will
  be 150-270× worse than FP16.
- **Block reconstruction**: Use `optimize_binary=False` (scales only)
  with standard mode. Avoid error mitigation and progressive modes
  for now — they can hurt global PPL.

### Files Modified

- `forge/engine/quant/novel_quant_r48.py` — ADMM factorization, SVID
  helpers, optimal scale computation, `hessian_norms` parameter
- `forge/engine/quant/block_recon.py` — latent STE optimization,
  `calibrate_kl()`, `reconstruct_with_error_mitigation()`,
  `optimize_binary` flag, CausalLMOutput handling
- `forge/engine/forge_engine.py` — new reconstruction modes
  (`error_mitigation`, `kl_calib`), `kl_iters` parameter
- `scripts/test_block_recon.py` — new CLI modes
- `scripts/test_recon_full.py` — full validation script (new)
- `scripts/test_recon_modes.py` — mode comparison script (new)
- `scripts/test_kl_tuning.py` — KL tuning script (new)
- `tests/unit/test_novel_quant_r48.py` — fixed flaky k-means test

### Test Results

- Unit tests: 137/137 pass (R44+R45+R46+R48)
- Smoke tests: 6/6 pass
- Real model validation: Qwen 2.5 0.5B, all modes tested
