# R&D Round 45: Online Quantization Research — New Ideas (2025-2026)

**Date**: 2026-09-05
**Purpose**: Catalog new quantization algorithms and ideas discovered via web research
that were NOT in the R44 master list, to inform next round of novel implementations.

## New Algorithms Found (not in previous research)

### Weight-Only PTQ — New 2025-2026 Methods

#### 1. REAL-Q (arXiv 2609.00049, Sep 2026)
**Core idea**: Dynamic block-wise gradient descent PTQ. Instead of GPTQ's single
closed-form second-order solver per layer, REAL-Q applies fine-grained gradient
corrections after every 128-column block, refreshing the Hessian as the loss
landscape shifts. Uses a sliding window for cross-layer transitions.
- **Key innovation**: "Information misalignment" — the Hessian frozen at layer
  start becomes stale as columns are quantized. Dynamic refresh fixes this.
- **Results**: Up to 49% KL divergence reduction vs SOTA on LLaMA-3.1/Qwen3 W4A16.
- **Novel twist for us**: Apply gradient descent (not just closed-form) to
  refine quantized weights block-by-block with Hessian refresh.

#### 2. HeRo-Q (arXiv 2601.21626, Jan 2026)
**Core idea**: Hessian-conditioned quantization. Applies a learnable
rotation-compression matrix to the weight space BEFORE quantization, specifically
targeting the largest Hessian eigenvalue. Reduces max eigenvalue → more robust
to quantization noise.
- **Key innovation**: "Low-error, high-loss" paradox — low reconstruction error
  doesn't mean low task loss. The Hessian geometry matters.
- **Results**: Outperforms GPTQ, AWQ, SpinQuant at W4A8 and W3A16. GSM8K on
  Llama-3-8B: 70.15% at W3A16.
- **Novel twist for us**: Learn a rotation that minimizes the max Hessian
  eigenvalue, not just weight magnitude variance (QuaRot/SpinQuant approach).

#### 3. SchurQuant (arXiv 2608.15567, Aug 2026)
**Core idea**: Groupwise discrete optimization with Schur-complement curvature.
Analytically eliminates the unquantized suffix's optimal continuous response,
yielding an exact groupwise quadratic. Alternates closed-form scale/zero-point
refitting with coordinate descent over integer codes.
- **Key innovation**: GPTQ ignores that the remaining unquantized suffix can
  absorb error. Schur-complement accounts for this exactly.
- **Results**: +11.88 pp on 2-bit Qwen3-4B. +9.65 pp over strongest baseline at 2-bit.
- **Novel twist for us**: Schur-complement correction for the "suffix absorption"
  effect in GPTQ-style column-wise quantization.

#### 4. KronQ (arXiv 2607.07964, Jul 2026)
**Core idea**: Kronecker-factored Hessian for PTQ. Uses gradient covariance
(not just activation covariance) in the quantization objective. Bidirectional
incoherence processing — rotates both input AND output dimensions.
- **Key innovation**: GPTQ assumes all output channels contribute equally.
  KronQ uses gradient covariance to weight them differently.
- **Results**: 2-bit LLaMA-3-70B: 7.93 PPL (GPTQ/GPTAQ diverge >2000 PPL).
- **Novel twist for us**: Add gradient covariance to the quantization objective
  and rotate the output dimension (not just input like QuaRot).

#### 5. OCGQuant (arXiv 2609.00066, Aug 2026, EMNLP 2026)
**Core idea**: Outlier-Companion Grouping for NVFP4. Pairs outlier channels with
low-magnitude companion channels within NVFP4 blocks to reduce "Collateral
Quantization Error" — the error incurred by non-outlier values sharing a scale
dominated by an outlier.
- **Key innovation**: Channel grouping/reordering specifically for NVFP4's
  16-element blocks. Adaptive pairing of outliers with companions.
- **Results**: Lowest WikiText-2 PPL among PTQ methods for NVFP4.
- **Novel twist for us**: Reorder channels within blocks to pair outliers with
  low-magnitude companions before NVFP4 quantization.

#### 6. ReQuant (arXiv 2608.07019, Aug 2026)
**Core idea**: Fixed-grid discrete refinement as a post-processing stage. Takes
any existing quantized model and iteratively revisits integer assignments on the
fixed quantization grid, accepting only MSE-reducing moves.
- **Key innovation**: Plug-and-play post-processing. Can refine RTN to approach
  GPTQ quality without any calibration data.
- **Novel twist for us**: Add a ReQuant-style refinement pass after our AB-FP4
  or SR-INT4 quantization.

#### 7. D2Quant (arXiv 2602.02546, Feb 2026)
**Core idea**: Dual-scale quantizer for down-projection matrices (the known
quantization bottleneck). Also corrects activation deviations caused by weight
quantization.
- **Key innovation**: Down-projections get a separate dual-scale quantizer.
  Activation deviation correction after weight quantization.
- **Novel twist for us**: Layer-type-aware quantization — down_proj gets special
  treatment with dual scales.

#### 8. DynamicPTQ (arXiv 2606.12487, Jun 2026)
**Core idea**: Phase-aware mixed-precision activation quantization based on
residual-stream dynamics. Identifies "jump" layers where massive activations
cause quantization instability and gives them 8-bit activations.
- **Key innovation**: "Jump Ratio" and "Historical Feature SNR" metrics to
  identify sensitive layers from residual stream dynamics.
- **Novel twist for us**: Analyze residual stream to determine which layers need
  higher precision, rather than uniform quantization.

### Extreme Low-Bit (<2-bit) — New 2025-2026 Methods

#### 9. BTC-LLM (arXiv 2506.12040, Jun 2025)
**Core idea**: Sub-1-bit quantization via binary codebook + learnable transformation.
Clusters recurring binary weight vectors into compact indices. Learnable transform
reduces outliers and promotes shared sign patterns.
- **Key innovation**: Binary codebook replaces sparse masks (vs STBLLM). No mask
  management overhead. Standard hardware compatible.
- **Results**: 0.8 bits on LLaMA-2-13B with only 3.1% accuracy drop. 1.6x speedup.
- **Novel twist for us**: Binary pattern clustering — find recurring ±1 patterns
  in weights and store them as codebook indices.

#### 10. CCQ (arXiv 2507.07145, Jul 2025)
**Core idea**: Convolutional code quantization. Lookup-free encoding with
bit-shift operations. Convolutional codes + hybrid encoding + code cluster.
- **Key innovation**: Lookup-free VQ — linear mapping between codebook and weight
  vectors via bit-shifts. No LUT needed at inference.
- **Results**: 2-bit DeepSeek-V3 (671B) → 184GB. 2-bit ERNIE-4.5-300B → 89GB
  (single-GPU deployment).
- **Novel twist for us**: Convolutional code encoding instead of codebook lookup
  for fast inference.

#### 11. LittleBit / LittleBit-2 (NeurIPS 2025 / ICML 2026)
**Core idea**: Sub-1-bit via low-rank latent factorization + binarization.
W ≈ UV^T where U,V are binarized. Multi-scale compensation across row, column,
and latent dimensions. LittleBit-2 adds Joint-ITQ for latent geometry alignment.
- **Key innovation**: Low-rank factorization BEFORE binarization. The rank
  controls effective bit-width: rank r → r*(out+in)/(out*in) bits/weight.
- **Results**: 0.1 BPW on LLaMA-2-7B beats leading 0.7 BPW methods. 31x compression.
- **Novel twist for us**: This is essentially what our HadamardLift tried, but
  with QAT (not PTQ) and proper initialization (SVD + Joint-ITQ, not random P).
  **Key lesson**: LiftQuant-style lifting REQUIRES QAT or SVD init, not STE.

#### 12. HBLLM (NeurIPS 2025 Spotlight)
**Core idea**: Haar wavelet transform for 1-bit quantization. Decomposes weights
into high/low frequency components. Frequency-aware grouping + saliency-driven
column selection. Shared mean for non-salient weights.
- **Key innovation**: Wavelet (not Hadamard) rotation for binary quantization.
  Haar is cheaper than Hadamard (O(n) vs O(n log n)) and separates frequency bands.
- **Results**: 1.08 bits on LLaMA-2-13B, PPL 6.71.
- **Novel twist for us**: Replace Hadamard with Haar wavelet in our rotation-based
  methods. Frequency-band-aware grouping for bit allocation.

#### 13. CRVQ (TACL 2025)
**Core idea**: Channel-Relaxed Vector Quantization. Selects and reorders critical
weight channels, uses extended codebooks to relax constraints on those channels.
- **Key innovation**: Critical channel identification + extended codebook for
  just those channels. Minimal extra bits for large quality improvement.
- **Results**: 38.9% improvement over strongest sub-2-bit PTQ baseline.
- **Novel twist for us**: Identify critical channels (by sensitivity) and give
  them extended codebook entries while keeping the rest at standard VQ.

#### 14. RSAVQ (NeurIPS 2025)
**Core idea**: Riemannian sensitivity-aware VQ. Uses Fisher Information Matrix
to project quantization errors onto low-sensitivity directions. Channel-wise
sensitivity for dynamic bit allocation.
- **Key innovation**: Information geometry for VQ. Error direction matters, not
  just error magnitude. FIM-induced Riemannian metric guides projection.
- **Results**: +0.4 PPL over VPTQ and QuIP# at 2-bit LLaMA-3-8B.
- **Novel twist for us**: FIM-based sensitivity for bit allocation in our AB-FP4.

### KV Cache — New 2025-2026 Methods

#### 15. KVarN (arXiv 2606.03458, Jun 2026)
**Core idea**: Hadamard rotation + Sinkhorn variance normalization for KV cache.
Dual-scaling across both axes of K and V matrices. Calibration-free.
- **Key innovation**: Hadamard alone fails for KV cache (doesn't fix token scaling).
  Sinkhorn variance normalization equalizes row/column variance before quantization.
- **Results**: SOTA at 2-bit KV on MATH500, AIME24, HumanEval. vLLM integrated.
- **Novel twist for us**: Sinkhorn variance normalization for our KV quantization.

#### 16. TurboQuant (ICLR 2026, arXiv 2504.19874)
**Core idea**: Random orthogonal rotation + Lloyd-Max optimal scalar quantization
+ QJL residual sign bits. Near-optimal distortion at 2-3 bits for KV cache.
- **Key innovation**: Random rotation → Gaussian distribution → fixed Lloyd-Max
  codebook is provably near-optimal. QJL adds 1-bit residual correction.
- **Results**: 4.6x KV compression at 3-bit with 4.6% PPL degradation.
- **Novel twist for us**: Lloyd-Max codebook for rotated values + QJL residual.

#### 17. SPECTRA (arXiv 2608.07915, Aug 2026)
**Core idea**: Spectral transform coding to push KV cache beyond the 2-bit cliff.
Transforms correlated channels into spectral domain where bit allocation is natural.
- **Key innovation**: Goes beyond scalar quantization — spectral transform coding
  exploits inter-channel correlation in KV cache.
- **Novel twist for us**: Spectral decomposition (DCT/DFT) of KV cache channels
  before quantization.

#### 18. SemKV (arXiv 2608.28911, Aug 2026)
**Core idea**: Semantic mixed-precision KV cache quantization guided by a
"quality cliff" measurement. All-token-preserving (no eviction).
- **Key innovation**: Empirically measures the quality cliff per model and
  allocates precision to stay just above it.
- **Novel twist for us**: Quality-cliff measurement to determine optimal
  mixed-precision allocation.

#### 19. VQKV (arXiv 2603.16435, Mar 2026)
**Core idea**: Vector quantization for KV cache. Thousands of FP values represented
by a few integer indices. Training-free.
- **Key innovation**: VQ for KV (not just weights). High compression + high fidelity.
- **Results**: 82.8% compression on LLaMA3.1-8B, 98.6% performance retained.
- **Novel twist for us**: VQ for KV cache, not just weights.

#### 20. Sequential KV Compression (arXiv 2604.15356, Apr 2026)
**Core idea**: Two-layer compression: probabilistic prefix deduplication + predictive
delta coding. Exploits the fact that KV tokens come from a formal language the model
  predicts.
- **Key innovation**: Sequential compression (not per-vector). Uses model's own
  predictions to encode KV residuals.
- **Novel twist for us**: Predictive delta coding for KV — store prediction residual.

### FP4/Microscaling — New 2025-2026 Methods

#### 21. MR-GPTQ (arXiv 2509.23202, Sep 2025)
**Core idea**: Micro-Rotated GPTQ for MXFP4/NVFP4. Block-wise Hadamard transforms
fused into weights + format-specific optimizations for FP4's unique properties.
- **Key innovation**: NVFP4's small group size (16) neutralizes traditional outlier
  mitigation. MR-GPTQ uses block-wise Hadamard specifically for FP4.
- **Results**: 2.2x end-to-end on B200, 4x on RTX5090. Matches SOTA accuracy.
- **Novel twist for us**: Block-wise Hadamard fused into NVFP4 weights.

#### 22. OAS + MBS (arXiv 2603.08713, Mar 2026)
**Core idea**: Overflow-Aware Scaling (OAS) + Macro Block Scaling (MBS) for MXFP4.
OAS increases dynamic range under power-of-2 scaling. MBS adds coarser-granularity
scales for outliers.
- **Key innovation**: Two-level scaling for MXFP4 (like NVFP4's approach but
  software-only, no hardware change needed).
- **Results**: Closes MXFP4 vs NVFP4 gap from 10% to <1%.
- **Novel twist for us**: Macro block scaling — add a coarser scale level on top
  of our existing block scales.

#### 23. Quartet (arXiv 2505.14669, May 2025)
**Core idea**: Native FP4 training for LLMs. All major computations in FP4 on
Blackwell. New low-precision scaling law.
- **Key innovation**: End-to-end FP4 training (not just inference). CUDA kernels
  for Blackwell.
- **Novel twist for us**: FP4 training path (if we ever do QAT).

### GPTQ Ecosystem — New 2025-2026 Methods

#### 24. GPTAQ / Compensation-Aware Error (arXiv 2604.07955, Apr 2026)
**Core idea**: Rethinks residual errors in compensation-based quantization.
Aligns quantized output with ORIGINAL full-precision output (not compensated output).
Identifies "compensation-aware error" from weight discrepancy.
- **Key innovation**: The residual error comes from BOTH output difference AND
  weight discrepancy, not just output difference.
- **Novel twist for us**: Fix GPTQ's objective to align with original output.

#### 25. GuidedQuant (arXiv 2505.07004, May 2025)
**Core idea**: End-loss gradient guidance for PTQ. Integrates gradient information
from the end loss while preserving cross-weight dependencies. Non-uniform scalar
quantization with monotonic objective decrease.
- **Key innovation**: Gradient-guided non-uniform scalar quantization.
- **Novel twist for us**: Non-uniform quantization grid guided by end-loss gradients.

#### 26. GPTQModel v6 (Apr 2026)
**Core idea**: Production GPTQ with new methods: ParoQuant, FOEM (First-Order
Error Matters), EXL3, FP8, GGUF integration. MoE routing controls.
- **Key innovation**: FOEM — first-order error matters (not just second-order).
  ParoQuant — optimization scope control (module vs layer).
- **Novel twist for us**: First-order error term in the quantization objective.

## Summary of Novel Ideas for Next R&D Round

Ranked by potential impact on our RTX 5070 / 12GB / Qwen 0.5B setup:

### Tier 1 — High impact, implementable now
1. **Haar Wavelet + Binary** (HBLLM): Replace Hadamard with Haar in our dead-end
   lifting methods. Haar is cheaper and separates frequency bands for better
   binary expressivity. Could revive HadamardLift as "WaveletLift".

2. **Schur-Complement Correction** (SchurQuant): Add suffix-absorption correction
   to our GPTQ-style quantization. Exact groupwise quadratic with Schur curvature.

3. **Sinkhorn Variance Normalization** (KVarN): Apply to our KV cache quantization.
   Hadamard + dual-scaling Sinkhorn equalizes variance before 2-bit quantization.

4. **Lloyd-Max + QJL Residual** (TurboQuant): For KV cache. Random rotation →
   Gaussian → fixed Lloyd-Max codebook + 1-bit QJL residual. Calibration-free.

5. **Low-Rank + Binarize with SVD Init** (LittleBit): Our HadamardLift failed
   because of random P + STE. SVD init + Joint-ITQ is the proper way. Even PTQ
   SVD init (without QAT) should be much better than random.

6. **ReQuant Refinement Pass** (ReQuant): Post-processing for our AB-FP4/SR-INT4.
   Iteratively revisits integer assignments, accepting only MSE-reducing moves.

### Tier 2 — Medium impact, needs more work
7. **Outlier-Companion Grouping** (OCGQuant): Channel reordering for NVFP4.
   Pair outliers with low-magnitude companions within blocks.

8. **Gradient Covariance Weighting** (KronQ): Add gradient covariance to
   quantization objective. Bidirectional incoherence (output rotation too).

9. **Convolutional Code Encoding** (CCQ): Lookup-free VQ with bit-shift decode.
   For extreme low-bit weight compression.

10. **Hessian Eigenvalue Minimization** (HeRo-Q): Learn rotation that minimizes
    max Hessian eigenvalue, not just weight variance.

### Tier 3 — Research stage
11. **Spectral Transform Coding** (SPECTRA): DCT/DFT of KV cache channels.
12. **Predictive Delta Coding** (Sequential KV): Model predicts its own KV.
13. **FP4 Native Training** (Quartet): QAT path for Blackwell FP4.
14. **Phase-Aware Mixed Precision** (DynamicPTQ): Residual stream dynamics for
    layer sensitivity.

## Cross-Domain Combination Ideas (Novel!)

1. **WaveletLift**: HBLLM's Haar wavelet + LittleBit's low-rank factorization.
   Decompose W into wavelet bands, factorize each band, binarize factors.
   Frequency-band-aware lifting.

2. **SchurAB-FP4**: SchurQuant's Schur-complement correction + our AB-FP4's
   kurtosis-based bit allocation. The Schur correction handles suffix absorption
   while kurtosis handles per-block precision allocation.

3. **TurboKVarN KV**: TurboQuant's Lloyd-Max + QJL + KVarN's Sinkhorn normalization.
   Rotation → Sinkhorn → Lloyd-Max → QJL residual. Best of both worlds for 2-bit KV.

4. **GradientGuided AB-FP4**: KronQ's gradient covariance + our AB-FP4's kurtosis
   bit allocation. Use gradient sensitivity (not just kurtosis) to decide which
   blocks get more bits.

5. **ReQuant + Everything**: ReQuant is a post-processing pass. Apply it after
   ANY of our quantization methods for free quality improvement.

6. **HaarKurt FP4**: HBLLM's Haar wavelet + our kurtosis-based bit allocation,
   applied to FP4 instead of binary. Frequency-band-aware FP4 with variable
   precision per band.
