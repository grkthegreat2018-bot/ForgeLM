# Quantization Research — 2026 State of the Art

## What ForgeAI already has
- **BitNet b1.58** (ternary QAT, STE, int8@int8 CUDA kernel + Triton b1.58 add-only kernel)
- **W8A8 INT8** (weight-only, `QuantizedLinear`, `FastINT8Linear`)
- **INT4 weight-only** (group_size=128, `quantize_model_int4`)
- **NLRQ** (Non-Linear RQ, used in V7-8B training, INT8 factor format)
- **OffQ, AAAC, SharQ, MosaicQuant** (various PTQ/QAT variants)
- **KV compression**: rotorquant, kv_2bit, paged_kv, kv_compress, kvzip
- **FP8 inference** (`fp8_infer.py`)

## What's missing / better (from web research, 2025-2026)

### Tier 1: High-impact, directly relevant to RTX 5070 SM120

1. **NVFP4 (hardware FP4 on Blackwell SM120)**
   - 4-bit floating-point (E2M1) with FP8 (E4M3) micro-scale per 32 elements
   - **RTX 5070 supports this natively** via CUTLASS Example 79b
   - 3x speedup over FP16 on RTX 5090, 4x memory savings
   - SM120 limitation: `kind::mxf8f6f4` stores 4-bit in 8-bit container (half throughput vs SM100)
   - **ARCQuant** (ACL 2026): augmented residual channels, SOTA NVFP4 accuracy, 3x speedup on RTX 5090/6000
   - **ScaleSweep** (arXiv 2606.07618): better scale initialization for NVFP4
   - **MR-GPTQ** (arXiv 2509.23202): block-wise Hadamard + format-specific for FP4
   - Code: CUTLASS 3.8 Example 79b, VincentKaufmann/fp4-cuda-kernel (SM120, 143 TFLOPS)
   - vLLM PR #21309 adds NVFP4 W4A4 for SM120

2. **QuEST (QAT, 1-bit to 4-bit, Pareto-optimal at 4-bit)**
   - Hadamard normalization + MSE-optimal fitting + trust gradient estimator
   - 4-bit W+A training is Pareto-optimal vs FP16 (better accuracy at lower size)
   - Stable down to 1-bit weights AND activations
   - Code: github.com/IST-DASLab/QuEST (updated May 2025)
   - **This is the QAT successor to BitNet** — better gradients, works at 4-bit not just ternary

3. **FlatQuant (ICML 2025, W4A4KV4 SOTA)**
   - Learnable affine transformations (Kronecker-decomposed) per layer
   - <1% accuracy drop on W4A4 for LLaMA-3-70B (beats SpinQuant by 7.5%)
   - 2.3x prefill speedup, 1.7x decode speedup
   - Fused into single kernel, minimal overhead
   - Code: github.com/ruikangliu/FlatQuant

### Tier 2: Strong PTQ methods

4. **SpinQuant** (learned rotations, W4A4 KV4)
   - Narrows gap to FP to 2.9 points on LLaMA-2-7B
   - Outperforms QuaRot (random rotations) by 45.1% on LLaMA-3-8B
   - Already partially in ForgeAI (Hadamard in rotorquant)

5. **ParoQuant** (pairwise Givens rotations, 2025)
   - 2.4% accuracy improvement over AWQ on reasoning tasks
   - <10% overhead, co-designed inference kernel
   - Good for reasoning LLMs (long CoT chains)

6. **HeRo-Q** (Hessian conditioning, 2026)
   - Joint rotation-compression, reduces largest Hessian eigenvalue
   - Beats GPTQ, AWQ, SpinQuant in W4A8 and W3A16

7. **FPTQuant** (function-preserving transforms, 2025)
   - 4 mergeable transforms (pre-RoPE, value, MLP, dynamic scaling)
   - 3.9x speedup, no custom kernels needed
   - Static INT4 with minimal overhead

### Tier 3: Training-focused

8. **StableQAT** (Microsoft, 2026)
   - Fourier-analysis-based surrogate for rounding (generalizes STE)
   - Stable 2-4 bit QAT, negligible overhead
   - Code: github.com/microsoft/StableQAT

9. **SiLQ** (Simple LLM QAT, 2025)
   - <0.1% training budget increase, STE + LSQ step refinement
   - Beats best PTQ methods on CSR + OLLM benchmarks
   - Dead simple to implement

10. **PE-QAT** (ACL 2026 SRW)
    - LoRA adapters + fake quant on merged weights
    - 0.11pp of FP baseline, trains only 1.26% of params
    - Scales QAT to large models

11. **Lattice VQ** (PMLR 2026)
    - E8/D4 lattice vector quantization, stable below 2 bits
    - Geometric structure reduces overload

## Recommendations for ForgeAI (RTX 5070, 12GB, SM120)

### Immediate (highest ROI):
1. **NVFP4 inference path** — the RTX 5070 has hardware FP4 tensor cores.
   CUTLASS 79b + VincentKaufmann's kernel prove 143 TFLOPS on SM120.
   This would make the 8B model fit in ~4GB (vs 16GB bf16) and run 3x faster.
   → New file: `research/quantization/nvfp4.py`

2. **QuEST for training** — replace/augment BitNet with QuEST's trust gradient.
   4-bit W+A training is Pareto-optimal. The Hadamard normalization is already
   partially in ForgeAI (rotorquant). The trust gradient estimator is the novel piece.
   → New file: `research/keys/compression/quest_key.py`

3. **FlatQuant for PTQ** — learnable affine transforms for W4A4.
   Best W4A4 accuracy published. Kronecker-decomposed = low overhead.
   → New file: `research/quantization/flatquant.py`

### Novel twists for ForgeAI:
- **NVFP4 + FreeToken overlap**: run FP4 GEMM on GPU while CPU does
  fp32 optimizer update (the 3-stage pipeline we just built). The 4x
  smaller weights mean 4x less PCIe transfer for the grad/master sync.
- **QuEST trust gradient + CPUAdamW**: the trust gradient estimator
  could improve the STE in BitNet training, especially at 1-bit.
- **FlatQuant + NVFP4**: FlatQuant's affine transforms + NVFP4 hardware
  format = best accuracy at 4-bit with hardware acceleration.
