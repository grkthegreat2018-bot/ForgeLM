# Weight Extraction from Activations

## What This Is

Techniques for recovering transformer weights from intermediate activations.
These are **known methods** from the model extraction and system identification
literature — not novel research. This document catalogs which techniques apply
to which transformer components, and what the theoretical limits are.

## The Core Idea

Given a trained model with known architecture, if you can observe the input and
output activations of each layer, you can recover the layer's weights:

```
input activations (X) ──[extraction]──→ layer weights (W)
```

For **linear layers** (Y = X @ W^T), this is just least squares — known since
Gauss (1795), applied to neural networks by Tramèr et al. (2016).

For **nonlinear layers** (Y = σ(X @ W^T)), iterative methods are needed —
Newton, Gauss-Newton, or specialized algorithms like Expand-and-Cluster
(Martinelli et al. 2024).

## Literature

| Paper | Contribution | Relevance |
|-------|-------------|-----------|
| Tramèr et al. 2016 | Equation-solving attacks for linear layers | V, O, W_up, W_down |
| Carlini et al. 2020 | Cryptanalytic extraction of ReLU networks | Piecewise-linear structure |
| Martinelli et al. 2024 | Expand-and-Cluster for any activation | W_gate (SiLU) |
| Head-channel non-identif. 2025 | O projection is non-identifiable | O projection limits |
| Provably Learning Attention 2026 | Q/K recovery theory | Q/K limits |

## Component Classification

### Trivial (no extraction needed)

| Component | Method | Notes |
|-----------|--------|-------|
| Embedding | Direct copy | Weight is a lookup table |
| LM Head (tied) | Direct copy | Same as embedding |
| RoPE | Deterministic | No learned weights — fixed rotation |
| Causal mask | Deterministic | No learned weights — fixed pattern |

### Linear (exact via least squares, conditioning-dependent)

| Component | Method | Notes |
|-----------|--------|-------|
| RMSNorm | `weight = mean(Y / X_norm)` | From the RMSNorm definition |
| V projection | `lstsq(X, Y)` | Tramèr 2016. Exact in float64 |
| W_up | `lstsq(X, Y)` | Tramèr 2016. Exact in float64 |
| O projection | `lstsq(X, Y)` | **Rank-deficient** — head-channel non-identifiability (2025) |
| W_down | `lstsq(X, Y)` | **Underdetermined** — intermediate_size >> seq_len |

**Numerical notes:**
- Use `lstsq` (QR decomposition), NOT `pinv` (SVD) — QR is more stable
- Use float64 for the solve — float32 fails at condition numbers > 1e8
- Center data before solving with bias — avoids collinearity with constant

### Nonlinear (iterative methods required)

| Component | Method | Notes |
|-----------|--------|-------|
| W_gate (SiLU) | Gauss-Newton / Expand-and-Cluster | SiLU is non-monotonic — no closed-form inverse |
| Q/K (softmax) | Gradient descent | Multi-head is **NOT identifiable** (2026 result) |

**W_gate specifics:**
- SiLU(x) = x * sigmoid(x) has a minimum at x ≈ -3.43 (silu_min ≈ -0.278)
- ~60% of gate targets fall in the ambiguous region [silu_min, 0) where two solutions exist
- Direct silu inversion fails — iterative methods are required
- Expand-and-Cluster (Martinelli 2024) handles SiLU's covert symmetry

**Q/K specifics:**
- Single-head: exactly recoverable with O(d²) queries (2026 paper)
- Multi-head: **provably NOT identifiable** — distinct parameterizations give same output
- "Fast uptraining" (GD on Q/K only) is just training a subset of parameters

## What Doesn't Work (and Why)

### O projection — rank-deficient
Multi-head attention sums per-head outputs through O projection. After summation,
individual head contributions are irrecoverable (head-channel non-identifiability).
The attention output matrix has ~640/1536 near-zero eigenvalues in practice.
Adding more data doesn't help — the rank deficiency is structural.

### W_down — underdetermined
The intermediate dimension (8960 for Qwen2.5-1.5B) far exceeds any feasible
sequence length. The system has fewer equations than unknowns — infinitely
many solutions exist. Minimum-norm (lstsq) gives one solution, but not the
original weights.

### Multi-head Q/K — non-identifiable
Multiple distinct (W_q, W_k) pairs produce identical attention patterns.
The 2026 paper proves this formally. Single-head is identifiable, multi-head is not.

## Two-Pronged Strategy

### Prong 1: Direct Weight Transforms (cross-arch port)

Transform weights **directly** from one architecture to another, without
going through the impossible extraction step. This is the viable path for
cross-architecture porting.

| Transform | Source → Target | Method | Status |
|-----------|----------------|--------|--------|
| SVD resize | Large model → smaller (same family) | Shared SVD projection from embedding | `convert_key_svd.py` |
| BitNet | bf16 → ternary {-1,0,+1} | absmean quantization | `convert_keys.py` — **WORKING** (2.29 GB, 26% smaller) |
| MLA | GQA K/V → low-rank compression | SVD on unexpanded K/V + head-preserving expansion | **WORKING** (cos=0.9999 at d_c=512) |
| MoE weights | Dense FFN → routed experts | Split intermediate + w_down scaling | **WORKING** (cos=1.0 all-experts, exact) |
| MoE router | Uniform → centroid routing | Weight slice centroids → router W | `moe_router_key.py` — **KEY** (PARTIAL, better than uniform) |
| SpinQuant | Rotation before quant | Fixed Hadamard (QuaRot) — no learning | `spinquant_key.py` — **KEY** (TRIVIAL, no calibration) |
| GateSkip | Token-wise layer skip | Gate from activation delta on calib data | `gateskip_key.py` — **KEY** (PARTIAL, needs calib) |
| MTP | Multi-token prediction | Copy LM head + temporal shift | `mtp_key.py` — **KEY** (FULL, head 1 exact) |
| DSpark | Semi-autoregressive spec decode | MTP key + zero-init RNN | `dspark_key.py` — **KEY** (PARTIAL, RNN needs fine-tune) |
| SSA | Sparse attention | Top-k from attention mass on calib data | `ssa_key.py` — **KEY** (PARTIAL, needs calib) |
| RotorQuant | KV cache compression | Fixed Givens rotation — no learning | `rotorquant_key.py` — **KEY** (TRIVIAL, 3.88x compression) |
| Liquid conv | Attention → short conv | Local attention pattern → conv kernel | `liquid_conv_key.py` — **KEY** (PARTIAL, needs calib) |

**Key classes (3 criteria for FULL):**
1. **Reversible**: `reverse(weights) → data` — can recover the input before the transform
2. **Data→weight**: `forward(data) → weights` — produces weights without traditional training
3. **Composable**: chains with other keys for weight-to-weight cross-arch conversion

| Class | Count | Criteria met | Keys |
|-------|-------|-------------|------|
| **FULL** | 7 | All 3 | MTP, Value Residual, SliceGPT, MRL, RotorQuant, SpinQuant, QuaRot R2 |
| **BI** | 5 | 1+2 (existing in source, exact copy) | Embedding, RMSNorm, LM Head, RoPE, Causal Mask |
| **PARTIAL** | 9 | 2 only (weight transform, not reversible) | GQA→MQA, Wanda, DSpark, MoE Router, SSA, GateSkip, Liquid Conv, SparDA, PartialRoPE |

**Architecture config keys (NOT in KeyStack — they define the target arch but don't transform weights):**
mHC, YaRN, StreamingLLM, KIVI, Gluon, DenseFormer, LearnedSink, ALiBi, MoD, MatFormer,
SwiGLU Clamp, SandwichNorm, Sliding Window, SnapKV. These are kept as files for reference.

**Unified pipeline**: `python -m research.forge_pipeline` chains MLA → MoE → BitNet.
Add `--phase2` for GateSkip + MTP + SpinQuant + fine-tuning (needs GPU).

These work because they transform the weight matrices directly — no need to
recover weights from activations. The math is well-defined and lossless (or
lossy with known bounds).

### Prong 2: Extraction + Fine-tune (same-arch, skip heavy training)

For same-architecture weight recovery (e.g. cloning a model from activations):

1. **Extract what's exact**: RMSNorm, V, W_up (float64 lstsq) — these work
2. **Approximate the rest**: W_gate (Gauss-Newton init), then fine-tune
3. **Copy the impossible**: O, W_down, Q/K — copy from source, fine-tune if needed
4. **Fine-tune**: short training run to close the gap on approximate weights

This is **distillation with a good initialization** — the extracted weights
provide a starting point much closer to the target than random init, reducing
training cost. It does NOT eliminate training entirely.

### Why the "data bridge" approach is dead

The original idea was: extract weights → data → re-encode into new architecture.
This fails because extraction is incomplete (O, W_down, Q/K are provably
impossible to recover). Direct weight transforms (Prong 1) bypass this by
transforming weights directly, never going through the lossy extraction step.

## Code

- `research/weight_extraction.py` — extraction functions (Prong 2)
- `research/convert_keys.py` — BitNet quantization + MLA + MoE (Prong 1)
- `research/convert_key_svd.py` — SVD resize (Prong 1)
- `research/keys/` — Key classes (base, keystack, all component keys)
- `research/keys/keystack.py` — `build_qwen2_keystack()`, `build_xp_keystack()`
- `.devin/test_all_keys.py` — comprehensive key verification (8/8 pass)
