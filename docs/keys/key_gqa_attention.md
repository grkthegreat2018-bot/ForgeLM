# GQA Attention: Q, K, V, O Projections

## Overview

Grouped-Query Attention has four linear projections:
- **Q projection** (query): `Y = X @ W_q^T` — feeds softmax
- **K projection** (key): `Y = X @ W_k^T` — feeds softmax
- **V projection** (value): `Y = X @ W_v^T` — linear, extracted via lstsq
- **O projection** (output): `Y = attn_out @ W_o^T` — linear, but rank-deficient

Plus structural KV head sharing (GQA groups K/V heads, shared via duplication).

## V Projection — Exact (lstsq)

V is a plain linear layer: `Y = X @ W^T`. Recover via least squares.

```python
W_v = extract_linear_weight(X, Y)  # lstsq in float64
```

**Status:** Exact with float64. Conditioning-dependent — some layers need float64
to avoid lstsq failure at condition numbers > 1e8.

**Reference:** Tramèr et al. 2016 (equation-solving attacks for linear layers).

## O Projection — Rank-Deficient (NOT recoverable)

The O projection input is the concatenated multi-head attention output. After
multi-head summation, individual head contributions are irrecoverable — this is
**head-channel non-identifiability** (2025 result).

In practice, the attention output matrix has ~640/1536 near-zero eigenvalues.
`lstsq` produces a minimum-norm solution, but it is NOT the original weight.

**Status:** NOT recoverable. Copy from source, fine-tune if needed.

**Reference:** "Head-channel non-identifiability" (2025) — after multi-head
attention sums per-head outputs through O projection, individual contributions
cannot be canonically attributed to a specific head.

## Q/K Projections — NOT Identifiable (multi-head)

Q and K feed into softmax attention. Two issues:

1. **Softmax nonlinearity**: Q/K outputs go through softmax, which is nonlinear.
   Single-head Q/K is exactly recoverable with O(d²) queries (2026 paper),
   but requires iterative methods (gradient descent).

2. **Multi-head non-identifiability**: Multiple distinct (W_q, W_k) pairs
   produce identical attention patterns. The 2026 paper proves this formally.
   Multi-head Q/K is **provably NOT identifiable**.

"Fast uptraining" (GD on Q/K only) is just training a subset of parameters —
it works but is not "extraction."

**Status:** NOT recoverable (multi-head). Copy from source, fine-tune if needed.

**Reference:** "Provably Learning Attention with Queries" (2026) — single-head
exactly recoverable, multi-head NOT identifiable.

## GQA Structure — Trivial

KV head sharing is a structural property (how many K/V heads vs Q heads).
The shared K/V weights are duplicated across Q head groups. No extraction —
just reshape/duplicate.

**Status:** Trivial. Structural, no extraction.

## References

- Tramèr et al. 2016: "Stealing Machine Learning Models via Prediction APIs"
- Head-channel non-identifiability 2025: O projection limits
- Provably Learning Attention 2026: Q/K recovery theory
