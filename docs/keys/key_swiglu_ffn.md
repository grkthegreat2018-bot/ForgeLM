# SwiGLU FFN: W_gate, W_up, W_down

## Overview

SwiGLU FFN has three linear projections:
- **W_gate**: `gate = silu(X @ W_gate^T)` — nonlinear (silu activation)
- **W_up**: `up = X @ W_up^T` — linear, extracted via lstsq
- **W_down**: `down = (gate * up) @ W_down^T` — linear, but underdetermined

## W_up — Exact (lstsq)

W_up is a plain linear layer: `Y = X @ W^T`. Recover via least squares.

```python
W_up = extract_linear_weight(X, Y)  # lstsq in float64
```

**Status:** Exact with float64.

**Reference:** Tramèr et al. 2016.

## W_gate — Approximate (Gauss-Newton)

W_gate feeds into silu: `gate_target = silu(X @ W_gate^T)`.

Silu(x) = x * sigmoid(x) is nonlinear with no closed-form inverse. Worse:
- silu has a minimum at x ≈ -3.43 (silu_min ≈ -0.278)
- ~60% of gate targets fall in [silu_min, 0) where TWO solutions exist
- Direct silu inversion is fundamentally ambiguous in this region

Iterative methods (Gauss-Newton with LM damping) provide an approximation:

```python
W_gate = extract_swiglu_gate(X, gate_target, n_iters=10)
```

This gives a good initialization but may not reach exact precision on the real
model. For exact recovery, consider Expand-and-Cluster (Martinelli 2024),
which handles SiLU's covert symmetry more robustly.

**Status:** Approximate. Use as initialization for fine-tuning.

**Reference:** Martinelli et al. 2024: "Expand-and-Cluster: Parameter Recovery
of Neural Networks" — handles SiLU and other non-monotonic activations.

## W_down — Underdetermined (NOT recoverable)

W_down maps from the intermediate dimension (8960 for Qwen2.5-1.5B) back to
the model dimension (1536). The input to W_down is `gate * up`, which has
dimension 8960.

The system `Y = (gate * up) @ W_down^T` has 8960 unknowns per output dimension
but only `seq_len` equations. For any feasible sequence length (<< 8960),
the system is massively underdetermined — infinitely many solutions exist.

`lstsq` gives the minimum-norm solution, but it is NOT the original weight.

**Status:** NOT recoverable. Copy from source, fine-tune if needed.

## References

- Tramèr et al. 2016: equation-solving attacks (W_up)
- Martinelli et al. 2024: Expand-and-Cluster (W_gate)
- Basic linear algebra: underdetermined systems (W_down)
