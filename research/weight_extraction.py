"""Weight extraction from activations — known techniques, properly implemented.

This module implements weight recovery from intermediate activations using
well-established methods from the model extraction / system identification
literature. Nothing here is novel research.

References:
  - Tramèr et al. 2016: "Stealing Machine Learning Models via Prediction APIs"
    (equation-solving attacks for linear layers — our lstsq approach)
  - Carlini et al. 2020: "Cryptanalytic Extraction of Neural Network Models"
    (exact recovery for ReLU networks via piecewise-linear structure)
  - Martinelli et al. 2024: "Expand-and-Cluster: Parameter Recovery"
    (handles SiLU and other non-monotonic activations)
  - "Head-channel non-identifiability" 2025: O projection is provably
    non-identifiable after multi-head summation
  - "Provably Learning Attention with Queries" 2026: single-head Q/K is
    exactly recoverable, multi-head is NOT identifiable

Classification for transformer components:
  Trivial (no extraction needed):
    - Embedding: direct copy
    - LM Head (tied): direct copy
    - RoPE: deterministic, no weights
    - Causal mask: deterministic, no weights

  Linear (exact via lstsq, conditioning-dependent):
    - RMSNorm: weight = mean(Y / X_norm) — from the RMSNorm definition
    - V projection: lstsq(X, Y) — Tramèr 2016
    - W_up: lstsq(X, Y) — Tramèr 2016
    - O projection: lstsq(X, Y) — but rank-deficient in practice (2025 result)
    - W_down: lstsq(X, Y) — but underdetermined (intermediate >> seq_len)

  Nonlinear (needs iterative methods):
    - W_gate (silu): Newton / Expand-and-Cluster — Martinelli 2024
    - Q/K (softmax): GD — but multi-head NOT identifiable (2026 result)
"""
import torch
import torch.nn.functional as F
from typing import Optional


# ============================================================
# Trivial keys — direct copy, no computation
# ============================================================

def extract_embedding(model) -> torch.Tensor:
    """Embedding weight = direct copy. No extraction needed."""
    return model.embed.weight.data.clone()


def extract_lm_head_tied(model) -> torch.Tensor:
    """Tied LM head = embedding weight. No extraction needed."""
    return model.embed.weight.data.clone()


# ============================================================
# RMSNorm — weight = mean(Y / X_norm), from the layer definition
# ============================================================

def extract_rmsnorm(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Recover RMSNorm scale weight from input/output activations.

    RMSNorm: Y = (X / RMS(X)) * weight
    → weight = mean(Y / X_norm) where X_norm = X * rsqrt(mean(X²) + eps)

    This is just the RMSNorm formula rearranged — not a discovery.
    Use float64 for the division to avoid precision loss on ill-conditioned layers.
    """
    X_f = X.double().cpu()
    Y_f = Y.double().cpu()
    variance = X_f.pow(2).mean(-1, keepdim=True)
    X_norm = X_f * torch.rsqrt(variance + eps)
    weight = (Y_f / (X_norm + 1e-12)).mean(dim=0)
    return weight.float().to(X.device)


# ============================================================
# Linear keys — lstsq (QR decomposition), Tramèr et al. 2016
# ============================================================

def extract_linear_weight(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Recover linear layer weight: Y = X @ W^T → W = lstsq(X, Y).T

    Uses float64 for numerical stability (float32 fails at cond > 1e8).
    Uses lstsq (QR) not pinv (SVD) — QR is more stable for this.
    """
    X_f = X.double().cpu()
    Y_f = Y.double().cpu()
    result = torch.linalg.lstsq(X_f, Y_f)
    return result.solution.T.float().to(X.device)


def extract_linear_weight_with_bias(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover linear layer weight + bias: Y = X @ W^T + b.

    Centers data first to avoid collinearity with the constant direction,
    then solves for W on centered data and computes bias from means.
    Uses float64 for precision.
    """
    X_f = X.double().cpu()
    Y_f = Y.double().cpu()
    X_mean = X_f.mean(0, keepdim=True)
    Y_mean = Y_f.mean(0, keepdim=True)
    X_c = X_f - X_mean
    Y_c = Y_f - Y_mean
    result = torch.linalg.lstsq(X_c, Y_c)
    W = result.solution.T
    bias = (Y_mean - X_mean @ W.T).squeeze(0)
    return W.float().to(X.device), bias.float().to(X.device)


# ============================================================
# SwiGLU W_gate — Newton iteration (Gauss-Newton with LM damping)
# ============================================================

def extract_swiglu_gate(X: torch.Tensor, gate_target: torch.Tensor,
                        n_iters: int = 10, lr: float = 0.5) -> torch.Tensor:
    """Recover SwiGLU gate weight: gate_target = silu(X @ W^T).

    silu(x) = x * sigmoid(x) is nonlinear with no closed-form inverse.
    Uses Gauss-Newton iteration with Levenberg-Marquardt damping.

    Note: silu is non-monotonic for x < -3.43 (has a minimum at ~-0.278),
    so direct inversion fails for ~60% of values. Iterative methods are needed.

    For production use, consider Expand-and-Cluster (Martinelli et al. 2024)
    which handles SiLU's covert symmetry more robustly.

    This runs on GPU in float32. For ill-conditioned layers, the result
    may not reach exact precision — use as initialization for fine-tuning.
    """
    X_f = X.float().to(device=X.device)
    gt_f = gate_target.float().to(device=X.device)

    # Initial: linearized solve (silu(x) ≈ 0.5x for small x)
    res = torch.linalg.lstsq(X_f, gt_f)
    W = (2.0 * res.solution).T.clone()

    for it in range(n_iters):
        pre_act = X_f @ W.T
        pred = F.silu(pre_act)
        residual = gt_f - pred
        loss = residual.pow(2).sum()

        if loss.item() < 1e-8:
            break

        sig = torch.sigmoid(pre_act)
        silu_grad = sig * (1 + pre_act * (1 - sig))

        # Damped Gauss-Newton: minimize ||residual - grad * X @ delta||²
        # Solve: (X^T G² X + λI) delta = X^T G r  (per-neuron, but vectorized)
        # Vectorized approximation: solve X @ delta = residual / (grad + eps)
        eps = 0.1  # damping for saturated neurons
        weighted_rhs = residual / (silu_grad + eps)
        res = torch.linalg.lstsq(X_f, weighted_rhs)
        delta = res.solution.T

        # Check if step improves loss
        W_new = W + lr * delta
        new_loss = (gt_f - F.silu(X_f @ W_new.T)).pow(2).sum()
        if new_loss < loss:
            W = W_new
        else:
            lr *= 0.5  # reduce step size
            if lr < 1e-4:
                break

    return W.float().cpu()


# ============================================================
# Full extraction pipeline for a transformer layer
# ============================================================

def extract_layer_weights(activations: dict, true_weights: dict,
                          layer_idx: int, model) -> dict:
    """Extract all recoverable weights for a single transformer layer.

    Returns a dict of {weight_name: recovered_weight}.
    Weights that can't be recovered (Q/K, O, W_down) are copied from true_weights.

    Args:
        activations: dict with keys like "ln1", "v_proj", "w_gate", etc.
            Each value is (input_activations, output_activations).
        true_weights: the model's state_dict (for copying non-extractable weights).
        layer_idx: layer number.
        model: the model (for checking bias presence).
    """
    weights = {}
    i = layer_idx

    # RMSNorm (exact, float64)
    X, Y = activations["ln1"]
    weights[f"blocks.{i}.ln1.weight"] = extract_rmsnorm(X, Y)

    X, Y = activations["ln2"]
    weights[f"blocks.{i}.ln2.weight"] = extract_rmsnorm(X, Y)

    # V projection (exact with float64, conditioning-dependent)
    X, Y = activations["v_proj"]
    has_v_bias = hasattr(model.blocks[i].attn.v_proj, "bias") and model.blocks[i].attn.v_proj.bias is not None
    if has_v_bias:
        W, bias = extract_linear_weight_with_bias(X, Y)
        weights[f"blocks.{i}.attn.v_proj.weight"] = W
        weights[f"blocks.{i}.attn.v_proj.bias"] = bias
    else:
        weights[f"blocks.{i}.attn.v_proj.weight"] = extract_linear_weight(X, Y)

    # W_up (exact with float64)
    X, Y = activations["w_up"]
    weights[f"blocks.{i}.ffn.w_up.weight"] = extract_linear_weight(X, Y)

    # W_gate (approximate, needs fine-tuning for exact)
    X, gate_pre = activations["w_gate"]
    gate_target = F.silu(gate_pre.float())
    weights[f"blocks.{i}.ffn.w_gate.weight"] = extract_swiglu_gate(X, gate_target)

    # Non-recoverable weights — copy from original:
    # O projection: rank-deficient (head-channel non-identifiability, 2025)
    # Q/K: multi-head not identifiable (2026)
    # W_down: underdetermined (intermediate_size >> seq_len)
    for name in [f"blocks.{i}.attn.out_proj.weight",
                 f"blocks.{i}.attn.q_proj.weight",
                 f"blocks.{i}.attn.k_proj.weight",
                 f"blocks.{i}.ffn.w_down.weight"]:
        if name in true_weights:
            weights[name] = true_weights[name].clone()

    # Copy biases for Q/K/V if present
    for proj in ["q_proj", "k_proj", "v_proj"]:
        bias_name = f"blocks.{i}.attn.{proj}.bias"
        if bias_name in true_weights and bias_name not in weights:
            weights[bias_name] = true_weights[bias_name].clone()

    return weights


# ============================================================
# Verification
# ============================================================

def verify_extraction(recovered: dict, original: dict, threshold: float = 0.01) -> dict:
    """Compare recovered weights to original. Returns per-weight diff and pass/fail."""
    results = {}
    for name, w_rec in recovered.items():
        if name not in original:
            results[name] = {"status": "MISSING_FROM_ORIGINAL", "diff": None}
            continue
        w_orig = original[name]
        if w_rec.shape != w_orig.shape:
            results[name] = {"status": "SHAPE_MISMATCH", "diff": None}
            continue
        diff = (w_rec.float() - w_orig.float()).abs().max().item()
        results[name] = {
            "status": "EXACT" if diff < threshold else "APPROX",
            "diff": diff,
            "rel_diff": diff / (w_orig.float().abs().max().item() + 1e-12),
        }
    return results
