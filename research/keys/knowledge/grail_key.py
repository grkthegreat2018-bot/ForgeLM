"""GRAIL Compensation Key — heal lossy transforms via Gram matrix + ridge regression.

Based on GRAIL (2026): "Post-hoc Compensation by Linear Reconstruction
for Compressed Networks"

After ANY lossy weight transform (pruning, WQ elimination, expert merging),
this key:
  1. Runs calibration data through original and transformed model
  2. Collects hidden activations at each layer boundary
  3. Computes Gram matrix: G = X_orig^T @ X_transformed
  4. Solves ridge regression: R = (X_trans^T @ X_trans + λI)^{-1} @ X_trans^T @ X_orig
  5. Absorbs reconstruction map R into the downstream layer's weights

The reconstruction map R is a (d_model × d_model) matrix that maps
transformed activations back to the original activation space.
When absorbed into the downstream Linear W: W' = W @ R

This makes lossy transforms (WQ elim, Wanda pruning, expert consolidation)
much less lossy — potentially near-lossless with enough calibration data.

Key class: PARTIAL — requires calibration data, not reversible.

Usage:
    from research.keys.grail_key import GRAILKey, apply_grail_compensation
    # After a lossy transform, heal with GRAIL
    state = apply_grail_compensation(state, model_orig, model_transformed,
                                      calib_data, n_layers=28)
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class GRAILKey(Key):
    """GRAIL Compensation key — heal lossy transforms via ridge regression.

    Collects activations from original and transformed models,
    computes reconstruction map, absorbs into downstream weights.

    Key class: PARTIAL — requires calibration data.
    """

    def __init__(self, n_calib_tokens: int = 512, ridge_lambda: float = 1e-4,
                 seq_len: int = 128):
        self.n_calib_tokens = n_calib_tokens
        self.ridge_lambda = ridge_lambda
        self.seq_len = seq_len

    @property
    def name(self) -> str:
        return "grail_compensation"

    @property
    def description(self) -> str:
        return "Heal lossy transforms via Gram matrix + ridge regression (GRAIL 2026)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """GRAIL requires calibration data — use apply_grail_compensation instead."""
        return KeyResult(
            success=True, weights=data,
            metadata={"note": "Use apply_grail_compensation for actual healing"},
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data=weights,
                         metadata={"reversible": False})

    def compute_reconstruction_map(self, x_orig: torch.Tensor,
                                    x_transformed: torch.Tensor) -> torch.Tensor:
        """Compute ridge regression reconstruction map.

        R = (X_t^T @ X_t + λI)^{-1} @ X_t^T @ X_o

        Such that X_o ≈ X_t @ R

        Args:
            x_orig: (N, d) original activations
            x_transformed: (N, d) transformed activations

        Returns:
            R: (d, d) reconstruction map
        """
        d = x_transformed.shape[-1]
        Xt = x_transformed.float()  # (N, d)
        Xo = x_orig.float()  # (N, d)

        # Ridge regression: R = (Xt^T Xt + λI)^{-1} Xt^T Xo
        XtX = Xt.t() @ Xt  # (d, d)
        XtX += self.ridge_lambda * torch.eye(d, device=Xt.device)
        XtXo = Xt.t() @ Xo  # (d, d)

        R = torch.linalg.solve(XtX, XtXo)  # (d, d)
        return R

    def absorb_into_downstream(self, state: dict[str, torch.Tensor],
                               R: torch.Tensor, layer_idx: int) -> dict[str, torch.Tensor]:
        """Absorb reconstruction map R into downstream layer weights.

        If the downstream layer is W (reads from the residual stream),
        then W' = W @ R (apply R to the input).

        For attention: fold R into out_proj (output writes to residual)
        Actually: R maps transformed→original, so we apply R BEFORE the
        downstream layer: W' = W @ R

        For pre-norm transformers, the downstream of layer i is:
          - ln2 → FFN (for the FFN sublayer)
          - ln_{i+1} → attn (for the next layer's attention)
          - ln_f → head (for the final output)

        We fold R into the readers of the NEXT sublayer:
          - After attention: fold R into ln2 (scale) then FFN reads from it
          - After FFN: fold R into ln1 of next layer
          - After last layer: fold R into ln_f then head
        """
        # Fold R into ln2 weight (next sublayer's pre-norm)
        ln2_key = f"blocks.{layer_idx}.ln2.weight"
        if ln2_key in state:
            # R is applied as: x' = x @ R, then ln2(x') = x' / rms * gamma
            # We can't fold a full matrix into a diagonal norm weight.
            # Instead, fold R into the FFN weights directly.
            pass

        # Fold R into FFN w_gate and w_up (they read from the residual stream)
        for reader in ["w_gate", "w_up"]:
            rk = f"blocks.{layer_idx}.ffn.{reader}.weight"
            if rk in state:
                w = state[rk].float()
                # W reads from residual: output = W @ x
                # With R: output = W @ (x @ R) = (W @ R) @ x
                # So: W' = W @ R (right-multiply)
                state[rk] = (w @ R).to(state[rk].dtype)

        # Also fold into MoE experts
        for ei in range(10):
            for part in ["w_gate", "w_up"]:
                rk = f"blocks.{layer_idx}.ffn.experts.{ei}.{part}.weight"
                if rk in state:
                    w = state[rk].float()
                    state[rk] = (w @ R).to(state[rk].dtype)

        # For the last layer, fold R into ln_f then head
        # Since ln_f is diagonal, we fold R into head directly
        # (This is handled separately for the final layer)

        return state


def apply_grail_compensation(state: dict[str, torch.Tensor],
                              model_orig: torch.nn.Module,
                              model_transformed: torch.nn.Module,
                              calib_tokens: torch.Tensor,
                              n_layers: int,
                              device: str = "cuda",
                              ridge_lambda: float = 1e-4) -> dict[str, torch.Tensor]:
    """Apply GRAIL compensation to heal a lossy transform.

    Args:
        state: transformed model state dict (will be modified)
        model_orig: original model (before transform)
        model_transformed: model with transformed weights
        calib_tokens: (N,) token IDs for calibration
        n_layers: number of transformer layers
        device: compute device
        ridge_lambda: ridge regression regularization

    Returns:
        healed state dict
    """
    key = GRAILKey(ridge_lambda=ridge_lambda)

    model_orig.eval()
    model_transformed.eval()

    # Collect activations layer by layer
    # We need the residual stream output of each layer
    seq_len = min(128, calib_tokens.shape[0])
    input_ids = calib_tokens[:seq_len].unsqueeze(0).to(device)

    print(f"  [GRAIL] Collecting activations from {seq_len} tokens...")

    # Run both models, collecting hidden states after each block
    with torch.inference_mode():
        # Original model
        x_orig = model_orig.embed(input_ids)
        orig_hiddens = [x_orig.clone()]
        for block in model_orig.blocks:
            x_orig, _ = block(x_orig)
            orig_hiddens.append(x_orig.clone())

        # Transformed model
        x_trans = model_transformed.embed(input_ids)
        trans_hiddens = [x_trans.clone()]
        for block in model_transformed.blocks:
            x_trans, _ = block(x_trans)
            trans_hiddens.append(x_trans.clone())

    # Compute reconstruction maps and absorb
    print(f"  [GRAIL] Computing reconstruction maps for {n_layers} layers...")
    total_healed = 0

    for i in range(n_layers):
        # Residual stream after layer i
        x_o = orig_hiddens[i + 1].view(-1, orig_hiddens[i + 1].shape[-1])
        x_t = trans_hiddens[i + 1].view(-1, trans_hiddens[i + 1].shape[-1])

        if x_o.shape != x_t.shape:
            continue

        # Compute reconstruction map
        R = key.compute_reconstruction_map(x_o, x_t)

        # Check if reconstruction is needed (R ≈ I means no healing needed)
        deviation = (R - torch.eye(R.shape[0], device=R.device)).abs().max().item()
        if deviation < 1e-4:
            continue  # Already lossless, skip

        # Absorb R into downstream weights
        state = key.absorb_into_downstream(state, R.cpu(), i)
        total_healed += 1

    # Final layer: fold R into head
    x_o_final = orig_hiddens[-1].view(-1, orig_hiddens[-1].shape[-1])
    x_t_final = trans_hiddens[-1].view(-1, trans_hiddens[-1].shape[-1])
    if x_o_final.shape == x_t_final.shape:
        R_final = key.compute_reconstruction_map(x_o_final, x_t_final)
        deviation = (R_final - torch.eye(R_final.shape[0], device=R_final.device)).abs().max().item()
        if deviation >= 1e-4:
            head_key = "head.weight"
            if head_key in state:
                w = state[head_key].float()
                state[head_key] = (w @ R_final.cpu()).to(state[head_key].dtype)
                total_healed += 1

    print(f"  [GRAIL] Healed {total_healed}/{n_layers} layers")
    return state


if __name__ == "__main__":
    key = GRAILKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test reconstruction map
    d = 64
    N = 256
    x_orig = torch.randn(N, d)
    # Create a transformed version (random rotation + noise)
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    x_trans = x_orig @ Q + 0.01 * torch.randn(N, d)

    R = key.compute_reconstruction_map(x_orig, x_trans)
    x_reconstructed = x_trans @ R

    error_before = (x_orig - x_trans).pow(2).mean().item()
    error_after = (x_orig - x_reconstructed).pow(2).mean().item()
    print(f"  Reconstruction error before: {error_before:.6f}")
    print(f"  Reconstruction error after:  {error_after:.6f}")
    assert error_after < error_before, "GRAIL should reduce error!"
    print("  Reconstruction verified ✓")
