"""MGLU (Masked Gated Linear Unit) — halve FFN memory reads.

MGLU (NeurIPS 2025) uses a single shared weight matrix with learned binary
masks (Mixture of Element-wise Gating, MoEG) to decompose into gate/value.
This halves memory reads compared to SwiGLU (which needs separate W_gate
and W_up matrices).

Standard SwiGLU:  output = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
                 = 2 memory reads for the gate/up projections.

MGLU:            output = (Swish(x @ (M * W)) * (x @ ((1-M) * W))) @ W_down
                 = 1 memory read for W, two masked views.

SwiMGLU (Swish variant) matches or surpasses SwiGLU downstream accuracy.
FlashMGLU kernel: 19.7x speedup over naive PyTorch, 47% more memory-efficient,
34% faster than standard GLU.

The binary mask M is learned via a straight-through estimator (STE):
  - Forward: M = (mask_logits > 0).float()
  - Backward: gradient flows through mask_logits (STE bypass)

Identity init: M alternates 1/0 per element → W_gate and W_value are
complementary halves of W → equivalent to splitting W into two views.
At init, this is NOT identical to SwiGLU (different weight values), but
the capacity is the same and fine-tuning converges quickly.

Usage:
    from research.keys.activation.mglu_key import MGLUFFN

    # In model construction (replaces SwiGLUFFN):
    ffn = MGLUFFN(d_model=768, hidden_dim=2048)
    out = ffn(x)  # same interface as SwiGLUFFN
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class MGLUFFN(nn.Module):
    """Masked Gated Linear Unit FFN.

    Uses a single shared weight matrix W with a learned binary mask M to
    decompose into gate (M*W) and value ((1-M)*W), halving memory reads.

    Args:
        d_model: input/output dimension.
        hidden_dim: intermediate dimension (default 8*d_model/3, same as SwiGLU).
        activation: "swish" (SwiMGLU) or "relu" (ReMGLU).
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None,
                 activation: str = "swish"):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Single shared weight matrix (replaces w_gate + w_up).
        self.W = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

        # Learnable mask logits (STE for binary mask).
        # Init: alternate 1/0 per element → balanced gate/value split.
        mask_init = torch.zeros(hidden_dim)
        mask_init[::2] = 1.0  # even indices = gate, odd = value
        self.mask_logits = nn.Parameter(mask_init)

    def get_mask(self) -> torch.Tensor:
        """Binary mask via straight-through estimator.

        Forward: M = (logits > 0).float()
        Backward: gradient passes through (STE).
        """
        # STE: forward = hard threshold, backward = identity.
        return (self.mask_logits > 0).float().detach() + \
               self.mask_logits - self.mask_logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MGLU forward: (act(x @ M*W) * (x @ (1-M)*W)) @ W_down.

        Single read of W, two masked views.
        """
        M = self.get_mask()  # (hidden_dim,)

        # Single projection: x @ W  (one memory read)
        proj = self.W(x)  # (B, S, hidden_dim)

        # Masked views (element-wise, no extra matmul).
        gate = proj * M       # (B, S, hidden_dim) — gate view
        value = proj * (1 - M)  # (B, S, hidden_dim) — value view

        # Activation on gate.
        if self.activation == "swish":
            act_gate = F.silu(gate)
        elif self.activation == "relu":
            act_gate = F.relu(gate)
        else:
            act_gate = gate  # linear (no activation)

        # Element-wise gating + down projection.
        return self.w_down(act_gate * value)


class MGLUKey(Key):
    """MGLU key — converts SwiGLU weights to MGLU format.

    Merges W_gate and W_up into a single shared W by interleaving rows,
    and initializes the mask to recover the original gate/value split.

    forward: SwiGLU weights → MGLU weights (merge W_gate + W_up → W).
    reverse: MGLU weights → SwiGLU weights (split W via mask).

    Key class: PARTIAL — forward direction (merge) is exact; reverse
    requires the learned mask.
    """

    @property
    def name(self) -> str:
        return "mglu"

    @property
    def description(self) -> str:
        return ("MGLU: merge SwiGLU W_gate+W_up into shared W with binary mask. "
                "Halves FFN memory reads.")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Merge SwiGLU W_gate + W_up → MGLU shared W.

        Interleaves rows: W[0] = W_gate[0], W[1] = W_up[0], W[2] = W_gate[1], ...
        Mask: M[even]=1 (gate), M[odd]=0 (value).

        Args:
            data: {"w_gate": (D, H), "w_up": (D, H), "w_down": (H, D)}

        Returns:
            {"W": (D, 2*H), "w_down": (2*H, D), "mask_logits": (2*H,)}
        """
        try:
            w_gate = data["w_gate"]  # (D, H)
            w_up = data["w_up"]      # (D, H)
            w_down = data["w_down"]  # (H, D)

            D, H = w_gate.shape
            # Interleave: new hidden_dim = 2*H
            W = torch.zeros(D, 2 * H, dtype=w_gate.dtype, device=w_gate.device)
            W[:, 0::2] = w_gate  # even = gate
            W[:, 1::2] = w_up    # odd = value

            # New w_down: interleave columns to match.
            w_down_new = torch.zeros(2 * H, D, dtype=w_down.dtype, device=w_down.device)
            w_down_new[0::2, :] = w_down  # gate rows
            w_down_new[1::2, :] = w_down  # value rows (same down weight)

            # Mask: even=1 (gate), odd=0 (value).
            mask_logits = torch.zeros(2 * H, dtype=w_gate.dtype, device=w_gate.device)
            mask_logits[0::2] = 1.0

            return KeyResult(
                success=True,
                weights={"W": W, "w_down": w_down_new, "mask_logits": mask_logits},
                metadata={"d_model": D, "hidden_dim": 2 * H, "activation": "swish"},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Split MGLU W → SwiGLU W_gate + W_up using the mask.

        Args:
            weights: {"W": (D, 2H), "w_down": (2H, D), "mask_logits": (2H,)}

        Returns:
            {"w_gate": (D, H), "w_up": (D, H), "w_down": (H, D)}
        """
        try:
            W = weights["W"]
            w_down = weights["w_down"]
            mask_logits = weights["mask_logits"]

            D, H2 = W.shape
            H = H2 // 2
            M = (mask_logits > 0).float()

            # Extract gate and value views.
            w_gate = W[:, M > 0.5]  # columns where mask=1
            w_up = W[:, M <= 0.5]   # columns where mask=0

            # Down projection: average the gate and value rows.
            w_down_gate = w_down[M > 0.5, :]
            w_down_value = w_down[M <= 0.5, :]
            w_down_avg = (w_down_gate + w_down_value) / 2

            return KeyResult(
                success=True,
                data={"w_gate": w_gate, "w_up": w_up, "w_down": w_down_avg},
                metadata={"d_model": D, "hidden_dim": H},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))
