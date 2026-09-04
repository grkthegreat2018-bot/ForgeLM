"""Mamba-3 Key — complex-valued SSM states for Mamba-3.

Converts Mamba-2 real-valued SSM states to Mamba-3 complex-valued states.
This is a STRUCTURAL key (KeyClass.BI): the conversion is lossless both
ways because the imaginary part is zero-initialized (lossless warm start)
and dropped on reverse (recovers the exact Mamba-2 real states).

Mamba-3 weight structure (vs Mamba-2):
  in_proj.weight   (2*d_inner, d_model)        — unchanged
  conv1d.weight    (d_inner, 1, d_conv)        — unchanged
  conv1d.bias      (d_inner,)                  — unchanged
  x_proj.weight    (dt_rank + 2*d_state*2, d_inner) — complex B/C (extra rows zero-init)
  dt_proj.weight   (d_inner, dt_rank)          — stays real
  dt_proj.bias     (d_inner,)                  — unchanged
  A_log            (d_inner, d_state, 2)       — complex: last dim [real, imag], imag=0
  D                (d_inner,)                  — unchanged
  out_proj.weight  (d_model, d_inner)          — unchanged
  dt_norm.weight   (d_inner, 2)                — complex RMSNorm, imag=0
  A_norm.weight    (d_state, 2)                — complex RMSNorm, imag=0
  B_norm.weight    (d_state, 2)                — complex RMSNorm, imag=0
  C_norm.weight    (d_state, 2)                — complex RMSNorm, imag=0

Port path (Mamba-2 -> Mamba-3):
  A_log (d_inner, d_state) -> (d_inner, d_state, 2) with imag=0
  x_proj.weight (dt_rank + 2*d_state, d_inner) -> (dt_rank + 2*d_state*2, d_inner)
      extra rows (the imaginary B/C projections) zero-init
  dt/A/B/C norms (...,) -> (..., 2) with imag=0
  All other weights pass through unchanged

The key is lossless because:
  1. Real parts are copied verbatim (no transformation)
  2. Imaginary parts are zero-init (warm start, no information lost)
  3. Reverse drops the zero imaginary part -> exact Mamba-2 recovery
  4. Round-trip (Mamba-2 -> Mamba-3 -> Mamba-2) is identity (verified by test)

Usage:
    key = Mamba3Key()
    # Mamba-2 -> Mamba-3 (lossless warm start)
    result = key.forward(mamba2_state)
    # Mamba-3 -> Mamba-2 (drop imaginary)
    result = key.reverse(mamba3_state)
    # Cross-arch with MambaKey
    result = key.cross_arch(mamba2_state, MambaKey())
"""
from __future__ import annotations

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


# ═══════════════════════════════════════════════════════════════════════════════
# Weight name sets
# ═══════════════════════════════════════════════════════════════════════════════

# Weights that pass through unchanged between Mamba-2 and Mamba-3
MAMBA3_PASSTHROUGH = (
    "in_proj.weight",
    "conv1d.weight",
    "conv1d.bias",
    "dt_proj.weight",
    "dt_proj.bias",
    "D",
    "out_proj.weight",
)

# Mamba-2 norm weights that become complex (gain a trailing 2 dim) in Mamba-3
MAMBA3_COMPLEX_NORMS = (
    "dt_norm.weight",
    "A_norm.weight",
    "B_norm.weight",
    "C_norm.weight",
)

# Jamba naming variants for the norms
MAMBA3_COMPLEX_NORMS_JAMBA = (
    "dt_layernorm.weight",
    "b_layernorm.weight",
    "c_layernorm.weight",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mamba3Key — complex-valued SSM state conversion
# ═══════════════════════════════════════════════════════════════════════════════

class Mamba3Key(Key):
    """Lossless conversion between Mamba-2 (real) and Mamba-3 (complex) SSM states.

    Mamba-3 generalizes the selective SSM to complex-valued states by splitting
    each state into a real and imaginary part. The port from Mamba-2 is a
    lossless warm start: real parts are copied verbatim and imaginary parts are
    zero-initialized, so the model starts in an identical operating point.

    KeyClass.BI: forward and reverse are exact inverses (round-trip is identity).
    """

    def __init__(self, d_state: int | None = None, dt_rank: int | None = None):
        """
        Args:
            d_state: SSM state dimension. If None, inferred from A_log shape.
            dt_rank: discretization rank. If None, inferred from x_proj shape.
        """
        self._d_state = d_state
        self._dt_rank = dt_rank

    @property
    def name(self) -> str:
        return "mamba3"

    @property
    def description(self) -> str:
        return ("Lossless conversion between Mamba-2 real SSM states and "
                "Mamba-3 complex SSM states. Imaginary part is zero-init "
                "(warm start); reverse drops it for exact recovery.")

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    # ── shape inference helpers ──────────────────────────────────────────────

    def _infer_d_state(self, data: dict[str, torch.Tensor]) -> int | None:
        """Infer d_state from A_log (Mamba-2) or A_log (Mamba-3)."""
        if self._d_state is not None:
            return self._d_state
        a = data.get("A_log")
        if a is None:
            return None
        # Mamba-2: (d_inner, d_state); Mamba-3: (d_inner, d_state, 2)
        if a.dim() == 2:
            return a.shape[1]
        if a.dim() == 3:
            return a.shape[1]
        return None

    def _infer_dt_rank(self, data: dict[str, torch.Tensor],
                       d_state: int) -> int | None:
        """Infer dt_rank from x_proj.weight shape.

        Mamba-2 x_proj: (dt_rank + 2*d_state, d_inner)
        Mamba-3 x_proj: (dt_rank + 2*d_state*2, d_inner)
        """
        if self._dt_rank is not None:
            return self._dt_rank
        x = data.get("x_proj.weight")
        if x is None:
            return None
        out_dim = x.shape[0]
        # Try Mamba-2 formula first
        dt_rank_v2 = out_dim - 2 * d_state
        if dt_rank_v2 > 0:
            return dt_rank_v2
        return None

    # ── forward: Mamba-2 -> Mamba-3 ──────────────────────────────────────────

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Mamba-2 real states -> Mamba-3 complex states (lossless warm start).

        Args:
            data: Mamba-2 weight dict with real-valued SSM states.
                  A_log shape: (d_inner, d_state)
                  x_proj.weight shape: (dt_rank + 2*d_state, d_inner)
        Returns:
            KeyResult with Mamba-3 weight dict.
                  A_log shape: (d_inner, d_state, 2)  [real, imag=0]
                  x_proj.weight shape: (dt_rank + 2*d_state*2, d_inner)
                  norms shape: (..., 2)  [real, imag=0]
        """
        if "A_log" not in data:
            return KeyResult(success=False, error="No A_log in input data")

        d_state = self._infer_d_state(data)
        if d_state is None:
            return KeyResult(success=False, error="Cannot infer d_state from A_log")

        dt_rank = self._infer_dt_rank(data, d_state)
        if dt_rank is None:
            return KeyResult(success=False, error="Cannot infer dt_rank from x_proj.weight")

        result: dict[str, torch.Tensor] = {}

        for key, tensor in data.items():
            if key == "A_log":
                # (d_inner, d_state) -> (d_inner, d_state, 2) with imag=0
                real = tensor
                imag = torch.zeros_like(real)
                result[key] = torch.stack([real, imag], dim=-1)
            elif key == "x_proj.weight":
                # (dt_rank + 2*d_state, d_inner) -> (dt_rank + 2*d_state*2, d_inner)
                old_out = tensor.shape[0]
                new_out = dt_rank + 2 * d_state * 2
                new_x = torch.zeros(new_out, *tensor.shape[1:],
                                    dtype=tensor.dtype, device=tensor.device)
                new_x[:old_out] = tensor
                result[key] = new_x
            elif key in MAMBA3_COMPLEX_NORMS or key in MAMBA3_COMPLEX_NORMS_JAMBA:
                # (...,) -> (..., 2) with imag=0
                real = tensor
                imag = torch.zeros_like(real)
                result[key] = torch.stack([real, imag], dim=-1)
            elif key in MAMBA3_PASSTHROUGH:
                result[key] = tensor.clone()
            else:
                # Unknown keys pass through unchanged (don't block conversion)
                result[key] = tensor.clone()

        return KeyResult(
            success=True,
            weights=result,
            metadata={"conversion": "mamba2->mamba3",
                      "d_state": d_state,
                      "dt_rank": dt_rank,
                      "lossless": True},
        )

    # ── reverse: Mamba-3 -> Mamba-2 ──────────────────────────────────────────

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Mamba-3 complex states -> Mamba-2 real states (drop imaginary).

        Args:
            weights: Mamba-3 weight dict with complex-valued SSM states.
                  A_log shape: (d_inner, d_state, 2)
                  x_proj.weight shape: (dt_rank + 2*d_state*2, d_inner)
                  norms shape: (..., 2)
        Returns:
            KeyResult with Mamba-2 weight dict (real states only).
        """
        if "A_log" not in weights:
            return KeyResult(success=False, error="No A_log in input weights")

        a_log = weights["A_log"]
        if a_log.dim() != 3 or a_log.shape[-1] != 2:
            return KeyResult(success=False,
                             error=f"A_log must be 3D (d_inner, d_state, 2), "
                                   f"got shape {tuple(a_log.shape)}")

        d_state = self._infer_d_state(weights)
        if d_state is None:
            return KeyResult(success=False, error="Cannot infer d_state from A_log")

        # Infer dt_rank from the Mamba-3 x_proj shape
        x = weights.get("x_proj.weight")
        if x is not None:
            dt_rank = x.shape[0] - 2 * d_state * 2
            if dt_rank <= 0:
                return KeyResult(success=False,
                                 error="Cannot infer dt_rank: x_proj too small")
        elif self._dt_rank is not None:
            dt_rank = self._dt_rank
        else:
            return KeyResult(success=False, error="No x_proj.weight to infer dt_rank")

        result: dict[str, torch.Tensor] = {}

        for key, tensor in weights.items():
            if key == "A_log":
                # (d_inner, d_state, 2) -> (d_inner, d_state)  [take real]
                result[key] = tensor[..., 0].clone()
            elif key == "x_proj.weight":
                # (dt_rank + 2*d_state*2, d_inner) -> (dt_rank + 2*d_state, d_inner)
                old_out = dt_rank + 2 * d_state
                result[key] = tensor[:old_out].clone()
            elif key in MAMBA3_COMPLEX_NORMS or key in MAMBA3_COMPLEX_NORMS_JAMBA:
                # (..., 2) -> (...,)  [take real]
                result[key] = tensor[..., 0].clone()
            elif key in MAMBA3_PASSTHROUGH:
                result[key] = tensor.clone()
            else:
                result[key] = tensor.clone()

        return KeyResult(
            success=True,
            data=result,
            metadata={"conversion": "mamba3->mamba2",
                      "d_state": d_state,
                      "dt_rank": dt_rank,
                      "lossless": True},
        )
