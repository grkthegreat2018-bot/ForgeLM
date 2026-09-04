"""R37-4: OutRo Key — Sink-Enhanced Contextual Representations.

OutRo (Outgoing-Rotary / sink-enhanced attention) modifies the attention
pattern so that the sink token (typically position 0, which absorbs a large
fraction of attention mass) can attend beyond the causal constraint, and so
that non-sink token representations are aligned with the sink representation.

Motivation:
  Attention sinks (Xiao et al., 2024) show that the first token acts as an
  "absorber" of redundant attention mass. Standard causal masking forbids the
  sink token from looking at future tokens, even though the sink is a global
  context aggregator. OutRo relaxes this: for sink positions, the causal mask
  is dropped (non-causal attention), while non-sink positions keep the causal
  constraint. Non-sink token representations are aligned toward the sink
  representation via a learned (or fixed identity) alignment matrix so that
  the global context captured by the sink propagates to every token.

Mechanism:
  1. Sink detection: a token is a "sink" if its attention to position 0
     exceeds a threshold (default 0.5). This identifies positions whose
     representation is dominated by the sink.
  2. Non-causal mask: for sink query positions, allow attending to ALL key
     positions (drop the upper-triangular causal mask). Non-sink positions
     retain the standard causal mask.
  3. Alignment: non-sink representations are aligned with the sink
     representation via an alignment matrix A (d x d, init = identity so the
     key is lossless at start).

Key class: PARTIAL — the forward direction modifies the attention pattern
(mask generation + alignment), which is the primary use. The reverse
direction extracts the sink mask and alignment parameters for inspection.

Usage:
    from research.keys.attention.outro_key import OutRoKey

    key = OutRoKey(sink_threshold=0.5)
    # data -> weights: build the non-causal mask + alignment from attn weights
    res = key.forward({"attn_weights": attn_weights, "d_model": 64})
    mask = res.weights["attention_mask"]      # (..., seq, seq) bool
    sink_mask = res.weights["sink_mask"]      # (..., seq) bool
    align = res.weights["alignment"]          # (d, d)

    # weights -> data: extract sink mask + alignment params
    res = key.reverse(res.weights)
"""
from __future__ import annotations

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class OutRoKey(Key):
    """OutRo (Outgoing-Rotary) key — sink-enhanced contextual representations.

    Detects sink tokens (high attention to position 0), enables non-causal
    attention for sink query positions, and aligns non-sink representations
    with the sink representation.

    Args:
        sink_threshold: attention-to-position-0 threshold above which a token
            is classified as a sink (default 0.5).
        align_strength: mixing factor for the alignment of non-sink tokens
            toward the sink representation (default 0.0 = lossless at start;
            the alignment matrix is identity and contributes nothing).
    """

    def __init__(self, sink_threshold: float = 0.5,
                 align_strength: float = 0.0):
        self.sink_threshold = sink_threshold
        self.align_strength = align_strength

    @property
    def name(self) -> str:
        return "outro"

    @property
    def description(self) -> str:
        return (
            "OutRo (Outgoing-Rotary): sink-enhanced contextual representations. "
            "Detects sink tokens, enables non-causal attention for sink "
            "positions, and aligns non-sink representations with the sink."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    # ── helpers ────────────────────────────────────────────────────────────

    def _detect_sinks(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Detect sink tokens from attention weights.

        A token is a sink if its attention mass on position 0 exceeds the
        threshold. Works for arbitrary batch/head dimensions.

        Args:
            attn_weights: (..., seq_q, seq_k) attention probabilities.

        Returns:
            sink_mask: (..., seq_q) boolean tensor. True where the query
            position is a sink.
        """
        # Attention paid to position 0 (the canonical sink position).
        attn_to_zero = attn_weights[..., 0]  # (..., seq_q)
        return attn_to_zero > self.sink_threshold

    @staticmethod
    def _causal_mask(seq_q: int, seq_k: int,
                     device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
        """Standard lower-triangular causal mask (True = allowed)."""
        i = torch.arange(seq_q, device=device).unsqueeze(1)
        j = torch.arange(seq_k, device=device).unsqueeze(0)
        return i >= j  # (seq_q, seq_k)

    def _build_non_causal_for_sinks(self, attn_weights: torch.Tensor,
                                    sink_mask: torch.Tensor) -> torch.Tensor:
        """Build an attention mask that is causal everywhere except sink rows.

        For sink query positions the mask allows attending to ALL key
        positions (non-causal). For non-sink query positions the standard
        causal mask is retained.

        Args:
            attn_weights: (..., seq_q, seq_k) — used only for shape/device.
            sink_mask: (..., seq_q) boolean sink indicator.

        Returns:
            mask: (..., seq_q, seq_k) boolean. True = attend (allowed).
        """
        seq_q = attn_weights.shape[-2]
        seq_k = attn_weights.shape[-1]
        device = attn_weights.device
        dtype = attn_weights.dtype

        # Base causal mask: (seq_q, seq_k)
        causal = self._causal_mask(seq_q, seq_k, device, dtype)

        # Broadcast causal to the batch/head shape: (..., seq_q, seq_k)
        # attn_weights.ndim - 2 is the number of leading dims.
        leading = attn_weights.shape[:-2]
        causal_b = causal.expand(*leading, seq_q, seq_k)

        # Full (non-causal) mask: everything allowed.
        full = torch.ones_like(causal_b)

        # Where sink_mask is True (over the query axis), use full; else causal.
        sink_b = sink_mask.unsqueeze(-1).expand(*leading, seq_q, seq_k)
        mask = torch.where(sink_b, full, causal_b)
        return mask

    @staticmethod
    def _default_alignment(d_model: int,
                           device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
        """Identity alignment matrix (lossless at start)."""
        return torch.eye(d_model, device=device, dtype=dtype)

    # ── forward / reverse ──────────────────────────────────────────────────

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights.

        Given attention weights, identify sink tokens (high attention to
        position 0), build a non-causal attention mask for sink positions,
        and produce an alignment matrix for non-sink representations.

        Args:
            data: {
                "attn_weights": (..., seq_q, seq_k) attention probabilities,
                "d_model": int (optional, for alignment matrix size),
            }

        Returns:
            KeyResult with weights {
                "attention_mask": (..., seq_q, seq_k) bool — allowed positions,
                "sink_mask": (..., seq_q) bool — sink query positions,
                "alignment": (d, d) — alignment matrix (identity at start),
            }
        """
        try:
            attn_weights = data.get("attn_weights")
            if attn_weights is None:
                return KeyResult(
                    success=False,
                    error="Missing 'attn_weights' in data",
                )
            d_model = data.get("d_model", attn_weights.shape[-1])

            # 1. Sink detection.
            sink_mask = self._detect_sinks(attn_weights)  # (..., seq_q)

            # 2. Non-causal mask for sink positions.
            attention_mask = self._build_non_causal_for_sinks(
                attn_weights, sink_mask)  # (..., seq_q, seq_k)

            # 3. Alignment matrix (identity = lossless at start).
            alignment = self._default_alignment(
                d_model, attn_weights.device, attn_weights.dtype)

            return KeyResult(
                success=True,
                weights={
                    "attention_mask": attention_mask,
                    "sink_mask": sink_mask,
                    "alignment": alignment,
                },
                metadata={
                    "sink_threshold": self.sink_threshold,
                    "align_strength": self.align_strength,
                    "n_sinks": int(sink_mask.sum().item()),
                    "seq_q": attn_weights.shape[-2],
                    "seq_k": attn_weights.shape[-1],
                    "d_model": d_model,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data.

        Extract the sink mask and alignment parameters from a set of OutRo
        weights for inspection / checkpoint round-tripping.

        Args:
            weights: {
                "attention_mask": (..., seq_q, seq_k) bool,
                "sink_mask": (..., seq_q) bool,
                "alignment": (d, d),
            }

        Returns:
            KeyResult with data {
                "sink_mask": (..., seq_q) bool,
                "alignment": (d, d),
                "n_sinks": int,
            }
        """
        try:
            sink_mask = weights.get("sink_mask")
            alignment = weights.get("alignment")
            if sink_mask is None or alignment is None:
                return KeyResult(
                    success=False,
                    error="Missing 'sink_mask' or 'alignment' in weights",
                )
            return KeyResult(
                success=True,
                data={
                    "sink_mask": sink_mask,
                    "alignment": alignment,
                    "n_sinks": int(sink_mask.sum().item()),
                },
                metadata={
                    "sink_threshold": self.sink_threshold,
                    "align_strength": self.align_strength,
                    "alignment_is_identity": bool(
                        torch.allclose(
                            alignment,
                            torch.eye(alignment.shape[0],
                                      device=alignment.device,
                                      dtype=alignment.dtype),
                            atol=1e-6)),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))
