"""SyncThink — Training-Free Reasoning Saturation Detection.

Reference: OpenReview Hc9jAnIB3f.

SyncThink monitors the reasoning-transition attention signal to detect when
a chain-of-thought has saturated and the model is ready to produce its final
answer. The core observation from the paper:

  During reasoning, each new token attends broadly across all prior reasoning
  tokens. When the model transitions to answer generation, answer tokens
  attend weakly to early reasoning and focus instead on boundary tokens
  (the end of the reasoning span, structural markers, etc.).

This attention shift is a reliable, training-free saturation signal. By
tracking the ratio of attention mass on the recent window (boundary region)
versus the early reasoning region, we can terminate generation as soon as
the ratio exceeds a threshold, yielding large token savings with no accuracy
loss. Reported result: 62% accuracy with 656 tokens vs 61.22% with 2141
tokens, and +8.1 on GPQA by preventing over-thinking.

VRAM budget: negligible. The monitor stores only a sliding history of
per-step attention-ratio floats (bounded by window_size), not full attention
matrices. All heavy tensors are reduced to scalars on-device before being
copied to host, so no attention tensor is retained.

This module is pure torch and runs on CPU; it operates on attention scores
already materialized by the engine, so no CUDA-specific code is required.
"""
from __future__ import annotations

from typing import Any

import torch


class SyncThinkMonitor:
    """Training-free reasoning saturation detector via attention-transition signal.

    Tracks how much each newly generated token attends to the recent boundary
    window versus the early reasoning region. When the recent/early attention
    ratio exceeds ``transition_threshold`` (and at least ``min_reasoning_tokens``
    have been generated), reasoning is considered saturated and ``should_terminate``
    returns True.

    Args:
        window_size: number of trailing positions defining the "recent" boundary
            window. Attention mass on these positions indicates the model is
            focusing on the end of the reasoning span (boundary tokens).
        transition_threshold: recent/early attention mass ratio above which
            reasoning is deemed saturated. Higher = more conservative (longer
            CoT), lower = more aggressive (earlier termination).
        min_reasoning_tokens: minimum tokens generated before termination is
            allowed. Prevents premature exit during the prompt/prefix phase
            where attention is naturally concentrated.

    Attributes:
        window_size: size of the recent boundary window.
        transition_threshold: saturation ratio threshold.
        min_reasoning_tokens: minimum tokens before termination.
    """

    def __init__(
        self,
        window_size: int = 32,
        transition_threshold: float = 0.15,
        min_reasoning_tokens: int = 128,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if transition_threshold <= 0.0:
            raise ValueError("transition_threshold must be > 0")
        if min_reasoning_tokens < 0:
            raise ValueError("min_reasoning_tokens must be >= 0")
        self.window_size = window_size
        self.transition_threshold = transition_threshold
        self.min_reasoning_tokens = min_reasoning_tokens
        self._tokens_generated: int = 0
        self._signal_history: list[float] = []
        self._last_signal: float = 0.0
        self._terminated: bool = False

    def update(
        self,
        token_id: int,
        attention_scores: torch.Tensor,
        position: int,
    ) -> None:
        """Record attention statistics for a newly generated token.

        Called after each token is produced. Reduces the per-head attention
        distribution to a scalar transition signal and appends it to the
        rolling history. No tensor is retained beyond this call.

        Args:
            token_id: id of the token just generated (unused for the signal
                itself, but accepted for logging/extension hooks).
            attention_scores: attention from the current token to all previous
                positions, shape ``(n_heads, seq_len)`` where ``seq_len`` is the
                number of prior positions (including the current one). Values
                are expected to be normalized attention weights summing to ~1
                per head. A 1-D tensor of shape ``(seq_len,)`` is also accepted
                and treated as a single head.
            position: 0-based absolute position of the current token in the
                full sequence. Used to gate on ``min_reasoning_tokens``.
        """
        self._tokens_generated += 1

        if attention_scores.dim() == 1:
            attn = attention_scores
        elif attention_scores.dim() == 2:
            attn = attention_scores.mean(dim=0)
        else:
            attn = attention_scores.view(attention_scores.shape[0], -1).mean(dim=0)

        attn = attn.to(dtype=torch.float32)
        seq_len = attn.shape[-1]
        if seq_len == 0:
            self._last_signal = 0.0
            self._signal_history.append(0.0)
            return

        recent_end = seq_len
        recent_start = max(0, seq_len - self.window_size)
        recent_mass = attn[recent_start:recent_end].sum().item()

        early_end = recent_start
        early_mass = attn[0:early_end].sum().item() if early_end > 0 else 0.0

        eps = 1e-8
        if early_mass < eps:
            signal = float("inf") if recent_mass > eps else 0.0
        else:
            signal = recent_mass / (early_mass + eps)

        self._last_signal = signal
        self._signal_history.append(signal)

    def compute_transition_signal(self) -> float:
        """Return the smoothed recent/early attention mass ratio.

        Uses the mean of the last ``window_size`` recorded signals to dampen
        per-step noise. A high value means recent tokens dominate attention
        over early reasoning — the reasoning-to-answer transition signature.

        Returns:
            Smoothed transition signal (float). ``0.0`` if no updates yet.
        """
        if not self._signal_history:
            return 0.0
        window = self._signal_history[-self.window_size:]
        finite = [s for s in window if s != float("inf")]
        if not finite:
            return float("inf")
        return sum(finite) / len(finite)

    def should_terminate(self) -> bool:
        """Decide whether reasoning has saturated and generation should stop.

        Returns True only when both conditions hold:
          1. At least ``min_reasoning_tokens`` have been generated.
          2. The smoothed transition signal exceeds ``transition_threshold``.

        Once True is returned, the monitor latches into the terminated state
        until ``reset`` is called, so repeated calls are idempotent.

        Returns:
            True if reasoning should terminate now.
        """
        if self._terminated:
            return True
        if self._tokens_generated < self.min_reasoning_tokens:
            return False
        signal = self.compute_transition_signal()
        if signal != float("inf") and signal > self.transition_threshold:
            self._terminated = True
            return True
        return False

    def reset(self) -> None:
        """Clear all internal state for a fresh generation pass."""
        self._tokens_generated = 0
        self._signal_history.clear()
        self._last_signal = 0.0
        self._terminated = False

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the monitor's current state.

        Returns:
            Dict with keys:
              - ``transition_signal``: latest smoothed signal value.
              - ``last_signal``: most recent raw per-step signal.
              - ``tokens_generated``: number of tokens observed via ``update``.
              - ``terminated``: whether ``should_terminate`` has latched True.
              - ``window_size``, ``transition_threshold``, ``min_reasoning_tokens``:
                the active configuration.
        """
        return {
            "transition_signal": self.compute_transition_signal(),
            "last_signal": self._last_signal,
            "tokens_generated": self._tokens_generated,
            "terminated": self._terminated,
            "window_size": self.window_size,
            "transition_threshold": self.transition_threshold,
            "min_reasoning_tokens": self.min_reasoning_tokens,
        }
