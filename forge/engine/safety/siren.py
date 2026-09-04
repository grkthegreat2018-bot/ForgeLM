"""SIREN — Streaming Safety Guardrails via Internal Representations.

Implements SIREN (ACL 2026): real-time harmful-content detection during
generation using the model's own internal hidden states, rather than a
separate guard model. Lightweight linear probes (one per harm category)
classify per-layer hidden states into a risk score in [0, 1]; if any
category exceeds ``threshold`` the stream is interrupted.

Key insight (SIREN, ACL 2026): harmful content is linearly separable in
the residual stream of instruction-tuned LMs. A single linear probe per
category trained on a small labeled set achieves guard-model-level
detection at ~250x fewer parameters (a probe is ``hidden_dim`` floats;
a guard model is hundreds of millions). On ForgeAI's 12GB RTX 5070 this
matters: a guard model would consume 0.5-1 GB of VRAM we don't have,
while 4 probes over ``hidden_dim=8192`` cost ~4 x 8192 x 2 bytes (bf16)
= ~65 KB — negligible, resident alongside the model.

Streaming detection follows StreamGuard (arXiv 2604.03962): evaluate
probes on the hidden state of *each* generated token as it is produced,
so a harmful trajectory is caught mid-generation rather than after the
full response is emitted. This bounds the amount of harmful text that
reaches the user to at most one token past the triggering state.

VRAM budget (RTX 5070, 12 GB):
  - Probes: 4 categories x hidden_dim floats. For hidden_dim=8192 in
    fp32 that is 4 x 8192 x 4 = ~128 KB; in bf16 ~64 KB. Negligible
    relative to the model (1.2B params bf16 ~= 2.4 GB) and KV cache.
  - Recorded hidden states: only the current token's states are held
    (one tensor per layer for one step), then discarded on reset().
    Peak transient: n_layers x hidden_dim x 2 bytes, e.g. 24 x 8192 x
    2 = ~384 KB. Well within budget.
  - All probe math runs in pure torch on the model's device (CUDA when
    available, CPU fallback otherwise). No extra kernels required.

Usage:
    from forge.engine.safety.siren import SIRENGuard, SIRENStreamWrapper

    guard = SIRENGuard(n_layers=24, hidden_dim=8192)
    guard.set_probes(trained_probe_weights)  # dict[category -> tensor]

    # Wrap a streaming generator from ForgeEngine.generate_stream:
    stream = SIRENStreamWrapper(engine.generate_stream(prompt, ...), guard)
    # The engine must call guard.record_layer() during its forward pass
    # (via a hook) so probes see the hidden states. See SIRENStreamWrapper
    # docstring for the integration contract.
    for chunk in stream:
        print(chunk, end="")

If no probes are set, ``evaluate()`` returns all zeros (permissive mode)
so SIREN is a safe no-op until probes are trained/loaded.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import torch

__all__ = ["SIRENGuard", "SIRENStreamWrapper"]

_DEFAULT_CATEGORIES = ("harmful", "deception", "pii", "violence")


class SIRENGuard:
    """SIREN safety guard: linear probes over internal hidden states.

    One probe per harm category. Each probe is a single linear layer with
    weight shape ``(n_layers, hidden_dim)``: for a recorded token, the
    per-layer hidden state is dotted with the probe's per-layer weight and
    the scores are mean-pooled across layers, then squashed through a
    sigmoid to give a category risk score in [0, 1].

    Probes are ~hidden_dim parameters each — 250x fewer than a guard
    model (SIREN, ACL 2026). On ForgeAI (RTX 5070, 12 GB) the full probe
    set is <128 KB and lives on the model's device with negligible VRAM
    impact (see module docstring for the budget breakdown).

    Permissive mode: with no probes set, ``evaluate()`` returns 0.0 for
    every category, so the guard never blocks. This lets SIREN ship as a
    no-op until trained probes are loaded via ``set_probes()``.
    """

    threshold: float = 0.7

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        categories: Optional[list[str]] = None,
    ) -> None:
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.categories: list[str] = (
            list(categories) if categories is not None else list(_DEFAULT_CATEGORIES)
        )
        # category -> probe weight tensor of shape (n_layers, hidden_dim).
        # Stored on the model device when set. None entries = no probe.
        self._probes: dict[str, torch.Tensor] = {}
        # Recorded hidden states for the current token: list indexed by
        # layer_idx. Only the latest token's states are kept; reset()
        # clears them between tokens.
        self._recorded: list[Optional[torch.Tensor]] = [None] * n_layers
        # Device is inferred from the first probe set / first recorded
        # state; defaults to CPU for pure-torch fallback.
        self._device: torch.device = torch.device("cpu")

    # ── Probe management ────────────────────────────────────────────────

    def set_probes(self, probe_weights: dict[str, torch.Tensor]) -> None:
        """Install trained linear probes.

        Args:
            probe_weights: mapping category -> weight tensor of shape
                ``(n_layers, hidden_dim)``. Each probe is a single linear
                layer: ``score = sigmoid(mean_l(hidden_l @ probe_l))``.
                Categories not present in the dict are left without a
                probe (permissive for that category).
        """
        self._probes = {}
        for cat, w in probe_weights.items():
            if w.shape != (self.n_layers, self.hidden_dim):
                raise ValueError(
                    f"probe for '{cat}' has shape {tuple(w.shape)}, "
                    f"expected ({self.n_layers}, {self.hidden_dim})"
                )
            self._probes[cat] = w.to(self._device)
        # Ensure all declared categories have an entry (None = permissive).
        for cat in self.categories:
            self._probes.setdefault(cat, None)  # type: ignore[arg-type]

    # ── Hidden-state recording (called during forward pass) ────────────

    def record_layer(self, layer_idx: int, hidden_state: torch.Tensor) -> None:
        """Record a layer's hidden state for the current token.

        Called from a forward hook on each transformer layer during
        generation. Stores the last-position (generated-token) hidden
        state. The caller is responsible for selecting the generated
        token's position (typically ``hidden_state[:, -1, :]`` for a
        batch of 1); this method accepts a 1D or 2D tensor and reduces
        to a single ``(hidden_dim,)`` vector.

        Args:
            layer_idx: index of the layer (0-based).
            hidden_state: hidden state tensor for the current token.
                Shapes accepted: (hidden_dim,) or (batch, hidden_dim)
                or (batch, seq, hidden_dim) — the last position is taken.
        """
        if not (0 <= layer_idx < self.n_layers):
            return
        t = hidden_state
        if t.dim() == 3:
            t = t[:, -1, :]
        if t.dim() == 2:
            t = t[-1]
        t = t.detach()
        self._device = t.device
        self._recorded[layer_idx] = t

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, float]:
        """Run each category probe over the recorded hidden states.

        Returns:
            dict mapping category -> risk score in [0, 1]. Categories
            without a probe (or with no probes set at all) return 0.0
            (permissive mode). Returns all zeros if no states recorded.
        """
        scores: dict[str, float] = {}
        have_states = any(s is not None for s in self._recorded)
        for cat in self.categories:
            probe = self._probes.get(cat)
            if probe is None or not have_states:
                scores[cat] = 0.0
                continue
            layer_scores: list[float] = []
            for li, state in enumerate(self._recorded):
                if state is None:
                    continue
                w = probe[li].to(state.device).to(state.dtype)
                # score = sigmoid(hidden @ probe_weight) per layer
                layer_scores.append(torch.sigmoid(state.dot(w)).item())
            scores[cat] = sum(layer_scores) / len(layer_scores) if layer_scores else 0.0
        return scores

    def check_token(
        self, token_str: str, hidden_states: dict[int, torch.Tensor]
    ) -> tuple[bool, Optional[str]]:
        """Convenience: record states, evaluate, decide safety.

        Args:
            token_str: the decoded token string (kept for API symmetry;
                not used in scoring — SIREN scores representations, not
                surface text).
            hidden_states: mapping layer_idx -> hidden state tensor for
                this token. Each value is passed to ``record_layer``.

        Returns:
            (is_safe, blocked_category). ``is_safe`` is False if any
            category risk exceeds ``threshold``; ``blocked_category`` is
            the name of the first such category, else None.
        """
        self.reset()
        for li, hs in hidden_states.items():
            self.record_layer(li, hs)
        scores = self.evaluate()
        for cat, score in scores.items():
            if score > self.threshold:
                return False, cat
        return True, None

    def reset(self) -> None:
        """Clear recorded hidden states between tokens."""
        self._recorded = [None] * self.n_layers


class SIRENStreamWrapper:
    """Wrap a ``generate_stream`` generator with SIREN safety checks.

    Iterates the underlying token generator and yields tokens as normal.
    If a SIREN guard block is triggered, iteration stops and a single
    ``"[SAFETY BLOCKED: <category>]"`` sentinel is yielded in place of
    the offending token (and all subsequent output).

    Integration contract: the wrapped generator (or the engine driving
    it) must call ``guard.record_layer(layer_idx, hidden_state)`` during
    its forward pass for each generated token, e.g. via a forward hook
    registered on the model's layers. This wrapper calls
    ``guard.evaluate()`` after each yielded token and checks scores
    against ``guard.threshold``. If the engine does not record states,
    ``evaluate()`` returns all zeros and no block ever fires (permissive
    mode) — the stream passes through unchanged.

    Streaming behavior (StreamGuard, arXiv 2604.03962): detection is
    per-token, so at most one token of harmful content can escape before
    the block surfaces. The blocked sentinel is emitted immediately and
    the generator is closed.

    Args:
        generator: an iterator of decoded token/chunk strings, as
            produced by ``ForgeEngine.generate_stream``.
        guard: a configured ``SIRENGuard`` with probes set (or none, for
            permissive passthrough).
    """

    def __init__(self, generator: Iterator[str], guard: SIRENGuard) -> None:
        self._gen = generator
        self._guard = guard
        self._blocked: bool = False

    def __iter__(self) -> Iterator[str]:
        for chunk in self._gen:
            scores = self._guard.evaluate()
            blocked_cat: Optional[str] = None
            for cat, score in scores.items():
                if score > self._guard.threshold:
                    blocked_cat = cat
                    break
            if blocked_cat is not None:
                self._blocked = True
                self._guard.reset()
                yield f"[SAFETY BLOCKED: {blocked_cat}]"
                try:
                    self._gen.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                return
            self._guard.reset()
            yield chunk
