"""ReplaySSM: input-caching for efficient SSM state reconstruction.

Based on "ReplaySSM: Accelerating SSM Inference via Input Replay" (Dao Lab, 2026).

Key insight
-----------
SSM (Mamba-1/2/3) state is *cheap to recompute* from a short window of
recent input tokens (one forward pass of the SSM scan), but *expensive to
checkpoint/restore* — the state is large and, for Mamba-3, complex-valued.

ReplaySSM therefore **caches recent SSM *inputs*** (the hidden states going
into the SSM block) instead of the states themselves.  When a state needs to
be restored (e.g. after speculative-decoding rejection), the cached inputs
are replayed through the SSM scan to reconstruct the state.

Benefits (from the paper):
  - 1.48x speedup for autoregressive generation
  - 1.87x speedup for speculative decoding (verify + rollback)
  - Works with any SSM (Mamba-1, Mamba-2, Mamba-3)

Implementation
--------------
``ReplaySSMCache`` maintains a **circular buffer (ring buffer)** of the last
``max_replay_tokens`` inputs **per layer**.  Each entry is the hidden state
that goes *into* the SSM block (before the input projection), shape
``(batch, d_model)`` for a single decode step.

The cache also tracks a monotonic **position** counter so that speculative
decoding can:
  1. ``checkpoint()`` — record the current position before drafting.
  2. ``rollback(position)`` — discard inputs after the checkpoint, then
     reconstruct the SSM state by replaying the remaining inputs.

Memory
------
For ``n_layers`` layers, ``max_replay_tokens=512``, ``d_model=2048``,
``batch=1``, fp16:
  ``n_layers * 512 * 2048 * 2 bytes``
  = ``n_layers * 2 MB``
  For 24 layers → 48 MB — far cheaper than storing full SSM states
  (which are ``n_layers * d_state * d_inner`` and complex-valued for Mamba-3).
"""
from __future__ import annotations

from typing import Any, Optional

import torch


class ReplaySSMCache:
    """Cache recent SSM *inputs* (not states) for efficient reconstruction.

    Uses a pre-allocated circular buffer per layer with a write index.

    Args:
        n_layers:          Number of SSM layers to cache inputs for.
        max_replay_tokens: Maximum number of input tokens to retain per
            layer (the ring-buffer capacity).  Older inputs are overwritten.
    """

    def __init__(self, n_layers: int, max_replay_tokens: int = 512):
        self.n_layers = n_layers
        self.max_replay_tokens = max_replay_tokens

        # Per-layer ring buffer storage (allocated lazily on first
        # ``cache_inputs`` call, since we need d_model / device / dtype).
        self._buffers: list[Optional[torch.Tensor]] = [None] * n_layers
        # Write index into each ring buffer.
        self._write_idx: list[int] = [0] * n_layers
        # Number of valid entries currently stored (capped at capacity).
        self._n_valid: list[int] = [0] * n_layers

        # Monotonic global position counter (total inputs cached across
        # all layers — they advance in lock-step during generation).
        self._position: int = 0

        # Saved checkpoint positions for speculative rollback.
        self._checkpoints: list[int] = []

    # ── Core API ────────────────────────────────────────────────────────

    def cache_inputs(self, layer_idx: int, inputs: torch.Tensor) -> None:
        """Store SSM inputs for a layer.

        Args:
            layer_idx: Index of the SSM layer.
            inputs:    Hidden states going *into* the SSM block.
                Shape ``(batch, T, d_model)`` or ``(batch, d_model)``.
                Multi-token inputs (T > 1) are split into individual
                time-steps so the ring buffer stores one entry per step.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})")

        # Normalise to (batch, T, d_model).
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # (batch, 1, d_model)
        bsz, T, d_model = inputs.shape

        buf = self._buffers[layer_idx]
        if buf is None:
            # Lazily allocate the ring buffer.
            buf = torch.zeros(
                bsz, self.max_replay_tokens, d_model,
                dtype=inputs.dtype, device=inputs.device,
            )
            self._buffers[layer_idx] = buf

        cap = self.max_replay_tokens
        w = self._write_idx[layer_idx]

        # Write each time-step into the ring buffer, wrapping as needed.
        for t in range(T):
            buf[:, w % cap, :] = inputs[:, t, :]
            w += 1

        self._write_idx[layer_idx] = w
        self._n_valid[layer_idx] = min(
            self._n_valid[layer_idx] + T, cap)

        # Advance the global position only on layer 0 (layers advance in
        # lock-step, so we use layer 0 as the canonical counter).  In
        # practice all layers receive the same number of inputs.
        if layer_idx == 0:
            self._position += T

    def reconstruct_state(
        self,
        layer_idx: int,
        ssm_module: Any,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct the SSM state by replaying cached inputs.

        Runs the cached inputs through the SSM scan to rebuild the state.
        This is a single forward pass of the SSM scan over the replay
        window — much cheaper than saving/restoring full states.

        Args:
            layer_idx:   Index of the SSM layer.
            ssm_module:  The SSM module (Mamba block or similar).  Must
                be callable: ``ssm_module(inputs, initial_state=...)``
                or ``ssm_module(inputs)`` returning ``(outputs, state)``
                or just ``outputs``.
            initial_state: Optional starting state.  If ``None``, the
                SSM module's default (typically zeros) is used.

        Returns:
            The reconstructed SSM state after processing all cached inputs.
        """
        inputs = self._get_ordered_inputs(layer_idx)
        if inputs is None:
            # Nothing cached — return initial state or zeros.
            if initial_state is not None:
                return initial_state
            raise RuntimeError(
                f"No cached inputs for layer {layer_idx}; cannot "
                "reconstruct state.")

        # inputs: (batch, n_valid, d_model)
        with torch.inference_mode():
            if initial_state is not None:
                out = ssm_module(inputs, initial_state=initial_state)
            else:
                out = ssm_module(inputs)

        # The SSM may return (outputs, state) or just outputs.
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            state = out[1]
        elif hasattr(ssm_module, "last_state"):
            state = ssm_module.last_state
        else:
            # Fallback: some SSM modules store state as an attribute.
            state = getattr(ssm_module, "_state", None)
            if state is None:
                raise RuntimeError(
                    "SSM module did not return a state and has no "
                    "`last_state` / `_state` attribute.")

        return state

    def checkpoint(self) -> int:
        """Return the current position for speculative rollback.

        Call this *before* drafting speculative tokens.  If the draft is
        rejected, call ``rollback(position)`` to discard the speculative
        inputs and restore the state to this point.

        Returns:
            The current global position (total inputs cached so far).
        """
        pos = self._position
        self._checkpoints.append(pos)
        return pos

    def rollback(self, position: int) -> None:
        """Discard inputs after the given checkpoint position.

        After rollback, ``reconstruct_state`` will replay only the inputs
        up to and including ``position``, effectively restoring the SSM
        state to what it was at the checkpoint.

        Args:
            position: A position previously returned by ``checkpoint()``.
        """
        if position > self._position:
            raise ValueError(
                f"Cannot rollback to future position {position} "
                f"(current: {self._position})")
        if position < 0:
            raise ValueError(f"position must be >= 0, got {position}")

        n_discard = self._position - position
        if n_discard == 0:
            return

        for layer_idx in range(self.n_layers):
            w = self._write_idx[layer_idx]
            nv = self._n_valid[layer_idx]
            # Move the write index back by n_discard (with wrap-around).
            self._write_idx[layer_idx] = (w - n_discard) % self.max_replay_tokens
            self._n_valid[layer_idx] = max(0, nv - n_discard)

        self._position = position

    def clear(self) -> None:
        """Reset the cache entirely (all layers, all positions)."""
        for layer_idx in range(self.n_layers):
            buf = self._buffers[layer_idx]
            if buf is not None:
                buf.zero_()
            self._write_idx[layer_idx] = 0
            self._n_valid[layer_idx] = 0
        self._position = 0
        self._checkpoints.clear()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_ordered_inputs(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return cached inputs in chronological order (oldest first).

        Always computes the oldest valid entry from ``(write_idx - n_valid)
        % capacity``, which is correct regardless of whether the buffer
        has wrapped or has been rolled back from a wrapped state.
        """
        buf = self._buffers[layer_idx]
        nv = self._n_valid[layer_idx]
        if buf is None or nv == 0:
            return None

        cap = self.max_replay_tokens
        w = self._write_idx[layer_idx]
        start = (w - nv) % cap  # index of the oldest valid entry

        if start + nv <= cap:
            # Valid range does not wrap around the buffer end.
            return buf[:, start:start + nv, :].clone()
        else:
            # Valid range wraps: [start, cap) ++ [0, nv - (cap - start))
            first_n = cap - start
            return torch.cat([
                buf[:, start:, :],          # old → buffer end
                buf[:, :nv - first_n, :],   # buffer start → new
            ], dim=1)

    @property
    def position(self) -> int:
        """Current global position (total inputs cached)."""
        return self._position

    def info(self) -> dict:
        """Return a summary dict (for logging / debugging)."""
        total_bytes = 0
        for buf in self._buffers:
            if buf is not None:
                total_bytes += buf.numel() * buf.element_size()
        return {
            "type": "replay_ssm",
            "n_layers": self.n_layers,
            "max_replay_tokens": self.max_replay_tokens,
            "position": self._position,
            "n_valid": list(self._n_valid),
            "bytes": total_bytes,
            "checkpoints": len(self._checkpoints),
        }
