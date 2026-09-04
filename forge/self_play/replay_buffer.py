"""FOREVER-style Replay Buffer — forgetting-curve-inspired replay scheduling.

Research basis:
  - FOREVER Replay (ACL 2026): forgetting-curve-inspired replay scheduling.
  - Model time = cumulative optimizer update magnitude (not wall-clock steps).
  - Replay intervals aligned with internal evolution of the model.
  - Prevents catastrophic forgetting in continual self-play.

Core idea:
  Each stored sample carries the optimizer magnitude at the time it was added
  ("added_at_magnitude").  Retrieval probability follows the Ebbinghaus
  forgetting curve  R(t) = exp(-t / tau)  where:
    t   = |current_magnitude - sample_magnitude|   (model time elapsed)
    tau = stability_constant                        (how slowly a memory decays)

  When the model has changed a lot (large optimizer updates since the sample
  was stored), the sample is "due" for replay and receives higher weight.

Buffer size is capped at 10K samples (FIFO eviction when full).

Usage:
    buf = ReplayBuffer(max_size=10000, stability_constant=1000.0)
    buf.add({"prompt": "...", "solution": "...", "quality": 0.8,
             "test_passed": True}, optimizer_magnitude=1.2)
    batch = buf.sample(n=32, current_magnitude=buf.cumulative_magnitude)
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Forgetting-curve replay buffer keyed on optimizer update magnitude."""

    def __init__(self, max_size: int = 10000, stability_constant: float = 1000.0):
        self.max_size = max_size
        self.tau = stability_constant
        self._buffer: list[dict] = []
        self.cumulative_magnitude: float = 0.0
        self._head: int = 0  # circular write pointer for FIFO eviction

    # ── public API ──────────────────────────────────────────────────────

    def add(self, sample: dict, optimizer_magnitude: float = 0.0) -> None:
        """Add a sample, stamping it with the current model time.

        ``optimizer_magnitude`` is the magnitude of the optimizer update that
        produced this sample (e.g. ``||grad||`` or ``||delta_theta||``).  It is
        accumulated internally so callers can pass per-step deltas.
        """
        self.update_magnitude(optimizer_magnitude)
        entry = dict(sample)  # shallow copy — don't mutate caller's dict
        entry["added_at_magnitude"] = self.cumulative_magnitude
        entry["retrieval_count"] = 0
        if len(self._buffer) < self.max_size:
            self._buffer.append(entry)
        else:
            # FIFO circular eviction
            self._buffer[self._head] = entry
            self._head = (self._head + 1) % self.max_size

    def update_magnitude(self, magnitude: float) -> None:
        """Accumulate optimizer update magnitude (model-time clock)."""
        self.cumulative_magnitude += float(magnitude)

    def sample(self, n: int, current_magnitude: float) -> list[dict]:
        """Sample ``n`` items using forgetting-curve weighting.

        Weight ∝ exp(-|current_magnitude - added_at_magnitude| / tau).
        Samples whose model-time distance is small (recently seen) are
        de-prioritised; samples that are "due" (large distance) are replayed.
        """
        if not self._buffer:
            return []
        n = min(n, len(self._buffer))

        mags = np.array([s["added_at_magnitude"] for s in self._buffer])
        # Forgetting-curve weights: Ebbinghaus retention R(t) = exp(-t/tau)
        # measures how much of a memory remains; "due-ness" for replay is the
        # complement — samples the model has drifted away from (large
        # model-time distance, low retention) get the highest weight.
        distances = np.abs(current_magnitude - mags)
        weights = (1.0 - np.exp(-distances / self.tau)) + 1e-3
        # Normalise to a probability distribution.
        probs = weights / weights.sum()

        # High-magnitude updates → more aggressive replay: when the model has
        # moved a lot, boost the effective temperature so stale memories are
        # revisited more eagerly.
        recent_movement = float(np.median(distances))
        if recent_movement > self.tau:
            probs = probs ** 0.5  # flatten → broader replay
            probs = probs / probs.sum()

        chosen_idx = np.random.choice(len(self._buffer), size=n, replace=False, p=probs)
        out: list[dict] = []
        for idx in chosen_idx:
            entry = self._buffer[int(idx)]
            entry["retrieval_count"] = entry.get("retrieval_count", 0) + 1
            out.append(dict(entry))  # return copies so callers can't mutate
        return out

    def __len__(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict:
        """Return buffer health metrics."""
        if not self._buffer:
            return {
                "size": 0,
                "mean_age": 0.0,
                "retrieval_count": 0,
                "cumulative_magnitude": self.cumulative_magnitude,
            }
        ages = [self.cumulative_magnitude - s["added_at_magnitude"] for s in self._buffer]
        total_retrievals = sum(s.get("retrieval_count", 0) for s in self._buffer)
        return {
            "size": len(self._buffer),
            "mean_age": float(np.mean(ages)),
            "retrieval_count": total_retrievals,
            "cumulative_magnitude": self.cumulative_magnitude,
        }
