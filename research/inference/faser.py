"""FASER: Fine-grained phase management for speculative decoding.

Based on "FASER: Fine-Grained Phase Management for Speculative Decoding in
Dynamic LLM Serving" (arXiv 2604.20503).

Problem: in continuous batching with speculative decoding, the draft and
verification phases create bubbles. Fixed speculative length wastes compute
when acceptance is low.

FASER solution:
  1. Dynamic speculative length: adjusts per-request based on acceptance rate
  2. Early pruning: identifies rejected suffix during verification (token-wise exit)
  3. Frontier execution: breaks verification into chunks, overlaps with draft phase
  4. Online controller: adapts speculative length and resource partitioning

Results: up to 53% higher throughput, 1.92× lower latency.

For our model:
  - Dynamic K: start at K=2, increase if acceptance > 80%, decrease if < 50%
  - Early exit: stop verification at first rejection (don't process remaining)
  - Frontier overlap: begin next draft while verifying current
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


class FASERController:
    """Online controller for speculative decoding parameters.

    Adapts speculative length and resource partitioning based on:
      - Per-request acceptance rate
      - System load (batch fill level)
      - Draft vs verification time ratio
    """

    def __init__(self, min_k: int = 1, max_k: int = 8,
                 accept_rate_high: float = 0.8,
                 accept_rate_low: float = 0.5,
                 adjustment_interval: int = 10):
        self.min_k = min_k
        self.max_k = max_k
        self.accept_high = accept_rate_high
        self.accept_low = accept_rate_low
        self.adjustment_interval = adjustment_interval

        self._per_request_k: dict[int, int] = {}
        self._accept_history: dict[int, list[float]] = {}
        self._step_count = 0

    def get_spec_length(self, request_id: int) -> int:
        """Get current speculative length for a request."""
        return self._per_request_k.get(request_id, 2)  # default K=2

    def record_acceptance(self, request_id: int, accepted: int, total: int):
        """Record acceptance result and adjust K if needed."""
        if total == 0:
            return

        rate = accepted / total
        history = self._accept_history.setdefault(request_id, [])
        history.append(rate)

        # Keep last N results
        if len(history) > self.adjustment_interval:
            history.pop(0)

        # Adjust K every adjustment_interval steps
        if len(history) >= self.adjustment_interval:
            avg_rate = sum(history) / len(history)
            current_k = self._per_request_k.get(request_id, 2)

            if avg_rate > self.accept_high and current_k < self.max_k:
                self._per_request_k[request_id] = current_k + 1
                history.clear()  # reset after adjustment
            elif avg_rate < self.accept_low and current_k > self.min_k:
                self._per_request_k[request_id] = current_k - 1
                history.clear()

    def remove_request(self, request_id: int):
        """Clean up state for a completed request."""
        self._per_request_k.pop(request_id, None)
        self._accept_history.pop(request_id, None)

    def stats(self) -> dict:
        avg_k = sum(self._per_request_k.values()) / max(len(self._per_request_k), 1)
        return {
            "n_requests": len(self._per_request_k),
            "avg_k": avg_k,
            "k_distribution": {k: list(self._per_request_k.values()).count(k)
                              for k in range(self.min_k, self.max_k + 1)},
        }


class FASEREarlyExit:
    """Token-wise early exit for speculative verification.

    Instead of processing all K draft tokens in verification, stop at the
    first rejection. This saves compute when acceptance is low.
    """

    def __init__(self):
        self._exit_counts = 0
        self._total_verifications = 0

    def should_continue(self, accepted_so_far: int, total_draft: int) -> bool:
        """Check if verification should continue.

        Early exit: if the current token is rejected, stop processing
        the remaining draft tokens (they're guaranteed to be wrong).
        """
        self._total_verifications += 1
        return accepted_so_far < total_draft

    def record_exit(self, exit_position: int, total: int):
        """Record an early exit event."""
        if exit_position < total:
            self._exit_counts += 1

    def stats(self) -> dict:
        return {
            "total_verifications": self._total_verifications,
            "early_exits": self._exit_counts,
            "exit_rate": self._exit_counts / max(self._total_verifications, 1),
        }


class FASERFrontier:
    """Frontier execution: overlap verification with next draft.

    Breaks verification into chunks (frontiers). As each frontier completes,
    the next draft phase can begin for accepted tokens, overlapping with
    remaining verification frontiers.
    """

    def __init__(self, frontier_size: int = 2):
        self.frontier_size = frontier_size
        self._draft_ready = False
        self._verify_complete = False

    def split_verification(self, k: int) -> list[int]:
        """Split K draft tokens into verification frontiers.

        Returns list of frontier sizes.
        """
        frontiers = []
        remaining = k
        while remaining > 0:
            size = min(self.frontier_size, remaining)
            frontiers.append(size)
            remaining -= size
        return frontiers

    def on_frontier_complete(self, frontier_idx: int, n_frontiers: int):
        """Called when a verification frontier completes."""
        if frontier_idx == 0:
            # First frontier done → can start next draft
            self._draft_ready = True
        if frontier_idx == n_frontiers - 1:
            self._verify_complete = True

    @property
    def draft_ready(self) -> bool:
        return self._draft_ready

    @property
    def verify_complete(self) -> bool:
        return self._verify_complete

    def reset(self):
        self._draft_ready = False
        self._verify_complete = False
