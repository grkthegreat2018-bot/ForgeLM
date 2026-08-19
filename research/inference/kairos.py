"""Kairos: SLO-aware scheduling for disaggregated LLM inference.

Based on "Taming Request Imbalance: SLO-Aware Scheduling for Disaggregated
LLM Inference" (arXiv 2605.02329).

Problem: FCFS scheduling for prefill causes head-of-line blocking (long
requests block short ones). Continuous batching for decode causes straggler
underutilization.

Kairos solution:
  1. Prefill: urgency-based priority scheduling
     - Predict prefill completion times
     - Dynamically select requests to maximize TTFT SLO attainment
  2. Decode: slack-guided adaptive batching
     - Leverage gap between per-step decode time and TPOT SLO
     - Greedily pack short requests to maximize throughput

Results: +23.9% TTFT SLO, +27.1% TPOT SLO, +33.8% e2e SLO, +19.3% decode throughput.

For our serving:
  - Prefill urgency: prioritize requests closest to missing TTFT deadline
  - Decode slack: pack more requests when decode is fast (under SLO)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SLORequest:
    """A request with SLO deadlines."""
    id: int
    input_ids: torch.Tensor
    max_new_tokens: int
    arrival_time: float
    ttft_deadline: float = 0.0  # time-to-first-token SLO
    tpot_deadline: float = 0.0  # time-per-output-token SLO
    prefill_done: bool = False
    first_token_time: float = 0.0
    decoded_tokens: int = 0
    estimated_prefill_time: float = 0.0

    def __post_init__(self):
        if self.ttft_deadline == 0.0:
            self.ttft_deadline = self.arrival_time + 2.0  # 2s default
        if self.tpot_deadline == 0.0:
            self.tpot_deadline = 0.05  # 50ms per token
        # Estimate prefill time (rough: 1ms per 100 tokens)
        self.estimated_prefill_time = self.input_ids.shape[1] * 0.00001


class KairosScheduler:
    """Kairos: SLO-aware prefill + decode scheduling.

    Prefill side: urgency-based priority (closest to missing TTFT deadline first).
    Decode side: slack-guided adaptive batching (pack more when under SLO).
    """

    def __init__(self, max_batch_size: int = 8,
                 ttft_slo: float = 2.0,
                 tpot_slo: float = 0.05,
                 max_gpu_tokens: int = 32768):
        self.max_batch_size = max_batch_size
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.max_gpu_tokens = max_gpu_tokens

        self._prefill_queue: deque[SLORequest] = deque()
        self._decode_active: list[SLORequest] = []
        self._next_id = 0
        self._gpu_tokens = 0
        self._last_decode_time = 0.0

    def submit(self, input_ids: torch.Tensor,
               max_new_tokens: int = 256) -> SLORequest:
        """Submit a new request."""
        req = SLORequest(
            id=self._next_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            arrival_time=time.time(),
            ttft_deadline=time.time() + self.ttft_slo,
            tpot_deadline=self.tpot_slo,
        )
        self._next_id += 1
        self._prefill_queue.append(req)
        return req

    def get_prefill_batch(self) -> list[SLORequest]:
        """Get batch for prefill, prioritized by urgency.

        Urgency = how close the request is to missing its TTFT deadline.
        """
        if not self._prefill_queue:
            return []

        # Sort by urgency (closest deadline first)
        pending = list(self._prefill_queue)
        now = time.time()
        pending.sort(key=lambda r: r.ttft_deadline - now - r.estimated_prefill_time)

        # Select requests that fit in memory and batch
        batch = []
        tokens = 0
        for req in pending:
            req_tokens = req.input_ids.shape[1]
            if len(batch) >= self.max_batch_size:
                break
            if self._gpu_tokens + tokens + req_tokens > self.max_gpu_tokens:
                continue
            batch.append(req)
            tokens += req_tokens

        # Remove selected from queue
        for req in batch:
            self._prefill_queue.remove(req)

        return batch

    def complete_prefill(self, req: SLORequest):
        """Mark prefill as complete, move to decode."""
        req.prefill_done = True
        req.first_token_time = time.time()
        self._decode_active.append(req)
        self._gpu_tokens += req.input_ids.shape[1] + req.decoded_tokens

    def get_decode_batch(self) -> list[SLORequest]:
        """Get batch for decode, using slack-guided adaptive batching.

        If decode is fast (under TPOT SLO), pack more requests.
        If decode is slow (over SLO), reduce batch size.
        """
        # Calculate slack: how much time budget is left per token
        now = time.time()
        if self._last_decode_time > 0:
            last_step_time = now - self._last_decode_time
        else:
            last_step_time = self.tpot_slo

        # Adaptive batch size based on slack
        if last_step_time < self.tpot_slo * 0.5:
            # Fast → can pack more
            effective_batch = self.max_batch_size
        elif last_step_time > self.tpot_slo:
            # Slow → reduce batch
            effective_batch = max(1, self.max_batch_size // 2)
        else:
            effective_batch = self.max_batch_size

        # Filter active requests (not yet complete)
        self._decode_active = [r for r in self._decode_active
                               if r.decoded_tokens < r.max_new_tokens]

        return self._decode_active[:effective_batch]

    def mark_decoded(self, req: SLORequest, n_tokens: int = 1):
        """Mark tokens as decoded for a request."""
        req.decoded_tokens += n_tokens
        self._last_decode_time = time.time()

    def stats(self) -> dict:
        now = time.time()
        n_at_risk = sum(1 for r in self._prefill_queue
                        if r.ttft_deadline - now < r.estimated_prefill_time)
        return {
            "prefill_pending": len(self._prefill_queue),
            "decode_active": len(self._decode_active),
            "at_risk_ttft": n_at_risk,
            "gpu_tokens": self._gpu_tokens,
            "max_gpu_tokens": self.max_gpu_tokens,
        }
