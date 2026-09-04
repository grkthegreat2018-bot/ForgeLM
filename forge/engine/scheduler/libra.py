"""Libra: flexible request partitioning and scheduling for unbalanced workloads.

Based on "Libra: Flexible Request Partitioning and Scheduling for Serving
Unbalanced and Dynamic LLM Workloads" (USENIX NSDI 2026).

Problem: real-world variability in prompt/response lengths skews prefill
(decode) phases. Both colocated and disaggregated deployments fail to
simultaneously deliver low tail latency and high throughput.

Libra solution:
  1. Micro-request based flexible partitioning (FPS):
     - Split each request at ANY token boundary into cooperating segments
  2. Two-level scheduling:
     - Global scheduler: selects per-request split points
     - Local scheduler: forms SLO-aware batches on each GPU instance
  3. Chunked KV cache transfers: support cross-instance micro-request execution

Results: up to 1.91× goodput, 1.15-3.07× serving capacity, 74.2% performance
improvement under strict SLOs.

For our single-GPU setup:
  - Micro-request partitioning: split long requests into chunks
  - Local scheduler: form SLO-aware batches
  - Chunked KV transfers: not needed (single GPU) but useful for future multi-GPU
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MicroRequest:
    """A segment of a larger request."""
    parent_id: int
    segment_idx: int
    input_ids: torch.Tensor
    start_pos: int
    end_pos: int
    is_prefill: bool
    arrival_time: float = field(default_factory=time.time)
    slo_deadline: float = 0.0
    estimated_time: float = 0.0


class LibraScheduler:
    """Libra: flexible request partitioning and scheduling.

    Splits requests into micro-requests at optimal token boundaries,
    then schedules them with SLO awareness.
    """

    def __init__(self, max_batch_size: int = 8,
                 chunk_size: int = 512,
                 ttft_slo: float = 2.0,  # seconds
                 tpot_slo: float = 0.05,  # seconds per token
                 max_gpu_tokens: int = 32768):
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.max_gpu_tokens = max_gpu_tokens

        self._pending: deque[MicroRequest] = deque()
        self._active: list[MicroRequest] = []
        self._next_parent_id = 0
        self._gpu_tokens = 0

    def submit(self, input_ids: torch.Tensor,
               max_new_tokens: int = 256) -> int:
        """Submit a request, partitioned into micro-requests.

        Returns parent request ID.
        """
        parent_id = self._next_parent_id
        self._next_parent_id += 1

        T = input_ids.shape[1]
        deadline = time.time() + self.ttft_slo + max_new_tokens * self.tpot_slo

        # Partition prefill into chunks
        n_prefill_chunks = (T + self.chunk_size - 1) // self.chunk_size
        for i in range(n_prefill_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, T)
            chunk = input_ids[:, start:end]

            micro = MicroRequest(
                parent_id=parent_id,
                segment_idx=i,
                input_ids=chunk,
                start_pos=start,
                end_pos=end,
                is_prefill=True,
                slo_deadline=deadline,
                estimated_time=(end - start) * 0.001,  # rough estimate
            )
            self._pending.append(micro)

        return parent_id

    def form_batch(self) -> list[MicroRequest]:
        """Form an SLO-aware batch of micro-requests.

        Prioritizes by urgency (closest deadline first).
        """
        # Free completed micro-requests
        self._active = [m for m in self._active if not m._done] if self._active else []
        # (In practice, micro-requests are removed when their segment completes)

        # Sort pending by urgency (deadline - estimated_time)
        pending_list = list(self._pending)
        pending_list.sort(key=lambda m: m.slo_deadline - m.estimated_time)

        while len(self._active) < self.max_batch_size and pending_list:
            micro = pending_list.pop(0)
            micro_tokens = micro.end_pos - micro.start_pos

            if self._gpu_tokens + micro_tokens > self.max_gpu_tokens:
                # Skip if over memory budget
                pending_list.append(micro)
                break

            self._active.append(micro)
            self._gpu_tokens += micro_tokens

        # Restore remaining to pending
        self._pending = deque(pending_list)

        return self._active

    def complete_micro(self, micro: MicroRequest):
        """Mark a micro-request as complete."""
        if micro in self._active:
            self._active.remove(micro)
            self._gpu_tokens -= (micro.end_pos - micro.start_pos)

    def stats(self) -> dict:
        return {
            "pending": len(self._pending),
            "active": len(self._active),
            "gpu_tokens": self._gpu_tokens,
            "max_gpu_tokens": self.max_gpu_tokens,
            "chunk_size": self.chunk_size,
        }
