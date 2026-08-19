"""Hybrid chunked prefill: adaptive chunking only when decode is active.

Based on vLLM PR #26625 (Hybrid Chunked Prefill).

Problem: chunked prefill reduces inter-token latency (ITL) by splitting
long prompts into chunks, interleaving with decode. BUT chunking also
splits long prefill segments, increasing launch/coordination overhead
and HURTING throughput.

The current approach applies chunked prefill UNCONDITIONALLY, so we keep
paying the throughput tax even when no decode requests are running.

Hybrid Chunked Prefill fixes this by enabling chunking ONLY when decode
is active. When no decode requests are running, it falls back to
continuous (non-chunked) prefill, recovering baseline throughput.

Results:
  - +2-5% higher total token throughput across all concurrency levels
  - 10-20% lower TTFT, especially under low concurrency
  - Stable scaling up to concurrency=8

For our BatchQueue:
  - Current: chunked prefill is always on (if enabled)
  - Hybrid: chunk only when decode requests are in the batch
  - When only prefill requests: use continuous prefill (no chunking)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from research.inference.chunked_prefill import ChunkedPrefiller, should_chunk


class HybridChunkedPrefiller(ChunkedPrefiller):
    """Adaptive chunked prefill: chunk only when decode is active.

    Extends ChunkedPrefiller with awareness of whether decode requests
    are currently running. When decode is active, uses chunked prefill
    to avoid blocking decode. When no decode is running, uses continuous
    (non-chunked) prefill for maximum throughput.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512,
                 device: str = "cuda"):
        super().__init__(model, chunk_size=chunk_size, device=device)
        self._decode_active = False

    def set_decode_active(self, active: bool):
        """Inform the prefiller whether decode requests are running."""
        self._decode_active = active

    def prefill_hybrid(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
        on_chunk_done: callable = None,
    ) -> tuple[torch.Tensor, list]:
        """Prefill with adaptive chunking.

        If decode is active: use chunked prefill (interleave with decode).
        If decode is NOT active: use continuous prefill (single forward pass).
        """
        B, T = input_ids.shape

        # Decision: chunk only when decode is active AND prompt is long
        use_chunking = self._decode_active and should_chunk(T, self.chunk_size)

        if not use_chunking:
            # Continuous prefill: single forward pass (maximum throughput)
            with torch.inference_mode():
                if T > 0:
                    out = self.model(input_ids, use_cache=use_cache)
                    if isinstance(out, tuple):
                        logits = out[0]
                        past_kv = out[1] if len(out) > 1 else None
                        if len(out) > 2:
                            past_kv = out[2]
                    else:
                        logits = out
                        past_kv = None
                    return logits, past_kv
                return None, None

        # Chunked prefill: interleave with decode
        return self.prefill_chunked(
            input_ids, use_cache=use_cache, on_chunk_done=on_chunk_done)


class HybridPrefillScheduler:
    """Scheduler that coordinates hybrid chunked prefill with decode.

    Tracks whether decode requests are active and informs the prefiller.
    Also manages the interleaving of prefill chunks and decode steps.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512,
                 device: str = "cuda"):
        self.prefiller = HybridChunkedPrefiller(model, chunk_size, device)
        self._decode_requests: list = []
        self._prefill_queue: list = []

    def add_decode_request(self, request):
        """Add an ongoing decode request."""
        self._decode_requests.append(request)
        self.prefiller.set_decode_active(True)

    def remove_decode_request(self, request):
        """Remove a completed decode request."""
        if request in self._decode_requests:
            self._decode_requests.remove(request)
        if not self._decode_requests:
            self.prefiller.set_decode_active(False)

    def add_prefill_request(self, input_ids: torch.Tensor):
        """Queue a prefill request."""
        self._prefill_queue.append(input_ids)

    def run_step(self) -> list:
        """Run one scheduling step.

        If decode is active and prefill queue is non-empty:
          → run one prefill chunk + one decode step (interleaved)

        If only prefill is queued:
          → run continuous prefill (no chunking)

        If only decode is active:
          → run decode step

        Returns:
            list of completed request results
        """
        results = []

        if self._prefill_queue and self._decode_requests:
            # Hybrid mode: chunked prefill + decode
            input_ids = self._prefill_queue.pop(0)

            def on_chunk_done(chunk_idx, n_chunks):
                # Run one decode step between prefill chunks
                for req in self._decode_requests[:]:
                    result = req.step()
                    if req.done:
                        results.append(result)
                        self.remove_decode_request(req)

            logits, past_kv = self.prefiller.prefill_hybrid(
                input_ids, use_cache=True, on_chunk_done=on_chunk_done)
            results.append(("prefill", logits, past_kv))

        elif self._prefill_queue:
            # Prefill-only mode: continuous prefill (no chunking)
            input_ids = self._prefill_queue.pop(0)
            logits, past_kv = self.prefiller.prefill_hybrid(input_ids)
            results.append(("prefill", logits, past_kv))

        elif self._decode_requests:
            # Decode-only mode
            for req in self._decode_requests[:]:
                result = req.step()
                if req.done:
                    results.append(result)
                    self.remove_decode_request(req)

        return results

    @property
    def decode_active(self) -> bool:
        return len(self._decode_requests) > 0

    @property
    def prefill_pending(self) -> int:
        return len(self._prefill_queue)

    def stats(self) -> dict:
        return {
            "decode_requests": len(self._decode_requests),
            "prefill_pending": self.prefill_pending,
            "decode_active": self.decode_active,
            "mode": "hybrid" if (self.decode_active and self.prefill_pending) else
                    "prefill_only" if self.prefill_pending else
                    "decode_only" if self.decode_active else "idle",
        }
