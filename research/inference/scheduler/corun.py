"""CoRun: deterministic LLM inference via padding + fixed-shape CUDA graphs.

Based on "CoRun: Padding is Simple and Efficient for Deterministic LLM
Inference" (arXiv 2608.14376).

Problem: LLM inference is nondeterministic even with fixed seeds because
batch-dependent GPU execution changes kernel tiling and FP reduction orders.
This breaks model evaluation and RL training reproducibility.

Existing solutions use batch-invariant kernels, but these:
  - Restrict optimized tiling → 2×+ latency increase
  - Reduce serving throughput by up to 74%

CoRun's insight: most kernels are POSITION-INVARIANT (not batch-invariant).
The output for position i doesn't depend on what other positions are in the
batch. So instead of making kernels batch-invariant, make the BATCH shape
invariant via padding.

CoRun:
  1. Isolate prefill (each request prefilled alone — natural shape)
  2. Fixed-shape batched decode (pad to max concurrency, one CUDA graph)
  3. Disable Split-KV for decode (deterministic)
  4. Bind RNG state to requests (not batch slots)

Results: determinism + 15-324% throughput improvement, -51.8% TTFT, -48.6% TPOT.

For our setup:
  - Determinism is valuable for self-play (reproducible reward signals)
  - Padding overhead is minimal (batch ≤ 8, pad to 8)
  - One CUDA graph for all decode steps (no re-capture)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class CoRunScheduler:
    """CoRun deterministic inference scheduler.

    Isolates prefill (one request at a time) and uses fixed-shape batched
    decode (pad to max concurrency) for deterministic outputs.
    """

    def __init__(self, model: nn.Module, max_concurrency: int = 8,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.max_concurrency = max_concurrency
        self.device = torch.device(device)
        self.dtype = dtype

        # Per-request RNG states (for deterministic sampling)
        self._rng_states: dict[int, torch.Generator] = {}

        # Fixed-shape decode buffers
        config = getattr(model, 'config', None)
        d_model = getattr(config, 'd_model', 2048) if config else 2048
        self._decode_input = torch.zeros(max_concurrency, 1, dtype=torch.long,
                                          device=self.device)
        self._decode_pos = torch.zeros(max_concurrency, 1, dtype=torch.long,
                                        device=self.device)

        # CUDA graph for fixed-shape decode
        self._decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._decode_output = None

    def capture_decode_graph(self):
        """Capture a single CUDA graph for fixed-shape batched decode."""
        if self.device.type != "cuda":
            return

        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                try:
                    self._decode_output = self.model(
                        self._decode_input, position_ids=self._decode_pos)
                except Exception:
                    return
            torch.cuda.synchronize()

        # Capture
        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self._decode_graph):
                try:
                    self._decode_output = self.model(
                        self._decode_input, position_ids=self._decode_pos)
                except Exception:
                    self._decode_graph = None
                    return

        print(f"  [CoRun] Captured fixed-shape decode graph "
              f"(concurrency={self.max_concurrency})")

    def prefill_isolated(self, input_ids: torch.Tensor,
                         request_id: int) -> torch.Tensor:
        """Isolated prefill: one request, natural shape (deterministic).

        Each request is prefilled alone, so kernel tiling is determined
        solely by the prompt length — no co-runner dependence.
        """
        with torch.inference_mode():
            pos = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
            output = self.model(input_ids, position_ids=pos)
            if isinstance(output, tuple):
                return output[0]
            return output

    def decode_batched(self, input_ids: torch.Tensor,
                       positions: torch.Tensor,
                       active_mask: torch.Tensor) -> torch.Tensor:
        """Fixed-shape batched decode (deterministic via padding).

        Pads to max_concurrency, replays fixed CUDA graph, trims output.
        All positions get the same kernel tiling regardless of batch fill.
        """
        B = input_ids.shape[0]

        # Pad to fixed shape
        self._decode_input.zero_()
        self._decode_pos.zero_()
        self._decode_input[:B] = input_ids
        self._decode_pos[:B] = positions

        if self._decode_graph is not None:
            # Replay fixed graph
            self._decode_graph.replay()
            output = self._decode_output
        else:
            # Fallback: run without graph
            with torch.inference_mode():
                output = self.model(self._decode_input, position_ids=self._decode_pos)

        # Trim to active requests
        if isinstance(output, tuple):
            logits = output[0][:B]
        else:
            logits = output[:B]

        return logits

    def sample_deterministic(self, logits: torch.Tensor,
                              request_id: int, temperature: float = 1.0,
                              top_k: int = 0) -> torch.Tensor:
        """Deterministic sampling with per-request RNG state.

        RNG state is bound to the request, not the batch slot, so the same
        request always produces the same output regardless of batch position.
        """
        if request_id not in self._rng_states:
            # Seed from request_id for reproducibility
            gen = torch.Generator(device=self.device)
            gen.manual_seed(request_id)
            self._rng_states[request_id] = gen

        gen = self._rng_states[request_id]

        if temperature > 0:
            logits = logits / temperature

        if top_k > 0:
            topk_logits, topk_indices = logits.topk(top_k, dim=-1)
            probs = torch.softmax(topk_logits, dim=-1)
            sampled = torch.multinomial(probs, 1, generator=gen)
            return topk_indices.gather(-1, sampled)
        else:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1, generator=gen)

    def clear_rng(self, request_id: int):
        """Clear RNG state for a completed request."""
        self._rng_states.pop(request_id, None)

    def stats(self) -> dict:
        return {
            "max_concurrency": self.max_concurrency,
            "graph_captured": self._decode_graph is not None,
            "active_rng_states": len(self._rng_states),
        }
