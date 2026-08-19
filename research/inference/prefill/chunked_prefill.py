"""Chunked prefill: split long prompts into chunks for interleaved decode.

Problem: when a long prompt (e.g., 8K tokens) enters the batch queue, it
blocks the decode queue — all short decodes stall until the long prefill
completes. This is especially bad for interactive workloads where a user
sends a long document and expects fast first-token response.

Solution: chunked prefill. Split the prompt into chunks (e.g., 512 tokens),
process one chunk per step, and interleave with ongoing decode steps. This
keeps decode latency low while making progress on the prefill.

Based on vLLM's chunked prefill design (vLLM v0.2+) and mini-infer's
implementation which reports significant TTFT improvements.

Usage:
    chunker = ChunkedPrefiller(model, chunk_size=512)
    kv = chunker.prefill_chunked(input_ids, use_cache=True)
    # kv can now be used for decode

The chunker processes the prompt in chunks, accumulating KV cache across
chunks. Between chunks, it can yield to let decode steps run.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChunkedPrefiller:
    """Chunked prefill for long prompts.

    Splits a long prompt into chunks of `chunk_size` tokens and processes
    them sequentially, accumulating the KV cache. This prevents long prompts
    from blocking the decode queue.

    Args:
        model: the LLM model
        chunk_size: tokens per prefill chunk (default 512)
        device: cuda or cpu
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512,
                 device: str = "cuda"):
        self.model = model
        self.chunk_size = chunk_size
        self.device = torch.device(device)

    def prefill_chunked(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
        on_chunk_done: callable = None,
    ) -> tuple[torch.Tensor, list]:
        """Process a long prompt in chunks, accumulating KV cache.

        Args:
            input_ids: (1, prompt_len) token IDs
            use_cache: whether to return KV cache
            on_chunk_done: optional callback(chunk_idx, n_chunks) for
                           yielding between chunks (e.g., to run decode steps)

        Returns:
            (logits, past_kv) — logits for the last token, past_kv for decode
        """
        B, T = input_ids.shape
        assert B == 1, "ChunkedPrefiller only supports batch=1"

        chunks = []
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunks.append((start, end))

        n_chunks = len(chunks)
        past_kv = None
        logits = None

        with torch.inference_mode():
            for i, (start, end) in enumerate(chunks):
                chunk_ids = input_ids[:, start:end]

                if past_kv is not None:
                    out = self.model(
                        chunk_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                else:
                    out = self.model(
                        chunk_ids,
                        use_cache=True,
                    )

                # Unpack output
                if isinstance(out, tuple):
                    if len(out) >= 3:
                        logits = out[0]
                        past_kv = out[2] if len(out) > 2 else out[1]
                    else:
                        logits = out[0]
                        past_kv = out[1] if len(out) > 1 else None
                else:
                    logits = out

                # Callback between chunks (yield to decode queue)
                if on_chunk_done is not None and i < n_chunks - 1:
                    on_chunk_done(i + 1, n_chunks)

        return logits, past_kv

    def prefill_interleaved(
        self,
        input_ids: torch.Tensor,
        decode_fn: callable,
        decode_state: dict,
    ) -> tuple[torch.Tensor, list]:
        """Interleave chunked prefill with ongoing decode steps.

        Between each prefill chunk, calls decode_fn(decode_state) to run
        one decode step for an ongoing generation. This keeps decode latency
        low while making progress on the prefill.

        Args:
            input_ids: the long prompt to prefill
            decode_fn: function(state) -> state, runs one decode step
            decode_state: mutable state for the ongoing decode

        Returns:
            (logits, past_kv) for the prefilled prompt
        """
        def on_chunk_done(chunk_idx, n_chunks):
            # Run one decode step between prefill chunks
            decode_state.update(decode_fn(decode_state))

        return self.prefill_chunked(
            input_ids, use_cache=True, on_chunk_done=on_chunk_done)


def should_chunk(prompt_len: int, chunk_size: int = 512,
                 threshold: int = 1024) -> bool:
    """Decide whether to use chunked prefill for a prompt.

    Chunking adds overhead for short prompts (extra forward calls).
    Only chunk if the prompt is longer than the threshold.
    """
    return prompt_len > threshold
