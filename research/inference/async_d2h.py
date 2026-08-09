"""Async device-to-host transfer utilities for CPU spike elimination.

During autoregressive generation, calling .item() or .tolist() on GPU tensors
forces a host-device synchronization — the CPU blocks until the GPU finishes,
then copies data back. This creates massive CPU spikes (up to 400%) and leaves
the GPU idle.

The solution (from vLLM, PyTorch gpt-fast, and CUDA stream interleaving research):
1. Use pinned (page-locked) host memory for non-blocking D2H copies.
2. Issue the copy asynchronously, record a CUDA event.
3. Let the CPU do other work (e.g. prepare next input) while the copy runs.
4. Only synchronize when the value is actually needed — by then the GPU has
   finished the copy, so the sync is essentially free.

This module provides:
  - PinnedTokenBuffer: pre-allocated pinned memory for token IDs + log probs.
  - AsyncTokenReader: async D2H copy + event-based sync for decode loops.
"""
import torch
import torch.nn.functional as F


class PinnedTokenBuffer:
    """Pre-allocated pinned host memory for async D2H token transfers.

    Allocates once, reuses across all decode steps — zero per-step allocation.
    Pinned memory enables non_blocking=True copies via DMA (independent of CUDA cores).

    Args:
        batch_size: max batch size for generation.
        vocab_size: model vocab size (for log_softmax computation on GPU).
        device: GPU device where tensors live.
    """

    def __init__(self, batch_size: int, vocab_size: int, device: str = "cuda"):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.device = device
        # Pinned host buffers — page-locked for non-blocking DMA transfer.
        self.tokens_pinned = torch.zeros(batch_size, dtype=torch.long, pin_memory=True)
        self.logprobs_pinned = torch.zeros(batch_size, dtype=torch.float32, pin_memory=True)
        # CUDA event for async synchronization.
        self.event = torch.cuda.Event()


class AsyncTokenReader:
    """Async D2H reader for decode loop token + log-prob transfer.

    Usage pattern (replaces blocking .item()/.tolist() in decode loops):

        reader = AsyncTokenReader(B, vocab_size, device)
        # ... prefill ...
        reader.issue(next_tokens, next_logits)  # async copy, non-blocking

        while not done:
            # ... launch next forward pass (GPU starts working) ...
            logits, _, past = model(cur, past_key_values=past, ...)

            # Get PREVIOUS step's tokens (already on CPU, no sync needed!)
            prev_tokens = reader.get_tokens()  # instant — already transferred
            prev_lps = reader.get_logprobs()

            # Issue THIS step's tokens for next iteration
            next_tokens = sample(logits[:, -1])
            reader.issue(next_tokens, logits[:, -1])

        # Final sync for last step
        reader.sync()
        last_tokens = reader.get_tokens()

    This overlaps D2H transfer with GPU computation, eliminating CPU spikes.
    """

    def __init__(self, batch_size: int, vocab_size: int, device: str = "cuda"):
        self.buf = PinnedTokenBuffer(batch_size, vocab_size, device)
        self._has_pending = False
        self._tokens_cpu = None
        self._logprobs_cpu = None

    def issue(self, tokens_gpu: torch.Tensor, logits_gpu: torch.Tensor):
        """Issue async D2H copy of tokens + log-probs. Non-blocking.

        Computes log_softmax on GPU, then copies tokens and log-probs to pinned
        host memory asynchronously. The copy runs on the default stream and
        overlaps with subsequent GPU work.

        Args:
            tokens_gpu: (B,) sampled token IDs on GPU.
            logits_gpu: (B, vocab) or (vocab,) logits on GPU (before sampling).
        """
        B = tokens_gpu.shape[0]
        # Compute log-probs on GPU (stays on GPU, no sync).
        log_probs = F.log_softmax(logits_gpu.float(), dim=-1)
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        token_lps = log_probs.gather(1, tokens_gpu.unsqueeze(1)).squeeze(1)

        # Async copy to pinned memory (non-blocking — uses DMA engine).
        self.buf.tokens_pinned[:B].copy_(tokens_gpu, non_blocking=True)
        self.buf.logprobs_pinned[:B].copy_(token_lps, non_blocking=True)
        # Record event — we can check/poll this later.
        self.buf.event.record()
        self._has_pending = True

    def sync(self):
        """Block until the pending async copy completes.

        Only call this when you actually need the values. By the time you do,
        the GPU has usually finished the copy, so this is effectively free.
        """
        if self._has_pending:
            self.buf.event.synchronize()
            self._has_pending = False
            B = self.buf.batch_size
            self._tokens_cpu = self.buf.tokens_pinned[:B].tolist()
            self._logprobs_cpu = self.buf.logprobs_pinned[:B].tolist()

    def get_tokens(self) -> list[int]:
        """Get transferred token IDs. Calls sync() if pending."""
        if self._has_pending:
            self.sync()
        return self._tokens_cpu

    def get_logprobs(self) -> list[float]:
        """Get transferred log-probs. Calls sync() if pending."""
        if self._has_pending:
            self.sync()
        return self._logprobs_cpu

    @property
    def has_pending(self) -> bool:
        """Whether there's an async copy still in flight."""
        return self._has_pending


class StreamedGenerator:
    """Async D2H generator for single-sequence decode.

    Uses pinned memory + non-blocking D2H copies to reduce CPU spikes
    during autoregressive generation. For B=1, the forward pass is too
    fast (~23ms) to overlap with the copy, but pinned memory avoids
    pageable-memory copy overhead and the DMA engine handles the transfer
    independently of CUDA cores.

    For batch generation, use AsyncTokenReader directly — the batch
    amortizes the sync over B sequences, giving much better overlap.

    Usage:
        gen = StreamedGenerator(model, device="cuda")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        tokens, log_probs = gen.generate(ids, max_tokens=50, eos_id=tokenizer.eos_token_id)
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        # Pinned memory for non-blocking D2H.
        self.token_pinned = torch.zeros(1, dtype=torch.long, pin_memory=True)
        self.logprob_pinned = torch.zeros(1, dtype=torch.float32, pin_memory=True)
        # Pre-allocated single-token input buffer.
        self.cur_token = torch.zeros(1, 1, dtype=torch.long, device=device)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_tokens: int = 200,
                 eos_id: int | None = None,
                 preallocated_cache=None) -> tuple[list[int], list[float]]:
        """Generate tokens with async D2H (pinned memory, non-blocking copy).

        Args:
            input_ids: (1, T) prompt token IDs on GPU.
            max_tokens: max tokens to generate.
            eos_id: EOS token ID for early stopping (None = no early stop).
            preallocated_cache: optional PreAllocatedKVCache for O(1) KV append.

        Returns:
            (token_ids, log_probs) — lists of ints and floats.
        """
        model = self.model
        token_pinned = self.token_pinned
        logprob_pinned = self.logprob_pinned
        cur_token = self.cur_token

        gen_ids = []
        log_probs = []
        past_kvs = None

        # ── Prefill ──────────────────────────────────────────────
        if preallocated_cache is not None:
            logits, _ = model(input_ids, preallocated_cache=preallocated_cache)
        else:
            logits, _, past_kvs = model(input_ids, past_key_values=None, use_cache=True)
        next_logits = logits[0, -1]
        next_token = next_logits.argmax()

        # Issue async D2H for first token (non-blocking, pinned memory).
        lp = torch.log_softmax(next_logits.float(), dim=-1)[next_token]
        token_pinned.copy_(next_token, non_blocking=True)
        logprob_pinned.copy_(lp, non_blocking=True)
        torch.cuda.synchronize()

        token_id = token_pinned.item()
        lp_val = logprob_pinned.item()
        gen_ids.append(token_id)
        log_probs.append(lp_val)

        if eos_id is not None and token_id == eos_id:
            return gen_ids, log_probs

        # ── Decode loop ──────────────────────────────────────────
        for _step in range(1, max_tokens):
            cur_token[0, 0] = next_token

            # Launch forward.
            if preallocated_cache is not None:
                logits, _ = model(cur_token, preallocated_cache=preallocated_cache)
            else:
                logits, _, past_kvs = model(cur_token, past_key_values=past_kvs, use_cache=True)
            next_logits = logits[0, -1]
            next_token = next_logits.argmax()

            # Async D2H (non-blocking, pinned memory — DMA engine handles it).
            lp = torch.log_softmax(next_logits.float(), dim=-1)[next_token]
            token_pinned.copy_(next_token, non_blocking=True)
            logprob_pinned.copy_(lp, non_blocking=True)
            torch.cuda.synchronize()

            token_id = token_pinned.item()
            lp_val = logprob_pinned.item()
            gen_ids.append(token_id)
            log_probs.append(lp_val)

            if eos_id is not None and token_id == eos_id:
                break

        if preallocated_cache is None:
            del past_kvs

        return gen_ids, log_probs
