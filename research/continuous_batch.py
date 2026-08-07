"""Continuous batching scheduler for inference serving.

Processes multiple concurrent requests in a single batch, with iteration-level
scheduling. New requests can join mid-generation, and finished requests leave
immediately (no waiting for the longest request in the batch).

This is the key technique behind vLLM's high throughput (2-4x over static
batching). For small models, the gains are even larger because the model
is memory-bound, not compute-bound.

Usage:
    from research.continuous_batch import ContinuousBatchScheduler

    scheduler = ContinuousBatchScheduler(model, tokenizer, max_batch_size=8)
    scheduler.submit("Hello", max_new_tokens=100)
    scheduler.submit("Write a poem", max_new_tokens=200)
    for output in scheduler.run():
        print(output)
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterator, Tuple
from queue import Queue
import time


@dataclass
class Request:
    """A single inference request."""
    id: int
    prompt_ids: torch.Tensor  # (1, T_prompt)
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    generated_ids: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""
    # KV cache slots for this request.
    kv_cache: Optional[Tuple] = None
    # Position offset (for RoPE).
    cur_pos: int = 0


class ContinuousBatchScheduler:
    """Continuous batching scheduler for inference.

    Args:
        model: the LLM (must support use_cache=True and past_key_values)
        tokenizer: the tokenizer
        max_batch_size: max concurrent requests
        max_total_tokens: max total tokens across all requests in batch
        eos_token_id: end-of-sequence token id
        device: cuda or cpu
    """

    def __init__(self, model, tokenizer, max_batch_size=8,
                 max_total_tokens=8192, eos_token_id=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id
        self.device = torch.device(device)
        self.queue: Queue = Queue()
        self.active: List[Request] = []
        self.next_id = 0
        self.model.eval()

    def submit(self, prompt: str, max_new_tokens: int = 100,
               temperature: float = 0.0, top_p: float = 1.0) -> int:
        """Submit a request. Returns request ID."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        req = Request(
            id=self.next_id,
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            cur_pos=ids.shape[1],
        )
        self.next_id += 1
        self.queue.put(req)
        return req.id

    def _admit_requests(self):
        """Admit queued requests into the active batch (up to max_batch_size)."""
        while (len(self.active) < self.max_batch_size and
               not self.queue.empty()):
            req = self.queue.get()
            self.active.append(req)

    def _prefill(self, req: Request):
        """Process the prompt (prefill phase) for a new request."""
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                out = self.model(req.prompt_ids, use_cache=True)
                logits = out[0] if isinstance(out, tuple) else out
                kv = out[2] if isinstance(out, tuple) and len(out) > 2 else None
                if kv is None and isinstance(out, tuple) and len(out) >= 2:
                    kv = out[1]

        req.kv_cache = kv
        # Generate first token.
        last_logits = logits[:, -1, :] / max(req.temperature, 1e-5)
        if req.top_p < 1.0:
            last_logits = self._top_p_filter(last_logits, req.top_p)
        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        req.generated_ids.append(token_id)
        req.cur_pos += 1

        if token_id == self.eos_token_id:
            req.finished = True
            req.finish_reason = "stop"

    def _decode_step(self):
        """Single decode step for all active requests (batched)."""
        if not self.active:
            return

        # Collect tokens to process (one per active request).
        # Each request has its own KV cache, so we process them individually
        # but in a tight loop (true batching would require paged KV cache).
        for req in self.active:
            if req.finished or len(req.generated_ids) >= req.max_new_tokens:
                if not req.finished:
                    req.finished = True
                    req.finish_reason = "length"
                continue

            # Single-token forward with KV cache.
            last_token = torch.tensor([[req.generated_ids[-1]]],
                                      device=self.device, dtype=torch.long)
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    out = self.model(last_token, past_key_values=req.kv_cache,
                                     use_cache=True)
                    logits = out[0] if isinstance(out, tuple) else out
                    kv = out[2] if isinstance(out, tuple) and len(out) > 2 else None
                    if kv is None and isinstance(out, tuple) and len(out) >= 2:
                        kv = out[1]

            req.kv_cache = kv
            last_logits = logits[:, -1, :] / max(req.temperature, 1e-5)
            if req.top_p < 1.0:
                last_logits = self._top_p_filter(last_logits, req.top_p)
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            req.generated_ids.append(token_id)
            req.cur_pos += 1

            if token_id == self.eos_token_id:
                req.finished = True
                req.finish_reason = "stop"

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus sampling filter."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    def _evict_finished(self) -> List[Request]:
        """Remove finished requests from active batch."""
        finished = [r for r in self.active if r.finished]
        self.active = [r for r in self.active if not r.finished]
        return finished

    def run(self) -> Iterator[Dict]:
        """Run the scheduler, yielding completed requests.

        Yields:
            dict with 'id', 'text', 'finish_reason'
        """
        while not self.queue.empty() or self.active:
            # Admit new requests.
            self._admit_requests()

            # Prefill new requests (those without KV cache).
            for req in self.active:
                if req.kv_cache is None and not req.finished:
                    self._prefill(req)
                    if req.finished:
                        yield {
                            "id": req.id,
                            "text": self.tokenizer.decode(req.generated_ids),
                            "finish_reason": req.finish_reason,
                        }

            # Evict finished.
            self._evict_finished()

            # Decode step for all active.
            self._decode_step()

            # Yield finished requests.
            for req in self._evict_finished():
                yield {
                    "id": req.id,
                    "text": self.tokenizer.decode(req.generated_ids),
                    "finish_reason": req.finish_reason,
                }

    def run_streaming(self) -> Iterator[Dict]:
        """Run the scheduler, yielding tokens as they're generated.

        Yields:
            dict with 'id', 'token', 'text', 'finished'
        """
        while not self.queue.empty() or self.active:
            self._admit_requests()

            for req in self.active:
                if req.kv_cache is None and not req.finished:
                    self._prefill(req)
                    if req.generated_ids:
                        yield {
                            "id": req.id,
                            "token": req.generated_ids[-1],
                            "text": self.tokenizer.decode([req.generated_ids[-1]]),
                            "finished": req.finished,
                        }

            self._evict_finished()
            self._decode_step()

            for req in self.active:
                if req.generated_ids:
                    yield {
                        "id": req.id,
                        "token": req.generated_ids[-1],
                        "text": self.tokenizer.decode([req.generated_ids[-1]]),
                        "finished": req.finished,
                    }

            self._evict_finished()
