"""RAIN — Rewindable Auto-regressive INference (Li et al.).

Evaluation-guided self-correction with zero training: the model generates,
scores its own output with an eval_fn, and rewinds the last k tokens to
re-generate when the score is poor. Pure forward passes — no gradients, no
optimizer, no parameter updates. Mimics RL self-improvement during inference.

The eval_fn is the pluggable "reward": pass a sandbox pass/fail checker or a
heuristic (mean logprob, length, keyword coverage).
"""
from __future__ import annotations

from typing import Callable

import torch

from research.training_free.decoder import generate_with_cache


def _default_eval(text: str, logprobs: list[float] | None) -> float:
    """Default self-evaluation: mean token log-prob (higher = more confident)."""
    if logprobs:
        return sum(logprobs) / len(logprobs)
    return 0.0


class RAINGenerator:
    """Rewindable autoregressive generator.

    Args:
        model: ConfigurableResearchLLM.
        tokenizer: HF-style tokenizer.
        device: "cuda" or "cpu".
        eval_fn: callable(generated_text, token_logprobs) -> float score.
            Higher = better. Defaults to mean token log-prob.
        threshold: rewind when score < threshold.
        rewind_tokens: how many tokens to truncate on each rewind.
        max_rewinds: maximum rewinds per request.
        max_tokens: per-attempt generation budget.
        temperature: sampling temperature (0 = greedy).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        eval_fn: Callable[[str, list[float] | None], float] | None = None,
        threshold: float = -1.0,
        rewind_tokens: int = 8,
        max_rewinds: int = 3,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eval_fn = eval_fn or _default_eval
        self.threshold = threshold
        self.rewind_tokens = max(1, rewind_tokens)
        self.max_rewinds = max(0, max_rewinds)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rewinds_used = 0

    def generate(self, prompt: str) -> tuple[str, float, list[float]]:
        """Generate with rewind-and-regenerate until the score clears the
        threshold or the rewind budget is exhausted.

        Returns:
            (final_text, final_score, per_attempt_scores).
        """
        self.rewinds_used = 0
        scores: list[float] = []

        text, logprobs = generate_with_cache(
            self.model, self.tokenizer, prompt,
            device=self.device, max_tokens=self.max_tokens,
            temperature=self.temperature, collect_logprobs=True,
        )
        score = self.eval_fn(text, logprobs)
        scores.append(score)

        while score < self.threshold and self.rewinds_used < self.max_rewinds:
            if len(text.split()) <= self.rewind_tokens:
                break  # nothing meaningful left to rewind
            self.rewinds_used += 1
            text, logprobs = generate_with_cache(
                self.model, self.tokenizer, prompt,
                device=self.device, max_tokens=self.max_tokens,
                temperature=self.temperature, collect_logprobs=True,
            )
            score = self.eval_fn(text, logprobs)
            scores.append(score)

        return text, score, scores
