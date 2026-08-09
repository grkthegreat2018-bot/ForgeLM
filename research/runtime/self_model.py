"""Self-Modeling — model predicts its own errors and retries (D5).

The model's own confidence (entropy, logit gap) is a signal about its likely
errors. When confidence is low, the model is likely wrong and should retry.

This module provides:
  1. ConfidenceScorer: measures model confidence from logits
  2. RetryPolicy: decides whether to retry based on confidence + attempt count
  3. SelfModelLoop: wraps generation with automatic retry on low confidence

Research basis: SYSTEMS_IDEATION.md D5 — "Self-Modeling"
  - Model predicts its own errors via confidence signal
  - Retry before outputting (avoid errors instead of making them)
  - Test-time compute scaling, but SELF-AWARE
  - Composes with self_play, infinite_curriculum, IterativeRefinement

Usage:
    from research.runtime.self_model import ConfidenceScorer, RetryPolicy
    scorer = ConfidenceScorer()
    policy = RetryPolicy(min_confidence=0.6, max_retries=3)
    # After generating:
    confidence = scorer.score(logits)
    if policy.should_retry(confidence, attempt=0):
        # regenerate with higher temperature
        ...
"""
import math
from typing import Dict, Optional

import torch


class ConfidenceScorer:
    """Score model confidence from logits.

    Multiple signals:
    1. Logit gap: difference between top-1 and top-2 logits (higher = more confident)
    2. Entropy: Shannon entropy of the softmax distribution (lower = more confident)
    3. Top-1 probability: softmax probability of the top token (higher = more confident)

    The combined score is a weighted average of normalized signals.
    """

    def __init__(self, logit_gap_weight: float = 0.4,
                 entropy_weight: float = 0.3,
                 top1_weight: float = 0.3):
        self.w_gap = logit_gap_weight
        self.w_entropy = entropy_weight
        self.w_top1 = top1_weight

    def score(self, logits: torch.Tensor) -> float:
        """Score confidence from logits. Returns [0, 1] where 1 = very confident.

        Args:
            logits: (vocab_size,) or (seq_len, vocab_size) — last token's logits

        Returns:
            Confidence score in [0, 1]
        """
        # Use the last token's logits
        if logits.dim() > 1:
            logits = logits[-1]

        # Top-2 logit gap
        top2 = torch.topk(logits, 2).values
        logit_gap = (top2[0] - top2[1]).item()
        # Normalize: gap of 0 = 0 confidence, gap of 10 = max confidence
        gap_score = min(max(logit_gap / 10.0, 0), 1)

        # Entropy of softmax
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = math.log(probs.shape[0])
        # Normalize: entropy of 0 = 1 confidence, max entropy = 0 confidence
        entropy_score = 1 - (entropy / max_entropy)

        # Top-1 probability
        top1_prob = top2[0]  # actually the logit, but we want the probability
        top1_prob = probs.max().item()
        prob_score = top1_prob  # already in [0, 1]

        # Weighted combination
        confidence = (self.w_gap * gap_score +
                      self.w_entropy * entropy_score +
                      self.w_top1 * prob_score)
        return confidence

    def score_sequence(self, logits_seq: torch.Tensor) -> float:
        """Average confidence over a sequence of token logits.

        Args:
            logits_seq: (seq_len, vocab_size) — logits for each generated token

        Returns:
            Average confidence score in [0, 1]
        """
        if logits_seq.dim() == 1:
            return self.score(logits_seq)

        scores = []
        for i in range(logits_seq.shape[0]):
            scores.append(self.score(logits_seq[i]))
        return sum(scores) / max(len(scores), 1)


class RetryPolicy:
    """Decide whether to retry generation based on confidence.

    Policy:
    - If confidence < min_confidence AND attempts < max_retries → retry
    - Each retry increases temperature (more exploration)
    - After max_retries, accept the best attempt (highest confidence)
    """

    def __init__(self,
                 min_confidence: float = 0.5,
                 max_retries: int = 2,
                 temp_increase: float = 0.15,
                 base_temp: float = 0.7):
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.temp_increase = temp_increase
        self.base_temp = base_temp

    def should_retry(self, confidence: float, attempt: int) -> bool:
        """Should the model retry generation?

        Args:
            confidence: confidence score from ConfidenceScorer
            attempt: current attempt number (0-indexed)

        Returns:
            True if should retry, False if should accept
        """
        if attempt >= self.max_retries:
            return False
        return confidence < self.min_confidence

    def temperature_for_attempt(self, attempt: int) -> float:
        """Get temperature for a given attempt number.

        Attempt 0: base temperature
        Attempt 1: base + increase
        Attempt 2: base + 2*increase
        """
        return self.base_temp + attempt * self.temp_increase

    def select_best(self, attempts: list) -> tuple:
        """Select the best attempt from a list of (code, confidence) pairs.

        Args:
            attempts: list of (code, confidence) tuples

        Returns:
            Best (code, confidence) pair (highest confidence)
        """
        if not attempts:
            return ("", 0.0)
        return max(attempts, key=lambda x: x[1])
