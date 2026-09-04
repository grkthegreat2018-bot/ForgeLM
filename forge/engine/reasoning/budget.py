"""Reasoning Budget Controller — inference-time reasoning length control.

Implements Budget Guidance (ACL 2026 findings.1866) and SelfBudgeter
(ACL 2026 findings.1063) concepts for controlling reasoning length at
inference time, without retraining.

Key insight: reasoning models (e.g. DeepSeek-R1, Qwen3-thinking) often
"over-think" — generating long CoT traces that don't improve accuracy.
Budget Guidance uses a Gamma distribution predictor to estimate the
optimal thinking length, then applies soft token-level guidance to steer
generation toward that budget.

This module provides:
  - ReasoningBudgetController: predicts optimal budget and applies guidance
  - GammaPredictor: lightweight Gamma distribution predictor for thinking length
  - SoftBudgetGuidance: token-level logit biasing to adhere to budget
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class BudgetConfig:
    """Configuration for reasoning budget control."""
    # Default thinking token budget (mode of Gamma distribution).
    default_budget: int = 1024
    # Minimum budget (never go below this).
    min_budget: int = 128
    # Maximum budget (never exceed this).
    max_budget: int = 8192
    # Gamma distribution shape parameter (controls distribution shape).
    gamma_shape: float = 2.0
    # Gamma distribution scale parameter (controls spread).
    gamma_scale: float = 512.0
    # Soft guidance strength: how aggressively to bias logits near budget.
    # 0 = no guidance, 1 = strong guidance.
    guidance_strength: float = 0.3
    # Budget utilization window: start applying guidance when generation
    # reaches this fraction of the budget.
    guidance_start_fraction: float = 0.8
    # Whether to use difficulty-based budget prediction.
    use_difficulty_prediction: bool = True
    # Difficulty signal: entropy of the first token's logit distribution.
    # Higher entropy = harder question = larger budget.
    difficulty_entropy_threshold: float = 2.0


class GammaPredictor:
    """Gamma distribution predictor for optimal thinking length.

    Models the optimal thinking length as a Gamma distribution, where:
    - Shape parameter controls the distribution shape (higher = more peaked)
    - Scale parameter controls the spread (higher = more variable)

    The predictor uses a difficulty signal (e.g. first-token entropy) to
    adjust the distribution parameters: harder questions get larger budgets.
    """

    def __init__(self, shape: float = 2.0, scale: float = 512.0):
        self.shape = shape
        self.scale = scale

    def predict_budget(
        self,
        difficulty_signal: float | None = None,
        min_budget: int = 128,
        max_budget: int = 8192,
    ) -> int:
        """Predict the optimal thinking token budget.

        Args:
            difficulty_signal: difficulty score in [0, 1] (0 = easy, 1 = hard).
                If None, uses the mode of the Gamma distribution.
            min_budget: minimum budget.
            max_budget: maximum budget.

        Returns:
            Predicted budget (int), clamped to [min_budget, max_budget].
        """
        if difficulty_signal is None:
            # Mode of Gamma distribution = (shape - 1) * scale.
            mode = max((self.shape - 1) * self.scale, min_budget)
            return int(min(max(mode, min_budget), max_budget))

        # Scale the budget based on difficulty.
        # Easy (0) -> 0.5x mode, Hard (1) -> 2x mode.
        base_mode = max((self.shape - 1) * self.scale, min_budget)
        budget = base_mode * (0.5 + difficulty_signal * 1.5)
        return int(min(max(budget, min_budget), max_budget))

    def predict_difficulty(self, first_token_logits: torch.Tensor) -> float:
        """Predict difficulty from the first token's logit distribution.

        Higher entropy = more uncertain = harder question.

        Args:
            first_token_logits: (vocab,) or (batch, vocab) logits.

        Returns:
            Difficulty score in [0, 1].
        """
        if first_token_logits.dim() > 1:
            first_token_logits = first_token_logits[-1]
        probs = F.softmax(first_token_logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        # Normalize: entropy of uniform distribution over vocab is log(vocab).
        # For typical vocab sizes (~150k), log(vocab) ~ 11.9.
        # We map [0, 5] -> [0, 1] (5 is a reasonable max for first-token entropy).
        return min(entropy / 5.0, 1.0)


class SoftBudgetGuidance:
    """Soft token-level guidance to adhere to a thinking budget.

    Applies a gentle logit bias that encourages the model to transition
    to the final answer as it approaches the budget. The bias increases
    gradually as generation approaches and exceeds the budget.

    The guidance is "soft" — it doesn't force termination but makes it
    more likely by boosting tokens that indicate answer transition
    (e.g. "The answer is", "Therefore", "In conclusion").
    """

    def __init__(self, strength: float = 0.3, start_fraction: float = 0.8):
        self.strength = strength
        self.start_fraction = start_fraction

    def apply_guidance(
        self,
        logits: torch.Tensor,
        current_tokens: int,
        budget: int,
        answer_transition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply soft budget guidance to logits.

        Args:
            logits: (batch, vocab) logits for the current step.
            current_tokens: number of tokens generated so far.
            budget: target thinking token budget.
            answer_transition_tokens: token IDs that indicate transition
                to final answer (e.g. "The", "answer", "Therefore").
                If None, no token-level boosting is applied (only entropy
                reduction via temperature-like scaling).

        Returns:
            Modified logits with budget guidance applied.
        """
        if budget <= 0 or current_tokens < budget * self.start_fraction:
            return logits

        # Compute how far past the start fraction we are.
        progress = (current_tokens - budget * self.start_fraction) / (
            budget * (1 - self.start_fraction) + 1e-8)
        progress = min(progress, 1.0)

        # Apply guidance: boost answer transition tokens.
        if answer_transition_tokens is not None and len(answer_transition_tokens) > 0:
            boost = self.strength * progress
            logits[:, answer_transition_tokens] += boost

        # Also slightly sharpen the distribution (reduce temperature-like effect)
        # to encourage more decisive tokens as we approach the budget.
        if progress > 0.5:
            sharpen = 1.0 + self.strength * progress * 0.5
            logits = logits * sharpen

        return logits


class ReasoningBudgetController:
    """Reasoning budget controller — predicts and enforces thinking length.

    Combines GammaPredictor (difficulty-aware budget prediction) with
    SoftBudgetGuidance (token-level guidance) to control reasoning length
    at inference time. Training-free.

    Usage:
        controller = ReasoningBudgetController(config)
        budget = controller.predict_budget(first_token_logits)
        for step, logits in enumerate(reasoning_loop):
            guided_logits = controller.apply_guidance(logits, step, budget)
            # ... sample from guided_logits ...

    Or with ForgeEngine:
        controller = ReasoningBudgetController()
        budget = controller.predict_budget(prompt_logits)
        # Pass budget as max_new_tokens for thinking phase
    """

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.predictor = GammaPredictor(
            shape=self.config.gamma_shape,
            scale=self.config.gamma_scale,
        )
        self.guidance = SoftBudgetGuidance(
            strength=self.config.guidance_strength,
            start_fraction=self.config.guidance_start_fraction,
        )
        self._predicted_budget: int | None = None
        self._difficulty: float | None = None

    def predict_budget(
        self,
        first_token_logits: torch.Tensor | None = None,
        difficulty: float | None = None,
    ) -> int:
        """Predict the optimal thinking token budget.

        Args:
            first_token_logits: logits from the first generated token.
                Used to estimate difficulty via entropy. If None, uses
                default budget.
            difficulty: explicit difficulty score [0, 1]. Overrides
                first_token_logits if provided.

        Returns:
            Predicted budget (int).
        """
        if difficulty is None and first_token_logits is not None:
            if self.config.use_difficulty_prediction:
                difficulty = self.predictor.predict_difficulty(first_token_logits)
            else:
                difficulty = None
        elif difficulty is None:
            difficulty = None

        self._difficulty = difficulty
        self._predicted_budget = self.predictor.predict_budget(
            difficulty,
            min_budget=self.config.min_budget,
            max_budget=self.config.max_budget,
        )
        return self._predicted_budget

    def apply_guidance(
        self,
        logits: torch.Tensor,
        current_tokens: int,
        answer_transition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply soft budget guidance to logits.

        Args:
            logits: (batch, vocab) logits for the current step.
            current_tokens: number of tokens generated so far.
            answer_transition_tokens: token IDs that indicate answer transition.

        Returns:
            Modified logits with budget guidance applied.
        """
        if self._predicted_budget is None:
            return logits
        return self.guidance.apply_guidance(
            logits, current_tokens, self._predicted_budget,
            answer_transition_tokens,
        )

    def reset(self):
        """Reset state for a new reasoning session."""
        self._predicted_budget = None
        self._difficulty = None

    @property
    def stats(self) -> dict:
        """Return statistics about the current reasoning session."""
        return {
            "predicted_budget": self._predicted_budget,
            "difficulty": self._difficulty,
            "config": {
                "default_budget": self.config.default_budget,
                "min_budget": self.config.min_budget,
                "max_budget": self.config.max_budget,
                "guidance_strength": self.config.guidance_strength,
            },
        }
