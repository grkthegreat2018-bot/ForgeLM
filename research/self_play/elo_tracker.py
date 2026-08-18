"""ELO-driven curriculum matchmaking for self-play.

Implements an ELO rating system that rates both the model and individual
prompts/tasks, then selects training prompts that match the model's current
skill level (targeting ~50% expected success rate — the Goldilocks zone
for maximum learning signal).

Research basis:
  - ELO (chess): rating system where expected score E = 1/(1+10^(ΔR/400))
  - AZR (NeurIPS 2025): Goldilocks difficulty, 50% success = max learnability
  - SPICE (Meta FAIR 2025): curriculum grounded in measured competence

The ELO approach improves on rolling-success-rate matching because:
  1. Per-prompt ratings capture individual difficulty (not just domain averages)
  2. The model rating is a single scalar that tracks overall skill progression
  3. Expected win probability gives a principled difficulty target (50%)
  4. Ratings auto-adjust after every interaction (no window-size tuning)

Usage:
    from research.self_play.elo_tracker import EloTracker
    elo = EloTracker(initial_rating=1200.0)
    elo.update_model_rating(prompt_id="task_42", success=True)
    prompt_ids = elo.select_prompts(model_rating=elo.model_rating, n=10)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class PromptRating:
    """ELO rating for a single prompt/task."""
    rating: float
    n_attempts: int = 0
    n_successes: int = 0
    last_attempt_step: int = 0


class EloTracker:
    """ELO matchmaking between model and prompts.

    Maintains:
      - model_rating: single ELO rating for the model's current skill
      - prompt_ratings: dict[prompt_id -> PromptRating] for each task

    The K-factor (sensitivity of rating updates) decreases as prompts accumulate
    more attempts, stabilizing ratings for well-tested prompts.
    """

    def __init__(self, initial_rating: float = 1200.0,
                 k_factor: float = 32.0,
                 k_factor_min: float = 8.0,
                 target_win_prob: float = 0.5,
                 win_prob_tolerance: float = 0.15,
                 seed: int = 42):
        """Initialize ELO tracker.

        Args:
            initial_rating: starting ELO for model and new prompts.
            k_factor: ELO K-factor (rating update sensitivity).
            k_factor_min: minimum K-factor for well-tested prompts.
            target_win_prob: target success probability for prompt selection
                             (0.5 = Goldilocks zone, max learning signal).
            win_prob_tolerance: ±tolerance around target for selection
                                (selects prompts with win_prob in
                                [target - tol, target + tol]).
            seed: RNG seed for reproducible prompt selection.
        """
        self.model_rating = initial_rating
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.k_factor_min = k_factor_min
        self.target_win_prob = target_win_prob
        self.win_prob_tolerance = win_prob_tolerance
        self.prompt_ratings: dict[str, PromptRating] = {}
        self.rng = random.Random(seed)
        self.step = 0

    def expected_win_prob(self, prompt_rating: float) -> float:
        """ELO expected win probability: E = 1 / (1 + 10^((Rp - Rm) / 400)).

        If model_rating > prompt_rating → E > 0.5 (model favored to solve).
        If model_rating < prompt_rating → E < 0.5 (prompt is harder).
        """
        return 1.0 / (1.0 + 10.0 ** ((prompt_rating - self.model_rating) / 400.0))

    def _effective_k(self, prompt_id: str) -> float:
        """K-factor that decreases as a prompt is tested more (stabilizes
        ratings for well-tested prompts)."""
        pr = self.prompt_ratings.get(prompt_id)
        if pr is None:
            return self.k_factor
        # K decreases linearly from k_factor to k_factor_min over 20 attempts
        decay = max(0.0, 1.0 - pr.n_attempts / 20.0)
        return self.k_factor_min + (self.k_factor - self.k_factor_min) * decay

    def register_prompt(self, prompt_id: str, rating: float | None = None):
        """Register a new prompt with an initial rating."""
        if prompt_id not in self.prompt_ratings:
            self.prompt_ratings[prompt_id] = PromptRating(
                rating=rating if rating is not None else self.initial_rating)

    def update_model_rating(self, prompt_id: str, success: bool):
        """Update both model and prompt ELO ratings after a solve attempt.

        Standard ELO update:
          Rm' = Rm + K * (actual - expected)
          Rp' = Rp - K * (actual - expected)  (zero-sum)

        where actual = 1.0 if success else 0.0.
        """
        self.register_prompt(prompt_id)
        pr = self.prompt_ratings[prompt_id]
        expected = self.expected_win_prob(pr.rating)
        actual = 1.0 if success else 0.0
        k = self._effective_k(prompt_id)

        delta = k * (actual - expected)
        self.model_rating += delta
        pr.rating -= delta  # zero-sum: model gains what prompt loses
        pr.n_attempts += 1
        if success:
            pr.n_successes += 1
        pr.last_attempt_step = self.step
        self.step += 1

    def select_prompts(self, prompt_ids: list[str] | None = None,
                       n: int = 10,
                       min_attempts: int = 0) -> list[str]:
        """Select prompts that match the model's current skill level.

        Targets prompts where the expected win probability is within
        [target - tolerance, target + tolerance] — the Goldilocks zone
        where the model has ~50% chance of solving (max learning signal).

        Args:
            prompt_ids: candidate prompt IDs (None = all registered).
            n: number of prompts to select.
            min_attempts: only select prompts with >= this many prior attempts
                          (0 = include never-tested prompts as exploration).

        Returns:
            list of prompt_ids, sorted by closeness to target win prob.
        """
        candidates = prompt_ids if prompt_ids is not None else list(
            self.prompt_ratings.keys())
        if not candidates:
            return []

        # Score each candidate by closeness to target win probability
        scored = []
        for pid in candidates:
            pr = self.prompt_ratings.get(pid)
            if pr is None:
                # Unregistered prompt — assume initial rating (neutral difficulty)
                exp = self.expected_win_prob(self.initial_rating)
            else:
                if pr.n_attempts < min_attempts:
                    continue
                exp = self.expected_win_prob(pr.rating)

            # Distance from target — lower is better (closer to 50%)
            distance = abs(exp - self.target_win_prob)
            # Add small random jitter to avoid always picking the same prompts
            jitter = self.rng.random() * 0.01
            scored.append((distance + jitter, pid))

        scored.sort()
        return [pid for _, pid in scored[:n]]

    def select_mixed_prompts(self, prompt_ids: list[str] | None = None,
                             n: int = 10,
                             exploration_ratio: float = 0.2) -> list[str]:
        """Select a mix of Goldilocks-matched and exploration prompts.

        exploration_ratio fraction of prompts are randomly selected (to
        discover new difficulty levels and avoid overfitting to known
        prompt ratings). The rest are ELO-matched to the Goldilocks zone.

        Args:
            prompt_ids: candidate prompt IDs (None = all registered).
            n: total number of prompts to select.
            exploration_ratio: fraction of prompts for random exploration.

        Returns:
            list of prompt_ids.
        """
        candidates = list(prompt_ids) if prompt_ids is not None else list(
            self.prompt_ratings.keys())
        if not candidates:
            return []

        n_explore = int(n * exploration_ratio)
        n_match = n - n_explore

        # ELO-matched prompts
        matched = self.select_prompts(candidates, n=n_match)

        # Exploration prompts (random, excluding already-matched)
        remaining = [p for p in candidates if p not in matched]
        if remaining and n_explore > 0:
            explore = self.rng.sample(remaining, min(n_explore, len(remaining)))
        else:
            explore = []

        return matched + explore

    def domain_stats(self) -> dict:
        """Return summary statistics for monitoring."""
        if not self.prompt_ratings:
            return {"model_rating": self.model_rating,
                    "n_prompts": 0, "mean_prompt_rating": 0.0,
                    "mean_success_rate": 0.0}
        ratings = [pr.rating for pr in self.prompt_ratings.values()]
        success_rates = [pr.n_successes / max(pr.n_attempts, 1)
                         for pr in self.prompt_ratings.values()]
        return {
            "model_rating": self.model_rating,
            "n_prompts": len(self.prompt_ratings),
            "mean_prompt_rating": sum(ratings) / len(ratings),
            "mean_success_rate": sum(success_rates) / len(success_rates),
            "step": self.step,
        }

    def get_prompt_rating(self, prompt_id: str) -> PromptRating | None:
        return self.prompt_ratings.get(prompt_id)
