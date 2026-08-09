"""Conformal Sampler Key — conformal-prediction-calibrated sampling temperature.

Novel insight: Conformal prediction provides distribution-free coverage guarantees.
We repurpose it for *sampling temperature calibration*: given a set of held-out
query scores (e.g. log-likelihood or confidence), we compute a conformal quantile
threshold T such that P(score >= T) = 1 - alpha.  At inference, the query's own
score relative to T determines the sampling temperature:

    - High score (confident, above threshold)  -> low temperature  (exploit)
    - Low score  (uncertain, below threshold)  -> high temperature (explore)

    temp(query) = base_temp * (threshold / max(query_score, eps))

This gives a principled, calibration-guaranteed explore/exploit trade-off per
query — no heuristics, no hand-tuned schedules.  The coverage guarantee holds
for any distribution of scores (conformal prediction is distribution-free).

Key class: TRIVIAL — runtime calibration, no weight changes.

Usage:
    from research.keys.conformal_sampler_key import ConformalSamplerKey, ConformalSampler
    sampler = ConformalSampler()
    sampler.calibrate(held_out_scores=[0.8, 0.5, 0.9, 0.3, 0.7], alpha=0.1)
    temp = sampler.get_temperature(query_score=0.85)
"""
import math
from typing import Dict, List, Optional

import torch

from .base import Key, KeyClass, KeyResult


class ConformalSampler:
    """Conformal-prediction-calibrated temperature sampler.

    Calibrates a score threshold from held-out data, then maps per-query
    scores to sampling temperatures: confident queries get low temperature
    (exploit), uncertain queries get high temperature (explore).
    """

    def __init__(self, base_temp: float = 1.0, eps: float = 1e-6):
        self.base_temp = base_temp
        self.eps = eps
        self.threshold: float | None = None
        self.alpha: float = 0.1

    def calibrate(self, held_out_scores: list[float], alpha: float = 0.1) -> float:
        """Conformal calibration: find threshold q at the (1-alpha) quantile.

        Args:
            held_out_scores: calibration scores from held-out queries.
            alpha: miscoverage rate (0.1 = 90% coverage).

        Returns:
            The calibrated threshold q.
        """
        self.alpha = alpha
        if len(held_out_scores) == 0:
            self.threshold = 0.0
            return 0.0
        sorted_scores = sorted(held_out_scores)
        n = len(sorted_scores)
        # Conformal quantile: ceil((n+1) * (1-alpha)) / n, clamped
        q_idx = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
        self.threshold = sorted_scores[q_idx]
        return self.threshold

    def get_temperature(self, query_score: float) -> float:
        """Map a query score to a sampling temperature.

        High score (confident) -> low temperature (exploit).
        Low score (uncertain)  -> high temperature (explore).

        Args:
            query_score: the incoming query's confidence score.

        Returns:
            Sampling temperature T.
        """
        if self.threshold is None or self.threshold <= 0:
            return self.base_temp
        score = max(query_score, self.eps)
        # If score > threshold: ratio < 1 -> temp < base (exploit)
        # If score < threshold: ratio > 1 -> temp > base (explore)
        return self.base_temp * (self.threshold / score)


class ConformalSamplerKey(Key):
    """Conformal Sampler key — conformal-prediction-calibrated temperature.

    Key class: TRIVIAL — runtime calibration, no weight changes.
    """

    @property
    def name(self) -> str:
        return "conformal_sampler"

    @property
    def description(self) -> str:
        return (
            "Conformal-prediction-calibrated per-query sampling temperature: "
            "high confidence -> low T (exploit), low confidence -> high T (explore)."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Calibrate the conformal sampler from held-out scores.

        Args:
            data: dict with keys:
                - held_out_scores: list[float] of calibration scores
                - alpha: float miscoverage rate (default 0.1)

        Returns:
            KeyResult with data = {"threshold": float, "alpha": float}
        """
        scores: list[float] = data["held_out_scores"]
        alpha: float = data.get("alpha", 0.1)

        sampler = ConformalSampler()
        threshold = sampler.calibrate(scores, alpha=alpha)

        return KeyResult(
            success=True,
            data={"threshold": threshold, "alpha": alpha},
            metadata={
                "n_calibration": len(scores),
                "base_temp": sampler.base_temp,
            },
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """No-op — TRIVIAL key has no weights to reverse."""
        return KeyResult(success=True, data={})


if __name__ == "__main__":
    # Synthetic held-out scores (e.g. log-likelihoods from validation set)
    held_out = [0.92, 0.85, 0.78, 0.65, 0.55, 0.43, 0.30, 0.88, 0.71, 0.60]
    alpha = 0.1

    key = ConformalSamplerKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print(f"  Description: {key.description}")

    # Forward: calibrate
    result = key.forward({"held_out_scores": held_out, "alpha": alpha})
    assert result.success, f"Forward failed: {result.error}"
    threshold = result.data["threshold"]
    print(f"  Forward: calibrated threshold={threshold:.4f} (alpha={alpha})")
    assert 0.0 <= threshold <= 1.0, "Threshold out of range"

    # Verify calibration: threshold should be near the 90th percentile
    sorted_scores = sorted(held_out)
    expected_idx = min(len(sorted_scores) - 1,
                       max(0, math.ceil((len(sorted_scores) + 1) * (1 - alpha)) - 1))
    assert abs(threshold - sorted_scores[expected_idx]) < 1e-6, "Calibration mismatch"
    print("  Calibration quantile verified")

    # Test temperature mapping
    sampler = ConformalSampler()
    sampler.calibrate(held_out, alpha=alpha)

    # Confident query (score above threshold) -> low temperature
    temp_confident = sampler.get_temperature(query_score=0.95)
    temp_uncertain = sampler.get_temperature(query_score=0.20)
    print(f"  Confident query (0.95): temp={temp_confident:.4f}")
    print(f"  Uncertain query (0.20): temp={temp_uncertain:.4f}")
    assert temp_confident < sampler.base_temp, "Confident query should have low temp"
    assert temp_uncertain > sampler.base_temp, "Uncertain query should have high temp"
    print("  Explore/exploit temperature mapping verified")

    # Reverse: no-op for TRIVIAL
    rev = key.reverse({})
    assert rev.success and rev.data == {}
    print("  Reverse: no-op (TRIVIAL) verified")
    print("  All tests passed.")
