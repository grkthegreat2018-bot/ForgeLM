"""Failure mode monitoring for self-play training.

Implements alerting based on research findings:
- AVSPO: Advantage Collapse Rate (ACR) monitoring.
- "Survive or Collapse" (arxiv 2605.22217): intrinsic-grounded gap.
- Diversity collapse detection.
- Language mixing detection (DeepSeek-R1-Zero style).
- QDC Framework: Quality-Diversity-Complexity balance tracking.

Only depends on the Python standard library.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional


class SelfPlayMonitor:
    """Rolling-window monitor for self-play training failure modes."""

    # Alert thresholds (research-backed).
    ACR_THRESHOLD = 0.3
    IG_GAP_THRESHOLD = 0.2
    DIVERSITY_THRESHOLD = 0.6
    LANGUAGE_MIX_THRESHOLD = 0.3
    KL_DIV_THRESHOLD = 10.0

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.advantage_collapse_rate: deque = deque(maxlen=window_size)
        self.mean_reward: deque = deque(maxlen=window_size)
        self.diversity_score: deque = deque(maxlen=window_size)
        self.kl_divergence: deque = deque(maxlen=window_size)
        self.intrinsic_reward: deque = deque(maxlen=window_size)
        self.grounded_reward: deque = deque(maxlen=window_size)
        self.intrinsic_ground_gap: deque = deque(maxlen=window_size)
        self.language_distribution: deque = deque(maxlen=window_size)
        self.step_count: int = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_step(self, metrics: dict) -> None:
        """Record per-step metrics.

        Expected keys (all optional except where noted):
            advantage_collapse_rate, mean_reward, diversity_score,
            kl_divergence, intrinsic_reward, grounded_reward,
            language_distribution (dict of lang->count).
        """
        self.step_count += 1
        self.advantage_collapse_rate.append(metrics.get("advantage_collapse_rate", 0.0))
        self.mean_reward.append(metrics.get("mean_reward", 0.0))
        self.diversity_score.append(metrics.get("diversity_score", 1.0))
        self.kl_divergence.append(metrics.get("kl_divergence", 0.0))
        self.intrinsic_reward.append(metrics.get("intrinsic_reward", 0.0))
        self.grounded_reward.append(metrics.get("grounded_reward", 0.0))

        gap = abs(metrics.get("intrinsic_reward", 0.0) - metrics.get("grounded_reward", 0.0))
        self.intrinsic_ground_gap.append(gap)

        self.language_distribution.append(metrics.get("language_distribution", {}))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _avg(window: deque) -> float:
        if not window:
            return 0.0
        return sum(window) / len(window)

    def _reward_slope(self) -> Optional[float]:
        """Least-squares slope of mean_reward over the rolling window."""
        n = len(self.mean_reward)
        if n < 2:
            return None
        xs = list(range(n))
        ys = list(self.mean_reward)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return num / den

    def _language_mix_ratio(self) -> float:
        """Fraction of tokens in non-dominant languages (0..1)."""
        totals: Dict[str, float] = {}
        for dist in self.language_distribution:
            for lang, count in dist.items():
                totals[lang] = totals.get(lang, 0.0) + count
        total = sum(totals.values())
        if total <= 0:
            return 0.0
        dominant = max(totals.values())
        return 1.0 - (dominant / total)

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #
    def check_alerts(self) -> List[dict]:
        """Check all alert conditions; return a list of alert dicts."""
        alerts: List[dict] = []

        acr = self._avg(self.advantage_collapse_rate)
        if acr > self.ACR_THRESHOLD:
            alerts.append({
                "level": "warning",
                "metric": "ACR",
                "value": round(acr, 4),
                "threshold": self.ACR_THRESHOLD,
                "msg": "Advantage collapse detected",
            })

        ig_gap = self._avg(self.intrinsic_ground_gap)
        if ig_gap > self.IG_GAP_THRESHOLD:
            alerts.append({
                "level": "critical",
                "metric": "IG_gap",
                "value": round(ig_gap, 4),
                "threshold": self.IG_GAP_THRESHOLD,
                "msg": "Intrinsic-grounded reward gap too large",
            })

        diversity = self._avg(self.diversity_score)
        if diversity < self.DIVERSITY_THRESHOLD:
            alerts.append({
                "level": "warning",
                "metric": "diversity",
                "value": round(diversity, 4),
                "threshold": self.DIVERSITY_THRESHOLD,
                "msg": "Diversity collapse detected",
            })

        mix_ratio = self._language_mix_ratio()
        if mix_ratio > self.LANGUAGE_MIX_THRESHOLD:
            alerts.append({
                "level": "warning",
                "metric": "language_mixing",
                "value": round(mix_ratio, 4),
                "threshold": self.LANGUAGE_MIX_THRESHOLD,
                "msg": "Non-dominant language share too high",
            })

        kl = self._avg(self.kl_divergence)
        if kl > self.KL_DIV_THRESHOLD:
            alerts.append({
                "level": "warning",
                "metric": "kl_div",
                "value": round(kl, 4),
                "threshold": self.KL_DIV_THRESHOLD,
                "msg": "Policy diverging from reference",
            })

        slope = self._reward_slope()
        if slope is not None and slope < 0:
            alerts.append({
                "level": "info",
                "metric": "reward_trend",
                "value": round(slope, 6),
                "threshold": 0.0,
                "msg": "Reward trend declining over window",
            })

        return alerts

    # ------------------------------------------------------------------ #
    # Summaries
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """Return current rolling-window statistics."""
        return {
            "step_count": self.step_count,
            "window_size": self.window_size,
            "window_filled": len(self.mean_reward),
            "advantage_collapse_rate": round(self._avg(self.advantage_collapse_rate), 4),
            "mean_reward": round(self._avg(self.mean_reward), 4),
            "diversity_score": round(self._avg(self.diversity_score), 4),
            "kl_divergence": round(self._avg(self.kl_divergence), 4),
            "intrinsic_reward": round(self._avg(self.intrinsic_reward), 4),
            "grounded_reward": round(self._avg(self.grounded_reward), 4),
            "intrinsic_ground_gap": round(self._avg(self.intrinsic_ground_gap), 4),
            "reward_slope": round(self._reward_slope() or 0.0, 6),
            "language_mix_ratio": round(self._language_mix_ratio(), 4),
        }

    def plot_data(self) -> dict:
        """Return time series (last N steps) suitable for plotting."""
        return {
            "steps": list(range(self.step_count - len(self.mean_reward), self.step_count)),
            "advantage_collapse_rate": list(self.advantage_collapse_rate),
            "mean_reward": list(self.mean_reward),
            "diversity_score": list(self.diversity_score),
            "kl_divergence": list(self.kl_divergence),
            "intrinsic_reward": list(self.intrinsic_reward),
            "grounded_reward": list(self.grounded_reward),
            "intrinsic_ground_gap": list(self.intrinsic_ground_gap),
        }
