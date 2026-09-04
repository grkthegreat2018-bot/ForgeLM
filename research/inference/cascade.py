"""R39-8: Model Cascade Routing.

Routes incoming queries across a tiered pair of models (small / large)
based on an estimated difficulty score.  Easy queries go to the cheap
small model; hard queries escalate to the large model.  This keeps
average latency / cost low while preserving quality on hard inputs.

Difficulty estimation is heuristic (no extra model call needed):

* prompt length          → short = easy, long = hard
* code / math keywords   → +0.2
* simple greetings       → 0.1
* everything else        → interpolated by length

Routing threshold is configurable (``difficulty_threshold``); queries
with difficulty >= threshold go to the large model.
"""
from __future__ import annotations

import re
from typing import Any, Protocol

__all__ = ["ModelCascade", "GenerativeModel"]


class GenerativeModel(Protocol):
    """Minimal model interface for cascade routing."""

    def generate(self, prompt: str, **kwargs: Any) -> str: ...


# ─── Difficulty heuristics ───────────────────────────────────────────────

# Keywords that signal a harder (code / math / reasoning) query.
_HARD_KEYWORDS = [
    # math
    "prove", "theorem", "integral", "derivative", "equation", "solve",
    "matrix", "eigen", "probability", "calculus", "algebra", "geometry",
    "combinatorics", "optimization", "lemma", "polynomial",
    # code
    "def ", "function", "class ", "import ", "algorithm", "debug",
    "compile", "runtime", "stack", "pointer", "recursion", "regex",
    "api", "async", "thread", "concurrency", "refactor", "sql",
    "python", "javascript", "rust", "c++", "java", "golang",
    # reasoning
    "analyze", "derive", "explain why", "compare", "trade-off",
    "architecture", "design pattern", "complex",
]

# Simple greeting / small-talk patterns.
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|good (morning|evening|afternoon)|"
    r"how (are|r) (you|u)|what'?s up|thanks|thank you|bye|goodbye)"
    r"[!?.\s]*$",
    re.IGNORECASE,
)

# Short-question patterns (yes/no, single-word answers).
_TRIVIAL_RE = re.compile(
    r"^\s*(yes|no|ok|sure|maybe|true|false|done|cool|nice|great)\s*[!?.]*$",
    re.IGNORECASE,
)


class ModelCascade:
    """Route queries between a small and a large model by difficulty.

    Parameters
    ----------
    small_model : object with ``generate(prompt, **kwargs) -> str``
    large_model : object with ``generate(prompt, **kwargs) -> str``
    difficulty_threshold : float
        Queries with estimated difficulty >= threshold are routed to the
        large model; the rest go to the small model.  Default 0.5.

    Example
    -------
    >>> cascade = ModelCascade(small, large, difficulty_threshold=0.5)
    >>> cascade.generate("Hi!")            # → small model
    >>> cascade.generate("Prove that...")  # → large model
    >>> cascade.stats()
    {'n_small': 1, 'n_large': 1, 'avg_difficulty': 0.45}
    """

    def __init__(self, small_model: GenerativeModel,
                 large_model: GenerativeModel,
                 difficulty_threshold: float = 0.5):
        if not 0.0 <= difficulty_threshold <= 1.0:
            raise ValueError(
                f"difficulty_threshold must be in [0, 1], got {difficulty_threshold}")
        self.small_model = small_model
        self.large_model = large_model
        self.difficulty_threshold = difficulty_threshold
        # stats
        self._n_small = 0
        self._n_large = 0
        self._difficulty_sum = 0.0
        self._n_total = 0

    # ─── difficulty estimation ─────────────────────────────────────────

    def estimate_difficulty(self, prompt: str) -> float:
        """Estimate query difficulty in [0, 1] using cheap heuristics.

        Rules (applied in priority order):
          1. Simple greeting / small-talk → 0.1
          2. Trivial one-word reply       → 0.05
          3. Short prompt (< 50 chars)    → 0.2
          4. Long prompt (> 500 chars)    → 0.8
          5. Otherwise                    → linear interpolation by length
          6. Code / math keywords         → +0.2 (capped at 1.0)
        """
        if not prompt or not prompt.strip():
            return 0.0

        text = prompt.strip()
        length = len(text)

        # 1. Greeting → very easy.
        if _GREETING_RE.match(text):
            base = 0.1
        # 2. Trivial reply.
        elif _TRIVIAL_RE.match(text):
            base = 0.05
        # 3. Short prompt.
        elif length < 50:
            base = 0.2
        # 4. Long prompt.
        elif length > 500:
            base = 0.8
        # 5. Interpolate between 0.2 and 0.8 by log-length.
        else:
            # Map [50, 500] → [0.2, 0.8] with a gentle log curve.
            import math
            t = (math.log(length) - math.log(50)) / (
                math.log(500) - math.log(50))
            t = max(0.0, min(1.0, t))
            base = 0.2 + 0.6 * t

        # 6. Keyword bump.
        lower = text.lower()
        if any(kw in lower for kw in _HARD_KEYWORDS):
            base = min(1.0, base + 0.2)

        return round(base, 4)

    # ─── routing ───────────────────────────────────────────────────────

    def route(self, prompt: str) -> str:
        """Return ``"small"`` or ``"large"`` based on estimated difficulty."""
        diff = self.estimate_difficulty(prompt)
        return "large" if diff >= self.difficulty_threshold else "small"

    # ─── generation ────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Route *prompt* to the appropriate model and generate."""
        target = self.route(prompt)
        diff = self.estimate_difficulty(prompt)

        # Update stats.
        self._n_total += 1
        self._difficulty_sum += diff
        if target == "small":
            self._n_small += 1
            return self.small_model.generate(prompt, **kwargs)
        else:
            self._n_large += 1
            return self.large_model.generate(prompt, **kwargs)

    # ─── stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return routing statistics.

        Returns
        -------
        dict with keys:
            n_small        : int    — queries routed to the small model
            n_large        : int    — queries routed to the large model
            n_total        : int    — total queries
            avg_difficulty : float  — mean estimated difficulty (0 if none)
        """
        avg = (self._difficulty_sum / self._n_total) if self._n_total else 0.0
        return {
            "n_small": self._n_small,
            "n_large": self._n_large,
            "n_total": self._n_total,
            "avg_difficulty": round(avg, 4),
        }

    def reset_stats(self) -> None:
        """Reset all routing statistics."""
        self._n_small = 0
        self._n_large = 0
        self._difficulty_sum = 0.0
        self._n_total = 0
