"""Random task domain simulators — math / algorithm / logic problem scoring.

These simulators do NOT generate the problem (the domain does that and stores
it on `domain.current_problem`). Instead they receive the GEN MODEL'S answer
in `config["answer"]`, look up the stored problem + its real answer, and score
the model's response.

Each simulator returns a metrics dict consumed by RewardGuard:
  - correct: 1.0 / 0.0
  - abs_error: |model_answer - real_answer|
  - relative_error: abs_error / max(|real_answer|, 1e-9)
  - time_s: wall-clock seconds the gen model took (set by the domain)
  - focus_score: 1.0 if the model ignored distractions (correct despite noise),
                 0.0 if distracted (wrong when noise was present)
  - speed_score: 1.0 / (1.0 + time_s) — faster is better
  - difficulty: the problem's difficulty level (1-5) for behavioral binning
  - correctness: alias of correct for behavioral_dims naming
"""
from __future__ import annotations

import math
import re
from typing import Any

from . import register


# ---------------------------------------------------------------------------
# Helpers — safe expression evaluation + answer parsing
# ---------------------------------------------------------------------------

# Only these characters are allowed in a math expression. Anything else
# (letters, underscores, dots) is stripped before eval so there is no way to
# reach builtins or construct arbitrary code.
_SAFE_EXPR_CHARS = set("0123456789+-*/%(). ")
_SAFE_OPS = {"+", "-", "*", "/", "%", "**"}


def _sanitize_expression(expr: str) -> str:
    """Strip everything except digits, operators, parens, spaces, dots."""
    # Keep '**' intact: the char filter keeps '*' so '**' survives.
    return "".join(c for c in expr if c in _SAFE_EXPR_CHARS).strip()


def _safe_eval(expr: str) -> float:
    """Evaluate a sanitized arithmetic expression with no builtins."""
    clean = _sanitize_expression(expr)
    if not clean:
        return float("nan")
    # Restrict globals/locals to empty dicts — no builtins reachable.
    try:
        result = eval(clean, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return float("inf")
    except Exception:
        return float("nan")
    return float(result)


def _extract_number(text: str) -> float | None:
    """Pull the first plausible numeric answer out of free-form model text."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Direct number
    try:
        return float(s)
    except ValueError:
        pass
    # "yes" / "no" → 1.0 / 0.0 (for logic domains)
    low = s.lower()
    if low.startswith("yes"):
        return 1.0
    if low.startswith("no"):
        return 0.0
    # First number (int or float, optional leading sign) in the text
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _base_metrics(real_answer: float, model_answer: float | None,
                  time_s: float, has_distractions: bool,
                  difficulty: int) -> dict[str, float]:
    """Build the common metrics dict shared by all three simulators."""
    if model_answer is None:
        correct = 0.0
        abs_error = float("inf")
    else:
        # Use a tolerance so integer-valued float answers match exactly.
        abs_error = abs(model_answer - real_answer)
        correct = 1.0 if abs_error <= 1e-6 else 0.0
    rel = abs_error / max(abs(real_answer), 1e-9) if math.isfinite(abs_error) else float("inf")
    # focus_score: only meaningful when distractions were present.
    #   correct despite noise → 1.0; wrong despite noise → 0.0.
    #   When no distractions, focus is trivially 1.0 (nothing to ignore).
    if has_distractions:
        focus_score = 1.0 if correct > 0.5 else 0.0
    else:
        focus_score = 1.0
    return {
        "correct": float(correct),
        "correctness": float(correct),
        "abs_error": float(abs_error) if math.isfinite(abs_error) else 1e9,
        "relative_error": float(rel) if math.isfinite(rel) else 1e9,
        "time_s": float(time_s),
        "focus_score": float(focus_score),
        "speed_score": float(1.0 / (1.0 + max(time_s, 0.0))),
        "difficulty": float(difficulty),
    }


# ---------------------------------------------------------------------------
# 1. random_math_simulate
# ---------------------------------------------------------------------------

@register("random_math_simulate")
def random_math_simulate(config: dict, domain=None) -> dict:
    """Score the gen model's answer to a random math problem.

    The problem (with its real answer + metadata) is stored on
    ``domain.current_problem`` by the domain's evaluate(). The model's answer
    arrives in ``config["answer"]``.
    """
    problem = _get_problem(domain, "math")
    real_answer = float(problem.get("real_answer", float("nan")))
    has_distractions = bool(problem.get("has_distractions", False))
    difficulty = int(problem.get("difficulty", 1))
    time_s = float(config.get("time_s", problem.get("time_s", 0.0)))
    model_answer = _extract_number(config.get("answer"))
    return _base_metrics(real_answer, model_answer, time_s, has_distractions, difficulty)


# ---------------------------------------------------------------------------
# 2. random_algorithm_simulate
# ---------------------------------------------------------------------------

@register("random_algorithm_simulate")
def random_algorithm_simulate(config: dict, domain=None) -> dict:
    """Score the gen model's answer to a random algorithmic puzzle."""
    problem = _get_problem(domain, "algorithm")
    real_answer = float(problem.get("real_answer", float("nan")))
    has_distractions = bool(problem.get("has_distractions", False))
    difficulty = int(problem.get("difficulty", 1))
    time_s = float(config.get("time_s", problem.get("time_s", 0.0)))
    model_answer = _extract_number(config.get("answer"))
    return _base_metrics(real_answer, model_answer, time_s, has_distractions, difficulty)


# ---------------------------------------------------------------------------
# 3. random_logic_simulate
# ---------------------------------------------------------------------------

@register("random_logic_simulate")
def random_logic_simulate(config: dict, domain=None) -> dict:
    """Score the gen model's yes/no answer to a random logic problem.

    The real answer is stored as 1.0 (yes) / 0.0 (no). The model's answer is
    parsed via _extract_number which maps "yes"/"no" to 1.0/0.0.
    """
    problem = _get_problem(domain, "logic")
    real_answer = float(problem.get("real_answer", float("nan")))
    has_distractions = bool(problem.get("has_distractions", False))
    difficulty = int(problem.get("difficulty", 1))
    time_s = float(config.get("time_s", problem.get("time_s", 0.0)))
    model_answer = _extract_number(config.get("answer"))
    return _base_metrics(real_answer, model_answer, time_s, has_distractions, difficulty)


# ---------------------------------------------------------------------------
# Shared problem lookup
# ---------------------------------------------------------------------------

def _get_problem(domain: Any, expected_kind: str) -> dict:
    """Retrieve the current problem dict from the domain instance.

    Falls back to an empty problem (→ all-zero metrics) when no domain is
    passed or no problem has been set, so the simulator is safe to call in
    isolation for testing.
    """
    if domain is None:
        return {"kind": expected_kind, "real_answer": float("nan"),
                "has_distractions": False, "difficulty": 1}
    problem = getattr(domain, "current_problem", None)
    if not problem:
        return {"kind": expected_kind, "real_answer": float("nan"),
                "has_distractions": False, "difficulty": 1}
    return problem
