"""Semantic input-output matching for self-play verification.

I/O-focus revamp: correctness verification should accept numerically or
structurally equivalent outputs, not just byte-identical strings. A model
that prints "3.0" when the expected answer is "3.00", or prints list items
in a different order for an unordered task, produces correct I/O behaviour
that a plain string comparison would reject — starving training of valid
positive signal.

Public API:
    io_similarity(expected, actual) -> float in [0, 1]
    io_match(expected, actual, threshold=0.99) -> bool

Deterministic, cheap, stdlib-only.
"""
from __future__ import annotations

import math
import re

_RE_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_RE_SPLIT = re.compile(r"[\s,;]+")


def _try_float(s: str) -> float | None:
    """Parse a string as a float; return None if not numeric."""
    s = s.strip()
    if not s or not _RE_FLOAT.match(s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _token_multiset(text: str) -> list[str]:
    """Split output into comparable tokens (newlines, commas, semicolons, spaces).

    Tokens are stripped of surrounding brackets/quotes so that a printed
    Python list "[1, 2, 3]" compares equal to "1 2 3".
    """
    toks = (t.strip("[](){}'\"") for t in _RE_SPLIT.split(text.strip()))
    return sorted(t for t in toks if t)


def io_similarity(expected: str | None, actual: str | None) -> float:
    """Semantic similarity of expected vs actual program output, in [0, 1].

    Tiers (first match wins):
      1. Exact match (stripped)                -> 1.0
      2. Case-insensitive exact match          -> 0.98
      3. Numeric equality (tolerant floats)    -> 1.0 / partial by rel. error
      4. Token multiset equality (unordered)   -> 0.95
      5. Line-level F1 overlap                 -> up to 0.9
      6. Otherwise                             -> 0.0
    """
    if expected is None or actual is None:
        return 0.0
    exp = expected.strip()
    act = actual.strip()
    if not exp:
        return 1.0 if not act else 0.0
    if not act:
        return 0.0

    # 1. Exact
    if exp == act:
        return 1.0

    # 2. Case-insensitive
    if exp.lower() == act.lower():
        return 0.98

    # 3. Numeric (single value, or equal-length numeric sequences)
    exp_f, act_f = _try_float(exp), _try_float(act)
    if exp_f is not None and act_f is not None:
        if math.isclose(exp_f, act_f, rel_tol=1e-6, abs_tol=1e-9):
            return 1.0
        # Partial credit scaled by relative error (decays to 0 at 100% off)
        denom = max(abs(exp_f), abs(act_f), 1e-9)
        rel_err = abs(exp_f - act_f) / denom
        return max(0.0, 0.9 * (1.0 - rel_err))

    exp_toks, act_toks = _token_multiset(exp), _token_multiset(act)
    exp_nums = [_try_float(t) for t in exp_toks]
    act_nums = [_try_float(t) for t in act_toks]

    # 3b. Numeric sequence (e.g. multi-line or space-separated numbers)
    if (len(exp_toks) == len(act_toks) and exp_toks
            and all(n is not None for n in exp_nums)
            and all(n is not None for n in act_nums)):
        if all(math.isclose(e, a, rel_tol=1e-6, abs_tol=1e-9)
               for e, a in zip(exp_nums, act_nums, strict=True)):
            return 1.0
        per_tok = []
        for e, a in zip(exp_nums, act_nums, strict=True):
            denom = max(abs(e), abs(a), 1e-9)
            per_tok.append(max(0.0, 1.0 - abs(e - a) / denom))
        return 0.9 * (sum(per_tok) / len(per_tok))

    # 4a. Ordered token equality (whitespace/punctuation normalization only)
    exp_seq = [t for t in (x.strip("[](){}'\"") for x in _RE_SPLIT.split(exp)) if t]
    act_seq = [t for t in (x.strip("[](){}'\"") for x in _RE_SPLIT.split(act)) if t]
    if exp_seq == act_seq:
        return 1.0

    # 4b. Unordered token multiset equality (order-insensitive tasks)
    if exp_toks == act_toks:
        return 0.95

    # 5. Line/token-level F1 overlap (partial credit, capped below threshold)
    exp_set, act_set = set(exp_toks), set(act_toks)
    if exp_set and act_set:
        inter = len(exp_set & act_set)
        precision = inter / len(act_set)
        recall = inter / len(exp_set)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            return 0.9 * f1

    return 0.0


def io_match(expected: str | None, actual: str | None, threshold: float = 0.99) -> bool:
    """Boolean correctness check: True when outputs are semantically equivalent."""
    if expected is None:
        return False
    return io_similarity(expected, actual) >= threshold
