"""Goal-Oriented Self-Play Scorer — multi-dimensional quality scoring.

Scores solutions on 5 soft dimensions (correctness is a gate):
  1. Minimalism  (0.35) — AST node count, soft floor (structural conciseness)
  2. Efficiency  (0.25) — exec time on stress-test input (algorithmic quality)
  3. Diversity   (0.20) — Dice's coefficient on AST fingerprints (creativity)
  4. Consistency (0.10) — K-sample output agreement (self-consistency, VERSE)
  5. Confidence  (0.10) — model mean logprob (model certainty)

Correctness is a GATE: if any I/O test case fails, quality = 0.0.
Among correct solutions, the 5 soft dims compose to [0, 1].

Anti-redundancy (Redundancy-Aware RLVR 2026): solutions with Dice similarity
> 0.85 to already-accepted solutions for the same goal are REJECTED to prevent
the model from spamming one approach.

AST diversity uses Python stdlib `ast` module — no external dependencies.
Dice's coefficient on node-type multisets (JPlag-inspired):
  Dice = 2 * |A ∩ B| / (|A| + |B|)
where A, B are multisets of AST node types and |A ∩ B| = sum(min(a[k], b[k])).

Usage:
    from research.evaluation.goal_scorer import GoalScorer
    scorer = GoalScorer()
    result = scorer.score(code, test_cases, exec_results, model_telemetry,
                          seen_fingerprints=accepted_fingerprints_for_this_goal)
    if result.accepted:
        scorer.record_fingerprint(goal_id, result.fingerprint)
"""
import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─── AST fingerprint ──────────────────────────────────────────────────

def extract_ast_fingerprint(code: str) -> dict[str, int]:
    """Extract AST node-type multiset from Python code.

    Returns a dict mapping node type names to counts.
    E.g. {"FunctionDef": 1, "For": 2, "BinOp": 5, "Return": 1, ...}

    This captures the STRUCTURAL approach (loops vs recursion vs math ops)
    without being sensitive to variable names or formatting.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def count_ast_nodes(code: str) -> int:
    """Count total AST nodes (structural complexity metric for minimalism)."""
    fp = extract_ast_fingerprint(code)
    return sum(fp.values())


def dice_coefficient(fp_a: dict[str, int], fp_b: dict[str, int]) -> float:
    """Dice's coefficient on two AST fingerprint multisets.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    where |A ∩ B| = sum of min(a[k], b[k]) for all keys k.
    Returns 0.0 if both are empty, 1.0 if identical.
    """
    size_a = sum(fp_a.values())
    size_b = sum(fp_b.values())
    if size_a == 0 and size_b == 0:
        return 1.0  # both empty = identical (degenerate)
    if size_a == 0 or size_b == 0:
        return 0.0  # one empty = completely different

    all_keys = set(fp_a.keys()) | set(fp_b.keys())
    intersection = sum(min(fp_a.get(k, 0), fp_b.get(k, 0)) for k in all_keys)
    return 2.0 * intersection / (size_a + size_b)


def max_similarity_to_seen(fingerprint: dict[str, int],
                           seen: list[dict[str, int]]) -> float:
    """Max Dice similarity of this fingerprint vs all seen fingerprints."""
    if not seen:
        return 0.0
    return max(dice_coefficient(fingerprint, s) for s in seen)


# ─── Scoring result ───────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Result of scoring a single solution attempt."""
    quality: float                        # composite [0, 1], 0 if failed gate
    correct: bool                         # passed all I/O test cases
    accepted: bool                        # correct AND not redundant
    rejected_reason: str = ""             # why rejected (if not accepted)
    scores: dict[str, float] = field(default_factory=dict)
    # individual dimension scores: minimalism, efficiency, diversity, etc.
    fingerprint: dict[str, int] = field(default_factory=dict)
    ast_node_count: int = 0
    exec_time_ms: float = 0.0
    max_similarity: float = 0.0           # max Dice sim to seen solutions
    tokens_generated: int = 0


# ─── Goal scorer ──────────────────────────────────────────────────────

class GoalScorer:
    """Multi-dimensional goal-oriented scorer.

    Scoring weights (among correct solutions):
      Minimalism  0.35 — AST node count (structural conciseness)
      Efficiency  0.25 — exec time on stress input (algorithmic quality)
      Diversity   0.20 — 1 - max_dice_sim_to_seen (creativity bonus)
      Consistency 0.10 — K-sample output agreement (self-consistency)
      Confidence  0.10 — model mean logprob (model certainty)

    Anti-redundancy: reject if max_dice_sim > REDUNDANCY_THRESHOLD.
    """

    # Soft floors and ceilings for normalization
    MIN_AST_FLOOR = 20       # <= 20 AST nodes = full minimalism score
    MIN_AST_CEILING = 150    # >= 150 nodes = zero minimalism score
    EFF_FLOOR_MS = 5.0       # <= 5ms = full efficiency score
    EFF_CEILING_MS = 200.0   # >= 200ms = zero efficiency score
    REDUNDANCY_THRESHOLD = 0.85  # reject if Dice sim > this (RLVR 2026)

    WEIGHTS = {
        "minimalism": 0.35,
        "efficiency": 0.25,
        "diversity": 0.20,
        "consistency": 0.10,
        "confidence": 0.10,
    }

    def __init__(self):
        # Per-goal accepted fingerprints for diversity tracking
        self._seen: dict[str, list[dict[str, int]]] = {}

    def record_fingerprint(self, goal_id: str, fingerprint: dict[str, int]):
        """Record an accepted solution's fingerprint for future diversity scoring."""
        self._seen.setdefault(goal_id, []).append(fingerprint)

    def get_seen(self, goal_id: str) -> list[dict[str, int]]:
        return self._seen.get(goal_id, [])

    def clear_goal(self, goal_id: str):
        """Clear seen fingerprints for a goal (e.g. between training runs)."""
        self._seen.pop(goal_id, None)

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def _score_minimalism(self, ast_nodes: int) -> float:
        """Soft floor minimalism: concise code rewarded, but no golf pressure."""
        if ast_nodes <= self.MIN_AST_FLOOR:
            return 1.0
        if ast_nodes >= self.MIN_AST_CEILING:
            return 0.0
        return self._clamp(1.0 - (ast_nodes - self.MIN_AST_FLOOR) /
                           (self.MIN_AST_CEILING - self.MIN_AST_FLOOR))

    def _score_efficiency(self, exec_time_ms: float) -> float:
        """Soft floor efficiency: fast code rewarded, but no micro-opt pressure."""
        if exec_time_ms <= self.EFF_FLOOR_MS:
            return 1.0
        if exec_time_ms >= self.EFF_CEILING_MS:
            return 0.0
        return self._clamp(1.0 - (exec_time_ms - self.EFF_FLOOR_MS) /
                           (self.EFF_CEILING_MS - self.EFF_FLOOR_MS))

    def _score_diversity(self, max_sim: float) -> float:
        """Diversity = 1 - max_similarity. New approaches get full score."""
        return self._clamp(1.0 - max_sim)

    @staticmethod
    def _score_consistency(k_samples: int, n_agreeing: int) -> float:
        """Self-consistency: fraction of K samples agreeing with majority."""
        if k_samples <= 1:
            return 0.5  # neutral when no multi-sampling
        return n_agreeing / k_samples

    @staticmethod
    def _score_confidence(mean_logprob: float) -> float:
        """Convert mean logprob to [0, 1] confidence score."""
        # logprob is negative; exp(logprob) = probability
        prob = math.exp(mean_logprob) if mean_logprob < 0 else 1.0
        return GoalScorer._clamp(prob)

    def score(self, code: str,
              correct: bool,
              exec_time_ms: float,
              mean_logprob: float,
              goal_id: str,
              tokens_generated: int = 0,
              k_samples: int = 1,
              n_agreeing: int = 1,
              stress_exec_ms: float | None = None) -> ScoreResult:
        """Score a solution attempt.

        Args:
            code: the generated Python code
            correct: whether all I/O test cases passed (GATE)
            exec_time_ms: execution time (use stress test time if available)
            mean_logprob: model's mean log probability of generated tokens
            goal_id: goal identifier for diversity tracking
            tokens_generated: number of tokens generated
            k_samples: number of samples for self-consistency
            n_agreeing: how many of k_samples agreed on output
            stress_exec_ms: exec time on stress test specifically (for efficiency)

        Returns:
            ScoreResult with composite quality and per-dimension scores
        """
        fingerprint = extract_ast_fingerprint(code)
        ast_nodes = sum(fingerprint.values())
        eff_time = stress_exec_ms if stress_exec_ms is not None else exec_time_ms

        # Diversity: compare to seen solutions for this goal
        seen = self.get_seen(goal_id)
        max_sim = max_similarity_to_seen(fingerprint, seen)

        # Per-dimension scores
        s_min = self._score_minimalism(ast_nodes)
        s_eff = self._score_efficiency(eff_time)
        s_div = self._score_diversity(max_sim)
        s_con = self._score_consistency(k_samples, n_agreeing)
        s_conf = self._score_confidence(mean_logprob)

        scores = {
            "minimalism": round(s_min, 4),
            "efficiency": round(s_eff, 4),
            "diversity": round(s_div, 4),
            "consistency": round(s_con, 4),
            "confidence": round(s_conf, 4),
        }

        # Correctness GATE
        if not correct:
            return ScoreResult(
                quality=0.0, correct=False, accepted=False,
                rejected_reason="failed I/O verification",
                scores=scores, fingerprint=fingerprint,
                ast_node_count=ast_nodes, exec_time_ms=eff_time,
                max_similarity=max_sim, tokens_generated=tokens_generated,
            )

        # Anti-redundancy check (RLVR 2026)
        if max_sim > self.REDUNDANCY_THRESHOLD and seen:
            return ScoreResult(
                quality=0.0, correct=True, accepted=False,
                rejected_reason=f"redundant (Dice sim {max_sim:.2f} > {self.REDUNDANCY_THRESHOLD})",
                scores=scores, fingerprint=fingerprint,
                ast_node_count=ast_nodes, exec_time_ms=eff_time,
                max_similarity=max_sim, tokens_generated=tokens_generated,
            )

        # Composite quality (among correct, non-redundant solutions)
        w = self.WEIGHTS
        quality = (w["minimalism"] * s_min +
                   w["efficiency"] * s_eff +
                   w["diversity"] * s_div +
                   w["consistency"] * s_con +
                   w["confidence"] * s_conf)

        return ScoreResult(
            quality=round(quality, 4), correct=True, accepted=True,
            scores=scores, fingerprint=fingerprint,
            ast_node_count=ast_nodes, exec_time_ms=eff_time,
            max_similarity=max_sim, tokens_generated=tokens_generated,
        )
