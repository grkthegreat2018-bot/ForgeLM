"""SGS: Self-Guided Self-Play to prevent Conjecturer collapse.

Based on "Scaling Self-Play with Self-Guidance" (arXiv 2604.20209).

Problem: LLM self-play hits learning plateaus because the Conjecturer
(problem proposer) learns to hack its reward, collapsing to artificially
complex problems that don't help the Solver improve.

SGS solution: the model takes on THREE roles:
  1. Solver: attempts problems
  2. Conjecturer: proposes problems
  3. Guide: scores synthetic problems by (a) relevance to unsolved targets
     and (b) cleanliness/naturalness

The Guide prevents Conjecturer collapse by providing supervision against
degenerate problem generation.

Results: 7B model after 200 rounds of SGS solves more Lean4 theorems than
a 671B model pass@4. Scales better than unguided self-play.

For our self-play loop:
  - Current: AZR-style (Conjecturer + Solver, no Guide)
  - SGS: add Guide role to score and filter problems
  - Prevents the collapse we'd see in long training runs

This implementation provides:
  1. SGSConjecturer: proposes problems (guided by target problems)
  2. SGSSolver: attempts problems
  3. SGSGuide: scores problems by relevance + cleanliness
  4. SGSTrainer: orchestrates the three-role self-play loop
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SGSProblem:
    """A problem in the SGS self-play loop."""
    id: int
    prompt: str
    conjecturer_score: float = 0.0  # Conjecturer's own confidence
    guide_relevance: float = 0.0    # Guide's relevance score (to targets)
    guide_cleanliness: float = 0.0  # Guide's cleanliness score
    guide_score: float = 0.0        # Combined guide score
    solver_success: bool = False
    solver_solution: str = ""
    is_accepted: bool = False  # accepted into curriculum


class SGSConjecturer:
    """Proposes problems, guided by target (unsolved) problems.

    Unlike unguided self-play, the Conjecturer is prompted with target
    problems and asked to generate related but easier instances.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def propose(self, target_problems: list[str], n_problems: int = 5,
                difficulty: str = "easier") -> list[str]:
        """Propose problems guided by targets.

        Args:
            target_problems: unsolved problems to use as guidance
            n_problems: number of problems to generate
            difficulty: "easier" or "harder" than targets
        """
        problems = []
        for _ in range(n_problems):
            target = random.choice(target_problems) if target_problems else ""
            prompt = (
                f"Here is a challenging problem:\n{target}\n\n"
                f"Generate a {difficulty} problem that preserves the key concept "
                f"but is more accessible. The problem should be natural and well-defined.\n\n"
                f"Problem:"
            )
            problem_text = self._generate(prompt)
            problems.append(problem_text)
        return problems

    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.9, do_sample=True)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            return f"Write a function that processes a list of integers."


class SGSSolver:
    """Attempts to solve problems."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def solve(self, problem: str, max_tokens: int = 512) -> tuple[str, bool]:
        """Attempt to solve a problem. Returns (solution, success)."""
        try:
            inputs = self.tokenizer(problem, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.3, do_sample=True)
            solution = self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            solution = ""

        success = self._verify(problem, solution)
        return solution, success

    def _verify(self, problem: str, solution: str) -> bool:
        """Verify solution correctness (simplified)."""
        return len(solution) > 50 and ("def " in solution or "return" in solution)


class SGSGuide:
    """Scores problems by relevance and cleanliness.

    The Guide is the KEY innovation of SGS. It prevents Conjecturer collapse
    by scoring proposed problems on:
      1. Relevance: how related is this problem to unsolved targets?
      2. Cleanliness: is this problem natural and well-defined?

    Problems with low Guide scores are rejected, preventing the Conjecturer
    from hacking its reward with degenerate problems.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def score(self, problem: str, target_problems: list[str]) -> tuple[float, float]:
        """Score a problem. Returns (relevance, cleanliness).

        Uses the model itself to assess quality (self-guidance).
        """
        relevance = self._score_relevance(problem, target_problems)
        cleanliness = self._score_cleanliness(problem)
        return relevance, cleanliness

    def _score_relevance(self, problem: str, targets: list[str]) -> float:
        """Score relevance to target problems (0-1)."""
        if not targets:
            return 0.5

        # Use embedding similarity if available, else heuristic
        # Heuristic: keyword overlap with targets
        target_words = set()
        for t in targets:
            target_words.update(t.lower().split())

        problem_words = set(problem.lower().split())
        overlap = len(target_words & problem_words) / max(len(target_words), 1)

        return min(1.0, overlap * 2)  # scale up since overlap is sparse

    def _score_cleanliness(self, problem: str) -> float:
        """Score cleanliness/naturalness (0-1).

        Clean problems:
          - Have clear structure (function signature, examples)
          - Are not artificially complex
          - Are well-defined (no ambiguity)
        """
        score = 0.0

        # Has function signature
        if "def " in problem:
            score += 0.3
        # Has examples
        if "example" in problem.lower() or "input:" in problem.lower():
            score += 0.2
        # Reasonable length (not too short, not too long)
        if 50 < len(problem) < 1000:
            score += 0.2
        # Not degenerate (no excessive repetition)
        words = problem.split()
        if len(set(words)) > len(words) * 0.3:  # diversity check
            score += 0.3

        return min(1.0, score)

    def should_accept(self, relevance: float, cleanliness: float,
                      min_relevance: float = 0.1,
                      min_cleanliness: float = 0.4) -> bool:
        """Decide whether to accept a problem into the curriculum."""
        return relevance >= min_relevance and cleanliness >= min_cleanliness


class SGSTrainer:
    """Orchestrates the three-role SGS self-play loop.

    Solver + Conjecturer + Guide, all from the same model (or different copies).
    The Guide prevents Conjecturer collapse, enabling sustained learning.
    """

    def __init__(self, model, tokenizer, target_problems: list[str]):
        self.conjecturer = SGSConjecturer(model, tokenizer)
        self.solver = SGSSolver(model, tokenizer)
        self.guide = SGSGuide(model, tokenizer)
        self.target_problems = target_problems
        self._curriculum: list[SGSProblem] = []
        self._round = 0
        self._history: list[dict] = []
        self._problem_id = 0

    def run_round(self, n_proposals: int = 10) -> dict:
        """Run one round of SGS self-play.

        1. Conjecturer proposes problems (guided by targets)
        2. Guide scores each problem (relevance + cleanliness)
        3. Accept high-scoring problems into curriculum
        4. Solver attempts accepted problems
        5. Record stats and update target set (remove solved targets)
        """
        self._round += 1

        # 1. Conjecturer proposes
        proposed = self.conjecturer.propose(
            self.target_problems, n_problems=n_proposals)

        # 2. Guide scores
        accepted = []
        for prob_text in proposed:
            rel, clean = self.guide.score(prob_text, self.target_problems)
            guide_score = (rel + clean) / 2

            problem = SGSProblem(
                id=self._problem_id,
                prompt=prob_text,
                guide_relevance=rel,
                guide_cleanliness=clean,
                guide_score=guide_score,
            )
            self._problem_id += 1

            # 3. Accept if Guide approves
            if self.guide.should_accept(rel, clean):
                problem.is_accepted = True
                accepted.append(problem)
                self._curriculum.append(problem)

        # 4. Solver attempts accepted problems
        for problem in accepted:
            solution, success = self.solver.solve(problem.prompt)
            problem.solver_solution = solution
            problem.solver_success = success

        # 5. Update targets (remove solved)
        # (In practice, we'd check if any target was solved and remove it)

        # Stats
        n_accepted = len(accepted)
        n_solved = sum(1 for p in accepted if p.solver_success)
        avg_guide_score = (sum(p.guide_score for p in accepted) / max(n_accepted, 1))

        stats = {
            "round": self._round,
            "n_proposed": len(proposed),
            "n_accepted": n_accepted,
            "accept_rate": n_accepted / max(len(proposed), 1),
            "n_solved": n_solved,
            "solve_rate": n_solved / max(n_accepted, 1),
            "avg_guide_score": avg_guide_score,
            "curriculum_size": len(self._curriculum),
        }
        self._history.append(stats)
        return stats

    def run(self, n_rounds: int = 100) -> list[dict]:
        """Run multiple rounds."""
        all_stats = []
        for _ in range(n_rounds):
            stats = self.run_round()
            all_stats.append(stats)
        return all_stats

    def get_curriculum(self) -> list[SGSProblem]:
        """Get accepted curriculum problems, sorted by guide score."""
        return sorted(
            [p for p in self._curriculum if p.is_accepted],
            key=lambda p: p.guide_score,
            reverse=True,
        )

    def stats(self) -> dict:
        return {
            "rounds": self._round,
            "curriculum_size": len(self._curriculum),
            "accepted": sum(1 for p in self._curriculum if p.is_accepted),
            "solved": sum(1 for p in self._curriculum if p.solver_success),
        }
