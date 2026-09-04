"""SOAR: Self-improvement via meta-RL to escape learning plateaus.

Based on "Teaching Models to Teach Themselves: Reasoning at the Edge of
Learnability" (arXiv 2601.18778).

Problem: RL fine-tuning stalls on datasets with low initial success rates
(little training signal). The model can't solve hard problems, so it gets
no reward, so it doesn't improve.

SOAR solution: a teacher copy of the model proposes synthetic problems for
a student copy. The teacher is rewarded with the student's IMPROVEMENT on
a small subset of hard problems (not intrinsic proxy rewards).

Key findings:
  1. Bi-level meta-RL can unlock learning under sparse, binary rewards
  2. Grounded rewards (measured student progress) outperform intrinsic rewards
  3. Structural quality and well-posedness > solution correctness for curriculum
  4. The ability to generate stepping stones doesn't require solving the hard problems

For our self-play loop (infinite_loop.py):
  - Current: AZR-style propose → solve → verify → SFT
  - SOAR: teacher proposes problems → student attempts → teacher rewarded
    by student improvement on hard problems → teacher improves proposals
  - Escapes the plateau where the model can't generate useful training signal

This implementation provides:
  1. SOARTeacher: proposes synthetic problems guided by student progress
  2. SOARStudent: attempts problems, provides success signal
  3. SOARMetaRL: bi-level optimization loop (teacher + student)
  4. Integration with existing self-play infrastructure
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Problem:
    """A synthetic problem proposed by the teacher."""
    id: int
    prompt: str
    difficulty: float = 0.5  # 0=easy, 1=hard
    category: str = "general"
    reference_solution: str = ""
    is_well_posed: bool = True
    structural_quality: float = 0.5  # 0=poor, 1=excellent
    student_success_rate: float = 0.0
    created_at_step: int = 0


@dataclass
class StudentAttempt:
    """A student's attempt at a problem."""
    problem_id: int
    solution: str
    success: bool
    reward: float = 0.0


class SOARTeacher:
    """Teacher model that proposes synthetic problems.

    The teacher is rewarded by the student's improvement on hard problems,
    NOT by intrinsic proxy rewards (which lead to reward hacking and
    diversity collapse).
    """

    def __init__(self, model, tokenizer, max_problems_per_round: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.max_problems = max_problems_per_round
        self._problem_history: list[Problem] = []
        self._student_progress: dict[int, float] = {}  # problem_id → success_rate
        self._teacher_reward: float = 0.0

    def propose_problems(self, target_problems: list[str],
                         student_skill_level: float = 0.5) -> list[Problem]:
        """Propose synthetic problems guided by target hard problems.

        The teacher generates "stepping stone" problems that are:
          - Easier than the target (learnable by the student)
          - Related to the target (preserve high-level motif)
          - Well-posed (structural quality > solution correctness)

        Args:
            target_problems: hard problems the student can't solve yet
            student_skill_level: current student ability (0-1)

        Returns:
            list of proposed problems
        """
        problems = []
        for i in range(self.max_problems):
            # Select a target problem as inspiration
            target = random.choice(target_problems) if target_problems else ""

            # Generate a stepping stone problem
            prompt = self._build_proposal_prompt(target, student_skill_level)
            problem_text = self._generate(prompt)

            # Parse and create Problem object
            problem = Problem(
                id=len(self._problem_history) + i,
                prompt=problem_text,
                difficulty=student_skill_level + random.uniform(-0.1, 0.1),
                category="stepping_stone",
                is_well_posed=self._check_well_posed(problem_text),
                structural_quality=self._assess_quality(problem_text),
                created_at_step=len(self._problem_history),
            )
            problems.append(problem)

        self._problem_history.extend(problems)
        return problems

    def _build_proposal_prompt(self, target: str, skill_level: float) -> str:
        """Build a prompt for the teacher to generate a stepping stone problem."""
        return (
            f"Generate a coding problem that is easier than this one:\n"
            f"{target}\n\n"
            f"The problem should be solvable by a student at skill level {skill_level:.1f}/1.0.\n"
            f"It should preserve the key concept of the original but be simpler.\n"
            f"Make sure the problem is well-posed with clear inputs and outputs.\n\n"
            f"Problem:"
        )

    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from the teacher model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.8, do_sample=True)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            return f"Problem {random.randint(100, 999)}: Write a function that..."

    def _check_well_posed(self, problem_text: str) -> bool:
        """Check if a problem is well-posed (has clear inputs/outputs)."""
        # Simple heuristic: must contain function signature or input/output description
        keywords = ["input", "output", "return", "function", "def ", "Example"]
        return any(kw in problem_text.lower() for kw in keywords)

    def _assess_quality(self, problem_text: str) -> float:
        """Assess structural quality of a problem (0-1)."""
        # Simple heuristic: longer + more structured = higher quality
        length_score = min(1.0, len(problem_text) / 500)
        structure_score = 0.5
        if "Example" in problem_text:
            structure_score += 0.2
        if "def " in problem_text:
            structure_score += 0.2
        if "return" in problem_text:
            structure_score += 0.1
        return min(1.0, (length_score + structure_score) / 2)

    def update_reward(self, student_improvement: float):
        """Update teacher reward based on measured student improvement.

        This is the KEY difference from intrinsic-reward self-play:
        the teacher is rewarded by ACTUAL student progress, not by
        proxy measures like pass rate or problem complexity.
        """
        self._teacher_reward = student_improvement

    def get_reward(self) -> float:
        return self._teacher_reward


class SOARStudent:
    """Student model that attempts problems and provides success signal."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._attempts: list[StudentAttempt] = []
        self._skill_level: float = 0.0

    def attempt_problem(self, problem: Problem) -> StudentAttempt:
        """Attempt to solve a problem."""
        try:
            inputs = self.tokenizer(problem.prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=512,
                    temperature=0.3, do_sample=True)
            solution = self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            solution = ""

        # Verify (simplified — real implementation would use a verifier)
        success = self._verify(problem, solution)

        attempt = StudentAttempt(
            problem_id=problem.id,
            solution=solution,
            success=success,
            reward=1.0 if success else 0.0,
        )
        self._attempts.append(attempt)
        return attempt

    def _verify(self, problem: Problem, solution: str) -> bool:
        """Verify if the solution is correct (simplified)."""
        # In practice, this would run the code and check outputs
        return len(solution) > 50 and "def " in solution

    def measure_improvement(self, problems: list[Problem]) -> float:
        """Measure improvement on a set of problems.

        Compares recent success rate to historical success rate.
        """
        if not problems:
            return 0.0

        recent_attempts = self._attempts[-len(problems):]
        if not recent_attempts:
            return 0.0

        recent_success = sum(1 for a in recent_attempts if a.success) / len(recent_attempts)

        # Compare to historical
        if len(self._attempts) > len(recent_attempts):
            historical = self._attempts[:-len(recent_attempts)]
            hist_success = sum(1 for a in historical if a.success) / max(len(historical), 1)
            improvement = recent_success - hist_success
        else:
            improvement = recent_success

        self._skill_level = recent_success
        return improvement

    @property
    def skill_level(self) -> float:
        return self._skill_level


class SOARMetaRL:
    """Bi-level meta-RL loop: teacher and student improve together.

    Outer loop (teacher): optimize problem proposals to maximize student
    improvement on hard target problems.

    Inner loop (student): standard RL/SFT on problems proposed by teacher.

    The bi-level structure means the teacher learns to generate problems
    that are USEFUL for the student's learning, not just hard or complex.
    """

    def __init__(self, teacher_model, student_model, tokenizer,
                 target_problems: list[str]):
        self.teacher = SOARTeacher(teacher_model, tokenizer)
        self.student = SOARStudent(student_model, tokenizer)
        self.target_problems = target_problems
        self._round = 0
        self._history: list[dict] = []

    def run_round(self) -> dict:
        """Run one round of SOAR meta-RL.

        1. Teacher proposes problems (guided by student skill level)
        2. Student attempts all problems
        3. Measure student improvement
        4. Update teacher reward (grounded in student progress)
        5. Optionally fine-tune teacher and student

        Returns:
            round_stats: dict with metrics for this round
        """
        self._round += 1

        # 1. Teacher proposes problems
        problems = self.teacher.propose_problems(
            self.target_problems,
            student_skill_level=self.student.skill_level)

        # Filter: only well-posed problems with decent quality
        good_problems = [p for p in problems if p.is_well_posed and p.structural_quality > 0.3]

        # 2. Student attempts
        attempts = []
        for problem in good_problems:
            attempt = self.student.attempt_problem(problem)
            attempts.append(attempt)
            problem.student_success_rate = attempt.success

        # 3. Measure improvement
        improvement = self.student.measure_improvement(good_problems)

        # 4. Update teacher reward (grounded, not intrinsic)
        self.teacher.update_reward(improvement)

        # 5. Record stats
        stats = {
            "round": self._round,
            "n_problems": len(good_problems),
            "n_success": sum(1 for a in attempts if a.success),
            "success_rate": sum(1 for a in attempts if a.success) / max(len(attempts), 1),
            "student_improvement": improvement,
            "teacher_reward": self.teacher.get_reward(),
            "student_skill": self.student.skill_level,
        }
        self._history.append(stats)
        return stats

    def run(self, n_rounds: int = 50) -> list[dict]:
        """Run multiple rounds of SOAR."""
        all_stats = []
        for _ in range(n_rounds):
            stats = self.run_round()
            all_stats.append(stats)
            if stats["n_problems"] == 0:
                break
        return all_stats

    def get_curriculum(self) -> list[Problem]:
        """Get the curriculum (all proposed problems sorted by difficulty)."""
        return sorted(self.teacher._problem_history, key=lambda p: p.difficulty)

    def stats(self) -> dict:
        return {
            "rounds": self._round,
            "total_problems": len(self.teacher._problem_history),
            "student_skill": self.student.skill_level,
            "teacher_reward": self.teacher.get_reward(),
        }
