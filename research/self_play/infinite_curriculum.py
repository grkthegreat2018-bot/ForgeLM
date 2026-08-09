"""Infinite Curriculum Engine — AZR-style self-play task generation.

Implements the Absolute Zero Reasoner (AZR) paradigm:
  - The model PROPOSES coding tasks with unit tests
  - A Python executor VERIFIES (ground truth, not model judgment)
  - Goldilocks difficulty: proposer rewarded when solver succeeds 40-60%
  - Three reasoning modes: deduction, abduction, induction
  - Infinite, adaptive curriculum — never runs out of questions

Architecture:
  1. Proposer: model generates (task_description, function_signature, test_cases)
  2. Validator: Python executor checks that test_cases are self-consistent
     (reference solution passes all tests — filters garbage proposals)
  3. Solver: model solves the task (existing self-play infrastructure)
  4. Difficulty tracker: per-domain rolling success rate
     - >60% success → proposer must generate harder tasks
     - <40% success → proposer must generate easier tasks
  5. Task queue: validated tasks stored for replay and curriculum ordering

Research basis:
  - AZR (NeurIPS 2025): zero-data SOTA via propose-solve-verify loop
  - SQLM (2025): asymmetric self-play, Goldilocks reward
  - SPICE (Meta FAIR, 2025): corpus grounding prevents collapse
  - SOAR (2026): reward = measured improvement, not intrinsic proxy

Usage:
    from research.self_play.infinite_curriculum import InfiniteCurriculum
    curriculum = InfiniteCurriculum(model, tokenizer, device="cuda")
    tasks = curriculum.propose_tasks(domain="algorithms", n=10)
    for task in tasks:
        result = curriculum.solve_task(task)
        curriculum.record_result(task, result["success"])
"""
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from research.inference.async_d2h import AsyncTokenReader
from research.json_compat import dumps, loads

# Pre-compiled regex patterns (avoids 450+ recompilations per session).
_RE_PYTHON_BLOCK = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
_RE_TASK_DESC = re.compile(r'#\s*Task:\s*(.+?)(?:\n|$)')
_RE_QUOTED = re.compile(r'["\'](.+?)["\']')
_RE_SOLVE_SIG = re.compile(r'def\s+solve\s*(\(.*?\))')
_RE_SOLVE_TEST = re.compile(r'#\s*solve\s*\((.+?)\)\s*==\s*(.+?)(?:\n|$)')
_RE_ASSERT_TEST = re.compile(r'#?\s*assert\s+solve\s*\((.+?)\)\s*==\s*(.+?)(?:,\s*["\']|\n|$)')
_RE_SOLVE_IMPL = re.compile(r'(def\s+solve\s*\(.*?\):.*?)(?:\nassert|\n#|\n\ndef|\Z)', re.DOTALL)
_RE_RETURN = re.compile(r'return\s+(.+)')
_RE_CODEBLOCK_OPEN = re.compile(r'```python\s*\n?')
_RE_CODEBLOCK_CLOSE = re.compile(r'```\s*$', re.MULTILINE)

from research.evaluation.goal_tasks import GoalTask

# ─── Data structures ──────────────────────────────────────────────────

@dataclass
class ProposedTask:
    """A task proposed by the model, with self-generated test cases."""
    id: str
    domain: str                          # algorithms, math, strings, logic
    difficulty: str                      # easy, medium, hard (model-declared)
    description: str                     # what the function should do
    signature: str                       # e.g. "(n: int) -> int"
    solve_name: str = "solve"
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    # each: {"args": (...,), "expected": ...}
    stress_index: int | None = None
    proposer_confidence: float = 0.0     # model's own confidence
    validated: bool = False              # passed self-consistency check
    archetype: str = "model_proposed"    # marks this as model-generated
    raw_output: str = ""                 # full model output for debugging
    generation_mode: str = "induction"   # induction/abduction/deduction


@dataclass
class CurriculumStats:
    """Tracks curriculum health metrics."""
    total_proposed: int = 0
    total_validated: int = 0       # passed self-consistency
    total_solved: int = 0          # solver succeeded
    total_failed: int = 0          # solver failed
    # Per-domain rolling success rate (for Goldilocks difficulty)
    domain_results: dict[str, list[bool]] = field(default_factory=dict)

    @property
    def validation_rate(self) -> float:
        return self.total_validated / max(self.total_proposed, 1)

    @property
    def solve_rate(self) -> float:
        return self.total_solved / max(self.total_validated, 1)

    def domain_success_rate(self, domain: str) -> float:
        results = self.domain_results.get(domain, [])
        if not results:
            return 0.5  # unknown → assume medium
        return sum(results) / len(results)


# ─── Task proposal prompts ────────────────────────────────────────────

# Mode 1: Induction — model writes a function + test cases (code completion style)
# Push-limits: the prompt asks for NON-TRIVIAL tasks with branching, edge cases
INDUCTION_PROMPT = '''# Implement: {desc_hint}
# Write solve{signature} that solves this problem using CONDITIONAL LOGIC or LOOPS.
# Do NOT just call a builtin (sum, max, len, sorted, etc.) — implement the logic yourself.
# After the function, write 4+ test cases with edge cases (empty, negative, single element, large).

def solve{signature}:
'''

# Mode 2: Abduction — given output, write code that produces it
ABLATION_PROMPT = """# You are a task designer. Create a Python programming challenge.
# Mode: ABDUCTION — given a target output, the solver must write code that produces it.
#
# Rules:
# 1. The task must be solvable with a single function called `solve`.
# 2. Provide the function signature and 3-5 (input, output) pairs.
# 3. The function should transform the input to produce the output.
# 4. Focus on: {domain}
#
# Generate a {difficulty} {domain} task in the same format:"""

# Mode 3: Deduction — given code, predict output (simpler, for warmup)
DEDUCTION_PROMPT = """# You are a task designer. Create a Python trace challenge.
# Mode: DEDUCTION — given code and input, the solver must predict the output.
#
# Rules:
# 1. Provide a short Python function (5-15 lines).
# 2. Provide 3-5 (input, expected_output) pairs.
# 3. The function should be deterministic (no randomness, no I/O).
# 4. Focus on: {domain}
#
# Generate a {difficulty} {domain} task:"""

# Solver prompt — shown to the model that must solve the task
SOLVER_PROMPT = """# Goal: {description}
# Define a function `solve{signature}`.
# It must produce the correct output for ALL of these test inputs:
{test_lines}
# Implement solve any way you choose."""


# ─── Infinite Curriculum Engine ───────────────────────────────────────

class InfiniteCurriculum:
    """AZR-style infinite curriculum: propose → validate → solve → score.

    The model generates tasks with test cases. A Python executor validates
    that the proposer's own reference solution passes the tests (self-consistency).
    Validated tasks are then given to the solver. Success rate feeds back into
    difficulty adjustment (Goldilocks zone: 40-60% solver success).
    """

    # Domains and their difficulty progression
    DOMAINS = ["algorithms", "math", "strings", "logic", "data_structures"]
    DIFFICULTIES = ["easy", "medium", "hard"]

    # Push-limits mode: start at hard, never go below medium.
    # The Goldilocks zone is LOWER than standard AZR — we WANT the model to
    # struggle (10-40% success). If it's solving everything, the tasks are too easy.
    # Standard AZR targets 50% success. We target 20% — the model should FAIL
    # most tasks, learning from the ones it does solve.
    GOLDILOCKS_LOW = 0.1
    GOLDILOCKS_HIGH = 0.4
    TARGET_SUCCESS = 0.2  # ideal: model fails 80% of the time (pushing limits)
    MIN_DIFFICULTY = "medium"  # never propose easy tasks
    DEFAULT_DIFFICULTY = "hard"  # start at hard

    # Minimum complexity requirements — reject trivial tasks
    MIN_TEST_CASES = 4          # need at least 4 test cases
    MIN_SIGNATURE_ARGS = 1      # at least 1 argument
    BANNED_PATTERNS = {         # reject tasks whose solution is just a builtin
        "sum", "max", "min", "len", "sorted", "abs", "round",
        "int", "str", "list", "set", "dict", "tuple", "bool",
        "print", "type", "hash", "id", "ord", "chr",
    }

    # Rolling window for per-domain success tracking
    WINDOW_SIZE = 15

    def __init__(self, model, tokenizer, device: str = "cuda",
                 max_gen_tokens: int = 200,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 task_queue_dir: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        if task_queue_dir is None:
            from research.paths import CURRICULUM_DIR
            task_queue_dir = CURRICULUM_DIR
        self.task_queue_dir = Path(task_queue_dir)
        self.task_queue_dir.mkdir(parents=True, exist_ok=True)

        self.stats = CurriculumStats()
        self._task_counter = 0
        self.rng = random.Random(42)

        # Task queue: validated tasks ready for training
        self._task_queue: list[ProposedTask] = []
        # Failed/rejected tasks (for diversity — don't re-propose similar)
        self._rejected_signatures: set = set()

        # Forward cache for repeated prompt prefixes (C2 — 20-40% fewer forward passes)
        from research.runtime.forward_cache import ForwardCache
        self.fwd_cache = ForwardCache(max_entries=200, device=device)

        # Self-modeling: confidence-based retry policy (D5)
        from research.runtime.self_model import ConfidenceScorer, RetryPolicy
        self.confidence_scorer = ConfidenceScorer()
        self.retry_policy = RetryPolicy(
            min_confidence=0.4,  # low bar — we're pushing limits, low confidence is normal
            max_retries=1,       # one retry on very low confidence
            temp_increase=0.2,
            base_temp=temperature,
        )

    # ─── Task Proposal ─────────────────────────────────────────────

    def propose_tasks(self, domain: str = "algorithms",
                      n: int = 5,
                      difficulty: str | None = None,
                      mode: str = "induction",
                      batch_size: int = 8) -> list[ProposedTask]:
        """Generate n tasks via the model, validate them, return valid ones.

        Uses batched generation: multiple prompts are generated in parallel
        in a single forward pass (batch_size at a time). This gives
        batch_size× speedup on GPU since batch=1 underutilizes compute.

        Args:
            domain: target domain (algorithms, math, strings, etc.)
            n: number of tasks to attempt (may return fewer if validation fails)
            difficulty: override adaptive difficulty (None = auto)
            mode: induction / abduction / deduction
            batch_size: number of prompts to generate in parallel (8 = good default)
        """
        if difficulty is None:
            difficulty = self._adaptive_difficulty(domain)

        valid_tasks = []
        proposed = 0

        while proposed < n:
            # Generate a batch of prompts
            current_batch = min(batch_size, n - proposed)
            prompts = []
            for _ in range(current_batch):
                sig_hint = self._random_signature(domain)
                t_args, t_expected = self._example_test_cases(domain, sig_hint)
                desc_hints = {
                    "algorithms": [
                        "Find the k-th smallest element using partitioning (not sorted())",
                        "Detect if a list contains a cycle using Floyd's algorithm",
                        "Merge two sorted lists into one sorted list (not using sorted())",
                        "Find the longest increasing subsequence length",
                        "Compute the running median of a stream",
                        "Rotate a matrix 90 degrees in-place",
                        "Find all pairs that sum to a target (no nested loops if possible)",
                        "Implement binary search on a rotated sorted array",
                        "Count inversions in a list using merge sort",
                        "Find the majority element (appears > n/2 times) in O(n) space",
                    ],
                    "math": [
                        "Compute the n-th prime number using the Sieve of Eratosthenes",
                        "Find all prime factors of n (not just check if prime)",
                        "Compute the digital root of n (recursive sum until single digit)",
                        "Check if n is a perfect number (sum of proper divisors equals n)",
                        "Compute the n-th Catalan number",
                        "Find the GCD of a list of numbers (not just two)",
                        "Compute the Euler totient function phi(n)",
                        "Check if n is a power of 2 using bit manipulation",
                        "Compute the continued fraction representation of sqrt(n)",
                        "Find the modular inverse of a mod m using extended Euclidean",
                    ],
                    "strings": [
                        "Find the longest palindromic substring (not just check palindrome)",
                        "Compress a string using run-length encoding",
                        "Check if two strings are anagrams without sorting",
                        "Find the first non-repeating character in a string",
                        "Implement string matching using KMP algorithm",
                        "Convert a string to title case handling edge cases",
                        "Find the minimum window in s that contains all chars of t",
                        "Validate if a string is a valid IPv4 address",
                        "Count distinct palindromic substrings",
                    ],
                    "logic": [
                        "Evaluate a boolean expression string with AND/OR/NOT",
                        "Check if a number is a valid Luhn checksum",
                        "Determine if a Sudoku row/column is valid",
                        "Check if parentheses/brackets are balanced",
                        "Evaluate a simple arithmetic expression string (no eval)",
                    ],
                    "data_structures": [
                        "Implement a min-stack (push/pop/min in O(1))",
                        "Evaluate reverse polish notation",
                        "Implement a circular queue with wraparound",
                        "Flatten a nested list (arbitrary depth) iteratively",
                        "Detect if a binary tree is a valid BST",
                    ],
                }
                desc_hint = self.rng.choice(desc_hints.get(domain, desc_hints["algorithms"]))
                prompt = INDUCTION_PROMPT.format(
                    domain=domain, difficulty=difficulty,
                    signature=sig_hint, desc_hint=desc_hint,
                )
                prompts.append(prompt)

            # Batched generation — all prompts in one forward pass sequence
            completions = self._generate_batch(prompts)

            # Parse and validate each completion
            for j, completion in enumerate(completions):
                proposed += 1
                full_code = prompts[j] + completion
                task = self._parse_proposal(full_code, domain, difficulty, mode)
                if task is not None:
                    task.raw_output = full_code

                if task is None:
                    try:
                        print(f"  [Curriculum] Task {proposed}/{n}: PARSE FAILED")
                    except OSError:
                        pass
                    continue

                self.stats.total_proposed += 1
                safe_desc = task.description[:40].encode('ascii', 'replace').decode('ascii')
                try:
                    print(f"  [Curriculum] Task {proposed}/{n}: sig={task.signature}, "
                          f"tests={len(task.test_cases)}, desc={safe_desc}")
                except OSError:
                    pass

                # Complexity filter: reject trivial tasks (push-limits mode)
                if not self._is_complex_enough(task):
                    print("    -> REJECTED (too simple)")
                    continue

                # Validate: run proposer's reference solution against test cases
                if self._validate_task(task):
                    task.validated = True
                    self.stats.total_validated += 1
                    valid_tasks.append(task)
                    self._task_queue.append(task)
                    self._save_task(task)
                    print(f"    -> VALIDATED (conf={task.proposer_confidence})")
                else:
                    self._rejected_signatures.add(task.signature)
                    print("    -> REJECTED (validation failed)")

        return valid_tasks

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate completions for multiple prompts in a single batched pass.

        Uses left-padding with proper attention_mask and position_ids to ensure
        pad tokens don't corrupt generation. Each sequence generates independently,
        stopping on EOS.

        Args:
            prompts: list of prompt strings (1-16 typically)

        Returns:
            list of completion strings (same length as prompts)
        """
        if len(prompts) == 1:
            return [self._generate(prompts[0])]

        B = len(prompts)
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id or 0

        # Tokenize all prompts
        all_ids = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt",
                                 truncation=True, max_length=512)
            all_ids.append(enc.input_ids[0])  # (T_i,)

        # Left-pad to max prompt length (left pad so generation positions align)
        max_prompt_len = max(ids.shape[0] for ids in all_ids)
        padded = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=self.device)
        for i, ids in enumerate(all_ids):
            t = ids.shape[0]
            padded[i, max_prompt_len - t:] = ids.to(self.device)
            attention_mask[i, max_prompt_len - t:] = 1

        # Generate
        with torch.no_grad():
            # Prefill: process all prompts at once with attention_mask
            logits, _, past_kvs = self.model(
                padded, past_key_values=None, use_cache=True, attention_mask=attention_mask)
            next_logits = logits[:, -1]  # (B, vocab)
            next_tokens = self._sample_batch(next_logits)  # (B,)

            # Async D2H: issue non-blocking copy for all B tokens.
            reader = AsyncTokenReader(B, next_logits.shape[-1], self.device)
            reader.issue(next_tokens, next_logits)

            gen_ids = [[] for _ in range(B)]
            # Sync for first token.
            token_list = reader.get_tokens()
            finished = [False] * B
            for b in range(B):
                gen_ids[b].append(token_list[b])
                if token_list[b] == eos_id:
                    finished[b] = True

            # Decode loop: all sequences continue in parallel
            # Pre-allocate mask buffer to avoid torch.cat per step (CPU spike fix).
            max_total = max_prompt_len + self.max_gen_tokens
            mask_buf = torch.zeros(B, max_total, dtype=torch.long, device=self.device)
            mask_buf[:, :max_prompt_len] = attention_mask
            step = 0
            while step < self.max_gen_tokens and not all(finished):
                cur = next_tokens.unsqueeze(-1)  # (B, 1)
                mask_buf[:, max_prompt_len + step] = 1
                # Launch forward (GPU starts working).
                logits, _, past_kvs = self.model(
                    cur, past_key_values=past_kvs, use_cache=True,
                    attention_mask=mask_buf[:, :max_prompt_len + step + 1])
                next_logits = logits[:, -1]  # (B, vocab)
                next_tokens = self._sample_batch(next_logits)  # (B,)

                # Async D2H: non-blocking copy of tokens.
                reader.issue(next_tokens, next_logits)

                # Sync to get tokens (GPU finished copy by now — near-free).
                token_list = reader.get_tokens()
                for b in range(B):
                    if not finished[b]:
                        gen_ids[b].append(token_list[b])
                        if token_list[b] == eos_id:
                            finished[b] = True

                step += 1

            del past_kvs

        # Decode each sequence
        completions = []
        for b in range(B):
            text = self.tokenizer.decode(
                torch.tensor(gen_ids[b]), skip_special_tokens=True)
            completions.append(text)
        return completions

    def _sample_batch(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next tokens for a batch. logits: (B, vocab) -> (B,)."""
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[-1]:
            topk_vals, _ = torch.topk(probs, self.top_k, dim=-1)
            mask = probs < topk_vals[:, -1:]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if 0.0 < self.top_p < 1.0:
            # Per-row top-p filtering
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            keep = cumsum <= self.top_p
            keep[..., 0] = True  # always keep at least one
            probs = torch.zeros_like(probs, dtype=probs.dtype)
            probs.scatter_(-1, sorted_idx, (sorted_probs * keep.to(probs.dtype)))
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _propose_one(self, domain: str, difficulty: str,
                     mode: str = "induction") -> ProposedTask | None:
        """Generate one task via the model and parse it."""
        # Pick a random signature pattern to guide the model
        sig_hint = self._random_signature(domain)

        # Generate concrete test case examples for the prompt template
        # These are EXAMPLES — the model will replace them with its own
        t_args, t_expected = self._example_test_cases(domain, sig_hint)

        # Description hint — challenging, non-trivial tasks that push the model
        desc_hints = {
            "algorithms": [
                "Find the k-th smallest element using partitioning (not sorted())",
                "Detect if a list contains a cycle using Floyd's algorithm",
                "Merge two sorted lists into one sorted list (not using sorted())",
                "Find the longest increasing subsequence length",
                "Compute the running median of a stream",
                "Rotate a matrix 90 degrees in-place",
                "Find all pairs that sum to a target (no nested loops if possible)",
                "Implement binary search on a rotated sorted array",
                "Count inversions in a list using merge sort",
                "Find the majority element (appears > n/2 times) in O(n) space",
            ],
            "math": [
                "Compute the n-th prime number using the Sieve of Eratosthenes",
                "Find all prime factors of n (not just check if prime)",
                "Compute the digital root of n (recursive sum until single digit)",
                "Check if n is a perfect number (sum of proper divisors equals n)",
                "Compute the n-th Catalan number",
                "Find the GCD of a list of numbers (not just two)",
                "Compute the Euler totient function phi(n)",
                "Check if n is a power of 2 using bit manipulation",
                "Compute the continued fraction representation of sqrt(n)",
                "Find the modular inverse of a mod m using extended Euclidean",
            ],
            "strings": [
                "Find the longest palindromic substring (not just check palindrome)",
                "Compress a string using run-length encoding",
                "Check if two strings are anagrams without sorting",
                "Find the first non-repeating character in a string",
                "Implement string matching using KMP algorithm",
                "Convert a string to title case handling edge cases",
                "Find the minimum window in s that contains all chars of t",
                "Validate if a string is a valid IPv4 address",
                "Count distinct palindromic substrings",
                "Implement regex for basic pattern matching (., *, +)",
            ],
            "logic": [
                "Evaluate a boolean expression string with AND/OR/NOT",
                "Check if a number is a valid Luhn checksum",
                "Determine if a Sudoku row/column is valid",
                "Check if parentheses/brackets are balanced",
                "Evaluate a simple arithmetic expression string (no eval)",
            ],
            "data_structures": [
                "Implement a min-stack (push/pop/min in O(1))",
                "Evaluate reverse polish notation",
                "Implement a circular queue with wraparound",
                "Flatten a nested list (arbitrary depth) iteratively",
                "Detect if a binary tree is a valid BST",
            ],
        }
        desc_hint = self.rng.choice(desc_hints.get(domain, desc_hints["algorithms"]))

        prompt = INDUCTION_PROMPT.format(
            domain=domain,
            difficulty=difficulty,
            signature=sig_hint,
            desc_hint=desc_hint,
        )

        # Generate — the model completes the function body
        completion = self._generate(prompt)

        # Full code = prompt + completion (the model sees the stub and completes it)
        # The prompt already has the function stub and example test cases.
        # The model's completion will have the implementation + possibly more tests.
        full_code = prompt + completion

        # Parse the full code into a ProposedTask
        task = self._parse_proposal(full_code, domain, difficulty, mode)
        if task:
            task.raw_output = full_code  # store full code for validation
        return task

    def _random_signature(self, domain: str) -> str:
        """Return a random function signature hint for the domain."""
        sigs = {
            "algorithms": ["(n: int) -> int", "(lst: list) -> int",
                          "(lst: list) -> list", "(n: int) -> list",
                          "(a: int, b: int) -> int"],
            "math": ["(n: int) -> int", "(x: int, y: int) -> int",
                    "(n: int) -> bool", "(n: int) -> float"],
            "strings": ["(s: str) -> str", "(s: str) -> int",
                       "(s: str, t: str) -> str", "(s: str) -> bool"],
            "logic": ["(n: int) -> bool", "(a: int, b: int) -> bool",
                     "(lst: list) -> bool"],
            "data_structures": ["(lst: list) -> list", "(lst: list, n: int) -> list",
                               "(lst: list) -> int", "(matrix: list) -> list"],
        }
        return self.rng.choice(sigs.get(domain, sigs["algorithms"]))

    def _example_test_cases(self, domain: str, signature: str) -> tuple[list[str], list[str]]:
        """Generate example test case strings for the prompt template.

        These are PLACEHOLDER examples that show the format — the model
        is expected to replace them with its own test cases.
        """
        # Parse the signature to know how many args
        # Simple heuristic based on domain
        if "lst" in signature:
            args = ["[1, 2, 3]", "[5]", "[]", "[3, 1, 4, 1, 5]"]
            expected = ["6", "5", "0", "14"]
        elif "s" in signature and "str" in signature:
            args = ["'hello'", "''", "'test'", "'abcabc'"]
            expected = ["5", "0", "4", "6"]
        elif "a" in signature and "b" in signature:
            args = ["3, 5", "0, 0", "10, 20", "7, 3"]
            expected = ["8", "0", "30", "10"]
        elif "n" in signature and "bool" in signature:
            args = ["7", "1", "10", "13"]
            expected = ["True", "False", "False", "True"]
        else:
            args = ["5", "0", "10", "20"]
            expected = ["25", "0", "100", "400"]
        return args, expected

    def _generate(self, prompt: str) -> str:
        """Generate text from the model using KV cache (O(n) not O(n²)).

        Uses async D2H (pinned memory + non-blocking copy) to eliminate
        CPU spikes from .item() host-device synchronization.
        """
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=512)
        input_ids = enc.input_ids.to(self.device)

        with torch.no_grad():
            past_kvs = None
            gen_ids = []
            eos_id = self.tokenizer.eos_token_id

            # Initial pass on full prompt
            logits, _, past_kvs = self.model(
                input_ids, past_key_values=past_kvs, use_cache=True)
            next_logits = logits[0, -1]
            next_token_gpu = self._sample_gpu(next_logits)

            # Async D2H: issue non-blocking copy instead of blocking .item().
            reader = AsyncTokenReader(1, next_logits.shape[-1], self.device)
            reader.issue(next_token_gpu.unsqueeze(0), next_logits.unsqueeze(0))

            # Pre-allocate single-token buffer.
            cur_token = torch.zeros(1, 1, dtype=input_ids.dtype, device=self.device)

            # Sync for first token.
            token_list = reader.get_tokens()
            next_token = token_list[0]
            gen_ids.append(next_token)

            # Subsequent tokens: feed only new token, reuse KV cache.
            # Async D2H pattern: issue copy, launch forward, sync for previous result.
            while len(gen_ids) < self.max_gen_tokens:
                if next_token == eos_id:
                    break
                cur_token[0, 0] = next_token_gpu
                # Launch forward (GPU starts working).
                logits, _, past_kvs = self.model(
                    cur_token, past_key_values=past_kvs, use_cache=True)
                next_logits = logits[0, -1]
                next_token_gpu = self._sample_gpu(next_logits)

                # Issue async D2H (non-blocking).
                reader.issue(next_token_gpu.unsqueeze(0), next_logits.unsqueeze(0))

                # Sync to get token (GPU finished copy by now — near-free).
                token_list = reader.get_tokens()
                next_token = token_list[0]
                gen_ids.append(next_token)

            del past_kvs

        text = self.tokenizer.decode(
            torch.tensor(gen_ids), skip_special_tokens=True)
        return text

    def _sample_gpu(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token, keeping result on GPU (no .item() sync).

        Unlike _sample() which returns an int (forcing D2H sync), this returns
        a GPU tensor for use with async D2H transfer.
        """
        if self.temperature <= 0.0:
            return logits.argmax()

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        if 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = sorted_probs.cumsum(0)
            cutoff = (cumsum > self.top_p).float().cumsum(0)
            sorted_probs = sorted_probs * (cutoff == 0).float()
            sorted_probs = sorted_probs / sorted_probs.sum()
            probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)

        return torch.multinomial(probs, 1)

    def _sample(self, logits: torch.Tensor) -> int:
        """Sample next token with temperature/top-k/top-p."""
        if self.temperature <= 0.0:
            return logits.argmax().item()

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        if 0.0 < self.top_p < 1.0:
            from research.sampling_utils import top_p_sample_probs
            probs = top_p_sample_probs(probs, self.top_p)

        return torch.multinomial(probs, num_samples=1).squeeze().item()

    def _parse_proposal(self, raw: str, domain: str, difficulty: str,
                        mode: str) -> ProposedTask | None:
        """Parse model output into a ProposedTask.

        Expected format:
          # Task: <description>
          def solve(...):
              ...
          # solve(args1) == expected1
          # solve(args2) == expected2
        """
        # Extract code block if wrapped in ```python ... ```
        code_match = _RE_PYTHON_BLOCK.search(raw)
        code = code_match.group(1) if code_match else raw

        # Extract task description — try "# Task:" or quoted description
        desc_match = _RE_TASK_DESC.search(code)
        if not desc_match:
            # Try quoted: "Given a list..." or 'Given a list...'
            desc_match = _RE_QUOTED.search(code)
        description = desc_match.group(1).strip() if desc_match else "Model-proposed task"

        # Extract function signature — if no def solve found, infer from test cases
        sig_match = _RE_SOLVE_SIG.search(code)
        if sig_match:
            signature = sig_match.group(1)
        else:
            # Infer signature from first test case args
            # Will be set after test cases are parsed
            signature = None

        # Extract test cases — handle multiple formats:
        #   Format 1: # solve(args) == expected
        #   Format 2: assert solve(args) == expected
        #   Format 3: # assert solve(args) == expected  (commented assert)
        test_cases = []
        # Pattern 1: comment-style # solve(args) == expected
        for m in _RE_SOLVE_TEST.finditer(code):
            args_str = m.group(1).strip()
            expected_str = m.group(2).strip().rstrip(',')
            try:
                args = ast.literal_eval(f"({args_str},)")
                expected = ast.literal_eval(expected_str)
                test_cases.append({"args": args, "expected": expected})
            except Exception:
                continue
        # Pattern 2: assert-style (active or commented)
        if not test_cases:
            for m in _RE_ASSERT_TEST.finditer(code):
                args_str = m.group(1).strip()
                expected_str = m.group(2).strip().rstrip(',')
                try:
                    args = ast.literal_eval(f"({args_str},)")
                    expected = ast.literal_eval(expected_str)
                    test_cases.append({"args": args, "expected": expected})
                except Exception:
                    continue

        if len(test_cases) < 2:
            return None  # need at least 2 test cases

        # Infer signature from test case args if not found above
        if signature is None:
            first_args = test_cases[0]["args"]
            type_hints = []
            for a in first_args:
                if isinstance(a, int):
                    type_hints.append("int")
                elif isinstance(a, str):
                    type_hints.append("str")
                elif isinstance(a, list):
                    type_hints.append("list")
                elif isinstance(a, float):
                    type_hints.append("float")
                elif isinstance(a, bool):
                    type_hints.append("bool")
                else:
                    type_hints.append("any")
            params = ", ".join(f"arg{i}: {t}" for i, t in enumerate(type_hints))
            # Infer return type from first expected
            ret = test_cases[0]["expected"]
            if isinstance(ret, int):
                ret_type = "int"
            elif isinstance(ret, str):
                ret_type = "str"
            elif isinstance(ret, list):
                ret_type = "list"
            elif isinstance(ret, bool):
                ret_type = "bool"
            else:
                ret_type = "any"
            signature = f"({params}) -> {ret_type}"

        # Extract the reference implementation (for validation)
        # Match from `def solve(...)` to the next assert/#/def or end
        impl_match = _RE_SOLVE_IMPL.search(code)
        ref_code = impl_match.group(1) if impl_match else ""

        self._task_counter += 1
        task_id = f"azr_{domain}_{difficulty}_{self._task_counter}"

        return ProposedTask(
            id=task_id,
            domain=domain,
            difficulty=difficulty,
            description=description,
            signature=signature,
            solve_name="solve",
            test_cases=test_cases,
            stress_index=len(test_cases) - 1 if len(test_cases) > 2 else None,
            archetype="model_proposed",
            raw_output=raw,
            generation_mode=mode,
        )

    # ─── Validation ────────────────────────────────────────────────

    def _validate_task(self, task: ProposedTask) -> bool:
        """Validate that the task is well-formed.

        In push-limits mode, we accept tasks even when the proposer's solution
        has bugs — the solver will determine if the task is solvable.
        We only reject tasks with:
        - No test cases
        - Syntax errors in the reference code (can't even parse)
        - Test cases that crash (invalid Python)
        """
        if not task.test_cases:
            return False

        # Extract reference implementation from raw output
        ref_code = self._extract_reference_code(task.raw_output)
        if not ref_code:
            # No reference code — accept with lower confidence
            task.proposer_confidence = 0.3
            return True

        # Try to validate: run ref code + check test cases
        test_lines = []
        for i, tc in enumerate(task.test_cases):
            args_str = ", ".join(repr(a) for a in tc["args"])
            expected_repr = repr(tc["expected"])
            test_lines.append(
                f"    result_{i} = solve({args_str})\n"
                f"    assert result_{i} == {expected_repr}, "
                f"Test {i}: solve({args_str}) = {{result_{i}!r}}, expected {expected_repr}"
            )

        validation_code = ref_code + "\n" + "\n".join(test_lines) + "\nprint('VALID')"

        # Execute in sandbox
        result = self._execute_sandbox(validation_code, timeout_s=3.0)
        if result.get("returncode") == 0 and "VALID" in result.get("stdout", ""):
            task.proposer_confidence = 1.0  # validated — high confidence
            return True

        # Proposer's solution failed validation — but the test cases might still be valid.
        # Check if the reference code at least parses (no syntax errors).
        try:
            compile(ref_code, "<proposer>", "exec")
            # Code parses but fails tests — accept with lower confidence.
            # The solver will try to solve it; if it can, we train on it.
            task.proposer_confidence = 0.4
            return True
        except SyntaxError:
            # Syntax error — the task is malformed, reject
            return False

    def _is_complex_enough(self, task: ProposedTask) -> bool:
        """Check if a task is complex enough to be worth training on (push-limits).

        Rejects:
        - Too few test cases (< MIN_TEST_CASES)
        - Solutions that are just a single builtin call (sum, max, len, etc.)
        - Solutions with no branching (no if/for/while)
        - Tasks with trivial signatures (no arguments)
        """
        # Minimum test cases
        if len(task.test_cases) < self.MIN_TEST_CASES:
            return False

        # Check signature has at least 1 argument
        sig_clean = task.signature.split("->")[0].strip().strip("()")
        if not sig_clean or sig_clean == "":
            return False

        # Check the reference implementation for complexity
        ref_code = self._extract_reference_code(task.raw_output)
        if ref_code:
            # Reject if the solution body is just a single builtin call
            # e.g., "return sum(lst)" or "return max(lst)"
            body = ref_code.split(":", 1)[-1].strip() if ":" in ref_code else ref_code
            # Remove common whitespace
            body_lines = [l.strip() for l in body.split("\n") if l.strip()]
            if body_lines:
                # Check if it's a single-line return of a builtin
                single_line = " ".join(body_lines)
                # Extract what's being returned
                ret_match = _RE_RETURN.search(single_line)
                if ret_match:
                    ret_expr = ret_match.group(1).strip()
                    # Check if it's just a builtin call: builtin(args)
                    builtin_match = re.match(
                        r'^(' + '|'.join(self.BANNED_PATTERNS) + r')\s*\(.+\)$',
                        ret_expr)
                    if builtin_match:
                        return False  # trivial: just a builtin wrapper

                # Check for branching (if/for/while) — encourage but don't require
                # (the model often generates test cases without reference code)
                has_branch = any(
                    kw in ref_code
                    for kw in ["if ", "for ", "while ", "elif ", "and ", "or "]
                )
                # Only reject if NO branching AND the solution is a single line
                if not has_branch and len(body_lines) <= 2:
                    return False  # trivial: single-line no-branch solution

        return True

    def _extract_reference_code(self, raw: str) -> str:
        """Extract the def solve(...) implementation from raw model output."""
        # Remove code block markers
        code = _RE_CODEBLOCK_OPEN.sub('', raw)
        code = _RE_CODEBLOCK_CLOSE.sub('', code)

        # Find the function definition — stop at assert/#/def or end
        match = _RE_SOLVE_IMPL.search(code)
        if match:
            return match.group(1).strip()
        return ""

    def _execute_sandbox(self, code: str, timeout_s: float = 3.0) -> dict:
        """Execute Python code in a subprocess sandbox."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        try:
            proc = subprocess.Popen(
                [sys.executable, temp_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                proc.kill()
                return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1}
        finally:
            try:
                os.unlink(temp_path)
            except OSError as e:
                import warnings
                warnings.warn(f"log rotation cleanup: {e}", RuntimeWarning, stacklevel=2)

    # ─── Solving ───────────────────────────────────────────────────

    def solve_task(self, task: ProposedTask,
                   engine: Any | None = None) -> dict:
        """Solve a proposed task using the model.

        Can use the existing RecursiveSelfPlay engine, or generate directly.
        Returns success/failure + solution code.
        """
        # Convert ProposedTask to GoalTask for compatibility with existing engine
        goal_task = GoalTask(
            id=task.id,
            domain=task.domain,
            difficulty=task.difficulty,
            description=task.description,
            input_signature=task.signature,
            solve_name=task.solve_name,
            test_cases=task.test_cases,
            stress_index=task.stress_index,
            archetype=task.archetype,
        )

        if engine is not None:
            # Use existing recursive self-play engine
            result = engine.run_goal_task(goal_task, k_samples=1, use_reasoning=True)
            return result
        else:
            # Direct generation (simpler, no retry loop)
            return self._solve_direct(goal_task)

    def _solve_direct(self, task: GoalTask) -> dict:
        """Solve a task with self-modeling retry (D5).

        On low confidence, retry with higher temperature.
        Accept the best attempt (highest confidence or first that passes tests).
        """
        from research.evaluation.goal_tasks import build_goal_prompt
        prompt = build_goal_prompt(task)

        attempts = []
        best_solution = None
        best_success = False

        for attempt in range(self.retry_policy.max_retries + 1):
            # Adjust temperature for retry (more exploration)
            old_temp = self.temperature
            self.temperature = self.retry_policy.temperature_for_attempt(attempt)

            completion = self._generate(prompt)
            self.temperature = old_temp  # restore

            # Extract code block if present
            code_match = _RE_PYTHON_BLOCK.search(completion)
            solution_body = code_match.group(1) if code_match else completion

            # Build full solution: prompt (has def solve) + completion (has body)
            full_solution = prompt + solution_body

            # Execute and check
            test_code = full_solution + "\n"
            for tc in task.test_cases:
                args_str = ", ".join(repr(a) for a in tc["args"])
                expected_repr = repr(tc["expected"])
                test_code += f"\nassert solve({args_str}) == {expected_repr}, "
                test_code += f"'solve({args_str}) failed'"
            test_code += "\nprint('PASS')"

            result = self._execute_sandbox(test_code, timeout_s=5.0)
            success = result.get("returncode") == 0 and "PASS" in result.get("stdout", "")

            attempts.append({
                "code": full_solution, "success": success,
                "error": result.get("stderr", ""),
                "round": 0, "sample": attempt,
                "temperature": self.retry_policy.temperature_for_attempt(attempt),
            })

            if success:
                best_solution = full_solution
                best_success = True
                break  # passed — no need to retry

            if best_solution is None:
                best_solution = full_solution  # keep first attempt as fallback

            # Self-modeling: check confidence to decide retry
            # (simplified — we use test failure as the error signal)
            # If this was the last attempt, stop
            if attempt >= self.retry_policy.max_retries:
                break

        return {
            "final_success": best_success,
            "attempts": attempts,
            "rounds_used": len(attempts),
            "best_quality": 1.0 if best_success else 0.0,
        }

    # ─── Difficulty Adaptation ─────────────────────────────────────

    def record_result(self, task: ProposedTask, success: bool):
        """Record solver result for adaptive difficulty."""
        self.stats.total_solved += success
        self.stats.total_failed += (not success)

        results = self.stats.domain_results.setdefault(task.domain, [])
        results.append(success)
        if len(results) > self.WINDOW_SIZE:
            results = results[-self.WINDOW_SIZE:]

    def _adaptive_difficulty(self, domain: str) -> str:
        """Pick difficulty based on recent solver success rate (push-limits mode).

        In push-limits mode, we START at hard and only ease to medium if the
        model is failing >90% of the time. We NEVER propose easy tasks.
        The goal is to keep the model at the edge of its capability.
        """
        rate = self.stats.domain_success_rate(domain)

        if rate > self.GOLDILOCKS_HIGH:
            # Model is solving too many → escalate to hard
            return "hard"
        elif rate < self.GOLDILOCKS_LOW:
            # Model is failing almost everything → ease to medium (never easy)
            return "medium"
        else:
            # In the push zone — mostly hard, some medium
            return self.rng.choice(["medium", "hard", "hard", "hard"])

    # ─── Task Queue Management ─────────────────────────────────────

    def get_training_batch(self, n: int,
                           domain: str | None = None,
                           difficulty: str | None = None) -> list[ProposedTask]:
        """Get n validated tasks from the queue for training.

        Returns tasks sorted by difficulty (easy first = curriculum order).
        """
        # Filter queue
        candidates = [t for t in self._task_queue
                      if (domain is None or t.domain == domain)
                      and (difficulty is None or t.difficulty == difficulty)]

        # Sort by difficulty (curriculum: easy → medium → hard)
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        candidates.sort(key=lambda t: diff_order.get(t.difficulty, 1))

        return candidates[:n]

    def queue_size(self, domain: str | None = None) -> int:
        """Number of validated tasks in the queue."""
        if domain is None:
            return len(self._task_queue)
        return sum(1 for t in self._task_queue if t.domain == domain)

    def _save_task(self, task: ProposedTask):
        """Save validated task to disk for replay."""
        path = self.task_queue_dir / f"{task.id}.json"
        data = {
            "id": task.id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "description": task.description,
            "signature": task.signature,
            "solve_name": task.solve_name,
            "test_cases": task.test_cases,
            "stress_index": task.stress_index,
            "archetype": task.archetype,
            "generation_mode": task.generation_mode,
            "proposer_confidence": task.proposer_confidence,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_queue(self, max_tasks: int = 1000):
        """Load previously saved tasks from disk into the queue."""
        files = sorted(self.task_queue_dir.glob("*.json"))[:max_tasks]
        for f in files:
            try:
                data = loads(f.read_text())
                task = ProposedTask(
                    id=data["id"],
                    domain=data["domain"],
                    difficulty=data["difficulty"],
                    description=data["description"],
                    signature=data["signature"],
                    solve_name=data.get("solve_name", "solve"),
                    test_cases=data["test_cases"],
                    stress_index=data.get("stress_index"),
                    archetype=data.get("archetype", "model_proposed"),
                    generation_mode=data.get("generation_mode", "induction"),
                    proposer_confidence=data.get("proposer_confidence", 0),
                    validated=True,
                )
                self._task_queue.append(task)
            except (json.JSONDecodeError, KeyError):
                continue

        print(f"  [Curriculum] Loaded {len(self._task_queue)} tasks from disk")

    # ─── Convert to GoalTask (for existing training pipeline) ──────

    def to_goal_tasks(self, tasks: list[ProposedTask]) -> list[GoalTask]:
        """Convert ProposedTasks to GoalTasks for the existing training pipeline."""
        return [GoalTask(
            id=t.id,
            domain=t.domain,
            difficulty=t.difficulty,
            description=t.description,
            input_signature=t.signature,
            solve_name=t.solve_name,
            test_cases=t.test_cases,
            stress_index=t.stress_index,
            archetype=t.archetype,
        ) for t in tasks]

    # ─── Stats ─────────────────────────────────────────────────────

    def print_stats(self):
        """Print curriculum health metrics."""
        try:
            print("\n  [Curriculum] Stats:")
            print(f"    Proposed: {self.stats.total_proposed}")
            print(f"    Validated: {self.stats.total_validated} "
                  f"({self.stats.validation_rate:.1%})")
            print(f"    Solved: {self.stats.total_solved} "
                  f"({self.stats.solve_rate:.1%})")
            print(f"    Queue: {len(self._task_queue)} tasks")
            for domain in self.DOMAINS:
                rate = self.stats.domain_success_rate(domain)
                n = len(self.stats.domain_results.get(domain, []))
                if n > 0:
                    print(f"    {domain}: {rate:.1%} success ({n} samples)")
        except OSError:
            pass  # Windows console encoding issue — skip stats print
