"""Tool-use self-play loop — agentic tool calling with reward collection.

This is the revamped self-play module that:
1. Gives the model a task + tool definitions in Qwen format
2. Runs the agentic loop: model emits tool calls → we execute → inject results
3. Collects the full trajectory as SFT training data
4. Computes rewards: format correctness, tool execution success, answer quality
5. Saves high-reward trajectories to the DB for SFT/GRPO training

Unlike the discovery loop (goal-free exploration), this loop uses concrete
tasks with verifiable outcomes — so we can compute meaningful rewards.

Tools available:
  - run_script: execute Python in sandbox (from discovery_tools)
  - web_search: DuckDuckGo search (from discovery_tools)
  - calculate: safe math evaluation
  - query_db: read model's own memory (from discovery_tools)
  - think / sudo_think: reasoning (not rewarded, but tracked)

Usage:
    from research.self_play.discovery.tool_use_loop import ToolUseSelfPlay
    loop = ToolUseSelfPlay.from_default_model()
    loop.run_session(n_tasks=20)
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Reduce CUDA memory fragmentation (critical for 12GB VRAM during self-play)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from research.self_play.discovery.discovery_db import DiscoveryDB
from research.self_play.discovery.discovery_tools import ToolRegistry
from research.self_play.discovery.qwen_adapter import (
    qwen_render_messages, qwen_parse_tool_calls, qwen_generate,
    create_tool_grammar, make_grammar_logits_processor,
    IM_START, IM_END, EOS_ID,
)


# ── Task pool for self-play ───────────────────────────────────────────────

# Difficulty tiers: easy (single tool, simple args) → medium (multi-turn or
# complex args) → hard (multi-tool chaining, nested args, rules, long chains).
# The curriculum generator mixes tiers based on rolling success rate.
#
# Task types covered:
#   - calculate: math via the calculate tool
#   - run_script: Python code execution
#   - web_search: internet research
#   - think: reasoning chains (think / sudo_think tools)
#   - query_db: database queries
#   - save_research: persisting findings
#   - multi-tool: chaining 2+ tools in sequence or parallel
#   - instruction_following: rules, format constraints

_TASKS_EASY = [
    # Calculate
    "What is 15 * 37 + 42?",
    "Calculate the factorial of 10.",
    "What is the square root of 144?",
    "Calculate (23 + 17) * 0.15 and round to 2 decimal places.",
    "What is 2^20?",
    "Calculate the area of a circle with radius 5 (use pi=3.14159).",
    "What is 100 / 7? Give the exact decimal to 4 places.",
    "Calculate the compound interest on $1000 at 5% for 3 years.",
    "What is the GCD of 48 and 36?",
    # Run script (simple)
    "Run a Python script that prints the first 10 Fibonacci numbers.",
    "Run a Python script that checks if 97 is a prime number.",
    "Run a script that reverses the string 'hello world'.",
    "Run a script that sorts the list [3, 1, 4, 1, 5, 9, 2, 6] in descending order.",
    "Run a script that creates a dictionary of squares from 1 to 5.",
    "Run a script that finds all even numbers in range(1, 21).",
    "Run a script that counts the vowels in 'artificial intelligence'.",
    # Web search (simple)
    "Search the web for 'python asyncio tutorial' and summarize the top result.",
    "Search for 'RTX 5070 specs' and tell me the VRAM amount.",
    "Search for 'Python dataclasses' and tell me what they are.",
    "Search for 'speculative decoding LLM' and explain the key idea.",
    # Think (simple reasoning)
    "Think about why Python uses indentation for blocks instead of braces. Record your thoughts.",
    "Think about the trade-offs between static and dynamic typing. Use the think tool.",
]

_TASKS_MEDIUM = [
    # Multi-tool: search + calculate
    "Search for 'GRPO reinforcement learning' and explain it briefly, then run a Python script that computes the mean of [10, 20, 30, 40, 50].",
    "Calculate the factorial of 15, then search for 'big numbers in Python' and explain how Python handles arbitrary precision.",
    "Search for 'climate change 2026' and save a research summary, then calculate the percentage increase from 1.5 to 2.0 degrees.",
    "Search for 'transformer architecture explained' and summarize, then run a script that computes the softmax of [1.0, 2.0, 3.0].",
    "Search for 'RTX 5070 specs' and find the VRAM, then run a script that calculates how many 1B-parameter models fit in that VRAM (assume 2 bytes per param).",
    # Multi-tool: script + think
    "Run a Python script that implements a simple stack class with push and pop, then think about why stacks are useful in parsing.",
    "Run a Python script that implements binary search on [1, 3, 5, 7, 9, 11, 13] for target 7, then think about the O(log n) complexity.",
    "Run a script that implements merge sort on [5, 2, 8, 1, 9, 3, 7, 4], then think about why merge sort is stable.",
    # Multi-tool: search + save_research
    "Search for 'liquid foundation models' and summarize what they are, then save the research summary.",
    "Search for 'SFT fine-tuning best practices' and list 3 key points, then save the research.",
    "Search for 'attention mechanism in transformers' and explain Q, K, V, then save the research summary.",
    # Scripting (moderate complexity)
    "Run a Python script that implements a simple calculator function supporting +, -, *, /, then test it with (10 + 5) * 3.",
    "Calculate the GCD of 1071 and 462 using the Euclidean algorithm — run a script to verify.",
    "Run a Python script that generates the first 20 prime numbers, then search for 'prime number theorem' and explain it.",
    "Run a script that implements a queue using two stacks, then think about amortized analysis.",
    # Long thinking chains
    "Think step by step about how you would design a URL shortener. Use the think tool at least 3 times to record your reasoning at each stage.",
    "Think through the pros and cons of microservices vs monoliths. Use sudo_think to meta-reason about whether your analysis is balanced.",
    "Think about how garbage collection works in Python — record at least 2 thoughts about reference counting and generational GC.",
    # Query DB
    "Query the database to see what thoughts you've recorded so far. Use query_db with 'SELECT * FROM thoughts LIMIT 5'.",
]

_TASKS_HARD = [
    # Multi-tool chaining (sequential, 3+ steps)
    "First run a Python script that computes the factorial of 12, then use the result to calculate how many trailing zeros it has by running another script.",
    "Search for 'largest prime under 1000', then run a Python script to verify the answer is indeed prime.",
    "Run a script that generates 100 random numbers and finds the max, then search for 'Python random module' and explain how it works.",
    "Calculate the sum of primes below 100 by running a script, then search for 'sieve of eratosthenes' and explain the algorithm.",
    "Run a script that implements a queue using two stacks, then search for 'amortized analysis' and explain why enqueue is O(1) amortized.",
    # Multi-tool parallel
    "Get the weather in Tokyo and the current time in JST — use both tools in one go.",
    "Search for 'Python list comprehension' and run a script that uses list comprehension to flatten [[1,2],[3,4],[5,6]] — do both at once.",
    "Calculate 15 * 37 and run a script that checks if 555 is a palindrome — do both at once.",
    # Rule-following + tools
    "Search for 'quantum computing basics' and explain it in exactly 15 words.",
    "Run a script that computes the Fibonacci sequence up to 100. Answer in exactly 10 words.",
    "Calculate 2^10 + 2^11. Respond with ONLY the number, no words.",
    "Search for 'gradient descent' and summarize. Your answer must be in ALL lowercase.",
    # Nested arguments
    "Run a Python script with a function that takes a config dict {'name': 'test', 'reps': 5} and prints the name 5 times.",
    "Search for 'Python dataclasses' and create a script with a dataclass Person that has fields name (str) and age (int), then instantiate it.",
    "Run a script that processes a nested structure: {'items': [{'id': 1, 'value': 10}, {'id': 2, 'value': 20}]} and computes the total value.",
    # Long thinking chains (complex)
    "Think through how you would implement a hash table from scratch. Use think 3 times: (1) design the array, (2) handle collisions, (3) think about resize strategy. Then run a script that implements a simple hash table.",
    "Think about the CAP theorem step by step — use think for each property (Consistency, Availability, Partition tolerance), then sudo_think about whether your understanding is correct. Then search for 'CAP theorem' to verify.",
    "Think through how transformer attention works: (1) think about Q, K, V matrices, (2) think about why we divide by sqrt(d_k), (3) think about multi-head attention. Then run a script that computes a simple attention score.",
    # Web research pipeline (search → save → think → query)
    "Search for 'speculative decoding', save the research, think about when it helps vs hurts, then query the DB to retrieve your saved research.",
    "Search for 'model distillation techniques', save the research, think about which technique is best for small models, then query the DB for your thoughts.",
    # Scripting + debugging
    "Run a Python script that has a bug: 'def add(a, b): return a - b'. Think about what the bug is, then run a corrected version.",
    "Run a script that tries to divide by zero. Think about why it fails, then run a script that handles the error gracefully with try/except.",
]

# Open-ended exploration tasks — no fixed answer, model must discover and reason.
# These push the model's limits: it must explore, form hypotheses, and report findings.
_TASKS_EXPLORE = [
    # Research + hypothesis
    "Search for 'small language model techniques 2026' and think about which technique would most help a 1.2B model. Save your research and record your hypothesis.",
    "Search for 'KV cache compression methods' and think about which method is best for long-context inference. Save the research and propose a theory about which is most memory-efficient.",
    "Search for 'mixture of experts small models' and think about whether MoE helps or hurts at the 1B scale. Record your reasoning and save the research.",
    "Search for 'speculative decoding speedup benchmarks' and think about when speculative decoding backfires. Save your findings.",
    "Search for 'quantization aware training vs post training quantization' and reason about which is better for deployment. Save the research and record your conclusion.",
    # Scripting exploration
    "Write and run a Python script that benchmarks list comprehension vs map() vs for-loop for squaring 10000 numbers. Think about why the winner is faster.",
    "Write and run a Python script that implements a simple neural network layer (matrix multiply + ReLU) from scratch using only lists. Think about why NumPy is faster.",
    "Write and run a Python script that measures how dictionary lookup time scales with size (100, 1000, 10000, 100000 keys). Think about why it's O(1).",
    "Write and run a Python script that implements a basic tokenizer (split on whitespace + punctuation). Think about what makes a good tokenizer.",
    "Write and run a Python script that simulates a simple Markov chain and prints the steady-state distribution. Think about what this tells us about language models.",
    # Multi-step discovery
    "Search for 'attention is all you need paper' and summarize the key innovation. Then think about why attention replaced RNNs. Then run a script that computes attention scores for a tiny example.",
    "Search for 'RLHF vs DPO training' and think about which is simpler to implement. Save the research, then run a script that demonstrates a simple preference comparison.",
    "Search for 'temperature in language models' and think about what temperature=0 vs temperature=1 means. Run a script that simulates sampling at different temperatures.",
    "Search for 'token embedding dimension trade-offs' and think about why larger embeddings help. Run a script that shows how cosine similarity changes with dimension.",
    # Meta-reasoning + exploration
    "Think about what you (the model) are good at and bad at. Use sudo_think to reason about your own limitations. Then search for 'small model limitations' and compare your self-assessment with the research.",
    "Think about how you would improve your own reasoning. Use think 3 times: (1) what works, (2) what fails, (3) what to try next. Then search for 'chain of thought prompting' and see if it matches your ideas.",
    "Think about what makes a good tool call vs a bad one. Use sudo_think to reason about common mistakes. Then run a script that demonstrates a well-structured function call.",
    # Creative scripting
    "Write and run a Python script that generates a simple maze using DFS. Think about why DFS works for maze generation.",
    "Write and run a Python script that implements a basic compression algorithm (run-length encoding). Think about when RLE is effective vs ineffective.",
    "Write and run a Python script that simulates a simple ecosystem (predator-prey Lotka-Volterra equations). Think about what the oscillation pattern means.",
    # Open-ended R&D — model builds its own knowledge base
    "Explore a topic of your choice. Search the web, run experiments, form hypotheses, and record what you discover. Use think and sudo_think to reason about your findings. Save your research and record a discovery.",
    "Pick any math concept you find interesting. Write a script to demonstrate it, think about why it works, and record your understanding as a discovery.",
    "Research 'emergent abilities in language models' — search, read, think critically, and form your own theory about what causes emergence. Save research and propose the theory.",
    "Investigate your own architecture: search for 'Liquid Foundation Model LFM2.5' and 'hybrid SSM attention models'. Think about what makes your architecture different from pure transformers. Record your findings.",
    "Design and run an experiment: write a script that tests a hypothesis about Python performance (e.g. 'list append is O(1)'). Think about the results and record a discovery about what you learned.",
    "Explore graph algorithms: write a script implementing BFS and DFS on a small graph. Think about when each is better. Search for 'graph traversal applications in AI' and save your research.",
    "Research 'retrieval augmented generation' and think about how it could help a small model like you. Run a script that demonstrates a simple retrieval concept. Record your conclusions.",
    "Free exploration: use any combination of tools to learn something new. Search the web, run scripts, think deeply, and record at least one discovery in your database.",
    "Investigate 'chain of thought vs direct answer' — run a script that simulates both approaches on a simple problem. Think about when CoT helps vs hurts. Save your findings.",
    "Build a small knowledge base: search for 3 topics of your choice, save each as research, propose a theory connecting them, and record a discovery. Use query_db to review what you've collected.",
]

# Legacy flat list (kept for backward compat)
_TASKS = _TASKS_EASY + _TASKS_MEDIUM + _TASKS_HARD + _TASKS_EXPLORE


class TaskCurriculum:
    """Adaptive task sampler with difficulty progression.

    Tracks rolling success rate per tier. If success > 70%, shifts toward
    harder tasks. If < 30%, shifts toward easier. Target: ~50% success
    (Goldillows zone for maximal learning signal).

    Tiers: easy → medium → hard → explore
    The 'explore' tier has open-ended tasks with no fixed answer — the model
    must discover, reason, and report findings. These push the model's limits.
    """

    TARGET_SUCCESS = 0.5
    WINDOW = 20  # rolling window size

    def __init__(self):
        self.tiers = ["easy", "medium", "hard", "explore"]
        self.tasks = {
            "easy": list(_TASKS_EASY),
            "medium": list(_TASKS_MEDIUM),
            "hard": list(_TASKS_HARD),
            "explore": list(_TASKS_EXPLORE),
        }
        self.results = {t: [] for t in self.tiers}  # rolling [bool]
        self.weights = {"easy": 0.35, "medium": 0.30, "hard": 0.20, "explore": 0.15}
        self.rng = random.Random()

    def record(self, tier: str, success: bool):
        """Record a task outcome and adapt weights."""
        if tier not in self.results:
            return
        self.results[tier].append(success)
        if len(self.results[tier]) > self.WINDOW:
            self.results[tier] = self.results[tier][-self.WINDOW:]
        self._adapt()

    def _adapt(self):
        """Adjust weights based on rolling success rates."""
        rates = {}
        for tier in self.tiers:
            r = self.results[tier]
            rates[tier] = sum(r) / len(r) if r else 0.5

        # Adaptive shifting: if a tier is mastered (>80%), shift weight to harder.
        # If a tier is too hard (<20%), shift weight to easier.
        # Always keep some explore weight — discovery is the goal.
        if rates["easy"] > 0.8 and rates.get("hard", 0.5) < 0.3:
            self.weights = {"easy": 0.15, "medium": 0.30, "hard": 0.35, "explore": 0.20}
        elif rates["easy"] > 0.8:
            self.weights = {"easy": 0.20, "medium": 0.35, "hard": 0.25, "explore": 0.20}
        elif rates.get("hard", 0.5) < 0.2:
            self.weights = {"easy": 0.45, "medium": 0.30, "hard": 0.10, "explore": 0.15}
        elif rates.get("medium", 0.5) > 0.8:
            self.weights = {"easy": 0.10, "medium": 0.25, "hard": 0.40, "explore": 0.25}
        elif rates.get("explore", 0.5) > 0.6:
            # Model is good at exploration — push it further
            self.weights = {"easy": 0.10, "medium": 0.20, "hard": 0.30, "explore": 0.40}
        # Default: keep current weights

    def sample(self, n: int) -> list[tuple[str, str]]:
        """Sample n tasks. Returns list of (tier, task_string)."""
        tiers = self.rng.choices(self.tiers, weights=[self.weights[t] for t in self.tiers], k=n)
        result = []
        for tier in tiers:
            task = self.rng.choice(self.tasks[tier])
            result.append((tier, task))
        return result

    def stats(self) -> dict:
        return {
            tier: {
                "n": len(self.results[tier]),
                "success_rate": round(sum(self.results[tier]) / max(len(self.results[tier]), 1), 2),
                "weight": round(self.weights[tier], 2),
            }
            for tier in self.tiers
        }


# ── Reward computation ────────────────────────────────────────────────────

@dataclass
class ToolUseReward:
    """Multi-component reward for a tool-use trajectory.

    Components:
    - format_ok: Did the model emit valid JSON tool calls? (0..1)
    - tool_executed: Did tools execute without errors? (0..1)
    - output_quality: Did script outputs actually contain useful content? (0..1)
    - task_completed: Did the trajectory actually accomplish the task? (0..1)
    - answer_given: Did the model produce a final answer? (0..1)
    - answer_relevant: Is the answer relevant to the task? (0..1)
    - discovery: Did the model find/learn something new? (0..1)
    - stopped_ok: Did the model stop correctly after tools + answer? (0..1)
    - conciseness: Did the model keep output concise? (0..1)
    """
    format_ok: float = 0.0
    tool_executed: float = 0.0
    output_quality: float = 0.0    # scripts produced real output, searches found results
    task_completed: float = 0.0    # trajectory actually solved the task
    answer_given: float = 0.0
    answer_relevant: float = 0.0
    discovery: float = 0.0         # model explored new info, not just repeated
    stopped_ok: float = 0.0
    conciseness: float = 0.0       # reward shorter answers when task is done

    @property
    def total(self) -> float:
        """Weighted sum. Task completion + output quality are most important.
        Conciseness gives a small bonus for not rambling."""
        return (
            self.format_ok * 0.10 +
            self.tool_executed * 0.15 +
            self.output_quality * 0.20 +
            self.task_completed * 0.20 +
            self.answer_given * 0.08 +
            self.answer_relevant * 0.10 +
            self.discovery * 0.07 +
            self.stopped_ok * 0.05 +
            self.conciseness * 0.05
        )

    def to_dict(self) -> dict:
        return {
            "format_ok": self.format_ok,
            "tool_executed": self.tool_executed,
            "output_quality": self.output_quality,
            "task_completed": self.task_completed,
            "answer_given": self.answer_given,
            "answer_relevant": self.answer_relevant,
            "discovery": self.discovery,
            "stopped_ok": self.stopped_ok,
            "conciseness": self.conciseness,
            "total": self.total,
        }


def _check_answer_relevance(task: str, answer: str) -> float:
    """Heuristic check: does the answer contain keywords from the task?

    Returns 0..1 score. Not a perfect metric but good enough for filtering.
    """
    if not answer or len(answer) < 5:
        return 0.0
    # Extract significant words from the task
    task_words = set(re.findall(r'\b[a-z]{3,}\b', task.lower()))
    # Remove common stopwords
    stopwords = {"the", "and", "for", "what", "how", "tell", "me", "about",
                 "search", "calculate", "script", "that", "with", "this",
                 "then", "your", "must", "only", "exactly", "answer"}
    task_words -= stopwords
    if not task_words:
        return 0.5  # can't evaluate, give neutral
    answer_lower = answer.lower()
    matches = sum(1 for w in task_words if w in answer_lower)
    return min(matches / max(len(task_words), 1), 1.0)


def _evaluate_script_output(task: str, tool_calls: list[dict]) -> float:
    """Judge whether script outputs are actually useful — not just that they ran.

    Checks:
    - Script ran successfully (returncode 0)
    - stdout is non-empty and substantive (>10 chars)
    - Output is relevant to the task (contains task keywords or numbers)
    - For calculation tasks: output contains a numeric result
    - For script tasks: output contains expected patterns
    """
    script_calls = [tc for tc in tool_calls if tc.get("name") == "run_script"]
    if not script_calls:
        return 0.0

    scores = []
    for tc in script_calls:
        result = tc.get("result", {})
        if not isinstance(result, dict):
            scores.append(0.0)
            continue

        score = 0.0
        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()
        ok = result.get("ok", False)

        # Ran successfully
        if ok:
            score += 0.3

        # Has substantive output (not just empty or whitespace)
        if len(stdout) > 10:
            score += 0.3
        elif len(stdout) > 0:
            score += 0.15

        # Output is relevant to the task
        task_lower = task.lower()
        stdout_lower = stdout.lower()
        task_words = set(re.findall(r'\b[a-z]{4,}\b', task_lower))
        task_words -= {"script", "python", "print", "that", "with", "this",
                       "then", "your", "answer", "calculate", "search"}
        if task_words:
            matches = sum(1 for w in task_words if w in stdout_lower)
            if matches > 0:
                score += 0.2

        # For calculation tasks: output contains a number
        if any(kw in task_lower for kw in ["calculate", "what is", "factorial",
                                            "square root", "gcd", "area"]):
            if re.search(r'\d+\.?\d*', stdout):
                score += 0.2

        # For list/sequence tasks: output contains multiple lines or list-like
        if any(kw in task_lower for kw in ["fibonacci", "prime", "sort",
                                            "even numbers", "squares"]):
            if stdout.count("\n") >= 3 or "[" in stdout:
                score += 0.2

        scores.append(min(score, 1.0))

    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_search_output(task: str, tool_calls: list[dict]) -> float:
    """Judge whether web search results are actually useful.

    Checks:
    - Search returned results (not empty)
    - Results have titles and snippets
    - Results are relevant to the search query in the task
    """
    search_calls = [tc for tc in tool_calls if tc.get("name") == "web_search"]
    if not search_calls:
        return 0.0

    scores = []
    for tc in search_calls:
        result = tc.get("result", {})
        if not isinstance(result, dict):
            scores.append(0.0)
            continue

        results = result.get("results", [])
        if not results:
            scores.append(0.0)
            continue

        score = 0.0
        # Has results
        score += 0.3

        # Results have substantive content
        substantive = sum(1 for r in results
                          if len(r.get("snippet", "")) > 20)
        score += min(substantial / max(len(results), 1), 1.0) * 0.3

        # Results are relevant to the task
        task_words = set(re.findall(r'\b[a-z]{4,}\b', task.lower()))
        task_words -= {"search", "tell", "about", "explain", "what", "find"}
        if task_words:
            all_text = " ".join(
                r.get("title", "") + " " + r.get("snippet", "")
                for r in results).lower()
            matches = sum(1 for w in task_words if w in all_text)
            score += min(matches / max(len(task_words), 1), 1.0) * 0.4

        scores.append(min(score, 1.0))

    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_task_completion(task: str, tool_calls: list[dict],
                              final_answer: str | None) -> float:
    """Judge whether the trajectory actually completed the task.

    This is the hardest signal to compute automatically. We use heuristics:
    - For calculation tasks: does the answer contain a number?
    - For script tasks: did a script run successfully with output?
    - For search tasks: did search return results AND model summarized them?
    - For multi-step tasks: were multiple tools called in sequence?
    - For instruction-following: does the answer meet the constraint?
    """
    task_lower = task.lower()
    score = 0.0

    # Calculation tasks
    if any(kw in task_lower for kw in ["calculate", "what is", "factorial",
                                        "square root", "gcd", "area",
                                        "compound interest"]):
        # Check if answer or script output contains a number
        has_number = False
        if final_answer and re.search(r'\d+\.?\d*', final_answer):
            has_number = True
        for tc in tool_calls:
            if tc.get("name") in ("calculate", "run_script"):
                result = tc.get("result", {})
                if isinstance(result, dict):
                    stdout = result.get("stdout", "")
                    if re.search(r'\d+\.?\d*', stdout):
                        has_number = True
        if has_number:
            score += 0.5
        if final_answer and len(final_answer) > 5:
            score += 0.3
        if any(tc.get("success") for tc in tool_calls):
            score += 0.2

    # Script tasks
    elif "run" in task_lower and "script" in task_lower:
        script_ok = any(
            tc.get("name") == "run_script" and tc.get("success")
            for tc in tool_calls
        )
        has_output = any(
            tc.get("name") == "run_script" and
            len(tc.get("result", {}).get("stdout", "")) > 5
            for tc in tool_calls
        )
        if script_ok:
            score += 0.4
        if has_output:
            score += 0.3
        if final_answer and len(final_answer) > 5:
            score += 0.3

    # Search tasks
    elif "search" in task_lower:
        search_ok = any(
            tc.get("name") == "web_search" and
            len(tc.get("result", {}).get("results", [])) > 0
            for tc in tool_calls
        )
        if search_ok:
            score += 0.4
        if final_answer and len(final_answer) > 10:
            score += 0.3
        # Did the model actually summarize the search results?
        if final_answer:
            search_snippets = ""
            for tc in tool_calls:
                if tc.get("name") == "web_search":
                    for r in tc.get("result", {}).get("results", []):
                        search_snippets += r.get("snippet", "") + " "
            if search_snippets:
                # Check if answer shares content with search results
                ans_words = set(re.findall(r'\b[a-z]{4,}\b', final_answer.lower()))
                snippet_words = set(re.findall(r'\b[a-z]{4,}\b', search_snippets.lower()))
                overlap = len(ans_words & snippet_words) / max(len(ans_words), 1)
                score += min(overlap, 0.3)

    # Think tasks
    elif "think" in task_lower:
        think_ok = any(
            tc.get("name") in ("think", "sudo_think") and tc.get("success")
            for tc in tool_calls
        )
        if think_ok:
            score += 0.5
        if final_answer and len(final_answer) > 20:
            score += 0.3
        # Count number of think calls (reward longer reasoning chains)
        n_thinks = sum(1 for tc in tool_calls if tc.get("name") in ("think", "sudo_think"))
        score += min(n_thinks * 0.1, 0.2)

    # Multi-step tasks (contains "then" or "first")
    elif "then" in task_lower or "first" in task_lower:
        n_tools = len(tool_calls)
        if n_tools >= 2:
            score += 0.4
        if n_tools >= 3:
            score += 0.2
        if final_answer and len(final_answer) > 10:
            score += 0.2
        if all(tc.get("success") for tc in tool_calls if tc):
            score += 0.2

    # Instruction following (word count, lowercase, json)
    elif "exactly" in task_lower or "only" in task_lower or "must" in task_lower:
        if final_answer:
            if "10 words" in task_lower:
                n = len(final_answer.split())
                if 8 <= n <= 12:
                    score += 0.6
                elif 6 <= n <= 14:
                    score += 0.3
            elif "lowercase" in task_lower:
                if not any(c.isupper() for c in final_answer if c.isalpha()):
                    score += 0.6
            elif "json" in task_lower:
                try:
                    json.loads(final_answer.strip())
                    score += 0.6
                except Exception:
                    pass
            else:
                score += 0.3  # gave some answer
            if len(final_answer) > 5:
                score += 0.2

    # Default: tools called + answer given
    else:
        if tool_calls:
            score += 0.3
        if final_answer and len(final_answer) > 10:
            score += 0.4
        if any(tc.get("success") for tc in tool_calls):
            score += 0.3

    return min(score, 1.0)


def _evaluate_discovery(task: str, tool_calls: list[dict],
                        final_answer: str | None,
                        seen_outputs: set | None = None) -> float:
    """Reward exploration and finding new information.

    Discovery signals:
    - Web search returned results the model hasn't seen before
    - Script output contains new information (not just echoing input)
    - Model used save_research to persist findings
    - Model used think/sudo_think for meta-reasoning
    - Model explored a topic beyond the minimum required
    - Model chained tools in a novel way
    """
    score = 0.0

    # Used save_research (explicitly persisting new knowledge)
    if any(tc.get("name") == "save_research" and tc.get("success")
           for tc in tool_calls):
        score += 0.3

    # Used think/sudo_think (meta-reasoning about process)
    n_thinks = sum(1 for tc in tool_calls
                   if tc.get("name") in ("think", "sudo_think"))
    score += min(n_thinks * 0.1, 0.2)

    # Web search found substantive results
    for tc in tool_calls:
        if tc.get("name") == "web_search":
            results = tc.get("result", {}).get("results", [])
            if len(results) >= 3:
                score += 0.15
            # Check if snippets have real content
            total_content = sum(len(r.get("snippet", "")) for r in results)
            if total_content > 200:
                score += 0.1

    # Script produced non-trivial output (not just echoing)
    for tc in tool_calls:
        if tc.get("name") == "run_script":
            stdout = tc.get("result", {}).get("stdout", "").strip()
            if len(stdout) > 50:
                # Check it's not just a copy of the task
                task_words = set(re.findall(r'\b[a-z]{4,}\b', task.lower()))
                stdout_words = set(re.findall(r'\b[a-z]{4,}\b', stdout.lower()))
                # If output has words NOT in the task, it's producing new info
                new_words = stdout_words - task_words - {"print", "true", "false", "none"}
                if len(new_words) >= 2:
                    score += 0.1

    # Novelty: output hasn't been seen before (if tracking)
    if seen_outputs is not None:
        for tc in tool_calls:
            result = tc.get("result", {})
            if isinstance(result, dict):
                output_key = (tc.get("name", ""),
                              result.get("stdout", "")[:100],
                              str(result.get("results", ""))[:100])
                if output_key not in seen_outputs:
                    score += 0.05
                    seen_outputs.add(output_key)

    # Multi-tool exploration (used 3+ different tools)
    unique_tools = set(tc.get("name") for tc in tool_calls)
    if len(unique_tools) >= 3:
        score += 0.1

    return min(score, 1.0)


def compute_reward(task: str, tool_calls: list[dict],
                   final_answer: str | None,
                   stopped_after_tools: bool,
                   stopped_after_answer: bool,
                   seen_outputs: set | None = None) -> ToolUseReward:
    """Compute multi-component reward for a tool-use trajectory.

    Args:
        seen_outputs: optional set for tracking novel outputs across sessions.
                      Updated in-place with new outputs.
    """
    r = ToolUseReward()

    # Format: did all tool calls parse as valid JSON?
    if tool_calls:
        r.format_ok = 1.0
    elif final_answer and not tool_calls:
        r.format_ok = 0.3

    # Tool execution: did tools execute successfully?
    if tool_calls:
        successes = sum(1 for tc in tool_calls if tc.get("success"))
        r.tool_executed = successes / len(tool_calls)

    # Output quality: did scripts/searches produce useful content?
    r.output_quality = max(
        _evaluate_script_output(task, tool_calls),
        _evaluate_search_output(task, tool_calls),
    )

    # Task completion: did the trajectory actually solve the task?
    r.task_completed = _evaluate_task_completion(task, tool_calls, final_answer)

    # Answer given
    r.answer_given = 1.0 if final_answer and len(final_answer) > 5 else 0.0

    # Answer relevance
    r.answer_relevant = _check_answer_relevance(task, final_answer or "")

    # Discovery: did the model find/explore new things?
    r.discovery = _evaluate_discovery(task, tool_calls, final_answer, seen_outputs)

    # Stopped correctly
    if stopped_after_tools and stopped_after_answer:
        r.stopped_ok = 1.0
    elif stopped_after_tools or stopped_after_answer:
        r.stopped_ok = 0.5

    # Conciseness: reward shorter final answers when task is completed.
    # Only applies when the task was actually solved — don't reward brevity
    # over correctness. Target: <50 tokens for the final answer.
    if r.task_completed > 0.5 and final_answer:
        ans_tokens = len(final_answer.split())
        if ans_tokens <= 20:
            r.conciseness = 1.0
        elif ans_tokens <= 50:
            r.conciseness = 0.5
        elif ans_tokens <= 100:
            r.conciseness = 0.2

    return r


# ── Self-play loop ────────────────────────────────────────────────────────

@dataclass
class SelfPlayConfig:
    max_turns: int = 10          # max tool-call turns (increased for R&D freedom)
    max_gen_tokens: int = 512    # per-turn generation limit (increased for reasoning)
    temperature: float = 0.2     # LFM2.5-recommended (low for tool use)
    top_k: int = 80              # LFM2.5-recommended top-k sampling
    repetition_penalty: float = 1.05  # LFM2.5-recommended repetition penalty
    min_reward: float = 0.4      # minimum reward to save trajectory
    n_tasks: int = 20            # tasks per session
    device: str = "cuda"


class ToolUseSelfPlay:
    """Tool-use self-play loop with reward collection.

    Runs the SFT-trained model on tasks, collects trajectories, computes
    rewards, and saves high-quality ones for SFT/GRPO training.

    Supports two backends:
      - ForgeEngine (recommended): uses KV cache, Triton conv, torch.compile.
        Pass ``engine=`` to constructor or ``from_checkpoint=`` to factory.
      - Raw model (legacy): uses qwen_generate with O(n²) re-computation.
        Pass ``model=`` and ``tokenizer=`` to constructor.
    """

    def __init__(self, model=None, tokenizer=None, db: DiscoveryDB | None = None,
                 config: SelfPlayConfig | None = None,
                 engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.db = db or DiscoveryDB()
        self.config = config or SelfPlayConfig()
        self._seen_outputs: set = set()  # tracks novel outputs for discovery reward
        # If engine given, extract model+tokenizer for compatibility
        if engine is not None:
            self.model = engine.model
            self.tokenizer = engine.tokenizer

    @classmethod
    def from_default_model(cls, db_path: str | None = None,
                           config: SelfPlayConfig | None = None,
                           checkpoint: str | None = None,
                           use_engine: bool = True) -> "ToolUseSelfPlay":
        """Load the SFT-trained model for self-play.

        Args:
            checkpoint: path to safetensors checkpoint. If None, auto-detects
                        the latest sft* checkpoint in research/checkpoints/.
            use_engine: if True (default), uses ForgeEngine with KV cache,
                        Triton conv, and warmup. ~5-10x faster than raw model.
        """
        import torch
        from research.paths import DATA_DIR
        import os
        import glob

        db = DiscoveryDB(db_path or str(DATA_DIR / "discovery" / "discovery.sqlite3"))

        # Auto-detect latest sft checkpoint if not specified
        if checkpoint is None:
            ckpt_dir = "research/checkpoints"
            candidates = sorted(glob.glob(f"{ckpt_dir}/ForgeLM_V2_LFM25-1.2B.sft*.safetensors"),
                                key=os.path.getmtime)
            checkpoint = candidates[-1] if candidates else None

        cfg = config or SelfPlayConfig()

        if use_engine:
            # Use ForgeEngine — gets KV cache, Triton conv, warmup for free
            from research.inference.forge_engine import ForgeEngine
            engine = ForgeEngine.from_checkpoint(
                checkpoint=checkpoint,
                config_name="lfm25_1.2b",
                tokenizer_path="research/checkpoints/lfm25_tokenizer",
                device=cfg.device,
            )
            engine.activate(
                kv_cache="standard",
                decoding="standard",
                use_triton_conv=True,
                warmup=True,
            )
            return cls(db=db, config=cfg, engine=engine)
        else:
            # Legacy: raw model + qwen_generate (O(n²), no KV cache)
            from research.model_loader import load_default_model
            from research.tokenizer_cache import get_tokenizer
            model, _ = load_default_model(
                "lfm25_1.2b", checkpoint_path=checkpoint,
                device=cfg.device, dtype=torch.bfloat16,
            )
            model.eval()
            tokenizer = get_tokenizer()
            return cls(model=model, tokenizer=tokenizer, db=db, config=cfg)

    def _build_tools(self, session_id: str) -> tuple[ToolRegistry, list[dict]]:
        """Build the tool registry and Qwen-format tool definitions."""
        registry = ToolRegistry(self.db, session_id)
        # Get tool definitions in Qwen format (name/description/parameters)
        tools = []
        for s in registry.schemas:
            tools.append({
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            })
        return registry, tools

    def run_task(self, task: str, registry: ToolRegistry,
                 tools: list[dict]) -> dict:
        """Run a single task through the agentic loop.

        Returns: {messages, tool_calls, reward, final_answer, ...}
        """
        cfg = self.config
        messages = [{"role": "user", "content": task}]
        tool_call_records = []  # {name, args, result, success}
        final_answer = None
        stopped_after_tools = False
        stopped_after_answer = False

        # Context manager for long conversations
        from research.self_play.discovery.context_manager import ContextManager, ContextManagerConfig
        ctx_config = ContextManagerConfig(
            max_seq_len=32768,
            reserved_for_generation=cfg.max_gen_tokens + 512,
            keep_recent_turns=6,
            use_model_summary=False,  # heuristic for speed
        )
        ctx_manager = ContextManager(ctx_config, tokenizer=self.tokenizer)

        for turn in range(cfg.max_turns):
            # Compress context if approaching token budget
            messages, was_compressed = ctx_manager.maybe_compress(messages)
            if was_compressed:
                # Log the compression event
                pass  # could emit to DB

            # Build prompt
            prompt = qwen_render_messages(messages, tools=tools,
                                          add_generation_prompt=True)

            # Create fresh grammar matcher for this turn (constrains tool calls)
            grammar_matcher, bitmask = create_tool_grammar(self.tokenizer, tools)

            # Generate: use ForgeEngine (fast) or qwen_generate (legacy)
            if self.engine is not None:
                # Fast path: ForgeEngine with KV cache + Triton conv
                lp = make_grammar_logits_processor(
                    grammar_matcher, bitmask, self.tokenizer)
                output = self.engine.generate_raw(
                    prompt,
                    max_new_tokens=cfg.max_gen_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    repetition_penalty=cfg.repetition_penalty,
                    logits_processor=lp,
                    eos_token_ids=[EOS_ID],
                    skip_special_tokens=False,
                )
            else:
                # Legacy path: raw model, O(n²) re-computation
                output = qwen_generate(
                    self.model, self.tokenizer, prompt,
                    max_new_tokens=cfg.max_gen_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    repetition_penalty=cfg.repetition_penalty,
                    device=cfg.device,
                    grammar_matcher=grammar_matcher,
                    bitmask=bitmask,
                )

            # Parse tool calls
            calls, musing = qwen_parse_tool_calls(output)

            if not calls:
                # No tool calls — this is the final answer
                final_answer = (musing or output).strip()
                messages.append({"role": "assistant", "content": final_answer})
                stopped_after_answer = True
                break

            # Execute tool calls
            stopped_after_tools = True
            tc_for_msg = []
            tool_results = []  # parallel list to tc_for_msg
            for call in calls:
                name = call["name"]
                args = call.get("arguments", call.get("args", {}))
                result = registry.call(name, args)
                success = "error" not in result if isinstance(result, dict) else False
                tool_call_records.append({
                    "name": name,
                    "args": args,
                    "result": result,
                    "success": success,
                })
                tc_for_msg.append({"name": name, "arguments": args})
                tool_results.append(result)

            # Add ONE assistant message with ALL tool calls, then tool results
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": tc_for_msg})
            for tc, result in zip(tc_for_msg, tool_results):
                messages.append({"role": "tool", "name": tc["name"],
                                 "content": json.dumps(result, ensure_ascii=False)[:2000]})

        # Compute reward
        reward = compute_reward(
            task=task,
            tool_calls=tool_call_records,
            final_answer=final_answer,
            stopped_after_tools=stopped_after_tools,
            stopped_after_answer=stopped_after_answer,
            seen_outputs=getattr(self, '_seen_outputs', None),
        )

        return {
            "task": task,
            "messages": messages,
            "tool_calls": tool_call_records,
            "reward": reward,
            "final_answer": final_answer,
            "stopped_after_tools": stopped_after_tools,
            "stopped_after_answer": stopped_after_answer,
            "n_turns": turn + 1,
        }

    def run_session(self, n_tasks: int | None = None,
                    curriculum: TaskCurriculum | None = None) -> dict:
        """Run a self-play session with multiple tasks.

        If curriculum is provided, uses adaptive difficulty sampling.
        Otherwise, uses the flat _TASKS list (backward compat).

        Returns session summary with stats.
        """
        cfg = self.config
        n = n_tasks or cfg.n_tasks
        session_id = f"sp_{uuid.uuid4().hex[:10]}"
        self.db.start_session(session_id)

        registry, tools = self._build_tools(session_id)

        # Sample tasks
        if curriculum is not None:
            sampled = curriculum.sample(n)
            tasks_with_tier = [(tier, task) for tier, task in sampled]
        else:
            rng = random.Random()
            tasks_with_tier = [("flat", t) for t in rng.choices(_TASKS, k=n)]

        results = []
        t0 = time.time()

        for i, (tier, task) in enumerate(tasks_with_tier):
            result = self.run_task(task, registry, tools)
            results.append(result)

            # Record outcome in curriculum
            reward_total = result["reward"].total
            success = reward_total >= cfg.min_reward
            if curriculum is not None:
                curriculum.record(tier, success)

            # Save high-reward trajectories
            if success:
                self.db.add_tool_trajectory(
                    session_id=session_id,
                    task=task,
                    messages=result["messages"],
                    tool_calls=result["tool_calls"],
                    reward=reward_total,
                    final_answer=result["final_answer"],
                )

            status = "SAVED" if success else "skip"
            print(f"  [{i+1}/{n}] [{tier}] reward={reward_total:.2f} "
                  f"tools={len(result['tool_calls'])} "
                  f"answer={'yes' if result['final_answer'] else 'no'} "
                  f"[{status}]", flush=True)

        elapsed = round(time.time() - t0, 1)
        self.db.end_session(session_id, f"tool-use self-play: {n} tasks in {elapsed}s")

        # Compute stats
        rewards = [r["reward"].total for r in results]
        n_saved = sum(1 for r in rewards if r >= cfg.min_reward)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        n_with_tools = sum(1 for r in results if r["tool_calls"])
        n_with_answers = sum(1 for r in results if r["final_answer"])

        stats = {
            "session_id": session_id,
            "n_tasks": n,
            "n_saved": n_saved,
            "avg_reward": round(avg_reward, 3),
            "n_with_tools": n_with_tools,
            "n_with_answers": n_with_answers,
            "elapsed_s": elapsed,
            "curriculum": curriculum.stats() if curriculum else None,
        }
        print(f"\n  Session done: {stats}")
        return stats

    def export_sft_dataset(self, min_reward: float = 0.5,
                           output_path: str | None = None) -> str:
        """Export high-quality trajectories as SFT-format JSONL.

        Each line is {"messages": [...]} in Qwen format, ready for sft_train.py.
        """
        from research.paths import DATA_DIR
        if output_path is None:
            output_path = str(DATA_DIR / "finetune" / "self_play_trajectories.jsonl")

        trajectories = self.db.get_trajectories(min_reward=min_reward, limit=500)
        with open(output_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps({"messages": traj["messages"]},
                                   ensure_ascii=False) + "\n")

        print(f"  Exported {len(trajectories)} trajectories to {output_path}")
        return output_path

    def collect_grpo_batch(self, tasks: list[str],
                           group_size: int = 4) -> dict:
        """Collect a GRPO training batch: G completions per task with rewards.

        For each task, generates `group_size` completions at temperature > 0,
        runs the agent loop on each, and computes tool-use rewards.

        Returns:
            {
                "prompts": list[str],          # B prompts (rendered)
                "completions": list[list[str]], # B × G completion strings
                "rewards": list[list[float]],   # B × G reward floats (0..1)
            }
        """
        cfg = self.config
        registry, tools = self._build_tools("grpo")
        prompts = []
        completions = []
        rewards = []

        for task in tasks:
            # Build the initial prompt (same for all G completions)
            messages = [{"role": "user", "content": task}]
            prompt = qwen_render_messages(messages, tools=tools,
                                          add_generation_prompt=True)

            group_completions = []
            group_rewards = []

            for _ in range(group_size):
                result = self.run_task(task, registry, tools)
                # Extract the first assistant turn as the "completion"
                # (GRPO trains on the first response; multi-turn is handled
                #  by the turn-level advantage in the trainer)
                first_asst = None
                for m in result["messages"]:
                    if m["role"] == "assistant":
                        if m.get("tool_calls"):
                            # Render tool calls as JSON string
                            body = "\n".join(
                                json.dumps(tc, ensure_ascii=False)
                                for tc in m["tool_calls"])
                            first_asst = body
                        else:
                            first_asst = m.get("content", "")
                        break

                if first_asst is None:
                    first_asst = ""

                group_completions.append(first_asst)
                group_rewards.append(result["reward"].total)

            prompts.append(prompt)
            completions.append(group_completions)
            rewards.append(group_rewards)

        return {
            "prompts": prompts,
            "completions": completions,
            "rewards": rewards,
        }
