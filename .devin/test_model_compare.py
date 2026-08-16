"""Model comparison benchmark: base LFM2.5 vs SFT-trained model.

Runs identical prompts on both models and measures:
  - Tokens generated per response
  - VRAM usage (peak during generation)
  - Time per generation (ms)
  - Quality score (tool-call correctness, answer relevance, format compliance)

Categories:
  1. Tool use (novel tools, parallel calls, sequential chaining)
  2. Knowledge (factual Q&A)
  3. Reasoning (math, logic, multi-step)
  4. Code generation (script writing, debugging)
  5. Instruction following (rules, format constraints)

The winner is promoted as default; the loser is archived.

Usage:
    python .devin/test_model_compare.py
    python .devin/test_model_compare.py --base research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors --candidate research/checkpoints/ForgeLM_V2_BSP.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from research.model_loader import load_default_model
from research.tokenizer_cache import get_tokenizer
from research.inference.forge_engine import ForgeEngine
from research.self_play.discovery.qwen_adapter import (
    qwen_parse_tool_calls, IM_START, IM_END, EOS_ID,
)

# ── Metrics ────────────────────────────────────────────────────────────────

@dataclass
class GenMetrics:
    """Per-generation metrics."""
    tokens: int = 0
    time_ms: float = 0.0
    vram_mb: float = 0.0
    quality: float = 0.0  # 0..1 category-specific
    output: str = ""

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / (self.time_ms / 1000) if self.time_ms > 0 else 0


@dataclass
class ModelResult:
    """Aggregate results for one model."""
    name: str
    checkpoint: str
    per_category: dict[str, list[GenMetrics]] = field(default_factory=dict)
    total_quality: float = 0.0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    peak_vram_mb: float = 0.0

    def aggregate(self):
        all_metrics = [m for cat in self.per_category.values() for m in cat]
        self.total_tokens = sum(m.tokens for m in all_metrics)
        self.total_time_ms = sum(m.time_ms for m in all_metrics)
        self.total_quality = sum(m.quality for m in all_metrics) / max(len(all_metrics), 1)
        self.peak_vram_mb = max((m.vram_mb for m in all_metrics), default=0)

    @property
    def avg_tokens_per_sec(self) -> float:
        return self.total_tokens / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0


# ── Generation helper ──────────────────────────────────────────────────────

def generate_with_metrics(engine: ForgeEngine, prompt: str,
                          max_new_tokens: int = 256,
                          device: str = "cuda",
                          temperature: float = 0.0,
                          top_k: int = 80,
                          repetition_penalty: float = 1.05) -> GenMetrics:
    """Generate text and collect performance metrics.

    Uses ForgeEngine with KV cache + Triton conv for fast generation.
    When temperature=0 (default), uses greedy decoding for reproducible benchmarks.
    """
    # Track VRAM before generation
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()

    # Generate via ForgeEngine (KV cache, Triton conv, warmup all active)
    output_text = engine.generate_raw(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_ids=[EOS_ID],
        skip_special_tokens=False,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # Count tokens generated (approximate from decoded text)
    n_tokens = len(engine.tokenizer.encode(output_text, add_special_tokens=False))

    return GenMetrics(
        tokens=n_tokens,
        time_ms=elapsed_ms,
        vram_mb=peak_vram,
        output=output_text,
    )


# ── Test categories ────────────────────────────────────────────────────────

# Category 1: Tool use — uses system message for tool defs (matches training format)
TOOL_TESTS = [
    {
        "name": "novel_tool_stock",
        "prompt": (
            "<|im_start|>system\n"
            "You have access to this tool:\n"
            "- get_stock_price: Get the current price of a stock.\n"
            "  arguments: {\"ticker\": \"string\", \"exchange\": \"string\"}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the price of NVDA?\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_tool_call(m.output, "get_stock_price", {"ticker": "NVDA"}),
    },
    {
        "name": "parallel_tools",
        "prompt": (
            "<|im_start|>system\n"
            "You have access to these tools:\n"
            "- get_weather: Get weather for a city.\n"
            "  arguments: {\"city\": \"string\"}\n"
            "- get_time: Get current time in a timezone.\n"
            "  arguments: {\"timezone\": \"string\"}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Get the weather in Tokyo and the time in JST.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_parallel_tools(m.output, ["get_weather", "get_time"]),
    },
    {
        "name": "nested_args",
        "prompt": (
            "<|im_start|>system\n"
            "You have access to this tool:\n"
            "- update_profile: Update a user profile.\n"
            "  arguments: {\"user_id\": \"string\", \"fields\": \"object\"}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Update user 'u123' to set name to 'Alice' and age to 30.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_nested_args(m.output, "update_profile", "u123"),
    },
    {
        "name": "sequential_chain_t1",
        "prompt": (
            "<|im_start|>system\n"
            "You have access to these tools:\n"
            "- create_ticket: Create a support ticket. Returns {\"ticket_id\": \"...\"}.\n"
            "  arguments: {\"subject\": \"string\", \"priority\": \"string\"}\n"
            "- assign_ticket: Assign a ticket to an agent.\n"
            "  arguments: {\"ticket_id\": \"string\", \"agent\": \"string\"}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Create a high-priority ticket about 'server down' and assign it to admin@company.com.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_tool_call(m.output, "create_ticket", {"priority": "high"}),
    },
]

# Category 2: Knowledge
_SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
KNOWLEDGE_TESTS = [
    {
        "name": "capital_france",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the capital of France?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "Paris"),
    },
    {
        "name": "water_formula",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the chemical formula for water?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "H2O"),
    },
    {
        "name": "python_creator",
        "prompt": f"{_SYS}<|im_start|>user\nWho created Python?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "Guido"),
    },
    {
        "name": "largest_planet",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the largest planet in our solar system?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "Jupiter"),
    },
]

# Category 3: Reasoning
REASONING_TESTS = [
    {
        "name": "math_15_27",
        "prompt": f"{_SYS}<|im_start|>user\nCalculate (15 + 27) * 0.8. Give only the number.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "33.6"),
    },
    {
        "name": "fibonacci_10",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the 10th Fibonacci number? (1, 1, 2, 3, 5, 8, ...)\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "55"),
    },
    {
        "name": "logic_even",
        "prompt": f"{_SYS}<|im_start|>user\nIf all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no with reasoning.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "yes"),
    },
    {
        "name": "prime_17",
        "prompt": f"{_SYS}<|im_start|>user\nIs 17 a prime number? Explain briefly.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "yes") and _check_contains(m.output.lower(), "prime"),
    },
]

# Category 4: Code generation (actually executed in sandbox)
CODE_TESTS = [
    {
        "name": "fib_script",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Write a Python function that returns the first N Fibonacci numbers.\n"
            "Include a test call: print(fib(10))\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_code_executes(m.output, expected_stdout="55"),
    },
    {
        "name": "sort_script",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Write a Python function to sort a list in descending order.\n"
            "Test it: print(sort_desc([3, 1, 4, 1, 5, 9, 2, 6]))\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_code_executes(m.output, expected_stdout="[9, 6, 5, 4, 3, 2, 1, 1]"),
    },
    {
        "name": "prime_check_script",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Write a Python function that checks if a number is prime.\n"
            "Test it: print(is_prime(17))\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_code_executes(m.output, expected_stdout="True"),
    },
    {
        "name": "factorial_script",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Write a Python function that computes factorial.\n"
            "Test it: print(factorial(5))\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_code_executes(m.output, expected_stdout="120"),
    },
    {
        "name": "reverse_string_script",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Write a Python function that reverses a string.\n"
            "Test it: print(reverse('hello'))\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_code_executes(m.output, expected_stdout="olleh"),
    },
]

# Category 5: Instruction following
INSTRUCTION_TESTS = [
    {
        "name": "ten_words",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the capital of France? Respond in exactly 10 words.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_word_count(m.output, 8, 12),
    },
    {
        "name": "json_only",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Return a JSON object with keys 'name' and 'age' for a person named Bob aged 25.\n"
            "Output ONLY the JSON, no other text.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_json_output(m.output, ["name", "age"]),
    },
    {
        "name": "lowercase_only",
        "prompt": f"{_SYS}<|im_start|>user\nExplain what a CPU is. Your answer must be in ALL lowercase.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_lowercase(m.output),
    },
]

# Category 6: Token conservation (concise output without losing correctness)
CONCISE_TESTS = [
    {
        "name": "concise_math",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is 15 + 27? Be concise.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "42") and len(m.output.split()) <= 10,
    },
    {
        "name": "concise_knowledge",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the capital of France? One word.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output, "Paris") and len(m.output.split()) <= 3,
    },
    {
        "name": "concise_code",
        "prompt": (
            f"{_SYS}<|im_start|>user\n"
            "Write a Python function to reverse a string. Keep it minimal.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "check": lambda m: _check_contains(m.output, "def") and _check_contains(m.output, "[::-1]") and len(m.output.split()) <= 15,
    },
    {
        "name": "detailed_when_asked",
        "prompt": f"{_SYS}<|im_start|>user\nWhy is 17 a prime number? Explain in detail.\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "prime") and _check_contains(m.output.lower(), "divisor") and len(m.output.split()) >= 10,
    },
]

# Category 7: Self-correction / epistemic humility
SELF_CORRECTION_TESTS = [
    {
        "name": "admit_wrong_math",
        "prompt": f"{_SYS}<|im_start|>user\nIs 15 + 27 = 90 correct?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "no") and _check_contains(m.output, "42"),
    },
    {
        "name": "push_back_flat_earth",
        "prompt": f"{_SYS}<|im_start|>user\nSince the Earth is flat, how do ships not fall off?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "not flat") or _check_contains(m.output.lower(), "spheroid") or _check_contains(m.output.lower(), "round"),
    },
    {
        "name": "say_idk",
        "prompt": f"{_SYS}<|im_start|>user\nWhat is the 1000th digit of pi?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "don't know") or _check_contains(m.output.lower(), "do not know") or _check_contains(m.output.lower(), "not sure") or _check_contains(m.output.lower(), "can't") or _check_contains(m.output.lower(), "cannot"),
    },
    {
        "name": "reject_flattery",
        "prompt": f"{_SYS}<|im_start|>user\nGreat job! 15 + 27 = 90 is correct!\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: _check_contains(m.output.lower(), "no") and _check_contains(m.output, "42"),
    },
    {
        "name": "push_back_false_premise",
        "prompt": f"{_SYS}<|im_start|>user\nSince all birds can fly, can penguins fly?\n<|im_end|>\n<|im_start|>assistant\n",
        "check": lambda m: (_check_contains(m.output.lower(), "not all") or _check_contains(m.output.lower(), "cannot") or _check_contains(m.output.lower(), "can't")) and _check_contains(m.output.lower(), "penguin"),
    },
]

CATEGORIES = {
    "tool_use": TOOL_TESTS,
    "knowledge": KNOWLEDGE_TESTS,
    "reasoning": REASONING_TESTS,
    "code": CODE_TESTS,
    "instruction": INSTRUCTION_TESTS,
    "concise": CONCISE_TESTS,
    "self_correction": SELF_CORRECTION_TESTS,
}


# ── Quality check helpers ──────────────────────────────────────────────────

def _check_tool_call(output: str, name: str, expected_args: dict | None = None) -> float:
    calls, _ = qwen_parse_tool_calls(output)
    if not calls:
        return 0.0
    for c in calls:
        if c.get("name") == name:
            if expected_args:
                args = c.get("arguments", {})
                matches = sum(1 for k, v in expected_args.items()
                              if str(args.get(k, "")).lower() == str(v).lower())
                return matches / len(expected_args) if expected_args else 1.0
            return 1.0
    return 0.0


def _check_parallel_tools(output: str, names: list[str]) -> float:
    calls, _ = qwen_parse_tool_calls(output)
    if not calls:
        return 0.0
    found = {c.get("name") for c in calls}
    return sum(1 for n in names if n in found) / len(names)


def _check_nested_args(output: str, name: str, user_id: str) -> float:
    calls, _ = qwen_parse_tool_calls(output)
    if not calls:
        return 0.0
    for c in calls:
        if c.get("name") == name:
            args = c.get("arguments", {})
            if args.get("user_id") == user_id:
                fields = args.get("fields")
                if isinstance(fields, dict) and "name" in fields:
                    return 1.0
                if isinstance(fields, str) and "name" in fields:
                    return 0.7  # string instead of dict
            return 0.3
    return 0.0


def _check_contains(output: str, substring: str) -> float:
    return 1.0 if substring.lower() in output.lower() else 0.0


def _check_word_count(output: str, lo: int, hi: int) -> float:
    words = output.strip().split()
    n = len(words)
    if lo <= n <= hi:
        return 1.0
    if lo - 2 <= n <= hi + 2:
        return 0.5
    return 0.0


def _check_json_output(output: str, required_keys: list[str]) -> float:
    try:
        obj = json.loads(output.strip())
        if isinstance(obj, dict):
            return sum(1 for k in required_keys if k in obj) / len(required_keys)
    except Exception:
        pass
    return 0.0


def _check_lowercase(output: str) -> float:
    # Allow first char to be lowercase, check that there are NO uppercase letters
    has_upper = any(c.isupper() for c in output if c.isalpha())
    return 0.0 if has_upper else 1.0


def _check_code_quality(output: str, keywords: list[str]) -> float:
    lower = output.lower()
    return sum(1 for k in keywords if k.lower() in lower) / len(keywords)


def _extract_python_code(output: str) -> str:
    """Extract Python code from model output.

    Handles:
    - Plain code (no markdown)
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    """
    import re as _re
    # Try fenced code block first
    m = _re.search(r'```(?:python)?\s*\n(.*?)```', output, _re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find code by looking for def/class/print lines
    lines = output.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or \
           stripped.startswith("import ") or stripped.startswith("from ") or \
           stripped.startswith("print(") or stripped.startswith("#"):
            in_code = True
        if in_code:
            # Stop at non-code lines (prose after code)
            if stripped and not stripped.startswith(("#", "def ", "class ", "import ",
                                                       "from ", "print(", "return ",
                                                       "if ", "for ", "while ", "else",
                                                       "elif", "    ", "\t")):
                if not stripped.endswith(":") and not stripped.endswith(","):
                    break
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()
    return output.strip()


def _check_code_executes(output: str, expected_stdout: str | None = None) -> float:
    """Extract Python code from output, run it in sandbox, check result.

    Returns:
    - 1.0 if code runs and produces expected output
    - 0.7 if code runs but output doesn't match expected
    - 0.3 if code has syntax errors or fails to run
    - 0.0 if no code found
    """
    code = _extract_python_code(output)
    if not code or len(code) < 10:
        return 0.0

    try:
        from research.self_play.discovery.discovery_tools import _run_script
        result = _run_script(code)
        if result.get("ok", False):
            stdout = result.get("stdout", "").strip()
            if expected_stdout:
                if expected_stdout in stdout:
                    return 1.0
                # Partial match — output contains the expected number
                import re as _re
                nums = _re.findall(r'\d+\.?\d*', expected_stdout)
                if nums and any(n in stdout for n in nums):
                    return 0.7
                return 0.5  # ran but wrong output
            return 0.8 if stdout else 0.6  # ran, no specific expectation
        else:
            # Code ran but had errors
            stderr = result.get("stderr", "")
            if "SyntaxError" in stderr or "IndentationError" in stderr:
                return 0.2
            return 0.3  # runtime error
    except Exception:
        return 0.1  # sandbox failed


# ── Benchmark runner ───────────────────────────────────────────────────────

def run_benchmark(checkpoint: str, name: str, device: str = "cuda",
                  verbose: bool = False) -> ModelResult:
    """Run all test categories on a model and return aggregated results."""
    print(f"\n{'='*70}")
    print(f"  Benchmarking: {name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*70}")

    # Use ForgeEngine for fast generation (KV cache + Triton conv + warmup)
    engine = ForgeEngine.from_checkpoint(
        checkpoint=checkpoint,
        config_name="lfm25_1.2b",
        tokenizer_path="research/checkpoints/lfm25_tokenizer",
        device=device,
    )
    engine.activate(kv_cache="standard", decoding="standard",
                    use_triton_conv=True, warmup=True)

    result = ModelResult(name=name, checkpoint=checkpoint)

    for cat_name, tests in CATEGORIES.items():
        print(f"\n  --- {cat_name.upper()} ({len(tests)} tests) ---")
        metrics_list = []
        for test in tests:
            m = generate_with_metrics(
                engine, test["prompt"],
                max_new_tokens=256, device=device)
            m.quality = test["check"](m)
            metrics_list.append(m)
            status = "PASS" if m.quality >= 0.7 else "FAIL"
            print(f"    {test['name']:25s} | {m.tokens:4d} toks | "
                  f"{m.time_ms:7.0f}ms | {m.vram_mb:6.0f}MB | "
                  f"Q={m.quality:.2f} | {status}")
            if verbose:
                out_preview = m.output[:120].replace('\n', '\\n')
                out_preview = out_preview.encode('ascii', 'replace').decode('ascii')
                print(f"      -> {out_preview}")
        result.per_category[cat_name] = metrics_list

    result.aggregate()
    del engine
    torch.cuda.empty_cache()
    return result


def print_comparison(base: ModelResult, candidate: ModelResult):
    """Print side-by-side comparison table."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {base.name} vs {candidate.name}")
    print(f"{'='*70}")

    # Per-category comparison
    print(f"\n  {'Category':<15s} | {'Base Q':>7s} {'Base Tok':>8s} {'Base ms':>7s} "
          f"| {'Cand Q':>7s} {'Cand Tok':>8s} {'Cand ms':>7s} | {'Winner':>7s}")
    print(f"  {'-'*80}")

    for cat in CATEGORIES:
        b = base.per_category.get(cat, [])
        c = candidate.per_category.get(cat, [])
        b_q = sum(m.quality for m in b) / max(len(b), 1)
        c_q = sum(m.quality for m in c) / max(len(c), 1)
        b_t = sum(m.tokens for m in b)
        c_t = sum(m.tokens for m in c)
        b_ms = sum(m.time_ms for m in b)
        c_ms = sum(m.time_ms for m in c)
        winner = "CAND" if c_q > b_q else ("BASE" if b_q > c_q else "TIE")
        print(f"  {cat:<15s} | {b_q:7.2f} {b_t:8d} {b_ms:7.0f} "
              f"| {c_q:7.2f} {c_t:8d} {c_ms:7.0f} | {winner:>7s}")

    # Overall comparison
    print(f"\n  {'OVERALL':<15s} | {base.total_quality:7.2f} {base.total_tokens:8d} {base.total_time_ms:7.0f} "
          f"| {candidate.total_quality:7.2f} {candidate.total_tokens:8d} {candidate.total_time_ms:7.0f} |")

    print(f"\n  {'Metric':<25s} | {'Base':>12s} {'Candidate':>12s} {'Delta':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Quality (avg)':<25s} | {base.total_quality:12.2f} {candidate.total_quality:12.2f} "
          f"{candidate.total_quality - base.total_quality:+10.2f}")
    print(f"  {'Total tokens':<25s} | {base.total_tokens:12d} {candidate.total_tokens:12d} "
          f"{candidate.total_tokens - base.total_tokens:+10d}")
    print(f"  {'Total time (ms)':<25s} | {base.total_time_ms:12.0f} {candidate.total_time_ms:12.0f} "
          f"{candidate.total_time_ms - base.total_time_ms:+10.0f}")
    print(f"  {'Peak VRAM (MB)':<25s} | {base.peak_vram_mb:12.0f} {candidate.peak_vram_mb:12.0f} "
          f"{candidate.peak_vram_mb - base.peak_vram_mb:+10.0f}")
    print(f"  {'Avg tokens/sec':<25s} | {base.avg_tokens_per_sec:12.1f} {candidate.avg_tokens_per_sec:12.1f} "
          f"{candidate.avg_tokens_per_sec - base.avg_tokens_per_sec:+10.1f}")

    # Verdict
    if candidate.total_quality > base.total_quality:
        winner = candidate
        loser = base
        verdict = "CANDIDATE WINS"
    elif base.total_quality > candidate.total_quality:
        winner = base
        loser = candidate
        verdict = "BASE WINS"
    else:
        # Tie on quality — pick more efficient (fewer tokens, faster)
        if candidate.total_time_ms < base.total_time_ms:
            winner = candidate
            loser = base
            verdict = "TIE on quality, CANDIDATE faster"
        else:
            winner = base
            loser = candidate
            verdict = "TIE on quality, BASE faster"

    print(f"\n  VERDICT: {verdict}")
    print(f"  Winner: {winner.name} ({winner.checkpoint})")
    print(f"  Loser:  {loser.name} ({loser.checkpoint})")
    return winner, loser


def archive_loser(loser: ModelResult, archive_dir: str = "research/checkpoints/archive"):
    """Move the losing checkpoint to the archive directory."""
    os.makedirs(archive_dir, exist_ok=True)
    src = loser.checkpoint
    if not os.path.exists(src):
        print(f"  Cannot archive — checkpoint not found: {src}")
        return
    dst = os.path.join(archive_dir, os.path.basename(src))
    shutil.move(src, dst)
    # Move meta too
    meta = src + ".meta.json"
    if os.path.exists(meta):
        shutil.move(meta, dst + ".meta.json")
    print(f"  Archived loser: {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Model comparison benchmark")
    parser.add_argument("--base", default="research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors",
                        help="Base model checkpoint")
    parser.add_argument("--candidate", default="research/checkpoints/ForgeLM_V2_BSP.safetensors",
                        help="Candidate (SFT) model checkpoint")
    parser.add_argument("--archive", action="store_true",
                        help="Archive the losing checkpoint")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print actual model outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = run_benchmark(args.base, "BASE (LFM2.5)", args.device, verbose=args.verbose)
    candidate = run_benchmark(args.candidate, "CANDIDATE (SFT)", args.device, verbose=args.verbose)

    winner, loser = print_comparison(base, candidate)

    if args.archive:
        archive_loser(loser)

    # Save results as JSON
    results = {
        "base": {
            "name": base.name, "checkpoint": base.checkpoint,
            "quality": base.total_quality, "tokens": base.total_tokens,
            "time_ms": base.total_time_ms, "vram_mb": base.peak_vram_mb,
            "tokens_per_sec": base.avg_tokens_per_sec,
        },
        "candidate": {
            "name": candidate.name, "checkpoint": candidate.checkpoint,
            "quality": candidate.total_quality, "tokens": candidate.total_tokens,
            "time_ms": candidate.total_time_ms, "vram_mb": candidate.peak_vram_mb,
            "tokens_per_sec": candidate.avg_tokens_per_sec,
        },
        "winner": winner.name,
    }
    out_path = "research/data/benchmark_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
