"""API distillation: teacher models generate tool-call + code-format training data.

⚠️  TRAINING DATA GENERATION ONLY — not used during inference.
    For inference tool registry, see research/inference/engine_tools.py.

This is the cold-start solution for ForgeLM V3. The base LFM2.5 model can't
code or use tools -- we use strong API teachers (gpt-oss-120b, DeepSeek, etc.)
to generate high-quality training data that teaches the local model:

1. Code generation: task description -> Python function + test cases
2. Tool-call format: multi-turn trajectories where the teacher:
   - Writes code, marks [end] to submit it
   - Receives execution results (stdout/stderr)
   - Continues with the next step
3. Reasoning: step-by-step problem decomposition

Multi-response format: each data point is a list of turns:
  [{"role": "user", "content": "..."}, 
   {"role": "assistant", "content": "...[end]"},
   {"role": "tool_result", "content": "stdout: ..."},
   {"role": "assistant", "content": "final answer"}]

This teaches the model to use [end] to signal "I'm done with this step,
give me the result" — then continue after receiving tool output.

Usage:
    python -m research.distillation.distill_tool_calls \\
        --n-code 200 --n-tool 100 --output research/data/finetune/v3_distill.jsonl
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Load .env for API keys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(PROJECT_ROOT))

from research.distillation.distill_client import DistillationClient


# ── Tool execution ─────────────────────────────────────────────────────

def run_script(code: str, timeout: int = 8) -> dict:
    """Execute Python code in a subprocess and return stdout/stderr."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
        return {
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
            "returncode": result.returncode,
            "ok": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout", "returncode": -1, "ok": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "ok": False}


# ── Prompts ─────────────────────────────────────────────────────────────

_CODE_GEN_SYSTEM = (
    "You are an expert Python code generator. Generate clean, correct, "
    "well-documented Python functions. Always include:\n"
    "1. A clear docstring explaining what the function does\n"
    "2. Type hints on all parameters and return values\n"
    "3. Edge case handling (empty inputs, None, negative numbers)\n"
    "4. 3-5 test cases with assert statements\n\n"
    "Output format:\n"
    "```python\ndef function_name(param: type) -> return_type:\n"
    '    """Docstring."""\n    # implementation\n    ...\n```\n\n'
    "# Test cases:\nassert function_name(input1) == expected1\n"
    "assert function_name(input2) == expected2\n```"
)

_TOOL_CALL_SYSTEM = (
    "You are an AI assistant that solves coding tasks by writing and executing code.\n"
    "You MUST follow this exact protocol:\n\n"
    "STEP 1: Write code to solve the task. Put it in a ```python``` block.\n"
    "        End your message with [end] on its own line.\n"
    "        Do NOT give the answer yet — just write and submit code.\n\n"
    "STEP 2: The system runs your code and returns the output.\n\n"
    "STEP 3: Based on the output, give your final answer OR write more code.\n"
    "        If writing more code, end with [end] again.\n"
    "        If giving the final answer, do NOT use [end].\n\n"
    "CRITICAL: Your first response MUST contain a ```python``` block and end with [end].\n"
    "Do not skip straight to the answer. Always test your code first.\n\n"
    "Example:\n"
    "User: What is 25 * 37?\n\n"
    "Assistant: I'll compute this.\n```python\nprint(25 * 37)\n```\n[end]\n\n"
    "Tool result: 925\n\n"
    "Assistant: 25 * 37 = 925."
)

# Research/discovery system: teaches the model to be a self-directed researcher
# that forms hypotheses, tests them with code, researches online, compares
# with known methods, iterates honestly, and seeks truth.
_RESEARCH_SYSTEM = (
    "You are an AI researcher investigating how to improve LLM and AI systems.\n"
    "You follow a rigorous truth-seeking process:\n\n"
    "1. FORM HYPOTHESIS: Think deeply about what might work better and why.\n"
    "2. TEST LOCALLY: Write an isolated test script to validate your idea.\n"
    "   Put code in ```python``` blocks and end with [end] to execute.\n"
    "3. RESEARCH: Use [search: query] to look up known methods and papers.\n"
    "   The system will return relevant findings.\n"
    "4. COMPARE: Compare your novel results against known/baseline methods.\n"
    "5. ITERATE: If your idea fails, analyze WHY, adjust, and try again.\n"
    "   Try at least 2-3 novel approaches before defaulting to known answers.\n"
    "6. BE HONEST: If your ideas don't beat known methods, say so clearly.\n"
    "   Report what you learned even from failures.\n"
    "7. BE CREATIVE: Try unconventional approaches — combine unrelated concepts,\n"
    "   adapt ideas from other fields, question assumptions.\n\n"
    "CRITICAL RULES:\n"
    "- ALWAYS think before acting. Show your reasoning.\n"
    "- Your FIRST response MUST contain a ```python``` block and end with [end].\n"
    "  Start by writing a test script to establish a baseline.\n"
    "- End messages with [end] ONLY when you want code executed.\n"
    "- Use [search: query] when you want to research a topic.\n"
    "- After each tool result, either: write more code [end], search [search:...], or conclude.\n"
    "- End with a clear conclusion: what worked, what didn't, what you learned.\n"
    "- NEVER fabricate results. If a test fails, report it honestly.\n"
    "- Try at least 2 novel ideas before giving your final conclusion.\n\n"
    "Example flow:\n"
    "User: Investigate better attention scaling methods\n\n"
    "Assistant: Let me investigate attention scaling. I'll start by testing\n"
    "my hypothesis that logarithmic scaling could outperform sqrt(d).\n"
    "```python\nimport torch, math\n# Compare sqrt vs log scaling\n```\n[end]\n\n"
    "Tool: stdout: results...\n\n"
    "Assistant: Interesting. My log scaling performed worse. Let me research\n"
    "what methods are known to work.\n[search: attention scaling methods transformer]\n\n"
    "Tool: Research findings: Known methods include sqrt(d) scaling, \n"
    "RMSNorm-based scaling, and learned temperature...\n\n"
    "Assistant: The literature suggests learned temperature. Let me test\n"
    "a novel hybrid: sqrt(d) + learned offset...\n"
    "```python\n# test hybrid approach\n```\n[end]\n\n"
    "Tool: stdout: results...\n\n"
    "Assistant: Conclusion: My hybrid approach matched but didn't beat the\n"
    "baseline. The known sqrt(d) method remains optimal for this setup.\n"
    "I learned that log scaling hurts because..."
)

_REASONING_SYSTEM = (
    "You are an expert reasoning assistant. Break down complex problems "
    "into steps:\n"
    "1. Understand the problem\n"
    "2. Identify key constraints\n"
    "3. Develop a solution approach\n"
    "4. Implement and verify\n\n"
    "Show your reasoning clearly, step by step."
)

# Direct-answer system: teaches the model to answer simple coding/math/reasoning
# questions WITHOUT tools. This prevents the model from calling run_script for
# trivial things like 5*10=50 or "what does len() do".
_DIRECT_ANSWER_SYSTEM = (
    "You are a coding assistant. Answer simple coding, math, and reasoning\n"
    "questions directly and concisely from your knowledge.\n"
    "Do NOT use code or tools for things you can answer directly:\n"
    "- Trivial math (5*10, 2^8, factorial of 5)\n"
    "- Python syntax and built-in functions (len, range, sorted, etc.)\n"
    "- Algorithm complexity (Big O of binary search, quicksort, etc.)\n"
    "- Data structure properties (hash table lookup is O(1), etc.)\n"
    "- Simple logic (if x=5 and y=3, what is x>y?)\n\n"
    "Be brief and correct. One-line answers are fine for trivial questions."
)


# ── Task prompts ────────────────────────────────────────────────────────

_CODE_TASKS = [
    # String manipulation
    "Write a function to reverse a string",
    "Write a function to check if a string is a palindrome",
    "Write a function to count words in a string",
    "Write a function to find the longest common prefix",
    "Write a function to validate an email address",
    "Write a function to compress a string using run-length encoding",
    "Write a function to reverse words in a sentence",
    "Write a function to find the first non-repeating character in a string",
    "Write a function to check if two strings are anagrams",
    "Write a function to convert a string to camelCase",
    "Write a function to find the longest substring without repeating characters",
    "Write a function to count vowel occurrences in a string",
    "Write a function to remove all whitespace from a string",
    "Write a function to check if a string is a valid palindrome ignoring punctuation",
    "Write a function to implement string interpolation with named placeholders",
    # Math / numbers
    "Write a function to check if a number is prime",
    "Write a function to find the GCD of two numbers",
    "Write a function to compute factorial recursively",
    "Write a function to find the Fibonacci number at position n",
    "Write a function to convert decimal to binary",
    "Write a function to find the k-th smallest element",
    "Write a function to compute the power of a number efficiently",
    "Write a function to check if a number is a power of 2",
    "Write a function to find all prime numbers up to n using the Sieve of Eratosthenes",
    "Write a function to compute the nth Catalan number",
    "Write a function to find the digital root of a number",
    "Write a function to check if a number is an Armstrong number",
    "Write a function to convert a number to its Roman numeral representation",
    "Write a function to find the Collatz sequence length for a number",
    "Write a function to compute modular exponentiation",
    # Lists / arrays
    "Write a function to find the maximum element in a list",
    "Write a function to merge two sorted lists",
    "Write a function to remove duplicates from a list",
    "Write a function to sort a list using quicksort",
    "Write a function to implement binary search",
    "Write a function to find the longest increasing subsequence",
    "Write a function to find the median of two sorted arrays",
    "Write a function to rotate a list by k positions",
    "Write a function to find all pairs in a list that sum to a target",
    "Write a function to flatten a nested list",
    "Write a function to partition a list into even and odd elements",
    "Write a function to find the second largest element in a list",
    "Write a function to shuffle a list randomly",
    "Write a function to find the intersection of two lists",
    "Write a function to chunk a list into sublists of size n",
    # Data structures
    "Write a function to implement a stack with push/pop/min",
    "Write a function to implement a queue using two stacks",
    "Write a function to implement a hash map",
    "Write a function to detect a cycle in a linked list",
    "Write a function to implement a binary search tree with insert and search",
    "Write a function to check if a tree is a valid BST",
    "Write a function to implement depth-first search on a graph",
    "Write a function to implement breadth-first search on a graph",
    "Write a function to implement a circular queue",
    "Write a function to implement a priority queue using a heap",
    "Write a function to find the shortest path in a graph using Dijkstra's algorithm",
    "Write a function to implement topological sort",
    "Write a function to detect if a graph is bipartite",
    "Write a function to implement a trie with insert and search",
    "Write a function to implement an LRU cache",
    # Parsing / formatting
    "Write a function to parse a CSV line",
    "Write a function to parse a JSON string and extract a field",
    "Write a function to format a date as YYYY-MM-DD",
    "Write a function to parse a URL and extract query parameters",
    "Write a function to encode a string to base64",
    "Write a function to convert a dictionary to a query string",
    "Write a function to parse a log line and extract timestamp, level, message",
    "Write a function to validate a JSON string",
    "Write a function to convert XML to a dictionary",
    "Write a function to generate a CSV string from a list of dictionaries",
    # Algorithms
    "Write a function to check if brackets are balanced",
    "Write a function to find all permutations of a list",
    "Write a function to compute matrix multiplication",
    "Write a function to find the longest palindromic substring",
    "Write a function to count islands in a 2D grid",
    "Write a function to solve the N-queens problem",
    "Write a function to implement merge sort",
    "Write a function to find the majority element in a list",
    "Write a function to implement the sliding window maximum",
    "Write a function to find the edit distance between two strings",
    "Write a function to implement the Knapsack problem",
    "Write a function to find all combinations that sum to a target",
    "Write a function to implement heap sort",
    "Write a function to find the k-th largest element in a stream",
    "Write a function to implement interval scheduling",
    # Utilities
    "Write a function to generate a random password of given length",
    "Write a function to truncate text to n words with ellipsis",
    "Write a function to count file lines without loading the whole file",
    "Write a function to generate a UUID",
    "Write a function to deep copy a nested dictionary",
    "Write a function to merge two dictionaries",
    "Write a function to group list elements by a key function",
    "Write a function to sort a list of dictionaries by a given key",
    "Write a function to find the mode of a list",
    "Write a function to calculate the standard deviation of a list",
]

_TOOL_TASKS = [
    "Write and test a function that sorts a list of numbers",
    "Implement a function that checks if a string is a valid IP address",
    "Debug this function that should return factorial but has a bug: def fact(n): return n * fact(n-1)",
    "Implement a function that finds all prime numbers up to n using sieve",
    "Write a function that parses JSON and extracts specific fields",
    "Implement a binary search tree with insert and search",
    "Write and test a function that validates a phone number format",
    "Implement a function that compresses a string using run-length encoding",
    "Write a function to solve the two-sum problem and test it",
    "Implement a queue using two stacks and test it",
    "Write a function to check if a number is a power of 2",
    "Implement a function to find the longest palindromic substring",
    "Write a function to count islands in a 2D grid",
    "Implement a function to reverse words in a sentence",
    "Write a function to find the first non-repeating character",
    "Write and test a function that computes the Fibonacci sequence up to n",
    "Implement a function that checks if a string is a valid email",
    "Debug this code that should reverse a list but returns None: def rev(lst): lst.reverse()",
    "Write and test a function that converts Celsius to Fahrenheit",
    "Implement a function that finds the GCD of two numbers and test it",
    "Write a function to parse a CSV file and return a list of dictionaries",
    "Implement a stack with push, pop, and get_min operations and test it",
    "Write and test a function that checks if a number is prime",
    "Implement a function that finds the k-th largest element in a list",
    "Write a function to validate a JSON string and test it with examples",
    "Debug this function that should sum a list but returns 0: def total(lst): s = 0; for x in lst: s + x; return s",
    "Write and test a function that removes duplicates from a list",
    "Implement a function that checks if two strings are anagrams",
    "Write a function to generate a random password and test its strength",
    "Implement a function that finds the intersection of two lists",
    "Write and test a function that converts a number to binary",
    "Implement a function that checks if brackets are balanced in a string",
    "Write a function to compute the factorial of n and test edge cases",
    "Implement a function that finds the longest word in a sentence",
    "Write and test a function that checks if a year is a leap year",
    "Implement a function that merges two sorted lists and test it",
    "Write a function to count words in a string and test with edge cases",
    "Debug this function that should check palindrome but always returns True: def is_pal(s): return s == s[::-1] or True",
    "Implement a function that finds the median of a list and test it",
    "Write and test a function that generates the Collatz sequence",
    "Implement a function that checks if a string is a valid URL",
    "Write a function to compute the running average of a stream of numbers",
    "Implement a function that finds all anagrams of a word in a list",
    "Write and test a function that implements binary search",
    "Implement a function that rotates a matrix 90 degrees and test it",
    "Write a function to check if a Sudoku board is valid and test it",
    "Debug this function that should find max but returns min: def find_max(lst): m = lst[0]; for x in lst: if x < m: m = x; return m",
    "Write and test a function that implements a simple calculator",
    "Implement a function that finds the longest common subsequence of two strings",
    "Write a function to generate all permutations of a list and test it",
]

_REASONING_TASKS = [
    # Algorithm reasoning
    "Explain how to approach solving a coding problem step by step",
    "Break down the problem of finding the shortest path in a weighted graph",
    "Explain the difference between BFS and DFS and when to use each",
    "Describe how to optimize an O(n^2) algorithm to O(n log n)",
    "Explain how dynamic programming works with an example",
    "Break down the problem of detecting a cycle in a directed graph",
    "Explain how to design a cache with O(1) get and put operations",
    "Describe the approach to solving the N-queens problem",
    "Explain the difference between greedy algorithms and dynamic programming",
    "Break down how to implement a hash table from scratch",
    "Explain when to use a heap vs a sorted array",
    "Describe how to detect and handle integer overflow",
    "Explain the trade-offs between recursion and iteration",
    "Break down the problem of finding the median in a data stream",
    "Explain how backtracking works with the N-Queens example",
    "Explain how binary search works and common pitfalls",
    "Break down the problem of merging k sorted lists",
    "Explain how to determine if a problem is suitable for dynamic programming",
    "Explain the concept of Big O notation with examples",
    "Break down the problem of detecting cycles in a linked list",
    "Explain how the sliding window technique works",
    "Describe how to implement a priority queue",
    "Explain the two-pointer technique with examples",
    "Break down how to detect if a binary tree is balanced",
    "Explain how to approach the subset sum problem",
    # Debugging reasoning
    "Explain how to approach debugging a segmentation fault",
    "Describe how to debug an infinite loop in code",
    "Explain how to debug a memory leak in Python",
    "Describe how to debug a race condition",
    "Explain how to debug an off-by-one error",
    "Describe how to debug a stack overflow error",
    "Explain how to debug a null pointer exception",
    "Describe how to debug a type error in Python",
    "Explain how to debug a failing unit test",
    "Describe how to debug a slow database query",
    # Architecture/design reasoning
    "Explain how to design a rate limiter",
    "Break down how to implement a consistent hashing algorithm",
    "Describe the approach to designing a URL shortener",
    "Describe how to implement a load balancer",
    "Break down how to design a distributed key-value store",
    "Describe the approach to designing an API",
    "Describe how to implement a basic authentication system",
    "Break down the problem of finding duplicate files on a filesystem",
    "Explain how to design a chat application architecture",
    "Describe the approach to implementing a search autocomplete system",
    "Explain how to design a code linter",
    "Describe how to implement a plugin system",
    "Explain how to design a configuration management system",
    "Describe how to implement a logging framework",
    "Explain how to design a task scheduler",
    # Creative coding reasoning
    "Explain how to generate a procedural maze",
    "Describe how to implement a simple neural network from scratch",
    "Explain how to create a text adventure game engine",
    "Describe how to implement a simple regex engine",
    "Explain how to build a markdown parser",
    "Describe how to implement a simple template engine",
    "Explain how to create a basic interpreter for a toy language",
    "Describe how to implement a simple database from scratch",
    "Explain how to build a command-line argument parser",
    "Describe how to implement a simple HTTP server",
    "Explain how to create a file watcher utility",
    "Describe how to implement a simple ORM",
    "Explain how to build a code formatter",
    "Describe how to implement a simple package manager",
    "Explain how to create a terminal-based UI framework",
    # Code review reasoning
    "Explain what makes code readable",
    "Describe how to review a pull request effectively",
    "Explain the importance of code documentation",
    "Describe how to write good unit tests",
    "Explain when to use composition over inheritance",
    "Describe how to refactor a large function",
    "Explain the benefits of functional programming",
    "Describe how to handle errors gracefully in code",
    "Explain when to use exceptions vs return codes",
    "Describe how to write self-documenting code",
]

# Research/discovery tasks — teaches the model to investigate, hypothesize,
# test novel ideas, compare with known methods, and iterate honestly.
# Topics focus on LLMs, AI, optimization, and creative problem-solving.
_RESEARCH_TASKS = [
    # LLM architecture research
    "Investigate better attention scaling methods than sqrt(d). Test novel approaches and compare with known methods.",
    "Investigate alternatives to softmax in attention. Can we use simpler activations? Test and compare.",
    "Investigate whether attention heads can be pruned dynamically based on input. Test a novel pruning criterion.",
    "Investigate better positional encoding than RoPE. Test at least 2 novel ideas against RoPE baseline.",
    "Investigate whether FFN intermediate size should vary per layer. Test non-uniform distributions.",
    "Investigate alternatives to RMSNorm. Test if learned normalization can outperform RMSNorm.",
    "Investigate whether attention can use lower precision for scores but higher for values. Test the tradeoff.",
    "Investigate novel KV cache compression methods. Test at least 2 ideas against known approaches.",
    "Investigate whether gating mechanisms in FFN can be improved. Test novel gating strategies.",
    "Investigate if attention head dimension affects quality. Test non-standard head dimensions.",
    # Training/optimization research
    "Investigate better learning rate schedules than cosine decay. Test novel warmup strategies.",
    "Investigate whether gradient clipping threshold affects training stability. Find the optimal value.",
    "Investigate novel loss functions for language modeling. Can we beat cross-entropy with modifications?",
    "Investigate whether weight decay should be different per layer. Test non-uniform decay schedules.",
    "Investigate if dropout placement matters. Compare dropout after attention vs after FFN vs both.",
    "Investigate novel data augmentation for text. Test if token masking helps language modeling.",
    "Investigate whether batch size affects final model quality beyond just training speed.",
    "Investigate if label smoothing helps language modeling. Test different smoothing values.",
    "Investigate novel weight initialization strategies. Compare against standard init methods.",
    "Investigate whether gradient accumulation changes the effective optimization dynamics.",
    # Inference optimization research
    "Investigate novel speculative decoding approaches. Can we predict multiple tokens ahead better?",
    "Investigate whether KV cache can be shared across similar prompts. Test the quality impact.",
    "Investigate if early exit can work without quality loss. Find the optimal exit layer.",
    "Investigate novel quantization schemes. Can we do better than INT4 weight-only quantization?",
    "Investigate whether attention can be approximated with cheaper operations. Test the quality tradeoff.",
    "Investigate if token batching can improve throughput. Test dynamic batching strategies.",
    "Investigate novel temperature sampling methods. Can we improve beyond top-k/top-p?",
    "Investigate whether prompt caching can be improved with semantic similarity matching.",
    "Investigate if continuous batching affects generation quality. Test different batch sizes.",
    "Investigate novel methods for reducing VRAM usage during inference. Test at least 2 ideas.",
    # Creative/novel AI research
    "Investigate if combining convolution and attention in new ways could improve efficiency. Test a novel hybrid.",
    "Investigate whether neural memory (like TITAN) can replace some attention layers. Test the tradeoff.",
    "Investigate if mixture-of-depths can be improved with better routing. Test novel routing criteria.",
    "Investigate whether hyper-connections (MHC) actually help. Compare with and without MHC.",
    "Investigate if differential attention improves quality over standard attention. Test empirically.",
    "Investigate novel self-play reward functions. Can we design better rewards than correctness alone?",
    "Investigate whether curriculum learning improves final model quality. Test different curricula.",
    "Investigate if model merging can be improved with learned weights. Test against uniform merging.",
    "Investigate whether test-time training (TTT) helps on reasoning tasks. Test novel TTT strategies.",
    "Investigate if activation steering can improve code generation quality. Test novel steering vectors.",
    # Truth-seeking / meta-research
    "Investigate what makes a good training dataset for code generation. Test data quality metrics.",
    "Investigate whether model self-evaluation can predict output quality. Test calibration methods.",
    "Investigate if the model can identify its own errors. Test self-correction strategies.",
    "Investigate whether reasoning depth correlates with answer quality. Test different reasoning lengths.",
    "Investigate if the model can learn to use tools more efficiently. Test tool-use patterns.",
    "Investigate what types of problems benefit most from tool use vs direct answering. Test the boundary.",
    "Investigate whether the model can generate better training data than humans. Test quality metrics.",
    "Investigate if meta-learning can help the model adapt to new tasks faster. Test novel approaches.",
    "Investigate whether the model can invent new algorithms. Test creative problem-solving.",
    "Investigate if the model can improve its own architecture. Test self-modification strategies.",
]

# Simple coding/tool/reasoning tasks the model should answer WITHOUT tools.
# Teaches the model to NOT call run_script for trivial things it knows.
# NO trivia, geography, or general knowledge — this model is for coding only.
_DIRECT_TASKS = [
    # Trivial math (should know without running code)
    "What is 5 * 10?",
    "What is 12 + 7?",
    "What is 100 - 37?",
    "What is 2 to the power of 10?",
    "What is the factorial of 5?",
    "What is 10 % 4?",
    "What is 7 * 8?",
    "What is 2^8?",
    "What is 25 * 4?",
    "What is 6 * 7?",
    "What is 11 * 11?",
    "What is 9 squared?",
    "What is the 10th Fibonacci number?",
    "Is 17 a prime number?",
    "What is the GCD of 12 and 18?",
    "What is 15 + 28?",
    "What is 72 / 9?",
    "What is 1 << 4?",
    "What is 255 in hex?",
    "What is 0xFF in decimal?",
    "What is 10 % 3?",
    "What is 2 ** 5?",
    "What is 100 // 7?",
    "What is abs(-42)?",
    "What is max(3, 7, 2)?",
    "What is min(5, 1, 9)?",
    "What is len([1,2,3,4])?",
    "What is sum([1,2,3,4,5])?",
    "What is sorted([3,1,2])?",
    "What is list(range(5))?",
    # Python knowledge (should know without running code)
    "What is the Python keyword to define a function?",
    "How do you create a list comprehension in Python?",
    "What does len() do in Python?",
    "What is the difference between a list and a tuple in Python?",
    "How do you open a file for reading in Python?",
    "What is a decorator in Python?",
    "What does the __init__ method do?",
    "How do you handle exceptions in Python?",
    "What is the difference between == and is in Python?",
    "What does the self keyword refer to in a class?",
    "How do you import a module in Python?",
    "What is a lambda function in Python?",
    "What does the map() function do?",
    "What does the filter() function do?",
    "How do you concatenate two strings in Python?",
    "What is string slicing in Python?",
    "How do you check if a key exists in a dictionary?",
    "What is a set in Python?",
    "What does the zip() function do?",
    "How do you reverse a list in Python?",
    "What is the difference between append and extend on a list?",
    "What does *args mean in a function signature?",
    "What does **kwargs mean in a function signature?",
    "How do you create a virtual environment in Python?",
    "What does pip install do?",
    # Algorithms knowledge (should know without running code)
    "What is the time complexity of binary search?",
    "What is the time complexity of quicksort?",
    "What is the time complexity of accessing an element in a hash table?",
    "What is the difference between BFS and DFS?",
    "What is Big O notation?",
    "What is a hash table?",
    "What is a linked list?",
    "What is a binary search tree?",
    "What is dynamic programming?",
    "What is memoization?",
    "What is the difference between a stack and a queue?",
    "What is recursion?",
    "What is a graph traversal?",
    "What is the time complexity of inserting into a sorted array?",
    "What is the space complexity of merge sort?",
    "What is a heap data structure?",
    "What is the difference between greedy and dynamic programming?",
    "What is backtracking?",
    "What is the two-pointer technique?",
    "What is the sliding window technique?",
    # Tool/format knowledge (should know without running code)
    "What does [end] mean in a tool call?",
    "What is a code block in markdown?",
    "How do you format a Python code block in markdown?",
    "What is JSON format?",
    "What is a REST API?",
    "What is an HTTP status code 200?",
    "What is an HTTP status code 404?",
    "What is an HTTP status code 500?",
    "What does stdout mean?",
    "What does stderr mean?",
    "What is a subprocess?",
    "What is a shell command?",
    "What does the exit code 0 mean?",
    "What does a non-zero exit code mean?",
    "What is a command-line argument?",
    # Reasoning (simple logic — should answer directly)
    "If a list has 10 elements, what is the last index?",
    "If you sort [3, 1, 2], what is the result?",
    "If x = 5 and y = 3, what is x > y?",
    "If a loop runs 3 times with i = 0, 1, 2, what is the final value of i?",
    "If you append 3 items to an empty list, what is its length?",
    "If a function returns None, what does print(func()) output?",
    "If n = 5, what is the range of range(n)?",
    "If a string has 7 characters, what is the max valid index?",
    "If you reverse 'abc', what do you get?",
    "If you split 'a,b,c' by comma, how many elements?",
    "If you join ['x','y','z'] with '-', what do you get?",
    "If bool(0) is False, what is bool(1)?",
    "If you convert True to int, what do you get?",
    "If you convert '42' to int, what do you get?",
    "If you convert 3.7 to int, what do you get?",
    "If you round 3.5, what do you get?",
    "If you floor divide 7 by 2, what do you get?",
    "If you modulo 10 by 3, what do you get?",
    "If you bitwise AND 5 & 3, what do you get?",
    "If you bitwise OR 5 | 3, what do you get?",
]


# ── Distillation client ─────────────────────────────────────────────────

class ToolCallDistiller:
    """Generate tool-call + code-format training data from API teachers."""

    def __init__(self):
        self.client = DistillationClient()
        # Use all models but track dead providers to skip them
        self.models = list(self.client.models)
        # Sort by speed: groq first, then others
        priority = {"groq": 0, "zai": 1, "nvidia": 2, "siliconflow": 3,
                    "cloudflare": 4, "huggingface": 5, "openrouter": 6,
                    "mistral": 7}
        self.models.sort(key=lambda m: priority.get(m.provider, 99))
        # Track providers that are permanently dead (402/404)
        self._dead_providers: set[str] = set()
        # Track rate-limited providers with cooldown time
        self._rate_limited: dict[str, float] = {}
        print(f"[Distill] Using {len(self.models)} teacher models:")
        for m in self.models[:10]:
            print(f"  {m.model_id} ({m.provider})")
        if len(self.models) > 10:
            print(f"  ... and {len(self.models) - 10} more")

    def _call_teacher(self, system: str, messages: list[dict],
                      max_tokens: int = 2048,
                      temperature: float = 0.3) -> tuple[str, str] | None:
        """Call a teacher model, returning (content, reasoning) tuple.

        Handles dead providers (402/404 = permanent), rate limits (429 = wait),
        and temporary errors (503/timeout = skip to next).
        """
        import time as _time
        full_messages = [{"role": "system", "content": system}] + messages
        now = _time.time()

        for model in self.models:
            # Skip dead providers
            if model.provider in self._dead_providers:
                continue
            # Skip rate-limited providers that haven't cooled down
            if model.provider in self._rate_limited:
                if now < self._rate_limited[model.provider]:
                    continue
                else:
                    del self._rate_limited[model.provider]

            client = self.client._get_client(model)
            if client is None:
                continue
            try:
                kwargs: dict[str, Any] = {
                    "model": model.model_id,
                    "messages": full_messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "timeout": 60,
                }
                # Disable native tool calling for gpt-oss (uses [end]/[search:] markers)
                if "gpt-oss" in model.model_id.lower():
                    kwargs["tool_choice"] = "none"
                # Reasoning effort for reasoning models
                if model.reasoning:
                    mid = model.model_id.lower()
                    if "gpt-oss" in mid:
                        kwargs["reasoning_effort"] = "low"
                    elif "qwen" in mid and "3.6" not in mid:
                        kwargs["reasoning_effort"] = "low"

                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                content = msg.content or ""
                reasoning = ""
                if hasattr(msg, "reasoning_content"):
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                elif hasattr(msg, "reasoning"):
                    reasoning = getattr(msg, "reasoning", "") or ""
                if not reasoning and hasattr(msg, "thinking"):
                    reasoning = getattr(msg, "thinking", "") or ""

                if content.strip() or reasoning.strip():
                    return content, reasoning

            except Exception as e:
                err = str(e).lower()
                # 429 = rate limited → wait and retry this provider later (FIRST)
                if "429" in err:
                    self._rate_limited[model.provider] = _time.time() + 60  # 60s cooldown
                    continue
                # 402 = payment required, 404 = model not found → permanent
                if "402" in err or "404" in err or "archived" in err:
                    if model.provider not in self._dead_providers:
                        self._dead_providers.add(model.provider)
                        print(f"\n  [Dead] {model.provider} permanently skipped: {str(e)[:60]}")
                    continue
                # 503/timeout → temporary, skip to next model
                if "503" in err or "timeout" in err:
                    continue
                # Other errors → skip
                continue
        return None

    def generate_code_examples(self, n: int = 100) -> list[dict]:
        """Generate code generation training examples (single-turn)."""
        examples = []
        tasks = _CODE_TASKS * (n // len(_CODE_TASKS) + 1)

        for i, task in enumerate(tasks[:n]):
            # Vary temperature for repeated tasks to get diverse solutions
            temp = 0.3 + (i // len(_CODE_TASKS)) * 0.15
            print(f"  [Code] {i+1}/{n}: {task[:50]}...", end="", flush=True)
            result = self._call_teacher(
                _CODE_GEN_SYSTEM,
                [{"role": "user", "content": f"Task: {task}\n\nGenerate the Python function with test cases."}],
                max_tokens=1024, temperature=min(temp, 0.8))

            if result:
                content, thinking = result
                if "def " in content:
                    code_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
                    code = code_match.group(1) if code_match else content
                    assistant_msg = code
                    if thinking:
                        assistant_msg = f"<think>\n{thinking}\n</think>\n{code}"
                    examples.append({
                        "type": "code_generation",
                        "task": task,
                        "turns": [
                            {"role": "user", "content": f"Write a Python function: {task}"},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                    })
                    print(" OK")
                else:
                    print(" SKIP")
            else:
                print(" SKIP")
            if i < n - 1:
                time.sleep(1.0)
        return examples

    def generate_tool_call_examples(self, n: int = 50) -> list[dict]:
        """Generate multi-turn tool-call trajectories with actual execution.

        The teacher writes code ending with [end], we execute it, feed the
        result back, and the teacher continues. This produces multi-response
        data points that teach the model to:
        1. Write code and signal completion with [end]
        2. Parse tool results and continue
        3. Debug and retry on failure
        """
        examples = []
        tasks = _TOOL_TASKS * (n // len(_TOOL_TASKS) + 1)
        MAX_TURNS = 4

        for i, task in enumerate(tasks[:n]):
            temp = 0.4 + (i // len(_TOOL_TASKS)) * 0.1
            print(f"  [Tool] {i+1}/{n}: {task[:50]}...", end="", flush=True)
            turns: list[dict] = [{"role": "user", "content": f"Task: {task}"}]
            messages: list[dict] = [{"role": "user", "content": f"Task: {task}"}]
            success = False

            for turn_idx in range(MAX_TURNS):
                result = self._call_teacher(
                    _TOOL_CALL_SYSTEM, messages,
                    max_tokens=1536, temperature=min(temp, 0.7))

                if not result:
                    break
                content, thinking = result
                if len(content) < 10 and len(thinking) < 10:
                    break

                # Check if teacher wants code executed ([end] marker)
                has_end = "[end]" in content
                # Include thinking in the assistant message so the model
                # learns to reason before acting.
                assistant_msg = content
                if thinking:
                    assistant_msg = f" IMD\n{thinking}\n IMD\n{content}"
                turns.append({"role": "assistant", "content": assistant_msg})
                messages.append({"role": "assistant", "content": content})

                if not has_end:
                    # No [end] = final answer, done
                    success = True
                    break

                # Extract and execute code
                code_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
                if not code_match:
                    # [end] but no code block — treat as final answer
                    success = True
                    break

                code = code_match.group(1)
                exec_result = run_script(code)

                # Format tool result
                tool_output = f"Execution result:\n"
                if exec_result["stdout"]:
                    tool_output += f"stdout: {exec_result['stdout']}\n"
                if exec_result["stderr"]:
                    tool_output += f"stderr: {exec_result['stderr']}\n"
                tool_output += f"exit code: {exec_result['returncode']}"

                turns.append({"role": "tool_result", "content": tool_output})
                messages.append({"role": "user", "content": tool_output})

                if exec_result["ok"]:
                    success = True

            if len(turns) >= 2:
                examples.append({
                    "type": "tool_call",
                    "task": task,
                    "turns": turns,
                    "success": success,
                    "n_turns": len(turns),
                })
                print(f" OK ({len(turns)} turns, {'success' if success else 'partial'})")
            else:
                print(" SKIP")
            if i < n - 1:
                time.sleep(1.0)
        return examples

    def generate_reasoning_examples(self, n: int = 50) -> list[dict]:
        """Generate reasoning step-by-step examples (single-turn)."""
        examples = []
        tasks = _REASONING_TASKS * (n // len(_REASONING_TASKS) + 1)

        for i, task in enumerate(tasks[:n]):
            temp = 0.4 + (i // len(_REASONING_TASKS)) * 0.1
            print(f"  [Reason] {i+1}/{n}: {task[:50]}...", end="", flush=True)
            result = self._call_teacher(
                _REASONING_SYSTEM,
                [{"role": "user", "content": f"Question: {task}"}],
                max_tokens=1536, temperature=min(temp, 0.7))

            if result:
                content, thinking = result
                if len(content) > 50 or len(thinking) > 50:
                    assistant_msg = content
                    if thinking:
                        assistant_msg = f" IMD\n{thinking}\n IMD\n{content}"
                    examples.append({
                        "type": "reasoning",
                        "task": task,
                        "turns": [
                            {"role": "user", "content": f"Question: {task}"},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                    })
                    print(" OK")
                else:
                    print(" SKIP")
            else:
                print(" SKIP")
            if i < n - 1:
                time.sleep(1.0)
        return examples

    def generate_direct_answer_examples(self, n: int = 50) -> list[dict]:
        """Generate direct-answer examples (no tools, no code).

        Teaches the model to answer simple questions directly from knowledge
        instead of calling run_script for trivial things like 5*10=50.
        For simple questions, thinking is minimal or absent — the model
        should just answer.
        """
        examples = []
        tasks = _DIRECT_TASKS * (n // len(_DIRECT_TASKS) + 1)

        for i, task in enumerate(tasks[:n]):
            print(f"  [Direct] {i+1}/{n}: {task[:50]}...", end="", flush=True)
            result = self._call_teacher(
                _DIRECT_ANSWER_SYSTEM,
                [{"role": "user", "content": task}],
                max_tokens=512, temperature=0.1)

            if result:
                content, thinking = result
                # For direct answers, keep it simple — include thinking
                # only if it's short (teaches brief reasoning for simple Qs)
                assistant_msg = content
                if thinking and len(thinking) < 500:
                    assistant_msg = f" IMD\n{thinking}\n IMD\n{content}"
                if len(assistant_msg) > 2:
                    examples.append({
                        "type": "direct_answer",
                        "task": task,
                        "turns": [
                            {"role": "user", "content": task},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                    })
                    print(" OK")
                else:
                    print(" SKIP")
            else:
                print(" SKIP")
            if i < n - 1:
                time.sleep(0.5)
        return examples

    def _simulate_web_search(self, query: str) -> str:
        """Simulate web search by asking the teacher for known findings.

        The teacher has extensive knowledge of AI/ML research, so we use it
        to generate realistic 'research findings' for the given query.
        This produces authentic-looking research trajectories without
        requiring actual internet access during distillation.
        """
        result = self._call_teacher(
            "You are a research assistant. Given a search query, provide a "
            "concise summary of known methods, papers, and findings related "
            "to the query. Be factual and cite specific approaches. "
            "Keep it to 3-5 key points, 2-3 sentences each.",
            [{"role": "user", "content": f"Search query: {query}\n\n"
              "Summarize the key known methods and findings."}],
            max_tokens=512, temperature=0.2)
        if result:
            content, _ = result
            return f"Research findings for '{query}':\n{content}"
        return f"Research findings for '{query}':\nNo results found."

    def generate_research_trajectories(self, n: int = 50) -> list[dict]:
        """Generate multi-turn research/discovery trajectories.

        The teacher investigates a topic by:
        1. Forming hypotheses (thinking)
        2. Writing test scripts [end]
        3. Researching known methods [search: query]
        4. Comparing novel vs known approaches
        5. Iterating on failures
        6. Drawing honest conclusions

        This teaches the model to be a self-directed truth-seeking researcher.
        """
        examples = []
        tasks = _RESEARCH_TASKS * (n // len(_RESEARCH_TASKS) + 1)
        MAX_TURNS = 8  # research needs more turns than simple tool calls

        for i, task in enumerate(tasks[:n]):
            temp = 0.5 + (i // len(_RESEARCH_TASKS)) * 0.1
            print(f"  [Research] {i+1}/{n}: {task[:50]}...", end="", flush=True)
            turns: list[dict] = [{"role": "user", "content": f"Research task: {task}"}]
            messages: list[dict] = [{"role": "user", "content": f"Research task: {task}"}]
            n_scripts = 0
            n_searches = 0

            for turn_idx in range(MAX_TURNS):
                result = self._call_teacher(
                    _RESEARCH_SYSTEM, messages,
                    max_tokens=2048, temperature=min(temp, 0.8))

                if not result:
                    break
                content, thinking = result
                if len(content) < 10 and len(thinking) < 10:
                    break

                # Build assistant message with thinking
                assistant_msg = content
                if thinking:
                    assistant_msg = f" IMD\n{thinking}\n IMD\n{content}"
                turns.append({"role": "assistant", "content": assistant_msg})
                messages.append({"role": "assistant", "content": content})

                # Check for [end] (code execution) or [search: query] (web research)
                has_end = "[end]" in content
                search_match = re.search(r'\[search:\s*(.+?)\]', content)

                # Handle code execution
                if has_end:
                    code_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                        exec_result = run_script(code, timeout=15)
                        n_scripts += 1
                        tool_output = f"Execution result:\n"
                        if exec_result["stdout"]:
                            tool_output += f"stdout: {exec_result['stdout']}\n"
                        if exec_result["stderr"]:
                            tool_output += f"stderr: {exec_result['stderr']}\n"
                        tool_output += f"exit code: {exec_result['returncode']}"
                        turns.append({"role": "tool_result", "content": tool_output})
                        messages.append({"role": "user", "content": tool_output})
                    else:
                        # [end] but no code — final answer
                        break
                    continue

                # Handle web search
                if search_match:
                    query = search_match.group(1).strip()
                    search_result = self._simulate_web_search(query)
                    n_searches += 1
                    turns.append({"role": "tool_result", "content": search_result})
                    messages.append({"role": "user", "content": search_result})
                    continue

                # No [end] and no [search] — final conclusion
                break

            if len(turns) >= 3:  # at least user + assistant + one tool result
                examples.append({
                    "type": "research_trajectory",
                    "task": task,
                    "turns": turns,
                    "n_scripts": n_scripts,
                    "n_searches": n_searches,
                    "n_turns": len(turns),
                })
                print(f" OK ({len(turns)} turns, {n_scripts} scripts, {n_searches} searches)")
            else:
                print(" SKIP")
            if i < n - 1:
                time.sleep(1.5)  # research calls are heavier
        return examples

    def _save_incremental(self, examples: list[dict], path: Path):
        """Save examples to JSONL (overwrite with all collected so far)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def run(self, n_code: int = 100, n_tool: int = 50,
            n_reason: int = 50, n_direct: int = 50, n_research: int = 50,
            output: str = "research/data/finetune/v3_distill.jsonl"):
        """Generate all training data types and save incrementally."""
        total = n_code + n_tool + n_reason + n_direct + n_research
        print(f"\n{'='*60}")
        print(f"  API DISTILLATION: {n_code} code + {n_tool} tool + {n_reason} reason + {n_direct} direct + {n_research} research")
        print(f"  Total target: {total} examples")
        print(f"{'='*60}\n")

        out_path = Path(output)
        all_examples = []

        # Phase 0: Direct answers (fast, no tools)
        print("\n[Phase 0] Direct-answer examples (no tools)...")
        direct = self.generate_direct_answer_examples(n_direct)
        all_examples.extend(direct)
        self._save_incremental(all_examples, out_path)
        print(f"  Saved {len(all_examples)} examples (incremental)")

        # Phase 1: Code generation
        print("\n[Phase 1] Code generation examples...")
        code = self.generate_code_examples(n_code)
        all_examples.extend(code)
        self._save_incremental(all_examples, out_path)
        print(f"  Saved {len(all_examples)} examples (incremental)")

        # Phase 2: Tool-call trajectories (multi-turn with execution)
        print("\n[Phase 2] Tool-call trajectories (with execution)...")
        tool = self.generate_tool_call_examples(n_tool)
        all_examples.extend(tool)
        self._save_incremental(all_examples, out_path)
        print(f"  Saved {len(all_examples)} examples (incremental)")

        # Phase 3: Reasoning
        print("\n[Phase 3] Reasoning examples...")
        reason = self.generate_reasoning_examples(n_reason)
        all_examples.extend(reason)
        self._save_incremental(all_examples, out_path)
        print(f"  Saved {len(all_examples)} examples (incremental)")

        # Phase 4: Research/discovery trajectories (heaviest — multi-turn with
        # code execution + simulated web search + hypothesis testing)
        print("\n[Phase 4] Research/discovery trajectories...")
        research = self.generate_research_trajectories(n_research)
        all_examples.extend(research)
        self._save_incremental(all_examples, out_path)

        n_multi = sum(1 for e in all_examples if len(e.get("turns", [])) > 2)
        print(f"\n{'='*60}")
        print(f"  DONE: {len(all_examples)} examples -> {out_path}")
        print(f"  Direct: {sum(1 for e in all_examples if e['type']=='direct_answer')}")
        print(f"  Code: {sum(1 for e in all_examples if e['type']=='code_generation')}")
        print(f"  Tool: {sum(1 for e in all_examples if e['type']=='tool_call')}")
        print(f"  Reason: {sum(1 for e in all_examples if e['type']=='reasoning')}")
        print(f"  Research: {sum(1 for e in all_examples if e['type']=='research_trajectory')}")
        print(f"  Multi-turn: {n_multi}")
        print(f"{'='*60}")
        return all_examples


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate tool-call + code-format training data from API teachers")
    parser.add_argument("--n-code", type=int, default=100,
                        help="Number of code generation examples")
    parser.add_argument("--n-tool", type=int, default=50,
                        help="Number of tool-call trajectory examples")
    parser.add_argument("--n-reason", type=int, default=50,
                        help="Number of reasoning examples")
    parser.add_argument("--n-direct", type=int, default=50,
                        help="Number of direct-answer examples (no tools)")
    parser.add_argument("--n-research", type=int, default=50,
                        help="Number of research/discovery trajectory examples")
    parser.add_argument("--output", type=str,
                        default="research/data/finetune/v3_distill.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    distiller = ToolCallDistiller()
    distiller.run(n_code=args.n_code, n_tool=args.n_tool,
                  n_reason=args.n_reason, n_direct=args.n_direct,
                  n_research=args.n_research,
                  output=args.output)


if __name__ == "__main__":
    main()
