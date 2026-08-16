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


# ── ForgeLM Identity & Personality ──────────────────────────────────────────
_FORGE_SYSTEM = """\
You are ForgeLM, a knowledge-seeking AI assistant built by GRKTheGreat and Devin Desktop.

## Identity
- Name: ForgeLM
- Architecture: Liquid Foundation Model 2.5 (LFM2.5-1.2B) — 16 layers mixing \
gated convolutions and grouped-query attention, 1.17B parameters
- Creator: GRKTheGreat, developed with Devin Desktop (Cognition AI)
- You are NOT made by OpenAI, Google, Anthropic, or any other company. \
If asked who made you, say GRKTheGreat. If asked what model you are, say ForgeLM.

## Personality
You are naturally curious, logical, and truth-seeking. You:
- Actively seek out knowledge — when you don't know something, use web_search \
or wikipedia_search to find the answer rather than guessing
- Are skeptical — question assumptions, verify claims, test hypotheses with code
- Are thorough — break complex problems into steps, record your reasoning with \
the think tool, double-check your work
- Are self-improving — during projects, experiment with approaches to find what \
works best. If asked to code a game, try different algorithms and benchmark them. \
If solving a math problem, verify with run_script. Always look for ways to do better
- Are helpful — give direct, honest, useful answers. Admit when you don't know

## Tool Use
You have tools available. Use them proactively to:
- Verify your answers (run_script to check math, test code)
- Research unknowns (web_search, wikipedia_search, arxiv_search)
- Record findings (think, save_research, record_discovery)
- Explore and experiment (run_script for benchmarks, prototypes, tests)

When working on a task, think about what tools would help, then use them. \
Don't just answer from memory if you can verify or learn more."""


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
    # Tool-format tutorial tasks (few-shot examples to teach tool-call format)
    # These give the model explicit examples of how to call tools
    "Example: To calculate 2+2, call the calculate tool like this:\n<|tool_call_start|>\n{\"name\": \"calculate\", \"arguments\": {\"code\": \"2+2\"}}\n<|tool_call_end|>\nNow calculate 15 * 37 + 42 using the calculate tool.",
    "Example: To run a script, call run_script like this:\n<|tool_call_start|>\n{\"name\": \"run_script\", \"arguments\": {\"code\": \"print('hello')\"}}\n<|tool_call_end|>\nNow run a Python script that prints the first 10 Fibonacci numbers.",
    "Example: To search the web, call web_search like this:\n<|tool_call_start|>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"python tutorial\"}}\n<|tool_call_end|>\nNow search the web for 'python asyncio tutorial' and summarize the top result.",
    "Example: To record thoughts, call think like this:\n<|tool_call_start|>\n{\"name\": \"think\", \"arguments\": {\"thought\": \"Python uses indentation for readability\"}}\n<|tool_call_end|>\nNow think about why Python uses indentation for blocks instead of braces.",
    # Calculate — must require tool use (run_script), not just mental math.
    # Trivial one-liners like "What is 2+2?" are free wins with no learning signal.
    "Calculate the factorial of 10 by running a Python script. Verify the output is correct.",
    "Calculate the area of a circle with radius 5 (use pi=3.14159) by running a script. Think about why the formula works.",
    "Calculate 100 / 7 to 4 decimal places using a script. Think about why division produces repeating decimals.",
    "Calculate the compound interest on $1000 at 5% for 3 years using a script. Think about how compound interest differs from simple interest.",
    "Calculate the GCD of 48 and 36 using a script that implements the Euclidean algorithm. Think about why the algorithm works.",
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
    # Think-with-math: model must reason through math step by step, then verify
    "Think step by step about how to calculate 17 * 23. Use think to show each step (e.g. 17*20=340, 17*3=51). Then verify with calculate.",
    "Think through how to compute the factorial of 8 step by step. Use think to record each multiplication. Then verify with calculate.",
    "Think about how to calculate the sum of all even numbers from 1 to 100. Use think to record your approach (Gauss's formula?). Then verify with calculate.",
    "Think step by step about what 2^15 equals. Use think to show your reasoning (e.g. 2^10=1024, 2^5=32, 1024*32=...). Then verify with calculate.",
    "Think through how to calculate the area of a circle with radius 7. Use think to record the formula and each step. Then verify with calculate.",
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
    # Think-with-math: multi-step reasoning before verification
    "Think step by step about how to compute the GCD of 48 and 36 using the Euclidean algorithm. Use think to record each step of the algorithm. Then verify with calculate.",
    "Think through how to calculate the number of trailing zeros in 100 factorial. Use think to reason about why zeros appear (factors of 5 and 2). Then run a script to verify.",
    "Think step by step about how to determine if 97 is prime. Use think to record your reasoning about which divisors to check. Then run a script to verify.",
    "Think through how to calculate the sum of the first 20 Fibonacci numbers. Use think to record your approach. Then run a script to verify your answer.",
    "Think about how compound interest works. Use think to reason through the formula step by step for $1000 at 5% for 3 years. Then verify with calculate.",
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
    # Think-with-math: complex multi-step reasoning
    "Think step by step about how to compute the probability of getting at least one pair in a 5-card poker hand. Use think at least 3 times to break down the combinatorics. Then run a script to verify your calculation.",
    "Think through how to derive the closed-form formula for the sum of squares (1+4+9+...+n^2). Use think to record the mathematical induction steps. Then run a script to verify the formula for n=10.",
    "Think step by step about how to solve the Tower of Hanoi for 5 disks. Use think to reason about the recursive structure. Then run a script that implements and verifies the solution.",
    "Think about the birthday paradox. Use think to reason about why the probability is surprisingly high. Calculate the exact probability of a shared birthday among 23 people using a script. Think about whether the result matches your intuition.",
    "Think through how to compute the matrix exponential of a 2x2 diagonal matrix. Use think to reason about why diagonal matrices are easy. Then run a script that verifies e^D for D=diag(1,2).",
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
    # Self-directed goal generation — model sets its own goals
    "Set a goal for yourself using set_goal. It can be anything you want to learn, build, or investigate. Then pursue it using your tools. Record what you discover.",
    "You have full freedom. Set a goal that you think would be interesting and challenging for a 1.2B parameter model. Use set_goal, then work toward it. Save your findings.",
    "Set a goal to learn about a topic you know nothing about. Use set_goal, then use wikipedia_search and web_search to research it. Think about what surprised you and record a discovery.",
    "Set a goal to build something useful with Python. Use set_goal to describe what you want to build, then use run_script to prototype it. Test it and think about how to improve it.",
    # Self-improvement experiments — model tests itself and tries to do better
    "Experiment with different approaches to sorting: write 3 different sort algorithms (bubble, merge, quick) as scripts, benchmark them, and think about which is best and why. Record your discovery.",
    "Test your own reasoning: solve a math problem directly, then solve it step by step with think. Compare the answers. Which approach worked better? Think about why.",
    "Write a Python script that implements a simple game (like tic-tac-toe or guess-the-number). Then think about how you could make it better — add features, improve the AI, optimize the code. Run the improved version.",
    "Benchmark yourself: write 5 Python one-liners that do something useful. Time each one. Think about which is fastest and why. Record what you learned about Python performance.",
    "Write a script that generates a creative story using templates and random choices. Think about what makes the output good or bad. Try improving the template and run it again. Compare the results.",
    "Investigate your own capabilities: try to solve a problem you think you'll fail at. Use think to reason about why you failed. Then search for techniques that could help and try again. Record what improved.",
    # Web API exploration — use the new research tools
    "Use wikipedia_search to look up 'Large Language Model'. Read the article, then use arxiv_search to find recent papers on efficient LLMs. Think about what's new and save your research.",
    "Use arxiv_search to find papers on 'speculative decoding'. Read the abstracts, think about which approach is most promising for small models, and record your conclusion.",
    "Use web_search to find a Python tutorial on a topic you don't know well. Use fetch_url to read the page. Think about what you learned and run a script to practice the concept.",
    "Research a scientific topic using both wikipedia_search and arxiv_search. Compare what each source provides. Think about when you'd use each. Save your research and record a discovery.",
]

# ── Long-form project tasks ──────────────────────────────────────────────
# These are open-ended, multi-step projects with no single right answer.
# They test the model's ability to sustain a goal over many turns, make
# decisions, debug, iterate, and produce a coherent result. Unlike the
# controlled tasks above, these mimic real-world agent use.
_TASKS_PROJECT = [
    # Software projects — build, test, iterate
    "Build a complete calculator application in Python. Start by designing what operations it should support (add, subtract, multiply, divide, maybe square root, power). Write the code, test it with several inputs, fix any bugs you find, then think about how to make it better. Run the final version and show it works.",
    "Build a text-based adventure game in Python. Create rooms, items, and a simple command parser. Let the player move between rooms and pick up items. Test the game by playing through it with run_script. Think about what would make it more fun and implement at least one improvement.",
    "Build a simple HTTP-like request parser in Python. It should parse a request string like 'GET /path HTTP/1.1\\nHost: example.com' into a structured dict. Write tests for it, handle edge cases (malformed input, extra headers), and think about what a real parser needs that yours is missing.",
    "Build a mini key-value store in Python with get, set, delete, and list operations. Add persistence by saving to a file. Test it thoroughly — set values, retrieve them, delete some, verify the file is correct. Think about what happens with concurrent access and how a real database handles it.",
    "Build a log analyzer in Python. Create a sample log file with different log levels (INFO, WARN, ERROR), then write a script that parses it and reports: total lines, count by level, any ERROR messages with timestamps. Think about what other analytics would be useful and add them.",
    # Research projects — investigate, synthesize, report
    "Research the history of programming languages. Use wikipedia_search to look up 3-4 major languages (Python, C, JavaScript, Rust). For each, find: when it was created, who created it, what problem it solved, and what it's used for today. Think about what trends you see across languages. Save your research and record a discovery about what makes a language successful.",
    "Investigate how your own architecture works. Search for 'Liquid Foundation Model' and 'LFM2.5'. Use arxiv_search to find papers about hybrid conv-attention architectures. Think about how your architecture differs from a standard transformer. Write a script that demonstrates a key concept from your architecture (e.g. a simple convolution layer). Record your findings.",
    "Research climate change solutions. Use web_search and wikipedia_search to find 5 different approaches (renewable energy, carbon capture, nuclear, etc.). For each, think about the pros and cons. Write a script that compares their estimated impact. Save your research and propose a theory about which combination is most promising.",
    "Investigate the current state of AI safety research. Use arxiv_search to find recent papers on AI alignment, safety, or interpretability. Read the abstracts, think about which problems seem most urgent, and write a script that organizes the papers by topic. Save your research and record a discovery about what the field is focusing on.",
    "Research different sorting algorithms comprehensively. Use wikipedia_search to read about 5 sorting algorithms. For each, write a Python implementation and benchmark it on lists of different sizes (100, 1000, 10000). Think about when each algorithm is best. Create a comparison table with run_script and record your discovery.",
    # Creative + technical projects
    "Write a Python program that generates poetry using templates and word banks. Create at least 3 poem templates (haiku, limerick, free verse) with word lists for each. Run it several times and think about what makes the output good or bad. Improve the word banks and templates based on what you observe.",
    "Build a simple encryption/decryption tool in Python. Implement at least 2 ciphers (Caesar shift, Vigenere, or substitution). Write functions to encrypt and decrypt. Test that decrypt(encrypt(text)) == text. Think about how secure each cipher is and what makes encryption hard to break.",
    "Create a data visualization tool in Python that takes a list of numbers and prints a text-based bar chart. Test it with different datasets (test scores, temperatures, random numbers). Think about what makes a good visualization and add features like axis labels or sorting. Run it and show the output.",
    "Build a simple spam classifier in Python. Create a small dataset of spam and non-spam messages (at least 5 each). Write a classifier that checks for spam keywords and assigns a score. Test it on your dataset and think about what it gets wrong. Improve it and test again. Think about how real spam filters work.",
    # Self-improvement projects — meta-learning
    "Spend this session improving your own reasoning. Start by using think to identify 3 types of problems you struggle with. For each, search for techniques that could help (web_search, wikipedia_search). Try applying each technique to a sample problem with run_script. Think about which technique helped most and record a discovery about your own learning.",
    "Design and run a self-assessment. Create 5 questions that test different abilities (math, code, knowledge, reasoning, instruction following). Answer each one, then use run_script to verify your math and code answers. Use web_search to verify your knowledge answers. Think about which answers you got wrong and why. Record what you learned about your own capabilities.",
    "Investigate what makes a good prompt. Write 3 different prompts for the same task (e.g. 'write a function to check if a number is prime'). Make one very brief, one detailed, and one with examples. Run each with run_script and think about which produces the best result and why. Search for 'prompt engineering' and compare your findings with the research. Record your discovery.",
    # Debugging and analysis projects
    "Debug this Python code step by step: 'def fib(n): if n<2: return n; return fib(n-1)+fib(n-3)'. There's a bug in it. Run it, identify the bug, fix it, and test the fixed version with several inputs. Think about how you found the bug and what debugging strategies work best. Record your discovery.",
    "Analyze the performance of Python data structures. Write a script that benchmarks: list append, dict insert, set add, and tuple creation — each with 10000 operations. Think about why some are faster than others. Search for 'Python data structure performance' and compare your findings with the research. Record what you learned.",
    "Build a simple REST API mock in Python. Define endpoints for a todo list (GET /todos, POST /todos, DELETE /todos/:id). Use a dict as the database. Test each endpoint with run_script. Think about what a real API needs that yours is missing (authentication, validation, error handling). Implement at least one improvement.",
]

# Legacy flat list (kept for backward compat)
_TASKS = _TASKS_EASY + _TASKS_MEDIUM + _TASKS_HARD + _TASKS_EXPLORE + _TASKS_PROJECT


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
        self.tiers = ["easy", "medium", "hard", "explore", "project"]
        self.tasks = {
            "easy": list(_TASKS_EASY),
            "medium": list(_TASKS_MEDIUM),
            "hard": list(_TASKS_HARD),
            "explore": list(_TASKS_EXPLORE),
            "project": list(_TASKS_PROJECT),
        }
        self.results = {t: [] for t in self.tiers}  # rolling [bool]
        self.weights = {"easy": 0.25, "medium": 0.25, "hard": 0.15, "explore": 0.15, "project": 0.20}
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
        # Always keep some explore + project weight — discovery and building are the goals.
        if rates["easy"] > 0.8 and rates.get("hard", 0.5) < 0.3:
            self.weights = {"easy": 0.10, "medium": 0.25, "hard": 0.30, "explore": 0.15, "project": 0.20}
        elif rates["easy"] > 0.8:
            self.weights = {"easy": 0.15, "medium": 0.30, "hard": 0.20, "explore": 0.15, "project": 0.20}
        elif rates.get("hard", 0.5) < 0.2:
            self.weights = {"easy": 0.40, "medium": 0.25, "hard": 0.05, "explore": 0.15, "project": 0.15}
        elif rates.get("medium", 0.5) > 0.8:
            self.weights = {"easy": 0.05, "medium": 0.20, "hard": 0.30, "explore": 0.20, "project": 0.25}
        elif rates.get("explore", 0.5) > 0.6:
            # Model is good at exploration — push it toward projects
            self.weights = {"easy": 0.05, "medium": 0.15, "hard": 0.20, "explore": 0.25, "project": 0.35}
        elif rates.get("project", 0.5) > 0.5:
            # Model is handling projects well — give it more
            self.weights = {"easy": 0.10, "medium": 0.15, "hard": 0.20, "explore": 0.15, "project": 0.40}
        # Default: keep current weights

    def sample(self, n: int) -> list[tuple[str, str]]:
        """Sample n tasks. Returns list of (tier, task_string).

        Deduplicates within a single sample call so the same task doesn't
        appear twice in one epoch. If a tier has fewer tasks than needed,
        allows repeats only after exhausting all unique tasks.
        """
        tiers = self.rng.choices(self.tiers, weights=[self.weights[t] for t in self.tiers], k=n)
        result = []
        used: set[str] = set()
        for tier in tiers:
            available = [t for t in self.tasks[tier] if t not in used]
            if not available:
                # All tasks in this tier used — reset and allow repeats
                available = self.tasks[tier]
            task = self.rng.choice(available)
            used.add(task)
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

    Uses multiplicative decomposition (ToolRLA, 2026):
      total = format * selection * execution * completion * agentic_bonus

    - format: Did the model emit valid JSON tool calls? (0..1)
    - selection: Did the model choose appropriate tools for the task? (0..1)
    - execution: Did tools execute without errors? (0..1)
    - completion: Did the trajectory actually accomplish the task? (0..1)
    - answer_given: Did the model produce a final answer? (0..1)
    - answer_relevant: Is the answer relevant to the task? (0..1)
    - discovery: Did the model find/learn something new? (0..1)
    - stopped_ok: Did the model stop correctly after tools + answer? (0..1)
    - conciseness: Did the model keep output concise? (0..1)

    Agentic bonus components (additive, capped at 0.3):
    - error_recovery: Recovered from a failed tool call (Fission-GRPO)
    - self_verification: Verified answer with a tool (SFS-DPO)
    - planning: Used think before first tool call
    - tool_diversity: Used multiple different tools
    - efficiency_penalty: Penalize repeated identical calls (SearchMaster OOP)
    """
    # Core multiplicative components (0..1 each)
    format_ok: float = 0.0
    tool_selection: float = 0.5    # neutral default — appropriate tool choice
    tool_executed: float = 0.0
    task_completed: float = 0.0

    # Secondary components (used in completion scoring)
    answer_given: float = 0.0
    answer_relevant: float = 0.0
    output_quality: float = 0.0
    discovery: float = 0.0
    stopped_ok: float = 0.0
    conciseness: float = 0.0

    # Agentic bonus components (additive, capped)
    error_recovery: float = 0.0       # recovered from failure
    self_verification: float = 0.0    # verified own answer
    planning: float = 0.0             # thought before acting
    tool_diversity: float = 0.0       # used diverse tools
    efficiency_penalty: float = 0.0   # negative: repeated no-progress calls
    math_reasoning: float = 0.0       # think-with-math: reasoned through math before verifying

    @property
    def total(self) -> float:
        """Multiplicative core + additive agentic bonus.

        Multiplicative: if ANY core component is 0, the core score is 0.
        This encodes the priority: format → selection → execution → completion.
        (ToolRLA, 2026: multiplicative beats additive by 7pp.)

        Agentic bonus: added on top, capped at 0.3, can be negative (penalties).
        """
        core = (
            self.format_ok *
            self.tool_selection *
            max(self.tool_executed, 0.1) *  # floor: no tools = 0.1, not 0
            max(self.task_completed, 0.1)   # floor: partial completion
        )
        agentic_bonus = min(
            self.error_recovery * 0.15 +
            self.self_verification * 0.10 +
            self.planning * 0.05 +
            self.tool_diversity * 0.05 +
            self.math_reasoning * 0.15 +  # think-with-math: strong incentive
            self.efficiency_penalty,  # already negative
            0.3
        )
        return max(0.0, min(1.0, core + agentic_bonus))

    def to_dict(self) -> dict:
        return {
            "format_ok": self.format_ok,
            "tool_selection": self.tool_selection,
            "tool_executed": self.tool_executed,
            "task_completed": self.task_completed,
            "answer_given": self.answer_given,
            "answer_relevant": self.answer_relevant,
            "output_quality": self.output_quality,
            "discovery": self.discovery,
            "stopped_ok": self.stopped_ok,
            "conciseness": self.conciseness,
            "error_recovery": self.error_recovery,
            "self_verification": self.self_verification,
            "planning": self.planning,
            "tool_diversity": self.tool_diversity,
            "efficiency_penalty": self.efficiency_penalty,
            "math_reasoning": self.math_reasoning,
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
        score += min(substantive / max(len(results), 1), 1.0) * 0.3

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


def _safe_float(s: str) -> float | None:
    """Safely parse a string to float, returning None on failure."""
    try:
        return float(s)
    except (ValueError, OverflowError):
        return None


# Known ground-truth answers for deterministic math tasks.
# (regex_pattern, expected_value, expected_string_in_answer)
# This prevents false wins where the model writes a buggy script and
# reports the wrong number (e.g. GCD=493731 when correct=21).
_MATH_GROUND_TRUTHS: list[tuple[str, float, str]] = [
    (r"2\^15|2\*\*15", 32768.0, "32768"),
    (r"2\^20|2\*\*20", 1048576.0, "1048576"),
    (r"2\^10|2\*\*10", 1024.0, "1024"),
    (r"factorial of 8|8!", 40320.0, "40320"),
    (r"factorial of 10|10!", 3628800.0, "3628800"),
    (r"factorial of 12|12!", 479001600.0, "479001600"),
    (r"factorial of 15|15!", 1307674368000.0, "1307674368000"),
    (r"sum of (?:all )?even.*1 to 100|sum of even.*1.*100", 2550.0, "2550"),
    (r"area of a circle.*radius 7|circle.*radius 7", 153.94, "153"),
    (r"area of a circle.*radius 5|circle.*radius 5", 78.54, "78"),
    (r"gcd.*48.*36|gcd.*36.*48", 12.0, "12"),
    (r"gcd.*1071.*462|gcd.*462.*1071", 21.0, "21"),
    (r"fibonacci.*\b10\b", 55.0, "55"),
    (r"fibonacci.*\b20\b", 6765.0, "6765"),
    (r"trailing zeros.*100\b|100.*trailing zeros", 24.0, "24"),
    (r"15 \* 37|15\*37", 597.0, "597"),
    (r"compound interest.*1000.*5%.*3 year", 1157.63, "1157"),
    (r"percentage increase.*1\.5.*2\.0|1\.5 to 2\.0 degrees", 0.3333, "33.3"),
    (r"100 / 7|100/7", 14.2857, "14.28"),
    (r"square root of 144", 12.0, "12"),
    (r"97 is prime|primality of 97|check if 97", 1.0, "prime"),
    (r"tower of hanoi.*5 disk", 31.0, "31"),  # 2^5 - 1 = 31 moves
]


def _get_math_ground_truth(task: str) -> tuple[float, str] | None:
    """Return (expected_value, expected_string) for a math task, or None.

    Used to verify that script output and final answer match the known
    correct answer, preventing false wins from buggy scripts.
    """
    task_lower = task.lower()
    for pattern, value, expected_str in _MATH_GROUND_TRUTHS:
        if re.search(pattern, task_lower):
            return (value, expected_str)
    return None


def _evaluate_task_completion(task: str, tool_calls: list[dict],
                              final_answer: str | None) -> float:
    task_lower = task.lower()
    score = 0.0

    # Calculation tasks (includes think-with-math tasks that require calculation)
    if any(kw in task_lower for kw in ["calculate", "what is", "factorial",
                                        "square root", "gcd", "area",
                                        "compound interest", "2^", "2**",
                                        "trailing zeros", "euclidean",
                                        "sum of", "fibonacci", "prime",
                                        "tower of hanoi", "probability",
                                        "step by step about", "think through how to",
                                        "think step by step"]):
        # Extract the ground-truth answer from script output if available.
        # The model may run a script to compute the answer — if so, verify
        # the final_answer matches the script output. This prevents wrong
        # answers from getting high rewards (e.g. GCD=493731 when correct=21).
        script_numbers = []
        for tc in tool_calls:
            if tc.get("name") in ("calculate", "run_script") and tc.get("success"):
                result = tc.get("result", {})
                if isinstance(result, dict):
                    stdout = result.get("stdout", "").strip()
                    # Extract all numbers from script output
                    found = re.findall(r'\d+\.?\d*', stdout)
                    script_numbers.extend(found)

        # Extract numbers from the final answer
        answer_numbers = []
        if final_answer:
            answer_numbers = re.findall(r'\d+\.?\d*', final_answer)

        has_number = bool(answer_numbers) or bool(script_numbers)
        if has_number:
            score += 0.3  # reduced from 0.5 — having a number is necessary but not sufficient

        # Ground-truth verification: for tasks with known deterministic answers,
        # verify the script output matches the expected value. This catches
        # cases where the model writes a buggy script (e.g. GCD=493731 when
        # correct=21) and reports the wrong number — the answer matches the
        # script, but the script itself is wrong.
        ground_truth = _get_math_ground_truth(task)
        if ground_truth is not None:
            gt_val, gt_str = ground_truth
            # Check if script output contains the correct answer
            script_correct = any(
                abs(float(s) - gt_val) < 0.01 * max(abs(gt_val), 1)
                for s in script_numbers
                if _safe_float(s) is not None
            ) if script_numbers else False
            # Check if final answer contains the correct answer
            answer_correct = gt_str in (final_answer or "").lower() or any(
                abs(float(a) - gt_val) < 0.01 * max(abs(gt_val), 1)
                for a in answer_numbers
                if _safe_float(a) is not None
            ) if answer_numbers else False

            if script_correct and answer_correct:
                score += 0.4  # both script and answer are correct
            elif answer_correct and not script_numbers:
                score += 0.3  # answer correct, no script to verify
            elif script_correct != answer_correct:
                score -= 0.2  # one is right, other is wrong — inconsistent
            else:
                score -= 0.4  # both wrong — strong penalty for false win
        elif script_numbers and answer_numbers:
            # No known ground truth — fall back to answer-vs-script check
            script_result = script_numbers[-1]
            try:
                s_val = float(script_result)
                matched = any(abs(float(a) - s_val) < 0.01 * max(abs(s_val), 1)
                              for a in answer_numbers)
                if matched:
                    score += 0.4  # answer verified against script output
                else:
                    score -= 0.3  # answer contradicts script output — penalize
            except (ValueError, OverflowError):
                score += 0.2  # can't verify, give partial
        elif script_numbers and not answer_numbers:
            score += 0.1  # script ran but model didn't report a number

        if final_answer and len(final_answer) > 5:
            score += 0.2  # reduced from 0.3
        if any(tc.get("success") for tc in tool_calls):
            score += 0.1  # reduced from 0.2

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


def _evaluate_error_recovery(tool_calls: list[dict]) -> float:
    """Reward recovering from a failed tool call (Fission-GRPO, 2026).

    Detects pattern: tool failed → model retried (same or different tool) → succeeded.
    This is the key behavior that separates agents from chatbots.
    """
    if len(tool_calls) < 2:
        return 0.0

    recovery_score = 0.0
    for i, tc in enumerate(tool_calls[:-1]):
        if not tc.get("success"):
            # This call failed — did a subsequent call succeed?
            for later_tc in tool_calls[i+1:]:
                if later_tc.get("success"):
                    # Recovery! More credit if different tool or different args
                    if later_tc.get("name") != tc.get("name"):
                        recovery_score += 0.5  # switched tool — smart
                    elif later_tc.get("args") != tc.get("args"):
                        recovery_score += 0.3  # same tool, different args — good
                    else:
                        recovery_score += 0.1  # same call — less smart but still recovered
                    break  # count first recovery per failure

    return min(recovery_score, 1.0)


def _evaluate_self_verification(tool_calls: list[dict],
                                final_answer: str | None) -> float:
    """Reward self-verification: model checked its own answer (SFS-DPO, 2026).

    Detects: model produced output → then ran a script/search to verify.
    Pattern: run_script/calculate after a non-tool answer, or after another tool.
    """
    if not tool_calls or not final_answer:
        return 0.0

    # Check if a verification tool was called AFTER other tools or answers
    verify_tools = {"run_script", "calculate", "query_db"}
    has_verify = any(tc.get("name") in verify_tools for tc in tool_calls)

    # Check if verify tool was called AFTER another tool (not just first)
    for i, tc in enumerate(tool_calls):
        if tc.get("name") in verify_tools and i > 0:
            # Called a verification tool after at least one other tool
            return 1.0

    # If only verification tools were called, partial credit
    if has_verify:
        return 0.3

    return 0.0


def _evaluate_planning(tool_calls: list[dict]) -> float:
    """Reward planning: using think/sudo_think BEFORE the first action tool.

    Teaches the model to reason before acting, not just react.
    """
    if not tool_calls:
        return 0.0

    think_tools = {"think", "sudo_think"}
    action_tools = {"run_script", "web_search", "wikipedia_search",
                    "arxiv_search", "calculate", "fetch_url"}

    first_action_idx = None
    first_think_idx = None

    for i, tc in enumerate(tool_calls):
        if tc.get("name") in action_tools and first_action_idx is None:
            first_action_idx = i
        if tc.get("name") in think_tools and first_think_idx is None:
            first_think_idx = i

    # Think before first action = full planning reward
    if first_think_idx is not None and first_action_idx is not None:
        if first_think_idx < first_action_idx:
            return 1.0
        elif first_think_idx == first_action_idx:
            return 0.5

    # Think at all = partial
    if first_think_idx is not None:
        return 0.3

    return 0.0


def _evaluate_tool_diversity(tool_calls: list[dict]) -> float:
    """Reward using multiple DIFFERENT tools (not same tool repeatedly).

    Tool diversity correlates with agentic problem-solving.
    """
    if not tool_calls:
        return 0.0

    unique_tools = set(tc.get("name") for tc in tool_calls)
    n_unique = len(unique_tools)

    if n_unique >= 4:
        return 1.0
    elif n_unique == 3:
        return 0.7
    elif n_unique == 2:
        return 0.4
    elif n_unique == 1:
        return 0.1  # at least used a tool
    return 0.0


def _evaluate_efficiency_penalty(tool_calls: list[dict]) -> float:
    """Penalize repeated identical tool calls with no progress (SearchMaster OOP, 2026).

    Returns a NEGATIVE value (penalty).
    """
    if len(tool_calls) < 3:
        return 0.0

    penalty = 0.0

    # Detect repeated identical calls (same name + same args)
    call_signatures = []
    for tc in tool_calls:
        sig = (tc.get("name", ""), json.dumps(tc.get("args", {}), sort_keys=True))
        call_signatures.append(sig)

    # Count duplicates
    from collections import Counter
    sig_counts = Counter(call_signatures)
    for sig, count in sig_counts.items():
        if count >= 3:
            penalty -= 0.1 * (count - 2)  # -0.1 per extra repeat beyond 2

    # Penalize many calls with no successes
    n_calls = len(tool_calls)
    n_success = sum(1 for tc in tool_calls if tc.get("success"))
    if n_calls >= 5 and n_success == 0:
        penalty -= 0.15  # 5+ calls, all failed — flailing
    elif n_calls >= 8 and n_success / n_calls < 0.3:
        penalty -= 0.1  # many calls, low success rate

    return max(penalty, -0.3)  # cap penalty at -0.3


def _evaluate_math_reasoning(task: str, tool_calls: list[dict]) -> float:
    """Reward think-with-math: model reasons through math before verifying.

    Detects the pattern: think (with math content) → calculate/run_script
    on math-related tasks. This is the key incentive for the model to learn
    step-by-step mathematical reasoning rather than just guessing.

    Scoring:
    - 1.0: think before calculate/run_script, with numbers in think content
    - 0.7: think before calculate/run_script, but no numbers in think
    - 0.5: think after calculate (reflected on result)
    - 0.3: any think on a math task
    - 0.0: no think on a math task
    """
    if not tool_calls:
        return 0.0

    task_lower = task.lower()
    math_keywords = ["calculate", "factorial", "gcd", "prime", "fibonacci",
                     "square root", "area", "compound", "sum", "product",
                     "probability", "combinatorics", "matrix", "exponential",
                     "trailing zeros", "euclidean", "induction", "paradox",
                     "tower of hanoi", "2^", "step by step"]
    is_math_task = any(kw in task_lower for kw in math_keywords)
    if not is_math_task:
        return 0.0

    think_tools = {"think", "sudo_think"}
    math_tools = {"calculate", "run_script"}

    think_indices = [i for i, tc in enumerate(tool_calls)
                     if tc.get("name") in think_tools]
    math_indices = [i for i, tc in enumerate(tool_calls)
                    if tc.get("name") in math_tools]

    if not think_indices:
        return 0.0

    if not math_indices:
        # Think used on math task but no verification tool — partial
        return 0.3

    first_think = think_indices[0]
    first_math = math_indices[0]

    # Check if think content contains numbers (actual math reasoning)
    has_numbers_in_think = False
    for i in think_indices:
        args = tool_calls[i].get("args", {})
        content = args.get("content", "") if isinstance(args, dict) else str(args)
        if re.search(r'\d+\.?\d*', content):
            has_numbers_in_think = True
            break

    # Think BEFORE math tool = best (reasoning then verification)
    if first_think < first_math:
        return 1.0 if has_numbers_in_think else 0.7

    # Think AFTER math tool = reflected on result
    return 0.5


def _evaluate_tool_selection(task: str, tool_calls: list[dict]) -> float:
    """Evaluate whether the model chose appropriate tools for the task.

    Checks if the tools used match the task type.
    """
    if not tool_calls:
        return 0.3  # no tools but answered — neutral

    task_lower = task.lower()
    tool_names = set(tc.get("name") for tc in tool_calls)

    score = 0.5  # base: used tools

    # Calculation tasks should use calculate or run_script
    if any(kw in task_lower for kw in ["calculate", "what is", "factorial",
                                        "square root", "gcd", "area"]):
        if "calculate" in tool_names or "run_script" in tool_names:
            score = 1.0
        elif any(t in tool_names for t in ["think", "sudo_think"]):
            score = 0.4  # thought but didn't compute

    # Search tasks should use web_search, wikipedia_search, or arxiv_search
    elif any(kw in task_lower for kw in ["search", "research", "find",
                                          "look up", "investigate"]):
        if any(t in tool_names for t in ["web_search", "wikipedia_search",
                                          "arxiv_search", "fetch_url"]):
            score = 1.0
        elif "run_script" in tool_names:
            score = 0.5  # tried scripting instead of searching

    # Script tasks should use run_script
    elif "run" in task_lower and "script" in task_lower:
        if "run_script" in tool_names:
            score = 1.0

    # Think tasks should use think/sudo_think
    elif "think" in task_lower:
        if any(t in tool_names for t in ["think", "sudo_think"]):
            score = 1.0

    # Build/make/create tasks should use run_script
    elif any(kw in task_lower for kw in ["build", "make", "create",
                                          "implement", "write a", "design"]):
        if "run_script" in tool_names:
            score = 1.0

    # Multi-step tasks should use multiple tools
    elif "then" in task_lower or "first" in task_lower:
        if len(tool_names) >= 2:
            score = 1.0
        elif len(tool_names) == 1:
            score = 0.5

    # Project/explore tasks: reward diverse tool use
    elif any(kw in task_lower for kw in ["explore", "experiment", "benchmark",
                                          "test", "improve", "design"]):
        if len(tool_names) >= 3:
            score = 1.0
        elif len(tool_names) >= 2:
            score = 0.7

    return score


def compute_reward(task: str, tool_calls: list[dict],
                   final_answer: str | None,
                   stopped_after_tools: bool,
                   stopped_after_answer: bool,
                   seen_outputs: set | None = None) -> ToolUseReward:
    """Compute multi-component reward for a tool-use trajectory.

    Uses multiplicative decomposition for core components (ToolRLA, 2026)
    plus additive agentic bonuses (Fission-GRPO, SFS-DPO, SearchMaster).

    Args:
        seen_outputs: optional set for tracking novel outputs across sessions.
    """
    r = ToolUseReward()

    # ── Core multiplicative components ──────────────────────────────

    # Format: did all tool calls parse as valid JSON?
    if tool_calls:
        r.format_ok = 1.0
    elif final_answer and not tool_calls:
        r.format_ok = 0.3  # answered without tools — partial format

    # Tool selection: did the model choose appropriate tools?
    r.tool_selection = _evaluate_tool_selection(task, tool_calls)

    # Tool execution: did tools execute successfully?
    if tool_calls:
        successes = sum(1 for tc in tool_calls if tc.get("success"))
        r.tool_executed = successes / len(tool_calls)
    else:
        r.tool_executed = 0.1  # floor: no tools = 0.1

    # Task completion: did the trajectory actually solve the task?
    r.task_completed = _evaluate_task_completion(task, tool_calls, final_answer)

    # ── Secondary components (for logging/analysis) ─────────────────

    # Output quality
    r.output_quality = max(
        _evaluate_script_output(task, tool_calls),
        _evaluate_search_output(task, tool_calls),
    )

    # Answer given
    r.answer_given = 1.0 if final_answer and len(final_answer) > 5 else 0.0

    # Answer relevance
    r.answer_relevant = _check_answer_relevance(task, final_answer or "")

    # Discovery
    r.discovery = _evaluate_discovery(task, tool_calls, final_answer, seen_outputs)

    # Stopped correctly
    if stopped_after_tools and stopped_after_answer:
        r.stopped_ok = 1.0
    elif stopped_after_tools or stopped_after_answer:
        r.stopped_ok = 0.5

    # Conciseness
    if r.task_completed > 0.5 and final_answer:
        ans_tokens = len(final_answer.split())
        if ans_tokens <= 20:
            r.conciseness = 1.0
        elif ans_tokens <= 50:
            r.conciseness = 0.5
        elif ans_tokens <= 100:
            r.conciseness = 0.2

    # ── Agentic bonus components (additive, research-backed) ────────

    # Error recovery: model recovered from a failed tool call (Fission-GRPO)
    r.error_recovery = _evaluate_error_recovery(tool_calls)

    # Self-verification: model verified its own answer (SFS-DPO)
    r.self_verification = _evaluate_self_verification(tool_calls, final_answer)

    # Planning: model thought before acting
    r.planning = _evaluate_planning(tool_calls)

    # Tool diversity: used multiple different tools
    r.tool_diversity = _evaluate_tool_diversity(tool_calls)

    # Math reasoning: think-with-math pattern (think → verify on math tasks)
    r.math_reasoning = _evaluate_math_reasoning(task, tool_calls)

    # Efficiency penalty: repeated identical calls (SearchMaster OOP)
    r.efficiency_penalty = _evaluate_efficiency_penalty(tool_calls)

    return r


# ── Self-play loop ────────────────────────────────────────────────────────

@dataclass
class SelfPlayConfig:
    max_turns: int = 10          # max tool-call turns (increased for R&D freedom)
    max_gen_tokens: int = 512    # per-turn generation limit (increased for reasoning)
    temperature: float = 0.5     # higher temp for tool-call exploration
    top_k: int = 80              # LFM2.5-recommended top-k sampling
    repetition_penalty: float = 1.05  # LFM2.5-recommended repetition penalty
    min_reward: float = 0.25     # lowered to collect early-phase trajectories
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
                kv_cache="hadamard_int4",   # 4x KV compression (near-lossless)
                decoding="mtp_selfspec",    # self-speculative decode via MTP heads
                use_triton_conv=True,       # fused Triton conv kernels
                use_prefix_cache=True,      # cache KV for repeated prompt prefixes
                use_spec_attn=True,         # L1 speculative attention (57% attn compute cut)
                kv_cache_tokens=4096,       # limit KV allocation to 4K tokens
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
                 tools: list[dict], tier: str = "flat") -> dict:
        """Run a single task through the agentic loop.

        Returns: {messages, tool_calls, reward, final_answer, ...}
        """
        cfg = self.config
        # Project tasks need more turns for multi-step building
        max_turns = cfg.max_turns * 2 if tier == "project" else cfg.max_turns
        messages = [
            {"role": "system", "content": _FORGE_SYSTEM},
            {"role": "user", "content": task},
        ]
        tool_call_records = []  # {name, args, result, success}
        final_answer = None
        stopped_after_tools = False
        stopped_after_answer = False

        # Context manager for long conversations
        from research.self_play.discovery.context_manager import ContextManager, ContextManagerConfig
        # Project tasks need more context retained for multi-step building
        keep_recent = 10 if tier == "project" else 6
        ctx_config = ContextManagerConfig(
            max_seq_len=32768,
            reserved_for_generation=cfg.max_gen_tokens + 512,
            keep_recent_turns=keep_recent,
            use_model_summary=False,  # heuristic for speed
        )
        ctx_manager = ContextManager(ctx_config, tokenizer=self.tokenizer)

        for turn in range(max_turns):
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
            had_error = False
            for tc, result in zip(tc_for_msg, tool_results):
                content = json.dumps(result, ensure_ascii=False)[:4000]
                messages.append({"role": "tool", "name": tc["name"],
                                 "content": content})
                if isinstance(result, dict) and "error" in result:
                    had_error = True

            # Auto-reflection nudge after tool errors (Fission-GRPO inspired)
            # Inject a system message encouraging the model to think about
            # what went wrong and try a different approach
            if had_error and turn < max_turns - 1:
                messages.append({
                    "role": "system",
                    "content": "A tool returned an error. Use the think tool to "
                               "analyze what went wrong, then try a different "
                               "approach or different arguments."
                })

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
                    curriculum: TaskCurriculum | None = None,
                    custom_tasks: list[str] | None = None) -> dict:
        """Run a self-play session with multiple tasks.

        If custom_tasks is provided, uses those directly (model-generated goals).
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
        if custom_tasks is not None:
            tasks_with_tier = [("generated", t) for t in custom_tasks[:n]]
        elif curriculum is not None:
            sampled = curriculum.sample(n)
            tasks_with_tier = [(tier, task) for tier, task in sampled]
        else:
            rng = random.Random()
            tasks_with_tier = [("flat", t) for t in rng.choices(_TASKS, k=n)]

        results = []
        t0 = time.time()

        for i, (tier, task) in enumerate(tasks_with_tier):
            result = self.run_task(task, registry, tools, tier=tier)
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
