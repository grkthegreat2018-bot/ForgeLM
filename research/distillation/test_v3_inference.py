"""Quick inference tests for ForgeLM V3 SFT checkpoint.

Tests easy, medium, and hard prompts across code, reasoning, math, and logic.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from research.inference.forge_engine import ForgeEngine
from research.tokenizer_cache import get_tokenizer

CHECKPOINT = "research/checkpoints/ForgeLM_V3_SFT.safetensors"
CONFIG = "forgelm_v3"

# ── Test prompts ──

TESTS = [
    # ── EASY ──
    {
        "name": "Easy: FizzBuzz",
        "difficulty": "EASY",
        "prompt": "Write a Python function called fizzbuzz(n) that prints Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for multiples of both, and the number otherwise.",
        "max_tokens": 300,
    },
    {
        "name": "Easy: List reverse",
        "difficulty": "EASY",
        "prompt": "Write a Python function to reverse a list in place without using built-in reverse methods.",
        "max_tokens": 200,
    },
    {
        "name": "Easy: Simple math",
        "difficulty": "EASY",
        "prompt": "If a train travels 60 mph for 2.5 hours, how far does it go? Show your work.",
        "max_tokens": 150,
    },
    # ── MEDIUM ──
    {
        "name": "Medium: Binary search",
        "difficulty": "MEDIUM",
        "prompt": "Implement binary search in Python. Include edge case handling for empty lists and targets not found. Return the index or -1.",
        "max_tokens": 400,
    },
    {
        "name": "Medium: LRU Cache",
        "difficulty": "MEDIUM",
        "prompt": "Implement an LRU (Least Recently Used) cache in Python with O(1) get and put operations. Use OrderedDict.",
        "max_tokens": 400,
    },
    {
        "name": "Medium: Logic puzzle",
        "difficulty": "MEDIUM",
        "prompt": "Three people (Alice, Bob, Carol) are standing in a line. Alice can see Bob and Carol. Bob can see Carol. Carol sees nobody. They each have a hat that is either red or blue. There are 3 red hats and 2 blue hats total. Alice says 'I don't know my hat color.' Bob then says 'I don't know my hat color.' What color is Carol's hat? Explain your reasoning.",
        "max_tokens": 400,
    },
    {
        "name": "Medium: Word problem",
        "difficulty": "MEDIUM",
        "prompt": "A store sells apples at $1.20 each, or 3 for $3.00. If you have $10.00, what is the maximum number of apples you can buy, and how much money is left over? Show your reasoning step by step.",
        "max_tokens": 300,
    },
    # ── HARD ──
    {
        "name": "Hard: Dynamic programming",
        "difficulty": "HARD",
        "prompt": "Implement the longest common subsequence (LCS) problem in Python using dynamic programming. Return both the length and the actual subsequence string. Include test cases.",
        "max_tokens": 600,
    },
    {
        "name": "Hard: Graph algorithm",
        "difficulty": "HARD",
        "prompt": "Implement Dijkstra's shortest path algorithm in Python using a priority queue. The graph is represented as an adjacency list with (neighbor, weight) tuples. Return the shortest distances from a source node to all other nodes.",
        "max_tokens": 600,
    },
    {
        "name": "Hard: Complex reasoning",
        "difficulty": "HARD",
        "prompt": "You have 12 coins, one of which is counterfeit and has a different weight (heavier or lighter). You have a balance scale that can only be used 3 times. Describe a strategy to identify the counterfeit coin and determine whether it is heavier or lighter.",
        "max_tokens": 800,
    },
    {
        "name": "Hard: System design",
        "difficulty": "HARD",
        "prompt": "Design a rate limiter that allows 100 requests per minute per user with a sliding window. Implement it in Python. Consider thread safety and memory efficiency.",
        "max_tokens": 600,
    },
]


def format_prompt(prompt: str) -> str:
    """Format as a chat prompt in Qwen style."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def run_tests():
    print("=" * 70)
    print("  ForgeLM V3 SFT — Inference Tests")
    print("=" * 70)
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Config: {CONFIG}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    engine = ForgeEngine.from_checkpoint(
        CHECKPOINT, config_name=CONFIG, device="cuda")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free = torch.cuda.mem_get_info(0)
    print(f"  VRAM after load: {free[0]/1024**3:.1f} GB free / {free[1]/1024**3:.1f} GB total")
    print()

    results = []
    for test in TESTS:
        name = test["name"]
        difficulty = test["difficulty"]
        prompt = test["prompt"]
        max_tokens = test["max_tokens"]

        print("-" * 70)
        print(f"  [{difficulty}] {name}")
        print("-" * 70)
        print(f"  PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print()

        formatted = format_prompt(prompt)
        t0 = time.time()
        try:
            output = engine.generate(
                formatted,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                top_k=80,
                repetition_penalty=1.05,
            )
            elapsed = time.time() - t0
            # Clean up the output
            output = output.strip()
            if "<|im_end|>" in output:
                output = output.split("<|im_end|>")[0].strip()

            n_tokens = len(output.split())  # rough estimate
            tps = n_tokens / elapsed if elapsed > 0 else 0

            print(f"  RESPONSE ({elapsed:.1f}s, ~{n_tokens} words, {tps:.0f} w/s):")
            print()
            # Indent the response
            for line in output.split("\n"):
                print(f"  {line}")
            print()

            results.append({
                "name": name,
                "difficulty": difficulty,
                "output": output,
                "elapsed": elapsed,
                "words": n_tokens,
                "tps": tps,
                "error": None,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            print()
            results.append({
                "name": name,
                "difficulty": difficulty,
                "output": "",
                "elapsed": elapsed,
                "words": 0,
                "tps": 0,
                "error": str(e),
            })

        # Check VRAM
        free = torch.cuda.mem_get_info(0)
        print(f"  VRAM: {free[0]/1024**3:.1f} GB free")
        print()

    # ── Summary ──
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        status = "OK" if r["error"] is None else "ERROR"
        print(f"  [{r['difficulty']:6s}] {status:5s} {r['name']:40s} "
              f"{r['elapsed']:5.1f}s  {r['words']:4d} words")
    print()

    # By difficulty
    for diff in ["EASY", "MEDIUM", "HARD"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            avg_time = sum(r["elapsed"] for r in subset) / len(subset)
            avg_words = sum(r["words"] for r in subset) / len(subset)
            errors = sum(1 for r in subset if r["error"])
            print(f"  {diff:6s}: {len(subset)} tests, avg {avg_time:.1f}s, "
                  f"avg {avg_words:.0f} words, {errors} errors")


if __name__ == "__main__":
    run_tests()
