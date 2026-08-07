"""Batch synthetic data generation via LM Studio's local OpenAI-compatible API.

Generates (prompt, completion) JSONL pairs by querying a locally-served model
(e.g., Qwen3-Coder-30B-A3B-Instruct GGUF via LM Studio). This is sequence-level
distillation data collection — works with ANY model LM Studio can serve.

Usage:
    # Start LM Studio server first (default port 1234)
    # Then run:
    python -m research.generate_synthetic --output research/data/lmstudio_synthetic.jsonl \
        --num-samples 1000 --domains coding reasoning knowledge math writing \
        --base-url http://localhost:1234/v1 --model qwen3-coder-30b

    # With temperature variation for diversity
    python -m research.generate_synthetic --output research/data/lmstudio_synthetic.jsonl \
        --num-samples 500 --temperature 0.9 --max-tokens 512
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import requests


# Prompt templates for each domain. Each generates a diverse set of instructions.
DOMAIN_PROMPTS = {
    "coding": [
        "Write a Python function to {task}",
        "Implement a {task} in Python with proper error handling",
        "Review this code for issues: {task}",
        "Debug this Python code: {task}",
        "Explain how {task} works with a code example",
        "Refactor this function to be more Pythonic: {task}",
        "Write a unit test for {task}",
        "Optimize this Python code for performance: {task}",
    ],
    "reasoning": [
        "Solve this step by step: {task}",
        "Explain the reasoning behind {task}",
        "If {task}, what can we conclude?",
        "Prove that {task}",
        "What is the logical flaw in this argument: {task}?",
        "Analyze the cause and effect of {task}",
        "Compare and contrast {task}",
        "What are the trade-offs of {task}?",
    ],
    "knowledge": [
        "Explain {task} in simple terms",
        "What is the difference between {task}?",
        "How does {task} work?",
        "Why is {task} important?",
        "Describe the history of {task}",
        "What are the key concepts of {task}?",
        "Give an overview of {task}",
        "What are common misconceptions about {task}?",
    ],
    "math": [
        "Solve: {task}",
        "Calculate {task} showing all steps",
        "If {task}, find the value of x",
        "Prove that {task}",
        "What is the derivative of {task}?",
        "Find the area of {task}",
        "How many ways can {task}?",
        "What is the probability that {task}?",
    ],
    "writing": [
        "Write a clear explanation of {task}",
        "Summarize {task} in 3 paragraphs",
        "Write a tutorial on {task}",
        "Create a step-by-step guide for {task}",
        "Write documentation for {task}",
        "Explain {task} to a beginner",
        "Write an FAQ about {task}",
        "Describe {task} with examples",
    ],
}

# Task seeds for each domain — these get substituted into the prompt templates.
TASK_SEEDS = {
    "coding": [
        "reverse a linked list iteratively",
        "implement a hash map from scratch",
        "find the longest palindromic substring",
        "implement a priority queue using a binary heap",
        "detect a cycle in a directed graph",
        "serialize and deserialize a binary tree",
        "implement quicksort with in-place partitioning",
        "find the kth smallest element in an array",
        "implement a rate limiter using token bucket algorithm",
        "validate balanced parentheses in a string",
        "implement a trie for autocomplete",
        "find duplicate elements in an array in O(n) time",
        "implement merge sort bottom-up iteratively",
        "check if a binary tree is a valid BST",
        "implement an LRU cache with O(1) operations",
        "find the shortest path in an unweighted graph using BFS",
        "implement a circular buffer",
        "detect if two strings are anagrams",
        "implement a thread-safe singleton pattern",
        "flatten a nested list of arbitrary depth",
        "implement a bloom filter",
        "find the median of two sorted arrays",
        "implement a basic regex matcher for * and .",
        "check if a number is prime using trial division",
        "implement a simple blockchain block hash",
        "write a decorator that retries on failure",
        "implement a context manager for database transactions",
        "create a generator that yields fibonacci numbers lazily",
        "implement a simple pub/sub system",
        "write a function to parse a CSV file without using csv module",
        "implement a basic HTTP server using only stdlib",
        "create a memoization decorator with LRU eviction",
        "implement a sliding window maximum algorithm",
        "write a function to compress a string using run-length encoding",
        "implement a basic interpreter for arithmetic expressions",
        "find all permutations of a string",
        "implement a union-find data structure with path compression",
        "write a function to validate an email address",
        "implement a simple cache with TTL support",
        "create a decorator that measures memory usage",
        "implement a basic state machine",
        "write a function to find the longest increasing subsequence",
        "implement a simple reverse proxy",
        "create a basic key-value store with persistence",
        "implement a simple neural network from scratch using only numpy",
        "write a function to detect language from text",
        "implement a basic spell checker using edit distance",
        "create a simple web scraper with rate limiting",
        "implement a circular linked list with insert and delete",
    ],
    "reasoning": [
        "if all cats are mammals and all mammals are animals, are all cats animals",
        "why does ice float on water but most solids sink in their liquid form",
        "if a clock shows 3:15, what is the angle between the hour and minute hands",
        "why does the sky appear blue during the day but red at sunset",
        "if you flip a fair coin 10 times and get heads each time, what is the probability of heads on the 11th flip",
        "why does hot water sometimes freeze faster than cold water",
        "if two trains are 100 miles apart heading toward each other at 60 and 40 mph, when do they meet",
        "why do mirrors appear to reverse left and right but not up and down",
        "if 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets",
        "why does division by zero produce an undefined result",
        "if the day after tomorrow is Wednesday, what day was yesterday",
        "why do objects with different masses fall at the same rate in a vacuum",
        "if a bat and ball cost $1.10 total and the bat costs $1 more than the ball, how much is the ball",
        "why does time slow down near massive objects according to relativity",
        "if you have 8 balls and one is heavier, find it in 2 weighings on a balance scale",
        "why does a rainbow always appear in an arc",
        "if 3x + 7 = 22, solve for x and explain each step",
        "why do we see lightning before we hear thunder",
        "if a store offers 20% then 30% off, what is the total discount percentage",
        "why does a spinning top stay upright while spinning but falls when stopped",
        "if all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly",
        "why does salt melt ice on roads in winter",
        "if a snail climbs 3 feet per day and slides back 2 feet per night, how long to climb a 10-foot wall",
        "why does a straw appear bent when placed in a glass of water",
        "if you roll two dice, what is the probability the sum is 7",
        "why do we season food with salt rather than sugar for savory dishes",
        "if a rectangle has perimeter 40m and area 96m², find its dimensions",
        "why does a boat float but a stone of the same material sinks",
        "if the sum of two odd numbers is always even, prove it",
        "why does the moon appear to change shape throughout the month",
    ],
    "knowledge": [
        "how HTTPS encryption works",
        "the difference between TCP and UDP",
        "how Docker containers work",
        "what caused the Industrial Revolution",
        "how neurons transmit signals in the brain",
        "the difference between REST and GraphQL APIs",
        "what quantum entanglement is",
        "how a CPU executes a program",
        "the difference between supervised and unsupervised learning",
        "how public-key cryptography works",
        "what the CAP theorem states about distributed systems",
        "how garbage collection works in modern languages",
        "the difference between a compiler and an interpreter",
        "how git stores data internally",
        "what a neural network is and how it learns",
        "the difference between SQL and NoSQL databases",
        "how a hash function works and where it is used",
        "what the difference between authentication and authorization is",
        "how a database index improves query performance",
        "the difference between concurrency and parallelism",
        "what closures are in programming",
        "how the DNS system resolves domain names",
        "the difference between a stack and a heap in memory",
        "what virtual memory is and how it works",
        "how load balancers distribute traffic",
        "the difference between a process and a thread",
        "what ACID properties mean in databases",
        "how CDNs improve website performance",
        "the difference between symmetric and asymmetric encryption",
        "what the OSI model layers are",
        "how websockets differ from HTTP polling",
        "what microservices architecture is",
        "how OAuth 2.0 authentication works",
        "the difference between IPv4 and IPv6",
        "what NAT does in home routers",
        "how SSDs differ from HDDs in storage",
        "the difference between BFS and DFS graph traversal",
        "what a Bloom filter is and when to use it",
        "how Raft consensus algorithm works",
        "the difference between strong and eventual consistency",
    ],
    "math": [
        "3x + 7 = 22",
        "the area of a circle with radius 7",
        "the derivative of x^3 * sin(x)",
        "the integral of e^x * cos(x) dx",
        "x^2 - 5x + 6 = 0",
        "the sum of the first 100 natural numbers",
        "log base 2 of 1024",
        "the probability of getting at least one 6 in 4 dice rolls",
        "15% of 240",
        "the volume of a sphere with radius 5",
        "2x + 3y = 12 and x - y = 1, find x and y",
        "the GCD of 48 and 72 using Euclidean algorithm",
        "the number of ways to arrange 5 books on a shelf",
        "the number of ways to choose 3 items from 10",
        "sin(30 degrees) + cos(60 degrees)",
        "the limit of (x^2 - 1)/(x - 1) as x approaches 1",
        "the Taylor series expansion of e^x around x=0",
        "the eigenvalues of the matrix [[2,1],[1,2]]",
        "the expected value of rolling a fair 6-sided die",
        "the variance of a binomial distribution with n=10, p=0.5",
        "the area of a triangle with vertices at (0,0), (4,0), (0,3)",
        "the dot product of vectors [1,2,3] and [4,5,6]",
        "the determinant of the matrix [[1,2],[3,4]]",
        "the Fourier transform of a square pulse",
        "the Maclaurin series for ln(1+x)",
        "the probability of a royal flush in poker",
        "the sum of the infinite geometric series 1 + 1/2 + 1/4 + ...",
        "the number of integer solutions to x + y + z = 10 where x,y,z >= 0",
        "the surface area of a cylinder with radius 3 and height 10",
        "the compound interest on $1000 at 5% annual rate for 3 years",
    ],
    "writing": [
        "how to set up a Python project with virtual environments",
        "the basics of REST API design",
        "how to write clean code principles",
        "a beginner's guide to version control with git",
        "how to structure a microservice",
        "best practices for error handling in Python",
        "how to optimize database queries",
        "writing effective unit tests",
        "how to design a scalable system",
        "the principles of object-oriented programming",
        "how to secure a web application",
        "a guide to Docker for beginners",
        "how to implement CI/CD pipelines",
        "writing technical documentation",
        "how to debug a complex issue",
        "the basics of machine learning model deployment",
        "how to choose the right data structure",
        "a guide to async programming in Python",
        "how to refactor legacy code",
        "best practices for API error responses",
        "how to monitor a production system",
        "writing efficient SQL queries",
        "how to handle concurrency in distributed systems",
        "a guide to code review best practices",
        "how to design a caching strategy",
        "the principles of functional programming",
        "how to migrate a monolith to microservices",
        "writing good commit messages",
        "how to choose between SQL and NoSQL",
        "a guide to logging and observability",
    ],
}


def generate_prompt(domain, idx=None):
    """Generate a random prompt for the given domain."""
    templates = DOMAIN_PROMPTS[domain]
    seeds = TASK_SEEDS[domain]
    template = random.choice(templates)
    seed = random.choice(seeds)
    return template.format(task=seed)


def query_lm_studio(base_url, model, prompt, temperature=0.8, max_tokens=512, timeout=120,
                    enable_thinking=False):
    """Query LM Studio's OpenAI-compatible API."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, knowledgeable assistant. Provide clear, accurate, educational answers."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # Gemma-4 thinking mode is buggy in LM Studio; disabled by default.
        # Set enable_thinking=True only for models that support it correctly.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Cannot connect to LM Studio at {base_url}. Is the server running?")
        return None
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Request timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:100]}")
        return None


def main():
    p = argparse.ArgumentParser(description="Generate synthetic data via LM Studio API")
    p.add_argument("--output", default="research/data/lmstudio_synthetic.jsonl")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--domains", nargs="+", default=["coding", "reasoning", "knowledge"],
                   choices=list(DOMAIN_PROMPTS.keys()))
    p.add_argument("--base-url", default="http://localhost:1234/v1")
    p.add_argument("--model", default="qwen3-coder-30b-a3b-instruct",
                   help="Model name as shown in LM Studio (use 'loaded' for whatever is loaded)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    p.add_argument("--retry-failures", type=int, default=2, help="Number of retries for failed requests")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Number of concurrent API requests (LM Studio supports parallel completions)")
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    print(f"Generating {args.num_samples} samples across domains: {args.domains}")
    print(f"Output: {output_path} (mode: {mode})")
    print(f"API: {args.base_url} | model: {args.model}")
    print(f"Temperature: {args.temperature} | max_tokens: {args.max_tokens}")
    print()

    # Test connection first.
    print("Testing connection to LM Studio...")
    test = query_lm_studio(args.base_url, args.model, "Say 'hello' in one word.",
                           temperature=0.1, max_tokens=10, timeout=max(args.timeout, 300))
    if test is None:
        print("Connection test failed. Make sure LM Studio server is running.")
        return
    print(f"Connection OK. Response: {test[:50]}")
    print()

    # Generate samples.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    written = 0
    failed = 0
    t0 = time.time()
    write_lock = threading.Lock()

    # Build all tasks upfront.
    tasks = []
    for i in range(args.num_samples):
        domain = args.domains[i % len(args.domains)]
        prompt = generate_prompt(domain)
        tasks.append((i, domain, prompt))

    def process_one(task):
        i, domain, prompt = task
        completion = None
        for attempt in range(args.retry_failures + 1):
            completion = query_lm_studio(
                args.base_url, args.model, prompt,
                temperature=args.temperature + (0.1 * attempt),
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            if completion is not None and len(completion.strip()) > 20:
                break
            if attempt < args.retry_failures:
                time.sleep(2)
        return (i, domain, prompt, completion)

    print(f"Concurrency: {args.concurrency} parallel requests")
    print()

    with open(output_path, mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(process_one, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                i, domain, prompt, completion = fut.result()
                if completion is None or len(completion.strip()) < 20:
                    with write_lock:
                        failed += 1
                        print(f"  [{i+1}/{args.num_samples}] FAILED")
                    continue
                record = {"prompt": prompt, "completion": completion.strip(), "domain": domain}
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with write_lock:
                    f.write(line)
                    f.flush()
                    written += 1
                    elapsed = time.time() - t0
                    rate = written / max(elapsed, 1)
                    eta = (args.num_samples - written) / max(rate, 0.01)
                    print(f"  [{written+failed}/{args.num_samples}] {domain:10s} | {len(completion):4d} chars | "
                          f"{rate:.1f}/s | ETA {eta:.0f}s | written={written} failed={failed}")

    print(f"\nDone. Written: {written}, Failed: {failed}")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
