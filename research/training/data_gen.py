"""LM Studio data generation pipeline for ForgeLM_V2 fine-tuning.

Two modes:
1. Direct: Teacher models generate training data from task prompts.
2. Web-augmented: LFM (fast) generates search queries → web scrapers fetch
   real content → teacher models create grounded training data.

Usage:
    python -m research.training.data_gen --task tool_use --n 500
    python -m research.training.data_gen --task code --n 500 --web
    python -m research.training.data_gen --task short_cot --n 500
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

from research.paths import LMSTUDIO_API as LMSTUDIO_BASE, LMSTUDIO_GGUF, LMSTUDIO_MODELS_ROOT
from research.training.web_scraper import WebScraper, search_all

# Teacher models (sorted by quality)
TEACHERS = {
    "general": "openai/gpt-oss-20b",
    "reasoning": "grok-3-reasoning-gemma3-12b-distilled",
    "code": "qwen2.5-coder-7b-instruct",
    "fast": "liquid/lfm2.5-1.2b",  # LFM for query generation
}

# Output directory
DATA_DIR = Path(__file__).parent.parent.parent / "research" / "data" / "finetune"


# ── Task definitions ─────────────────────────────────────────────────────────

TOOL_USE_TASKS = [
    "Schedule a meeting between 3 people, finding a common time slot",
    "Search for flights from NYC to London on a specific date",
    "Calculate the total cost of a shopping cart with discounts",
    "Fetch weather data for 5 cities and summarize",
    "Parse a CSV file and extract specific columns",
    "Send an email notification to a list of recipients",
    "Query a database for users who signed up in the last 7 days",
    "Convert a JSON response to a formatted table",
    "Extract entities (names, dates, amounts) from a legal document",
    "Validate a form submission and return error messages",
    "Translate text to 3 languages using an API",
    "Resize and compress an image batch",
    "Generate a PDF report from template data",
    "Monitor a server's CPU/memory and alert if threshold exceeded",
    "Scrape product prices from an e-commerce page",
    "Create a calendar event with reminders",
    "Process a payment with retry logic",
    "Geocode addresses and calculate distances",
    "Filter and sort a list of products by multiple criteria",
    "Authenticate a user with OAuth2",
]

CODE_TASKS = [
    "Write a Python function to merge two sorted lists",
    "Implement a binary search tree with insert and search",
    "Create a REST API endpoint for user registration",
    "Write a SQL query to find top 10 customers by revenue",
    "Implement a LRU cache in Python",
    "Write a function to detect cycles in a linked list",
    "Create a React component for a todo list",
    "Write a regex to validate email addresses",
    "Implement quicksort with in-place partitioning",
    "Write a Dockerfile for a Python Flask app",
    "Create a Python decorator for rate limiting",
    "Write a function to parse and evaluate arithmetic expressions",
    "Implement a simple pub/sub message queue",
    "Write a script to backup files to S3",
    "Create a TypeScript interface for a paginated API response",
    "Write a function to find the longest common subsequence",
    "Implement a thread-safe singleton in Python",
    "Write a git pre-commit hook for linting",
    "Create a CSS grid layout for a dashboard",
    "Write a function to generate all permutations of a string",
]

SHORT_COT_TASKS = [
    "If a train travels 60 km/h for 2.5 hours, how far does it go?",
    "A store sells apples at 3 for $2. How much for 12 apples?",
    "If x + 5 = 12, what is 2x?",
    "A rectangle has area 24 and width 4. What is the perimeter?",
    "If 20% of a number is 15, what is the number?",
    "A shirt costs $40 after a 20% discount. What was the original price?",
    "How many ways to arrange 5 books on a shelf?",
    "If two angles in a triangle are 45° and 65°, what is the third?",
    "A car uses 8 liters per 100km. How much for 350km?",
    "If 3 workers finish a job in 6 hours, how long for 4 workers?",
    "What is the next number: 2, 6, 12, 20, 30, ?",
    "A clock shows 3:15. What is the angle between the hands?",
    "If you flip 3 coins, what's the probability of exactly 2 heads?",
    "Simplify: (3x² - 6x) / 3x",
    "A pizza is cut into 8 slices. 3 people eat 2 each. What fraction remains?",
]


def build_prompt(task: str, task_type: str) -> str:
    """Build a system+user prompt for the teacher model."""
    if task_type == "tool_use":
        return (
            "You are a helpful assistant that demonstrates tool use. "
            "Given a task, show which tools you would use and how.\n\n"
            f"Task: {task}\n\n"
            "Respond in this format:\n"
            "Thought: <brief reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <parameters as JSON>\n"
            "Observation: <expected result>\n"
            "Final Answer: <summary>"
        )
    elif task_type == "code":
        return (
            "You are an expert programmer. Write clean, well-documented code.\n\n"
            f"Task: {task}\n\n"
            "Provide the solution with a brief explanation."
        )
    elif task_type == "short_cot":
        return (
            "Solve this step by step. Keep each step to one sentence.\n\n"
            f"Problem: {task}\n\n"
            "Show your work in 2-3 steps, then give the final answer."
        )
    return task


async def generate_one(client: httpx.AsyncClient, model: str,
                       prompt: str, temperature: float = 0.7) -> dict:
    """Generate one training example from a teacher model."""
    try:
        resp = await client.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1024,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "prompt": prompt,
            "response": data["choices"][0]["message"]["content"],
            "model": model,
            "task_type": prompt.split("Task: ")[1].split("\n")[0]
            if "Task: " in prompt else "unknown",
        }
    except Exception as e:
        return {"prompt": prompt, "error": str(e), "model": model}


async def generate_batch(tasks: list[str], task_type: str,
                         teacher: str, n_parallel: int = 4) -> list[dict]:
    """Generate training data for a batch of tasks."""
    model = TEACHERS.get(teacher, teacher)
    prompts = [build_prompt(t, task_type) for t in tasks]

    async with httpx.AsyncClient() as client:
        # Process in parallel batches
        results = []
        for i in range(0, len(prompts), n_parallel):
            batch = prompts[i:i + n_parallel]
            coros = [generate_one(client, model, p) for p in batch]
            batch_results = await asyncio.gather(*coros)
            results.extend(batch_results)
            done = min(i + n_parallel, len(prompts))
            print(f"  [{done}/{len(prompts)}] generated", flush=True)
    return results


def save_dataset(data: list[dict], name: str):
    """Save generated data as JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            if "error" not in item:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {sum(1 for d in data if 'error' not in d)} examples to {path}")


# ── Web-augmented generation ─────────────────────────────────────────────────

# Topics for web scraping (coding, LLM, logic)
WEB_TOPICS = {
    "code": [
        "Python async asyncio best practices 2024",
        "Rust ownership borrowing patterns",
        "TypeScript generics advanced patterns",
        "Go concurrency goroutines channels",
        "SQL window functions optimization",
        "React hooks performance optimization",
        "Docker multi-stage build best practices",
        "Git rebase vs merge workflows",
        "Python dataclasses vs pydantic",
        "FastAPI dependency injection patterns",
        "PostgreSQL indexing strategies",
        "Redis caching patterns",
        "Kubernetes deployment yaml examples",
        "CSS grid flexbox layout patterns",
        "Python typing protocol mypy",
    ],
    "llm": [
        "LLM inference optimization techniques 2024",
        "KV cache quantization methods",
        "Mixture of experts architecture",
        "Flash attention implementation",
        "Speculative decoding algorithms",
        "LoRA fine-tuning best practices",
        "RLHF DPO training comparison",
        "Tokenization BPE SentencePiece",
        "Rotary position embedding RoPE",
        "Grouped query attention GQA",
        "Model quantization INT4 INT8",
        "Chain of thought reasoning distillation",
        "Tool use function calling LLM",
        "RAG retrieval augmented generation",
        "MTP multi-token prediction",
    ],
    "logic": [
        "Algorithm design patterns dynamic programming",
        "Graph traversal BFS DFS shortest path",
        "Binary search edge cases patterns",
        "Tree traversal recursion iteration",
        "Bit manipulation tricks algorithms",
        "Sorting algorithms comparison complexity",
        "Hash table collision resolution",
        "Dynamic programming memoization tabulation",
        "Greedy algorithms correctness proof",
        "Backtracking pruning strategies",
    ],
}


# ── Multi-agent collaboration system ─────────────────────────────────────────

# Agent roles — each slot gets a different perspective and seed
AGENT_ROLES = [
    {"name": "questioner", "perspective": "Ask the most fundamental question a beginner would have", "temp": 0.3},
    {"name": "deep_diver", "perspective": "Focus on edge cases and advanced usage", "temp": 0.5},
    {"name": "pragmatist", "perspective": "Focus on practical real-world usage", "temp": 0.4},
    {"name": "analogist", "perspective": "Explain using analogies from other domains", "temp": 0.6},
    {"name": "critic", "perspective": "Find common misconceptions and correct them", "temp": 0.4},
    {"name": "architect", "perspective": "Focus on design patterns and structure", "temp": 0.3},
    {"name": "optimizer", "perspective": "Focus on performance and efficiency", "temp": 0.4},
    {"name": "teacher", "perspective": "Create the clearest explanation for a student", "temp": 0.3},
    {"name": "interviewer", "perspective": "Frame as a technical interview question", "temp": 0.5},
    {"name": "debugger", "perspective": "Focus on common bugs and how to fix them", "temp": 0.5},
    {"name": "comparator", "perspective": "Compare alternatives and trade-offs", "temp": 0.4},
    {"name": "summarizer", "perspective": "Distill to the essential 3 points", "temp": 0.3},
    {"name": "explorer", "perspective": "Explore unusual or creative applications", "temp": 0.7},
    {"name": "historian", "perspective": "Explain the evolution and why it exists", "temp": 0.4},
    {"name": "reviewer", "perspective": "Review as if evaluating a PR, suggest improvements", "temp": 0.3},
    {"name": "challenger", "perspective": "Challenge assumptions, what if the opposite is true?", "temp": 0.6},
    {"name": "simplifier", "perspective": "Explain in the fewest words possible", "temp": 0.3},
    {"name": "detailer", "perspective": "Go deep into implementation details", "temp": 0.4},
    {"name": "tester", "perspective": "Focus on testing strategies and examples", "temp": 0.4},
    {"name": "refactorer", "perspective": "Show bad code then refactor to good code", "temp": 0.5},
    {"name": "theorist", "perspective": "Explain the theoretical foundation", "temp": 0.4},
    {"name": "practitioner", "perspective": "Show what a working professional actually does", "temp": 0.3},
    {"name": "antipattern", "perspective": "Show what NOT to do and why", "temp": 0.5},
    {"name": "connector", "perspective": "Connect this to other related concepts", "temp": 0.5},
    {"name": "troubleshooter", "perspective": "Focus on debugging and error messages", "temp": 0.4},
    {"name": "migrator", "perspective": "Show how to migrate from older approaches", "temp": 0.4},
    {"name": "scalability", "perspective": "Focus on scaling and production concerns", "temp": 0.4},
    {"name": "security", "perspective": "Focus on security implications", "temp": 0.4},
    {"name": "minimalist", "perspective": "Show the absolute minimal working example", "temp": 0.3},
    {"name": "completeness", "perspective": "Cover every aspect exhaustively", "temp": 0.5},
    {"name": "contrarian", "perspective": "Argue against the conventional approach", "temp": 0.6},
    {"name": "synthesizer", "perspective": "Synthesize multiple viewpoints into one answer", "temp": 0.4},
]


async def lfm_call(client: httpx.AsyncClient, messages: list[dict],
                   temperature: float = 0.4, max_tokens: int = 768,
                   seed: int = None, semaphore: asyncio.Semaphore = None) -> str:
    """Core LFM API call with optional seed for diversity."""
    payload = {
        "model": TEACHERS["fast"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    async def _call():
        resp = await client.post(
            f"{LMSTUDIO_BASE}/chat/completions", json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    if semaphore:
        async with semaphore:
            return await _call()
    return await _call()


async def agent_generate_query(client: httpx.AsyncClient, topic: str,
                               role: dict, seed: int,
                               semaphore: asyncio.Semaphore) -> dict:
    """Agent slot generates a search query from its unique perspective."""
    prompt = (
        f"You are a {role['name']} exploring: {topic}\n"
        f"Your perspective: {role['perspective']}\n\n"
        f"Generate a web search query to find content that helps you explain "
        f"this topic from your perspective. Reply with ONLY the search query."
    )
    try:
        query = await lfm_call(client, [{"role": "user", "content": prompt}],
                               temperature=role["temp"], max_tokens=64,
                               seed=seed, semaphore=semaphore)
        return {"role": role["name"], "seed": seed, "query": query.strip().strip('"'), "ok": True}
    except Exception as e:
        return {"role": role["name"], "seed": seed, "error": str(e), "ok": False}


async def agent_generate_draft(client: httpx.AsyncClient, topic: str,
                               content: str, role: dict, seed: int,
                               task_type: str, semaphore: asyncio.Semaphore) -> dict:
    """Agent slot generates a draft Q&A from its perspective with sudo-think."""
    think_prefix = (
        f"<think>\n"
        f"You are a {role['name']}. Your perspective: {role['perspective']}\n"
        f"Topic: {topic}\n"
        f"Think step by step about what makes the best training example from your angle.\n"
        f"What question would capture this? What answer would be most useful?\n"
        f"Consider: accuracy, clarity, completeness, and your unique angle.\n"
        f"</think>\n\n"
    )

    if task_type == "code":
        format_guide = "Q: <specific coding question>\nA: <explanation + working code>"
    elif task_type == "llm":
        format_guide = "Q: <question about the concept>\nA: <concise grounded explanation>"
    else:
        format_guide = "Q: <problem>\nA: <2-3 step solution>"

    prompt = (
        f"{think_prefix}"
        f"Based on this web content, create the best training Q&A you can.\n\n"
        f"Web content:\n{content[:3000]}\n\n"
        f"Format:\n{format_guide}"
    )
    try:
        response = await lfm_call(client, [{"role": "user", "content": prompt}],
                                  temperature=role["temp"], max_tokens=768,
                                  seed=seed, semaphore=semaphore)
        # Strip the think block from final output
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        return {"role": role["name"], "seed": seed, "topic": topic,
                "response": response, "ok": True}
    except Exception as e:
        return {"role": role["name"], "seed": seed, "topic": topic,
                "error": str(e), "ok": False}


async def agent_critique(client: httpx.AsyncClient, drafts: list[dict],
                         topic: str, seed: int,
                         semaphore: asyncio.Semaphore) -> dict:
    """Agent reviews multiple drafts and selects/synthesizes the best."""
    drafts_text = "\n\n---\n\n".join(
        f"[{d['role']}]: {d['response']}" for d in drafts if d.get("ok"))
    if not drafts_text:
        return {"error": "no drafts to critique", "ok": False}

    prompt = (
        f"<think>\n"
        f"You are a senior reviewer. Multiple agents generated training data for: {topic}\n"
        f"Review each draft for accuracy, clarity, and educational value.\n"
        f"Pick the best elements and synthesize them into one superior Q&A pair.\n"
        f"</think>\n\n"
        f"Drafts from {len(drafts)} agents:\n{drafts_text[:6000]}\n\n"
        f"Create the BEST possible training Q&A by combining the strongest elements.\n"
        f"Format:\nQ: <question>\nA: <answer>"
    )
    try:
        response = await lfm_call(client, [{"role": "user", "content": prompt}],
                                  temperature=0.3, max_tokens=768,
                                  seed=seed, semaphore=semaphore)
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        return {"synthesized": response, "ok": True,
                "n_drafts": len([d for d in drafts if d.get("ok")])}
    except Exception as e:
        return {"error": str(e), "ok": False}


async def generate_multi_agent(client: httpx.AsyncClient, topic: str,
                               category: str, n_agents: int,
                               lfm_sem: asyncio.Semaphore,
                               scrape_sem: asyncio.Semaphore) -> dict:
    """Multi-agent pipeline for one topic.

    1. N agents generate diverse search queries (different seeds + perspectives)
    2. Scrape web for all queries (dedup)
    3. N agents write draft Q&A from their perspectives (sudo-think)
    4. One agent synthesizes the best from all drafts
    """
    # Select n_agents roles (cycle through with offset for variety)
    roles = [AGENT_ROLES[i % len(AGENT_ROLES)] for i in range(n_agents)]
    seeds = [random.randint(0, 999999) for _ in range(n_agents)]

    # Phase 1: Diverse query generation
    query_coros = [
        agent_generate_query(client, topic, roles[i], seeds[i], lfm_sem)
        for i in range(n_agents)
    ]
    query_results = await asyncio.gather(*query_coros)

    # Phase 2: Scrape (dedup queries, limit concurrency)
    queries = list(set(r["query"] for r in query_results if r.get("ok") and r.get("query")))
    if not queries:
        return {"topic": topic, "error": "no queries generated", "ok": False}

    async def scrape_one(q: str) -> str:
        async with scrape_sem:
            results = search_all(q, n=2)
            return "\n\n".join(r.content for r in results if r.content)

    scrape_coros = [scrape_one(q) for q in queries[:4]]  # max 4 queries per topic
    scrape_results = await asyncio.gather(*scrape_coros)
    content = "\n\n".join(c for c in scrape_results if c)

    if not content:
        return {"topic": topic, "error": "no content scraped", "ok": False}

    # Phase 3: Draft generation (all agents, different seeds + perspectives)
    draft_coros = [
        agent_generate_draft(client, topic, content, roles[i], seeds[i],
                             category, lfm_sem)
        for i in range(n_agents)
    ]
    drafts = await asyncio.gather(*draft_coros)
    ok_drafts = [d for d in drafts if d.get("ok")]
    if not ok_drafts:
        return {"topic": topic, "error": "no drafts generated", "ok": False}

    # Phase 4: Synthesis — one agent reviews all drafts and creates best
    synth_seed = random.randint(0, 999999)
    synth = await agent_critique(client, drafts, topic, synth_seed, lfm_sem)

    if not synth.get("ok"):
        # Fallback: use best individual draft (first that succeeded)
        return {"topic": topic, "task_type": category,
                "response": ok_drafts[0]["response"],
                "model": TEACHERS["fast"],
                "n_agents": n_agents, "n_drafts": len(ok_drafts),
                "synthesized": False, "ok": True}

    return {"topic": topic, "task_type": category,
            "response": synth["synthesized"],
            "model": TEACHERS["fast"],
            "n_agents": n_agents, "n_drafts": len(ok_drafts),
            "synthesized": True, "ok": True,
            "web_content": content[:500]}


async def generate_grounded_example(client: httpx.AsyncClient, model: str,
                                    topic: str, scraped_content: str,
                                    task_type: str,
                                    semaphore: asyncio.Semaphore = None) -> dict:
    """Use a model to create training data grounded in web content.

    With LFM as the processor, this runs at 32 concurrent — no teacher bottleneck.
    LFM summarizes scraped content directly into Q&A format.
    """
    if task_type == "code":
        prompt = (
            "You are creating training data. Summarize this web content into a "
            "clear coding Q&A pair.\n\n"
            f"Topic: {topic}\n"
            f"Web content:\n{scraped_content[:4000]}\n\n"
            "Format:\n"
            "Q: <specific coding question>\n"
            "A: <concise answer with code example>"
        )
    elif task_type == "llm":
        prompt = (
            "You are creating training data. Summarize this web content into a "
            "clear Q&A pair about LLM concepts.\n\n"
            f"Topic: {topic}\n"
            f"Web content:\n{scraped_content[:4000]}\n\n"
            "Format:\n"
            "Q: <question about the concept>\n"
            "A: <concise explanation grounded in the content>"
        )
    else:  # logic
        prompt = (
            "You are creating training data. Summarize this web content into a "
            "logic/algorithm problem with solution.\n\n"
            f"Topic: {topic}\n"
            f"Web content:\n{scraped_content[:4000]}\n\n"
            "Format:\n"
            "Q: <problem>\n"
            "A: <2-3 step solution>"
        )

    async def _call():
        resp = await client.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 768,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    if semaphore:
        async with semaphore:
            response = await _call()
    else:
        response = await _call()

    return {
        "topic": topic,
        "task_type": task_type,
        "web_content": scraped_content[:500],
        "prompt": prompt,
        "response": response,
        "model": model,
    }


async def generate_web_augmented(category: str, n: int, teacher: str,
                                 n_parallel: int = 32, teacher_parallel: int = 4,
                                 n_agents: int = 4) -> list[dict]:
    """Multi-agent web-augmented generation pipeline.

    For each topic, n_agents LFM slots collaborate:
    1. Each agent generates a unique search query (different seed + perspective)
    2. Web scrapers fetch content for all queries
    3. Each agent writes a draft Q&A from its perspective (sudo-think)
    4. A synthesis agent combines the best elements into one superior example

    All 32 LM Studio slots are used concurrently across topics + agents.
    Each agent gets a random seed for diverse outputs.
    """
    topics = WEB_TOPICS.get(category, [])
    if not topics:
        print(f"  Unknown category: {category}")
        return []

    # Expand topics list to n entries
    topic_list = []
    for i in range(n):
        base = topics[i % len(topics)]
        topic_list.append(base if i < len(topics) else f"{base} (aspect {i // len(topics) + 1})")

    lfm_sem = asyncio.Semaphore(n_parallel)
    scrape_sem = asyncio.Semaphore(4)

    async with httpx.AsyncClient() as client:
        print(f"  Multi-agent: {n} topics x {n_agents} agents = {n * n_agents} LFM calls", flush=True)
        print(f"  {n_parallel} concurrent slots, each agent has unique seed + perspective", flush=True)

        # Run all topics concurrently — each topic spawns n_agents internally
        topic_coros = [
            generate_multi_agent(client, t, category, n_agents, lfm_sem, scrape_sem)
            for t in topic_list
        ]
        results = await asyncio.gather(*topic_coros)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning data via LM Studio")
    parser.add_argument("--task", choices=["tool_use", "code", "short_cot", "all",
                                           "web_code", "web_llm", "web_logic", "web_all"],
                        default="all", help="Type of training data to generate")
    parser.add_argument("--n", type=int, default=100, help="Examples per task type")
    parser.add_argument("--teacher", default="fast",
                        help="Processor model: 'fast' (LFM, 32 concurrent) or teacher key (4-8 concurrent)")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel teacher requests (ignored with --teacher fast)")
    parser.add_argument("--lfm-parallel", type=int, default=32,
                        help="Parallel LFM requests. Max 32 for LM Studio.")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of agent slots per topic (multi-agent collab). 4-8 recommended.")
    args = parser.parse_args()

    # Web-augmented tasks
    web_categories = {
        "web_code": "code",
        "web_llm": "llm",
        "web_logic": "logic",
    }

    if args.task in web_categories or args.task == "web_all":
        cats = [web_categories[args.task]] if args.task != "web_all" else list(web_categories.values())
        for cat in cats:
            print(f"\n{'='*60}")
            print(f"Web-augmented generation: {cat} ({args.n} examples)")
            print(f"  LFM parallel: {args.lfm_parallel}, agents per topic: {args.agents}")
            print(f"{'='*60}")

            teacher = args.teacher
            t0 = time.time()
            results = asyncio.run(generate_web_augmented(
                cat, args.n, teacher,
                n_parallel=args.lfm_parallel, teacher_parallel=args.parallel,
                n_agents=args.agents))
            t1 = time.time()
            save_dataset(results, f"web_{cat}_{args.n}")
            print(f"Completed in {t1-t0:.1f}s")
        return

    # Direct generation tasks
    task_configs = {
        "tool_use": (TOOL_USE_TASKS, "adi-qwen2.5-14b-glm5.2-general"),
        "code": (CODE_TASKS, "qwen2.5-coder-7b-instruct"),
        "short_cot": (SHORT_COT_TASKS, "grok-3-reasoning-gemma3-12b-distilled"),
    }

    tasks_to_run = [args.task] if args.task != "all" else list(task_configs.keys())

    for task_type in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Generating {task_type} data ({args.n} examples)")
        print(f"{'='*60}")

        base_tasks, default_teacher = task_configs[task_type]
        teacher = args.teacher if args.teacher != "general" else default_teacher

        tasks = []
        for i in range(args.n):
            base = base_tasks[i % len(base_tasks)]
            if i >= len(base_tasks):
                tasks.append(f"{base} (variation {i // len(base_tasks) + 1})")
            else:
                tasks.append(base)

        t0 = time.time()
        results = asyncio.run(generate_batch(tasks, task_type, teacher, args.parallel))
        t1 = time.time()

        save_dataset(results, f"{task_type}_{args.n}")
        print(f"Completed in {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
