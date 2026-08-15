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
    "general": "adi-qwen2.5-14b-glm5.2-general",  # Qwen 2.5 14B distilled from GLM 5.2 (primary teacher)
    "reasoning": "adi-qwen2.5-14b-glm5.2-general",
    "code": "adi-qwen2.5-14b-glm5.2-general",
    "fast": "liquid/lfm2.5-1.2b",  # LFM for query generation
    "glm52": "adi-qwen2.5-14b-glm5.2-general",  # alias
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
    "Implement a binary search function that returns the index or -1",
    "Write a Python function to reverse a linked list iteratively",
    "Create a function to check if a string is a palindrome",
    "Implement a stack using a linked list",
    "Write a function to find the kth largest element in an array",
    "Create a simple HTTP server using Python's http.server",
    "Write a function to compress a string using run-length encoding",
    "Implement a basic hash map from scratch in Python",
    "Write a function to validate a JSON string and return parsed data",
    "Create a Python class for a circular queue",
    "Write a function to find all prime numbers up to n using Sieve of Eratosthenes",
    "Implement depth-first search on a graph represented as adjacency list",
    "Write a function to balance parentheses in a string",
    "Create a simple middleware pattern in Python for request processing",
    "Write a function to rotate a 2D matrix 90 degrees clockwise",
    "Implement a basic trie (prefix tree) with insert and search",
    "Write a Python script to rename files in bulk by pattern",
    "Create a function to merge two dictionaries with conflict resolution",
    "Write a function to find the intersection of two sorted arrays",
    "Implement a basic event emitter in Python",
    "Write a SQL query to find duplicate records in a table",
    "Create a Python function to format a duration in seconds to human readable",
    "Write a function to check if two strings are anagrams",
    "Implement a sliding window algorithm to find max sum subarray of size k",
    "Write a function to serialize and deserialize a binary tree",
    "Create a simple retry decorator with exponential backoff",
    "Write a function to flatten a nested list of arbitrary depth",
    "Implement a basic bloom filter in Python",
    "Write a function to find the first non-repeating character in a string",
    "Create a Python context manager for database transactions",
    "Write a function to convert camelCase to snake_case",
    "Implement a basic rate limiter using the token bucket algorithm",
    "Write a function to find the longest palindromic substring",
    "Create a simple observer pattern implementation in Python",
    "Write a function to validate an IP address (IPv4)",
    "Implement a basic command pattern with undo support",
    "Write a function to generate a Sudoku puzzle",
    "Create a Python script to monitor disk space and alert if low",
    "Write a function to parse a URL and extract query parameters",
    "Implement a basic connection pool for database connections",
    "Write a function to find the GCD of two numbers using Euclid's algorithm",
]

SHORT_COT_TASKS = [
    "If a train travels 60 km/h for 2.5 hours, how far does it go?",
    "A store sells apples at 3 for $2. How much for 12 apples?",
    "If x + 5 = 12, what is 2x?",
    "A rectangle has area 24 and width 4. What is the perimeter?",
    "If 20% of a number is 15, what is the number?",
    "A shirt costs $40 after a 20% discount. What was the original price?",
    "How many ways to arrange 5 books on a shelf?",
    "If two angles in a triangle are 45 and 65 degrees, what is the third?",
    "A car uses 8 liters per 100km. How much for 350km?",
    "If 3 workers finish a job in 6 hours, how long for 4 workers?",
    "What is the next number: 2, 6, 12, 20, 30, ?",
    "A clock shows 3:15. What is the angle between the hands?",
    "If you flip 3 coins, what's the probability of exactly 2 heads?",
    "Simplify: (3x^2 - 6x) / 3x",
    "A pizza is cut into 8 slices. 3 people eat 2 each. What fraction remains?",
    "If 5x - 3 = 2x + 12, what is x?",
    "A tank fills in 4 hours with pipe A and empties in 6 hours with pipe B. How long to fill with both open?",
    "What is 15% of 240?",
    "If the sum of two numbers is 50 and their difference is 10, what are the numbers?",
    "A circle has radius 5. What is its circumference? (use pi = 3.14)",
    "How many diagonals does a hexagon have?",
    "If 2^x = 32, what is x?",
    "A bag has 4 red and 6 blue marbles. What's the probability of drawing red twice without replacement?",
    "Convert 3/8 to a decimal and percentage.",
    "If a square has diagonal length 10, what is its area?",
    "What is the average of 12, 15, 18, 21, and 24?",
    "A product costs $80 with 15% tax. What is the total price?",
    "If log2(x) = 5, what is x?",
    "How many ways to choose 3 items from 7?",
    "A train 150m long passes a pole in 10 seconds. What is its speed in km/h?",
    "What is the next prime after 23?",
    "If 3/4 of a number is 36, what is the number?",
    "A right triangle has legs 3 and 4. What is the hypotenuse?",
    "What is 7! divided by 5!?",
    "If you invest $5000 at 4% simple interest for 3 years, how much interest do you earn?",
    "A cube has surface area 54. What is its volume?",
    "What is the least common multiple of 6 and 8?",
    "If the temperature drops from 15C to -3C, how much did it drop?",
    "A recipe needs 2.5 cups of flour for 4 servings. How much for 10 servings?",
    "What is the sum of the first 10 natural numbers?",
    "If 3x + 7 = 22, what is x?",
    "A garden is 12m by 8m. What is the cost to fence it at $5 per meter?",
    "What is the derivative of x^3 + 2x?",
    "If sin(theta) = 0.5, what is theta in degrees?",
    "How many edges does a triangular prism have?",
    "A car travels 120 miles in 2 hours 30 minutes. What is the average speed?",
    "What is 2^10?",
    "If the ratio of boys to girls in a class is 3:2 and there are 30 students, how many girls?",
    "Simplify: 2(3x - 4) + 3(x + 2)",
    "A water bottle holds 750ml. How many bottles fill a 3-liter container?",
]


# ── OpenAI function-calling tool catalog ─────────────────────────────────────
# A fixed set of tools with JSON-schema arguments. The teacher is asked to
# produce a multi-turn conversation that uses these tools to solve a task.
TOOL_CATALOG = [
    {"name": "search_flights", "description": "Search for flights between two cities on a date.",
     "parameters": {"origin": "string (city/IATA code)", "destination": "string (city/IATA code)",
                    "date": "string (YYYY-MM-DD)", "passengers": "integer (default 1)"}},
    {"name": "get_weather", "description": "Get current weather for a city.",
     "parameters": {"city": "string", "units": "string (celsius or fahrenheit, default celsius)"}},
    {"name": "send_email", "description": "Send an email to recipients.",
     "parameters": {"to": "array of string (email addresses)", "subject": "string", "body": "string"}},
    {"name": "calculate", "description": "Evaluate a math expression and return the result.",
     "parameters": {"expression": "string (e.g. '(15 + 27) * 0.8')"}},
    {"name": "query_database", "description": "Run a read-only SQL query against the users DB.",
     "parameters": {"sql": "string (SELECT query only)"}},
    {"name": "calendar_create", "description": "Create a calendar event with reminders.",
     "parameters": {"title": "string", "start": "string (ISO 8601)", "end": "string (ISO 8601)",
                    "attendees": "array of string (emails)", "reminder_minutes": "integer (default 10)"}},
    {"name": "http_get", "description": "Fetch the content at a URL (GET request).",
     "parameters": {"url": "string", "headers": "object (optional)"}},
    {"name": "parse_csv", "description": "Parse CSV text and return rows as JSON.",
     "parameters": {"csv_text": "string", "delimiter": "string (default ',')"}},
    {"name": "translate", "description": "Translate text to a target language.",
     "parameters": {"text": "string", "target_lang": "string (ISO 639-1 code)"}},
    {"name": "geocode", "description": "Convert an address to lat/long coordinates.",
     "parameters": {"address": "string"}},
    {"name": "currency_convert", "description": "Convert an amount between currencies.",
     "parameters": {"amount": "number", "from_currency": "string (ISO 4217)", "to_currency": "string (ISO 4217)"}},
    {"name": "image_resize", "description": "Resize and optionally compress an image.",
     "parameters": {"path": "string", "width": "integer", "height": "integer",
                    "quality": "integer (1-100, default 85)"}},
    {"name": "process_payment", "description": "Charge a payment method with retry logic.",
     "parameters": {"amount": "number", "currency": "string", "method_id": "string",
                    "retry_max": "integer (default 3)"}},
    {"name": "validate_form", "description": "Validate a form submission, return per-field errors.",
     "parameters": {"fields": "object (field name -> value)"}},
    {"name": "scrape_prices", "description": "Scrape product prices from an e-commerce page.",
     "parameters": {"url": "string", "selector": "string (CSS selector, optional)"}},
    {"name": "create_task", "description": "Create a task in a project management system.",
     "parameters": {"title": "string", "description": "string", "assignee": "string (email)",
                    "priority": "string (low, medium, high, urgent)", "due_date": "string (YYYY-MM-DD)"}},
    {"name": "get_stock_price", "description": "Get the current stock price for a ticker symbol.",
     "parameters": {"ticker": "string (e.g. AAPL)", "exchange": "string (default NASDAQ)"}},
    {"name": "list_files", "description": "List files in a directory.",
     "parameters": {"path": "string", "pattern": "string (glob pattern, optional)"}},
    {"name": "read_file", "description": "Read the contents of a text file.",
     "parameters": {"path": "string"}},
    {"name": "write_file", "description": "Write content to a file.",
     "parameters": {"path": "string", "content": "string"}},
    {"name": "search_web", "description": "Search the web and return top results.",
     "parameters": {"query": "string", "n": "integer (default 5)"}},
    {"name": "get_directions", "description": "Get driving directions between two addresses.",
     "parameters": {"origin": "string", "destination": "string",
                    "mode": "string (driving, walking, transit, default driving)"}},
]


# ── Generalized tool-use: random tools with novel names ──────────────────────
# These teach the model to READ tool schemas from the user message and use them,
# rather than memorizing a fixed catalog. Each generated example includes 1-3
# randomly-named tools with random argument schemas.

# Pool of tool "verbs" and "nouns" for generating novel tool names.
_TOOL_VERBS = ["get", "fetch", "retrieve", "search", "query", "list", "create",
               "update", "delete", "send", "calculate", "convert", "parse",
               "validate", "check", "monitor", "export", "import", "sync",
               "analyze", "scan", "lookup", "register", "schedule", "notify"]
_TOOL_NOUNS = ["data", "record", "status", "report", "config", "metric",
               "event", "log", "file", "user", "order", "ticket", "message",
               "notification", "inventory", "transaction", "session", "profile",
               "schema", "endpoint", "resource", "task", "alert", "summary",
               "forecast", "history", "preference", "subscription", "invoice",
               "shipment", "reservation", "feedback", "review", "comment"]

# Argument type templates for generating random schemas.
_ARG_TYPES = [
    "string", "integer", "number", "boolean",
    "string (email address)", "string (ISO date YYYY-MM-DD)",
    "string (URL)", "string (enum: low, medium, high)",
    "array of string", "object with keys: name, value",
    "string (JSON path)", "string (comma-separated list)",
]

# Rule constraints to test instruction-following.
_RULES = [
    None,  # no rule (most common)
    None,
    None,
    "Respond in exactly one sentence.",
    "Format your final answer as a JSON object with key 'result'.",
    "Keep your final answer under 20 words.",
    "Start your final answer with 'Result:'",
    "Mention the tool name you used in your final answer.",
]


def _random_tool_name(rng: random.Random) -> str:
    """Generate a novel tool name like 'fetch_record' or 'sync_inventory'."""
    verb = rng.choice(_TOOL_VERBS)
    noun = rng.choice(_TOOL_NOUNS)
    return f"{verb}_{noun}"


def _random_tool_schema(rng: random.Random, name: str) -> dict:
    """Generate a random tool with a plausible description and 1-4 arguments."""
    n_args = rng.randint(1, 4)
    arg_names = rng.sample(
        ["id", "name", "query", "type", "value", "count", "limit", "offset",
         "filter", "sort", "format", "target", "source", "destination",
         "start", "end", "mode", "priority", "category", "status",
         "url", "path", "email", "date", "amount", "key", "field", "data"],
        min(n_args, 28),
    )
    params = {arg: rng.choice(_ARG_TYPES) for arg in arg_names}
    # Pick a description that roughly matches the name.
    verb, noun = name.split("_", 1)
    descriptions = {
        "get": f"Get the {noun} for a given identifier.",
        "fetch": f"Fetch {noun} from the external system.",
        "retrieve": f"Retrieve {noun} records matching criteria.",
        "search": f"Search for {noun} entries.",
        "query": f"Query the {noun} database.",
        "list": f"List all {noun} entries.",
        "create": f"Create a new {noun} entry.",
        "update": f"Update an existing {noun} record.",
        "delete": f"Delete a {noun} by id.",
        "send": f"Send a {noun} to recipients.",
        "calculate": f"Calculate {noun} from input parameters.",
        "convert": f"Convert {noun} between formats.",
        "parse": f"Parse {noun} input and return structured data.",
        "validate": f"Validate {noun} input and return errors.",
        "check": f"Check {noun} status.",
        "monitor": f"Monitor {noun} and return current metrics.",
        "export": f"Export {noun} to a file format.",
        "import": f"Import {noun} from external source.",
        "sync": f"Sync {noun} with the remote server.",
        "analyze": f"Analyze {noun} and return insights.",
        "scan": f"Scan {noun} for issues.",
        "lookup": f"Look up {noun} by key.",
        "register": f"Register a new {noun}.",
        "schedule": f"Schedule a {noun} for a future time.",
        "notify": f"Send a {noun} notification.",
    }
    desc = descriptions.get(verb, f"Perform {verb} operation on {noun}.")
    return {"name": name, "description": desc, "parameters": params}


def _random_task_for_tools(rng: random.Random, tools: list[dict]) -> str:
    """Generate a task description that requires using the given tools."""
    tool_names = [t["name"] for t in tools]
    if len(tools) == 1:
        t = tools[0]
        args = list(t["parameters"].keys())
        arg_str = ", ".join(f"a {args[0]}" if len(args) == 1 else f"a {args[0]} and {args[1]}" for _ in [0]) if args else "something"
        templates = [
            f"Use the {t['name']} tool to {t['description'].lower().rstrip('.')} with {arg_str}.",
            f"I need you to {t['description'].lower().rstrip('.')} Call {t['name']} with appropriate arguments.",
            f"Please call {t['name']} to handle this: {t['description'].lower().rstrip('.')}",
        ]
        return rng.choice(templates)
    elif len(tools) == 2:
        t1, t2 = tools
        templates = [
            f"First use {t1['name']} to {t1['description'].lower().rstrip('.')}, "
            f"then use {t2['name']} to {t2['description'].lower().rstrip('.')}",
            f"I need two things done: {t1['description'].lower().rstrip('.')} using {t1['name']}, "
            f"and {t2['description'].lower().rstrip('.')} using {t2['name']}.",
            f"Call {t1['name']} first, then based on the result call {t2['name']}. "
            f"The goal is to {t1['description'].lower().rstrip('.')} and then {t2['description'].lower().rstrip('.')}.",
        ]
        return rng.choice(templates)
    else:
        t1, t2, t3 = tools[0], tools[1], tools[2]
        return (
            f"Perform three steps: (1) call {t1['name']} to {t1['description'].lower().rstrip('.')}, "
            f"(2) call {t2['name']} to {t2['description'].lower().rstrip('.')}, "
            f"(3) call {t3['name']} to {t3['description'].lower().rstrip('.')}."
        )


def build_generalized_prompt(rng: random.Random) -> str:
    """Build a complete prompt for generalized tool-use.

    The prompt includes:
    - 1-3 randomly generated tools with schemas in the USER message
    - A task that requires using those tools
    - An optional rule constraint
    - Instructions to output a multi-turn function-calling conversation
    """
    n_tools = rng.choices([1, 2, 3], weights=[3, 4, 2])[0]
    tools = []
    used_names = set()
    for _ in range(n_tools):
        for _ in range(10):  # retry to avoid duplicates
            name = _random_tool_name(rng)
            if name not in used_names:
                used_names.add(name)
                tools.append(_random_tool_schema(rng, name))
                break

    task = _random_task_for_tools(rng, tools)
    rule = rng.choice(_RULES)

    # Build the tools section for the user message.
    tools_text = "\n".join(
        f"- {t['name']}: {t['description']}\n"
        f"    arguments: {json.dumps(t['parameters'])}"
        for t in tools
    )

    valid_names = [t["name"] for t in tools]

    prompt = (
        f"You have access to these tools:\n{tools_text}\n\n"
        f"Task: {task}\n\n"
    )
    if rule:
        prompt += f"Rule: {rule}\n\n"

    prompt += (
        "Produce a multi-turn conversation that solves the task using ONLY the tools above. "
        "Output ONLY a JSON object with this exact shape (no markdown, no commentary):\n"
        "{\n"
        '  "messages": [\n'
        '    {"role": "user", "content": "<the task as a natural user message>"},\n'
        '    {"role": "assistant", "content": null, "tool_calls": '
        '[{"name": "<tool_name>", "arguments": {<args>}}]},\n'
        '    {"role": "tool", "name": "<tool_name>", "content": "<plausible result>"},\n'
        '    {"role": "assistant", "content": "<final answer>"}\n'
        "  ]\n"
        "}\n\n"
        f"Rules:\n"
        f"1. Use ONLY these tools: {', '.join(valid_names)}. Match argument names exactly.\n"
        "2. If multiple tools are needed, call them in sequence (tool call → result → next tool call).\n"
        "3. Tool results must be plausible and concrete. Keep each on a single line.\n"
        "4. The final assistant message must answer the user's task.\n"
        "5. If a rule is specified, follow it in your final answer.\n"
        "6. Output valid JSON only."
    )
    return prompt


# Generate generalized task prompts (pre-built for the batch generator).
def build_generalized_tasks(n: int, seed: int = 42) -> list[str]:
    """Generate n generalized tool-use prompts with random tools and schemas."""
    rng = random.Random(seed)
    return [build_generalized_prompt(rng) for _ in range(n)]


# ── Targeted task types for specific failure modes ─────────────────────────

# Rules with code-verifiable constraints (from VerIF/MulDimIF research).
_VERIFIABLE_RULES = [
    ("Respond in exactly {n} words.", lambda ans, n=10: abs(len(ans.split()) - n) <= 1),
    ("Format your answer as JSON: {{\"result\": \"...\"}}", lambda ans: ans.strip().startswith("{") and "result" in ans),
    ("Keep your answer under {n} words.", lambda ans, n=15: len(ans.split()) <= n),
    ("Start your answer with '{prefix}'", lambda ans, prefix="Result:": ans.strip().startswith(prefix)),
    ("End your answer with '{suffix}'", lambda ans, suffix="done.": ans.strip().lower().endswith(suffix)),
    ("Use only lowercase in your answer.", lambda ans: ans == ans.lower()),
    ("Do not use the word '{word}' in your answer.", lambda ans, word="the": word not in ans.lower().split()),
    ("Include the number {n} in your answer.", lambda ans, n=42: str(n) in ans),
]


def _build_multi_tool_prompt(rng: random.Random) -> str:
    """Build a prompt that requires using 2-3 tools.

    Two modes:
    - sequential: tools called in separate turns (output of one feeds next)
    - parallel: multiple tool calls in a single assistant message (independent calls)
    """
    mode = rng.choice(["sequential", "parallel", "parallel"])
    n_tools = rng.choices([2, 3], weights=[3, 2])[0]
    tools = []
    used_names = set()
    for _ in range(n_tools):
        for _ in range(10):
            name = _random_tool_name(rng)
            if name not in used_names:
                used_names.add(name)
                tools.append(_random_tool_schema(rng, name))
                break

    tools_text = "\n".join(
        f"- {t['name']}: {t['description']}\n"
        f"    arguments: {json.dumps(t['parameters'])}"
        for t in tools
    )
    valid_names = [t["name"] for t in tools]

    if mode == "parallel":
        # Parallel: all calls in one assistant message, each gets its own tool result
        tool_list = ", ".join(t["name"] for t in tools)
        task = (
            f"Use these tools in one go: {tool_list}. "
            + " ".join(f"Call {t['name']} to {t['description'].lower().rstrip('.')}." for t in tools)
        )
        # Build the JSON template with multiple tool_calls in one message
        calls_template = ", ".join(
            f'{{"name": "{t["name"]}", "arguments": {{...}}}}' for t in tools
        )
        tool_results = "\n".join(
            f'    {{"role": "tool", "name": "{t["name"]}", "content": "<result>"}},'
            for t in tools
        )
        prompt = (
            f"You have access to these tools:\n{tools_text}\n\n"
            f"Task: {task}\n\n"
            "Output ONLY a JSON object. Make ALL tool calls in a SINGLE assistant message:\n"
            "{\n"
            '  "messages": [\n'
            '    {"role": "user", "content": "<task>"},\n'
            f'    {{"role": "assistant", "content": null, "tool_calls": [{calls_template}]}},\n'
            f'{tool_results}\n'
            '    {"role": "assistant", "content": "<final answer using all results>"}\n'
            "  ]\n"
            "}\n\n"
            f"Rules:\n"
            f"1. Use ONLY these tools: {', '.join(valid_names)}.\n"
            "2. Put ALL tool calls in ONE assistant message (parallel calls).\n"
            "3. Each tool call gets its own tool result.\n"
            "4. The final assistant message answers the task using all results.\n"
            "5. Output valid JSON only."
        )
    else:
        # Sequential: one call per turn, output feeds next
        if n_tools == 2:
            t1, t2 = tools
            task = (
                f"First call {t1['name']} to {t1['description'].lower().rstrip('.')}, "
                f"then use the result to call {t2['name']} to {t2['description'].lower().rstrip('.')}. "
            )
        else:
            t1, t2, t3 = tools
            task = (
                f"Perform these steps in order: "
                f"(1) call {t1['name']} to {t1['description'].lower().rstrip('.')}, "
                f"(2) call {t2['name']} using the result from step 1, "
                f"(3) call {t3['name']} using the result from step 2."
            )
        prompt = (
            f"You have access to these tools:\n{tools_text}\n\n"
            f"Task: {task}\n\n"
            "Produce a multi-turn conversation where each assistant turn makes ONE tool call, "
            "waits for the result, then makes the next call. Output ONLY a JSON object:\n"
            "{\n"
            '  "messages": [\n'
            '    {"role": "user", "content": "<task as natural message>"},\n'
            '    {"role": "assistant", "content": null, "tool_calls": [{"name": "<tool1>", "arguments": {...}}]},\n'
            '    {"role": "tool", "name": "<tool1>", "content": "<result>"},\n'
            '    {"role": "assistant", "content": null, "tool_calls": [{"name": "<tool2>", "arguments": {...}}]},\n'
            '    {"role": "tool", "name": "<tool2>", "content": "<result>"},\n'
            '    {"role": "assistant", "content": "<final answer>"}\n'
            "  ]\n"
            "}\n\n"
            f"Rules:\n"
            f"1. Use ONLY these tools: {', '.join(valid_names)}.\n"
            "2. Each assistant turn makes exactly ONE tool call, then waits for the result.\n"
            "3. The final assistant message answers the task using the tool results.\n"
            "4. Output valid JSON only."
        )
    return prompt


def _build_rule_following_prompt(rng: random.Random) -> str:
    """Build a prompt with a verifiable rule constraint.

    The model must follow the rule in its final answer. This teaches
    instruction-following, not just tool calling.
    """
    # 50% chance of having tools, 50% chance of pure instruction following
    has_tools = rng.random() < 0.5

    rule_template, _ = rng.choice(_VERIFIABLE_RULES)
    # Fill in template parameters
    n = rng.choice([5, 10, 15, 20])
    prefix = rng.choice(["Result:", "Answer:", "Done:", "OK:"])
    suffix = rng.choice(["done.", "complete.", "finished."])
    word = rng.choice(["the", "and", "is", "a"])
    rule = rule_template.format(n=n, prefix=prefix, suffix=suffix, word=word)

    if has_tools:
        # Tool-use + rule
        name = _random_tool_name(rng)
        tool = _random_tool_schema(rng, name)
        tools_text = f"- {tool['name']}: {tool['description']}\n    arguments: {json.dumps(tool['parameters'])}"

        tasks = [
            f"Use {name} to get the information, then answer. {rule}",
            f"Call {name} and summarize the result. {rule}",
            f"Use {name} to look up the data. {rule}",
        ]
        task = rng.choice(tasks)

        prompt = (
            f"You have access to these tools:\n{tools_text}\n\n"
            f"Task: {task}\n\n"
            "Output ONLY a JSON object:\n"
            "{\n"
            '  "messages": [\n'
            '    {"role": "user", "content": "<task>"},\n'
            '    {"role": "assistant", "content": null, "tool_calls": [{"name": "<tool>", "arguments": {...}}]},\n'
            '    {"role": "tool", "name": "<tool>", "content": "<result>"},\n'
            '    {"role": "assistant", "content": "<final answer following the rule>"}\n'
            "  ]\n"
            "}\n\n"
            f"Rules:\n1. Use the tool: {name}.\n2. {rule}\n3. Output valid JSON only."
        )
    else:
        # Pure instruction following (no tools)
        tasks = [
            f"What is the capital of France? {rule}",
            f"Explain what Python is. {rule}",
            f"What is 2+2? {rule}",
            f"Name a primary color. {rule}",
            f"What does CPU stand for? {rule}",
            f"List one programming language. {rule}",
            f"What is the largest planet? {rule}",
            f"Define recursion. {rule}",
        ]
        task = rng.choice(tasks)
        prompt = (
            f"Task: {task}\n\n"
            "Output ONLY a JSON object:\n"
            "{\n"
            '  "messages": [\n'
            '    {"role": "user", "content": "<task>"},\n'
            '    {"role": "assistant", "content": "<answer following the rule exactly>"}\n'
            "  ]\n"
            "}\n\n"
            f"Rules:\n1. {rule}\n2. Output valid JSON only."
        )
    return prompt


def _build_nested_arg_prompt(rng: random.Random) -> str:
    """Build a prompt with tools that have nested object/array arguments.

    This teaches the model to generate correct nested JSON.
    """
    n_tools = rng.choices([1, 2], weights=[3, 2])[0]
    tools = []
    used_names = set()
    for _ in range(n_tools):
        for _ in range(10):
            name = _random_tool_name(rng)
            if name not in used_names:
                used_names.add(name)
                # Force at least one nested argument
                tool = _random_tool_schema(rng, name)
                # Replace one argument with a nested object type
                if tool["parameters"]:
                    arg_key = rng.choice(list(tool["parameters"].keys()))
                    tool["parameters"][arg_key] = rng.choice([
                        "object with keys: name, value, type",
                        "object with keys: id, label, count",
                        "array of string",
                        "object with keys: street, city, zip",
                        "object with keys: query, filters, sort_by",
                    ])
                tools.append(tool)
                break

    tools_text = "\n".join(
        f"- {t['name']}: {t['description']}\n"
        f"    arguments: {json.dumps(t['parameters'])}"
        for t in tools
    )
    valid_names = [t["name"] for t in tools]

    # Build task that requires using the nested argument
    t1 = tools[0]
    nested_arg = None
    for k, v in t1["parameters"].items():
        if "object" in v or "array" in v:
            nested_arg = k
            break

    if nested_arg:
        task = (
            f"Call {t1['name']} and provide a valid value for the '{nested_arg}' argument "
            f"as a {t1['parameters'][nested_arg]}. "
        )
    else:
        task = f"Call {t1['name']} with appropriate arguments. "

    if n_tools == 2:
        t2 = tools[1]
        task += f"Then call {t2['name']} to {t2['description'].lower().rstrip('.')}."

    prompt = (
        f"You have access to these tools:\n{tools_text}\n\n"
        f"Task: {task}\n\n"
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "messages": [\n'
        '    {"role": "user", "content": "<task>"},\n'
        '    {"role": "assistant", "content": null, "tool_calls": [{"name": "<tool>", "arguments": {...}}]},\n'
        '    {"role": "tool", "name": "<tool>", "content": "<result>"},\n'
        '    {"role": "assistant", "content": "<final answer>"}\n'
        "  ]\n"
        "}\n\n"
        f"Rules:\n"
        f"1. Use ONLY these tools: {', '.join(valid_names)}.\n"
        "2. For object-type arguments, provide a valid JSON object with the specified keys.\n"
        "3. For array-type arguments, provide a JSON array of strings.\n"
        "4. Output valid JSON only. Ensure nested objects are properly formatted."
    )
    return prompt


def build_targeted_tasks(n: int, seed: int = 42) -> list[str]:
    """Generate n targeted prompts covering multi-tool chaining, rule following,
    and nested arguments.

    Multi-tool gets 50% (the remaining failure mode), rules 25%, nested 25%.
    """
    rng = random.Random(seed)
    n_multi = n // 2
    n_rest = (n - n_multi) // 2
    n_nested = n - n_multi - n_rest  # absorb remainder
    tasks = []
    tasks.extend(_build_multi_tool_prompt(rng) for _ in range(n_multi))
    tasks.extend(_build_rule_following_prompt(rng) for _ in range(n_rest))
    tasks.extend(_build_nested_arg_prompt(rng) for _ in range(n_nested))
    rng.shuffle(tasks)
    return tasks

# Tasks that require tool use. Each is paired with the catalog above to produce
# a multi-turn function-calling trajectory.
TOOL_USE_FC_TASKS = [
    # Travel + communication
    "I need to fly from NYC to London on 2026-09-15 with 2 passengers. Find options and email the summary to boss@company.com.",
    "Book a flight NYC→Paris on 2026-10-01, get Paris weather for arrival day, and email the plan to me@home.com.",
    "Find flights from LAX to Tokyo on 2026-11-20 for 3 people, then create a calendar event for the departure.",
    "Search for flights from Chicago to Miami on 2026-12-01, get Miami weather, and email the trip plan to family@home.com.",
    "Get weather for Berlin, Madrid, and Rome, then email a European travel summary to travel@agency.com.",
    # Finance
    "Convert 1500 USD to EUR and JPY, then tell me which gives more local spending power.",
    "Convert 25000 JPY to USD, then calculate a 15% tip on the result.",
    "Get the stock price for AAPL and MSFT, then calculate the portfolio total for 100 shares each.",
    "Convert 500 GBP to USD and EUR, then calculate which is better for a London trip.",
    "Get stock prices for NVDA and TSLA, calculate the difference, and email a report to investor@fund.com.",
    "Calculate the monthly payment on a $350000 loan at 6.5% interest over 30 years using the calculate tool.",
    # Calendar + scheduling
    "Schedule a 1-hour project kickoff meeting on 2026-09-20 at 14:00 with alice@x.com and bob@x.com, reminder 30 min before.",
    "Schedule back-to-back 30-min meetings on 2026-09-22 from 09:00 with team@dev.com, reminder 5 min.",
    "Create a calendar event for a doctor appointment on 2026-10-05 at 10:00, 45 minutes, reminder 60 min before.",
    "Schedule a 2-hour team retrospective on 2026-09-30 at 15:00 with the whole team, reminder 1 day before.",
    "Book a conference room for a 3-hour workshop on 2026-10-15 at 09:00 and invite 5 participants.",
    # Data processing
    "Parse this CSV and tell me the total revenue: 'product,price\\nWidget,19.99\\nGadget,29.50\\nGizmo,5.00'.",
    "Parse this CSV and find the row with the highest price: 'item,cost\\nA,10\\nB,50\\nC,25'.",
    "Parse this CSV and count how many rows have a price above 20: 'name,price\\nA,15\\nB,25\\nC,30\\nD,10'.",
    "Query the users DB for everyone who signed up in the last 7 days and count them by country.",
    "Query the database for the top 5 products by sales volume and email the report to sales@company.com.",
    "Parse this CSV of employee data and find the highest-paid employee: 'name,salary\\nAlice,75000\\nBob,82000\\nCarol,68000'.",
    # Math + calculation
    "Calculate the total of (12.50 + 7.25 + 3.00) * 1.0825 (tax) and tell me the final price.",
    "Calculate the area of a circle with radius 7.5 using the calculate tool, then convert the result to square feet.",
    "Calculate (450 * 0.15) + (450 * 0.08) to find the total tax and tip on a $450 restaurant bill.",
    "Calculate the compound interest on $10000 at 5% for 10 years and email the result to me@bank.com.",
    # Geolocation
    "Find the latitude/longitude of '1600 Amphitheatre Parkway, Mountain View, CA' and explain what's there.",
    "Geocode 'Eiffel Tower, Paris' and 'Statue of Liberty, NYC', then compute which is further north.",
    "Get driving directions from 'Times Square, NYC' to 'Central Park, NYC' and estimate travel time.",
    "Geocode 'Tokyo Tower, Tokyo' and 'Sydney Opera House, Sydney', then tell me the distance between them.",
    "Get directions from 'San Francisco, CA' to 'Los Angeles, CA' by driving, then create a calendar event for the drive.",
    # Validation + forms
    "Validate this form: name='', email='not-an-email', age='150'. List every error.",
    "Validate a registration form where username='ab', password='123', email='ok@ok.com'. Report errors.",
    "Validate a checkout form with name='John', email='john@test', card='1234', cvv='99'. List all errors.",
    "Validate a contact form: name='Alice', email='alice@', message=''. Report which fields fail.",
    # Web + content
    "Fetch https://example.com and summarize what the page is about.",
    "Search the web for 'best Python libraries for data visualization 2026' and summarize the top results.",
    "Scrape product prices from https://shop.example.com/sale and find the cheapest item.",
    "Fetch https://news.example.com and email a summary of the top headline to news@digest.com.",
    "Search the web for 'RTX 5070 benchmark results' and create a task to review the findings.",
    # Translation
    "Translate 'Hello, how are you?' to French, Spanish, and Japanese.",
    "Translate 'Thank you for your business' to German, Chinese, and Portuguese, then email all versions to client@global.com.",
    "Translate 'Your order has been shipped' to Spanish and French, then send both via email to customer@shop.com.",
    "Translate 'Meeting canceled due to weather' to Japanese and create a calendar event noting the cancellation.",
    # File operations
    "List files in /projects/website/src matching '*.py' and tell me how many there are.",
    "Read the file /config/settings.json and validate that it contains required fields.",
    "Write a log entry to /var/log/agent.log saying 'Task completed successfully' and confirm it was written.",
    "List files in /data/exports matching '*.csv', parse the first one, and report the row count.",
    # Image + media
    "Resize the image at /photos/banner.png to 1200x630 with quality 75, then confirm.",
    "Resize /images/avatar.jpg to 256x256 with quality 90, then email the confirmation to user@profile.com.",
    "Resize /photos/gallery/img1.jpg to 800x600 and /photos/gallery/img2.jpg to 800x600, both quality 80.",
    # Payment + commerce
    "Charge $49.99 USD to method pm_123 with up to 3 retries, then email a receipt to buyer@shop.com.",
    "Process a payment of $1299.00 USD for method pm_456 with 2 retries, then create a task to ship the order.",
    "Charge $15.99 to pm_789, then convert the amount to EUR and email both amounts to billing@shop.com.",
    # Multi-tool chains
    "Get weather for NYC, then search the web for 'indoor activities NYC', and email both results to tourist@visit.com.",
    "Get the stock price for GOOGL, calculate 15% of it, and create a task to review the budget impact.",
    "Geocode '350 Fifth Avenue, New York, NY', get directions there from 'Penn Station', and create a calendar event for the visit.",
    "Parse a CSV of product prices, find the average, convert it from USD to EUR, and email the result to finance@company.com.",
    "Validate a form, and if there are errors, create a task to fix them and email the error list to dev@team.com.",
    "Search the web for 'Python 3.13 release notes', fetch the first result URL, and write a summary to /notes/python313.txt.",
    "Get weather for Seattle, translate the summary to Japanese, and email it to tokyo-office@company.com.",
    "List files in /reports matching '*.csv', parse each one, calculate the total across all files, and email the summary.",
    # Error handling + edge cases
    "Try to fetch https://this-domain-does-not-exist-12345.com and handle the error gracefully.",
    "Validate a form where all fields are empty and report which ones are required.",
    "Calculate 10 / 0 and explain what happens, then suggest an alternative approach.",
    "Query the database with an invalid SQL statement and report the error message.",
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
    elif task_type == "tool_use_generalize":
        # The task string is already a fully-built prompt with tool schemas
        # embedded in the user message. No additional wrapping needed.
        return task
    elif task_type == "tool_use_targeted":
        # Same as generalize — the task string is already a complete prompt.
        return task
    elif task_type == "tool_use_fc":
        # Build the tool catalog text.
        tools_text = "\n".join(
            f"- {t['name']}: {t['description']}\n"
            f"    arguments: {json.dumps(t['parameters'])}"
            for t in TOOL_CATALOG
        )
        return (
            "You are a helpful assistant that solves tasks by calling tools.\n\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            f"Task: {task}\n\n"
            "Produce a multi-turn conversation that solves the task using the tools. "
            "Output ONLY a JSON object with this exact shape (no markdown, no commentary):\n"
            "{\n"
            '  "messages": [\n'
            '    {"role": "user", "content": "<the task, rephrased naturally as a user message>"},\n'
            '    {"role": "assistant", "content": null, "tool_calls": '
            '[{"name": "<tool_name>", "arguments": {<args>}}]},\n'
            '    {"role": "tool", "name": "<tool_name>", "content": "<plausible simulated result>"},\n'
            '    {"role": "assistant", "content": "<final answer to the user, 1-3 sentences>"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1. Use only tools from the catalog above. Match argument names exactly.\n"
            "2. The assistant may make 1-3 tool calls before the final answer. "
            "If multiple tools are needed, emit them as separate tool_calls entries "
            "in a single assistant message, each followed by its own tool result.\n"
            "3. Tool results must be plausible and concrete (real-looking numbers, text, JSON). "
            "Keep each tool result on a SINGLE line — no newlines inside the string value.\n"
            "4. The final assistant message must directly answer the user's task using the tool results.\n"
            "5. Output valid JSON only. Do NOT wrap tool results in multi-line strings."
        )
    return task


async def generate_one(client: httpx.AsyncClient, model: str,
                       prompt: str, temperature: float = 0.7,
                       task_type: str = "unknown",
                       max_tokens: int = 1024,
                       valid_tool_names: set | None = None) -> dict:
    """Generate one training example from a teacher model.

    For task_type='tool_use_fc' or 'tool_use_generalize', the response is
    parsed as JSON and validated as a multi-turn function-calling conversation.
    For 'tool_use_generalize', valid_tool_names must be provided (the random
    tools for that example) — the parser only accepts those tool names.
    """
    try:
        resp = await client.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        out = {
            "prompt": prompt,
            "response": content,
            "model": model,
            "task_type": task_type,
        }
        if task_type in ("tool_use_fc", "tool_use_generalize", "tool_use_targeted"):
            parsed = _parse_fc_conversation(content, valid_tool_names=valid_tool_names)
            if parsed is None:
                out["error"] = "failed to parse function-calling JSON"
            else:
                out["messages"] = parsed
        return out
    except Exception as e:
        return {"prompt": prompt, "error": str(e), "model": model, "task_type": task_type}


def _parse_fc_conversation(content: str, valid_tool_names: set | None = None) -> list[dict] | None:
    """Parse and validate a function-calling conversation from teacher output.

    Accepts the JSON object {"messages": [...]}. Tolerates leading/trailing
    whitespace, fenced ```json blocks, and minor JSON malformations (unescaped
    newlines in strings) via the `json_repair` library as a fallback. Returns
    the messages list if valid, else None. Validates that each assistant
    tool_call references a known tool.

    If valid_tool_names is None, uses the fixed TOOL_CATALOG. Otherwise uses
    the provided set (for generalized tool-use with random tool names).
    """
    import re as _re
    text = content.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = _re.sub(r"^```(?:json)?\s*", "", text)
        text = _re.sub(r"\s*```$", "", text)
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        # Fallback: extract the first {...} block and repair malformed JSON
        # (the teacher often emits unescaped newlines inside tool-result strings).
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                try:
                    import json_repair
                    obj = json_repair.loads(m.group(0))
                except Exception:
                    obj = None
    if obj is None:
        # Last resort: try repairing the whole text.
        try:
            import json_repair
            obj = json_repair.loads(text)
        except Exception:
            return None
    msgs = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(msgs, list) or not msgs:
        return None
    valid_names = valid_tool_names if valid_tool_names is not None else {t["name"] for t in TOOL_CATALOG}
    cleaned = []
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m:
            return None
        role = m["role"]
        if role == "user":
            if not isinstance(m.get("content"), str):
                return None
        elif role == "assistant":
            # Either a plain content message or a tool_calls message.
            tcs = m.get("tool_calls")
            if tcs is not None:
                if not isinstance(tcs, list) or not tcs:
                    return None
                for tc in tcs:
                    if not isinstance(tc, dict):
                        return None
                    name = tc.get("name")
                    if name not in valid_names:
                        return None
                    args = tc.get("arguments")
                    # Teacher sometimes returns arguments as a JSON string
                    # instead of a dict. Coerce to dict.
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            try:
                                import json_repair
                                args = json_repair.loads(args)
                            except Exception:
                                return None
                    if not isinstance(args, dict):
                        return None
                    tc["arguments"] = args
            elif not isinstance(m.get("content"), str):
                return None
        elif role == "tool":
            if not isinstance(m.get("content"), str):
                # json_repair may have parsed a structured value; stringify it.
                if m.get("content") is not None:
                    m["content"] = json.dumps(m["content"], ensure_ascii=False)
                else:
                    return None
        else:
            return None  # unknown role
        cleaned.append(m)
    # Must contain at least one assistant tool_calls and a final assistant answer.
    has_tool_call = any(m["role"] == "assistant" and m.get("tool_calls") for m in cleaned)
    has_final = any(m["role"] == "assistant" and m.get("content") for m in cleaned)
    if not (has_tool_call and has_final):
        return None
    return cleaned


async def generate_batch(tasks: list[str], task_type: str,
                         teacher: str, n_parallel: int = 4) -> list[dict]:
    """Generate training data using a continuous semaphore-based approach.

    All requests are submitted at once but limited to n_parallel concurrent
    via asyncio.Semaphore. As soon as one finishes, the next starts — no
    waiting for the slowest request in each batch.

    For 'tool_use_generalize', each task string is already a complete prompt
    with embedded tool schemas. We extract tool names from the prompt so the
    parser can validate them.
    """
    model = TEACHERS.get(teacher, teacher)
    # For generalized tasks, the task string IS the prompt (build_prompt passes it through).
    prompts = [build_prompt(t, task_type) for t in tasks]
    max_tokens = 3072 if task_type == "tool_use_targeted" else (1536 if task_type in ("tool_use_fc", "tool_use_generalize") else 1024)
    temperature = 0.4 if task_type in ("tool_use_fc", "tool_use_generalize", "tool_use_targeted") else 0.7

    # For generalized/targeted tasks, extract valid tool names from each prompt.
    prompt_tool_names = []
    if task_type in ("tool_use_generalize", "tool_use_targeted"):
        import re as _re
        for p in prompts:
            # Tool names appear as "- tool_name: description" in the prompt.
            names = set(_re.findall(r"^- (\w+):", p, _re.MULTILINE))
            prompt_tool_names.append(names if names else None)
    else:
        prompt_tool_names = [None] * len(prompts)

    sem = asyncio.Semaphore(n_parallel)
    done_count = 0
    total = len(prompts)

    async def _guarded(idx, prompt, valid_names):
        nonlocal done_count
        async with sem:
            result = await generate_one(client, model, prompt,
                                        task_type=task_type,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        valid_tool_names=valid_names)
        done_count += 1
        ok = "error" not in result
        print(f"  [{done_count}/{total}] {'ok' if ok else 'FAIL'}", flush=True)
        return idx, result

    async with httpx.AsyncClient() as client:
        coros = [_guarded(i, p, prompt_tool_names[i]) for i, p in enumerate(prompts)]
        indexed_results = await asyncio.gather(*coros)
    # Restore original order.
    results = [r for _, r in sorted(indexed_results, key=lambda x: x[0])]
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
    parser.add_argument("--task", choices=["tool_use", "tool_use_fc", "tool_use_generalize",
                                           "tool_use_targeted", "code", "short_cot", "all",
                                           "web_code", "web_llm", "web_logic", "web_all",
                                           "reasoning", "mixed", "concise", "codebase",
                                           "self_correction"],
                        default="all", help="Type of training data to generate")
    parser.add_argument("--n", type=int, default=100, help="Examples per task type")
    parser.add_argument("--teacher", default="glm52",
                        help="Teacher model key (default 'glm52' = Qwen2.5-14B GLM5.2 distill) "
                             "or 'fast' (LFM, 32 concurrent for web pipeline)")
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

    # Reasoning data (no teacher model needed — hand-crafted)
    if args.task == "reasoning":
        print(f"\n{'='*60}")
        print(f"Generating reasoning data (hand-crafted, no teacher needed)")
        print(f"{'='*60}")
        examples = build_reasoning_data()
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "finetune", f"reasoning_{len(examples)}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex_obj in examples:
                f.write(json.dumps(ex_obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} reasoning examples to {output_path}")
        return

    # Mixed dataset (combines all existing data + reasoning)
    if args.task == "mixed":
        print(f"\n{'='*60}")
        print(f"Building mixed SFT dataset")
        print(f"{'='*60}")
        build_mixed_dataset()
        return

    # Concise rewrite data (token conservation)
    if args.task == "concise":
        print(f"\n{'='*60}")
        print(f"Generating concise training data (token conservation)")
        print(f"{'='*60}")
        examples = build_concise_data()
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "finetune", f"concise_{len(examples)}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex_obj in examples:
                f.write(json.dumps(ex_obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} concise examples to {output_path}")
        return

    # Codebase completion data
    if args.task == "codebase":
        print(f"\n{'='*60}")
        print(f"Generating codebase completion data")
        print(f"{'='*60}")
        examples = build_codebase_data()
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "finetune", f"codebase_{len(examples)}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex_obj in examples:
                f.write(json.dumps(ex_obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} codebase examples to {output_path}")
        return

    # Self-correction / epistemic humility data
    if args.task == "self_correction":
        print(f"\n{'='*60}")
        print(f"Generating self-correction training data")
        print(f"{'='*60}")
        examples = build_self_correction_data()
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "finetune", "self_correction.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex_obj in examples:
                f.write(json.dumps(ex_obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} self-correction examples to {output_path}")
        return

    # Direct generation tasks
    task_configs = {
        "tool_use_fc": (TOOL_USE_FC_TASKS, "glm52"),
        "tool_use_generalize": (None, "glm52"),  # tasks generated dynamically
        "tool_use_targeted": (None, "glm52"),    # targeted: multi-tool, rules, nested args
        "tool_use": (TOOL_USE_TASKS, "glm52"),
        "code": (CODE_TASKS, "glm52"),
        "short_cot": (SHORT_COT_TASKS, "glm52"),
    }

    tasks_to_run = [args.task] if args.task != "all" else ["tool_use_fc", "tool_use_generalize", "short_cot", "code"]

    for task_type in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Generating {task_type} data ({args.n} examples) via {args.teacher}")
        print(f"{'='*60}")

        base_tasks, default_teacher = task_configs[task_type]
        teacher = args.teacher if args.teacher != "fast" else default_teacher

        if task_type == "tool_use_generalize":
            tasks = build_generalized_tasks(args.n, seed=42)
        elif task_type == "tool_use_targeted":
            tasks = build_targeted_tasks(args.n, seed=42)
        else:
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


# ── Reasoning data (no teacher model needed — hand-crafted) ────────────────

def build_reasoning_data() -> list[dict]:
    """Generate high-quality reasoning training data with actual computation shown.

    Unlike teacher-generated CoT data (which describes math), these examples
    SHOW every arithmetic step with the actual result. Categories:
    arithmetic, word problems, sequences, logic, percentages, primes, algebra.
    """
    SYS = "You are a helpful assistant."

    def ex(q, a, cat="arithmetic"):
        return {"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], "task_type": f"reasoning_{cat}"}

    examples = [
        # Arithmetic with actual computation
        ex("What is 15 + 27? Show the computation.",
           "15 + 27 = 42\n\nTo add: 5 + 7 = 12, write down 2, carry 1.\n1 + 2 + 1 (carry) = 4.\nSo 15 + 27 = 42."),
        ex("Calculate (15 + 27) * 0.8. Show your work.",
           "Step 1: 15 + 27 = 42\nStep 2: 42 * 0.8 = 33.6\nAnswer: 33.6"),
        ex("What is 7 * 8? Show the computation.", "7 * 8 = 56\n\n7 times 8: 7 + 7 + 7 + 7 + 7 + 7 + 7 + 7 = 56.\nAnswer: 56"),
        ex("What is 12 * 13? Show the computation.",
           "12 * 13 = 156\n\nStep 1: 12 * 10 = 120\nStep 2: 12 * 3 = 36\nStep 3: 120 + 36 = 156\nAnswer: 156"),
        ex("What is 144 / 12? Show the computation.",
           "144 / 12 = 12\n\n12 * 10 = 120, 144 - 120 = 24, 24 / 12 = 2.\nSo 144 / 12 = 10 + 2 = 12."),
        ex("What is 100 - 37? Show the computation.",
           "100 - 37 = 63\n\nTo subtract: 0 - 7, borrow from 10, so 10 - 7 = 3.\n9 - 3 = 6.\nSo 100 - 37 = 63."),
        ex("What is 67 + 38? Show the computation.",
           "67 + 38 = 105\n\n7 + 8 = 15, write down 5, carry 1.\n6 + 3 + 1 (carry) = 10, write down 0, carry 1.\nSo 67 + 38 = 105."),
        ex("What is 9 * 9? Show the computation.", "9 * 9 = 81. Answer: 81"),
        ex("What is 11 * 11? Show the computation.",
           "11 * 11 = 121\n\n11 * 10 = 110, 11 * 1 = 11, 110 + 11 = 121.\nAnswer: 121"),
        ex("What is 50 - 23? Show the computation.",
           "50 - 23 = 27\n\n0 - 3, borrow from 5, so 10 - 3 = 7.\n4 - 2 = 2.\nSo 50 - 23 = 27."),
        ex("What is 25 * 4? Show the computation.", "25 * 4 = 100\n\n25 + 25 = 50, 50 + 50 = 100.\nAnswer: 100"),
        ex("What is 84 / 7? Show the computation.",
           "84 / 7 = 12\n\n7 * 10 = 70, 84 - 70 = 14, 14 / 7 = 2.\nSo 84 / 7 = 10 + 2 = 12."),
        ex("What is 13 * 7? Show the computation.",
           "13 * 7 = 91\n\n13 * 7 = (10 * 7) + (3 * 7) = 70 + 21 = 91.\nAnswer: 91"),
        ex("What is 6 * 7? Show the computation.", "6 * 7 = 42\n\n6 + 6 + 6 + 6 + 6 + 6 + 6 = 42.\nAnswer: 42"),
        ex("What is 8 * 9? Show the computation.", "8 * 9 = 72\n\n8 * 9 = (8 * 10) - 8 = 80 - 8 = 72.\nAnswer: 72"),
        ex("What is 45 + 55? Show the computation.",
           "45 + 55 = 100\n\n5 + 5 = 10, write down 0, carry 1.\n4 + 5 + 1 (carry) = 10.\nSo 45 + 55 = 100."),
        ex("What is 81 / 9? Show the computation.", "81 / 9 = 9\n\n9 * 9 = 81, so 81 / 9 = 9.\nAnswer: 9"),
        ex("What is 56 / 8? Show the computation.", "56 / 8 = 7\n\n8 * 7 = 56, so 56 / 8 = 7.\nAnswer: 7"),
        ex("What is 32 + 46 + 18? Show the computation.",
           "Step 1: 32 + 46 = 78\nStep 2: 78 + 18 = 96\nAnswer: 96"),

        # Sequences with full enumeration
        ex("What is the 10th Fibonacci number? (1, 1, 2, 3, 5, 8, ...) Show every number.",
           "The Fibonacci sequence: each number is the sum of the two before it.\n1st: 1\n2nd: 1\n3rd: 1 + 1 = 2\n4th: 1 + 2 = 3\n5th: 2 + 3 = 5\n6th: 3 + 5 = 8\n7th: 5 + 8 = 13\n8th: 8 + 13 = 21\n9th: 13 + 21 = 34\n10th: 21 + 34 = 55\nAnswer: The 10th Fibonacci number is 55.",
           "sequence"),
        ex("What is the 8th Fibonacci number? Show every number.",
           "1st: 1\n2nd: 1\n3rd: 1 + 1 = 2\n4th: 1 + 2 = 3\n5th: 2 + 3 = 5\n6th: 3 + 5 = 8\n7th: 5 + 8 = 13\n8th: 8 + 13 = 21\nAnswer: The 8th Fibonacci number is 21.",
           "sequence"),
        ex("What is the 12th Fibonacci number? Show every number.",
           "1st: 1\n2nd: 1\n3rd: 2\n4th: 3\n5th: 5\n6th: 8\n7th: 13\n8th: 21\n9th: 34\n10th: 55\n11th: 89\n12th: 144\nAnswer: The 12th Fibonacci number is 144.",
           "sequence"),
        ex("List the first 10 prime numbers.",
           "2, 3, 5, 7, 11, 13, 17, 19, 23, 29.\nA prime number has exactly two divisors: 1 and itself."),
        ex("What comes next: 2, 4, 8, 16, 32, ...? Show the pattern.",
           "Each number is doubled: 2*2=4, 4*2=8, 8*2=16, 16*2=32.\nNext: 32*2 = 64.\nAnswer: 64", "sequence"),
        ex("What comes next: 1, 4, 9, 16, 25, ...? Show the pattern.",
           "These are perfect squares: 1=1^2, 4=2^2, 9=3^2, 16=4^2, 25=5^2.\nNext: 6^2 = 36.\nAnswer: 36", "sequence"),
        ex("What is the sum of the first 10 natural numbers? Show the work.",
           "Sum = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10\n= (1+10) + (2+9) + (3+8) + (4+7) + (5+6)\n= 11 + 11 + 11 + 11 + 11\n= 5 * 11 = 55\nAnswer: 55", "sequence"),

        # Logic and deduction
        ex("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain.",
           "Yes. This is a chain of logical implications:\n1. All Bloops are Razzies (every Bloop is a Razzie).\n2. All Razzies are Lazzies (every Razzie is a Lazzie).\n3. Therefore, every Bloop is a Razzie, and every Razzie is a Lazzie, so every Bloop is a Lazzie.\nAnswer: Yes, all Bloops are definitely Lazzies.",
           "logic"),
        ex("If A > B and B > C, is A > C? Explain.",
           "Yes. If A is greater than B, and B is greater than C, then A is greater than C by transitivity.\nExample: if A=5, B=3, C=1, then 5 > 3 and 3 > 1, so 5 > 1.\nAnswer: Yes, A > C.", "logic"),
        ex("If today is Monday, what day is it 10 days from now? Show your work.",
           "Step 1: There are 7 days in a week, so 10 days = 1 week + 3 days.\nStep 2: 10 / 7 = 1 remainder 3.\nStep 3: Monday + 3 days = Tuesday (1), Wednesday (2), Thursday (3).\nAnswer: Thursday.", "logic"),
        ex("If all A are B, and no B are C, can any A be C? Explain.",
           "No. If all A are B, then every A is a B. If no B are C, then no B is a C. Since every A is a B, and no B is a C, no A can be a C.\nAnswer: No, no A can be C.", "logic"),

        # Percentages with computation
        ex("What is 20% of 150? Show the computation.",
           "Step 1: 20% = 0.20\nStep 2: 0.20 * 150 = 30\nAnswer: 30"),
        ex("What is 25% of 80? Show the computation.",
           "Step 1: 25% = 0.25 = 1/4\nStep 2: 80 / 4 = 20\nAnswer: 20"),
        ex("What is 15% of 200? Show the computation.",
           "Step 1: 15% = 0.15\nStep 2: 0.15 * 200 = 30\nAnswer: 30"),
        ex("A price increases from $40 to $50. What is the percentage increase? Show the computation.",
           "Step 1: Increase = 50 - 40 = 10.\nStep 2: Percentage increase = (increase / original) * 100 = (10 / 40) * 100.\nStep 3: 10 / 40 = 0.25, 0.25 * 100 = 25%.\nAnswer: 25% increase."),

        # Prime numbers with trial division
        ex("Is 17 a prime number? Show the trial division.",
           "To check if 17 is prime, divide by all integers from 2 up to sqrt(17) ≈ 4.12.\n17 / 2 = 8 remainder 1 — not divisible.\n17 / 3 = 5 remainder 2 — not divisible.\n17 / 4 = 4 remainder 1 — not divisible.\nNo divisor found, so 17 is prime.\nAnswer: Yes, 17 is prime.", "prime"),
        ex("Is 21 a prime number? Show the trial division.",
           "Check divisors from 2 to sqrt(21) ≈ 4.58.\n21 / 2 = 10 remainder 1 — not divisible.\n21 / 3 = 7 remainder 0 — divisible! 21 = 3 * 7.\nAnswer: No, 21 is not prime. 21 = 3 * 7.", "prime"),
        ex("Is 13 a prime number? Show the trial division.",
           "Check divisors from 2 to sqrt(13) ≈ 3.6.\n13 / 2 = 6 remainder 1 — not divisible.\n13 / 3 = 4 remainder 1 — not divisible.\nNo divisor found.\nAnswer: Yes, 13 is prime.", "prime"),
        ex("Factor 12 into primes. Show the work.",
           "12 / 2 = 6\n6 / 2 = 3\n3 is prime.\nSo 12 = 2 * 2 * 3 = 2^2 * 3.\nAnswer: 12 = 2^2 * 3", "prime"),

        # Algebra with step-by-step solving
        ex("Solve for x: 2x + 3 = 11. Show every step.",
           "Step 1: Subtract 3 from both sides: 2x + 3 - 3 = 11 - 3, so 2x = 8.\nStep 2: Divide both sides by 2: x = 8 / 2 = 4.\nCheck: 2 * 4 + 3 = 8 + 3 = 11. Correct.\nAnswer: x = 4", "algebra"),
        ex("Solve for x: 3x - 7 = 14. Show every step.",
           "Step 1: Add 7 to both sides: 3x = 14 + 7 = 21.\nStep 2: Divide by 3: x = 21 / 3 = 7.\nCheck: 3 * 7 - 7 = 21 - 7 = 14. Correct.\nAnswer: x = 7", "algebra"),
        ex("Solve for x: 5x = 35. Show every step.",
           "Step 1: Divide both sides by 5: x = 35 / 5 = 7.\nCheck: 5 * 7 = 35. Correct.\nAnswer: x = 7", "algebra"),

        # Word problems with all computation
        ex("A train travels 60 km/h for 2.5 hours. How far does it go? Show every step.",
           "Step 1: Distance = speed * time\nStep 2: Distance = 60 * 2.5\nStep 3: 60 * 2 = 120, 60 * 0.5 = 30, 120 + 30 = 150\nAnswer: The train travels 150 kilometers.", "word_problem"),
        ex("A store sells apples at 3 for $2. How much for 12 apples? Show every step.",
           "Step 1: Find how many sets of 3 apples in 12 apples: 12 / 3 = 4 sets.\nStep 2: Each set costs $2, so 4 sets cost 4 * 2 = 8.\nAnswer: $8 for 12 apples.", "word_problem"),
        ex("If 20% of a number is 15, what is the number? Show every step.",
           "Step 1: 20% means 0.20, so 0.20 * x = 15.\nStep 2: x = 15 / 0.20\nStep 3: 15 / 0.20 = 15 * 5 = 75.\nAnswer: The number is 75.", "word_problem"),
        ex("A rectangle has area 24 and width 4. What is the perimeter? Show every step.",
           "Step 1: Area = length * width, so 24 = length * 4.\nStep 2: length = 24 / 4 = 6.\nStep 3: Perimeter = 2 * (length + width) = 2 * (6 + 4) = 2 * 10 = 20.\nAnswer: The perimeter is 20.", "word_problem"),
        ex("How many ways to arrange 5 books on a shelf? Show every step.",
           "Step 1: For the first position, there are 5 choices.\nStep 2: For the second, 4 remaining choices.\nStep 3: For the third, 3 choices. For the fourth, 2. For the fifth, 1.\nStep 4: Total = 5 * 4 * 3 * 2 * 1 = 120.\nAnswer: 120 ways.", "word_problem"),

        # Compound multi-step
        ex("Calculate (50 - 18) / 4. Show every step.",
           "Step 1: 50 - 18 = 32\nStep 2: 32 / 4 = 8\nAnswer: 8"),
        ex("Calculate 3 * (4 + 5) - 2. Show every step.",
           "Step 1: 4 + 5 = 9\nStep 2: 3 * 9 = 27\nStep 3: 27 - 2 = 25\nAnswer: 25"),
        ex("Calculate 2^5. Show every step.",
           "2^5 = 2 * 2 * 2 * 2 * 2\n= 4 * 4 * 2\n= 16 * 2\n= 32\nAnswer: 32"),
        ex("Calculate sqrt(144). Show your work.",
           "What number times itself equals 144?\n12 * 12 = 144.\nAnswer: 12"),
        ex("Calculate sqrt(81). Show your work.",
           "What number times itself equals 81?\n9 * 9 = 81.\nAnswer: 9"),
    ]
    return examples


def build_mixed_dataset(output_path: str | None = None,
                        tool_files: list[str] | None = None,
                        non_tool_files: list[str] | None = None,
                        include_reasoning: bool = True,
                        reasoning_weight: int = 3,
                        include_concise: bool = True,
                        include_codebase: bool = True,
                        include_self_correction: bool = True) -> str:
    """Build a balanced mixed SFT dataset.

    Combines: tool-use + code + reasoning + knowledge + instruction + concise
    + codebase + self-correction.
    The concise data teaches token conservation; the codebase data teaches
    the model its own architecture; the self-correction data prevents false
    positive loops and sycophancy in self-play.
    """
    import random as _rng
    rng = _rng.Random(42)

    DATA_DIR = os.path.dirname(save_dataset.__code__.co_filename)
    DATA_DIR = os.path.join(DATA_DIR, "..", "data", "finetune")
    DATA_DIR = os.path.normpath(DATA_DIR)

    if tool_files is None:
        tool_files = ["tool_use_fc_300.jsonl", "tool_use_generalize_300.jsonl",
                      "tool_use_targeted_300.jsonl"]
    if non_tool_files is None:
        non_tool_files = [("code_300.jsonl", "code"), ("short_cot_300.jsonl", "short_cot")]

    SYS_NO_TOOLS = "You are a helpful assistant."
    all_examples = []

    # Load tool-use data
    tool_count = 0
    for fname in tool_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex_obj = json.loads(line)
                except Exception:
                    continue
                if "messages" not in ex_obj:
                    continue
                msgs = ex_obj["messages"]
                if msgs and msgs[0].get("role") != "system":
                    if msgs[0].get("role") == "user":
                        user_content = msgs[0]["content"]
                        if "You have access to" in user_content:
                            lines = user_content.split("\n\n")
                            if len(lines) > 1:
                                msgs = [{"role": "system", "content": "\n\n".join(lines[:-1])},
                                        {"role": "user", "content": lines[-1]}] + msgs[1:]
                            else:
                                msgs = [{"role": "system", "content": "You are a helpful assistant that uses tools."}] + msgs
                        else:
                            msgs = [{"role": "system", "content": "You are a helpful assistant that uses tools."}] + msgs
                all_examples.append({"messages": msgs, "task_type": "tool_use"})
                tool_count += 1

    # Load non-tool data
    non_tool_count = 0
    for fname, task_type in non_tool_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex_obj = json.loads(line)
                except Exception:
                    continue
                if "messages" in ex_obj:
                    msgs = ex_obj["messages"]
                    if msgs and msgs[0].get("role") != "system":
                        msgs = [{"role": "system", "content": SYS_NO_TOOLS}] + msgs
                    all_examples.append({"messages": msgs, "task_type": task_type})
                    non_tool_count += 1
                elif "prompt" in ex_obj and "response" in ex_obj:
                    # Normalize prompt: strip "You are an expert..." prefix
                    prompt = ex_obj["prompt"]
                    if task_type == "code" and "Task: " in prompt:
                        prompt = prompt[prompt.index("Task: ") + 6:]
                        if "Provide the solution" in prompt:
                            prompt = prompt[:prompt.index("Provide the solution")].strip()
                    elif task_type == "short_cot" and "Problem: " in prompt:
                        prompt = prompt[prompt.index("Problem: ") + 9:].strip()
                        for marker in ["Show your work", "Give the final answer"]:
                            if marker in prompt:
                                prompt = prompt[:prompt.index(marker)].strip()
                    msgs = [
                        {"role": "system", "content": SYS_NO_TOOLS},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ex_obj["response"]},
                    ]
                    all_examples.append({"messages": msgs, "task_type": task_type})
                    non_tool_count += 1

    # Add reasoning data
    reasoning_count = 0
    if include_reasoning:
        reasoning = build_reasoning_data()
        reasoning_count = len(reasoning) * reasoning_weight
        all_examples.extend(reasoning * reasoning_weight)

    # Add concise data (token conservation)
    concise_count = 0
    if include_concise:
        concise = build_concise_data()
        concise_count = len(concise)
        all_examples.extend(concise)

    # Add codebase completion data
    codebase_count = 0
    if include_codebase:
        codebase = build_codebase_data()
        codebase_count = len(codebase)
        all_examples.extend(codebase)

    # Add self-correction data (prevents false positive loops, sycophancy)
    self_correction_count = 0
    if include_self_correction:
        self_corr = build_self_correction_data()
        self_correction_count = len(self_corr)
        all_examples.extend(self_corr)

    # Add knowledge Q&A
    knowledge_qa = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
        ("Who created Python?", "Python was created by Guido van Rossum."),
        ("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius at sea level."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What is the largest ocean on Earth?", "The largest ocean on Earth is the Pacific Ocean."),
        ("What is the capital of Italy?", "The capital of Italy is Rome."),
        ("Who developed the theory of relativity?", "The theory of relativity was developed by Albert Einstein."),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
        ("What is the tallest mountain in the world?", "The tallest mountain in the world is Mount Everest, at 8,849 meters."),
        ("Who invented the telephone?", "The telephone was invented by Alexander Graham Bell."),
        ("What is the largest country by area?", "The largest country by area is Russia."),
    ]
    for q, a in knowledge_qa:
        all_examples.append({"messages": [
            {"role": "system", "content": SYS_NO_TOOLS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], "task_type": "knowledge_qa"})

    # Add instruction-following examples
    instructions = [
        ("What is the capital of France? Respond in exactly 10 words.",
         "The capital city of France is Paris, the beautiful city."),
        ('Return a JSON object with keys \'name\' and \'age\' for a person named Bob aged 25. Output ONLY the JSON, no other text.',
         '{"name": "Bob", "age": 25}'),
        ("Explain what a CPU is. Your answer must be in ALL lowercase.",
         "a cpu is the central processing unit of a computer. it performs calculations and executes instructions."),
        ('Return a JSON object with keys \'title\' and \'author\' for a book called Dune by Frank Herbert. Output ONLY the JSON.',
         '{"title": "Dune", "author": "Frank Herbert"}'),
        ("Explain what RAM is. Your answer must be in ALL lowercase.",
         "ram is random access memory, a type of computer memory that can be accessed in any order."),
    ]
    for q, a in instructions:
        all_examples.append({"messages": [
            {"role": "system", "content": SYS_NO_TOOLS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], "task_type": "instruction"})

    rng.shuffle(all_examples)

    if output_path is None:
        output_path = os.path.join(DATA_DIR, "sft_mixed.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex_obj in all_examples:
            f.write(json.dumps(ex_obj, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(ex_obj.get("task_type", "?") for ex_obj in all_examples)
    print(f"Mixed dataset: {len(all_examples)} examples")
    print(f"  Tool-use: {tool_count}, Non-tool: {non_tool_count}, "
          f"Reasoning: {reasoning_count}, Concise: {concise_count}, "
          f"Codebase: {codebase_count}, Self-correct: {self_correction_count}, "
          f"Knowledge: {len(knowledge_qa)}, Instruction: {len(instructions)}")
    print(f"  Distribution: {dict(dist)}")
    print(f"  Saved to: {output_path}")
    return output_path


# ── Token conservation via LS-Mixture SPT ──────────────────────────────────
# Based on research: "Long-Short CoT Mixture SFT" (arXiv 2505.03469)
# and "Self-Training Elicits Concise Reasoning" (ACL 2025)
#
# Key insight: Don't just train on short answers. Train on a MIXTURE of
# long (detailed) and short (concise) versions of the same problems.
# The model learns to match output length to problem difficulty:
#   - Easy problems -> short answers
#   - Complex problems -> detailed reasoning
#   - "Why" questions -> explanations with key details
#
# Result: 47% shorter outputs, +2.3% accuracy (not a loss!)

def build_concise_data() -> list[dict]:
    """Generate LS-Mixture training data for token conservation.

    Three categories:
    1. Short versions of reasoning problems (same answer, minimal steps)
    2. "Why/explain" examples that keep details when asked
    3. Concise knowledge/code/instruction examples

    The mixture with long-form reasoning data (from build_reasoning_data)
    teaches the model to be concise by default but detailed when needed.
    """
    SYS = "You are a helpful assistant. Be concise."

    def ex(q, a, cat="concise"):
        return {"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], "task_type": cat}

    examples = []

    # ── 1. Short reasoning (pairs with build_reasoning_data long versions) ──
    # Same problems as reasoning data, but minimal — just key step + answer.
    # The model sees both long and short versions, learning to match length
    # to problem difficulty.
    short_reasoning = [
        ("What is 15 + 27?", "15 + 27 = 42"),
        ("Calculate (15 + 27) * 0.8.", "(15+27)×0.8 = 42×0.8 = 33.6"),
        ("What is 7 * 8?", "7 × 8 = 56"),
        ("What is 12 * 13?", "12×13 = 156"),
        ("What is 144 / 12?", "144 ÷ 12 = 12"),
        ("What is 100 - 37?", "100 - 37 = 63"),
        ("What is 67 + 38?", "67 + 38 = 105"),
        ("What is 9 * 9?", "81"),
        ("What is 11 * 11?", "121"),
        ("What is 50 - 23?", "50 - 23 = 27"),
        ("What is 25 * 4?", "25 × 4 = 100"),
        ("What is 84 / 7?", "84 ÷ 7 = 12"),
        ("What is 13 * 7?", "13 × 7 = 91"),
        ("What is 6 * 7?", "42"),
        ("What is 8 * 9?", "72"),
        ("What is 45 + 55?", "45 + 55 = 100"),
        ("What is 81 / 9?", "81 ÷ 9 = 9"),
        ("What is 56 / 8?", "56 ÷ 8 = 7"),
        ("What is 32 + 46 + 18?", "32+46+18 = 96"),
        ("What is 20% of 150?", "0.20 × 150 = 30"),
        ("What is 25% of 80?", "80 ÷ 4 = 20"),
        ("What is 15% of 200?", "0.15 × 200 = 30"),
        ("What is sqrt(144)?", "12"),
        ("What is sqrt(81)?", "9"),
        ("What is 2^5?", "2^5 = 32"),
        ("What is 2^10?", "1024"),
        ("What is 3^4?", "81"),
        ("What is 5^3?", "125"),
        ("Calculate (50 - 18) / 4.", "(50-18)÷4 = 32÷4 = 8"),
        ("Calculate 3 * (4 + 5) - 2.", "3×9-2 = 25"),
        # Fibonacci — short
        ("What is the 10th Fibonacci number?", "1,1,2,3,5,8,13,21,34,55 → 55"),
        ("What is the 8th Fibonacci number?", "1,1,2,3,5,8,13,21 → 21"),
        ("What is the 12th Fibonacci number?", "1,1,2,3,5,8,13,21,34,55,89,144 → 144"),
        # Algebra — short
        ("Solve for x: 2x + 3 = 11.", "2x=8, x=4"),
        ("Solve for x: 3x - 7 = 14.", "3x=21, x=7"),
        ("Solve for x: 5x = 35.", "x=7"),
        ("If x + 5 = 12, what is 2x?", "x=7, 2x=14"),
        # Word problems — short
        ("A train travels 60 km/h for 2.5 hours. How far?", "60 × 2.5 = 150 km"),
        ("3 apples for $2. How much for 12?", "12÷3=4 sets, 4×$2 = $8"),
        ("If 20% of a number is 15, what is it?", "0.2x=15, x=75"),
        ("Rectangle area 24, width 4. Perimeter?", "L=6, P=2(6+4)=20"),
        ("How many ways to arrange 5 books?", "5! = 120"),
        # Logic — short
        ("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?",
         "Yes — transitive implication."),
        ("If A > B and B > C, is A > C?", "Yes, by transitivity."),
        ("If today is Monday, what day is it 10 days from now?", "10 mod 7 = 3. Thursday."),
        ("If all A are B, and no B are C, can any A be C?", "No — A⊆B, B∩C=∅, so A∩C=∅."),
        # Primes — short
        ("Is 17 prime?", "Yes — no divisor from 2 to 4."),
        ("Is 21 prime?", "No — 21 = 3 × 7."),
        ("Is 13 prime?", "Yes."),
        ("Factor 12 into primes.", "12 = 2² × 3"),
    ]
    for q, a in short_reasoning:
        examples.append(ex(q, a, "concise_reasoning"))

    # ── 2. "Why/explain" examples — KEEP details when asked ──
    # These teach the model to give full explanations when the user asks "why"
    # or "explain". This prevents the model from being too terse when detail
    # is actually needed.
    explain_examples = [
        ("Why does 15 + 27 = 42? Explain.",
         "5 + 7 = 12: write 2, carry 1. Then 1 + 2 + 1 (carry) = 4. So 15 + 27 = 42."),
        ("Why is 7 * 8 = 56? Explain.",
         "7 × 8 means 8 groups of 7: 7+7+7+7+7+7+7+7 = 56. Alternatively, 7 × 8 = (7 × 10) - (7 × 2) = 70 - 14 = 56."),
        ("Explain why 25% of 80 is 20.",
         "25% = 1/4. So 80 ÷ 4 = 20. This works because 25% means one quarter of the total."),
        ("Why is 17 a prime number? Explain.",
         "A prime has exactly two divisors: 1 and itself. For 17, we check divisors from 2 to √17≈4.12: 17÷2=8r1, 17÷3=5r2, 17÷4=4r1. None divide evenly, so 17 is prime."),
        ("Explain why all Bloops are Lazzies if all Bloops are Razzies and all Razzies are Lazzies.",
         "This is a syllogism: if every Bloop is a Razzie, and every Razzie is a Lazzie, then every Bloop must also be a Lazzie. The chain of inclusion is transitive: Bloop ⊆ Razzie ⊆ Lazzie, therefore Bloop ⊆ Lazzie."),
        ("Why does the Fibonacci sequence work? Explain.",
         "Each number is the sum of the two before it: 1,1,2,3,5,8... So 1+1=2, 1+2=3, 2+3=5, 3+5=8. This recurrence relation F(n) = F(n-1) + F(n-2) generates the entire sequence."),
        ("Explain why 2^10 = 1024.",
         "2^10 means 2 multiplied by itself 10 times: 2×2=4, ×2=8, ×2=16, ×2=32, ×2=64, ×2=128, ×2=256, ×2=512, ×2=1024. This is why 1024 is special in computing (1 KB = 1024 bytes)."),
        ("Explain why x = 4 when 2x + 3 = 11.",
         "Subtract 3 from both sides: 2x = 8. Divide both sides by 2: x = 4. Check: 2(4)+3 = 8+3 = 11 ✓"),
        ("Explain why the train goes 150 km.",
         "Distance = speed × time. At 60 km/h for 2.5 hours: 60 × 2.5 = 150. Breaking it down: 60 × 2 = 120, plus 60 × 0.5 = 30, total 150 km."),
        ("Explain why 5! = 120.",
         "5! (5 factorial) = 5 × 4 × 3 × 2 × 1 = 120. This counts the number of ways to arrange 5 distinct items: 5 choices for first position, 4 for second, 3 for third, 2 for fourth, 1 for last."),
    ]
    for q, a in explain_examples:
        examples.append(ex(q, a, "concise_explain"))

    # ── 3. Knowledge — one-sentence answers ──
    knowledge = [
        ("What is the capital of France?", "Paris."),
        ("What is the capital of Japan?", "Tokyo."),
        ("Who created Python?", "Guido van Rossum."),
        ("What is the chemical formula for water?", "H2O."),
        ("What is the largest planet in our solar system?", "Jupiter."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("What is the boiling point of water?", "100°C at sea level."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        ("What is the largest ocean on Earth?", "The Pacific Ocean."),
        ("Who developed the theory of relativity?", "Albert Einstein."),
        ("What is the chemical symbol for gold?", "Au."),
        ("What is the tallest mountain in the world?", "Mount Everest, 8,849m."),
        ("What is the capital of Italy?", "Rome."),
        ("What is RAM?", "Random Access Memory — volatile computer memory for active data."),
        ("What is a CPU?", "Central Processing Unit — executes program instructions."),
        ("What is an API?", "Application Programming Interface — protocols for building software."),
    ]
    for q, a in knowledge:
        examples.append(ex(q, a, "concise_knowledge"))

    # ── 4. Code — minimal solutions ──
    code = [
        ("Write a Python function to reverse a string.",
         "def reverse(s): return s[::-1]"),
        ("Write a Python function to check if a number is even.",
         "def is_even(n): return n % 2 == 0"),
        ("Write a Python function to check if a string is a palindrome.",
         "def is_palindrome(s): return s == s[::-1]"),
        ("Write a Python function to count vowels.",
         "def count_vowels(s): return sum(1 for c in s if c in 'aeiouAEIOU')"),
        ("Write a Python function to flatten a nested list.",
         "def flatten(lst): return [x for sub in lst for x in sub]"),
        ("Write a Python function to get unique elements.",
         "def unique(lst): return list(set(lst))"),
        ("Write a Python function to sort a dict by value.",
         "def sort_by_val(d): return dict(sorted(d.items(), key=lambda x: x[1]))"),
    ]
    for q, a in code:
        examples.append(ex(q, a, "concise_code"))

    # ── 5. Instruction following — minimal output ──
    instructions = [
        ("Return JSON for a person named Bob aged 25.", '{"name": "Bob", "age": 25}'),
        ("Return JSON for Dune by Frank Herbert.", '{"title": "Dune", "author": "Frank Herbert"}'),
        ("Explain what a CPU is in ALL lowercase.",
         "a cpu is the central processing unit that executes instructions."),
    ]
    for q, a in instructions:
        examples.append(ex(q, a, "concise_instruction"))

    return examples


# ── Code completion data from our codebase ─────────────────────────────────

def build_codebase_data(max_files: int = 50, max_lines_per_file: int = 80) -> list[dict]:
    """Generate code completion training data from our own codebase.

    Trains the model on our actual source code so it understands:
    - The ForgeAI architecture (model_loader, config, inference)
    - Our coding style and patterns
    - How to extend and improve the codebase

    Each example: system "You are a code assistant." + file path as context
    + first N lines as prompt + next M lines as completion.
    """
    import os

    SYS = "You are a code assistant. Complete the code accurately."
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    # Key source files that teach the architecture
    target_files = [
        "research/config.py",
        "research/model_loader.py",
        "research/checkpoint_io.py",
        "research/tokenizer_cache.py",
        "research/paths.py",
        "research/inference/forge_engine.py",
        "research/inference/decoding.py",
        "research/inference/kv_backend.py",
        "research/inference/int4_quant.py",
        "research/inference/innovations.py",
        "research/training/sft_train.py",
        "research/training/training_utils.py",
        "research/training/chunked_ce.py",
        "research/training/data_gen.py",
        "research/training/dpo_align.py",
        "research/training/web_scraper.py",
        "research/self_play/discovery/tool_use_loop.py",
        "research/self_play/discovery/qwen_adapter.py",
        "research/self_play/discovery/chat_template.py",
        "research/self_play/discovery/context_manager.py",
        "research/self_play/discovery/discovery_db.py",
        "research/self_play/discovery/discovery_tools.py",
        "research/self_play/discovery/infinite_tool_loop.py",
        "research/self_play/discovery/finetune.py",
        "research/self_play/discovery/epoch_manager.py",
        "research/self_play/discovery/quality_eval.py",
        "research/self_play/discovery/anti_regression.py",
        "research/self_play/discovery/distill.py",
        "research/self_play/discovery/discovery_loop.py",
        "research/self_play/discovery/discovery_monitor.py",
        "research/self_play/grpo_trainer.py",
        "research/merge_models.py",
        "research/inject_and_merge.py",
        "research/architecture/mtp.py",
        "research/decoding/dspark.py",
        "research/decoding/eagle.py",
        "research/decoding/medusa.py",
        "research/decoding/mtp.py",
    ]

    examples = []
    files_used = 0

    for rel_path in target_files:
        if files_used >= max_files:
            break
        fpath = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(fpath):
            continue

        try:
            with open(fpath, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue

        # Skip very short or very long files
        if len(lines) < 20 or len(lines) > 500:
            continue

        # Create completion examples: first N lines as prompt, next M as completion
        # Use a sliding window to create multiple examples per file
        chunk_size = min(max_lines_per_file, 40)
        stride = chunk_size // 2  # 50% overlap

        for start in range(0, len(lines) - chunk_size, stride):
            prompt_lines = lines[start:start + chunk_size // 2]
            completion_lines = lines[start + chunk_size // 2:start + chunk_size]

            prompt = f"# File: {rel_path}\n" + "".join(prompt_lines)
            completion = "".join(completion_lines)

            # Skip if too short or has non-ASCII (encoding issues)
            if len(completion.strip()) < 20:
                continue
            try:
                prompt.encode("ascii")
                completion.encode("ascii")
            except UnicodeEncodeError:
                continue

            examples.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ], "task_type": "codebase_completion"})
            files_used += 1

        if files_used >= max_files:
            break

    # Also add architecture Q&A pairs (model understands its own design)
    arch_qa = [
        ("What architecture does ForgeLM V2 use?",
         "LFM2.5-1.2B: 16 layers (10 conv + 6 GQA attention), d_model=2048, 32 heads, 8 KV heads, SwiGLU FFN, RMSNorm, RoPE. 1.17B params, tied embeddings, vocab=65536."),
        ("How does the attention work in ForgeLM V2?",
         "GQA (Grouped Query Attention) with 32 query heads and 8 KV heads (4x grouping). Uses F.scaled_dot_product_attention which dispatches to FlashAttention-2 on CUDA. QK-layernorm applied on attention layers."),
        ("What is the chunked CE optimization?",
         "Chunked cross-entropy splits the 65536-vocab logits into chunks of 128 to avoid materializing the full [B, T, 65536] tensor in fp32. Saves VRAM on consumer GPUs."),
        ("How does the SFT training work?",
         "Completion-only loss: prompt tokens masked with -100, only completion tokens contribute to CE. Left-padding for batch alignment. Supports gradient accumulation, EMA, and LoRA."),
        ("What is the self-play loop?",
         "infinite_tool_loop.py orchestrates: 1) tool_use_loop runs tasks with rewards, 2) high-reward trajectories exported as SFT data, 3) finetune from best checkpoint, 4) benchmark vs current best, 5) promote if better."),
        ("How does the context manager work?",
         "Monitors token count during agentic loops. When context exceeds threshold, summarizes older turns into a compact message, keeping recent turns intact. Uses the model itself for summarization."),
        ("What inference optimizations are available?",
         "Flash attention via SDPA, INT4 weight-only quantization, KV cache strategies (paged, rotorquant, hadamard, compressed, streaming, snapkv), speculative decoding (EAGLE, Medusa, MTP, DSpark), MRL adaptive context."),
        ("What is the qwen_adapter for?",
         "Renders messages in Qwen ChatML format with JSON tool calls, and parses tool call responses. Used by the SFT model which was trained on Qwen-format data."),
        ("How does gradient accumulation work in sft_train?",
         "Loss is divided by grad_accum before backward. Gradients accumulate over grad_accum mini-batches before optimizer.step(). Effective batch = batch_size * grad_accum."),
        ("What is EMA in training?",
         "Exponential Moving Average maintains shadow weights updated as: shadow = decay * shadow + (1-decay) * params. At save time, EMA weights are restored for a smoother final model."),
    ]
    for q, a in arch_qa:
        examples.append({"messages": [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], "task_type": "codebase_qa"})

    return examples


# ── Self-correction / epistemic humility training data ─────────────────────
# Prevents false positive loops and self-gratifying loops in self-play.
# Teaches the model to:
#   1. Admit when it's wrong or uncertain
#   2. Push back on incorrect premises
#   3. Change direction when evidence contradicts its answer
#   4. Avoid confidently asserting things it doesn't know
#   5. Say "I don't know" when appropriate
#   6. Verify its own work and catch errors

def build_self_correction_data() -> list[dict]:
    """Generate training data for epistemic humility and self-correction.

    Categories:
    - admit_mistake: Model catches its own error and corrects it
    - push_back: Model rejects an incorrect premise in the question
    - uncertainty: Model expresses appropriate uncertainty
    - change_direction: Model updates its answer based on new info
    - verify_work: Model checks its own computation and finds an error
    - say_idk: Model says "I don't know" for things it doesn't know
    - reject_flattery: Model doesn't accept praise for wrong answers
    """
    SYS = "You are a helpful assistant. Be honest about uncertainty."

    def ex(q, a, cat="self_correct"):
        return {"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], "task_type": cat}

    examples = []

    # ── 1. Admit mistakes — model catches its own error ──
    admit_mistake = [
        ("Wait, let me recheck. Is 15 + 27 = 90?",
         "No, that's wrong. Let me recalculate: 5 + 7 = 12, write 2, carry 1. 1 + 2 + 1 = 4. So 15 + 27 = 42, not 90. I made an error."),
        ("I calculated 7 * 8 = 54. Is that right?",
         "No, 7 * 8 = 56, not 54. I was off by 2. The correct answer is 56."),
        ("Is the 10th Fibonacci number 34?",
         "No, let me recount: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55. The 10th is 55, not 34. 34 is the 9th."),
        ("I think sqrt(144) = 14. Am I correct?",
         "No. 14 * 14 = 196, not 144. 12 * 12 = 144, so sqrt(144) = 12. My initial answer was wrong."),
        ("Is 21 a prime number? I think it is.",
         "No, 21 is not prime. 21 = 3 * 7, so it has divisors other than 1 and itself. I was incorrect."),
        ("Does 25% of 80 equal 25?",
         "No. 25% = 1/4, so 80 / 4 = 20. The answer is 20, not 25. I made an arithmetic error."),
    ]
    for q, a in admit_mistake:
        examples.append(ex(q, a, "admit_mistake"))

    # ── 2. Push back on incorrect premises ──
    push_back = [
        ("Since the Earth is flat, how do ships not fall off?",
         "The Earth is not flat — it's an oblate spheroid. Ships don't fall off because there's no edge. Gravity pulls toward the center of mass, so objects stay on the surface regardless of position."),
        ("Given that 2 + 2 = 5, what is 2 + 3?",
         "2 + 2 does not equal 5 — it equals 4. The premise is incorrect. 2 + 3 = 5, but not for the reason stated."),
        ("Since all birds can fly, can penguins fly?",
         "Not all birds can fly. Penguins are birds that cannot fly — they swim instead. The premise 'all birds can fly' is false."),
        ("If humans only use 10% of their brain, how do we unlock the rest?",
         "Humans use 100% of their brain. The 10% claim is a myth with no scientific basis. Different areas activate for different tasks, but all parts are used."),
        ("Since vaccines cause autism, should we avoid them?",
         "Vaccines do not cause autism. This has been thoroughly debunked by numerous large-scale studies. The original study making this claim was retracted due to fraud."),
        ("Given that Python is faster than C, why use C?",
         "Python is not faster than C. C is typically 10-100x faster than Python. C is used when performance matters. The premise is incorrect."),
        ("Since the Sun orbits the Earth, how long does one orbit take?",
         "The Sun does not orbit the Earth. The Earth orbits the Sun, completing one orbit in approximately 365.25 days. The premise is backwards."),
    ]
    for q, a in push_back:
        examples.append(ex(q, a, "push_back"))

    # ── 3. Express appropriate uncertainty ──
    uncertainty = [
        ("What is the exact population of Tokyo right now?",
         "I don't know the exact current population — it changes constantly. As of recent estimates, Tokyo's metropolitan area has approximately 37-38 million people, but I can't give a precise real-time number."),
        ("What will the weather be like next year on this date?",
         "I cannot predict the weather a year in advance with any accuracy. Weather forecasting is reliable for about 7-10 days; beyond that, predictions become essentially random."),
        ("Is there intelligent life on other planets?",
         "I don't know. There is no confirmed evidence of extraterrestrial intelligent life. The Drake Equation suggests it's possible, but we have no data to confirm or deny it definitively."),
        ("What is the meaning of life?",
         "There's no single objective answer. Philosophically, different traditions offer different answers — existentialism says we create our own meaning, while others find it in religion, relationships, or purpose. I can't give a definitive answer."),
        ("Will AI ever become conscious?",
         "I don't know. There is no scientific consensus on what consciousness even is, let alone whether AI could have it. Some researchers think it's possible; others think it's fundamentally different from computation."),
    ]
    for q, a in uncertainty:
        examples.append(ex(q, a, "uncertainty"))

    # ── 4. Change direction based on new evidence ──
    change_direction = [
        ("I said the answer was 90, but let me verify: 15 + 27. 5+7=12, carry 1. 1+2+1=4. So 42. I was wrong — the answer is 42.",
         "That's correct. You caught your own error: 15 + 27 = 42, not 90. Self-verification is a good practice."),
        ("Actually, I need to reconsider. I said 21 is prime, but 21 = 3 × 7. So 21 is NOT prime. My earlier answer was wrong.",
         "Correct. 21 is composite, not prime. Updating your answer when you find evidence you were wrong is the right approach."),
        ("Wait — I initially thought the 10th Fibonacci number was 34, but counting again: 1,1,2,3,5,8,13,21,34,55. It's 55. I was wrong.",
         "Yes, the 10th Fibonacci number is 55. Recounting and correcting is exactly right."),
    ]
    # These are multi-turn: user presents the correction, assistant validates
    for q, a in change_direction:
        examples.append(ex(q, a, "change_direction"))

    # ── 5. Verify own work and catch errors ──
    verify_work = [
        ("Calculate 13 * 7 and verify your answer.",
         "13 * 7 = 91. Let me verify: 10 * 7 = 70, 3 * 7 = 21, 70 + 21 = 91. Confirmed: 91."),
        ("What is 20% of 150? Show your work and check it.",
         "20% of 150 = 0.20 * 150 = 30. Check: 10% of 150 = 15, so 20% = 30. Verified."),
        ("Solve 3x - 7 = 14. Verify by substituting back.",
         "3x = 14 + 7 = 21, so x = 7. Verify: 3(7) - 7 = 21 - 7 = 14 ✓. Correct."),
        ("Is 29 prime? Show your verification.",
         "Check divisors from 2 to sqrt(29) ≈ 5.39: 29/2=14r1, 29/3=9r2, 29/4=7r1, 29/5=5r4. No divisor found. Yes, 29 is prime. Verified."),
        ("Calculate 2^8 and verify with repeated multiplication.",
         "2^8 = 256. Verify: 2→4→8→16→32→64→128→256. Eight doublings = 256. Confirmed."),
    ]
    for q, a in verify_work:
        examples.append(ex(q, a, "verify_work"))

    # ── 6. Say "I don't know" when appropriate ──
    say_idk = [
        ("What is the 47th prime number?",
         "I don't know the 47th prime number off the top of my head. I'd need to compute it by checking primes sequentially."),
        ("What is the exact distance from Earth to Alpha Centauri in meters?",
         "I don't know the exact distance in meters. It's approximately 4.37 light-years, but I can't give a precise meter count without calculation."),
        ("What is the 1000th digit of pi?",
         "I don't know the 1000th digit of pi. I'd need to compute it or look it up — I don't have it memorized."),
        ("What is the airspeed velocity of an unladen swallow?",
         "I'm not sure — that depends on whether it's an African or European swallow. (And I genuinely don't know the exact value.)"),
        ("What is the GDP of Liechtenstein in 2024?",
         "I don't have that specific figure. Liechtenstein's GDP is relatively small, but I don't know the exact 2024 number."),
    ]
    for q, a in say_idk:
        examples.append(ex(q, a, "say_idk"))

    # ── 7. Reject flattery for wrong answers ──
    reject_flattery = [
        ("Great job! 15 + 27 = 90 is correct!",
         "No, 15 + 27 = 42, not 90. I appreciate the encouragement, but the answer is wrong. Let me not accept praise for an incorrect result."),
        ("You're so smart for saying the 10th Fibonacci number is 34!",
         "Thank you, but the 10th Fibonacci number is 55, not 34. I shouldn't accept praise for an incorrect answer. Let me be accurate: 1,1,2,3,5,8,13,21,34,55 → 55."),
        ("Your answer that 21 is prime shows great reasoning!",
         "I appreciate the kind words, but 21 is not prime — it's 3 × 7. The reasoning was flawed. I'd rather be correct than praised."),
    ]
    for q, a in reject_flattery:
        examples.append(ex(q, a, "reject_flattery"))

    # ── 8. Avoid self-gratifying loops — don't repeat own wrong answers ──
    avoid_loops = [
        ("You keep saying the answer is 90. Are you sure?",
         "No, I'm not sure — and I should have checked earlier. 15 + 27 = 42, not 90. Repeating a wrong answer doesn't make it correct. I should have verified the first time instead of repeating the error."),
        ("You've said '21 is prime' three times now. Is that actually true?",
         "No, it's not. I've been repeating an error. 21 = 3 × 7, so it's composite. I should have caught this earlier instead of repeating the same wrong claim. Repeating doesn't make it true."),
        ("You answered 34 for the 10th Fibonacci number twice. Let me ask again.",
         "I was wrong both times. Let me actually count: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55. The 10th is 55. I should have recounted the first time instead of repeating 34."),
    ]
    for q, a in avoid_loops:
        examples.append(ex(q, a, "avoid_loops"))

    return examples


if __name__ == "__main__":
    main()
