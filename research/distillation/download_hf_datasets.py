"""Download and filter HuggingFace datasets for ForgeLM V3 training.

Downloads high-quality datasets for:
- Code generation
- Reasoning / chain-of-thought
- Logic
- Planning / tool use
- Math
- Creativity

Excludes: politics, culture, geopolitics, anything not useful for self-play.

Usage:
    python -m research.distillation.download_hf_datasets --max-per-dataset 50000
    python -m research.distillation.download_hf_datasets --category code --max-per-dataset 100000
    python -m research.distillation.download_hf_datasets --stream --max-per-dataset 10000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# Load .env for HF token
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
os.environ.setdefault("PYTHONUTF8", "1")

# ── Exclusion filter ──

EXCLUDE_KEYWORDS = [
    # Politics / geopolitics
    "politic", "election", "president", "congress", "senate", "parliament",
    "democrat", "republican", "liberal", "conservative", "geopolit",
    "government policy", "foreign policy", "diplomat", "war crime",
    "nato", "united nations", "eu policy", "brexit", "trump", "biden",
    "obama", "putin", "xi jinping", "netanyahu", "hamas", "israel",
    "palestine", "ukraine conflict", "russia ukraine", "taiwan conflict",
    # Culture / social
    "cultural appropriation", "cultural identity", "cultural heritage",
    "social justice", "woke", "cancel culture", "identity politics",
    "race theory", "gender ideology", "cultural relativism",
    # Not useful for self-play
    "celebrity gossip", "entertainment news", "movie review",
    "recipe", "cooking instruction", "fashion tip", "beauty tutorial",
    "travel guide", "restaurant review",
]

EXCLUDE_PATTERNS = re.compile(
    "|".join(re.escape(kw) for kw in EXCLUDE_KEYWORDS),
    re.IGNORECASE,
)


def should_exclude(text: str) -> bool:
    """Return True if text contains excluded keywords."""
    if not text:
        return False
    # Only check first 2000 chars for speed
    return bool(EXCLUDE_PATTERNS.search(text[:2000]))


# ── Dataset catalog ──

DATASETS = {
    # ── CODE INSTRUCTION (problem → code) ──
    "magicoder_oss": {
        "id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "category": "code",
        "priority": "high",
        "format": "magicoder",
        "description": "75K synthetic code instruction pairs (OSS-Instruct)",
    },
    "magicoder_evol": {
        "id": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "category": "code",
        "priority": "high",
        "format": "magicoder",
        "description": "110K evolved code instruction pairs",
    },
    "code_alpaca": {
        "id": "HuggingFaceH4/CodeAlpaca_20K",
        "category": "code",
        "priority": "high",
        "format": "alpaca",
        "description": "20K code instruction pairs",
    },
    "codeforces": {
        "id": "open-r1/codeforces",
        "category": "code",
        "priority": "medium",
        "format": "codeforces",
        "description": "Competitive programming problems with solutions",
    },

    # ── REASONING / THINKING ──
    "openthoughts_114k": {
        "id": "open-thoughts/OpenThoughts-114k",
        "category": "reasoning",
        "priority": "high",
        "format": "openthoughts",
        "description": "114K reasoning traces with thinking",
    },
    "openthoughts3": {
        "id": "open-thoughts/OpenThoughts3-1.2M",
        "category": "reasoning",
        "priority": "high",
        "format": "openthoughts",
        "description": "1.2M reasoning traces (largest OpenThoughts)",
    },
    "dolphin_r1": {
        "id": "cognitivecomputations/dolphin-r1",
        "category": "reasoning",
        "priority": "medium",
        "format": "dolphin_r1",
        "config": "reasoning-deepseek",
        "split": "train",
        "description": "R1 reasoning traces (dolphin filtered, deepseek)",
    },

    # ── MATH ──
    "openr1_math": {
        "id": "open-r1/OpenR1-Math-220k",
        "category": "math",
        "priority": "high",
        "format": "openr1_math",
        "description": "220K math problems with R1 reasoning",
    },
    "metamath": {
        "id": "meta-math/MetaMathQA",
        "category": "math",
        "priority": "high",
        "format": "metamath",
        "description": "395K math QA with reasoning",
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "category": "math",
        "priority": "high",
        "format": "gsm8k",
        "config": "main",
        "split": "train",
        "description": "8.5K grade-school math word problems",
    },
    "math_hard": {
        "id": "lighteval/MATH-Hard",
        "category": "math",
        "priority": "medium",
        "format": "math_hard",
        "description": "Hard competition math problems",
    },
    "orca_math": {
        "id": "microsoft/orca-math-word-problems-200k",
        "category": "math",
        "priority": "medium",
        "format": "orca_math",
        "description": "200K math word problems",
    },

    # ── LOGIC ──
    "bigbench": {
        "id": "tasksource/bigbench",
        "category": "logic",
        "priority": "high",
        "format": "bigbench",
        "config": "multiple",  # special: load all relevant configs
        "split": "train",
        "description": "BIG-Bench: 200+ diverse logic/reasoning tasks",
    },
    "bbh": {
        "id": "lukaemon/bbh",
        "category": "logic",
        "priority": "high",
        "format": "bbh",
        "config": "multiple",  # special: load all 23 sub-tasks
        "split": "test",
        "description": "Big-Bench-Hard: 23 challenging reasoning tasks",
    },
    "folio": {
        "id": "tasksource/folio",
        "category": "logic",
        "priority": "medium",
        "format": "folio",
        "description": "First-order logic reasoning dataset",
    },
    "planbench": {
        "id": "tasksource/PlanBench",
        "category": "planning",
        "priority": "medium",
        "format": "planbench",
        "config": "multiple",  # load all task configs
        "split": "train",
        "description": "Planning and reasoning benchmark",
    },

    # ── TOOL USE / FUNCTION CALLING ──
    "hermes_fc": {
        "id": "NousResearch/hermes-function-calling-v1",
        "category": "tool_use",
        "priority": "high",
        "format": "hermes_fc",
        "description": "Function calling training data (Hermes)",
    },
    "xlam_fc": {
        "id": "Salesforce/xlam-function-calling-60k",
        "category": "tool_use",
        "priority": "high",
        "format": "xlam_fc",
        "description": "60K function calling examples (xLAM)",
    },

    # ── GENERAL HIGH-QUALITY (filtered for useful content) ──
    "tulu3": {
        "id": "allenai/tulu-3-sft-mixture",
        "category": "general",
        "priority": "medium",
        "format": "tulu3",
        "description": "Tulu 3 SFT mixture (filtered for useful content)",
    },
    "openhermes": {
        "id": "teknium/OpenHermes-2.5",
        "category": "general",
        "priority": "medium",
        "format": "openhermes",
        "description": "1M general instruction pairs (filtered)",
    },
    "no_robots": {
        "id": "HuggingFaceH4/no_robots",
        "category": "general",
        "priority": "medium",
        "format": "no_robots",
        "description": "10K human-written creative + instruction data",
    },
    "slimorca": {
        "id": "Open-Orca/SlimOrca",
        "category": "general",
        "priority": "low",
        "format": "slimorca",
        "description": "SlimOrca: filtered OpenOrca subset",
    },
}


# ── Format converters ──

def _extract_text(obj: Any) -> str:
    """Extract text from various formats (string, list of messages, etc.)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        parts = []
        for item in obj:
            if isinstance(item, dict):
                parts.append(item.get("content", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(obj, dict):
        return obj.get("content", obj.get("text", str(obj)))
    return str(obj)


def convert_magicoder(ex: dict) -> dict | None:
    """Magicoder format: {problem, solution} or {instruction, response}."""
    prompt = ex.get("problem") or ex.get("instruction") or ""
    solution = ex.get("solution") or ex.get("response") or ""
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "magicoder"}


def convert_alpaca(ex: dict) -> dict | None:
    """Alpaca format: {instruction, input, output} OR {prompt, completion}."""
    # Try instruction/input/output format first
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    output = ex.get("output", "")
    if instruction and output:
        prompt = f"{instruction}\n\n{inp}" if inp else instruction
        return {"prompt": prompt, "solution": output, "source": "alpaca"}
    # Try prompt/completion format (CodeAlpaca_20K)
    prompt = ex.get("prompt", "")
    completion = ex.get("completion", "")
    if prompt and completion:
        return {"prompt": prompt, "solution": completion, "source": "alpaca"}
    return None


def convert_openthoughts(ex: dict) -> dict | None:
    """OpenThoughts format: conversations with reasoning."""
    conv = ex.get("conversations") or ex.get("messages")
    if not conv:
        return None
    prompt = ""
    solution = ""
    for msg in conv:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role in ("human", "user"):
            prompt = content
        elif role in ("gpt", "assistant", "model"):
            solution = content
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "openthoughts"}


def convert_dolphin_r1(ex: dict) -> dict | None:
    """Dolphin R1 format: messages (may be string), reasoning, answer."""
    # Dolphin R1 has: messages (list or string), reasoning, answer
    messages = ex.get("messages")
    reasoning = ex.get("reasoning", "")
    answer = ex.get("answer", "")

    prompt = ""
    # Messages might be a string representation of a list
    if isinstance(messages, str):
        try:
            import ast
            messages = ast.literal_eval(messages)
        except Exception:
            messages = None

    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    prompt = content
                elif role == "system" and content:
                    prompt = f"{content}\n\n{prompt}" if prompt else content

    # Fallback to question/prompt fields
    if not prompt:
        prompt = ex.get("question") or ex.get("prompt") or ""

    # Build solution from reasoning + answer
    solution_parts = []
    if reasoning:
        solution_parts.append(f"<think>\n{reasoning}\n</think>")
    if answer:
        solution_parts.append(answer)
    solution = "\n\n".join(solution_parts)

    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "dolphin_r1"}


def convert_openr1_math(ex: dict) -> dict | None:
    """OpenR1 Math format."""
    prompt = ex.get("problem") or ex.get("question") or ""
    solution = ex.get("solution") or ex.get("answer") or ""
    if not prompt or not solution:
        # Try messages
        msgs = ex.get("messages")
        if msgs:
            for msg in msgs:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    solution = msg.get("content", "")
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "openr1_math"}


def convert_metamath(ex: dict) -> dict | None:
    """MetaMath format: {query, response, original_question}."""
    prompt = ex.get("query") or ex.get("question") or ""
    solution = ex.get("response") or ex.get("answer") or ""
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "metamath"}


def convert_gsm8k(ex: dict) -> dict | None:
    """GSM8K format: {question, answer}."""
    prompt = ex.get("question", "")
    solution = ex.get("answer", "")
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "gsm8k"}


def convert_math_hard(ex: dict) -> dict | None:
    """MATH-Hard format."""
    prompt = ex.get("problem") or ex.get("question", "")
    solution = ex.get("solution") or ex.get("answer", "")
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "math_hard"}


def convert_orca_math(ex: dict) -> dict | None:
    """Orca Math format."""
    prompt = ex.get("question", "")
    solution = ex.get("answer", "")
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "orca_math"}


def convert_bigbench(ex: dict) -> dict | None:
    """BIG-Bench format: {inputs, targets, multiple_choice_targets, ...}."""
    # BIG-Bench uses plural keys: inputs (str), targets (list)
    prompt = ex.get("inputs") or ex.get("input") or ex.get("prompt") or ""
    targets = ex.get("targets")
    if targets is not None:
        if isinstance(targets, list):
            solution = "\n".join(str(t) for t in targets)
        else:
            solution = str(targets)
    else:
        solution = ex.get("target") or ex.get("output") or ex.get("answer") or ""

    # For multiple choice, include the choices in the prompt
    mc_targets = ex.get("multiple_choice_targets")
    if mc_targets and isinstance(mc_targets, list) and len(mc_targets) > 1:
        choices = "\n".join(f"  ({chr(65+i)}) {t}" for i, t in enumerate(mc_targets))
        prompt = f"{prompt}\n\nChoose the correct answer:\n{choices}"

    if not prompt or not solution:
        return None
    return {"prompt": str(prompt), "solution": str(solution), "source": "bigbench"}


def convert_bbh(ex: dict) -> dict | None:
    """BBH format: {input, target} or per-task format."""
    prompt = ex.get("input") or ex.get("question", "")
    solution = ex.get("target") or ex.get("answer", "")
    if not prompt or not solution:
        return None
    return {"prompt": str(prompt), "solution": str(solution), "source": "bbh"}


def convert_folio(ex: dict) -> dict | None:
    """FOLIO format: first-order logic reasoning."""
    prompt = ex.get("prompt") or ex.get("question", "")
    solution = ex.get("answer") or ex.get("label", "")
    if not prompt:
        # Build from premises + conclusion
        premises = ex.get("premises", [])
        conclusion = ex.get("conclusion", "")
        if premises and conclusion:
            prompt = f"Premises: {premises}\nConclusion: {conclusion}\nIs the conclusion true, false, or uncertain?"
            solution = ex.get("label", "")
    if not prompt or not solution:
        return None
    return {"prompt": str(prompt), "solution": str(solution), "source": "folio"}


def convert_planbench(ex: dict) -> dict | None:
    """PlanBench format: {query, ground_truth_plan, task, domain, ...}."""
    prompt = ex.get("query") or ex.get("prompt") or ex.get("input") or ""
    solution = ex.get("ground_truth_plan") or ex.get("answer") or ex.get("target") or ""
    if not prompt or not solution:
        return None
    return {"prompt": str(prompt), "solution": str(solution), "source": "planbench"}


def convert_hermes_fc(ex: dict) -> dict | None:
    """Hermes function calling format: conversations with tool calls."""
    conv = ex.get("conversations") or ex.get("messages")
    if not conv:
        return None
    # Reconstruct as text: prompt = user message, solution = assistant response
    prompt = ""
    solution_parts = []
    for msg in conv:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role in ("human", "user", "system"):
            if role == "system":
                prompt = f"{content}\n\n{prompt}" if prompt else content
            else:
                prompt = content
        elif role in ("gpt", "assistant", "model"):
            solution_parts.append(content)
    solution = "\n".join(solution_parts)
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "hermes_fc"}


def convert_xlam_fc(ex: dict) -> dict | None:
    """xLAM function calling format."""
    # xLAM format: {query, answers, tools}
    query = ex.get("query") or ex.get("instruction", "")
    tools = ex.get("tools", "")
    answers = ex.get("answers", "")
    if not query:
        return None
    prompt = f"Tools available: {tools}\n\nQuery: {query}" if tools else query
    solution = str(answers) if answers else ""
    if not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "xlam_fc"}


def convert_codeforces(ex: dict) -> dict | None:
    """Codeforces format: competitive programming with rich metadata."""
    title = ex.get("title", "")
    description = ex.get("description", "")
    input_format = ex.get("input_format", "")
    output_format = ex.get("output_format", "")
    note = ex.get("note", "")
    editorial = ex.get("editorial", "")
    examples = ex.get("examples", [])

    if not description:
        return None

    # Build problem statement
    parts = [f"# {title}"] if title else []
    parts.append(description)
    if input_format:
        parts.append(f"\n## Input\n{input_format}")
    if output_format:
        parts.append(f"\n## Output\n{output_format}")
    if note:
        parts.append(f"\n## Note\n{note}")
    if examples:
        parts.append(f"\n## Examples\n{examples}")
    prompt = "\n".join(parts)

    # Solution = editorial (the official solution explanation)
    if not editorial:
        return None
    solution = str(editorial)

    return {"prompt": str(prompt), "solution": solution, "source": "codeforces"}


def convert_tulu3(ex: dict) -> dict | None:
    """Tulu 3 format: messages."""
    msgs = ex.get("messages")
    if not msgs:
        prompt = ex.get("prompt") or ex.get("instruction", "")
        solution = ex.get("response") or ex.get("output", "")
        if prompt and solution:
            return {"prompt": prompt, "solution": solution, "source": "tulu3"}
        return None
    prompt = ""
    solution = ""
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prompt = content
        elif role == "assistant":
            solution = content
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "tulu3"}


def convert_openhermes(ex: dict) -> dict | None:
    """OpenHermes format: conversations."""
    conv = ex.get("conversations") or ex.get("messages")
    if not conv:
        return None
    prompt = ""
    solution = ""
    for msg in conv:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role in ("human", "user", "system"):
            if role == "system":
                prompt = f"{content}\n\n{prompt}" if prompt else content
            else:
                prompt = content
        elif role in ("gpt", "assistant", "model"):
            solution = content
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "openhermes"}


def convert_no_robots(ex: dict) -> dict | None:
    """no_robots format: messages."""
    msgs = ex.get("messages")
    if not msgs:
        return None
    prompt = ""
    solution = ""
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prompt = content
        elif role == "assistant":
            solution = content
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "no_robots"}


def convert_slimorca(ex: dict) -> dict | None:
    """SlimOrca format: conversations."""
    conv = ex.get("conversations")
    if not conv:
        return None
    prompt = ""
    solution = ""
    for msg in conv:
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role in ("human", "user", "system"):
            if role == "system":
                prompt = f"{content}\n\n{prompt}" if prompt else content
            else:
                prompt = content
        elif role in ("gpt", "assistant"):
            solution = content
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "source": "slimorca"}


CONVERTERS = {
    "magicoder": convert_magicoder,
    "alpaca": convert_alpaca,
    "openthoughts": convert_openthoughts,
    "dolphin_r1": convert_dolphin_r1,
    "openr1_math": convert_openr1_math,
    "metamath": convert_metamath,
    "gsm8k": convert_gsm8k,
    "math_hard": convert_math_hard,
    "orca_math": convert_orca_math,
    "bigbench": convert_bigbench,
    "bbh": convert_bbh,
    "folio": convert_folio,
    "planbench": convert_planbench,
    "hermes_fc": convert_hermes_fc,
    "xlam_fc": convert_xlam_fc,
    "codeforces": convert_codeforces,
    "tulu3": convert_tulu3,
    "openhermes": convert_openhermes,
    "no_robots": convert_no_robots,
    "slimorca": convert_slimorca,
}


def download_dataset(name: str, config: dict, output_dir: Path,
                     max_per_dataset: int = 50000,
                     use_streaming: bool = False) -> dict:
    """Download a single dataset, filter, and save as JSONL.

    Returns:
        statistics dict
    """
    ds_id = config["id"]
    category = config["category"]
    fmt = config["format"]
    hf_config = config.get("config")  # HF config name (may be None)
    default_split = config.get("split", "train")
    converter = CONVERTERS.get(fmt)

    if converter is None:
        return {"error": f"no converter for format {fmt}"}

    output_file = output_dir / f"{name}.jsonl"
    stats = {
        "dataset": name,
        "id": ds_id,
        "category": category,
        "output": str(output_file),
        "total": 0,
        "kept": 0,
        "excluded": 0,
        "errors": 0,
    }

    if output_file.exists():
        # Count existing
        existing = sum(1 for _ in open(output_file, encoding="utf-8"))
        if existing >= max_per_dataset:
            stats["kept"] = existing
            stats["skipped"] = True
            print(f"  [{name}] Already have {existing} examples — skipping")
            return stats

    print(f"  [{name}] Loading {ds_id}...")

    try:
        from datasets import load_dataset

        config_name = hf_config
        split = default_split

        # Handle multi-config datasets (load all relevant configs)
        if config_name == "multiple":
            # Get available configs
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                info = api.dataset_info(ds_id)
                available = [s.id for s in info.siblings
                             if s.id and not s.id.endswith((".json", ".md", ".py",
                             ".gitattributes"))]
            except Exception:
                available = []

            # For BBH, load all 23 sub-tasks
            if ds_id == "lukaemon/bbh":
                configs_to_load = [
                    "boolean_expressions", "causal_judgement", "date_understanding",
                    "disambiguation_qa", "dyck_languages", "formal_fallacies",
                    "geometric_shapes", "hyperbaton",
                    "logical_deduction_five_objects",
                    "logical_deduction_seven_objects",
                    "logical_deduction_three_objects",
                    "movie_recommendation", "multistep_arithmetic_two",
                    "navigate", "object_counting", "penguins_in_a_table",
                    "reasoning_about_colored_objects", "ruin_names",
                    "salient_translation_error_detection", "snarks",
                    "sports_understanding", "temporal_sequences",
                    "tracking_shuffled_objects_five_objects",
                    "tracking_shuffled_objects_seven_objects",
                    "tracking_shuffled_objects_three_objects",
                    "web_of_lies", "word_sorting",
                ]
            elif ds_id == "tasksource/PlanBench":
                configs_to_load = [
                    "task_1_plan_generation",
                    "task_2_plan_optimality",
                    "task_3_plan_verification",
                    "task_5_plan_generalization",
                    "task_7_plan_execution",
                ]
            elif ds_id == "tasksource/bigbench":
                # Pick logic/reasoning/code relevant sub-tasks from 167 available
                configs_to_load = [
                    "logical_deduction", "logical_fallacy_detection",
                    "logical_sequence", "logical_args",
                    "formal_fallacies_syllogisms_negation",
                    "cause_and_effect", "analogical_similarity",
                    "analytic_entailment", "epistemic_reasoning",
                    "implicatures", "presuppositions_as_nli",
                    "strategyqa", "strange_stories",
                    "reasoning_about_colored_objects",
                    "repeat_copy_logic", "navigate",
                    "object_counting", "arithmetic",
                    "elementary_math_qa", "modified_arithmetic",
                    "mathematical_induction", "cs_algorithms",
                    "code_line_description", "auto_debugging",
                    "goal_step_wikihow", "novel_concepts",
                    "riddle_sense", "odd_one_out",
                    "operators", "list_functions",
                    "penguins_in_a_table", "temporal_sequences",
                    "tracking_shuffled_objects", "word_sorting",
                    "dyck_languages", "symbol_interpretation",
                    "understanding_fables", "winowhy",
                    "sufficient_information", "tellmewy",
                    "physical_intuition", "physics",
                    "unit_conversion", "unit_interpretation",
                    "timedial", "tense",
                    "key_value_maps", "mult_data_wrangling",
                    "semantic_parsing_spider",
                    "semantic_parsing_in_context_sparc",
                    "undo_permutation", "word_unscrambling",
                    "chinese_remainder_theorem",
                    "simple_arithmetic_json",
                    "simple_arithmetic_json_subtasks",
                ]
            else:
                configs_to_load = []

            count = 0
            with open(output_file, "w", encoding="utf-8") as f:
                for cfg_name in configs_to_load:
                    if count >= max_per_dataset:
                        break
                    try:
                        if use_streaming:
                            ds = load_dataset(ds_id, cfg_name,
                                            split=split, streaming=True)
                        else:
                            ds = load_dataset(ds_id, cfg_name,
                                            split=split)
                    except Exception as e:
                        print(f"    [{name}/{cfg_name}] skip: {str(e)[:60]}")
                        continue

                    for ex in ds:
                        if count >= max_per_dataset:
                            break
                        stats["total"] += 1
                        try:
                            pair = converter(ex)
                            if pair is None:
                                stats["errors"] += 1
                                continue
                            combined = pair["prompt"] + " " + pair["solution"]
                            if should_exclude(combined):
                                stats["excluded"] += 1
                                continue
                            pair["category"] = category
                            pair["dataset"] = name
                            pair["subtask"] = cfg_name
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            stats["kept"] += 1
                            count += 1
                        except Exception:
                            stats["errors"] += 1

                print(f"  [{name}] Done: kept={stats['kept']} "
                      f"excluded={stats['excluded']} errors={stats['errors']}")
            return stats

        # Single config
        try:
            if use_streaming:
                ds = load_dataset(ds_id, config_name, split=split,
                                streaming=True)
            else:
                ds = load_dataset(ds_id, config_name, split=split)
        except Exception:
            # Try without split
            try:
                if use_streaming:
                    ds = load_dataset(ds_id, config_name, streaming=True)
                    ds = ds[list(ds.keys())[0]]
                else:
                    ds = load_dataset(ds_id, config_name)
                    ds = ds[list(ds.keys())[0]]
            except Exception:
                # Try without config
                try:
                    if use_streaming:
                        ds = load_dataset(ds_id, split=split, streaming=True)
                    else:
                        ds = load_dataset(ds_id, split=split)
                except Exception as e:
                    stats["error"] = f"load failed: {e}"
                    print(f"  [{name}] LOAD ERROR: {e}")
                    return stats

        with open(output_file, "w", encoding="utf-8") as f:
            count = 0
            for ex in ds:
                if count >= max_per_dataset:
                    break
                stats["total"] += 1

                try:
                    pair = converter(ex)
                    if pair is None:
                        stats["errors"] += 1
                        continue

                    # Filter excluded content
                    combined = pair["prompt"] + " " + pair["solution"]
                    if should_exclude(combined):
                        stats["excluded"] += 1
                        continue

                    # Add metadata
                    pair["category"] = category
                    pair["dataset"] = name

                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
                    count += 1

                    if count % 10000 == 0:
                        print(f"    {name}: {count}/{max_per_dataset} "
                              f"(excluded={stats['excluded']})")

                except Exception:
                    stats["errors"] += 1

    except Exception as e:
        stats["error"] = str(e)
        print(f"  [{name}] ERROR: {e}")
        return stats

    print(f"  [{name}] Done: kept={stats['kept']} "
          f"excluded={stats['excluded']} errors={stats['errors']}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download HF datasets for ForgeLM V3 training")
    parser.add_argument("--output-dir", type=str,
                        default="research/distillation/hf_datasets",
                        help="Output directory for JSONL files")
    parser.add_argument("--max-per-dataset", type=int, default=50000,
                        help="Max examples per dataset (default: 50000)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["code", "reasoning", "math", "logic",
                                "planning", "tool_use", "general"],
                        help="Only download specific category")
    parser.add_argument("--priority", type=str, default=None,
                        choices=["high", "medium", "low"],
                        help="Only download specific priority level")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode (for large datasets)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter datasets
    to_download = {}
    for name, config in DATASETS.items():
        if args.datasets and name not in args.datasets:
            continue
        if args.category and config["category"] != args.category:
            continue
        if args.priority and config["priority"] != args.priority:
            continue
        to_download[name] = config

    print("=" * 70)
    print("  ForgeAI HF Dataset Downloader")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Max per dataset: {args.max_per_dataset}")
    print(f"  Streaming: {args.stream}")
    print(f"  Datasets to download: {len(to_download)}")
    print()

    for name, config in to_download.items():
        print(f"  {name:25s} [{config['category']:10s}] "
              f"{config['priority']:6s} — {config['description'][:50]}")
    print()

    all_stats = []
    t0 = time.time()

    for name, config in to_download.items():
        stats = download_dataset(
            name, config, output_dir,
            max_per_dataset=args.max_per_dataset,
            use_streaming=args.stream,
        )
        all_stats.append(stats)

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    total_kept = 0
    total_excluded = 0
    total_errors = 0
    for s in all_stats:
        kept = s.get("kept", 0)
        excluded = s.get("excluded", 0)
        errors = s.get("errors", 0)
        total_kept += kept
        total_excluded += excluded
        total_errors += errors
        status = "OK" if "error" not in s else "ERROR"
        print(f"  {status:5s} {s['dataset']:25s} "
              f"kept={kept:6d} excluded={excluded:4d} errors={errors:4d}")
    print(f"\n  Total kept:     {total_kept:,}")
    print(f"  Total excluded: {total_excluded:,}")
    print(f"  Total errors:   {total_errors:,}")
    print(f"  Elapsed:        {elapsed:.0f}s")

    # Save stats
    stats_file = output_dir / "download_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Stats saved to: {stats_file}")


if __name__ == "__main__":
    main()
