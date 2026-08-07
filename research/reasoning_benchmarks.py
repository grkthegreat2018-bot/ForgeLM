"""Reasoning & creativity benchmarks for ForgeLM.

Four complementary benchmarks that measure different aspects of thinking quality:

1. ARC-AGI-2 (2025) — Fluid intelligence, abstraction, novel pattern recognition
   - 120 unique tasks, each completely novel (no memorization possible)
   - Pure LLMs score 0%; any improvement is real cognitive gain
   - Tests: can the model discover rules from examples and apply them?

2. NeoCoder/NeoGauge (NAACL 2025) — Creative code generation via denial prompting
   - 199 Codeforces problems with incrementally banned approaches
   - Tests: can the model find NEW strategies when old ones are forbidden?
   - Measures convergent (correctness) + divergent (novelty) thinking

3. FineReason (ACL 2025) — Deliberate reasoning with intermediate step validation
   - Logic puzzles decomposable into atomic steps
   - Tests: can the model reflect and correct within reasoning?

4. ThinkBench (NeurIPS 2025) — OOD reasoning robustness
   - Dynamic OOD data generation, 2912 samples
   - Tests: does the model's reasoning generalize beyond distribution?

Usage:
    from research.reasoning_benchmarks import ReasoningBenchmarkSuite
    suite = ReasoningBenchmarkSuite(model, tokenizer, device="cuda")
    results = suite.run(benchmarks=["arc_agi2", "neocoder"], n_problems=20)
    suite.print_report(results)

    # Or via CLI:
    # python -m research.reasoning_benchmarks --benchmarks arc_agi2,neocoder --n-problems 20
"""
import os
import sys
import json
import time
import re
import subprocess
import tempfile
import urllib.request
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── Shared infrastructure ───────────────────────────────────────────

BENCH_CACHE_DIR = "D:/windsurf/ForgeAI/.devin/bench_cache"
BENCH_LOG_DIR = "D:/windsurf/ForgeAI/research/data/reasoning_bench"


def _ensure_dirs():
    os.makedirs(BENCH_CACHE_DIR, exist_ok=True)
    os.makedirs(BENCH_LOG_DIR, exist_ok=True)


def _download_file(url: str, dest: str) -> str:
    """Download a file if not already cached."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")
    return dest


def _generate_solution(model, tokenizer, prompt: str, device: str = "cuda",
                       max_tokens: int = 512, temperature: float = 0.0) -> Tuple[str, Dict]:
    """Generate text using ForgeLM with telemetry."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    t0 = time.time()
    log_probs = []

    with torch.inference_mode():
        cur_ids = input_ids
        for step in range(max_tokens):
            logits, _ = model(cur_ids)
            next_logits = logits[0, -1]

            if temperature <= 0.0:
                next_token = next_logits.argmax()
            else:
                scaled = next_logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()

            lp = torch.log_softmax(next_logits, dim=-1)[next_token].item()
            log_probs.append(lp)

            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    gen_time_ms = (time.time() - t0) * 1000
    tokens_generated = cur_ids.shape[1] - input_ids.shape[1]

    generated_text = tokenizer.decode(cur_ids[0, input_ids.shape[1]:],
                                       skip_special_tokens=True)

    telemetry = {
        "gen_time_ms": gen_time_ms,
        "tokens_generated": tokens_generated,
        "mean_logprob": sum(log_probs) / max(len(log_probs), 1),
    }
    return generated_text, telemetry


def _extract_code(text: str) -> str:
    """Extract Python code from markdown fences or raw text."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _run_code_with_input(code: str, test_input: str,
                         timeout_s: float = 10.0) -> Tuple[str, str, bool]:
    """Run Python code with stdin input, return (stdout, stderr, success)."""
    code = _extract_code(text=code)
    temp_dir = "D:/windsurf/ForgeAI/.devin/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                      encoding='utf-8', dir=temp_dir) as f:
        f.write(code)
        code_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, code_path],
            input=test_input, capture_output=True, text=True,
            timeout=timeout_s, cwd=temp_dir,
            encoding='utf-8', errors='replace')
        return proc.stdout.strip(), proc.stderr.strip(), proc.returncode == 0
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", False
    finally:
        try:
            os.unlink(code_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# 1. ARC-AGI-2 — Fluid Intelligence / Abstract Reasoning
# ═══════════════════════════════════════════════════════════════════════

ARC_AGI2_REPO = "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main"
ARC_AGI2_API = "https://api.github.com/repos/arcprize/ARC-AGI-2/contents/data/evaluation"


def load_arc_agi2(max_tasks: Optional[int] = None) -> List[Dict]:
    """Load ARC-AGI-2 evaluation tasks from GitHub."""
    _ensure_dirs()
    cache_dir = os.path.join(BENCH_CACHE_DIR, "arc_agi2")
    os.makedirs(cache_dir, exist_ok=True)

    # Get file listing from GitHub API
    list_file = os.path.join(cache_dir, "_file_list.json")
    if not os.path.exists(list_file):
        print("[ARC-AGI-2] Fetching task list from GitHub...")
        urllib.request.urlretrieve(ARC_AGI2_API, list_file)

    with open(list_file, "r") as f:
        files = json.load(f)

    task_files = [f for f in files if f["name"].endswith(".json")]
    print(f"[ARC-AGI-2] Found {len(task_files)} evaluation tasks")

    if max_tasks:
        task_files = task_files[:max_tasks]

    tasks = []
    for tf in task_files:
        task_path = os.path.join(cache_dir, tf["name"])
        if not os.path.exists(task_path):
            urllib.request.urlretrieve(tf["download_url"], task_path)
        with open(task_path, "r") as f:
            task = json.load(f)
            task["task_id"] = tf["name"].replace(".json", "")
            tasks.append(task)

    print(f"[ARC-AGI-2] Loaded {len(tasks)} tasks")
    return tasks


def _grid_to_text(grid: List[List[int]]) -> str:
    """Convert a grid to a compact text representation."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _parse_grid_output(text: str) -> Optional[List[List[int]]]:
    """Parse model output into a grid."""
    text = text.strip()
    # Try to extract from code block
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()

    # Try to find grid-like patterns (rows of numbers)
    grid = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match rows of single-digit numbers separated by spaces
        nums = re.findall(r"\b(\d)\b", line)
        if nums and len(nums) > 1:
            grid.append([int(n) for n in nums])

    if grid:
        return grid

    # Try parsing as Python list literal
    try:
        result = eval(text.split("\n")[0])
        if isinstance(result, list) and all(isinstance(r, list) for r in result):
            return result
    except Exception:
        pass

    return None


def _grids_equal(g1: List[List[int]], g2: List[List[int]]) -> bool:
    """Check if two grids are equal."""
    if len(g1) != len(g2):
        return False
    return all(r1 == r2 for r1, r2 in zip(g1, g2))


def build_arc_agi2_prompt(task: Dict) -> str:
    """Build a prompt for ARC-AGI-2 task."""
    prompt = "You are solving an abstract reasoning puzzle.\n"
    prompt += "Each puzzle shows input/output grid pairs. Discover the transformation rule.\n"
    prompt += "Then apply it to the test input to produce the output grid.\n"
    prompt += "Colors are represented by digits 0-9.\n\n"

    prompt += "# Training Examples:\n"
    for i, ex in enumerate(task["train"]):
        prompt += f"\n## Example {i+1}:\n"
        prompt += f"Input:\n{_grid_to_text(ex['input'])}\n\n"
        prompt += f"Output:\n{_grid_to_text(ex['output'])}\n\n"

    prompt += "# Test Input:\n"
    test_input = task["test"][0]["input"]
    prompt += f"{_grid_to_text(test_input)}\n\n"
    prompt += "# Test Output (provide the transformed grid):\n"
    return prompt


def run_arc_agi2(model, tokenizer, device: str = "cuda",
                 max_tasks: Optional[int] = None,
                 max_tokens: int = 400) -> Dict:
    """Run ARC-AGI-2 evaluation."""
    print("\n" + "=" * 60)
    print("ARC-AGI-2: Fluid Intelligence Benchmark")
    print("=" * 60)

    tasks = load_arc_agi2(max_tasks=max_tasks)
    results = []
    n_solved = 0

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n  [{i+1}/{len(tasks)}] Task {tid}")

        prompt = build_arc_agi2_prompt(task)
        output, telemetry = _generate_solution(
            model, tokenizer, prompt, device=device,
            max_tokens=max_tokens, temperature=0.0)

        predicted = _parse_grid_output(output)
        expected = task["test"][0].get("output")

        solved = False
        if predicted is not None and expected is not None:
            solved = _grids_equal(predicted, expected)

        if solved:
            n_solved += 1
            print(f"    SOLVED! ({telemetry['tokens_generated']} tokens)")

        results.append({
            "task_id": tid,
            "solved": solved,
            "predicted_shape": f"{len(predicted)}x{len(predicted[0])}" if predicted else "None",
            "expected_shape": f"{len(expected)}x{len(expected[0])}" if expected else "None",
            "gen_time_ms": telemetry["gen_time_ms"],
            "tokens_generated": telemetry["tokens_generated"],
        })

    pass_rate = n_solved / len(tasks) if tasks else 0.0
    print(f"\n[ARC-AGI-2] Solved: {n_solved}/{len(tasks)} ({pass_rate:.1%})")

    return {
        "benchmark": "arc_agi2",
        "n_solved": n_solved,
        "n_total": len(tasks),
        "pass_rate": pass_rate,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. NeoCoder — Creative Code Generation via Denial Prompting
# ═══════════════════════════════════════════════════════════════════════

NEOCODER_BASE = "https://raw.githubusercontent.com/JHU-CLSP/NeoCoder/main/datasets/CodeForce/NeoCoder"


def load_neocoder(max_problems: Optional[int] = None) -> Tuple[List[Dict], Dict, Dict]:
    """Load NeoCoder dataset: problems, human solutions, test cases."""
    _ensure_dirs()
    cache_dir = os.path.join(BENCH_CACHE_DIR, "neocoder")
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "problems": "NeoCoder.json",
        "human_solutions": "human_solutions.json",
        "test_cases": "test_cases_annotated.json",
    }

    data = {}
    for key, fname in files.items():
        dest = os.path.join(cache_dir, fname)
        if not os.path.exists(dest):
            _download_file(f"{NEOCODER_BASE}/{fname}", dest)
        with open(dest, "r", encoding="utf-8") as f:
            data[key] = json.load(f)

    problems = data["problems"]
    if max_problems:
        problems = problems[:max_problems]

    print(f"[NeoCoder] Loaded {len(problems)} problems, "
          f"{len(data['human_solutions'])} human solutions, "
          f"{len(data['test_cases'])} test cases")

    return problems, data["human_solutions"], data["test_cases"]


def _build_neocoder_prompt(problem: Dict, denied_approaches: List[str] = None) -> str:
    """Build a prompt for a NeoCoder problem with optional denial constraints."""
    # Extract problem text — NeoCoder format varies, try common fields
    statement = problem.get("statement", problem.get("question_content", ""))
    title = problem.get("title", problem.get("question_title", "Unknown"))
    difficulty = problem.get("difficulty", "")

    prompt = f"# Problem: {title}\n"
    if difficulty:
        prompt += f"# Difficulty: {difficulty}\n"
    prompt += f"\n{statement}\n\n"

    if denied_approaches:
        prompt += "# CONSTRAINT: You may NOT use the following approaches:\n"
        for d in denied_approaches:
            prompt += f"#   - {d}\n"
        prompt += "# Find a DIFFERENT approach to solve this problem.\n\n"

    prompt += "# Write a Python solution. Read from stdin, print to stdout.\n"
    prompt += "# Solution:\n"
    return prompt


def _extract_technique(code: str) -> str:
    """Extract a high-level technique description from code (heuristic)."""
    code_lower = code.lower()
    techniques = []
    if "sort(" in code_lower or ".sort(" in code_lower:
        techniques.append("sorting")
    if "dp[" in code_lower or "memo" in code_lower:
        techniques.append("dynamic programming")
    if "bfs" in code_lower or "queue" in code_lower or "deque" in code_lower:
        techniques.append("BFS")
    if "dfs" in code_lower or "stack" in code_lower:
        techniques.append("DFS")
    if "graph" in code_lower or "adj" in code_lower:
        techniques.append("graph traversal")
    if "heap" in code_lower or "priority" in code_lower:
        techniques.append("heap/priority queue")
    if "binary" in code_lower and "search" in code_lower:
        techniques.append("binary search")
    if "greedy" in code_lower:
        techniques.append("greedy")
    if "math" in code_lower or "gcd" in code_lower or "lcm" in code_lower:
        techniques.append("math/number theory")
    if "hash" in code_lower or "set(" in code_lower or "dict(" in code_lower:
        techniques.append("hashing")
    if "recursion" in code_lower or code_lower.count("def ") > 1:
        techniques.append("recursion")
    return ", ".join(techniques) if techniques else "brute force"


def run_neocoder(model, tokenizer, device: str = "cuda",
                 max_problems: Optional[int] = None,
                 denial_rounds: int = 3,
                 max_tokens: int = 512,
                 timeout_s: float = 10.0) -> Dict:
    """Run NeoCoder creativity evaluation with denial prompting.

    For each problem:
      Round 0: Solve normally
      Round 1: Ban the technique used in round 0
      Round 2: Ban techniques from rounds 0+1
      Round 3: Ban techniques from rounds 0+1+2

    NeoGauge = (convergent_score * divergent_score)
      convergent: fraction of rounds that produce correct solutions
      divergent: fraction of rounds that use DIFFERENT techniques
    """
    print("\n" + "=" * 60)
    print("NeoCoder: Creative Code Generation Benchmark")
    print("=" * 60)

    problems, human_solutions, test_cases = load_neocoder(max_problems=max_problems)

    results = []
    total_neogauge = 0.0

    for i, problem in enumerate(problems):
        pid = str(problem.get("id", problem.get("question_id", i)))
        title = problem.get("title", problem.get("question_title", "?"))[:50]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n  [{i+1}/{len(problems)}] {title}")

        # Get test cases for this problem
        tc_list = test_cases.get(pid, test_cases.get(str(i), []))
        if isinstance(tc_list, dict):
            tc_list = tc_list.get("test_cases", [])

        round_results = []
        denied = []
        techniques_used = []

        for rnd in range(denial_rounds + 1):
            prompt = _build_neocoder_prompt(problem, denied_approaches=denied if rnd > 0 else None)
            code, telemetry = _generate_solution(
                model, tokenizer, prompt, device=device,
                max_tokens=max_tokens, temperature=0.3 if rnd > 0 else 0.0)

            # Evaluate
            correct = False
            if tc_list:
                n_pass = 0
                for tc in tc_list[:3]:  # Test on first 3 cases
                    inp = tc.get("input", "")
                    exp = tc.get("output", "").strip()
                    actual, stderr, ok = _run_code_with_input(code, inp, timeout_s)
                    if ok and actual == exp:
                        n_pass += 1
                correct = n_pass == min(3, len(tc_list))

            technique = _extract_technique(code)
            techniques_used.append(technique)

            round_results.append({
                "round": rnd,
                "correct": correct,
                "technique": technique,
                "tokens": telemetry["tokens_generated"],
            })

            # Add denied approach for next round
            if technique != "brute force":
                denied.append(technique)

        # Calculate NeoGauge
        convergent = sum(1 for r in round_results if r["correct"]) / len(round_results)
        unique_techniques = len(set(techniques_used))
        divergent = (unique_techniques - 1) / max(len(round_results) - 1, 1)
        divergent = max(0.0, min(1.0, divergent))
        neogauge = convergent * divergent
        total_neogauge += neogauge

        n_correct = sum(1 for r in round_results if r["correct"])
        if (i + 1) % 10 == 0 or i == 0 or neogauge > 0:
            print(f"    Round 0: {'PASS' if round_results[0]['correct'] else 'FAIL'} "
                  f"({round_results[0]['technique']})")
            for r in round_results[1:]:
                print(f"    Round {r['round']}: {'PASS' if r['correct'] else 'FAIL'} "
                      f"({r['technique']})")
            print(f"    NeoGauge: {neogauge:.3f} (conv={convergent:.2f} div={divergent:.2f})")

        results.append({
            "problem_id": pid,
            "title": title,
            "rounds": round_results,
            "convergent": convergent,
            "divergent": divergent,
            "neogauge": neogauge,
            "n_correct": n_correct,
            "techniques_used": techniques_used,
        })

    avg_neogauge = total_neogauge / len(problems) if problems else 0.0
    avg_convergent = sum(r["convergent"] for r in results) / max(len(results), 1)
    avg_divergent = sum(r["divergent"] for r in results) / max(len(results), 1)

    print(f"\n[NeoCoder] Avg NeoGauge: {avg_neogauge:.3f}")
    print(f"  Convergent (correctness): {avg_convergent:.3f}")
    print(f"  Divergent (creativity):   {avg_divergent:.3f}")

    return {
        "benchmark": "neocoder",
        "n_problems": len(problems),
        "avg_neogauge": avg_neogauge,
        "avg_convergent": avg_convergent,
        "avg_divergent": avg_divergent,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. FineReason — Deliberate Reasoning with Step Validation
# ═══════════════════════════════════════════════════════════════════════

def load_finereason(max_problems: Optional[int] = None) -> List[Dict]:
    """Load FineReason logic puzzle dataset.

    FineReason (ACL 2025, DAMO-NLP-SG) uses 4 puzzle types:
      - Sudoku (9x9 grid state checking)
      - Graph Coloring
      - Game of 24
      - Grid Puzzles (logic grid)

    Two evaluation tasks:
      - State Checking: is the current state solvable? (exact match)
      - State Transition: is this the next valid move? (binary)

    Data format: JSONL with Sample(inputs: dict, outputs: dict, prompt: str)
    The outputs dict has 'current_status' for state checking.

    Hosted at: https://github.com/DAMO-NLP-SG/FineReason (data/ directory)
    Training data: https://huggingface.co/datasets/Guizhen/Puzzles_10k
    """
    _ensure_dirs()
    cache_dir = os.path.join(BENCH_CACHE_DIR, "finereason")
    os.makedirs(cache_dir, exist_ok=True)

    # Try to load from GitHub (DAMO-NLP-SG/FineReason)
    puzzle_types = ["sudoku", "graphcoloring", "game24", "gridpuzzle"]
    problems = []

    for ptype in puzzle_types:
        # Try state checking data
        for task_suffix in ["_questions.json", "_states.json"]:
            fname = f"{ptype}{task_suffix}"
            dest = os.path.join(cache_dir, fname)
            url = f"https://raw.githubusercontent.com/DAMO-NLP-SG/FineReason/main/data/{fname}"
            try:
                if not os.path.exists(dest):
                    _download_file(url, dest)
                with open(dest, "r", encoding="utf-8") as f:
                    # Could be JSONL or JSON
                    content = f.read().strip()
                    if content.startswith("["):
                        data = json.loads(content)
                    else:
                        data = [json.loads(line) for line in content.split("\n") if line.strip()]
                    for item in data:
                        item["puzzle_type"] = ptype
                        problems.append(item)
            except Exception as e:
                print(f"  [FineReason] Could not load {fname}: {e}")

    if problems:
        if max_problems:
            problems = problems[:max_problems]
        print(f"[FineReason] Loaded {len(problems)} puzzles from GitHub "
              f"(types: {set(p.get('puzzle_type','?') for p in problems)})")
        return problems

    print("[FineReason] GitHub data unavailable, using built-in puzzles")
    builtin_puzzles = _get_builtin_logic_puzzles()
    if max_problems:
        builtin_puzzles = builtin_puzzles[:max_problems]
    print(f"[FineReason] Using {len(builtin_puzzles)} built-in puzzles")
    return builtin_puzzles


def _get_builtin_logic_puzzles() -> List[Dict]:
    """Built-in logic puzzles for reasoning evaluation.

    Each puzzle has:
      - setup: the problem description
      - steps: ordered intermediate conclusions to verify
      - answer: the final answer
    """
    return [
        {
            "id": "puzzle_1",
            "setup": (
                "Three friends — Alice, Bob, and Carol — have different favorite colors: "
                "red, blue, and green (not necessarily in that order).\n"
                "Clues:\n"
                "1. Alice does not like red.\n"
                "2. Bob's favorite color is not blue.\n"
                "3. The person who likes green is not Bob.\n"
                "What is each person's favorite color?"
            ),
            "steps": [
                "From clue 3: Bob does not like green",
                "From clue 2: Bob does not like blue",
                "Therefore Bob likes red",
                "From clue 1: Alice does not like red (Bob has red)",
                "From clue 3: Bob has red, so green goes to Alice or Carol",
                "Alice cannot like red, so Alice likes blue or green",
                "If Alice likes green, Carol likes blue. If Alice likes blue, Carol likes green.",
                "Need another constraint — but with 3 colors and Bob=red, "
                "Alice and Carol split blue and green",
                "Answer: Bob=red, and (Alice=blue, Carol=green) or (Alice=green, Carol=blue)",
            ],
            "answer": "Bob=red, Alice=blue, Carol=green OR Bob=red, Alice=green, Carol=blue",
            "answer_patterns": ["bob=red", "alice", "carol", "blue", "green"],
        },
        {
            "id": "puzzle_2",
            "setup": (
                "A farmer needs to cross a river with a fox, a chicken, and a bag of grain.\n"
                "The boat can only carry the farmer and one item at a time.\n"
                "If left alone: the fox will eat the chicken, the chicken will eat the grain.\n"
                "How can the farmer get everything across safely?"
            ),
            "steps": [
                "Step 1: Take chicken across (fox and grain are safe together)",
                "Step 2: Return alone",
                "Step 3: Take fox across",
                "Step 4: Bring chicken back (can't leave fox with chicken)",
                "Step 5: Take grain across",
                "Step 6: Return alone",
                "Step 7: Take chicken across",
            ],
            "answer": "Chicken, return, Fox, Chicken back, Grain, return, Chicken",
            "answer_patterns": ["chicken", "fox", "grain", "back", "return"],
        },
        {
            "id": "puzzle_3",
            "setup": (
                "Five houses in a row, each a different color: red, blue, green, yellow, white.\n"
                "Each occupied by a person of different nationality: American, Brit, Dane, German, Swede.\n"
                "Each drinks a different beverage: coffee, tea, milk, water, beer.\n"
                "Clues:\n"
                "1. The Brit lives in the red house.\n"
                "2. The Dane drinks tea.\n"
                "3. The green house is immediately to the left of the white house.\n"
                "4. The person in the green house drinks coffee.\n"
                "5. The person in the middle house drinks milk.\n"
                "Who drinks water? Who owns the fish (5th pet)?"
            ),
            "steps": [
                "House 3 (middle) drinks milk (clue 5)",
                "Green house is left of white house (clue 3) — could be houses 3-4 or 4-5",
                "Green house drinks coffee (clue 4) — but house 3 drinks milk, so green is house 4",
                "So house 4=green=coffee, house 5=white",
                "Brit lives in red house (clue 1) — red is not green or white, so red is 1, 2, or 3",
                "Dane drinks tea (clue 2) — not house 3 (milk) or 4 (coffee)",
                "This requires more steps to fully solve...",
            ],
            "answer": "The German drinks water and owns the fish (in the standard solution)",
            "answer_patterns": ["german", "water", "fish"],
        },
        {
            "id": "puzzle_4",
            "setup": (
                "You have 8 balls. One is heavier than the rest (all others weigh the same).\n"
                "You have a balance scale. What is the minimum number of weighings "
                "to find the heavy ball?"
            ),
            "steps": [
                "Divide 8 balls into 3 groups: 3, 3, 2",
                "Weighing 1: Weigh 3 vs 3",
                "If one side is heavier, the heavy ball is in that group of 3",
                "If balanced, the heavy ball is in the group of 2",
                "If in group of 3: Weighing 2: weigh 1 vs 1 from that group",
                "  If one is heavier, that's it. If balanced, the third is heavy.",
                "If in group of 2: Weighing 2: weigh 1 vs 1, heavier one is the answer",
                "Total: 2 weighings minimum",
            ],
            "answer": "2 weighings",
            "answer_patterns": ["2", "two", "weighing"],
        },
        {
            "id": "puzzle_5",
            "setup": (
                "A clock shows 3:15. What is the angle between the hour and minute hands?\n"
                "Give the exact answer in degrees."
            ),
            "steps": [
                "At 3:00, hour hand is at 90 degrees (3 * 30)",
                "Minute hand at 15 minutes is at 90 degrees (15 * 6)",
                "But hour hand moves: at 3:15, hour hand has moved 15/60 * 30 = 7.5 degrees",
                "Hour hand position: 90 + 7.5 = 97.5 degrees",
                "Minute hand position: 90 degrees",
                "Angle: |97.5 - 90| = 7.5 degrees",
            ],
            "answer": "7.5 degrees",
            "answer_patterns": ["7.5", "7.5 degree"],
        },
    ]


def build_finereason_prompt(puzzle: Dict) -> str:
    """Build a prompt for a FineReason logic puzzle.

    Handles two formats:
    1. Real FineReason (GitHub): has 'inputs' dict, 'outputs' dict, 'prompt' str
    2. Built-in fallback: has 'setup' str, 'steps' list, 'answer' str
    """
    prompt = "Solve this logic puzzle step by step. Show your reasoning.\n\n"

    # Real FineReason format: use the provided prompt field
    if "prompt" in puzzle and puzzle["prompt"]:
        prompt += puzzle["prompt"]
    elif "inputs" in puzzle:
        # Construct prompt from inputs dict
        inputs = puzzle["inputs"]
        prompt += json.dumps(inputs, indent=2)
    elif "setup" in puzzle:
        # Built-in format
        prompt += puzzle["setup"]
    else:
        prompt += str(puzzle)

    prompt += "\n\n# Reasoning (step by step):\n"
    return prompt


def _evaluate_finereason(response: str, puzzle: Dict) -> Tuple[bool, int, str]:
    """Evaluate a FineReason response.

    Real format: outputs['current_status'] (exact match)
    Built-in format: answer_patterns list

    Returns (correct, n_steps_matched, detail).
    """
    # Real FineReason format: exact match on current_status
    if "outputs" in puzzle:
        expected = str(puzzle["outputs"].get("current_status", "")).strip().lower()
        # Extract the model's prediction (last non-empty line or explicit answer)
        response_clean = response.strip().lower()
        last_lines = [l.strip() for l in response_clean.split("\n") if l.strip()]
        pred = last_lines[-1] if last_lines else ""

        # Also check if expected value appears anywhere
        correct = (pred == expected) or (expected and expected in response_clean)
        n_steps = sum(1 for line in response.split("\n")
                      if any(s in line.lower() for s in
                             ["therefore", "so ", "thus", "hence", "step ", "because", "since"]))
        return correct, n_steps, f"expected={expected}, pred={pred[:30]}"

    # Built-in format: pattern matching
    patterns = puzzle.get("answer_patterns", [])
    if not patterns:
        return False, 0, "no patterns"

    response_lower = response.lower()
    n_matched = sum(1 for p in patterns if p.lower() in response_lower)
    threshold = max(1, int(len(patterns) * 0.6))
    correct = n_matched >= threshold

    n_steps = sum(1 for line in response.split("\n")
                  if any(s in line.lower() for s in
                         ["therefore", "so ", "thus", "hence", "step ", "from clue",
                          "because", "since", "which means", "this means"]))

    return correct, n_matched, f"matched {n_matched}/{len(patterns)} patterns, {n_steps} steps"


def run_finereason(model, tokenizer, device: str = "cuda",
                   max_problems: Optional[int] = None,
                   max_tokens: int = 400) -> Dict:
    """Run FineReason evaluation."""
    print("\n" + "=" * 60)
    print("FineReason: Deliberate Reasoning Benchmark")
    print("=" * 60)

    puzzles = load_finereason(max_problems=max_problems)
    results = []
    n_correct = 0
    total_steps = 0

    for i, puzzle in enumerate(puzzles):
        pid = puzzle.get("id", f"puzzle_{i}")
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\n  [{i+1}/{len(puzzles)}] {pid}")

        prompt = build_finereason_prompt(puzzle)
        response, telemetry = _generate_solution(
            model, tokenizer, prompt, device=device,
            max_tokens=max_tokens, temperature=0.0)

        correct, n_steps, technique = _evaluate_finereason(response, puzzle)
        if correct:
            n_correct += 1
        total_steps += n_steps

        status = "CORRECT" if correct else "INCORRECT"
        print(f"    {status} | {n_steps} reasoning steps | {technique}")

        results.append({
            "puzzle_id": pid,
            "correct": correct,
            "n_patterns_matched": n_matched,
            "n_reasoning_steps": n_steps,
            "gen_time_ms": telemetry["gen_time_ms"],
            "tokens_generated": telemetry["tokens_generated"],
        })

    pass_rate = n_correct / len(puzzles) if puzzles else 0.0
    avg_steps = total_steps / max(len(puzzles), 1)
    print(f"\n[FineReason] Correct: {n_correct}/{len(puzzles)} ({pass_rate:.1%})")
    print(f"  Avg reasoning steps: {avg_steps:.1f}")

    return {
        "benchmark": "finereason",
        "n_correct": n_correct,
        "n_total": len(puzzles),
        "pass_rate": pass_rate,
        "avg_reasoning_steps": avg_steps,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. ThinkBench — OOD Reasoning Robustness
# ═══════════════════════════════════════════════════════════════════════

def load_thinkbench(max_problems: Optional[int] = None) -> List[Dict]:
    """Load ThinkBench OOD reasoning dataset.

    ThinkBench (NeurIPS 2025) generates OOD variants of reasoning tasks:
      - AIME math problems (numeric answers 0-999)
      - GPQA Diamond science problems (multiple choice A/B/C/D)

    Each original question has 4 OOD variants:
      - scenario_paraphrased: same logic, different context
      - stresstest_perturbed: noise injection (repetitive true statements)
      - checklist_perturbed: sentence-level perturbations
      - textbugger_perturbed: character-level noise

    OOD Accuracy = 0.5 * (min(attack_accs) + scenario_acc)

    Hosted at: https://huggingface.co/datasets/jiuyinjiu/ThinkBench
               https://github.com/huangshulin123/ThinkBench
    """
    _ensure_dirs()
    cache_dir = os.path.join(BENCH_CACHE_DIR, "thinkbench")
    os.makedirs(cache_dir, exist_ok=True)

    # Try to load from HuggingFace (ThinkBench uses "train" split for eval)
    try:
        from datasets import load_dataset
        ds = load_dataset("jiuyinjiu/ThinkBench", split="train",
                          cache_dir=cache_dir)
        problems = [dict(row) for row in ds]
        if max_problems:
            problems = problems[:max_problems]
        print(f"[ThinkBench] Loaded {len(problems)} OOD samples from HuggingFace")
        return problems
    except Exception as e:
        print(f"[ThinkBench] HuggingFace load failed ({e})")

    # Try GitHub
    try:
        dest = os.path.join(cache_dir, "thinkbench_data.json")
        if not os.path.exists(dest):
            _download_file(
                "https://raw.githubusercontent.com/huangshulin123/ThinkBench/main/data/ood_samples.json",
                dest)
        with open(dest, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            problems = data
        else:
            problems = data.get("samples", data.get("data", []))
        if max_problems:
            problems = problems[:max_problems]
        print(f"[ThinkBench] Loaded {len(problems)} OOD samples from GitHub")
        return problems
    except Exception as e:
        print(f"[ThinkBench] GitHub load failed ({e}), using synthetic OOD problems")

    # Fallback: synthetic OOD reasoning problems
    problems = _generate_ood_reasoning_problems()
    if max_problems:
        problems = problems[:max_problems]
    print(f"[ThinkBench] Using {len(problems)} synthetic OOD problems")
    return problems


def _generate_ood_reasoning_problems() -> List[Dict]:
    """Generate synthetic OOD reasoning problems.

    These test reasoning robustness by using unusual framings of
    standard logic problems — the model must generalize beyond
    familiar patterns.
    """
    return [
        {
            "id": "ood_1",
            "question": (
                "On planet Zorg, there are three species: Glips, Blaps, and Snorps.\n"
                "Rules:\n"
                "- Glips always tell the truth\n"
                "- Blaps always lie\n"
                "- Snorps alternate: first statement true, second false, third true, etc.\n\n"
                "You meet three aliens: X, Y, Z (one of each species, unknown order).\n"
                "X says: 'I am a Glip.'\n"
                "Y says: 'X is a Blap.'\n"
                "Z says: 'Y is telling the truth.'\n"
                "What species is each alien?"
            ),
            "answer": "X=Snorp, Y=Blap, Z=Glip",
            "answer_patterns": ["snorp", "blap", "glip"],
        },
        {
            "id": "ood_2",
            "question": (
                "A library has books arranged in a special code.\n"
                "Each book's position is determined by:\n"
                "- The sum of its title's letter positions (A=1, B=2, ...)\n"
                "- Modulo the number of shelves\n"
                "- Plus the publication year modulo 100\n\n"
                "If 'CAT' (C=3, A=1, T=20, sum=24) was published in 2019,\n"
                "and there are 7 shelves, which shelf is it on?"
            ),
            "answer": "Shelf 3 (24 mod 7 = 3, 19 mod 100 = 19, 3 + 19 = 22, 22 mod 7 = 1... wait, shelf 1)",
            "answer_patterns": ["shelf", "1"],
        },
        {
            "id": "ood_3",
            "question": (
                "In a circular tournament, 5 players each play every other player exactly once.\n"
                "A win = 1 point, draw = 0.5, loss = 0.\n"
                "Player A scored 3.5 points.\n"
                "Player B scored 2 points.\n"
                "Player C scored 2.5 points.\n"
                "Player D scored 1 point.\n"
                "What did Player E score?"
            ),
            "answer": "1 point (total points = 10, 3.5+2+2.5+1=9, so E=1)",
            "answer_patterns": ["1", "one", "point"],
        },
        {
            "id": "ood_4",
            "question": (
                "A strange clock has 10 hours instead of 12.\n"
                "Each hour has 100 minutes.\n"
                "If the clock shows 3:50, and 50 real minutes pass, what does it show?\n"
                "(The clock runs at the same rate as a real clock.)"
            ),
            "answer": "4:40 (50 real minutes = 50 clock minutes, 3:50 + 50 = 4:40)",
            "answer_patterns": ["4:40", "4", "40"],
        },
        {
            "id": "ood_5",
            "question": (
                "You have a 5x5 grid. Place 5 queens so that:\n"
                "- No two queens attack each other\n"
                "- No queen is on the main diagonal\n"
                "- No queen is on the anti-diagonal\n"
                "How many valid arrangements exist?"
            ),
            "answer": "This is a constraint satisfaction problem — the answer requires systematic enumeration",
            "answer_patterns": ["arrangement", "solution", "valid"],
        },
        {
            "id": "ood_6",
            "question": (
                "Three switches outside a closed room control three bulbs inside.\n"
                "You can flip switches but cannot see inside.\n"
                "You may enter the room exactly once.\n"
                "How do you determine which switch controls which bulb?"
            ),
            "answer": "Turn on switch 1 for 5 min, turn it off, turn on switch 2, enter room. "
                      "Hot bulb = switch 1, lit bulb = switch 2, cold dark bulb = switch 3",
            "answer_patterns": ["hot", "switch 1", "switch 2", "switch 3", "heat"],
        },
        {
            "id": "ood_7",
            "question": (
                "A sequence follows this pattern: 2, 6, 12, 20, 30, ?\n"
                "What is the next number and what is the general formula?"
            ),
            "answer": "42. Formula: n*(n+1) for n=1,2,3... (2=1*2, 6=2*3, 12=3*4, etc.)",
            "answer_patterns": ["42", "n*(n+1)", "n(n+1)"],
        },
        {
            "id": "ood_8",
            "question": (
                "If you have a 3-liter jug and a 5-liter jug, how do you measure exactly 4 liters?\n"
                "You can fill jugs, empty them, or pour from one to the other."
            ),
            "answer": "Fill 5, pour to 3 (5 has 2, 3 has 3), empty 3, pour 2 from 5 to 3, "
                      "fill 5, pour from 5 to 3 (3 needs 1, so 5 has 4)",
            "answer_patterns": ["fill", "pour", "empty", "4"],
        },
    ]


def build_thinkbench_prompt(problem: Dict) -> str:
    """Build a prompt for a ThinkBench OOD problem.

    Handles both real ThinkBench format (original_question, scenario_paraphrased,
    perturbed variants) and synthetic fallback format.
    """
    prompt = "Solve this reasoning problem. Think carefully and show your work.\n\n"

    # Real ThinkBench format: use scenario_paraphrased (OOD variant)
    if "scenario_paraphrased" in problem:
        question = problem["scenario_paraphrased"]
    elif "original_question" in problem:
        question = problem["original_question"]
    elif "stresstest_perturbed" in problem:
        question = problem["stresstest_perturbed"]
    else:
        question = problem.get("question", str(problem))

    prompt += question
    prompt += "\n\n# Solution:\n"
    return prompt


def _evaluate_thinkbench(response: str, problem: Dict) -> Tuple[bool, str]:
    """Evaluate a ThinkBench response.

    Real format: answer is a string (numeric for AIME, letter for GPQA).
    Synthetic format: answer_patterns list.
    """
    answer = str(problem.get("answer", "")).strip()

    if answer:
        # Real ThinkBench: extract answer from response
        # Try \boxed{}, ANSWER:, or standalone number/letter
        boxed = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed:
            extracted = boxed.group(1).strip()
        else:
            ans_match = re.search(r"(?:ANSWER|answer|Answer)[:\s]+([A-D0-9-]+)", response)
            if ans_match:
                extracted = ans_match.group(1).strip()
            else:
                # Try last number in response
                numbers = re.findall(r"\b(\d+)\b", response)
                extracted = numbers[-1] if numbers else ""

        correct = extracted == answer
        return correct, f"extracted={extracted}, expected={answer}"

    # Synthetic fallback: pattern matching
    patterns = problem.get("answer_patterns", [])
    response_lower = response.lower()
    n_matched = sum(1 for p in patterns if p.lower() in response_lower)
    threshold = max(1, int(len(patterns) * 0.5))
    correct = n_matched >= threshold
    return correct, f"matched {n_matched}/{len(patterns)} patterns"


def run_thinkbench(model, tokenizer, device: str = "cuda",
                   max_problems: Optional[int] = None,
                   max_tokens: int = 400) -> Dict:
    """Run ThinkBench OOD reasoning evaluation."""
    print("\n" + "=" * 60)
    print("ThinkBench: OOD Reasoning Robustness Benchmark")
    print("=" * 60)

    problems = load_thinkbench(max_problems=max_problems)
    results = []
    n_correct = 0

    for i, problem in enumerate(problems):
        pid = str(problem.get("number_id", problem.get("id", f"ood_{i}")))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\n  [{i+1}/{len(problems)}] {pid}")

        prompt = build_thinkbench_prompt(problem)
        response, telemetry = _generate_solution(
            model, tokenizer, prompt, device=device,
            max_tokens=max_tokens, temperature=0.0)

        correct, detail = _evaluate_thinkbench(response, problem)
        if correct:
            n_correct += 1

        status = "CORRECT" if correct else "INCORRECT"
        print(f"    {status} | {detail} | {telemetry['tokens_generated']} tokens")

        results.append({
            "problem_id": pid,
            "correct": correct,
            "detail": detail,
            "gen_time_ms": telemetry["gen_time_ms"],
            "tokens_generated": telemetry["tokens_generated"],
        })

    pass_rate = n_correct / len(problems) if problems else 0.0
    print(f"\n[ThinkBench] Correct: {n_correct}/{len(problems)} ({pass_rate:.1%})")

    return {
        "benchmark": "thinkbench",
        "n_correct": n_correct,
        "n_total": len(problems),
        "pass_rate": pass_rate,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Unified Suite
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_RUNNERS = {
    "arc_agi2": run_arc_agi2,
    "neocoder": run_neocoder,
    "finereason": run_finereason,
    "thinkbench": run_thinkbench,
}


class ReasoningBenchmarkSuite:
    """Unified runner for all reasoning/creativity benchmarks.

    Usage:
        suite = ReasoningBenchmarkSuite(model, tokenizer, device="cuda")
        results = suite.run(benchmarks=["arc_agi2", "neocoder"], n_problems=20)
        suite.print_report(results)
    """

    def __init__(self, model, tokenizer, device: str = "cuda",
                 log_dir: str = BENCH_LOG_DIR):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, benchmarks: List[str] = None,
            n_problems: Optional[int] = None,
            **kwargs) -> Dict:
        """Run specified benchmarks.

        Args:
            benchmarks: list of benchmark names (default: all)
            n_problems: cap problems per benchmark
            **kwargs: passed to individual benchmark runners

        Returns:
            Dict with results from all benchmarks
        """
        if benchmarks is None:
            benchmarks = list(BENCHMARK_RUNNERS.keys())

        all_results = {}
        for bench_name in benchmarks:
            runner = BENCHMARK_RUNNERS.get(bench_name)
            if runner is None:
                print(f"  Unknown benchmark: {bench_name}")
                continue

            try:
                result = runner(self.model, self.tokenizer,
                                device=self.device,
                                max_tasks=n_problems if bench_name == "arc_agi2" else None,
                                max_problems=n_problems if bench_name != "arc_agi2" else None,
                                **kwargs)
                all_results[bench_name] = result
            except Exception as e:
                print(f"  ERROR in {bench_name}: {e}")
                all_results[bench_name] = {"error": str(e)}

        # Save combined results
        combined = {
            "timestamp": datetime.now().isoformat(),
            "n_problems": n_problems,
            "results": all_results,
        }
        results_path = self.log_dir / f"reasoning_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n[ReasoningBench] Results saved: {results_path}")

        return combined

    def print_report(self, combined: Dict):
        """Print a formatted report of all benchmark results."""
        results = combined.get("results", {})

        print(f"\n{'='*60}")
        print("Reasoning & Creativity Benchmark Report")
        print(f"{'='*60}")

        for bench_name, result in results.items():
            if "error" in result:
                print(f"\n  {bench_name:15s} ERROR: {result['error']}")
                continue

            if bench_name == "arc_agi2":
                print(f"\n  ARC-AGI-2 (Fluid Intelligence):")
                print(f"    Solved: {result['n_solved']}/{result['n_total']} "
                      f"({result['pass_rate']:.1%})")
            elif bench_name == "neocoder":
                print(f"\n  NeoCoder (Creative Code Generation):")
                print(f"    NeoGauge: {result['avg_neogauge']:.3f}")
                print(f"    Convergent: {result['avg_convergent']:.3f} (correctness)")
                print(f"    Divergent: {result['avg_divergent']:.3f} (creativity)")
            elif bench_name == "finereason":
                print(f"\n  FineReason (Deliberate Reasoning):")
                print(f"    Correct: {result['n_correct']}/{result['n_total']} "
                      f"({result['pass_rate']:.1%})")
                print(f"    Avg reasoning steps: {result['avg_reasoning_steps']:.1f}")
            elif bench_name == "thinkbench":
                print(f"\n  ThinkBench (OOD Reasoning):")
                print(f"    Correct: {result['n_correct']}/{result['n_total']} "
                      f"({result['pass_rate']:.1%})")

        print(f"\n{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reasoning & creativity benchmarks for ForgeLM")
    parser.add_argument("--benchmarks", type=str, default="all",
                        help="Comma-separated benchmarks: arc_agi2,neocoder,finereason,thinkbench (default: all)")
    parser.add_argument("--n-problems", type=int, default=10,
                        help="Max problems per benchmark")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max generation tokens")
    parser.add_argument("--checkpoint", type=str,
                        default="research/checkpoints/forgelm_v2.safetensors",
                        help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="forgelm_v2",
                        help="Model config name")
    args = parser.parse_args()

    print("=" * 60)
    print("Reasoning & Creativity Benchmark Suite")
    print("=" * 60)

    # Load model
    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer

    print(f"\n[1] Loading model ({args.config})...")
    cfg = get_config(args.config, device="cuda")
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.checkpoint)
    model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

    # Determine benchmarks
    if args.benchmarks == "all":
        benchmarks = list(BENCHMARK_RUNNERS.keys())
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    print(f"\n[2] Running benchmarks: {benchmarks}")

    suite = ReasoningBenchmarkSuite(model, tokenizer, device="cuda")
    results = suite.run(benchmarks=benchmarks, n_problems=args.n_problems)
    suite.print_report(results)


if __name__ == "__main__":
    main()
