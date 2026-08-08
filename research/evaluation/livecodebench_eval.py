"""LiveCodeBench evaluation for ForgeLM — contamination-free benchmarking.

LiveCodeBench (ICLR 2025) collects coding problems from LeetCode, AtCoder, and
CodeForces, tagged with release dates. By filtering to problems released AFTER
Qwen2.5-Coder's training cutoff (~Sept 2024), we get a contamination-free
evaluation that measures true generalization, not memorization.

This module:
  1. Downloads LiveCodeBench code_generation_lite from HuggingFace
  2. Filters to problems released after the cutoff date
  3. Runs ForgeLM on each problem (using our own inference code)
  4. Evaluates solutions against the provided test cases
  5. Reports per-platform and per-difficulty breakdowns to identify sore spots

The evaluation uses the custom_evaluator format from LiveCodeBench:
  [{"question_id": "...", "code_list": ["solution1", ...]}, ...]

Usage:
    from research.evaluation.livecodebench_eval import LiveCodeBenchEvaluator
    evaluator = LiveCodeBenchEvaluator(model, tokenizer, device="cuda")
    results = evaluator.run(start_date="2024-09-01", n_problems=50)
    evaluator.print_sore_spots(results)

    # Or via CLI:
    # python -m research.livecodebench_eval --start-date 2024-09-01 --n-problems 50
"""
import os
import sys
import json
import time
import re
import subprocess
import tempfile
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── LiveCodeBench data loading ──────────────────────────────────────

LCB_DATASET_ID = "livecodebench/code_generation_lite"
LCB_CACHE_DIR = "D:/windsurf/ForgeAI/.devin/hf_cache"
# Qwen2.5-Coder was released Sept 2024 with training data "before 2024"
# Using Sept 1, 2024 as a conservative cutoff
DEFAULT_CUTOFF = "2024-09-01"


def load_livecodebench(start_date: str = DEFAULT_CUTOFF,
                       max_problems: Optional[int] = None,
                       cache_dir: str = LCB_CACHE_DIR,
                       release_version: str = "v6") -> List[Dict]:
    """Load LiveCodeBench problems released after the cutoff date.

    Downloads JSONL files directly from HuggingFace (bypasses the deprecated
    dataset script mechanism). Uses the latest release (v6 = up to Apr 2025).

    Args:
        start_date: only include problems released on/after this date (YYYY-MM-DD)
        max_problems: cap the number of problems (None = all)
        cache_dir: HuggingFace cache directory
        release_version: which release file to use (v1-v6, default v6 = latest)

    Returns:
        List of problem dicts with keys: question_id, question_title, question_content,
        difficulty, raw_tags, contest_date, public_test_cases, private_test_cases
    """
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    from huggingface_hub import hf_hub_download

    # Map release version to JSONL filename
    file_map = {"v1": "test.jsonl", "v2": "test2.jsonl", "v3": "test3.jsonl",
                "v4": "test4.jsonl", "v5": "test5.jsonl", "v6": "test6.jsonl"}
    filename = file_map.get(release_version, "test6.jsonl")

    print(f"[LiveCodeBench] Downloading {filename} (release {release_version})...")
    filepath = hf_hub_download(
        repo_id=LCB_DATASET_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"[LiveCodeBench] Downloaded: {filepath}")

    # Load and filter JSONL
    cutoff = datetime.strptime(start_date, "%Y-%m-%d")
    filtered = []
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            date_str = row.get("contest_date", "")
            if not date_str:
                continue
            try:
                row_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            if row_date >= cutoff:
                filtered.append(row)

    print(f"[LiveCodeBench] Total problems in file: {total}")
    print(f"[LiveCodeBench] Problems after {start_date}: {len(filtered)}")

    if max_problems is not None and len(filtered) > max_problems:
        filtered = filtered[:max_problems]
        print(f"[LiveCodeBench] Capped to {max_problems} problems")

    return filtered


# ─── Test case execution ──────────────────────────────────────────────

def extract_code_blocks(text: str) -> str:
    """Extract Python code from markdown fences or raw text."""
    # Try markdown code blocks first
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Return the last (most complete) code block
        return matches[-1].strip()
    return text.strip()


def run_test_cases(code: str, test_cases: List[Dict],
                   timeout_s: float = 10.0) -> Tuple[bool, str, int]:
    """Run generated code against test cases.

    LCB problems use stdin/stdout format: the solution reads from stdin,
    processes, and prints to stdout. Test cases specify input/output strings.

    Args:
        code: the generated Python solution
        test_cases: list of {"input": "...", "output": "...", "testtype": "stdin"} dicts
        timeout_s: execution timeout per test

    Returns:
        (all_passed, error_message, n_passed)
    """
    # Extract code from markdown if needed
    code = extract_code_blocks(code)

    # Write solution to temp file
    temp_dir = "D:/windsurf/ForgeAI/.devin/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                      delete=False, encoding='utf-8',
                                      dir=temp_dir) as f:
        f.write(code)
        code_path = f.name

    n_passed = 0
    n_total = len(test_cases)
    error_msg = ""

    for i, tc in enumerate(test_cases):
        test_input = tc.get("input", "")
        expected_output = tc.get("output", "").strip()

        try:
            proc = subprocess.run(
                [sys.executable, code_path],
                input=test_input,
                capture_output=True, text=True,
                timeout=timeout_s,
                cwd=temp_dir,
                encoding='utf-8', errors='replace',
            )
            actual = proc.stdout.strip()
            if actual == expected_output:
                n_passed += 1
            else:
                if not error_msg:
                    error_msg = (f"Test {i}: expected {expected_output[:80]!r}, "
                                 f"got {actual[:80]!r}")
        except subprocess.TimeoutExpired:
            if not error_msg:
                error_msg = f"Test {i}: timed out after {timeout_s}s"
        except Exception as e:
            if not error_msg:
                error_msg = f"Test {i}: {e}"

    # Cleanup
    try:
        os.unlink(code_path)
    except OSError:
        pass

    all_passed = (n_passed == n_total) and n_total > 0
    return all_passed, error_msg, n_passed


# ─── ForgeLM inference for LCB ────────────────────────────────────────

def build_lcb_prompt(problem: Dict) -> str:
    """Build a prompt for ForgeLM from a LiveCodeBench problem.

    Uses a simple format that works with base/instruct models.
    """
    content = problem.get("question_content", "")
    title = problem.get("question_title", "")

    prompt = (
        f"# Problem: {title}\n"
        f"{content}\n\n"
        f"# Write a Python solution. Read input from stdin, print output to stdout.\n"
        f"# Solution:\n"
    )
    return prompt


def generate_solution(model, tokenizer, prompt: str, device: str = "cuda",
                      max_tokens: int = 512, temperature: float = 0.0) -> Tuple[str, Dict]:
    """Generate a solution using ForgeLM.

    Returns (generated_code, telemetry).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    t0 = time.time()
    log_probs = []
    newline_id = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_id[0] if newline_id else 198

    with torch.inference_mode():
        cur_ids = input_ids
        newline_count = 0
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

            if next_token.item() == newline_id:
                newline_count += 1
                if newline_count >= 4 and step > 10:
                    break
            else:
                newline_count = 0

    gen_time_ms = (time.time() - t0) * 1000
    tokens_generated = cur_ids.shape[1] - input_ids.shape[1]

    generated_text = tokenizer.decode(cur_ids[0, input_ids.shape[1]:],
                                       skip_special_tokens=True)

    telemetry = {
        "gen_time_ms": gen_time_ms,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_generated / (gen_time_ms / 1000) if gen_time_ms > 0 else 0,
        "mean_logprob": sum(log_probs) / max(len(log_probs), 1),
    }

    return generated_text, telemetry


# ─── Evaluator ────────────────────────────────────────────────────────

class LiveCodeBenchEvaluator:
    """Run LiveCodeBench evaluation on ForgeLM.

    Downloads problems from HuggingFace, filters to post-cutoff, runs the model,
    evaluates against test cases, and reports per-category sore spots.
    """

    def __init__(self, model, tokenizer, device: str = "cuda",
                 log_dir: str = "D:/windsurf/ForgeAI/research/data/lcb_eval"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, start_date: str = DEFAULT_CUTOFF,
            n_problems: Optional[int] = None,
            max_tokens: int = 512,
            temperature: float = 0.0,
            timeout_s: float = 10.0) -> Dict:
        """Run LiveCodeBench evaluation.

        Args:
            start_date: only problems released on/after this date
            n_problems: cap number of problems
            max_tokens: max generation tokens per problem
            temperature: 0 for greedy, >0 for sampling
            timeout_s: test execution timeout

        Returns:
            Results dict with per-problem and per-category breakdowns
        """
        problems = load_livecodebench(start_date=start_date, max_problems=n_problems)
        if not problems:
            print("[LiveCodeBench] No problems found after cutoff date")
            return {"total": 0, "passed": 0, "pass_rate": 0.0}

        print(f"\n[LiveCodeBench] Evaluating {len(problems)} problems...")
        print(f"  Cutoff: {start_date}")
        print(f"  Max tokens: {max_tokens}, Temperature: {temperature}")

        results = []
        n_passed = 0

        for i, problem in enumerate(problems):
            qid = problem.get("question_id", f"q{i}")
            title = problem.get("question_title", "unknown")[:60]
            difficulty = problem.get("difficulty", "Unknown")
            platform = self._detect_platform(problem)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"\n  [{i+1}/{len(problems)}] {title} ({difficulty}, {platform})")

            # Build prompt and generate
            prompt = build_lcb_prompt(problem)
            code, telemetry = generate_solution(
                self.model, self.tokenizer, prompt,
                device=self.device, max_tokens=max_tokens, temperature=temperature)

            # Parse test cases
            test_cases = self._parse_test_cases(problem)

            # Evaluate
            if test_cases:
                all_pass, error, n_tests_passed = run_test_cases(
                    code, test_cases, timeout_s=timeout_s)
            else:
                all_pass = False
                error = "no test cases"
                n_tests_passed = 0

            if all_pass:
                n_passed += 1
                status = "PASS"
            else:
                status = "FAIL"

            if (i + 1) % 10 == 0 or i == 0 or all_pass:
                print(f"    {status} | {n_tests_passed}/{len(test_cases)} tests | "
                      f"{telemetry['tokens_generated']} tokens | "
                      f"{telemetry['gen_time_ms']:.0f}ms")
                if not all_pass and error:
                    print(f"    Error: {error[:80]}")

            results.append({
                "question_id": qid,
                "title": title,
                "difficulty": difficulty,
                "platform": platform,
                "passed": all_pass,
                "n_tests_passed": n_tests_passed,
                "n_tests_total": len(test_cases),
                "error": error,
                "gen_time_ms": telemetry["gen_time_ms"],
                "tokens_generated": telemetry["tokens_generated"],
                "code": code[:500],  # truncated for logging
            })

        pass_rate = n_passed / len(problems) if problems else 0.0
        print(f"\n[LiveCodeBench] Results: {n_passed}/{len(problems)} passed "
              f"({pass_rate:.1%})")

        # Build summary with sore spot analysis
        summary = self._build_summary(results, start_date)
        summary["pass_rate"] = pass_rate
        summary["n_passed"] = n_passed
        summary["n_total"] = len(problems)
        summary["results"] = results
        summary["start_date"] = start_date
        summary["timestamp"] = datetime.now().isoformat()

        # Save results
        results_path = self.log_dir / f"lcb_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[LiveCodeBench] Results saved: {results_path}")

        return summary

    def _detect_platform(self, problem: Dict) -> str:
        """Detect which platform a problem came from."""
        # LCB data has a 'platform' field directly
        platform = problem.get("platform", "")
        if platform:
            return str(platform).capitalize()
        # Fallback: detect from tags/qid
        tags = problem.get("raw_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        qid = str(problem.get("question_id", "")).lower()
        if "leetcode" in str(tags).lower() or "leetcode" in qid:
            return "LeetCode"
        if "atcoder" in str(tags).lower() or "atcoder" in qid:
            return "AtCoder"
        if "codeforces" in str(tags).lower() or "codeforces" in qid:
            return "CodeForces"
        return "Unknown"

    def _parse_test_cases(self, problem: Dict) -> List[Dict]:
        """Parse test cases from a LiveCodeBench problem.

        LCB stores test cases as JSON strings in public_test_cases / private_test_cases.
        Format: [{"input": "...", "output": "..."}, ...]
        """
        test_cases = []

        # Try public test cases first
        for field in ["public_test_cases", "private_test_cases"]:
            raw = problem.get(field)
            if not raw:
                continue
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            if isinstance(raw, list):
                for tc in raw:
                    if isinstance(tc, dict) and "input" in tc and "output" in tc:
                        test_cases.append({"input": tc["input"], "output": tc["output"]})

        return test_cases

    def _build_summary(self, results: List[Dict], start_date: str) -> Dict:
        """Build per-category summary for sore spot identification."""
        categories = {}

        for r in results:
            # By platform
            platform = r["platform"]
            cat_key = f"platform_{platform}"
            if cat_key not in categories:
                categories[cat_key] = {"total": 0, "passed": 0}
            categories[cat_key]["total"] += 1
            if r["passed"]:
                categories[cat_key]["passed"] += 1

            # By difficulty
            difficulty = r["difficulty"]
            cat_key = f"difficulty_{difficulty}"
            if cat_key not in categories:
                categories[cat_key] = {"total": 0, "passed": 0}
            categories[cat_key]["total"] += 1
            if r["passed"]:
                categories[cat_key]["passed"] += 1

        # Calculate rates
        for key, val in categories.items():
            val["pass_rate"] = val["passed"] / max(val["total"], 1)

        return {"categories": categories}

    def print_sore_spots(self, summary: Dict):
        """Print per-category breakdown to identify sore spots."""
        categories = summary.get("categories", {})
        if not categories:
            print("[LiveCodeBench] No category data available")
            return

        print(f"\n{'='*60}")
        print("LiveCodeBench Sore Spot Analysis")
        print(f"{'='*60}")
        print(f"  Overall: {summary.get('n_passed', 0)}/{summary.get('n_total', 0)} "
              f"({summary.get('pass_rate', 0):.1%})")

        # Platform breakdown
        print(f"\n  By Platform:")
        platforms = {k: v for k, v in categories.items() if k.startswith("platform_")}
        for key in sorted(platforms.keys()):
            v = platforms[key]
            name = key.replace("platform_", "")
            bar = "#" * int(v["pass_rate"] * 20)
            print(f"    {name:12s} {v['passed']:3d}/{v['total']:3d} "
                  f"({v['pass_rate']:5.1%}) {bar}")

        # Difficulty breakdown
        print(f"\n  By Difficulty:")
        diffs = {k: v for k, v in categories.items() if k.startswith("difficulty_")}
        for key in sorted(diffs.keys()):
            v = diffs[key]
            name = key.replace("difficulty_", "")
            bar = "#" * int(v["pass_rate"] * 20)
            print(f"    {name:12s} {v['passed']:3d}/{v['total']:3d} "
                  f"({v['pass_rate']:5.1%}) {bar}")

        # Sore spots (lowest pass rate categories with >=3 problems)
        print(f"\n  Sore Spots (lowest pass rate, >=3 problems):")
        sore = [(k, v) for k, v in categories.items() if v["total"] >= 3]
        sore.sort(key=lambda x: x[1]["pass_rate"])
        for key, v in sore[:5]:
            print(f"    {key:25s} {v['pass_rate']:.1%} ({v['passed']}/{v['total']})")

        print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LiveCodeBench evaluation for ForgeLM")
    parser.add_argument("--start-date", type=str, default=DEFAULT_CUTOFF,
                        help="Only problems released on/after this date (YYYY-MM-DD)")
    parser.add_argument("--n-problems", type=int, default=None,
                        help="Max number of problems to evaluate")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max generation tokens per problem")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (0=greedy)")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Test execution timeout in seconds")
    parser.add_argument("--checkpoint", type=str,
                        default="research/checkpoints/forgelm_v2.safetensors",
                        help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="forgelm_v2",
                        help="Model config name")
    args = parser.parse_args()

    print("=" * 60)
    print("LiveCodeBench Evaluation for ForgeLM")
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

    # Run evaluation
    print(f"\n[2] Running LiveCodeBench evaluation...")
    evaluator = LiveCodeBenchEvaluator(model, tokenizer, device="cuda")
    results = evaluator.run(
        start_date=args.start_date,
        n_problems=args.n_problems,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
    )

    # Print sore spot analysis
    evaluator.print_sore_spots(results)


if __name__ == "__main__":
    main()
