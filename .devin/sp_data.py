"""Self-play data manager: inspect and wipe trajectories, checkpoints, and finetune data.

Usage:
    python -m .devin.sp_data inspect [--min-reward N] [--check-correctness]
    python -m .devin.sp_data wipe [--db] [--ckpt] [--finetune] [--all]
    python -m .devin.sp_data stats

Examples:
    # Show all trajectories with reward >= 0.99 and check for false wins
    python .devin/sp_data.py inspect --min-reward 0.99 --check-correctness

    # Wipe everything (DBs + sp checkpoints + finetune data)
    python .devin/sp_data.py wipe --all

    # Just show stats
    python .devin/sp_data.py stats
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DB_PATHS = [
    ROOT / "research/data/discovery/tool_infinite.sqlite3",
    ROOT / "research/data/discovery/discovery.sqlite3",
]
CKPT_DIR = ROOT / "research/checkpoints"
FINETUNE_DIR = ROOT / "research/data/finetune"

TABLES_TO_WIPE = [
    "tool_trajectories", "distill_runs", "epochs", "events",
    "discoveries", "theories", "research", "scripts",
    "thoughts", "sessions",
]

# ── Known math answers for correctness checking ──────────────────────────
MATH_CHECKS = [
    # (pattern_in_task, expected_answer_substring, description)
    (r"2\^15|2\*\*15", "32768", "2^15=32768"),
    (r"factorial of 8|8!", "40320", "8!=40320"),
    (r"factorial of 10|10!", "3628800", "10!=3628800"),
    (r"factorial of 12|12!", "479001600", "12!=479001600"),
    (r"factorial of 15|15!", "1307674368000", "15!=1307674368000"),
    (r"sum of all even.*1 to 100|sum of even.*1.*100", "2550", "sum_evens_1_100=2550"),
    (r"area of a circle.*radius 7|circle.*radius 7", "153", "area_circle_r7~153.94"),
    (r"gcd.*48.*36|gcd.*36.*48", "12", "GCD(48,36)=12"),
    (r"gcd.*1071.*462|gcd.*462.*1071", "21", "GCD(1071,462)=21"),
    (r"fibonacci.*10\b", "55", "fib(10)=55"),
    (r"fibonacci.*20\b", "6765", "fib(20)=6765"),
    (r"trailing zeros.*100\b|100.*trailing zeros", "24", "100!_trailing_zeros=24"),
    (r"2\^20|2\*\*20", "1048576", "2^20=1048576"),
    (r"15 \* 37|15\*37", "597", "15*37+42=597"),
    (r"compound interest.*1000.*5%.*3 year", "1157", "compound_interest~1157.63"),
    (r"percentage increase.*1\.5.*2\.0|1\.5 to 2\.0 degrees", "33.3", "pct_increase_1.5to2.0=33.3%"),
]

TRIVIAL_PATTERNS = [
    "what is 2+", "what is 15 *", "what is 2^20", "what is 2^10",
    "what is the square root of 144",
]


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r'\d+\.?\d*', text or "")


def _check_correctness(task: str, answer: str, tool_calls: list[dict]) -> list[str]:
    """Check if a trajectory is a false win. Returns list of issue strings."""
    issues = []
    task_lower = task.lower()
    answer_lower = (answer or "").lower()
    answer_nums = _extract_numbers(answer)

    # Extract script output numbers (ground truth)
    script_nums = []
    for tc in tool_calls:
        if tc.get("name") in ("calculate", "run_script") and tc.get("success"):
            result = tc.get("result", {})
            if isinstance(result, dict):
                stdout = result.get("stdout", "").strip()
                script_nums.extend(_extract_numbers(stdout))

    # Check against known math answers
    for pattern, expected, desc in MATH_CHECKS:
        if re.search(pattern, task_lower):
            if expected not in answer_lower:
                issues.append(f"WRONG ANSWER ({desc}, got {answer_nums[:3]})")
            break  # only one math check applies

    # Check answer vs script output mismatch
    if script_nums and answer_nums:
        try:
            s_val = float(script_nums[-1])
            matched = any(abs(float(a) - s_val) < 0.01 * max(abs(s_val), 1)
                          for a in answer_nums)
            if not matched and "WRONG ANSWER" not in " ".join(issues):
                issues.append(f"ANSWER/SCRIPT MISMATCH (answer={answer_nums[:2]} script={script_nums[-1]})")
        except (ValueError, OverflowError):
            pass

    # Check trivial task
    for pat in TRIVIAL_PATTERNS:
        if pat in task_lower:
            issues.append("TRIVIAL TASK (free win)")
            break

    # Check no tools
    if not tool_calls:
        issues.append("NO TOOLS USED")

    return issues


def _classify_source(task: str) -> str:
    """Classify as MODEL-GEN or CURRICULUM based on template patterns."""
    templates = ["Search for '", "Research '", "Investigate '", "Explore '",
                 "Study '", "Compare 3 approaches to '"]
    if any(task.startswith(t) for t in templates):
        return "MODEL-GEN"
    return "CURRICULUM"


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_stats(args):
    """Show summary stats."""
    db = sqlite3.connect(str(DB_PATHS[0]))
    db.row_factory = sqlite3.Row

    total = db.execute("SELECT COUNT(*) FROM tool_trajectories").fetchone()[0]
    if total == 0:
        print("No trajectories in DB.")
        return

    saved = db.execute("SELECT COUNT(*) FROM tool_trajectories WHERE reward >= 0.25").fetchone()[0]
    avg_reward = db.execute("SELECT AVG(reward) FROM tool_trajectories").fetchone()[0] or 0
    perfect = db.execute("SELECT COUNT(*) FROM tool_trajectories WHERE reward >= 0.99").fetchone()[0]
    avg_tools = db.execute("SELECT AVG(n_tool_calls) FROM tool_trajectories").fetchone()[0] or 0

    print(f"  Total trajectories: {total}")
    print(f"  Saved (reward >= 0.25): {saved}")
    print(f"  Perfect (reward >= 0.99): {perfect}")
    print(f"  Avg reward: {avg_reward:.3f}")
    print(f"  Avg tool calls: {avg_tools:.1f}")

    # Checkpoints
    sp_ckpts = list(CKPT_DIR.glob("*.sp*.safetensors"))
    print(f"\n  Self-play checkpoints: {len(sp_ckpts)}")
    for c in sp_ckpts:
        size_gb = c.stat().st_size / 1e9
        print(f"    {c.name} ({size_gb:.2f} GB)")

    # Finetune data
    ft_files = list(FINETUNE_DIR.glob("self_play_*.jsonl")) if FINETUNE_DIR.exists() else []
    print(f"\n  Finetune data files: {len(ft_files)}")
    for f in ft_files:
        lines = sum(1 for _ in open(f))
        print(f"    {f.name} ({lines} examples)")

    db.close()


def cmd_inspect(args):
    """Inspect trajectories with optional filtering and correctness checking."""
    db = sqlite3.connect(str(DB_PATHS[0]))
    db.row_factory = sqlite3.Row

    query = "SELECT id, task, reward, n_tool_calls, n_successful, format_ok, final_answer, tool_calls FROM tool_trajectories"
    params = []
    if args.min_reward is not None:
        query += " WHERE reward >= ?"
        params.append(args.min_reward)
    query += " ORDER BY id"

    rows = list(db.execute(query, params))

    if not rows:
        print(f"No trajectories found (min_reward={args.min_reward}).")
        return

    print(f"{'='*90}")
    print(f"  {len(rows)} trajectories" + (f" (reward >= {args.min_reward})" if args.min_reward else ""))
    print(f"{'='*90}\n")

    false_wins = 0
    for r in rows:
        task = r["task"]
        fa = r["final_answer"] or ""
        source = _classify_source(task)

        try:
            tool_calls = json.loads(r["tool_calls"] or "[]")
        except (json.JSONDecodeError, TypeError):
            tool_calls = []

        issues = []
        if args.check_correctness:
            issues = _check_correctness(task, fa, tool_calls)
            if not r["format_ok"]:
                issues.append("FORMAT FAIL")

        if issues:
            false_wins += 1

        status = " | ".join(issues) if issues else "OK"
        print(f"  #{r['id']:2d} [{source:9s}] r={r['reward']:.2f} tools={r['n_tool_calls']} | {status}")
        print(f"       task: {task[:100]}")
        if fa:
            print(f"       answer: {fa[:150]}")
        if args.check_correctness and issues:
            # Show script output for false wins
            for tc in tool_calls:
                if tc.get("name") in ("calculate", "run_script") and tc.get("success"):
                    result = tc.get("result", {})
                    if isinstance(result, dict):
                        stdout = result.get("stdout", "")[:100]
                        if stdout:
                            print(f"       script_out: {stdout}")
        print()

    print(f"{'='*90}")
    print(f"  Summary: {len(rows)} trajectories, {false_wins} false wins")
    if args.check_correctness and false_wins > 0:
        print(f"  ** {false_wins} trajectories have reward >= {args.min_reward or 0} but failed correctness checks **")
    print(f"{'='*90}")

    db.close()


def cmd_wipe(args):
    """Wipe DBs, checkpoints, and/or finetune data."""
    wiped_total = 0

    if args.all:
        args.db = args.ckpt = args.finetune = True

    if not any([args.db, args.ckpt, args.finetune]):
        print("Nothing to wipe. Use --db, --ckpt, --finetune, or --all.")
        return

    # Wipe DBs
    if args.db:
        for db_path in DB_PATHS:
            if not db_path.exists():
                continue
            db = sqlite3.connect(str(db_path))
            existing = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            wiped = 0
            for t in TABLES_TO_WIPE:
                if t in existing:
                    cnt = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    if cnt > 0:
                        db.execute(f"DELETE FROM {t}")
                        wiped += cnt
            db.execute("DELETE FROM sqlite_sequence")
            db.commit()
            db.close()
            print(f"  [DB] {db_path.name}: wiped {wiped} rows")
            wiped_total += wiped

    # Wipe self-play checkpoints (keep base + BSP)
    if args.ckpt:
        sp_ckpts = list(CKPT_DIR.glob("*.sp*.safetensors"))
        sp_ckpts += list(CKPT_DIR.glob("*.sp*.meta.json"))
        sp_ckpts += list(CKPT_DIR.glob("*.sp*.train.pt"))
        for c in sp_ckpts:
            c.unlink()
            print(f"  [CKPT] Deleted: {c.name}")

    # Wipe finetune data
    if args.finetune:
        if FINETUNE_DIR.exists():
            for f in FINETUNE_DIR.glob("self_play_*.jsonl"):
                f.unlink()
                print(f"  [FT] Deleted: {f.name}")

    print(f"\nDone. Wiped {wiped_total} DB rows total.")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-play data manager: inspect and wipe trajectories, checkpoints, and finetune data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    sub.add_parser("stats", help="Show summary stats")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect trajectories")
    p_inspect.add_argument("--min-reward", type=float, default=None,
                           help="Only show trajectories with reward >= N")
    p_inspect.add_argument("--check-correctness", action="store_true",
                           help="Check for false wins (wrong answers, trivial tasks)")

    # wipe
    p_wipe = sub.add_parser("wipe", help="Wipe DBs, checkpoints, and/or finetune data")
    p_wipe.add_argument("--db", action="store_true", help="Wipe self-play DBs")
    p_wipe.add_argument("--ckpt", action="store_true", help="Wipe self-play checkpoints (sp*)")
    p_wipe.add_argument("--finetune", action="store_true", help="Wipe finetune JSONL files")
    p_wipe.add_argument("--all", action="store_true", help="Wipe everything")

    args = parser.parse_args()

    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "wipe":
        cmd_wipe(args)


if __name__ == "__main__":
    main()
