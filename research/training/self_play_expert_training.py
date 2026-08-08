"""Self-Play Expert Training — fine-tune AirMoE experts via recursive self-play.

Pipeline per topic:
  1. Load V2 base model (Qwen quality, always in VRAM)
  2. Route topic-specific prompts via expert router
  3. Generate code with recursive self-play (generate → execute → fix → retry)
  4. Score successful solutions (quality + correctness)
  5. Collect successful (prompt, solution, reasoning) pairs
  6. Fine-tune the topic expert via closed-form fact injection
  7. Save updated expert back to v4 library (SVD + int4 compressed)

The base model stays frozen. Only the topic experts are updated.
Each expert learns from its own successes — self-play improvement.

Usage:
    python -u -m research.self_play_expert_training --topics python_algorithms,math_arithmetic
    python -u -m research.self_play_expert_training --all-topics --rounds 50
"""
import os
import sys
import time
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.config import get_config
from research.model_loader import ModelLoader
from research.self_play.recursive_self_play import RecursiveSelfPlay
from research.moe.airmoe_infinite import InfiniteAirMoE
from research.evaluation.goal_tasks import GoalTaskGenerator, GoalTask
from research.self_play.infinite_curriculum import InfiniteCurriculum, ProposedTask
from research.runtime.vram_manager import VRAMManager
from research.keys.softpick_key import apply_softpick
from research.keys.per_query_temp_key import apply_per_query_temp
from research.keys.norm_gated_mod_key import apply_norm_gated_mod
from transformers import AutoTokenizer

# Goal-oriented task generator (replaces function-stub prompt library)
# Generates GoalTasks with adaptive difficulty and I/O verification pairs
_goal_gen = GoalTaskGenerator()

# Domain -> list of archetype names (for topic-based training)
DOMAIN_ARCHETYPES = {
    "python_algorithms": ["fibonacci", "sum_list", "sort_list", "collatz_steps",
                          "max_element", "linear_search"],
    "math_arithmetic": ["factorial", "is_prime", "gcd", "power", "digit_sum"],
    "python_strings": ["reverse_string", "count_vowels", "is_palindrome", "word_count"],
    "python_general": ["fibonacci", "factorial", "is_prime", "gcd", "reverse_string",
                       "sum_list", "sort_list", "count_vowels", "is_palindrome",
                       "power", "digit_sum", "word_count", "max_element", "linear_search"],
}


class ExpertSelfPlayTrainer:
    """Train AirMoE experts via recursive self-play.

    For each topic:
      1. Load the topic expert into the model
      2. Run recursive self-play on topic prompts
      3. Collect successful (prompt, solution) pairs
      4. Fine-tune the expert weights via gradient updates
      5. Save the improved expert back to disk
    """

    def __init__(self, model, tokenizer, airmoe: InfiniteAirMoE,
                 log_dir: str = "D:/windsurf/ForgeAI/research/data/expert_training",
                 device: str = "cuda",
                 learning_rate: float = 1e-4,
                 max_rounds: int = 3,
                 temperature: float = 0.7,
                 top_k: int = 40,
                 top_p: float = 0.9,
                 lcb_n_problems: int = 0,
                 reasoning_n_problems: int = 0,
                 reasoning_benchmarks: list = None,
                 vram_manager: 'VRAMManager' = None,
                 max_gen_tokens: int = 120,
                 use_curriculum: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.airmoe = airmoe
        self.device = device
        self.lr = learning_rate
        self.max_rounds = max_rounds
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.vram = vram_manager  # VRAM manager for dynamic memory management
        # LiveCodeBench contamination-free benchmark (0 = disabled)
        # Set to ~20-50 for per-epoch eval; problems are post-Sept 2024 (unseen by Qwen)
        self.lcb_n_problems = lcb_n_problems
        # Reasoning & creativity benchmarks (0 = disabled)
        # ARC-AGI-2, NeoCoder, FineReason, ThinkBench
        self.reasoning_n_problems = reasoning_n_problems
        self.reasoning_benchmarks = reasoning_benchmarks or [
            "arc_agi2", "neocoder", "finereason", "thinkbench"]

        # Self-play engine — temperature>0 for solution diversity across epochs
        # (prevents greedy decoding from producing identical solutions every epoch,
        # which causes overfitting to a single solution per prompt)
        self.engine = RecursiveSelfPlay(
            model, tokenizer,
            log_dir=str(self.log_dir / "self_play"),
            device=device,
            max_gen_tokens=max_gen_tokens,
            max_rounds=max_rounds,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            vram_manager=vram_manager,
        )

        # Calibrate conformal sampler on held-out prompts (NOVEL key)
        # Per-query temperature: confident prompts → precise, uncertain → creative
        if temperature > 0.0:
            self.engine.calibrate_conformal(alpha=0.1)

        # Infinite curriculum engine (AZR-style: model proposes tasks, executor verifies)
        # When enabled, replaces fixed GoalTaskGenerator with infinite adaptive tasks
        self.use_curriculum = use_curriculum
        self.curriculum = None
        if use_curriculum:
            self.curriculum = InfiniteCurriculum(
                model, tokenizer, device=device,
                max_gen_tokens=max_gen_tokens,
                temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
            self.curriculum.load_queue()
            print(f"  [Curriculum] Enabled — {self.curriculum.queue_size()} tasks in queue")

    def train_topic(self, topic: str, n_tasks: int = 50,
                    n_epochs: int = 3,
                    patience: int = 2,
                    val_ratio: float = 0.2,
                    k_samples: int = 1) -> Dict:
        """Train one topic expert via goal-oriented self-play.

        Generates GoalTasks with adaptive difficulty, runs recursive self-play
        with multi-dimensional scoring (minimalism, efficiency, diversity),
        and fine-tunes the expert on accepted solutions.

        Args:
            topic: topic/domain name (e.g., "python_algorithms")
            n_tasks: number of goal tasks to generate per epoch
            n_epochs: max epochs
            patience: stop if val success rate doesn't improve for N epochs
            val_ratio: fraction of tasks held out for validation
            k_samples: independent samples for self-consistency (VERSE)

        Returns:
            Training stats dict
        """
        import random
        rng = random.Random(42)

        print(f"\n{'='*60}")
        print(f"Training Expert: {topic} (goal-oriented)")
        print(f"{'='*60}")

        # Map topic to archetypes
        archetypes = DOMAIN_ARCHETYPES.get(topic, list(GoalTaskGenerator.__init__.__code__.co_consts) if False else
                                            ["fibonacci", "factorial", "is_prime", "gcd",
                                             "reverse_string", "sum_list", "sort_list",
                                             "count_vowels", "is_palindrome", "power"])
        print(f"  Archetypes: {archetypes}")

        # ── Task generation: curriculum or fixed archetypes ──────
        # When curriculum is enabled, the model proposes tasks with unit tests
        # (AZR-style). Otherwise, use the fixed GoalTaskGenerator.
        all_tasks = []

        if self.use_curriculum and self.curriculum is not None:
            # Map topic to curriculum domain
            curr_domain = {
                "python_algorithms": "algorithms",
                "math_arithmetic": "math",
                "python_strings": "strings",
                "python_general": "algorithms",
                "python_math": "math",
                "python_oop": "algorithms",
                "python_file_io": "algorithms",
            }.get(topic, "algorithms")

            print(f"  [Curriculum] Proposing {n_tasks} tasks for domain '{curr_domain}'...")
            # Propose tasks via the model, validate with executor
            proposed = self.curriculum.propose_tasks(
                domain=curr_domain, n=n_tasks, mode="induction")

            # Convert to GoalTasks for the existing training pipeline
            all_tasks = self.curriculum.to_goal_tasks(proposed)

            # If curriculum didn't produce enough, fall back to fixed archetypes
            if len(all_tasks) < n_tasks // 2:
                print(f"  [Curriculum] Only {len(all_tasks)} validated, "
                      f"supplementing with fixed archetypes")
                for arch in archetypes:
                    for diff in ["easy", "medium"]:
                        try:
                            task = _goal_gen.generate(archetype=arch, difficulty=diff)
                            all_tasks.append(task)
                        except (ValueError, KeyError):
                            pass

            self.curriculum.print_stats()
        else:
            # Fixed archetype-based task generation
            for arch in archetypes:
                for diff in ["easy", "medium"]:
                    try:
                        task = _goal_gen.generate(archetype=arch, difficulty=diff)
                        all_tasks.append(task)
                    except (ValueError, KeyError):
                        pass
            # Add more tasks by generating with adaptive difficulty
            for _ in range(n_tasks - len(all_tasks)):
                try:
                    # Pick random archetype from topic
                    arch = rng.choice(archetypes)
                    task = _goal_gen.generate(archetype=arch)
                    all_tasks.append(task)
                except (ValueError, KeyError):
                    pass

        if not all_tasks:
            print(f"  No goal tasks generated for topic '{topic}', skipping")
            return {"topic": topic, "saved": False, "n_tasks": 0}

        # ── Train/val split with minimum val size ────────────────────
        # Minimum 10 val tasks for statistical significance (25% increments
        # with 4 tasks is random noise). If not enough tasks, use 30% val.
        rng.shuffle(all_tasks)
        n_val = max(10, int(len(all_tasks) * max(val_ratio, 0.3)))
        n_val = min(n_val, len(all_tasks) // 2)  # never more than half
        if n_val < 5:
            n_val = max(1, len(all_tasks) // 3)  # fallback for tiny datasets
        val_tasks = all_tasks[:n_val]
        train_tasks = all_tasks[n_val:]
        print(f"  Tasks: {len(train_tasks)} train / {len(val_tasks)} val "
              f"(total {len(all_tasks)})")

        # Load fine-tuned topic expert from AirMoE library (if previously trained)
        # The base v4 library contains SVD-compressed copies of base model experts —
        # loading those is redundant (base model already has them) and the SVD
        # compression at 90% energy compounds across 28 layers causing output
        # degradation. Only load if the expert was previously FINE-TUNED by this
        # training loop (saved with svd_energy=0.99 after DoRA fine-tuning).
        if self.airmoe is not None:
            # Check if this topic has been fine-tuned (look for a marker file)
            marker = self.airmoe.module_dir / "experts" / f".trained_{topic}"
            if marker.exists():
                loaded = self.airmoe.load_topic(topic)
                if loaded:
                    print(f"  [AirMoE] Loaded fine-tuned expert for topic '{topic}'")
                else:
                    print(f"  [AirMoE] Load failed — using base expert 0 weights")
            else:
                print(f"  [AirMoE] No fine-tuned expert for '{topic}' — using base weights")
        else:
            print(f"  (AirMoE disabled — using base expert 0 weights)")

        # Collect successful solutions
        successes = []
        failures = []
        val_history = []
        lcb_history = []        # LiveCodeBench pass rate per epoch
        best_val_rate = -1.0    # best rolling average
        best_single = -1.0      # best single-epoch val rate
        epochs_without_improve = 0
        best_expert_state = None
        best_ft_stats = None
        stopped_early = False

        # ── Curriculum: sort by difficulty (easy first) ──────────────
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        train_tasks_sorted = sorted(train_tasks, key=lambda t: diff_order.get(t.difficulty, 1))

        # Track per-task success across epochs — skip tasks solved in prior epochs
        # to force the model to attempt unsolved tasks (prevents memorizing solved ones)
        solved_tasks = set()  # set of task descriptions that were solved

        for epoch in range(n_epochs):
            print(f"\n  Epoch {epoch+1}/{n_epochs}")
            epoch_successes = 0
            epoch_skipped = 0

            for i, task in enumerate(train_tasks_sorted):
                # Skip tasks already solved in previous epochs (anti-overfitting)
                if task.description in solved_tasks and epoch > 0:
                    epoch_skipped += 1
                    continue

                task_short = f"{task.archetype}({task.difficulty}): {task.description[:40]}"
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"\n    [{i+1}/{len(train_tasks_sorted)}] {task_short}")

                # Run goal-oriented self-play with multi-dim scoring
                result = self.engine.run_goal_task(task, k_samples=k_samples,
                                                   use_reasoning=True)

                attempts = result.get("attempts", [])
                success = result.get("final_success", False)
                best_quality = result.get("best_quality", 0.0)

                # Record result for adaptive difficulty
                _goal_gen.record_result(task.domain, success)

                # Record result for curriculum difficulty adaptation (Goldilocks)
                if self.curriculum is not None:
                    # Find the ProposedTask that generated this GoalTask
                    proposed = next((t for t in self.curriculum._task_queue
                                     if t.id == task.id), None)
                    if proposed is not None:
                        self.curriculum.record_result(proposed, success)

                # Show results
                show = success or (i + 1) % 10 == 0
                if show:
                    for att in attempts:
                        if att.get("sample", 0) > 0:
                            continue  # only show sample 0 for brevity
                        code_lines = att["code"].strip().split("\n")
                        code_preview = code_lines[0][:80] if code_lines else ""
                        status = "OK" if att["success"] else "FAIL"
                        q = att.get("quality", 0)
                        acc = "acc" if att.get("accepted") else "rej"
                        print(f"      R{att['round']}: {status} {acc} q={q:.2f} | {code_preview}")
                        if att["success"]:
                            ems = att.get("exec_time_ms", 0)
                            gms = att.get("gen_time_ms", 0)
                            nodes = att.get("ast_nodes", 0)
                            print(f"        {gms:.0f}ms gen + {ems:.0f}ms exec | {nodes} AST nodes")
                        elif att.get("error"):
                            err_lines = att["error"].split("\n")
                            for el in err_lines[:2]:
                                if el.strip():
                                    print(f"        Error: {el.strip()[:100]}")

                if success:
                    epoch_successes += 1
                    solved_tasks.add(task.description)  # mark as solved (skip next epoch)
                    # Find the best accepted attempt
                    best_att = max(
                        (a for a in attempts if a.get("accepted")),
                        key=lambda a: a.get("quality", 0),
                        default=attempts[-1] if attempts else {})
                    successes.append({
                        "prompt": task.description,
                        "goal_id": task.id,
                        "solution": best_att.get("code", ""),
                        "reasoning": result.get("reasoning_text", ""),
                        "rounds": result.get("rounds_used", 0),
                        "quality": best_quality,
                        "scores": best_att.get("scores", {}),
                        "ast_nodes": best_att.get("ast_nodes", 0),
                        "fingerprint": best_att.get("fingerprint", {}),
                        "gen_ms": best_att.get("gen_time_ms", 0),
                        "exec_ms": best_att.get("exec_time_ms", 0),
                        "archetype": task.archetype,
                        "difficulty": task.difficulty,
                        "epoch": epoch,
                    })
                else:
                    last_err = attempts[-1].get("error", "no output") if attempts else "no output"
                    failures.append({
                        "goal_id": task.id,
                        "archetype": task.archetype,
                        "last_error": last_err,
                        "rounds": result.get("rounds_used", 0),
                    })

            train_rate = epoch_successes / len(train_tasks_sorted)
            skip_info = f" (skipped {epoch_skipped} solved)" if epoch_skipped > 0 else ""
            print(f"\n  Epoch {epoch+1} train: {epoch_successes}/{len(train_tasks_sorted)} "
                  f"({train_rate:.1%}){skip_info}")

            # ── Deduplicate THIS EPOCH's new successes only ───────────
            # Only train on solutions from this epoch (not accumulated).
            # Re-training on old successes every epoch causes memorization.
            epoch_new = [s for s in successes if s.get("epoch", 0) == epoch]
            high_quality = [s for s in epoch_new if s.get("quality", 0) > 0.7]
            if len(high_quality) < len(epoch_new):
                print(f"  Quality filter: {len(epoch_new)} → {len(high_quality)} (q>0.7)")
            epoch_successes_dedup = self._dedup_successes(high_quality)
            if len(epoch_successes_dedup) < len(high_quality):
                print(f"  Dedup: {len(high_quality)} → {len(epoch_successes_dedup)} unique")

            # ── Fine-tune expert on this epoch's new successes ───────
            # Only new data per epoch prevents repetitive gradient on same samples.
            ft_stats = None
            if epoch_successes_dedup:
                # Free KV cache from generation before training (critical for VRAM)
                if self.vram:
                    self.vram.check_before_generation(f"epoch_{epoch}_pre_ft")
                    self.vram.empty_cache()
                else:
                    torch.cuda.empty_cache()
                print(f"\n  Fine-tuning expert on {len(epoch_successes_dedup)} solutions...")
                ft_stats = self._finetune_expert(topic, epoch_successes_dedup)
                self.model.eval()  # back to eval mode for validation
                # Free training activations before validation/benchmarks
                if self.vram:
                    self.vram.empty_cache()
                else:
                    torch.cuda.empty_cache()

            # ── Validation (measures FINE-TUNED model) ────────────────
            val_successes = 0
            for j, vt in enumerate(val_tasks):
                vresult = self.engine.run_goal_task(vt, k_samples=1, use_reasoning=True)
                if vresult.get("final_success", False):
                    val_successes += 1

            val_rate = val_successes / len(val_tasks)
            val_history.append(val_rate)
            print(f"  Epoch {epoch+1} val:   {val_successes}/{len(val_tasks)} "
                  f"({val_rate:.1%})")

            # ── LiveCodeBench contamination-free benchmark ────────────
            # Clear CUDA cache before benchmarks to free training activations
            if self.vram:
                self.vram.empty_cache()
            else:
                torch.cuda.empty_cache()
            # Run a small LCB eval after each epoch to track true generalization
            # on problems Qwen2.5-Coder couldn't have seen (post-Sept 2024)
            lcb_stats = None
            if self.lcb_n_problems > 0:
                try:
                    from research.evaluation.livecodebench_eval import LiveCodeBenchEvaluator
                    if not hasattr(self, '_lcb_evaluator'):
                        self._lcb_evaluator = LiveCodeBenchEvaluator(
                            self.model, self.tokenizer, device=self.device)
                    lcb_results = self._lcb_evaluator.run(
                        start_date="2024-09-01",
                        n_problems=self.lcb_n_problems,
                        max_tokens=400, temperature=0.0, timeout_s=10.0)
                    lcb_stats = {
                        "pass_rate": lcb_results.get("pass_rate", 0.0),
                        "n_passed": lcb_results.get("n_passed", 0),
                        "n_total": lcb_results.get("n_total", 0),
                        "categories": lcb_results.get("categories", {}),
                    }
                    lcb_history.append(lcb_stats["pass_rate"])
                    print(f"  Epoch {epoch+1} LCB:  {lcb_stats['n_passed']}/{lcb_stats['n_total']} "
                          f"({lcb_stats['pass_rate']:.1%}) [contamination-free]")
                    # Print sore spots on last epoch
                    if epoch == n_epochs - 1 or stopped_early:
                        self._lcb_evaluator.print_sore_spots(lcb_results)
                except Exception as e:
                    print(f"  (LCB eval skipped: {e})")
                    lcb_stats = None

            # ── Reasoning & creativity benchmarks ─────────────────────
            if self.vram:
                self.vram.empty_cache()
            else:
                torch.cuda.empty_cache()
            # ARC-AGI-2 (fluid intelligence), NeoCoder (creativity),
            # FineReason (step reasoning), ThinkBench (OOD reasoning)
            reasoning_stats = None
            if self.reasoning_n_problems > 0:
                try:
                    from research.evaluation.reasoning_benchmarks import ReasoningBenchmarkSuite
                    if not hasattr(self, '_reasoning_suite'):
                        self._reasoning_suite = ReasoningBenchmarkSuite(
                            self.model, self.tokenizer, device=self.device)
                    rb_results = self._reasoning_suite.run(
                        benchmarks=self.reasoning_benchmarks,
                        n_problems=self.reasoning_n_problems)
                    reasoning_stats = {}
                    for bname, bresult in rb_results.get("results", {}).items():
                        if "error" in bresult:
                            continue
                        if "pass_rate" in bresult:
                            reasoning_stats[bname] = bresult["pass_rate"]
                        elif "avg_neogauge" in bresult:
                            reasoning_stats[bname] = bresult["avg_neogauge"]
                    if reasoning_stats:
                        parts = [f"{k}={v:.1%}" for k, v in reasoning_stats.items()]
                        print(f"  Epoch {epoch+1} Reasoning: {' | '.join(parts)}")
                        if epoch == n_epochs - 1 or stopped_early:
                            self._reasoning_suite.print_report(rb_results)
                except Exception as e:
                    print(f"  (Reasoning eval skipped: {e})")
                    reasoning_stats = None

            # ── Early stopping with rolling average ──────────────────
            # Use rolling average of last 2 val rates to reduce noise from
            # small val sets. Compare rolling avg vs best rolling avg.
            val_history.append(val_rate)
            rolling_avg = sum(val_history[-2:]) / min(len(val_history), 2)
            if rolling_avg > best_val_rate:
                best_val_rate = rolling_avg
                best_single = max(best_single, val_rate)
                epochs_without_improve = 0
                best_expert_state = self._snapshot_expert()
                best_ft_stats = ft_stats
                print(f"  * New best rolling val: {best_val_rate:.1%} "
                      f"(single={val_rate:.1%})")
            else:
                epochs_without_improve += 1
                # Also check if single epoch is new best (catch non-monotonic improvement)
                if val_rate > best_single:
                    best_single = val_rate
                    best_expert_state = self._snapshot_expert()
                    best_ft_stats = ft_stats
                    print(f"  * New best single val: {best_single:.1%} "
                          f"(rolling={rolling_avg:.1%})")
                else:
                    print(f"  No improvement ({epochs_without_improve}/{patience})")
                if epochs_without_improve >= patience:
                    print(f"  ! Early stopping at epoch {epoch+1}")
                    stopped_early = True
                    break

        # Restore best expert weights
        if best_expert_state is not None:
            self._restore_expert(best_expert_state)
            print(f"  Restored best expert (val rate: {best_val_rate:.1%})")

        # Save the best expert (already restored to best state)
        stats = {
            "topic": topic,
            "n_tasks": len(all_tasks),
            "n_train": len(train_tasks),
            "n_val": len(val_tasks),
            "n_epochs_run": len(val_history),
            "stopped_early": stopped_early,
            "val_history": val_history,
            "lcb_history": lcb_history,
            "best_val_rate": best_val_rate,
            "total_successes": len(successes),
            "total_failures": len(failures),
            "train_success_rate": len(successes) / max(len(train_tasks) * len(val_history), 1),
        }

        if best_expert_state is not None:
            print(f"\n  Saving best expert (val rate: {best_val_rate:.1%})...")
            self._save_expert(topic)
            stats["saved"] = True
            if best_ft_stats:
                stats["finetune"] = best_ft_stats
        else:
            print(f"  No verified successes - skipping save")
            stats["saved"] = False

        # Log
        log_path = self.log_dir / f"{topic}_training.json"
        with open(log_path, "w") as f:
            json.dump({**stats, "successes": successes, "failures": failures},
                      f, indent=2)

        return stats

    def _dedup_successes(self, successes: List[Dict]) -> List[Dict]:
        """Deduplicate successful solutions by (prompt, solution) key.

        Also caps per-prompt solutions to avoid dominance of any single prompt
        (default max 2 solutions per prompt for diversity). Uses fuzzy matching
        to reject near-identical solutions (not just exact duplicates).
        """
        PER_PROMPT_CAP = 2  # reduced from 3 — less memorization per prompt
        SIMILARITY_THRESHOLD = 0.85  # reject solutions > 85% similar to a kept one

        def _solution_similarity(a: str, b: str) -> float:
            """Jaccard similarity on token sets (fast, no deps)."""
            ta, tb = set(a.split()), set(b.split())
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)

        by_prompt: Dict[str, List[Dict]] = {}
        for s in successes:
            key = s.get("prompt", "")
            by_prompt.setdefault(key, []).append(s)

        deduped = []
        kept_solutions = []  # all kept solutions for cross-prompt similarity check
        for prompt, items in by_prompt.items():
            items.sort(key=lambda x: x.get("quality", 0.0), reverse=True)
            kept = 0
            for item in items:
                sol = item.get("solution", "")
                # Check similarity against ALL kept solutions (not just same prompt)
                too_similar = False
                for prev_sol in kept_solutions:
                    if _solution_similarity(sol[:300], prev_sol[:300]) > SIMILARITY_THRESHOLD:
                        too_similar = True
                        break
                if too_similar:
                    continue
                kept_solutions.append(sol[:300])
                deduped.append(item)
                kept += 1
                if kept >= PER_PROMPT_CAP:
                    break
        return deduped

    def _snapshot_expert(self) -> Dict:
        """Snapshot current expert 0 weights across all layers."""
        state = {}
        for i, block in enumerate(self.model.blocks):
            if hasattr(block.ffn, 'experts') and len(block.ffn.experts) > 0:
                exp = block.ffn.experts[0]
                state[i] = {
                    "w1": exp.w1.weight.data.clone(),
                    "w2": exp.w2.weight.data.clone(),
                    "w3": exp.w3.weight.data.clone(),
                }
        return state

    def _restore_expert(self, state: Dict):
        """Restore expert 0 weights from snapshot."""
        for i, block in enumerate(self.model.blocks):
            if i in state and hasattr(block.ffn, 'experts') and len(block.ffn.experts) > 0:
                exp = block.ffn.experts[0]
                exp.w1.weight.data.copy_(state[i]["w1"])
                exp.w2.weight.data.copy_(state[i]["w2"])
                exp.w3.weight.data.copy_(state[i]["w3"])

    def _finetune_expert(self, topic: str, successes: List[Dict]) -> Dict:
        """Fine-tune on successful solutions using LoRA adapters.

        Applies LoRA to FFN layers (w1, w2, w3 or fc1, fc2) of each block.
        For MoE models: targets expert 0 only.
        For dense models: targets the FFN directly.
        Keeps trainable params to ~4-8M (vs 560M full), saving ~2GB VRAM.
        """
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        from research.architecture.dora import apply_dora_to_linear
        trainable = []
        n_lora = 0

        # Only apply LoRA to last N layers — early layers learn general features,
        # late layers are task-specific. This prevents catastrophic forgetting.
        n_blocks = len(self.model.blocks)
        LORA_START = max(0, n_blocks - 8)  # last 8 layers only

        for block_idx, block in enumerate(self.model.blocks):
            if block_idx < LORA_START:
                continue  # skip early layers
            if hasattr(block.ffn, 'experts') and len(block.ffn.experts) > 0:
                # MoE model: target expert 0
                expert = block.ffn.experts[0]
                for attr_name in ['w1', 'w2', 'w3']:
                    if hasattr(expert, attr_name):
                        layer = getattr(expert, attr_name)
                        if hasattr(layer, 'weight') and not hasattr(layer, 'lora_A'):
                            wrapped = apply_dora_to_linear(layer, rank=8, alpha=16, dropout=0.1)
                            setattr(expert, attr_name, wrapped)
                            n_lora += 1
                            for p in wrapped.parameters():
                                if p.requires_grad:
                                    trainable.append(p)
            else:
                # Dense model: target FFN directly
                ffn = block.ffn
                for attr_name in ['w1', 'w2', 'w3', 'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(ffn, attr_name):
                        layer = getattr(ffn, attr_name)
                        if hasattr(layer, 'weight') and not hasattr(layer, 'lora_A'):
                            wrapped = apply_dora_to_linear(layer, rank=8, alpha=16, dropout=0.1)
                            setattr(ffn, attr_name, wrapped)
                            n_lora += 1
                            for p in wrapped.parameters():
                                if p.requires_grad:
                                    trainable.append(p)

        # Ensure only LoRA params are trainable
        for param in self.model.parameters():
            param.requires_grad = False
        for p in trainable:
            p.requires_grad = True

        print(f"    LoRA adapters: {n_lora} layers, "
              f"trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params")

        # 8-bit AdamW for LoRA params
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable, lr=self.lr, weight_decay=0.15)
            print(f"    Optimizer: bnb AdamW8bit (wd=0.15)")
        except ImportError:
            optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.15)
            print(f"    Optimizer: torch AdamW")

        # Gradient checkpointing to save activation VRAM
        use_grad_ckpt = hasattr(self.model, 'gradient_checkpointing_enable')
        if use_grad_ckpt:
            self.model.gradient_checkpointing_enable()
            print(f"    Gradient checkpointing: ON")

        # Regularization: label smoothing prevents overconfidence on self-generated
        # data; input dropout noise adds robustness. Both fight overfitting to the
        # model's own outputs (self-reinforcing loop).
        LABEL_SMOOTHING = 0.2  # aggressive: self-generated data may contain errors
        INPUT_NOISE_STD = 0.05  # stronger noise fights self-reinforcing loop
        GRAD_ACCUM_STEPS = 4  # accumulate gradients for effective batch size = 4
        MAX_EPOCHS_PER_SESSION = 2  # cap epochs per fine-tune call (prevents memorization)

        # Shuffle successes to avoid curriculum bias in gradient updates
        import random as _rng
        shuffled_successes = successes[:]
        _rng.Random().shuffle(shuffled_successes)

        # Training loop: supervised on successful (prompt → solution) pairs
        total_loss = 0.0
        n_batches = 0
        accum_count = 0

        for item in shuffled_successes:
            prompt = item["prompt"]
            solution = item["solution"]
            quality = item.get("quality", 0.5)  # weighted by efficiency + test pass

            # Construct training text: prompt + solution
            full_text = prompt + solution
            if not full_text.endswith(self.tokenizer.eos_token or "<|endoftext|>"):
                full_text += self.tokenizer.eos_token or ""

            # Tokenize
            enc = self.tokenizer(full_text, return_tensors="pt",
                                 truncation=True, max_length=256)
            input_ids = enc.input_ids.to(self.device)

            # Find where the prompt ends (only compute loss on solution tokens)
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]

            if input_ids.shape[1] <= prompt_len + 1:
                continue  # Solution too short

            # Forward pass (gradient accumulation)
            self.model.train()

            logits, _ = self.model(input_ids)

            # Loss only on solution tokens (shift by 1 for next-token prediction)
            solution_logits = logits[0, prompt_len-1:-1, :]  # predict solution tokens
            solution_targets = input_ids[0, prompt_len:]

            if solution_logits.shape[0] == 0:
                continue

            # Label smoothing regularization — prevents overconfidence on
            # self-generated solutions (which may contain subtle errors)
            loss = torch.nn.functional.cross_entropy(
                solution_logits, solution_targets, label_smoothing=LABEL_SMOOTHING)

            # Weight loss by quality — efficient + test-passed solutions get more gradient
            # Higher quality = lower weighted loss (we want to learn more from good solutions)
            # So we scale the loss: quality=1.0 → full loss, quality=0.3 → 0.3x loss
            weighted_loss = loss * quality / GRAD_ACCUM_STEPS
            weighted_loss.backward()
            accum_count += 1

            # Only step every GRAD_ACCUM_STEPS samples (effective batch size = 4)
            if accum_count >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            total_loss += loss.item()
            n_batches += 1

        # Final step for remaining accumulated gradients
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        self.model.eval()

        # Disable gradient checkpointing for generation/eval
        if use_grad_ckpt:
            self.model.gradient_checkpointing_disable()

        # Merge LoRA weights back and remove adapters
        # (frees LoRA VRAM, keeps the learned knowledge in base weights)
        for block in self.model.blocks:
            if hasattr(block.ffn, 'experts') and len(block.ffn.experts) > 0:
                target = block.ffn.experts[0]
            else:
                target = block.ffn
            for attr_name in ['w1', 'w2', 'w3', 'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']:
                if hasattr(target, attr_name):
                    layer = getattr(target, attr_name)
                    if hasattr(layer, 'merge_and_unload'):
                        merged = layer.merge_and_unload()
                        setattr(target, attr_name, merged)

        # Count trainable params before deleting
        trainable_params = sum(p.numel() for p in trainable)

        # Free optimizer and LoRA tensors
        del optimizer, trainable
        if self.vram:
            self.vram.empty_cache()
        else:
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Fine-tune: {n_batches} batches, avg_loss={avg_loss:.4f}")

        return {
            "n_samples": len(successes),
            "n_batches": n_batches,
            "avg_loss": avg_loss,
            "trainable_params_m": trainable_params / 1e6,
        }

    def _save_expert(self, topic: str):
        """Save the fine-tuned expert back to the v4 library."""
        from safetensors.torch import save_file

        def _compress_svd(w, energy=0.99):
            U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
            cumsum = (S ** 2).cumsum(0)
            total = cumsum[-1]
            rank = max(1, (cumsum < energy * total).sum().item() + 1)
            return {
                "U": U[:, :rank].contiguous().to(torch.bfloat16),
                "U_shape": torch.tensor(U[:, :rank].shape, dtype=torch.int32),
                "S": S[:rank].to(torch.float16),
                "Vh": Vh[:rank, :].contiguous().to(torch.bfloat16),
                "Vh_shape": torch.tensor(Vh[:rank, :].shape, dtype=torch.int32),
                "rank": torch.tensor([rank], dtype=torch.int32),
            }

        # If AirMoE is disabled, save to local directory instead
        if self.airmoe is None:
            experts_dir = Path(self.log_dir / "experts")
            experts_dir.mkdir(parents=True, exist_ok=True)
            n_layers = len(self.model.blocks)
            v4_dir = None
        else:
            experts_dir = self.airmoe.module_dir / "experts"
            n_layers = self.airmoe.n_layers
            v4_dir = str(self.airmoe.module_dir)

        n_saved = 0

        for layer in range(n_layers):
            try:
                block = self.model.blocks[layer]
                ffn = block.ffn
                if not (hasattr(ffn, 'experts') and len(ffn.experts) > 0):
                    continue

                expert = ffn.experts[0]

                # Compress each weight part (SVD only — near-lossless at 0.99)
                compressed = {}
                for part_name, param in [("w1", expert.w1.weight.data),
                                          ("w2", expert.w2.weight.data),
                                          ("w3", expert.w3.weight.data)]:
                    w = param.detach().cpu().float()
                    comp = _compress_svd(w, energy=0.99)
                    for k, v in comp.items():
                        compressed[f"{part_name}_{k}"] = v

                shard_name = f"expert_l{layer}_{topic}.safetensors"
                shard_path = experts_dir / shard_name
                save_file(compressed, str(shard_path))
                n_saved += 1

            except (IndexError, AttributeError):
                continue

        print(f"    Saved {n_saved} expert files for topic '{topic}'")

        # Invalidate AirMoE cache for this topic so next load reads from disk
        if self.airmoe is not None:
            if topic in self.airmoe.cache:
                del self.airmoe.cache[topic]
                if self.airmoe.current_topic == topic:
                    self.airmoe.current_topic = None
                    self.airmoe.current_experts = None
            # Write marker file so future runs know this topic was fine-tuned
            marker = self.airmoe.module_dir / "experts" / f".trained_{topic}"
            marker.write_text("trained")

        # Update manifest.json so router knows about this topic
        if v4_dir:
            try:
                from train_expert import update_manifest
                update_manifest(v4_dir, topic)
            except Exception as e:
                print(f"    (Manifest update skipped: {e})")

    def train_all_topics(self, n_epochs: int = 3, n_tasks: int = 50) -> List[Dict]:
        """Train all available topics in the library.

        Reads topics from the V4 manifest (not hardcoded list).
        Falls back to DOMAIN_ARCHETYPES if no manifest found.
        """
        topics = list(DOMAIN_ARCHETYPES.keys())
        # Also include topics from the manifest
        if self.airmoe is not None:
            manifest_topics = self.airmoe.router.list_topics()
            for t in manifest_topics:
                if t not in topics:
                    topics.append(t)
        print(f"\nTraining {len(topics)} topics: {topics}")

        all_stats = []
        for topic in topics:
            stats = self.train_topic(topic, n_tasks=n_tasks, n_epochs=n_epochs)
            all_stats.append(stats)

        return all_stats


def main():
    parser = argparse.ArgumentParser(description="Self-play expert training")
    parser.add_argument("--topics", type=str, default="",
                        help="Comma-separated topics to train (default: all)")
    parser.add_argument("--all-topics", action="store_true",
                        help="Train all available topics")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Max self-play rounds per goal task")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Max training epochs per topic")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stop patience (epochs without val improvement)")
    parser.add_argument("--n-tasks", type=int, default=50,
                        help="Number of goal tasks to generate per topic")
    parser.add_argument("--k-samples", type=int, default=1,
                        help="Independent samples for self-consistency (VERSE)")
    parser.add_argument("--lcb-problems", type=int, default=0,
                        help="LiveCodeBench problems per epoch (0=disabled, ~20-50 recommended)")
    parser.add_argument("--reasoning-problems", type=int, default=0,
                        help="Reasoning benchmark problems per epoch (0=disabled, ~5-10 recommended)")
    parser.add_argument("--reasoning-benchmarks", type=str, default="all",
                        help="Comma-separated: arc_agi2,neocoder,finereason,thinkbench (default: all)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default 2e-5 — lower prevents catastrophic forgetting)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0=greedy, >0=diverse solutions)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling for solution diversity")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling for solution diversity")
    parser.add_argument("--vram-budget", type=float, default=0,
                        help="VRAM budget in GB (0=auto, set to 8-10 for 12GB cards to avoid OOM)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use fp16 (default for 12GB cards, saves ~3GB weight VRAM)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Use fp32 instead (higher precision, 2x VRAM)")
    parser.add_argument("--use-curriculum", action="store_true", default=False,
                        help="Use AZR-style infinite curriculum (model proposes tasks)")
    parser.add_argument("--use-mac-attn", action="store_true", default=False,
                        help="Enable MAC-Attention (reuse attention for similar queries, long context)")
    parser.add_argument("--use-spec-attn", action="store_true", default=False,
                        help="Enable L1 Speculative Attention (low-rank draft + entropy verify, 57% attn cut, lossless)")
    args = parser.parse_args()

    print("=" * 70)
    print("Self-Play Expert Training")
    print("=" * 70)

    # ── VRAM Manager (BOOT_TIME_AUDIT: Stage 3 + Stage 4) ──────────
    # Persistent compile cache + dynamic KV cache sizing
    vram = VRAMManager(
        total_vram_gb=12.0,
        safety_margin_gb=1.0 if not args.fp16 else 0.5,
    )
    vram.setup_compile_cache()

    # Load ForgeLM v2 (fixed — rebuilt from v1 with identity-init keys only)
    # V2 = V1 + QK-Norm + DenseFormer + SandwichNorm + LogitCap + SwiGLUClamp
    # All lossless at init, verified to produce identical output to V1
    # Load in bf16 directly (checkpoint is bf16, avoids fp32 upcast: 3.2GB vs 6.3GB)
    print("\n[1] Loading ForgeLM v2 (rebuilt, bf16)...")
    cfg = get_config("forgelm_v2", device="cuda")
    load_dtype = torch.bfloat16 if args.fp16 else torch.float32
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path="research/checkpoints/forgelm_v2.safetensors",
        moe_top_k=0,  # 0 = dense_bypass: skip untrained router, run all experts
        dtype=load_dtype)
    model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

    # Profile VRAM after model load (Stage 4: growable KV cache pattern)
    vram.profile_after_model_load(model, "v2_loaded")

    # ── Apply runtime keys (all lossless at init) ─────────────────
    # From KEY_MAPPING_MASTER.md + KEY_NOVELTY_AUDIT.md:
    #   Per-Query Temp: T(Q)=softplus(w·Q+b), init=standard attention (F5)
    #   Norm-Gated MoD: skip layers by residual-delta norm, threshold=0 (novel)
    # NOTE: Softpick disabled — materializes full attention matrix (no FA2),
    #   causes OOM on 12GB. Re-enable after implementing custom softpick FA2 kernel.
    print("\n[2] Applying runtime keys...")
    apply_per_query_temp(model)
    apply_norm_gated_mod(model)

    # MAC-Attention: reuse attention for similar queries (context-independent speed)
    # Only activates for long contexts (> 512 tokens). Lossless at init.
    mac_key = None
    if getattr(args, 'use_mac_attn', False):
        from research.keys.attn_reuse_key import AttnReuseKey
        mac_key = AttnReuseKey(max_entries=16, match_threshold=0.85)
        mac_key.apply(model)
        print(f"  [MAC-Attention] Patched {len(mac_key._patched_layers)} layers "
              f"(activates at >512 token context)")

    # L1 Speculative Attention: low-rank draft + entropy-based verify
    # 56.8% attention compute saved, cos=1.0 (lossless), 71% accept rate
    spec_attn_key = None
    if getattr(args, 'use_spec_attn', False):
        from research.keys.speculative_keys import SpeculativeAttentionKey
        spec_attn_key = SpeculativeAttentionKey(draft_rank=32)
        spec_attn_key.apply(model)
        print(f"  [SpecAttn] Patched {len(spec_attn_key._patched)} attention layers "
              f"(lossless, 57% attn compute cut on 71% accept rate)")

    vram.empty_cache()

    # Warmup generation to measure peak VRAM (captures CUDA kernel overhead)
    vram.profile_after_warmup(model, tokenizer, max_warmup_tokens=16)

    # Calculate dynamic max generation tokens based on measured free VRAM
    # ForgeLM v2: 28 layers, 12 heads, 128 head_dim, bf16 (2 bytes)
    # Cap at 120 for code generation (tasks are short, prevents VRAM pressure)
    dynamic_max_tokens = min(
        vram.max_gen_tokens(n_layers=28, n_heads=12, head_dim=128,
                           dtype_bytes=2, overhead_mb=512),
        120)
    # With KV cache, we can afford more tokens (O(n) not O(n²) per step)
    # Curriculum task proposal needs ~200 tokens for task+tests+reference
    if args.use_curriculum:
        dynamic_max_tokens = min(dynamic_max_tokens + 80, 200)
    print(f"  [VRAM] Dynamic max generation: {dynamic_max_tokens} tokens")

    # ── AirMoE: Infinite expert library with VRAM-limited hotswap ──
    # v4 expert library: 14 topics × 28 layers, SVD-compressed (~15 MB each)
    # Experts were built from v2 MoE experts (slices of original Qwen FFN).
    # Only math_arithmetic (expert 0) was affected by the v2 corruption in
    # layers 20-27 — all other topics use experts 1/2/3 which were not corrupted.
    # v2 has since been rebuilt from v1 (clean). The v4 library will be rebuilt
    # from clean v2 in a future pass; for now, non-math topics are safe to use.
    v4_dir = "D:/windsurf/ForgeAI/research/checkpoints/forgelm_v4"
    v4_manifest = os.path.join(v4_dir, "manifest.json")
    if os.path.exists(v4_manifest):
        print(f"\n[3] Creating AirMoE (v4 expert library)...")
        airmoe = InfiniteAirMoE(
            model, tokenizer, v4_dir,
            device="cuda",
            vram_budget_gb=2.0,       # 2 GB for expert cache (model uses ~3.2 GB)
            max_cached_topics=5,       # 5 topics × ~56 MB = ~280 MB per topic set
        )
        print(f"  Available topics: {airmoe.router.list_topics()}")
    else:
        print(f"\n[3] AirMoE disabled (no v4 manifest at {v4_manifest})")
        airmoe = None

    # Create trainer
    print("\n[4] Creating self-play trainer...")
    # Parse reasoning benchmarks
    if args.reasoning_benchmarks == "all":
        reasoning_benchmarks = ["arc_agi2", "neocoder", "finereason", "thinkbench"]
    else:
        reasoning_benchmarks = [b.strip() for b in args.reasoning_benchmarks.split(",")]

    trainer = ExpertSelfPlayTrainer(
        model, tokenizer, airmoe,
        learning_rate=args.lr,
        max_rounds=args.rounds,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        lcb_n_problems=args.lcb_problems,
        reasoning_n_problems=args.reasoning_problems,
        reasoning_benchmarks=reasoning_benchmarks,
        vram_manager=vram,
        max_gen_tokens=dynamic_max_tokens,
        use_curriculum=args.use_curriculum,
    )

    # Determine topics
    if args.all_topics or not args.topics:
        topics = list(DOMAIN_ARCHETYPES.keys())
    else:
        topics = [t.strip() for t in args.topics.split(",")]

    print(f"\n[5] Training topics: {topics}")

    all_stats = []
    for topic in topics:
        stats = trainer.train_topic(topic, n_tasks=args.n_tasks,
                                    n_epochs=args.epochs,
                                    patience=args.patience,
                                    k_samples=args.k_samples)
        all_stats.append(stats)

    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    for s in all_stats:
        status = "TRAINED" if s.get("saved") else "SKIPPED"
        sr = s.get("train_success_rate", 0)
        vr = s.get("best_val_rate", 0)
        early = " [early stop]" if s.get("stopped_early") else ""
        vh = " → ".join(f"{v:.0%}" for v in s.get("val_history", []))
        print(f"  {s['topic']:25s} {status:8s} "
              f"train={sr:.1%} val={vr:.1%}{early}")
        if vh:
            print(f"    val history: {vh}")
        lh = " -> ".join(f"{v:.1%}" for v in s.get("lcb_history", []))
        if lh:
            print(f"    LCB history: {lh} [contamination-free]")
        # Time stats from successes
        successes = s.get("successes", [])
        if successes:
            avg_gen = sum(x.get("gen_ms", 0) for x in successes) / len(successes)
            avg_exec = sum(x.get("exec_ms", 0) for x in successes) / len(successes)
            avg_nodes = sum(x.get("ast_nodes", 0) for x in successes) / len(successes)
            avg_q = sum(x.get("quality", 0) for x in successes) / len(successes)
            print(f"    time: {avg_gen:.0f}ms gen + {avg_exec:.0f}ms exec | "
                  f"avg AST nodes: {avg_nodes:.0f} | avg quality: {avg_q:.2f}")
        print(f"    ({s.get('total_successes', 0)} successes, "
              f"{s.get('n_train', 0)} train / {s.get('n_val', 0)} val)")

    # Save summary
    summary_path = trainer.log_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "topics_trained": len(all_stats),
            "results": all_stats,
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    if airmoe is not None:
        airmoe.print_stats()

    if trainer.curriculum is not None:
        trainer.curriculum.print_stats()

    if mac_key is not None:
        mac_key.print_stats()


if __name__ == "__main__":
    main()
