"""Infinite self-play training loop for tool-use.

Orchestrates the full cycle:
  1. Self-play: model runs tasks, collects tool-use trajectories + rewards
  2. Export: high-reward trajectories → SFT-format JSONL
  3. Finetune: SFT continuation from current best checkpoint
  4. Evaluate: run generalization + agent-loop tests on new checkpoint
  5. Promote/Demote: if new checkpoint passes, it becomes the new best;
     otherwise revert to previous best
  6. Repeat forever (or until max_epochs)

Usage:
    python -m research.self_play.discovery.infinite_tool_loop \\
        --checkpoint research/checkpoints/ForgeLM_V2_BSP.safetensors \\
        --epochs 10 --tasks-per-epoch 50

The loop is designed to be resumable: if interrupted, it picks up from the
last completed epoch using the saved checkpoint + DB state.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from research.paths import DATA_DIR
from research.self_play.discovery.discovery_db import DiscoveryDB
from research.self_play.discovery.tool_use_loop import (
    ToolUseSelfPlay, SelfPlayConfig, TaskCurriculum,
)


@dataclass
class LoopConfig:
    """Configuration for the infinite self-play loop."""
    # Self-play — more tasks per epoch since self-play is cheap (generation only)
    tasks_per_epoch: int = 50      # self-play is fast, FT is the bottleneck
    max_turns: int = 5
    max_gen_tokens: int = 256
    temperature: float = 0.5       # higher for tool-call exploration
    top_k: int = 80                # LFM2.5-recommended
    repetition_penalty: float = 1.05  # LFM2.5-recommended
    min_reward: float = 0.25

    # Context management
    max_seq_len: int = 32768       # model's context window
    context_threshold: float = 0.75  # summarize at 75% of budget
    keep_recent_turns: int = 6     # turns to keep intact during compression

    # Finetune — reduced steps since more trajectories provide richer signal
    ft_max_steps: int = 100        # reduced from 200: more data compensates
    ft_lr: float = 2e-5
    ft_batch_size: int = 1         # batch 1 + grad_accum 4 = effective batch 4
    ft_grad_accum: int = 4         # grad accumulation steps
    ft_seq_len: int = 1024
    ft_grad_checkpoint: bool = True  # activation checkpointing (saves ~0.5GB)
    ft_compile: bool = False  # torch.compile fails on Windows with vocab>32767
    ft_optimizer: str = "bnb"      # 8-bit AdamW (saves ~7GB vs fp32 AdamW)
    ft_lora: bool = True           # LoRA: train ~1M params instead of 1.17B
    ft_lora_r: int = 16            # LoRA rank
    ft_lora_alpha: int = 32        # LoRA alpha
    ft_entropy_alpha: float = 0.0  # disable entropy weighting (saves 1GB, enables chunked CE)
    ft_vram_limit_gb: float = 11.0 # emergency abort before OOM

    # GRPO (optional, alternates with SFT)
    use_grpo: bool = False
    grpo_group_size: int = 4
    grpo_tasks_per_step: int = 8
    grpo_steps: int = 20
    grpo_lr: float = 5e-6
    grpo_kl_coeff: float = 0.02

    # Loop control
    max_epochs: int = 10
    eval_threshold: float = 0.7  # min test pass rate to promote
    device: str = "cuda"

    # Paths
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data/finetune"
    # Model-generated goals
    use_generated_goals: bool = False  # model generates its own task goals
    goals_per_gen: int = 5            # generate 5 goals per batch, pick best
    goal_min_tokens: int = 15         # reject goals shorter than this (filler)
    goal_max_tokens: int = 120        # reject goals longer than this (rambling)
    goal_temperature: float = 0.6     # lower for coherent task descriptions


# ── Goal quality filter ────────────────────────────────────────────────────

# Goals that are too generic / filler — rejected by the filter.
_FILLER_PATTERNS = [
    "hello", "test", "hi ", "what is your name", "say ",
    "tell me a joke", "what can you do", "introduce yourself",
    "write hello world", "print hello", "count to ten",
    "what is 1+1", "what is 2+2", "simple", "basic test",
    "just ", "anything", "something", "whatever",
]

# Goals must require at least one of these to be non-filler.
_TOOL_KEYWORDS = ["search", "research", "script", "code", "build", "implement",
                  "investigate", "analyze", "compare", "benchmark", "design",
                  "experiment", "debug", "create", "explore", "study",
                  "write a", "run a", "test a", "find", "calculate",
                  "summarize", "propose", "record", "think about",
                  "hypothesis", "theory", "discovery"]


def _is_filler_goal(goal: str) -> bool:
    """Return True if goal is too easy/generic/filler or a tool-call JSON."""
    g = goal.lower().strip()
    if len(g) < 20:
        return True
    # Reject JSON tool calls (model defaulting to tool-call mode)
    if g.startswith("{") or g.startswith("[") or '"name"' in g or '"arguments"' in g:
        return True
    if "tool_call" in g or "function_call" in g:
        return True
    # Check filler patterns
    for pat in _FILLER_PATTERNS:
        if pat in g:
            return True
    # Must require some tool use or substantive thinking
    has_tool_keyword = any(kw in g for kw in _TOOL_KEYWORDS)
    if not has_tool_keyword:
        return True
    return False


# Sample curriculum tasks used as few-shot examples for goal generation.
# These teach the model the STYLE of task descriptions (imperative, natural
# language, multi-step) without mentioning tool-call JSON format.
_SAMPLE_TASKS = [
    "Search for 'speculative decoding', save the research, think about when it helps vs hurts, then query the DB to retrieve your saved research.",
    "Write and run a Python script that benchmarks list comprehension vs map() vs for-loop for squaring 10000 numbers. Think about why the winner is faster.",
    "Research 3 sorting algorithms on Wikipedia, implement each in Python, benchmark them on lists of size 100/1000/10000, and think about when each is superior. Record your discovery.",
    "Search for 'KV cache compression methods' and think about which method is best for long-context inference. Save the research and propose a theory about which is most memory-efficient.",
    "Investigate 'chain of thought vs direct answer' — run a script that simulates both approaches on a simple problem. Think about when CoT helps vs hurts. Save your findings.",
]

# Task templates — the model generates a {topic}, we fill it in.
# This plays to a 1.2B model's strength (short phrase generation) rather than
# its weakness (long structured instruction generation).
_TASK_TEMPLATES = [
    "Search for '{topic}' and think about how it applies to small language models. Save your research and record a discovery.",
    "Research '{topic}' on Wikipedia. Write a Python script that demonstrates the core concept. Think about what you learned and save your findings.",
    "Search for '{topic}' and think about whether it would help a 1.2B parameter model. Propose a theory and save your research.",
    "Investigate '{topic}' — search for it, read about it, think critically about the trade-offs. Run a script that illustrates the concept. Record a discovery.",
    "Compare 3 approaches to '{topic}' by searching for information on each. Write a script that benchmarks or demonstrates the differences. Think about which is best.",
    "Research '{topic}' using web search and Wikipedia. Think about what surprised you. Run a script that tests a key assumption. Save your research and record a discovery.",
    "Study '{topic}' — search for papers on arxiv, read the abstracts, and think about which approach is most promising. Write a script that demonstrates the key idea. Save your findings.",
    "Explore '{topic}' — search the web, run an experiment with a script, think about the results, and propose a theory. Record what you discovered.",
    "Investigate '{topic}' by searching Wikipedia and arxiv. Think about how it relates to your own architecture. Write a script that demonstrates a concept. Save your research.",
    "Research '{topic}' and think about its limitations. Write a script that shows where it fails or has trade-offs. Think about how to improve it. Record your discovery.",
    # Think-with-math templates: model must reason through math step by step
    "Think step by step about the mathematics behind '{topic}'. Use think to record each step of your reasoning. Then run a script to verify your calculations.",
    "Think through how to calculate key properties of '{topic}'. Use think to show your mathematical work. Then verify with the calculate tool.",
    "Research '{topic}' and think about the mathematical foundations. Use think to reason through the key equations step by step. Run a script to demonstrate. Save your findings.",
]

# Topic prompt — asks the model to generate diverse research topics.
# Uses raw completion (not chat) to avoid triggering tool-call mode.
_TOPIC_PROMPT = (
    "Here are diverse research topics for an AI to investigate:\n"
    "1. speculative decoding\n"
    "2. mixture of experts\n"
    "3. rotary position embeddings\n"
    "4. knowledge distillation\n"
    "5. quantization aware training\n"
    "6. retrieval augmented generation\n"
    "7. chain of thought reasoning\n"
    "8. attention mechanism optimization\n"
    "9. matrix multiplication optimization\n"
    "10. probability theory foundations\n"
    "11. modular arithmetic\n"
    "12. "
)


class GoalGenerator:
    """Uses the model to generate challenging, non-filler task goals.

    Strategy: a 1.2B model can't reliably generate full multi-step task
    instructions (it copies examples or produces degenerate output). Instead,
    we have it generate **topics** (short phrases — its strength), then fill
    those into curriculum-style task templates. This produces goals that
    match the style of non-generated goals while introducing novel content.

    Falls back to curriculum tasks if the model can't produce enough topics.
    """

    def __init__(self, engine, tokenizer, config: LoopConfig):
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self._seen_topics: set[str] = set()
        self._seen_goals: set[str] = set()
        self._template_idx = 0

    def _generate_topics(self, n: int) -> list[str]:
        """Generate n diverse topics using raw completion."""
        try:
            output = self.engine.generate_raw(
                _TOPIC_PROMPT,
                max_new_tokens=200,
                temperature=self.config.goal_temperature,
                top_k=60,
                repetition_penalty=1.15,
                skip_special_tokens=True,
            )
        except Exception as e:
            print(f"  [GoalGen] Topic generation failed: {e}")
            return []

        # Parse numbered lines from output
        import re
        topics = []
        full_text = _TOPIC_PROMPT + output
        for line in full_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip "N. " prefix
            m = re.match(r'^\d+\.\s*(.+)', line)
            if m:
                topic = m.group(1).strip().rstrip(".")
                # Filter: 3+ words, not too long, not a duplicate
                if len(topic) < 5 or len(topic) > 60:
                    continue
                if topic.lower() in self._seen_topics:
                    continue
                # Reject if it looks like a sentence (too many words)
                words = topic.split()
                if len(words) > 12:
                    continue
                # Reject if mostly numbers/punctuation
                alpha_chars = sum(1 for c in topic if c.isalpha())
                if alpha_chars < len(topic) * 0.5:
                    continue
                # Reject garbage topics: numbers, percentages, coordinates,
                # numbered list fragments ("7,8: cross-trait"), etc.
                import re as _re
                # Reject if starts with a digit or contains "N,M:" pattern
                if _re.match(r'^\d', topic):
                    continue
                if _re.search(r'\d+[,\.]\d+', topic):
                    continue
                # Reject if contains percentage signs or coordinate patterns
                if '%' in topic or ':' in topic[:10]:
                    continue
                # Reject if it's a known seed topic (already in the prompt)
                seed_topics = {"speculative decoding", "mixture of experts",
                               "rotary position embeddings", "knowledge distillation",
                               "quantization aware training", "retrieval augmented generation",
                               "chain of thought reasoning", "attention mechanism optimization",
                               "matrix multiplication optimization", "probability theory foundations",
                               "modular arithmetic"}
                if topic.lower() in seed_topics:
                    continue
                # Must have at least 2 alphabetic words
                alpha_words = [w for w in words if any(c.isalpha() for c in w)]
                if len(alpha_words) < 2:
                    continue
                self._seen_topics.add(topic.lower())
                topics.append(topic)
        return topics[:n]

    def _topic_to_goal(self, topic: str) -> str:
        """Fill a topic into a rotating task template."""
        template = _TASK_TEMPLATES[self._template_idx % len(_TASK_TEMPLATES)]
        self._template_idx += 1
        return template.format(topic=topic)

    def generate(self, n_needed: int) -> list[str]:
        """Generate n_needed quality goals from model-generated topics."""
        goals: list[str] = []
        max_attempts = max(3, n_needed // 5 + 2)

        for attempt in range(max_attempts):
            if len(goals) >= n_needed:
                break
            n_topics = min(10, n_needed - len(goals) + 5)
            topics = self._generate_topics(n_topics)
            for topic in topics:
                goal = self._topic_to_goal(topic)
                if goal.lower() in self._seen_goals:
                    continue
                if _is_filler_goal(goal):
                    continue
                self._seen_goals.add(goal.lower())
                goals.append(goal)
                if len(goals) >= n_needed:
                    break
            print(f"  [GoalGen] Attempt {attempt+1}: +{len(topics)} topics -> "
                  f"{len(goals)}/{n_needed} goals")

        return goals[:n_needed]


class InfiniteToolLoop:
    """The infinite self-play → finetune → evaluate → promote loop."""

    def __init__(self, checkpoint: str, config: LoopConfig | None = None):
        self.config = config or LoopConfig()
        self.best_checkpoint = checkpoint
        self.best_score = 0.0
        self.epoch = 0
        self.curriculum = TaskCurriculum()

        # DB for trajectory storage
        db_path = str(DATA_DIR / "discovery" / "tool_infinite.sqlite3")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = DiscoveryDB(db_path)

        # History tracking
        self.history: list[dict] = []

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.normpath(os.path.join(
            self.config.checkpoint_dir,
            f"ForgeLM_V2_LFM25-1.2B.sp{epoch}.safetensors"))

    def _export_trajectories(self, epoch: int) -> str:
        """Export high-reward trajectories from DB as SFT JSONL.

        Also mixes in non-tool examples (code, Q&A, reasoning) to prevent
        the model from forgetting how to answer directly without tools.
        This is critical to avoid the SFT4 overfitting issue where the model
        hallucinated tool calls for every non-tool question.
        """
        output_path = os.path.normpath(os.path.join(
            self.config.data_dir, f"self_play_epoch{epoch}.jsonl"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        trajectories = self.db.get_trajectories(
            min_reward=self.config.min_reward, limit=500)

        # Non-tool replay files — prevent catastrophic forgetting of direct Q&A
        # Reuses existing valid data files for compute efficiency.
        # Includes: code, reasoning, concise, self-correction, tool-use examples,
        # and expert training data (math, theory, creativity, general knowledge).
        non_tool_files = [
            # Core replay files (finetune dir)
            "code_300.jsonl",
            "short_cot_300.jsonl",
            "reasoning_500.jsonl",
            # Tool-use format examples (teaches correct tool call format)
            "tool_use_fc_70.jsonl",
            "tool_use_generalize_300.jsonl",
            # Expert training data (large, high-quality datasets)
            # Sampled at small ratios to avoid overwhelming self-play signal
        ]
        # Also check expert training dir for additional data
        expert_dir = os.path.normpath(os.path.join(
            self.config.data_dir, "..", "expert_training", "hf_datasets"))
        expert_files = []
        if os.path.isdir(expert_dir):
            expert_files = [
                (os.path.join(expert_dir, "math.jsonl"), 30),       # math problems
                (os.path.join(expert_dir, "python.jsonl"), 30),     # python code
                (os.path.join(expert_dir, "general.jsonl"), 20),    # general knowledge
                (os.path.join(expert_dir, "coding.jsonl"), 20),     # coding tasks
                (os.path.join(expert_dir, "algorithms.jsonl"), 15), # algorithms
            ]
        non_tool_examples = []
        for fname in non_tool_files:
            fpath = os.path.normpath(os.path.join(self.config.data_dir, fname))
            if not os.path.exists(fpath):
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    if "messages" in ex:
                        msgs = ex["messages"]
                        if msgs and msgs[0].get("role") != "system":
                            msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
                        non_tool_examples.append({"messages": msgs})
                    elif "prompt" in ex and "response" in ex:
                        msgs = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": ex["prompt"]},
                            {"role": "assistant", "content": ex["response"]},
                        ]
                        non_tool_examples.append({"messages": msgs})

        # Load expert training data with sampling (compute-efficient reuse)
        for epath, sample_n in expert_files:
            if not os.path.exists(epath):
                continue
            loaded = 0
            with open(epath, encoding="utf-8") as f:
                for line in f:
                    if loaded >= sample_n:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    if "messages" in ex:
                        msgs = ex["messages"]
                        if msgs and msgs[0].get("role") != "system":
                            msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
                        non_tool_examples.append({"messages": msgs})
                        loaded += 1
                    elif "prompt" in ex and "response" in ex:
                        msgs = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": ex["prompt"]},
                            {"role": "assistant", "content": ex["response"]},
                        ]
                        non_tool_examples.append({"messages": msgs})
                        loaded += 1

        # Mix: ~85% tool-use trajectories, ~15% non-tool replay.
        # IMPORTANT: the non-tool floor must stay SMALL. A floor of 30 meant
        # 21 tool-use + 30 non-tool = 59% direct-answer data, which trained
        # the model to STOP using tools entirely (epoch 3 collapsed to all
        # 0-tool tasks). Cap at 15% with a floor of 5 examples only.
        import random as _rng
        rng = _rng.Random(42 + epoch)
        # Include reward score for reward-weighted SFT training
        all_examples = []
        for t in trajectories:
            ex = {"messages": t["messages"]}
            if "reward" in t:
                ex["reward"] = t["reward"]
            all_examples.append(ex)
        # Cap non-tool at 15% of total (floor 5 to prevent forgetting Q&A)
        n_non_tool = min(len(non_tool_examples),
                         max(len(all_examples) * 15 // 100, 5))
        if n_non_tool > 0:
            rng.shuffle(non_tool_examples)
            all_examples.extend(non_tool_examples[:n_non_tool])
        rng.shuffle(all_examples)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  Exported {len(trajectories)} tool-use + {n_non_tool} non-tool = {len(all_examples)} total")
        return output_path

    def _run_self_play(self) -> tuple[dict, object]:
        """Phase 1: Run self-play session to collect trajectories.

        Returns (stats, engine) — engine is kept for reuse in eval phase.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 1: SELF-PLAY (epoch {self.epoch})")
        print(f"{'='*70}")

        sp_config = SelfPlayConfig(
            max_turns=self.config.max_turns,
            max_gen_tokens=self.config.max_gen_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            min_reward=self.config.min_reward,
            n_tasks=self.config.tasks_per_epoch,
            device=self.config.device,
        )

        loop = ToolUseSelfPlay.from_default_model(
            config=sp_config,
            checkpoint=self.best_checkpoint,
            db_path=str(self.db.path),
        )

        # ── Model-generated goals ──
        custom_tasks = None
        if self.config.use_generated_goals and loop.engine is not None:
            print(f"\n  [GoalGen] Generating {self.config.tasks_per_epoch} "
                  f"model-directed goals...")
            gen = GoalGenerator(loop.engine, loop.tokenizer, self.config)
            custom_tasks = gen.generate(self.config.tasks_per_epoch)
            n_gen = len(custom_tasks)
            if n_gen < self.config.tasks_per_epoch:
                # Top up with curriculum tasks
                deficit = self.config.tasks_per_epoch - n_gen
                sampled = self.curriculum.sample(deficit)
                custom_tasks.extend(task for _, task in sampled)
                print(f"  [GoalGen] Topped up with {deficit} curriculum tasks")
            print(f"  [GoalGen] {n_gen} model-generated + "
                  f"{self.config.tasks_per_epoch - n_gen} curriculum = "
                  f"{len(custom_tasks)} total tasks\n")
            # Print sample goals
            for i, g in enumerate(custom_tasks[:3]):
                print(f"    Goal {i+1}: {g[:100]}...")
            if len(custom_tasks) > 3:
                print(f"    ... and {len(custom_tasks) - 3} more")

        stats = loop.run_session(
            n_tasks=self.config.tasks_per_epoch,
            curriculum=self.curriculum if custom_tasks is None else None,
            custom_tasks=custom_tasks,
        )

        # Keep the engine for eval reuse; just free the loop wrapper
        engine = getattr(loop, 'engine', None)
        loop.engine = None  # prevent deletion
        del loop
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # RAM safeguard: check system memory after self-play
        try:
            import psutil
            ram_pct = psutil.virtual_memory().percent
            if ram_pct > 90:
                print(f"  [RAM] High memory usage ({ram_pct:.0f}%) after self-play; gc.collect()")
                gc.collect()
            if ram_pct > 95:
                print(f"  [RAM] Critical ({ram_pct:.0f}%) — clearing trajectory cache")
                self._seen_outputs = set() if hasattr(self, '_seen_outputs') else None
                gc.collect()
        except ImportError:
            pass

        return stats, engine

    def _finetune(self, data_path: str, epoch: int) -> str:
        """Phase 2: SFT continuation from best checkpoint."""
        print(f"\n{'='*70}")
        print(f"  PHASE 2: FINETUNE (epoch {self.epoch})")
        print(f"{'='*70}")

        save_path = self._epoch_checkpoint_path(epoch)

        # Set PYTHONPATH so the subprocess can find research.* modules
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))

        cmd = [
            os.sys.executable, "-m", "research.training.sft_train",
            "--data", data_path,
            "--checkpoint", self.best_checkpoint,
            "--save", save_path,
            "--max-steps", str(self.config.ft_max_steps),
            "--lr", str(self.config.ft_lr),
            "--batch-size", str(self.config.ft_batch_size),
            "--grad-accum", str(self.config.ft_grad_accum),
            "--seq-len", str(self.config.ft_seq_len),
            "--optimizer", self.config.ft_optimizer,
            "--entropy-alpha", str(self.config.ft_entropy_alpha),
            "--vram-limit-gb", str(self.config.ft_vram_limit_gb),
            "--ram-limit-percent", "90",  # RAM safeguard: throttle at 90%, abort at 95%+
        ]
        if self.config.ft_lora:
            cmd.extend(["--lora", "--lora-r", str(self.config.ft_lora_r),
                        "--lora-alpha", str(self.config.ft_lora_alpha)])
        if self.config.ft_grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if self.config.ft_compile:
            cmd.append("--compile")

        print(f"  Running: {' '.join(cmd)}")
        import subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        result = subprocess.run(cmd, env=env)
        exit_code = result.returncode
        if exit_code != 0 or not os.path.exists(save_path):
            raise RuntimeError(f"Finetune failed (exit code {exit_code})")

        return save_path

    def _evaluate(self, checkpoint: str, engine=None) -> dict:
        """Phase 3: Evaluate candidate against current best (in-process).

        If engine is provided (from self-play), reuses it to skip model reload (~40s saved).
        Uses weight swapping — loads one model, swaps state_dict.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 3: EVALUATE (fast in-process benchmark)")
        print(f"{'='*70}")

        from research.self_play.discovery.fast_eval import fast_eval

        try:
            results = fast_eval(
                base_checkpoint=self.best_checkpoint,
                candidate_checkpoint=checkpoint,
                device=self.config.device,
                engine=engine,
            )
            exit_code = 0
        except Exception as e:
            print(f"  Fast eval failed: {e}")
            import traceback; traceback.print_exc()
            results = {}
            exit_code = 1

        base_q = results.get("base", {}).get("quality", 0)
        cand_q = results.get("candidate", {}).get("quality", 0)
        cand_tokens = results.get("candidate", {}).get("tokens", 0)
        cand_time = results.get("candidate", {}).get("time_ms", 0)
        cand_vram = results.get("candidate", {}).get("vram_mb", 0)
        winner = results.get("winner", "BASE")

        # Don't promote if eval failed (no results)
        if exit_code != 0 or not results:
            passed = False
        elif winner == "CANDIDATE":
            passed = True
        elif cand_q > base_q:
            passed = True
        elif cand_q == base_q and cand_time < results.get("base", {}).get("time_ms", float("inf")):
            passed = True  # tie on quality, but faster → promote
        else:
            passed = False  # tie or worse → don't promote (avoid ratcheting)

        return {
            "exit_code": exit_code,
            "passed": passed,
            "base_quality": base_q,
            "candidate_quality": cand_q,
            "candidate_tokens": cand_tokens,
            "candidate_time_ms": cand_time,
            "candidate_vram_mb": cand_vram,
            "winner": winner,
        }

    def _maybe_promote(self, candidate: str, eval_result: dict) -> bool:
        """Phase 4: Promote candidate if it passes evaluation."""
        if eval_result["passed"]:
            self.best_checkpoint = candidate
            print(f"  PROMOTED: {candidate} is the new best checkpoint")
            return True
        else:
            # Archive the failed candidate
            archive_dir = os.path.join(self.config.checkpoint_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            archived = os.path.join(archive_dir, os.path.basename(candidate))
            if os.path.exists(candidate):
                shutil.move(candidate, archived)
                # Move meta too
                meta = candidate + ".meta.json"
                if os.path.exists(meta):
                    shutil.move(meta, archived + ".meta.json")
            print(f"  DEMOTED: reverted to {self.best_checkpoint}")
            return False

    def _run_grpo(self, epoch: int) -> str:
        """Phase 2b: GRPO training on tool-use trajectories.

        Collects G completions per task, computes tool-use rewards, and
        runs GRPO training steps. Saves the result as a new checkpoint.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 2b: GRPO (epoch {self.epoch})")
        print(f"{'='*70}")

        from research.model_loader import load_default_model
        from research.tokenizer_cache import get_tokenizer
        from research.self_play.grpo_trainer import GRPOTrainer, GRPOConfig
        from research.self_play.discovery.tool_use_loop import (
            ToolUseSelfPlay, SelfPlayConfig,
        )

        # Load policy + ref model
        model, tokenizer = load_default_model(
            "lfm25_1.2b", checkpoint_path=self.best_checkpoint,
            device=self.config.device, dtype=torch.bfloat16)
        model.train()

        ref_model, _ = load_default_model(
            "lfm25_1.2b", checkpoint_path=self.best_checkpoint,
            device=self.config.device, dtype=torch.bfloat16)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        grpo_config = GRPOConfig(
            learning_rate=self.config.grpo_lr,
            kl_coefficient=self.config.grpo_kl_coeff,
            group_size=self.config.grpo_group_size,
            temperature=self.config.temperature,
            use_tool_use_rewards=True,
        )
        trainer = GRPOTrainer(
            model, tokenizer, ref_model,
            device=self.config.device, config=grpo_config)

        # Self-play loop for trajectory collection
        sp_config = SelfPlayConfig(
            max_turns=self.config.max_turns,
            max_gen_tokens=self.config.max_gen_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            min_reward=self.config.min_reward,
            n_tasks=self.config.grpo_tasks_per_step,
            device=self.config.device,
        )
        sp_loop = ToolUseSelfPlay.from_default_model(
            config=sp_config, checkpoint=self.best_checkpoint,
            db_path=str(self.db.path))

        # Run GRPO steps
        for step in range(self.config.grpo_steps):
            # Sample tasks from curriculum
            sampled = self.curriculum.sample(self.config.grpo_tasks_per_step)
            tasks = [task for _, task in sampled]

            # Collect G completions per task with rewards
            batch = sp_loop.collect_grpo_batch(
                tasks, group_size=self.config.grpo_group_size)

            if not batch["prompts"]:
                continue

            # Run GRPO training step
            stats = trainer.train_step(
                prompts=batch["prompts"],
                completions=batch["completions"],
                rewards=batch["rewards"],
            )
            print(f"  GRPO step {step+1}/{self.config.grpo_steps}: "
                  f"loss={stats.get('loss', 0):.4f} "
                  f"reward={stats.get('mean_reward', 0):.3f} "
                  f"kl={stats.get('mean_kl', 0):.4f}", flush=True)

        # Save GRPO checkpoint
        save_path = self._epoch_checkpoint_path(epoch)
        from research.checkpoint_io import save_checkpoint
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_checkpoint(state, save_path)

        # Cleanup ref model from VRAM
        del ref_model, model, trainer
        torch.cuda.empty_cache()

        return save_path

    def _free_engine(self, engine):
        """Free a ForgeEngine from VRAM + clear model cache."""
        if engine is not None:
            del engine
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                from research.model_loader import ModelLoader
                ModelLoader.clear_cache()

    def run_epoch(self) -> dict:
        """Run one complete epoch: self-play → finetune/GRPO → eval → promote."""
        self.epoch += 1
        epoch_start = time.time()
        phases = {}

        # Phase 1: Self-play
        sp_engine = None
        try:
            sp_stats, sp_engine = self._run_self_play()
            phases["self_play"] = sp_stats
        except Exception as e:
            print(f"  Self-play failed: {e}")
            return {"epoch": self.epoch, "error": f"self_play: {e}"}

        # Phase 2: Export + Finetune (or GRPO)
        try:
            if self.config.use_grpo:
                # GRPO path (in-process, needs engine for eval)
                candidate = self._run_grpo(self.epoch)
                phases["grpo"] = {"checkpoint": candidate}

                # Phase 3: Evaluate (reuse self-play engine)
                eval_result = self._evaluate(candidate, engine=sp_engine)
                phases["evaluate"] = eval_result

                # Phase 4: Promote/Demote
                promoted = self._maybe_promote(candidate, eval_result)
                phases["promoted"] = promoted
            else:
                # SFT path — free engine BEFORE finetune subprocess so it has
                # the full VRAM budget. The subprocess runs in a separate
                # process and can't share the parent's VRAM allocation on
                # Windows. Cost: +40s for eval model reload (vs OOM crash).
                self._free_engine(sp_engine)
                sp_engine = None

                data_path = self._export_trajectories(self.epoch)
                n_examples = sum(1 for _ in open(data_path, encoding='utf-8'))
                print(f"  Exported {n_examples} trajectories to {data_path}")

                if n_examples < 8:
                    print(f"  Too few trajectories ({n_examples}), skipping finetune")
                    phases["finetune"] = {"skipped": True, "reason": "too_few_trajectories"}
                else:
                    candidate = self._finetune(data_path, self.epoch)
                    phases["finetune"] = {"checkpoint": candidate, "n_examples": n_examples}

                    # Phase 3: Evaluate (engine=None → fast_eval reloads model)
                    eval_result = self._evaluate(candidate, engine=None)
                    phases["evaluate"] = eval_result

                    # Phase 4: Promote/Demote
                    promoted = self._maybe_promote(candidate, eval_result)
                    phases["promoted"] = promoted
        except Exception as e:
            print(f"  Finetune/eval failed: {e}")
            phases["error"] = str(e)
        finally:
            # Ensure engine is freed even on exceptions.
            self._free_engine(sp_engine)

        elapsed = round(time.time() - epoch_start, 1)
        epoch_summary = {
            "epoch": self.epoch,
            "best_checkpoint": self.best_checkpoint,
            "elapsed_s": elapsed,
            "curriculum": self.curriculum.stats(),
            **phases,
        }
        self.history.append(epoch_summary)
        # Cap history at 100 epochs to prevent unbounded RAM growth
        if len(self.history) > 100:
            self.history = self.history[-100:]
        print(f"\n  Epoch {self.epoch} done in {elapsed}s")
        print(f"  Best checkpoint: {self.best_checkpoint}")
        return epoch_summary

    def run(self, max_epochs: int | None = None) -> list[dict]:
        """Run the infinite loop for max_epochs (or until interrupted)."""
        n = max_epochs or self.config.max_epochs
        print(f"\n{'#'*70}")
        print(f"#  INFINITE TOOL-USE SELF-PLAY LOOP")
        print(f"#  Starting checkpoint: {self.best_checkpoint}")
        print(f"#  Max epochs: {n}")
        print(f"#  Tasks per epoch: {self.config.tasks_per_epoch}")
        print(f"{'#'*70}")

        for _ in range(n):
            try:
                self.run_epoch()
            except KeyboardInterrupt:
                print(f"\n  Interrupted at epoch {self.epoch}")
                break
            except Exception as e:
                print(f"\n  Epoch {self.epoch} crashed: {e}")
                # Continue to next epoch — don't let one failure kill the loop

        self._print_summary()
        return self.history

    def _print_summary(self):
        print(f"\n{'#'*70}")
        print(f"#  LOOP COMPLETE — {self.epoch} epochs")
        print(f"#  Final best checkpoint: {self.best_checkpoint}")
        print(f"{'#'*70}")
        for h in self.history:
            ep = h["epoch"]
            sp = h.get("self_play", {})
            ft = h.get("finetune", {})
            prom = h.get("promoted", "—")
            avg_r = sp.get("avg_reward", "—") if isinstance(sp, dict) else "—"
            n_saved = sp.get("n_saved", "—") if isinstance(sp, dict) else "—"
            print(f"  Epoch {ep}: saved={n_saved} avg_reward={avg_r} "
                  f"promoted={prom}")
        print(f"\n  Curriculum stats: {self.curriculum.stats()}")


def main():
    parser = argparse.ArgumentParser(
        description="Infinite tool-use self-play training loop")
    parser.add_argument("--checkpoint", required=True,
                        help="Starting checkpoint (safetensors)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max epochs to run")
    parser.add_argument("--tasks-per-epoch", type=int, default=50,
                        help="Self-play tasks per epoch")
    parser.add_argument("--ft-steps", type=int, default=100,
                        help="Finetune steps per epoch (reduced: more trajectories compensate)")
    parser.add_argument("--ft-lr", type=float, default=2e-5,
                        help="Finetune learning rate")
    parser.add_argument("--ft-batch-size", type=int, default=1,
                        help="Finetune batch size (default 1, use grad-accum for effective batch)")
    parser.add_argument("--ft-grad-accum", type=int, default=4,
                        help="Finetune gradient accumulation steps")
    parser.add_argument("--ft-optimizer", type=str, default="bnb",
                        choices=["fused", "bnb", "lion"],
                        help="Finetune optimizer (bnb=8-bit AdamW, saves ~7GB VRAM)")
    parser.add_argument("--ft-lora", action="store_true", default=True,
                        help="Use LoRA for finetuning (default True, saves ~9GB VRAM)")
    parser.add_argument("--ft-no-lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--ft-grad-checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing (default True)")
    parser.add_argument("--ft-entropy-alpha", type=float, default=0.0,
                        help="Token entropy weighting alpha (default 0=disabled, saves 1GB)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Self-play exploration temperature")
    parser.add_argument("--top-k", type=int, default=80,
                        help="Top-k sampling (LFM2.5 default: 80)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Repetition penalty (LFM2.5 default: 1.05)")
    parser.add_argument("--min-reward", type=float, default=0.25,
                        help="Min reward to save trajectory")
    parser.add_argument("--grpo", action="store_true",
                        help="Use GRPO instead of SFT for training phase")
    parser.add_argument("--grpo-steps", type=int, default=20,
                        help="GRPO steps per epoch (if --grpo)")
    parser.add_argument("--grpo-group", type=int, default=4,
                        help="GRPO group size (completions per task)")
    parser.add_argument("--generated-goals", action="store_true",
                        help="Model generates its own task goals (filtered for quality)")
    args = parser.parse_args()

    config = LoopConfig(
        tasks_per_epoch=args.tasks_per_epoch,
        ft_max_steps=args.ft_steps,
        ft_lr=args.ft_lr,
        ft_batch_size=args.ft_batch_size,
        ft_grad_accum=args.ft_grad_accum,
        ft_optimizer=args.ft_optimizer,
        ft_lora=not args.ft_no_lora,
        ft_grad_checkpoint=args.ft_grad_checkpoint,
        ft_entropy_alpha=args.ft_entropy_alpha,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_reward=args.min_reward,
        max_epochs=args.epochs,
        use_grpo=args.grpo,
        grpo_steps=args.grpo_steps,
        grpo_group_size=args.grpo_group,
        use_generated_goals=args.generated_goals,
    )

    loop = InfiniteToolLoop(args.checkpoint, config)
    loop.run()


if __name__ == "__main__":
    main()
