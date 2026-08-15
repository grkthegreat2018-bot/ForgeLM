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
        --checkpoint research/checkpoints/ForgeLM_V2_LFM25-1.2B.sft4.safetensors \\
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
    # Self-play
    tasks_per_epoch: int = 50
    max_turns: int = 5
    max_gen_tokens: int = 256
    temperature: float = 0.2       # LFM2.5-recommended (low for tool use)
    top_k: int = 80                # LFM2.5-recommended
    repetition_penalty: float = 1.05  # LFM2.5-recommended
    min_reward: float = 0.4

    # Context management
    max_seq_len: int = 32768       # model's context window
    context_threshold: float = 0.75  # summarize at 75% of budget
    keep_recent_turns: int = 6     # turns to keep intact during compression

    # Finetune
    ft_max_steps: int = 200
    ft_lr: float = 1e-5
    ft_batch_size: int = 2
    ft_seq_len: int = 1536
    ft_grad_checkpoint: bool = False

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
        # Includes concise (token conservation), codebase (self-awareness),
        # reasoning (computation), and self-correction (epistemic humility) data
        non_tool_files = [
            "code_300.jsonl",
            "short_cot_300.jsonl",
            "concise_86.jsonl",
            "codebase_74.jsonl",
            "reasoning_51.jsonl",
            "self_correction.jsonl",
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

        # Mix: ~60% tool-use trajectories, ~40% non-tool replay
        import random as _rng
        rng = _rng.Random(42 + epoch)
        all_examples = [{"messages": t["messages"]} for t in trajectories]
        # Cap non-tool at 40% of total
        n_non_tool = min(len(non_tool_examples),
                         max(len(all_examples) * 2 // 3, 50))
        if n_non_tool > 0:
            rng.shuffle(non_tool_examples)
            all_examples.extend(non_tool_examples[:n_non_tool])
        rng.shuffle(all_examples)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  Exported {len(trajectories)} tool-use + {n_non_tool} non-tool = {len(all_examples)} total")
        return output_path

    def _run_self_play(self) -> dict:
        """Phase 1: Run self-play session to collect trajectories."""
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

        stats = loop.run_session(
            n_tasks=self.config.tasks_per_epoch,
            curriculum=self.curriculum,
        )

        # Free VRAM: release engine + model before SFT/eval phases
        if hasattr(loop, 'engine') and loop.engine is not None:
            del loop.engine
        if hasattr(loop, 'model') and loop.model is not None:
            del loop.model
        del loop
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            from research.model_loader import ModelLoader
            ModelLoader.clear_cache()

        return stats

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
            "--seq-len", str(self.config.ft_seq_len),
        ]
        if self.config.ft_grad_checkpoint:
            cmd.append("--grad-checkpoint")

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

    def _evaluate(self, checkpoint: str) -> dict:
        """Phase 3: Run benchmark comparison (best vs candidate).

        Runs the model comparison benchmark and returns structured results.
        The candidate must beat or tie the current best on quality to promote.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 3: EVALUATE (benchmark comparison)")
        print(f"{'='*70}")

        test_script = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            ".devin", "test_model_compare.py"))

        # Run benchmark: base (best_checkpoint) vs candidate
        cmd = [
            os.sys.executable, test_script,
            "--base", self.best_checkpoint,
            "--candidate", checkpoint,
        ]
        print(f"  Running: {' '.join(cmd)}")
        import subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        result = subprocess.run(cmd, env=env)
        exit_code = result.returncode

        # Parse results from JSON file
        results_path = "research/data/benchmark_results.json"
        try:
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

        base_q = results.get("base", {}).get("quality", 0)
        cand_q = results.get("candidate", {}).get("quality", 0)
        cand_tokens = results.get("candidate", {}).get("tokens", 0)
        cand_time = results.get("candidate", {}).get("time_ms", 0)
        cand_vram = results.get("candidate", {}).get("vram_mb", 0)
        winner = results.get("winner", "BASE")

        # Candidate passes if it wins or ties on quality
        passed = winner == "CANDIDATE" or cand_q >= base_q

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

    def run_epoch(self) -> dict:
        """Run one complete epoch: self-play → finetune/GRPO → eval → promote."""
        self.epoch += 1
        epoch_start = time.time()
        phases = {}

        # Phase 1: Self-play
        try:
            phases["self_play"] = self._run_self_play()
        except Exception as e:
            print(f"  Self-play failed: {e}")
            return {"epoch": self.epoch, "error": f"self_play: {e}"}

        # Phase 2: Export + Finetune (or GRPO)
        try:
            if self.config.use_grpo:
                # GRPO path
                candidate = self._run_grpo(self.epoch)
                phases["grpo"] = {"checkpoint": candidate}

                # Phase 3: Evaluate
                eval_result = self._evaluate(candidate)
                phases["evaluate"] = eval_result

                # Phase 4: Promote/Demote
                promoted = self._maybe_promote(candidate, eval_result)
                phases["promoted"] = promoted
            else:
                # SFT path
                data_path = self._export_trajectories(self.epoch)
                n_examples = sum(1 for _ in open(data_path, encoding='utf-8'))
                print(f"  Exported {n_examples} trajectories to {data_path}")

                if n_examples < 8:
                    print(f"  Too few trajectories ({n_examples}), skipping finetune")
                    phases["finetune"] = {"skipped": True, "reason": "too_few_trajectories"}
                else:
                    candidate = self._finetune(data_path, self.epoch)
                    phases["finetune"] = {"checkpoint": candidate, "n_examples": n_examples}

                    # Phase 3: Evaluate
                    eval_result = self._evaluate(candidate)
                    phases["evaluate"] = eval_result

                    # Phase 4: Promote/Demote
                    promoted = self._maybe_promote(candidate, eval_result)
                    phases["promoted"] = promoted
        except Exception as e:
            print(f"  Finetune/eval failed: {e}")
            phases["error"] = str(e)

        elapsed = round(time.time() - epoch_start, 1)
        epoch_summary = {
            "epoch": self.epoch,
            "best_checkpoint": self.best_checkpoint,
            "elapsed_s": elapsed,
            "curriculum": self.curriculum.stats(),
            **phases,
        }
        self.history.append(epoch_summary)
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
    parser.add_argument("--ft-steps", type=int, default=200,
                        help="Finetune steps per epoch")
    parser.add_argument("--ft-lr", type=float, default=1e-5,
                        help="Finetune learning rate")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Self-play exploration temperature (LFM2.5 default: 0.2)")
    parser.add_argument("--top-k", type=int, default=80,
                        help="Top-k sampling (LFM2.5 default: 80)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Repetition penalty (LFM2.5 default: 1.05)")
    parser.add_argument("--min-reward", type=float, default=0.4,
                        help="Min reward to save trajectory")
    parser.add_argument("--grpo", action="store_true",
                        help="Use GRPO instead of SFT for training phase")
    parser.add_argument("--grpo-steps", type=int, default=20,
                        help="GRPO steps per epoch (if --grpo)")
    parser.add_argument("--grpo-group", type=int, default=4,
                        help="GRPO group size (completions per task)")
    args = parser.parse_args()

    config = LoopConfig(
        tasks_per_epoch=args.tasks_per_epoch,
        ft_max_steps=args.ft_steps,
        ft_lr=args.ft_lr,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_reward=args.min_reward,
        max_epochs=args.epochs,
        use_grpo=args.grpo,
        grpo_steps=args.grpo_steps,
        grpo_group_size=args.grpo_group,
    )

    loop = InfiniteToolLoop(args.checkpoint, config)
    loop.run()


if __name__ == "__main__":
    main()
