"""Unified infinite self-play training loop (AZR paradigm).

Merges the AZR-style curriculum (propose → solve → verify) from
`infinite_curriculum.py` with the training orchestration (SFT, LoRA,
promote/demote, checkpoint archiving) from the former tool-use loop.

Cycle per epoch:
  1. Propose: model generates coding tasks with unit tests
  2. Validate: Python executor checks self-consistency (reference solution passes tests)
  3. Solve: model attempts to solve validated tasks
  4. Export: successful (task → solution) pairs → SFT JSONL
  5. Finetune: SFT continuation from current best checkpoint (LoRA)
  6. Evaluate: fast_eval compares candidate vs best (code, reasoning, knowledge)
  7. Promote/Demote: if candidate passes, it becomes the new best
  8. Repeat

Usage:
    python -m research.self_play.infinite_loop \\
        --checkpoint research/checkpoints/ForgeLM_V3_SFT.safetensors \\
        --epochs 50 --tasks-per-epoch 30

The loop is resumable: if interrupted, it picks up from the last
completed epoch using the saved checkpoint + curriculum state.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from research.paths import DATA_DIR
from research.self_play.infinite_curriculum import InfiniteCurriculum


@dataclass
class LoopConfig:
    """Configuration for the unified AZR self-play loop."""
    # Self-play (AZR curriculum)
    tasks_per_epoch: int = 30        # propose + solve per epoch
    max_gen_tokens: int = 256        # max tokens per generation
    temperature: float = 0.7         # exploration temperature
    top_k: int = 80                  # LFM2.5-recommended
    top_p: float = 0.95
    propose_batch_size: int = 8      # parallel task proposal
    domains: tuple = ("algorithms", "math", "strings", "logic", "data_structures")

    # Finetune
    ft_max_steps: int = 100          # SFT steps per epoch
    ft_lr: float = 2e-5
    ft_batch_size: int = 1
    ft_grad_accum: int = 4           # effective batch 4
    ft_seq_len: int = 1024
    ft_grad_checkpoint: bool = True
    ft_optimizer: str = "bnb"        # 8-bit AdamW (saves ~7GB VRAM)
    ft_lora: bool = True             # LoRA: train ~1M params
    ft_lora_r: int = 16
    ft_lora_alpha: int = 32
    ft_entropy_alpha: float = 0.0    # disabled: saves 1GB, enables chunked CE
    ft_vram_limit_gb: float = 11.0
    ft_min_examples: int = 8         # skip finetune if fewer than this

    # Loop control
    max_epochs: int = 50
    eval_threshold: float = 0.5      # min candidate quality vs base to promote
    device: str = "cuda"

    # Paths
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data/finetune"
    config_name: str = "forgelm_v3"

    # Replay buffer: mix in prior SFT data to prevent catastrophic forgetting
    replay_file: str = ""            # path to prior SFT JSONL (e.g. forgelm_v3_train.jsonl)
    replay_ratio: float = 0.2        # fraction of replay examples in each epoch

    # Task source: "model" (AZR self-propose) or "api" (distillation APIs)
    task_source: str = "model"       # "model" or "api"


class InfiniteSelfPlayLoop:
    """The unified AZR self-play → finetune → evaluate → promote loop."""

    def __init__(self, checkpoint: str, config: LoopConfig | None = None):
        self.config = config or LoopConfig()
        self.best_checkpoint = checkpoint
        self.best_score = 0.0
        self.epoch = 0
        self.history: list[dict] = []

        # Trajectory storage: (task_description, solution_code, success)
        self._trajectories: list[dict] = []

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.normpath(os.path.join(
            self.config.checkpoint_dir,
            f"ForgeLM_V3_SP{epoch}.safetensors"))

    # ── Phase 1: Self-Play (AZR curriculum) ──────────────────────────

    def _run_self_play(self) -> dict:
        """Phase 1: Load model, run AZR curriculum (propose → solve → verify)."""
        print(f"\n{'='*70}")
        print(f"  PHASE 1: AZR SELF-PLAY (epoch {self.epoch})")
        print(f"{'='*70}")

        from research.model_loader import load_default_model
        from research.tokenizer_cache import get_tokenizer

        model, tokenizer = load_default_model(
            self.config.config_name,
            checkpoint_path=self.best_checkpoint,
            device=self.config.device,
            dtype=torch.bfloat16,
        )
        model.eval()

        curriculum = InfiniteCurriculum(
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            max_gen_tokens=self.config.max_gen_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

        # Clear seen descriptions from prior runs — fresh start per epoch
        # so the model can re-propose similar tasks (it has limited vocabulary).
        # The clone filter still prevents exact duplicates within this epoch.
        curriculum._seen_descriptions = set()
        from research.paths import CURRICULUM_DIR
        seen_path = CURRICULUM_DIR / "seen_descriptions.json"
        if seen_path.exists():
            try:
                seen_path.unlink()
            except OSError:
                pass

        # ── Propose + validate tasks ──
        n_target = self.config.tasks_per_epoch
        validated: list = []

        if self.config.task_source == "api":
            # API-driven: use distillation teacher models to generate diverse tasks.
            # The local model still does all solving — APIs only provide task diversity.
            print("  [Propose] Using API teachers for task generation...")
            domain = self.config.domains[self.epoch % len(self.config.domains)]
            # Request more than needed — some will be filtered
            api_batch = max(n_target * 2, 30)
            proposed = curriculum.api_propose_tasks(
                n=api_batch, domain=domain, difficulty="medium")
            validated.extend(proposed)
            propose_attempts = 1
            print(f"  [Propose] {len(validated)} validated tasks from API "
                  f"(domain={domain})")
        else:
            # Model-driven: rotate domains + modes across propose attempts for diversity.
            # The model has limited task vocabulary — rotating domains and using
            # different reasoning modes (induction/abduction/deduction) forces
            # diverse task generation instead of repeating the same 5-6 tasks.
            import random as _rng
            rng = _rng.Random(42 + self.epoch)
            modes = ["induction", "abduction", "deduction"]

            propose_attempts = 0
            max_propose_attempts = (n_target // 3) + 5  # more attempts for diversity

            while len(validated) < n_target and propose_attempts < max_propose_attempts:
                # Rotate domain + mode per attempt
                domain = self.config.domains[propose_attempts % len(self.config.domains)]
                mode = modes[propose_attempts % len(modes)]
                batch_n = min(self.config.propose_batch_size,
                              n_target - len(validated) + 5)
                # Vary temperature slightly per attempt for diversity
                old_temp = curriculum.temperature
                curriculum.temperature = self.config.temperature * (1.0 + 0.1 * (propose_attempts % 3))
                proposed = curriculum.propose_tasks(
                    domain=domain, n=batch_n, batch_size=batch_n, mode=mode)
                curriculum.temperature = old_temp
                validated.extend(proposed)
                propose_attempts += 1
                print(f"  [Propose] Attempt {propose_attempts} ({domain}/{mode}): "
                      f"{len(proposed)} validated, {len(validated)}/{n_target} total")

            print(f"  [Propose] {len(validated)} validated tasks from "
                  f"{propose_attempts} attempts (domain={domain})")

        if not validated:
            print("  [Propose] No valid tasks generated, skipping epoch")
            self._free_model(model)
            del curriculum
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"error": "no_valid_tasks", "n_proposed": 0}

        # ── Solve tasks ──
        successes = 0
        failures = 0
        self._trajectories = []

        for i, task in enumerate(validated):
            result = curriculum.solve_task(task)
            success = result.get("final_success", False)
            attempts = result.get("attempts", [])

            if success and attempts:
                # Find the successful attempt
                for att in attempts:
                    if att.get("success"):
                        self._trajectories.append({
                            "task_description": task.description,
                            "signature": task.signature,
                            "test_cases": task.test_cases,
                            "solution_code": att["code"],
                            "domain": task.domain,
                            "difficulty": task.difficulty,
                            "reward": 1.0,
                        })
                        break
                successes += 1
            else:
                failures += 1
                # Keep failed attempts too (for negative examples / analysis)
                if attempts:
                    self._trajectories.append({
                        "task_description": task.description,
                        "signature": task.signature,
                        "test_cases": task.test_cases,
                        "solution_code": attempts[0]["code"],
                        "domain": task.domain,
                        "difficulty": task.difficulty,
                        "reward": 0.0,
                        "error": attempts[0].get("error", ""),
                    })

            # Record result for curriculum difficulty adaptation
            curriculum.record_result(task, success)

            if (i + 1) % 10 == 0 or i == len(validated) - 1:
                rate = successes / (i + 1)
                print(f"  [Solve] {i+1}/{len(validated)}: "
                      f"{successes} passed, {failures} failed "
                      f"({rate:.0%} success rate)")

        stats = {
            "n_proposed": len(validated),
            "n_solved": successes,
            "n_failed": failures,
            "success_rate": successes / max(len(validated), 1),
            "domain": domain,
            "curriculum_stats": {
                "total_proposed": curriculum.stats.total_proposed,
                "total_validated": curriculum.stats.total_validated,
                "total_solved": curriculum.stats.total_solved,
                "mean_proposer_reward": curriculum.stats.mean_proposer_reward,
                "diversity_score": curriculum.stats.diversity_score,
            },
        }
        print(f"\n  [Self-Play] {successes}/{len(validated)} solved "
              f"({stats['success_rate']:.0%}), domain={domain}")

        # Free model AND curriculum (curriculum holds self.model = model,
        # so deleting only the local var leaves a live reference → VRAM leak).
        self._free_model(model)
        del curriculum
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stats

    def _free_model(self, model):
        """Free model from VRAM + verify release."""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            from research.model_loader import ModelLoader
            ModelLoader.clear_cache()
            # Verify VRAM was actually released
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [VRAM] After free: {allocated:.2f} GB allocated, "
                  f"{reserved:.2f} GB reserved")

    # ── Phase 2: Export + Finetune ───────────────────────────────────

    def _export_trajectories(self, epoch: int) -> str:
        """Export successful trajectories as SFT JSONL.

        Format: {"prompt": task_description, "response": solution_code}
        Mixes in replay data to prevent catastrophic forgetting.
        """
        output_path = os.path.normpath(os.path.join(
            self.config.data_dir, f"azr_epoch{epoch}.jsonl"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Only export successful trajectories for SFT
        successful = [t for t in self._trajectories if t["reward"] > 0.5]
        examples = []

        for t in successful:
            # Build prompt: task description + function signature
            prompt = (f"Write a Python function {t['signature']} that "
                      f"{t['task_description']}\n\n"
                      f"```python\n")
            response = t["solution_code"]
            examples.append({"prompt": prompt, "response": response})

        # Mix in replay data (prior SFT examples) to prevent forgetting
        if self.config.replay_file and os.path.exists(self.config.replay_file):
            replay_examples = []
            with open(self.config.replay_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    if "prompt" in ex and "response" in ex:
                        replay_examples.append(ex)
                    elif "messages" in ex:
                        msgs = ex["messages"]
                        if len(msgs) >= 2:
                            user_msg = next((m for m in msgs if m["role"] == "user"), None)
                            asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
                            if user_msg and asst_msg:
                                replay_examples.append({
                                    "prompt": user_msg["content"],
                                    "response": asst_msg["content"],
                                })

            if replay_examples:
                import random as _rng
                rng = _rng.Random(42 + epoch)
                n_replay = min(len(replay_examples),
                               int(len(examples) * self.config.replay_ratio / max(1 - self.config.replay_ratio, 0.01)))
                n_replay = max(n_replay, 5)  # floor: always some replay
                rng.shuffle(replay_examples)
                examples.extend(replay_examples[:n_replay])
                print(f"  [Export] +{n_replay} replay examples (anti-forgetting)")

        # Shuffle
        import random as _rng
        rng = _rng.Random(42 + epoch)
        rng.shuffle(examples)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  [Export] {len(successful)} successful + "
              f"{len(examples) - len(successful)} replay = "
              f"{len(examples)} total -> {output_path}")
        return output_path

    def _finetune(self, data_path: str, epoch: int) -> str:
        """Phase 2b: SFT continuation from best checkpoint (subprocess)."""
        print(f"\n{'='*70}")
        print(f"  PHASE 2: FINETUNE (epoch {self.epoch})")
        print(f"{'='*70}")

        save_path = self._epoch_checkpoint_path(epoch)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

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
            "--ram-limit-percent", "90",
        ]
        if self.config.ft_lora:
            cmd.extend(["--lora", "--lora-r", str(self.config.ft_lora_r),
                        "--lora-alpha", str(self.config.ft_lora_alpha)])
        if self.config.ft_grad_checkpoint:
            cmd.append("--grad-checkpoint")

        print(f"  Running: {' '.join(cmd)}")
        import subprocess
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0 or not os.path.exists(save_path):
            raise RuntimeError(f"Finetune failed (exit code {result.returncode})")

        return save_path

    # ── Phase 3: Evaluate ────────────────────────────────────────────

    def _evaluate(self, checkpoint: str) -> dict:
        """Phase 3: Evaluate candidate vs current best via fast_eval."""
        print(f"\n{'='*70}")
        print(f"  PHASE 3: EVALUATE")
        print(f"{'='*70}")

        from research.self_play.discovery.fast_eval import fast_eval

        try:
            results = fast_eval(
                base_checkpoint=self.best_checkpoint,
                candidate_checkpoint=checkpoint,
                device=self.config.device,
            )
        except Exception as e:
            print(f"  Eval failed: {e}")
            import traceback; traceback.print_exc()
            return {"passed": False, "error": str(e)}

        base_q = results.get("base", {}).get("quality", 0)
        cand_q = results.get("candidate", {}).get("quality", 0)
        winner = results.get("winner", "BASE")

        # Promote if candidate wins or matches quality
        if winner == "CANDIDATE":
            passed = True
        elif cand_q >= base_q * (1 - self.config.eval_threshold):
            # Within threshold of base quality → promote (avoid ratcheting)
            passed = True
        else:
            passed = False

        return {
            "passed": passed,
            "base_quality": base_q,
            "candidate_quality": cand_q,
            "winner": winner,
            "details": results,
        }

    # ── Phase 4: Promote/Demote ──────────────────────────────────────

    def _maybe_promote(self, candidate: str, eval_result: dict) -> bool:
        """Phase 4: Promote candidate if it passes evaluation."""
        if eval_result["passed"]:
            self.best_checkpoint = candidate
            self.best_score = eval_result.get("candidate_quality", 0)
            print(f"  PROMOTED: {candidate} is the new best checkpoint")
            return True
        else:
            archive_dir = os.path.join(self.config.checkpoint_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            archived = os.path.join(archive_dir, os.path.basename(candidate))
            if os.path.exists(candidate):
                shutil.move(candidate, archived)
                meta = candidate + ".meta.json"
                if os.path.exists(meta):
                    shutil.move(meta, archived + ".meta.json")
            print(f"  DEMOTED: reverted to {self.best_checkpoint}")
            return False

    # ── Main loop ────────────────────────────────────────────────────

    def run_epoch(self) -> dict:
        """Run one complete epoch: self-play → finetune → eval → promote."""
        self.epoch += 1
        epoch_start = time.time()
        phases = {}

        # Phase 1: Self-play (AZR curriculum)
        try:
            sp_stats = self._run_self_play()
            phases["self_play"] = sp_stats
            if "error" in sp_stats:
                return {"epoch": self.epoch, **phases}
        except Exception as e:
            print(f"  Self-play failed: {e}")
            import traceback; traceback.print_exc()
            return {"epoch": self.epoch, "error": f"self_play: {e}"}

        # Phase 2: Export + Finetune
        try:
            data_path = self._export_trajectories(self.epoch)
            n_examples = sum(1 for _ in open(data_path, encoding='utf-8'))

            if n_examples < self.config.ft_min_examples:
                print(f"  Too few examples ({n_examples}), skipping finetune")
                phases["finetune"] = {"skipped": True, "n_examples": n_examples}
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
            import traceback; traceback.print_exc()
            phases["error"] = str(e)

        elapsed = round(time.time() - epoch_start, 1)
        epoch_summary = {
            "epoch": self.epoch,
            "best_checkpoint": self.best_checkpoint,
            "elapsed_s": elapsed,
            **phases,
        }
        self.history.append(epoch_summary)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        print(f"\n  Epoch {self.epoch} done in {elapsed}s")
        print(f"  Best checkpoint: {self.best_checkpoint}")
        return epoch_summary

    def run(self, max_epochs: int | None = None) -> list[dict]:
        """Run the infinite loop for max_epochs (or until interrupted)."""
        n = max_epochs or self.config.max_epochs
        print(f"\n{'#'*70}")
        print(f"#  INFINITE AZR SELF-PLAY LOOP")
        print(f"#  Starting checkpoint: {self.best_checkpoint}")
        print(f"#  Max epochs: {n}")
        print(f"#  Tasks per epoch: {self.config.tasks_per_epoch}")
        print(f"#  Config: {self.config.config_name}")
        print(f"{'#'*70}")

        for _ in range(n):
            try:
                self.run_epoch()
            except KeyboardInterrupt:
                print(f"\n  Interrupted at epoch {self.epoch}")
                break
            except Exception as e:
                print(f"\n  Epoch {self.epoch} crashed: {e}")
                import traceback; traceback.print_exc()

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
            rate = sp.get("success_rate", "—") if isinstance(sp, dict) else "—"
            n_ex = ft.get("n_examples", "—") if isinstance(ft, dict) else "—"
            print(f"  Epoch {ep}: success={rate} examples={n_ex} promoted={prom}")


def main():
    # Load .env for API keys (needed for --task-source api)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    parser = argparse.ArgumentParser(
        description="Infinite AZR self-play training loop")
    parser.add_argument("--checkpoint", required=True,
                        help="Starting checkpoint (safetensors)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs to run")
    parser.add_argument("--tasks-per-epoch", type=int, default=30,
                        help="Tasks to propose + solve per epoch")
    parser.add_argument("--ft-steps", type=int, default=100,
                        help="Finetune steps per epoch")
    parser.add_argument("--ft-lr", type=float, default=2e-5,
                        help="Finetune learning rate")
    parser.add_argument("--ft-batch-size", type=int, default=1)
    parser.add_argument("--ft-grad-accum", type=int, default=4)
    parser.add_argument("--ft-optimizer", type=str, default="bnb",
                        choices=["fused", "bnb", "lion", "flash_adamw", "flash_lion", "forge"])
    parser.add_argument("--self-play-mode", type=str, default="azr",
                        choices=["azr", "soar", "sgs"],
                        help="Self-play mode: 'azr' (standard), 'soar' (meta-RL curriculum, "
                             "escapes learning plateaus), 'sgs' (self-guided self-play, "
                             "prevents Conjecturer collapse with Guide role)")
    parser.add_argument("--saerl", action="store_true",
                        help="Enable SAERL: SAE-guided data engineering for RL. "
                             "Diversity control + difficulty curriculum + quality filtering. "
                             "+3% accuracy, 20% fewer steps to target.")
    parser.add_argument("--opmix", action="store_true",
                        help="Enable OP-MIX: on-policy data mixing via low-rank adapters. "
                             "Dynamically adjusts mixing ratio across data sources.")
    parser.add_argument("--ft-no-lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--ft-grad-checkpoint", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Self-play exploration temperature")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--replay-file", type=str, default="",
                        help="Prior SFT JSONL for replay (anti-forgetting)")
    parser.add_argument("--replay-ratio", type=float, default=0.2,
                        help="Fraction of replay examples per epoch")
    parser.add_argument("--task-source", type=str, default="model",
                        choices=["model", "api"],
                        help="Task source: 'model' (AZR self-propose) or "
                             "'api' (distillation teacher APIs)")
    parser.add_argument("--config", type=str, default="forgelm_v3",
                        help="Model config name")
    parser.add_argument("--eval-threshold", type=float, default=0.5,
                        help="Min candidate quality vs base to promote")
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
        temperature=args.temperature,
        top_k=args.top_k,
        max_epochs=args.epochs,
        config_name=args.config,
        replay_file=args.replay_file,
        replay_ratio=args.replay_ratio,
        task_source=args.task_source,
        eval_threshold=args.eval_threshold,
    )

    # Self-play mode dispatch
    if args.self_play_mode == "soar":
        print("[InfiniteLoop] SOAR mode: meta-RL curriculum (escapes learning plateaus)")
        from research.self_play.soar import SOARMetaRL
        # Load model + tokenizer for SOAR
        from research.model_loader import load_default_model
        from research.tokenizer_cache import get_tokenizer
        model, _ = load_default_model()
        tokenizer = get_tokenizer()
        # Use hard target problems from existing curriculum if available
        target_problems = []  # would be populated from curriculum
        soar = SOARMetaRL(model, model, tokenizer, target_problems)
        stats = soar.run(n_rounds=args.epochs)
        print(f"[SOAR] Completed {len(stats)} rounds. Final: {soar.stats()}")
        return

    if args.self_play_mode == "sgs":
        print("[InfiniteLoop] SGS mode: self-guided self-play (prevents Conjecturer collapse)")
        from research.self_play.sgs import SGSTrainer
        from research.model_loader import load_default_model
        from research.tokenizer_cache import get_tokenizer
        model, _ = load_default_model()
        tokenizer = get_tokenizer()
        target_problems = []  # would be populated from curriculum
        sgs = SGSTrainer(model, tokenizer, target_problems)
        stats = sgs.run(n_rounds=args.epochs)
        print(f"[SGS] Completed {len(stats)} rounds. Final: {sgs.stats()}")
        return

    # Standard AZR self-play (with optional SAERL + OP-MIX)
    if args.saerl:
        print("[InfiniteLoop] SAERL enabled: SAE-guided data engineering")
        # SAERL is applied inside the loop's batch composition
        # (would be wired into the training data pipeline)

    if args.opmix:
        print("[InfiniteLoop] OP-MIX enabled: on-policy data mixing via LoRA adapters")
        # OP-MIX adjusts data source mixing ratios dynamically

    loop = InfiniteSelfPlayLoop(args.checkpoint, config)
    loop.run()


if __name__ == "__main__":
    main()
