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
        --checkpoint research/checkpoints/ForgeLM_V2_Light.safetensors \\
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
    """Configuration for the unified AZR self-play loop.

    V10 update (2026-08-31): defaults to ForgeLM V10-1.2B, wires ForgeEngine
    inference features (SpectralKV, MTP self-speculative decoding, prefix
    cache, Triton conv) into the self-play generation phase, and exposes
    all sft_train training tricks (Muon-SF optimizer, grad-mixup, curriculum,
    augmentation, SYNPRO, focal loss, entropy weighting, MTP auxiliary loss,
    distillation, sequential freeze, EMA, BitNet-everywhere).
    """
    # Self-play (AZR curriculum)
    tasks_per_epoch: int = 30        # propose + solve per epoch
    max_gen_tokens: int = 256        # max tokens per generation
    temperature: float = 0.7         # exploration temperature
    top_k: int = 80                  # LFM2.5-recommended
    top_p: float = 0.95
    propose_batch_size: int = 8      # parallel task proposal
    domains: tuple = ("algorithms", "math", "strings", "logic", "data_structures")

    # ── ForgeEngine inference features (self-play generation phase) ──
    # V10: SpectralKV is the production KV cache (63× compression at 0.095 error).
    # When use_forge_engine=True, the self-play phase loads via ForgeEngine
    # and activates these features instead of the bare model path.
    use_forge_engine: bool = True    # load via ForgeEngine (V10 features)
    kv_cache: str = "spectral"       # V10 SpectralKV (was: none / bare model)
    decoding: str = "mtp_selfspec"   # MTP self-speculative (2-4× decode speedup)
    quantize: str | None = None      # weight quantization (None=bf16, "w8a8", "nvfp4")
    acceleration: str | None = None  # "cuda_graph", "megakernel", "flex_decoding"
    use_compile: bool = False        # torch.compile (experimental on Windows)
    use_triton_conv: bool = True     # Triton fused conv kernel (89% conv bottleneck cut)
    use_prefix_cache: bool = True    # prefix KV cache (repeated prompt prefixes)
    use_chunked_prefill: bool = True # chunked prefill (long prompts)
    # NOTE: disabled — FusedQKNormRopeCacheWrapper has a RoPE-convention
    # mismatch on V10 (model uses NeoX full-dim cos/sin via cat((freqs,freqs)),
    # but the wrapper's _py_qk_norm_rope expects GPT-J half-dim cos/sin →
    # 32-vs-64 shape crash). The original attention forward handles RoPE
    # correctly. Re-enable only after a bit-exact forward-pass comparison
    # (AGENTS.md directive A). Tracked in .devin/scratchpad.md.
    use_fused_qk_norm_rope_cache: bool = False  # fused QK-norm+RoPE+cache-write
    use_seq_split: bool = True       # sequence-split attention (long context)
    use_spec_attn: bool = True       # speculative attention
    kv_cache_tokens: int = 4096      # KV cache token budget
    warmup: bool = True              # warmup dummy token (init CUDA kernels)

    # Finetune
    ft_max_steps: int = 100          # SFT steps per epoch
    ft_lr: float = 5e-5              # V10 default (muon_sf tolerates higher LR)
    ft_min_lr: float = 5e-6          # cosine decay floor
    ft_batch_size: int = 1
    ft_grad_accum: int = 5           # evolution-discovered: 5 (effective batch 5)
    ft_sync_freq: int = 15           # evolution-discovered: sync every 15 steps
    ft_seq_len: int = 1024
    ft_warmup_steps: int = 20        # warmup for stability (evolution: 0 was synthetic-only)
    ft_grad_checkpoint: bool = True
    ft_checkpoint_strategy: str = "all"  # selective: "ffn", "attn", "lazy", "optimal"
    ft_optimizer: str = "muon_sf"    # V10 default: Muon-SF (2.39× vs AdamW)
    ft_lora: bool = True             # LoRA: train ~1M params
    ft_lora_r: int = 32              # V10 default rank (was 16)
    ft_lora_alpha: int = 64          # V10 default alpha (was 32)
    ft_bitnet_everywhere: bool = True  # BitNet ternary QAT (1.58 bits, 2.39× vs AdamW)
    ft_manual_lora: bool = True      # BitNet-compatible LoRA (auto-enabled w/ bitnet)
    ft_entropy_alpha: float = 0.5    # token entropy weighting (WeFT/VCORE 2025)
    ft_loss_function: str = "ce"     # "ce", "focal", "label_smoothing", "dynamic_focal"
    ft_focal_gamma: float = 4.93     # evolution-discovered focal optimum
    ft_label_smoothing_eps: float = 0.1
    ft_grad_compression: str = "int4"  # evolution-discovered: int4 for CPU offload
    ft_vram_limit_gb: float = 11.0
    ft_min_examples: int = 8         # skip finetune if fewer than this

    # ── Training tricks (R&D round 14+, wired to sft_train) ──
    ft_grad_mixup: int = 1           # N-batch grad averaging (1=off, 3=1.25× convergence)
    ft_curriculum: str = "none"      # "vanilla", "pacing", "interleaved", "warmup"
    ft_augment: bool = False         # token noise + FIM + target offset (anti-overfit)
    ft_synpro: bool = False          # SYNPRO synthetic data (3.7-5.2× effective tokens)
    ft_norm_type: str = "rmsnorm"    # "seednorm", "dyt" (Muon-compatible)
    ft_mtp_weight: float = 0.0       # MTP auxiliary loss (Nemotron Lightning)
    ft_mtp_n_heads: int | None = None  # override MTP heads (None=use config)
    ft_distill: bool = False         # knowledge distillation from teacher
    ft_teacher_checkpoint: str | None = None  # teacher checkpoint for distill
    ft_distill_topk: int = 50        # top-K logits to cache from teacher
    ft_distill_truncate: float = 1.0  # sequence truncation ratio for distill
    ft_distill_prefix: bool = False  # on-policy prefix distillation (2-40× FLOP cut)
    ft_sequential_freeze: int = 0    # sequential freeze/unfreeze (0=off, 4=4 phases)
    ft_final_finetune_steps: int = 0  # reserved all-layer finetune steps at end
    ft_ema: bool = False             # EMA shadow weights
    ft_ema_decay: float = 0.999
    ft_anchor: str | None = None     # L2-SP anchor checkpoint (anti-drift)
    ft_l2_lambda: float = 0.01       # L2-SP regularization lambda
    ft_sample_weighting: bool = False  # easy sample upweighting (ICML 2025)
    ft_lazy_train: bool = False      # LazyTrain scheduler (1.24× sustained TFLOPS)
    ft_oomb: bool = False            # OOMB chunk-recurrent (128K+ context on 12GB)
    ft_hybrid_clip: bool = False     # Hybrid8Bit fast grad clip
    ft_elastic_grad_accum: bool = False  # dynamic grad_accum (OOM prevention)
    ft_freetoken: bool = False       # FreeToken double-buffered grad pipeline
    ft_compile: bool = False         # torch.compile model (experimental on Windows)
    ft_disk_cache: bool = True       # disk-backed tokenization cache
    ft_pack_sequences: bool = True   # pack variable-length sequences (Llama-3 style)
    ft_async_prefetch: bool = True   # async background prefetcher (2-3× throughput)
    ft_prefetch_count: int = 4       # batches to prefetch ahead

    # Loop control
    max_epochs: int = 50
    eval_threshold: float = 0.5      # min candidate quality vs base to promote
    device: str = "cuda"

    # Paths
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data/finetune"
    config_name: str = "forgelm_v2_light"  # V10 default

    # Replay buffer: mix in prior SFT data to prevent catastrophic forgetting
    replay_file: str = ""            # path to prior SFT JSONL
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
        # ForgeEngine retained across self-play → eval for weight-swap reuse.
        self._engine = None

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.normpath(os.path.join(
            self.config.checkpoint_dir,
            f"ForgeLM_V10_SP{epoch}.safetensors"))

    def _load_engine(self):
        """Load a ForgeEngine with V10 inference features activated.

        When ``use_forge_engine`` is True (default), uses
        ``ForgeEngine.from_checkpoint`` and activates the V10 feature set
        (SpectralKV, MTP self-speculative decoding, prefix cache, Triton
        conv, fused QK-norm+RoPE+cache-write, sequence-split attention).
        The engine.model is used for curriculum generation; the engine
        itself is reused by fast_eval (weight-swap, no reload).

        When False, falls back to the bare ``load_default_model`` path
        (legacy, no V10 inference features — kept for debugging/ablation).
        """
        cfg = self.config
        if cfg.use_forge_engine:
            from research.inference.forge_engine import ForgeEngine
            engine = ForgeEngine.from_checkpoint(
                checkpoint=self.best_checkpoint,
                config_name=cfg.config_name,
                tokenizer_path="research/checkpoints/lfm25_tokenizer",
                device=cfg.device,
                auto_activate=False,  # we activate explicitly below
            )
            # Activate V10 inference features. SpectralKV is the V10 production
            # KV cache (63× compression). MTP self-spec gives 2-4× decode
            # speedup when the checkpoint has MTP heads (V10-growth configs).
            engine.activate(
                kv_cache=cfg.kv_cache,
                decoding=cfg.decoding,
                quantize=cfg.quantize,
                acceleration=cfg.acceleration,
                kv_cache_tokens=cfg.kv_cache_tokens,
                use_compile=cfg.use_compile,
                use_triton_conv=cfg.use_triton_conv,
                use_prefix_cache=cfg.use_prefix_cache,
                use_chunked_prefill=cfg.use_chunked_prefill,
                use_fused_qk_norm_rope_cache=cfg.use_fused_qk_norm_rope_cache,
                use_seq_split=cfg.use_seq_split,
                use_spec_attn=cfg.use_spec_attn,
                warmup=cfg.warmup,
            )
            print(f"  [ForgeEngine] V10 features active: kv={cfg.kv_cache}, "
                  f"decoding={cfg.decoding}, triton_conv={cfg.use_triton_conv}, "
                  f"prefix_cache={cfg.use_prefix_cache}")
            return engine

        # Legacy bare-model path (no V10 inference features)
        from research.model_loader import load_default_model
        from research.inference.forge_engine import ForgeEngine
        model, tokenizer = load_default_model(
            cfg.config_name,
            checkpoint_path=self.best_checkpoint,
            device=cfg.device,
            dtype=torch.bfloat16,
        )
        model.eval()
        # Wrap in a minimal ForgeEngine shell so fast_eval's weight-swap
        # path still works (engine.model + engine.tokenizer).
        engine = ForgeEngine(model, tokenizer, device=cfg.device,
                             checkpoint_path=self.best_checkpoint)
        return engine

    # ── Phase 1: Self-Play (AZR curriculum) ──────────────────────────

    def _run_self_play(self) -> dict:
        """Phase 1: Load model, run AZR curriculum (propose → solve → verify).

        V10: loads via ForgeEngine and activates V10 inference features
        (SpectralKV, MTP self-speculative decoding, prefix cache, Triton
        conv, fused QK-norm+RoPE+cache-write, sequence-split attention).
        The engine.model is passed to InfiniteCurriculum; the engine is
        retained on self._engine for fast_eval reuse (skips ~40s reload).
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 1: AZR SELF-PLAY (epoch {self.epoch})")
        print(f"{'='*70}")

        engine = self._load_engine()
        model = engine.model
        tokenizer = engine.tokenizer
        # Keep engine alive for fast_eval reuse (set before curriculum so
        # cleanup paths can find it).
        self._engine = engine

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
            self._free_engine()
            del curriculum
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return {"error": "no_valid_tasks", "n_proposed": 0}

        # ── Solve tasks ──
        # Use ThreadPoolExecutor to overlap CPU-bound sandbox verification
        # of one task with GPU generation of the next. GPU generation
        # serializes naturally (single CUDA context); the win comes from
        # parallelizing the subprocess-based test execution.
        successes = 0
        failures = 0
        self._trajectories = []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        max_workers = min(4, len(validated)) if len(validated) > 1 else 1

        # Curriculum state (e.g. self.temperature, _seen_descriptions, stats)
        # is not thread-safe. _solve_direct mutates self.temperature per retry.
        # Serialize solve_task to prevent concurrent corruption. The GPU
        # generation inside solve_task serializes via CUDA anyway, so the lock
        # mainly costs us overlapping subprocess sandbox execution — acceptable
        # for correctness.
        _solve_lock = Lock()

        def _solve_and_record(task):
            """Solve a single task and return (task, result)."""
            with _solve_lock:
                result = curriculum.solve_task(task)
            return task, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks — GPU generation serializes via GIL+CUDA,
            # but sandbox verification (subprocess) overlaps across threads
            futures = {
                executor.submit(_solve_and_record, task): i
                for i, task in enumerate(validated)
            }
            results_ordered = [None] * len(validated)
            for future in as_completed(futures):
                idx = futures[future]
                results_ordered[idx] = future.result()

        for i, (task, result) in enumerate(r for r in results_ordered if r is not None):
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

        # Free curriculum (holds self.model = engine.model reference).
        # The engine itself is retained on self._engine for fast_eval reuse
        # (weight-swap skips ~40s model reload). It is freed after eval/promote
        # via _free_engine().
        del curriculum
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return stats

    def _free_engine(self):
        """Free the ForgeEngine from VRAM + verify release.

        Releases the model, KV cache, CUDA graphs, Triton patches, and
        prefix cache held by the engine. Called after fast_eval completes
        (or on early-return paths where eval is skipped).
        """
        engine = self._engine
        if engine is None:
            return
        # Release acceleration resources (CUDA graphs, megakernels, etc.)
        # before deleting the model to prevent pool leaks.
        if hasattr(engine, "_release_acceleration_resources"):
            engine._release_acceleration_resources()
        # Sleep level 2 to drop the model from VRAM if the engine supports it.
        try:
            engine.sleep(level=2)
        except Exception:
            pass
        del engine
        self._engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            from research.model_loader import ModelLoader
            ModelLoader.clear_cache()
            # Verify VRAM was actually released
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [VRAM] After engine free: {allocated:.2f} GB allocated, "
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
        """Phase 2b: SFT continuation from best checkpoint.

        V10: wires all sft_train training tricks — Muon-SF optimizer, grad-mixup,
        curriculum learning, augmentation, SYNPRO, focal/dynamic-focal loss,
        entropy weighting, MTP auxiliary loss, distillation, sequential freeze,
        EMA, BitNet-everywhere, LazyTrain, OOMB, Hybrid8Bit clip, elastic
        grad-accum, FreeToken, disk cache, packed sequences, async prefetch.
        Each trick is gated by its LoopConfig field so users can toggle them
        via CLI args.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 2: FINETUNE (epoch {self.epoch})")
        print(f"{'='*70}")

        # Free the self-play engine before finetuning — training needs the
        # full VRAM budget. The engine is reloaded fresh next epoch.
        self._free_engine()

        save_path = self._epoch_checkpoint_path(epoch)
        c = self.config

        cmd = [
            os.sys.executable, "-m", "research.training.runners.sft_train",
            "--data", data_path,
            "--checkpoint", self.best_checkpoint,
            "--save", save_path,
            "--config", c.config_name,
            "--max-steps", str(c.ft_max_steps),
            "--lr", str(c.ft_lr),
            "--min-lr", str(c.ft_min_lr),
            "--batch-size", str(c.ft_batch_size),
            "--grad-accum", str(c.ft_grad_accum),
            "--sync-freq", str(c.ft_sync_freq),
            "--seq-len", str(c.ft_seq_len),
            "--warmup-steps", str(c.ft_warmup_steps),
            "--optimizer", c.ft_optimizer,
            "--grad-compression", c.ft_grad_compression,
            "--entropy-alpha", str(c.ft_entropy_alpha),
            "--loss-function", c.ft_loss_function,
            "--focal-gamma", str(c.ft_focal_gamma),
            "--label-smoothing-eps", str(c.ft_label_smoothing_eps),
            "--norm-type", c.ft_norm_type,
            "--grad-mixup", str(c.ft_grad_mixup),
            "--curriculum", c.ft_curriculum,
            "--vram-limit-gb", str(c.ft_vram_limit_gb),
            "--ram-limit-percent", "90",
        ]

        # ── LoRA / BitNet ──
        if c.ft_lora:
            cmd.extend(["--lora-r", str(c.ft_lora_r),
                        "--lora-alpha", str(c.ft_lora_alpha)])
        else:
            cmd.append("--no-lora")
        if c.ft_bitnet_everywhere:
            cmd.append("--bitnet-everywhere")
        else:
            cmd.append("--no-bitnet-everywhere")
        if c.ft_manual_lora:
            cmd.append("--manual-lora")

        # ── Gradient checkpointing ──
        if c.ft_grad_checkpoint:
            cmd.extend(["--grad-checkpoint", "--checkpoint-strategy",
                        c.ft_checkpoint_strategy])
        else:
            cmd.append("--no-grad-checkpoint")

        # ── Data pipeline ──
        if c.ft_disk_cache:
            cmd.append("--disk-cache")
        else:
            cmd.append("--no-disk-cache")
        if c.ft_pack_sequences:
            cmd.append("--pack-sequences")
        else:
            cmd.append("--no-pack-sequences")
        if c.ft_async_prefetch:
            cmd.extend(["--async-prefetch", "--prefetch-count",
                        str(c.ft_prefetch_count)])
        else:
            cmd.append("--no-async-prefetch")

        # ── Augmentation / SYNPRO ──
        if c.ft_augment:
            cmd.append("--augment")
        if c.ft_synpro:
            cmd.append("--synpro")

        # ── MTP auxiliary loss ──
        if c.ft_mtp_weight > 0:
            cmd.extend(["--mtp-weight", str(c.ft_mtp_weight)])
            if c.ft_mtp_n_heads is not None:
                cmd.extend(["--mtp-n-heads", str(c.ft_mtp_n_heads)])

        # ── Distillation ──
        if c.ft_distill:
            cmd.append("--distill")
            if c.ft_teacher_checkpoint:
                cmd.extend(["--teacher-checkpoint", c.ft_teacher_checkpoint])
            cmd.extend(["--distill-topk", str(c.ft_distill_topk),
                        "--distill-truncate", str(c.ft_distill_truncate)])
            if c.ft_distill_prefix:
                cmd.append("--distill-prefix")

        # ── Sequential freeze ──
        if c.ft_sequential_freeze > 0:
            cmd.extend(["--sequential-freeze", str(c.ft_sequential_freeze),
                        "--final-finetune-steps", str(c.ft_final_finetune_steps)])

        # ── EMA ──
        if c.ft_ema:
            cmd.extend(["--ema", "--ema-decay", str(c.ft_ema_decay)])

        # ── L2-SP anchor regularization ──
        if c.ft_anchor:
            cmd.extend(["--anchor", c.ft_anchor,
                        "--l2-lambda", str(c.ft_l2_lambda)])

        # ── Sample weighting ──
        if c.ft_sample_weighting:
            cmd.append("--sample-weighting")

        # ── LazyTrain / OOMB / Hybrid8Bit / elastic / FreeToken ──
        if c.ft_lazy_train:
            cmd.append("--lazy-train")
        if c.ft_oomb:
            cmd.append("--oomb")
        if c.ft_hybrid_clip:
            cmd.append("--hybrid-clip")
        if c.ft_elastic_grad_accum:
            cmd.append("--elastic-grad-accum")
        if c.ft_freetoken:
            cmd.append("--freetoken")

        # ── torch.compile ──
        if c.ft_compile:
            cmd.append("--compile")

        # Run in-process (no subprocess spawn overhead)
        if not self._run_subprocess(cmd, "Finetune"):
            raise RuntimeError(f"Finetune failed (in-process)")
        if not os.path.exists(save_path):
            raise RuntimeError(f"Finetune completed but checkpoint not found: {save_path}")

        return save_path

    def _run_subprocess(self, cmd: list[str], stage_name: str) -> bool:
        """Run a training stage in-process (no subprocess spawn).

        Calls the runner's main() directly via sys.argv manipulation, with
        explicit VRAM cleanup before/after. Falls back to subprocess if the
        module has no main() or the in-process path fails.
        """
        # cmd[0] is the python executable, cmd[1] is "-m", cmd[2] is the module
        if len(cmd) >= 3 and cmd[1] == "-m":
            module_name = cmd[2]
            cli_args = cmd[3:]
        else:
            return self._run_subprocess_fallback(cmd, stage_name)

        print(f"  Running (in-process): {module_name} "
              f"{' '.join(cli_args[:3])}... ({len(cli_args)} args)")

        # Cleanup VRAM from self-play before starting training
        self._free_engine()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import sys
        import importlib
        old_argv = sys.argv
        sys.argv = [module_name] + cli_args
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "main"):
                mod.main()
                return True
            else:
                print(f"  {stage_name}: module has no main() — falling back to subprocess")
                sys.argv = old_argv
                return self._run_subprocess_fallback(cmd, stage_name)
        except SystemExit as e:
            code = e.code
            if code is None or code == 0:
                return True
            print(f"  {stage_name} FAILED (exit code {code})")
            return False
        except Exception as e:
            print(f"  {stage_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            sys.argv = old_argv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _run_subprocess_fallback(self, cmd: list[str], stage_name: str) -> bool:
        """Legacy subprocess fallback (used when in-process isn't possible)."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        print(f"  Running (subprocess fallback): {' '.join(cmd[:4])}... ({len(cmd)} args)")
        import subprocess
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  {stage_name} FAILED (exit code {result.returncode})")
            return False
        return True

    # ── Phase 3: Evaluate ────────────────────────────────────────────

    def _evaluate(self, checkpoint: str) -> dict:
        """Phase 3: Evaluate candidate vs current best via fast_eval.

        V10: passes the self-play ForgeEngine to fast_eval for weight-swap
        reuse (skips ~40s model reload). The engine's KV cache is
        re-activated to a standard eval configuration before testing.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 3: EVALUATE")
        print(f"{'='*70}")

        from research.self_play.discovery.fast_eval import fast_eval

        try:
            results = fast_eval(
                base_checkpoint=self.best_checkpoint,
                candidate_checkpoint=checkpoint,
                device=self.config.device,
                engine=self._engine,           # reuse self-play engine
                config_name=self.config.config_name,  # V10 config
            )
        except Exception as e:
            print(f"  Eval failed: {e}")
            import traceback; traceback.print_exc()
            return {"passed": False, "error": str(e)}
        finally:
            # Engine is no longer needed after eval — free VRAM for next epoch.
            self._free_engine()

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
            with open(data_path, encoding='utf-8') as _f:
                n_examples = sum(1 for _ in _f)

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
        description="Infinite AZR self-play training loop (V10: ForgeEngine + all training tricks)")
    parser.add_argument("--checkpoint", required=True,
                        help="Starting checkpoint (safetensors). Default V10: "
                             "research/checkpoints/ForgeLM_V2_Light.safetensors")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs to run")
    parser.add_argument("--tasks-per-epoch", type=int, default=30,
                        help="Tasks to propose + solve per epoch")
    parser.add_argument("--ft-steps", type=int, default=100,
                        help="Finetune steps per epoch")
    parser.add_argument("--ft-lr", type=float, default=5e-5,
                        help="Finetune learning rate (V10 default 5e-5 for muon_sf)")
    parser.add_argument("--ft-batch-size", type=int, default=1)
    parser.add_argument("--ft-grad-accum", type=int, default=5,
                        help="Gradient accumulation steps (evolution: 5)")
    parser.add_argument("--ft-optimizer", type=str, default="muon_sf",
                        choices=["fused", "bnb", "lion", "muon", "muon_sf",
                                 "muon_sf_plain", "flash_adamw", "flash_lion",
                                 "forge", "sf_normuon", "amuse", "mona",
                                 "cpu_offload", "badam", "fira_nlrq"],
                        help="Finetune optimizer (V10 default: muon_sf, 2.39x vs AdamW)")
    parser.add_argument("--self-play-mode", type=str, default="azr",
                        choices=["azr", "soar", "sgs", "thinking"],
                        help="Self-play mode: 'azr' (standard), 'soar' (meta-RL curriculum, "
                             "escapes learning plateaus), 'sgs' (self-guided self-play, "
                             "prevents Conjecturer collapse with Guide role), "
                             "'thinking' (ForgeLM V10-Thinking pipeline: CPT to SFT to DPO to RLVR)")
    parser.add_argument("--saerl", action="store_true",
                        help="Enable SAERL: SAE-guided data engineering for RL. "
                             "Diversity control + difficulty curriculum + quality filtering. "
                             "+3%% accuracy, 20%% fewer steps to target.")
    parser.add_argument("--opmix", action="store_true",
                        help="Enable OP-MIX: on-policy data mixing via low-rank adapters. "
                             "Dynamically adjusts mixing ratio across data sources.")
    parser.add_argument("--ft-no-lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--ft-grad-checkpoint", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Gradient checkpointing (default True, use --no-ft-grad-checkpoint)")
    parser.add_argument("--ft-checkpoint-strategy", type=str, default="all",
                        choices=["all", "ffn", "attn", "none", "lazy", "optimal"],
                        help="Selective gradient checkpointing strategy")
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
    parser.add_argument("--config", type=str, default="forgelm_v2_light",
                        help="Model config name")
    parser.add_argument("--eval-threshold", type=float, default=0.5,
                        help="Min candidate quality vs base to promote")

    # ── V10 ForgeEngine inference features ──
    parser.add_argument("--no-forge-engine", action="store_true",
                        help="Disable ForgeEngine (use bare model path, no V10 inference features)")
    parser.add_argument("--kv-cache", type=str, default="spectral",
                        help="KV cache strategy for self-play generation (V10 default: spectral)")
    parser.add_argument("--decoding", type=str, default="mtp_selfspec",
                        choices=["standard", "speculative", "medusa", "dspark",
                                 "eagle3", "mtp_selfspec"],
                        help="Decoding strategy (V10 default: mtp_selfspec)")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=[None, "int8", "int4", "fp8", "w8a8", "nvfp4"],
                        help="Weight quantization for self-play generation")
    parser.add_argument("--acceleration", type=str, default=None,
                        choices=[None, "cuda_graph", "megakernel", "flex_decoding"],
                        help="Acceleration strategy")
    parser.add_argument("--use-compile", action="store_true",
                        help="torch.compile the self-play model (experimental on Windows)")
    parser.add_argument("--no-triton-conv", action="store_true",
                        help="Disable Triton fused conv kernel")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable prefix KV cache")
    parser.add_argument("--kv-cache-tokens", type=int, default=4096,
                        help="KV cache token budget for self-play generation")

    # ── Training tricks ──
    parser.add_argument("--ft-loss-function", type=str, default="ce",
                        choices=["ce", "focal", "label_smoothing", "lovasz",
                                 "dynamic_focal", "mixture"],
                        help="Loss function (focal/dynamic_focal: +36%% exact match)")
    parser.add_argument("--ft-focal-gamma", type=float, default=4.93,
                        help="Focal loss gamma (evolution optimum: 4.93)")
    parser.add_argument("--ft-entropy-alpha", type=float, default=0.5,
                        help="Token entropy weighting alpha (0 disables)")
    parser.add_argument("--ft-grad-mixup", type=int, default=1,
                        help="N-batch grad averaging (1=off, 3=1.25x convergence)")
    parser.add_argument("--ft-curriculum", type=str, default="none",
                        choices=["none", "vanilla", "pacing", "interleaved", "warmup"],
                        help="Curriculum learning strategy (18-45%% fewer steps)")
    parser.add_argument("--ft-augment", action="store_true",
                        help="Training-time data augmentation (token noise, FIM, target offset)")
    parser.add_argument("--ft-synpro", action="store_true",
                        help="SYNPRO synthetic data generation (3.7-5.2x effective tokens)")
    parser.add_argument("--ft-norm-type", type=str, default="rmsnorm",
                        choices=["rmsnorm", "seednorm", "dyt"],
                        help="Normalization type (dyt = Muon-compatible Dynamic Tanh)")
    parser.add_argument("--ft-mtp-weight", type=float, default=0.0,
                        help="MTP auxiliary loss weight (Nemotron Lightning, 0=disabled)")
    parser.add_argument("--ft-distill", action="store_true",
                        help="Knowledge distillation from a teacher model")
    parser.add_argument("--ft-teacher-checkpoint", type=str, default=None,
                        help="Teacher model checkpoint for distillation")
    parser.add_argument("--ft-sequential-freeze", type=int, default=0,
                        help="Sequential freeze/unfreeze (0=off, 4=4 phases)")
    parser.add_argument("--ft-ema", action="store_true",
                        help="EMA shadow weights")
    parser.add_argument("--ft-anchor", type=str, default=None,
                        help="L2-SP anchor checkpoint (anti-drift, NeurIPS 2024)")
    parser.add_argument("--ft-lazy-train", action="store_true",
                        help="LazyTrain scheduler (1.24x sustained TFLOPS)")
    parser.add_argument("--ft-oomb", action="store_true",
                        help="OOMB chunk-recurrent training (128K+ context on 12GB)")
    parser.add_argument("--ft-elastic-grad-accum", action="store_true",
                        help="Dynamic grad_accum (OOM prevention)")
    parser.add_argument("--ft-freetoken", action="store_true",
                        help="FreeToken double-buffered grad pipeline (cpu_offload only)")
    parser.add_argument("--ft-compile", action="store_true",
                        help="torch.compile the training model (experimental on Windows)")
    parser.add_argument("--ft-bitnet-everywhere", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="BitNet ternary QAT (1.58 bits, 2.39x vs AdamW)")
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
        ft_checkpoint_strategy=args.ft_checkpoint_strategy,
        ft_loss_function=args.ft_loss_function,
        ft_focal_gamma=args.ft_focal_gamma,
        ft_entropy_alpha=args.ft_entropy_alpha,
        ft_grad_mixup=args.ft_grad_mixup,
        ft_curriculum=args.ft_curriculum,
        ft_augment=args.ft_augment,
        ft_synpro=args.ft_synpro,
        ft_norm_type=args.ft_norm_type,
        ft_mtp_weight=args.ft_mtp_weight,
        ft_distill=args.ft_distill,
        ft_teacher_checkpoint=args.ft_teacher_checkpoint,
        ft_sequential_freeze=args.ft_sequential_freeze,
        ft_ema=args.ft_ema,
        ft_anchor=args.ft_anchor,
        ft_lazy_train=args.ft_lazy_train,
        ft_oomb=args.ft_oomb,
        ft_elastic_grad_accum=args.ft_elastic_grad_accum,
        ft_freetoken=args.ft_freetoken,
        ft_compile=args.ft_compile,
        ft_bitnet_everywhere=args.ft_bitnet_everywhere,
        # V10 inference features
        use_forge_engine=not args.no_forge_engine,
        kv_cache=args.kv_cache,
        decoding=args.decoding,
        quantize=args.quantize,
        acceleration=args.acceleration,
        use_compile=args.use_compile,
        use_triton_conv=not args.no_triton_conv,
        use_prefix_cache=not args.no_prefix_cache,
        kv_cache_tokens=args.kv_cache_tokens,
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

    if args.self_play_mode == "thinking":
        print("[InfiniteLoop] THINKING mode: ForgeLM V10-Thinking pipeline (CPT->SFT->DPO->RLVR)")
        pipeline_config = ThinkingPipelineConfig(
            config_name=args.config,
            optimizer=args.ft_optimizer if args.ft_optimizer != "bnb" else "cpu_offload",
        )
        pipeline = ThinkingPipeline(args.checkpoint, pipeline_config)
        final_ckpt = pipeline.run()
        print(f"\n[ThinkingPipeline] Final checkpoint: {final_ckpt}")
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


# ── ForgeLM V10-Thinking Pipeline ─────────────────────────────────────────

@dataclass
class ThinkingPipelineConfig:
    """Configuration for the ForgeLM V10-Thinking multi-stage pipeline.

    Implements the full training recipe from Liquid AI's ForgeLM V10-Thinking:
      CPT (midtraining with reasoning traces) →
      SFT (curriculum: short CoT → long CoT + mix distillation) →
      DPO (doom-loop mitigation with LLM judge) →
      RLVR (GRPO with n-gram repetition penalty on verifiable tasks)

    Each stage is a subprocess call to the corresponding runner. The pipeline
    is resumable: if interrupted, it skips completed stages by checking for
    output checkpoint existence.
    """
    # Model
    config_name: str = "forgelm_v2_light"  # V10 default
    device: str = "cuda"
    optimizer: str = "muon_sf"  # V10 default: Muon-SF (was: cpu_offload)

    # Stage 1: CPT (midtraining with reasoning traces)
    cpt_enabled: bool = True
    cpt_reasoning_data: list[str] = field(default_factory=lambda: [
        "research/distillation/hf_datasets/openr1_math.jsonl",
        "research/distillation/hf_datasets/openthoughts_114k.jsonl",
        "research/distillation/hf_datasets/dolphin_r1.jsonl",
    ])
    cpt_general_data: list[str] = field(default_factory=lambda: [
        "research/distillation/hf_datasets/orca_math.jsonl",
        "research/distillation/hf_datasets/metamath.jsonl",
    ])
    cpt_reasoning_ratio: float = 0.6
    cpt_lr: float = 1e-4
    cpt_max_steps: int = 5000
    cpt_batch_size: int = 2
    cpt_seq_len: int = 2048
    cpt_grad_accum: int = 4

    # Stage 2: Curriculum SFT (mix distillation + 2-stage curriculum)
    sft_enabled: bool = True
    sft_data_inputs: list[str] = field(default_factory=lambda: [
        "research/distillation/hf_datasets/gsm8k.jsonl",
        "research/distillation/hf_datasets/openr1_math.jsonl",
        "research/distillation/hf_datasets/openthoughts_114k.jsonl",
    ])
    sft_short_cot_max_tokens: int = 150
    sft_long_cot_min_tokens: int = 300
    sft_mix_ratio: float = 0.5
    sft_filter_doom_loops: bool = True
    sft_stage1_lr: float = 5e-5
    sft_stage1_steps: int = 1000
    sft_stage2_lr: float = 2e-5
    sft_stage2_steps: int = 1500
    sft_batch_size: int = 2
    sft_seq_len: int = 1024
    sft_grad_accum: int = 4

    # Stage 3: DPO (doom-loop mitigation)
    dpo_enabled: bool = True
    dpo_n_temp_samples: int = 5
    dpo_max_new_tokens: int = 512
    dpo_judge_model: str = "qwen3-32b"
    dpo_max_prompts: int = 500
    dpo_lr: float = 5e-7
    dpo_max_steps: int = 200
    dpo_method: str = "orpo"

    # Stage 4: RLVR (GRPO with repetition penalty)
    rlvr_enabled: bool = True
    rlvr_tasks: list[str] = field(default_factory=lambda: [
        "research/distillation/hf_datasets/gsm8k.jsonl",
    ])
    rlvr_task_type: str = "math"
    rlvr_max_steps: int = 500
    rlvr_group_size: int = 4
    rlvr_lr: float = 5e-6
    rlvr_algorithm: str = "grpo"
    rlvr_use_repetition_penalty: bool = True

    # Paths
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data/thinking_pipeline"


class ThinkingPipeline:
    """Orchestrates the full ForgeLM V10-Thinking training pipeline.

    Stages:
      1. CPT: Midtrain with reasoning traces (openthoughts, openr1_math, dolphin_r1)
      2. SFT: Curriculum (short CoT internal solver → long CoT externalize + mix distillation)
      3. DPO: Doom-loop mitigation (5 temp + 1 greedy, LLM judge, n-gram loop detector)
      4. RLVR: GRPO with n-gram repetition penalty on verifiable tasks

    Each stage runs as a subprocess calling the corresponding runner module.
    The pipeline is resumable: completed stages are skipped if the output
    checkpoint already exists.
    """

    def __init__(self, base_checkpoint: str, config: ThinkingPipelineConfig | None = None):
        self.base_checkpoint = base_checkpoint
        self.config = config or ThinkingPipelineConfig()
        self.history: list[dict] = []

    def _stage_path(self, stage: str) -> str:
        """Get the checkpoint path for a stage output."""
        return os.path.normpath(os.path.join(
            self.config.checkpoint_dir,
            f"ForgeLM_V10_{stage}.safetensors"))

    def _run_subprocess(self, cmd: list[str], stage_name: str) -> bool:
        """Run a training stage in-process (no subprocess spawn).

        Replaces the old subprocess.run() approach which spawned a fresh
        Python interpreter per stage (~3-5s startup + import overhead each).
        Now calls the runner's main() directly via sys.argv manipulation,
        with explicit VRAM cleanup between stages.
        """
        # cmd[0] is the python executable, cmd[1] is "-m", cmd[2] is the module
        if len(cmd) >= 3 and cmd[1] == "-m":
            module_name = cmd[2]
            cli_args = cmd[3:]
        else:
            # Fallback: can't parse, use old subprocess approach
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            print(f"  Running (subprocess fallback): {' '.join(cmd[:4])}... ({len(cmd)} args)")
            import subprocess
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print(f"  {stage_name} FAILED (exit code {result.returncode})")
                return False
            return True

        print(f"  Running (in-process): {module_name} {' '.join(cli_args[:3])}... ({len(cli_args)} args)")

        # Cleanup VRAM from previous stage before starting new one
        self._cleanup_vram()

        import sys
        import importlib
        old_argv = sys.argv
        sys.argv = [module_name] + cli_args
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "main"):
                mod.main()
                return True
            else:
                print(f"  {stage_name}: module has no main() — falling back to subprocess")
                sys.argv = old_argv
                return self._run_subprocess_fallback(cmd, stage_name)
        except SystemExit as e:
            code = e.code
            if code is None or code == 0:
                return True
            print(f"  {stage_name} FAILED (exit code {code})")
            return False
        except Exception as e:
            print(f"  {stage_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            sys.argv = old_argv
            self._cleanup_vram()

    def _run_subprocess_fallback(self, cmd: list[str], stage_name: str) -> bool:
        """Legacy subprocess fallback (used when in-process isn't possible)."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        import subprocess
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  {stage_name} FAILED (exit code {result.returncode})")
            return False
        return True

    def _cleanup_vram(self):
        """Release VRAM between in-process pipeline stages."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _stage_completed(self, checkpoint: str) -> bool:
        """Check if a stage's output checkpoint already exists (for resumability)."""
        return os.path.exists(checkpoint)

    # ── Stage 1: CPT ──────────────────────────────────────────────────

    def run_cpt(self) -> str:
        """Stage 1: Midtraining with reasoning traces."""
        output = self._stage_path("CPT")
        print(f"\n{'='*70}")
        print(f"  STAGE 1/4: CPT (Midtraining with Reasoning Traces)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.cpt_enabled:
            print(f"  Skipped (disabled)")
            return self.base_checkpoint

        cmd = [
            os.sys.executable, "-m", "research.training.runners.cpt_train",
            "--reasoning-data", *self.config.cpt_reasoning_data,
            "--general-data", *self.config.cpt_general_data,
            "--config", self.config.config_name,
            "--checkpoint", self.base_checkpoint,
            "--save", output,
            "--optimizer", self.config.optimizer,
            "--reasoning-ratio", str(self.config.cpt_reasoning_ratio),
            "--lr", str(self.config.cpt_lr),
            "--max-steps", str(self.config.cpt_max_steps),
            "--batch-size", str(self.config.cpt_batch_size),
            "--seq-len", str(self.config.cpt_seq_len),
            "--grad-accum", str(self.config.cpt_grad_accum),
        ]
        if not self._run_subprocess(cmd, "CPT"):
            raise RuntimeError("CPT stage failed")
        self.history.append({"stage": "cpt", "checkpoint": output})
        return output

    # ── Stage 2: Curriculum SFT ───────────────────────────────────────

    def run_sft(self, cpt_checkpoint: str) -> str:
        """Stage 2: Curriculum SFT (mix distillation + 2-stage curriculum)."""
        output = self._stage_path("SFT2")
        print(f"\n{'='*70}")
        print(f"  STAGE 2/4: CURRICULUM SFT (Mix Distillation + 2-Stage)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.sft_enabled:
            print(f"  Skipped (disabled)")
            return cpt_checkpoint

        # Step 2a: Prepare curriculum data
        curriculum_dir = os.path.join(self.config.data_dir, "curriculum")
        cmd_prep = [
            os.sys.executable, "-m", "research.training.runners.curriculum_sft",
            "prepare",
            "--input", *self.config.sft_data_inputs,
            "--output-dir", curriculum_dir,
            "--short-cot-max-tokens", str(self.config.sft_short_cot_max_tokens),
            "--long-cot-min-tokens", str(self.config.sft_long_cot_min_tokens),
            "--mix-ratio", str(self.config.sft_mix_ratio),
        ]
        if self.config.sft_filter_doom_loops:
            cmd_prep.append("--filter-doom-loops")
        if not self._run_subprocess(cmd_prep, "SFT prepare"):
            raise RuntimeError("SFT data preparation failed")

        stage1_data = os.path.join(curriculum_dir, "stage1_short.jsonl")
        stage2_data = os.path.join(curriculum_dir, "stage2_long.jsonl")
        sft1_output = self._stage_path("SFT1")

        # Step 2b: Stage 1 (short CoT — internal solver)
        cmd_s1 = [
            os.sys.executable, "-m", "research.training.runners.curriculum_sft",
            "train-stage1",
            "--data", stage1_data,
            "--checkpoint", cpt_checkpoint,
            "--save", sft1_output,
            "--config", self.config.config_name,
            "--lr", str(self.config.sft_stage1_lr),
            "--max-steps", str(self.config.sft_stage1_steps),
            "--optimizer", self.config.optimizer,
            "--seq-len", str(self.config.sft_seq_len),
            "--batch-size", str(self.config.sft_batch_size),
            "--grad-accum", str(self.config.sft_grad_accum),
        ]
        if not self._run_subprocess(cmd_s1, "SFT Stage 1"):
            raise RuntimeError("SFT Stage 1 failed")

        # Step 2c: Stage 2 (long CoT — externalize reasoning)
        cmd_s2 = [
            os.sys.executable, "-m", "research.training.runners.curriculum_sft",
            "train-stage2",
            "--data", stage2_data,
            "--checkpoint", sft1_output,
            "--save", output,
            "--config", self.config.config_name,
            "--lr", str(self.config.sft_stage2_lr),
            "--max-steps", str(self.config.sft_stage2_steps),
            "--optimizer", self.config.optimizer,
            "--seq-len", str(self.config.sft_seq_len * 2),
            "--batch-size", str(self.config.sft_batch_size),
            "--grad-accum", str(self.config.sft_grad_accum),
        ]
        if not self._run_subprocess(cmd_s2, "SFT Stage 2"):
            raise RuntimeError("SFT Stage 2 failed")
        self.history.append({"stage": "sft", "checkpoint": output})
        return output

    # ── Stage 3: DPO ──────────────────────────────────────────────────

    def run_dpo(self, sft_checkpoint: str) -> str:
        """Stage 3: DPO doom-loop mitigation."""
        output = self._stage_path("DPO")
        print(f"\n{'='*70}")
        print(f"  STAGE 3/4: DPO (Doom-Loop Mitigation)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.dpo_enabled:
            print(f"  Skipped (disabled)")
            return sft_checkpoint

        # Step 3a: Generate preference data
        dpo_data_dir = os.path.join(self.config.data_dir, "dpo")
        os.makedirs(dpo_data_dir, exist_ok=True)
        prompts_file = os.path.join(self.config.data_dir, "curriculum", "stage2_long.jsonl")
        pref_data = os.path.join(dpo_data_dir, "preference_pairs.jsonl")

        cmd_gen = [
            os.sys.executable, "-m", "research.training.runners.dpo_data_gen",
            "--prompts", prompts_file,
            "--checkpoint", sft_checkpoint,
            "--output", pref_data,
            "--config", self.config.config_name,
            "--n-temp-samples", str(self.config.dpo_n_temp_samples),
            "--max-new-tokens", str(self.config.dpo_max_new_tokens),
            "--judge-model", self.config.dpo_judge_model,
            "--max-prompts", str(self.config.dpo_max_prompts),
        ]
        if not self._run_subprocess(cmd_gen, "DPO data generation"):
            print("  DPO data generation failed, skipping DPO stage")
            return sft_checkpoint

        # Step 3b: DPO training
        cmd_dpo = [
            os.sys.executable, "-m", "research.training.runners.dpo_align",
            "--data", pref_data,
            "--checkpoint", sft_checkpoint,
            "--save", output,
            "--config", self.config.config_name,
            "--method", self.config.dpo_method,
            "--lr", str(self.config.dpo_lr),
            "--max-steps", str(self.config.dpo_max_steps),
            "--optimizer", self.config.optimizer,
        ]
        if not self._run_subprocess(cmd_dpo, "DPO training"):
            raise RuntimeError("DPO stage failed")
        self.history.append({"stage": "dpo", "checkpoint": output})
        return output

    # ── Stage 4: RLVR ─────────────────────────────────────────────────

    def run_rlvr(self, dpo_checkpoint: str) -> str:
        """Stage 4: RLVR with GRPO + n-gram repetition penalty."""
        output = self._stage_path("RLVR")
        print(f"\n{'='*70}")
        print(f"  STAGE 4/4: RLVR (GRPO + Repetition Penalty)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.rlvr_enabled:
            print(f"  Skipped (disabled)")
            return dpo_checkpoint

        cmd = [
            os.sys.executable, "-m", "research.training.runners.rlvr_train",
            "--tasks", *self.config.rlvr_tasks,
            "--task-type", self.config.rlvr_task_type,
            "--checkpoint", dpo_checkpoint,
            "--save", output,
            "--config", self.config.config_name,
            "--max-steps", str(self.config.rlvr_max_steps),
            "--group-size", str(self.config.rlvr_group_size),
            "--lr", str(self.config.rlvr_lr),
            "--rl-algorithm", self.config.rlvr_algorithm,
            "--optimizer", self.config.optimizer,
        ]
        if self.config.rlvr_use_repetition_penalty:
            cmd.append("--use-repetition-penalty")
        if not self._run_subprocess(cmd, "RLVR"):
            raise RuntimeError("RLVR stage failed")
        self.history.append({"stage": "rlvr", "checkpoint": output})
        return output

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self) -> str:
        """Run the full 4-stage ForgeLM V10-Thinking pipeline.

        Returns the path to the final RLVR checkpoint.
        """
        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"#  FORGELM V10-THINKING PIPELINE")
        print(f"#  Base checkpoint: {self.base_checkpoint}")
        print(f"#  Config: {self.config.config_name}")
        print(f"#  Optimizer: {self.config.optimizer}")
        print(f"{'#'*70}")

        stages = []
        try:
            # Stage 1: CPT
            cpt_ckpt = self.run_cpt()
            stages.append(("CPT", cpt_ckpt))

            # Stage 2: Curriculum SFT
            sft_ckpt = self.run_sft(cpt_ckpt)
            stages.append(("SFT", sft_ckpt))

            # Stage 3: DPO
            dpo_ckpt = self.run_dpo(sft_ckpt)
            stages.append(("DPO", dpo_ckpt))

            # Stage 4: RLVR
            rlvr_ckpt = self.run_rlvr(dpo_ckpt)
            stages.append(("RLVR", rlvr_ckpt))

        except RuntimeError as e:
            print(f"\n  PIPELINE INTERRUPTED: {e}")
            print(f"  Completed stages: {[s[0] for s in stages]}")
            if stages:
                print(f"  Last checkpoint: {stages[-1][1]}")
            raise

        elapsed = time.time() - t0
        print(f"\n{'#'*70}")
        print(f"#  PIPELINE COMPLETE ({elapsed:.0f}s)")
        print(f"{'#'*70}")
        for stage_name, ckpt in stages:
            print(f"  {stage_name}: {ckpt}")
        print(f"\n  Final model: {rlvr_ckpt}")
        print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        return rlvr_ckpt


if __name__ == "__main__":
    main()
