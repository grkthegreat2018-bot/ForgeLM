"""GenModelManager — lifecycle management for LLM gen models in ForgeEvolve.

Manages a single global LLM gen model size across all LLM-based domains
(the "golden ratio"). Tracks aggregate performance vs model size and
automatically grows or shrinks the model to find the most efficient point
on the ability-per-param curve.

Golden Ratio:
  The target ability curve is logarithmic: target_ability = A * log(P) + B
  where P = param_count, and A, B are fitted from performance history.
  The manager finds the point on this curve where the gen model is most
  efficient (best ability per param).

Growth/Shrink:
  - Grow: when performance plateaus (no aggregate improvement for N rounds)
    but compute budget allows (VRAM < 80%). Increases d_model or n_layers.
    Distills weights from old model (interpolate/zero-init new layers).
  - Shrink: when overperforming (aggregate > 1.2x target for 5 rounds).
    Distills to smaller model via logit-matching.

Fine-tuning:
  fine_tune_on_solutions(solutions) — cross-entropy loss on output tokens
  from high-scoring solutions. Supports CPUAdamW for VRAM-constrained training.

Persistence:
  Saves gen model state + manager state (current size, performance history,
  golden ratio curve) to DB via FindingsDB.

Usage:
    from research.evolution.gen_model_manager import GenModelManager
    from research.evolution.database import FindingsDB

    db = FindingsDB("forge_evolve.db")
    manager = GenModelManager(db=db)
    manager.record_round("math", score=0.65, gen_model_size=manager.get_current_size())
    if manager.should_grow():
        manager.grow()
    manager.fine_tune_on_solutions([
        {"input": "2+2=?", "output": "4", "score": 1.0},
    ])
    manager.save_state()
"""
from __future__ import annotations

import math
import time
import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from research.evolution.llm_gen_model import LLMGenModel
from research.evolution.database import FindingsDB
from research.model_loader import unpack_output_with_kv


# ── Constants ──────────────────────────────────────────────────────────

# Grow: no aggregate improvement for this many consecutive rounds
GROW_PLATEAU_ROUNDS = 3

# Shrink: aggregate performance > 1.2x target for this many consecutive rounds
SHRINK_OVERPERFORM_ROUNDS = 5
SHRINK_MARGIN = 1.2

# VRAM threshold for growth (don't grow if VRAM usage > 80%)
VRAM_GROW_THRESHOLD = 0.80

# Growth increments
D_MODEL_INCREMENT = 64
N_LAYERS_INCREMENT = 2

# Minimum sizes (don't shrink below these)
MIN_D_MODEL = 128
MIN_N_LAYERS = 2

# Fine-tuning defaults
FINE_TUNE_LR = 1e-4
FINE_TUNE_EPOCHS = 3
FINE_TUNE_BATCH_SIZE = 4
FINE_TUNE_SCORE_THRESHOLD = 0.5  # only train on solutions with score >= this

# Distillation defaults
DISTILL_EPOCHS = 5
DISTILL_LR = 5e-4
DISTILL_TEMPERATURE = 4.0  # softmax temperature for logit matching

# Calibration prompts for distillation (diverse task types)
_DEFAULT_CALIBRATION_PROMPTS = [
    "Solve: What is 15 * 23?",
    "Write a function to reverse a list in Python.",
    "Explain the concept of recursion.",
    "What is the time complexity of binary search?",
    "Solve: If x + 5 = 12, what is x?",
    "Write a haiku about the ocean.",
    "Sort the numbers: 5, 2, 8, 1, 9, 3.",
    "What is the capital of France?",
]


class GenModelManager:
    """Manages the lifecycle of LLM gen models for ForgeEvolve.

    Maintains ONE gen model size for all LLM-based domains. Tracks aggregate
    performance vs size and adjusts the model to find the golden ratio
    (most efficient ability-per-param point).

    Args:
        db: FindingsDB instance for persistence. If None, uses in-memory only.
        config_name: Config preset name for the gen model.
        device: "cuda", "cpu", or None (auto-detect).
        tokenizer_path: Path to tokenizer directory.
        vram_limit_fraction: Max VRAM fraction to use (default 0.80).
    """

    def __init__(
        self,
        db: FindingsDB | None = None,
        config_name: str = "gen_model_tiny",
        device: str | None = None,
        tokenizer_path: str = "research/checkpoints/lfm25_tokenizer",
        vram_limit_fraction: float = VRAM_GROW_THRESHOLD,
    ):
        self.db = db
        self.config_name = config_name
        self.tokenizer_path = tokenizer_path
        self.vram_limit_fraction = vram_limit_fraction

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Performance tracking
        # per-domain history: {domain: [(round_idx, score, size_dict), ...]}
        self.domain_history: dict[str, list[tuple[int, float, dict]]] = {}
        # aggregate history: [(round_idx, avg_score, param_count), ...]
        self.aggregate_history: list[tuple[int, float, int]] = []
        self.round_counter = 0

        # Golden ratio curve parameters: target_ability = A * log(P) + B
        self.golden_ratio_A: float = 0.1  # initial slope
        self.golden_ratio_B: float = -1.0  # initial intercept
        self.golden_ratio_fitted: bool = False

        # Consecutive round trackers
        self._plateau_count: int = 0
        self._overperform_count: int = 0
        self._last_aggregate_score: float = 0.0

        # Growth alternation: alternate between d_model and n_layers growth
        self._last_growth_dim: str = "n_layers"  # next growth will be d_model

        # Build or load the gen model
        self.gen_model = LLMGenModel(
            config_name=config_name,
            device=str(self.device),
            tokenizer_path=tokenizer_path,
        )

        # Try loading saved state from DB
        if self.db is not None:
            self._load_from_db()

    # ── Performance Tracking ──────────────────────────────────────────

    def record_round(self, domain_name: str, score: float,
                     gen_model_size: dict | None = None) -> None:
        """Record performance after an evolution round.

        Called after each evolution round for each LLM-based domain. Tracks
        per-domain and aggregate performance for grow/shrink decisions.

        The aggregate score is computed as the average of the most recent
        scores across all tracked domains. Each call updates the aggregate
        with the latest per-domain scores.

        Args:
            domain_name: Name of the domain (e.g. "math", "algorithms").
            score: Performance score for this round (higher = better).
            gen_model_size: Current gen model size dict (optional, uses
                current model size if None).
        """
        if gen_model_size is None:
            gen_model_size = self.gen_model.get_size_config()

        # Record per-domain
        if domain_name not in self.domain_history:
            self.domain_history[domain_name] = []
        self.domain_history[domain_name].append(
            (self.round_counter, score, dict(gen_model_size))
        )

        # Compute aggregate score: average of latest scores across all domains
        latest_scores = [
            hist[-1][1] for hist in self.domain_history.values()
            if hist
        ]
        if latest_scores:
            aggregate = sum(latest_scores) / len(latest_scores)
            param_count = self.gen_model.param_count()
            self.aggregate_history.append(
                (self.round_counter, aggregate, param_count)
            )

            # Update plateau/overperform trackers
            if aggregate > self._last_aggregate_score + 1e-6:
                self._plateau_count = 0
            else:
                self._plateau_count += 1

            target = self._get_target_ability(param_count)
            if aggregate > target * SHRINK_MARGIN:
                self._overperform_count += 1
            else:
                self._overperform_count = 0

            self._last_aggregate_score = aggregate

        self.round_counter += 1

        # Refit golden ratio curve if we have enough data
        if len(self.aggregate_history) >= 4:
            self._fit_golden_ratio()

    def get_current_performance(self) -> float:
        """Return the latest aggregate performance score."""
        if not self.aggregate_history:
            return 0.0
        return self.aggregate_history[-1][1]

    def get_current_size(self) -> dict:
        """Return the current gen model size configuration."""
        return self.gen_model.get_size_config()

    def get_current_param_count(self) -> int:
        """Return the current gen model parameter count."""
        return self.gen_model.param_count()

    # ── Golden Ratio ──────────────────────────────────────────────────

    def _get_target_ability(self, param_count: int) -> float:
        """Compute target ability for a given param count from the golden ratio curve.

        target_ability = A * log(param_count) + B
        """
        if param_count <= 0:
            return 0.0
        return self.golden_ratio_A * math.log(param_count) + self.golden_ratio_B

    def _fit_golden_ratio(self) -> None:
        """Fit the golden ratio curve (A, B) from performance history.

        Uses least-squares regression on (log(param_count), aggregate_score)
        pairs from the aggregate history.
        """
        if len(self.aggregate_history) < 2:
            return

        # Collect (log(P), score) pairs
        xs = []
        ys = []
        for _, score, param_count in self.aggregate_history:
            if param_count > 0:
                xs.append(math.log(param_count))
                ys.append(score)

        if len(xs) < 2:
            return

        # Least-squares linear regression: y = A*x + B
        x_arr = np.array(xs)
        y_arr = np.array(ys)
        n = len(xs)
        x_mean = x_arr.mean()
        y_mean = y_arr.mean()

        # A = cov(x,y) / var(x)
        x_var = np.sum((x_arr - x_mean) ** 2)
        if x_var < 1e-10:
            return  # all same param count, can't fit slope

        self.golden_ratio_A = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / x_var)
        self.golden_ratio_B = float(y_mean - self.golden_ratio_A * x_mean)
        self.golden_ratio_fitted = True

    def get_golden_ratio_curve(self) -> dict:
        """Return the current golden ratio curve parameters."""
        return {
            "A": self.golden_ratio_A,
            "B": self.golden_ratio_B,
            "fitted": self.golden_ratio_fitted,
        }

    def get_efficiency_score(self, param_count: int | None = None) -> float:
        """Compute efficiency score: ability per param (log-scaled).

        Higher = more efficient. Used to compare different sizes.

        Args:
            param_count: Param count to evaluate (default: current model).
        """
        if param_count is None:
            param_count = self.gen_model.param_count()
        if param_count <= 0:
            return 0.0
        target = self._get_target_ability(param_count)
        return target / math.log(param_count)

    # ── Size Decisions ────────────────────────────────────────────────

    def should_grow(self) -> bool:
        """Determine if the gen model should grow.

        Grow when:
        1. Performance has plateaued (no aggregate improvement for
           GROW_PLATEAU_ROUNDS consecutive rounds), AND
        2. Compute budget allows (current VRAM usage < vram_limit_fraction).

        Returns:
            True if the model should grow.
        """
        # Check plateau
        if self._plateau_count < GROW_PLATEAU_ROUNDS:
            return False

        # Check VRAM budget
        if self.device.type == "cuda":
            vram_free, vram_total = torch.cuda.mem_get_info(self.device)
            vram_used = 1.0 - (vram_free / vram_total if vram_total > 0 else 0)
            if vram_used >= self.vram_limit_fraction:
                return False

        return True

    def should_shrink(self) -> bool:
        """Determine if the gen model should shrink.

        Shrink when:
        1. Aggregate performance > 1.2x target ability for
           SHRINK_OVERPERFORM_ROUNDS consecutive rounds.

        Returns:
            True if the model should shrink.
        """
        return self._overperform_count >= SHRINK_OVERPERFORM_ROUNDS

    def get_target_size(self) -> dict:
        """Compute the target size for the next grow/shrink operation.

        Returns:
            Dict with target d_model, n_layers, intermediate_size, nlrq_rank.
        """
        current = self.gen_model.get_size_config()
        d_model = current["d_model"]
        n_layers = current["n_layers"]
        intermediate = current["intermediate_size"]
        nlrq_rank = current["nlrq_rank"]

        if self.should_grow():
            # Alternate between d_model and n_layers growth
            if self._last_growth_dim == "n_layers":
                d_model += D_MODEL_INCREMENT
                self._last_growth_dim = "d_model"
            else:
                n_layers += N_LAYERS_INCREMENT
                self._last_growth_dim = "n_layers"
            # Scale intermediate proportionally with d_model
            intermediate = max(512, d_model * 2)
            # Scale NLRQ rank with intermediate (keep ~1/8 ratio)
            nlrq_rank = max(64, intermediate // 8)

        elif self.should_shrink():
            # Reverse of growth
            if self._last_growth_dim == "d_model":
                d_model = max(MIN_D_MODEL, d_model - D_MODEL_INCREMENT)
                self._last_growth_dim = "n_layers"
            else:
                n_layers = max(MIN_N_LAYERS, n_layers - N_LAYERS_INCREMENT)
                self._last_growth_dim = "d_model"
            intermediate = max(256, d_model * 2)
            nlrq_rank = max(32, intermediate // 8)

        # Compute heads: head_dim=32, GQA with n_kv_heads that divides n_heads
        n_heads = d_model // 32  # head_dim=32
        n_kv_heads = max(2, n_heads // 4)  # target GQA ~4x
        # Ensure n_heads % n_kv_heads == 0 (required by ModelConfig)
        while n_kv_heads > 1 and n_heads % n_kv_heads != 0:
            n_kv_heads -= 1

        return {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "intermediate_size": intermediate,
            "nlrq_rank": nlrq_rank,
        }

    # ── Grow / Shrink ─────────────────────────────────────────────────

    def grow(self) -> bool:
        """Grow the gen model to a larger size with weight distillation.

        Builds a new larger model and distills weights from the old model
        (interpolate for different-sized weights, zero-init new layers).

        Returns:
            True if growth was performed.
        """
        if not self.should_grow():
            return False

        target = self.get_target_size()
        old_model = self.gen_model
        old_size = old_model.get_size_config()

        print(f"  [GenModelManager] Growing: {old_size['d_model']}d x "
              f"{old_size['n_layers']}L -> {target['d_model']}d x "
              f"{target['n_layers']}L")

        # Build new larger model
        new_model = LLMGenModel(
            config_name=self.config_name,
            device=str(self.device),
            **target,
        )

        # Weight distillation: copy/interpolate old weights to new model
        self._distill_weights(old_model, new_model)

        # Logit-matching distillation for quality preservation
        self._distill_logits(old_model, new_model, _DEFAULT_CALIBRATION_PROMPTS)

        # Replace model
        self.gen_model = new_model
        del old_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Reset trackers
        self._plateau_count = 0
        self._overperform_count = 0

        print(f"  [GenModelManager] Grown to {new_model.param_count():,} params")
        return True

    def shrink(self) -> bool:
        """Shrink the gen model to a smaller size with logit-matching distillation.

        Returns:
            True if shrink was performed.
        """
        if not self.should_shrink():
            return False

        target = self.get_target_size()
        old_model = self.gen_model
        old_size = old_model.get_size_config()

        print(f"  [GenModelManager] Shrinking: {old_size['d_model']}d x "
              f"{old_size['n_layers']}L -> {target['d_model']}d x "
              f"{target['n_layers']}L")

        # Build new smaller model
        new_model = LLMGenModel(
            config_name=self.config_name,
            device=str(self.device),
            **target,
        )

        # Weight distillation (copy/interpolate where possible)
        self._distill_weights(old_model, new_model)

        # Logit-matching distillation (primary method for shrink)
        self._distill_logits(old_model, new_model, _DEFAULT_CALIBRATION_PROMPTS)

        # Replace model
        self.gen_model = new_model
        del old_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Reset trackers
        self._overperform_count = 0
        self._plateau_count = 0

        print(f"  [GenModelManager] Shrunk to {new_model.param_count():,} params")
        return True

    # ── Distillation ──────────────────────────────────────────────────

    def _distill(self, old_model: LLMGenModel, new_model: LLMGenModel,
                 calibration_prompts: list[str]) -> None:
        """Logit-matching distillation from old model to new model.

        Runs calibration prompts through both models and minimizes the
        KL divergence between their output logit distributions. This
        transfers knowledge from the old (larger/smaller) model to the
        new one.

        Args:
            old_model: Teacher model (frozen).
            new_model: Student model (trained).
            calibration_prompts: Prompts to use for distillation.
        """
        self._distill_weights(old_model, new_model)
        self._distill_logits(old_model, new_model, calibration_prompts)

    def _distill_weights(self, old_model: LLMGenModel,
                         new_model: LLMGenModel) -> None:
        """Transfer weights from old model to new model (interpolate/zero-init).

        For parameters with matching shapes: copy directly.
        For parameters with different shapes: bilinear interpolate if 2D,
        zero-init otherwise.
        For new parameters (not in old model): zero-init (already done by
        random init, but we zero them for cleaner distillation start).
        """
        old_state = old_model.model.state_dict()
        new_state = new_model.model.state_dict()

        copied = 0
        interpolated = 0
        zeroed = 0

        with torch.no_grad():
            for name, new_param in new_state.items():
                if name not in old_state:
                    # New parameter (e.g. new layers) — zero-init
                    new_param.zero_()
                    zeroed += 1
                    continue

                old_param = old_state[name]
                if old_param.shape == new_param.shape:
                    # Direct copy
                    new_param.copy_(old_param.to(new_param.device,
                                                 dtype=new_param.dtype))
                    copied += 1
                elif old_param.dim() == 2 and new_param.dim() == 2:
                    # 2D weight: bilinear interpolate
                    interpolated += 1
                    new_param.copy_(
                        _interpolate_2d(old_param, new_param.shape)
                        .to(new_param.device, dtype=new_param.dtype)
                    )
                elif old_param.dim() == 1 and new_param.dim() == 1:
                    # 1D weight (norm, bias): interpolate or truncate
                    interpolated += 1
                    new_param.copy_(
                        _interpolate_1d(old_param, new_param.shape[0])
                        .to(new_param.device, dtype=new_param.dtype)
                    )
                else:
                    # Can't interpolate — zero-init
                    new_param.zero_()
                    zeroed += 1

        print(f"  [Distill] Weights: {copied} copied, {interpolated} "
              f"interpolated, {zeroed} zero-init")

    def _distill_logits(self, old_model: LLMGenModel,
                        new_model: LLMGenModel,
                        calibration_prompts: list[str]) -> None:
        """Logit-matching distillation via KL divergence.

        Runs calibration prompts through both models and trains the new
        model to match the old model's output logit distributions.

        Args:
            old_model: Teacher model (frozen, eval mode).
            new_model: Student model (train mode).
            calibration_prompts: Prompts for distillation.
        """
        if not calibration_prompts:
            return

        # Prepare calibration data: tokenize prompts
        tokenizer = new_model.tokenizer
        max_len = min(new_model.config.max_seq_len, 256)

        input_ids_list = []
        for prompt in calibration_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids
            if hasattr(ids, "to"):
                ids = ids.to(self.device)
            input_ids_list.append(ids)

        if not input_ids_list:
            return

        # Set up training
        new_model.train_mode()
        old_model.eval_mode()

        # Freeze old model
        for p in old_model.model.parameters():
            p.requires_grad = False

        # Optimizer: AdamW on new model parameters
        optimizer = torch.optim.AdamW(
            new_model.model.parameters(),
            lr=DISTILL_LR,
            weight_decay=0.01,
        )

        # Autocast context
        use_autocast = self.device.type == "cuda"
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_autocast
            else nullcontext()
        )

        for epoch in range(DISTILL_EPOCHS):
            epoch_loss = 0.0
            n_batches = 0

            for input_ids in input_ids_list:
                optimizer.zero_grad()

                with autocast_ctx:
                    # Teacher forward (no grad)
                    with torch.no_grad():
                        teacher_out = old_model.model(input_ids)
                        teacher_logits = unpack_output_with_kv(teacher_out)[0]

                    # Student forward
                    student_out = new_model.model(input_ids)
                    student_logits = unpack_output_with_kv(student_out)[0]

                    # KL divergence loss on logits
                    # Use temperature-scaled softmax for softer distributions
                    T = DISTILL_TEMPERATURE
                    teacher_probs = F.softmax(teacher_logits.float() / T, dim=-1)
                    student_log_probs = F.log_softmax(
                        student_logits.float() / T, dim=-1
                    )
                    # KL(teacher || student) = sum teacher * (log(teacher) - log(student))
                    # We minimize -teacher * log(student) (cross-entropy)
                    loss = F.kl_div(
                        student_log_probs, teacher_probs,
                        reduction="batchmean", log_target=False
                    ) * (T * T)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(new_model.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  [Distill] Epoch {epoch+1}/{DISTILL_EPOCHS}: "
                  f"loss={avg_loss:.4f}")

        # Unfreeze old model (in case it's reused)
        for p in old_model.model.parameters():
            p.requires_grad = True

        new_model.eval_mode()

    # ── Fine-tuning ───────────────────────────────────────────────────

    def fine_tune_on_solutions(
        self,
        solutions: list[dict],
        epochs: int = FINE_TUNE_EPOCHS,
        lr: float = FINE_TUNE_LR,
        batch_size: int = FINE_TUNE_BATCH_SIZE,
        use_cpu_adamw: bool = False,
    ) -> dict:
        """Fine-tune the gen model on high-scoring solutions.

        Uses cross-entropy loss on the output tokens. Solutions are dicts
        with {"input": str, "output": str, "score": float}. Only solutions
        with score >= FINE_TUNE_SCORE_THRESHOLD are used for training.

        Args:
            solutions: List of solution dicts.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Training batch size.
            use_cpu_adamw: If True, use CPUAdamW (optimizer state on CPU)
                for VRAM-constrained training.

        Returns:
            Dict with training stats {"loss_history": [...], "n_examples": int}.
        """
        # Filter high-scoring solutions
        good_solutions = [
            s for s in solutions if s.get("score", 0) >= FINE_TUNE_SCORE_THRESHOLD
        ]
        if not good_solutions:
            return {"loss_history": [], "n_examples": 0}

        tokenizer = self.gen_model.tokenizer
        max_len = self.gen_model.config.max_seq_len

        # Prepare training data: tokenize input + output, compute loss only
        # on output tokens.
        train_examples = []
        for sol in good_solutions:
            input_text = sol["input"]
            output_text = sol["output"]

            # Tokenize input and output separately to find the boundary
            input_ids = tokenizer(input_text, return_tensors="pt",
                                  truncation=True, max_length=max_len // 2).input_ids
            output_ids = tokenizer(output_text, return_tensors="pt",
                                   truncation=True, max_length=max_len // 2).input_ids

            # Concatenate: input + output + EOS
            if hasattr(input_ids, "shape"):
                input_len = input_ids.shape[1]
            else:
                input_len = len(input_ids)

            eos_id = self.gen_model.eos_token_id
            full_ids = torch.cat([
                input_ids.flatten(),
                output_ids.flatten(),
                torch.tensor([eos_id], dtype=input_ids.dtype),
            ])
            full_ids = full_ids[:max_len]

            # Targets: shift by 1 (predict next token)
            # Loss mask: only compute loss on output tokens (positions >= input_len)
            train_examples.append({
                "input_ids": full_ids,
                "input_len": input_len,
            })

        if not train_examples:
            return {"loss_history": [], "n_examples": 0}

        # Set up training
        self.gen_model.train_mode()
        device = self.device

        # Optimizer
        if use_cpu_adamw or self.device.type == "cpu":
            # CPUAdamW: keep optimizer state on CPU to save VRAM
            optimizer = _CPUAdamW(
                self.gen_model.model.parameters(),
                lr=lr,
                weight_decay=0.01,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.gen_model.model.parameters(),
                lr=lr,
                weight_decay=0.01,
            )

        # Autocast context
        use_autocast = self.device.type == "cuda"
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_autocast
            else nullcontext()
        )

        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle examples
            indices = torch.randperm(len(train_examples)).tolist()

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_examples = [train_examples[j] for j in batch_indices]

                # Pad batch to same length
                max_seq = max(ex["input_ids"].shape[0] for ex in batch_examples)
                batch_ids = torch.zeros(
                    len(batch_examples), max_seq,
                    dtype=batch_examples[0]["input_ids"].dtype,
                    device=device,
                )
                batch_targets = torch.full(
                    (len(batch_examples), max_seq), -100,
                    dtype=torch.long, device=device,
                )
                # -100 in CE loss = ignore index

                for j, ex in enumerate(batch_examples):
                    ids = ex["input_ids"].to(device)
                    seq_len = ids.shape[0]
                    input_len = ex["input_len"]

                    batch_ids[j, :seq_len] = ids
                    # Targets: predict next token, loss only on output tokens
                    # Position t predicts t+1, so target[t] = ids[t+1]
                    # Loss mask: t >= input_len - 1 (output region)
                    if seq_len > 1:
                        batch_targets[j, :seq_len - 1] = ids[1:]
                        # Mask out input region (set to -100)
                        batch_targets[j, :input_len - 1] = -100

                optimizer.zero_grad()

                with autocast_ctx:
                    out = self.gen_model.model(batch_ids)
                    logits = unpack_output_with_kv(out)[0]
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch_targets.view(-1),
                        ignore_index=-100,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.gen_model.model.parameters(), 1.0
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)
            print(f"  [FineTune] Epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss:.4f} ({len(good_solutions)} examples)")

        self.gen_model.eval_mode()

        return {
            "loss_history": loss_history,
            "n_examples": len(good_solutions),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save_state(self, state_path: str | Path | None = None) -> None:
        """Save gen model state + manager state to DB and/or file.

        Args:
            state_path: Optional file path for model weights. If None,
                saves to a temp file and stores the blob in the DB.
        """
        # Save model weights to file if path provided
        if state_path is not None:
            self.gen_model.save_state(state_path)

        # Save manager state to DB
        if self.db is not None:
            self._save_to_db()

    def _save_to_db(self) -> None:
        """Save manager state to FindingsDB."""
        manager_state = {
            "config_name": self.config_name,
            "round_counter": self.round_counter,
            "domain_history": self.domain_history,
            "aggregate_history": self.aggregate_history,
            "golden_ratio_A": self.golden_ratio_A,
            "golden_ratio_B": self.golden_ratio_B,
            "golden_ratio_fitted": self.golden_ratio_fitted,
            "plateau_count": self._plateau_count,
            "overperform_count": self._overperform_count,
            "last_aggregate_score": self._last_aggregate_score,
            "last_growth_dim": self._last_growth_dim,
            "current_size": self.gen_model.get_size_config(),
            "param_count": self.gen_model.param_count(),
        }

        # Save model weights as blob
        model_state = {}
        for name, p in self.gen_model.model.named_parameters():
            model_state[name] = p.detach().cpu().numpy()
        # Also save buffers (norm weights, etc.)
        for name, b in self.gen_model.model.named_buffers():
            if b is not None:
                model_state[name] = b.detach().cpu().numpy()

        c = self.db.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS gen_model_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                manager_state_json TEXT NOT NULL,
                weights_blob BLOB NOT NULL,
                config_json TEXT NOT NULL,
                param_count INTEGER NOT NULL,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)
        self.db.conn.commit()

        c.execute("""
            INSERT OR REPLACE INTO gen_model_state
            (id, manager_state_json, weights_blob, config_json, param_count)
            VALUES (1, ?, ?, ?, ?)
        """, (
            json.dumps(manager_state),
            pickle.dumps(model_state),
            json.dumps(self.gen_model.config.__dict__),
            self.gen_model.param_count(),
        ))
        self.db.conn.commit()

    def _load_from_db(self) -> bool:
        """Load manager state from FindingsDB.

        Returns:
            True if state was loaded.
        """
        c = self.db.conn.cursor()
        # Ensure table exists
        c.execute("""
            CREATE TABLE IF NOT EXISTS gen_model_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                manager_state_json TEXT NOT NULL,
                weights_blob BLOB NOT NULL,
                config_json TEXT NOT NULL,
                param_count INTEGER NOT NULL,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)
        self.db.conn.commit()

        c.execute("""
            SELECT manager_state_json, weights_blob, config_json, param_count
            FROM gen_model_state WHERE id = 1
        """)
        row = c.fetchone()
        if row is None:
            return False

        manager_state = json.loads(row[0])
        weights = pickle.loads(row[1])
        saved_config = json.loads(row[2])

        # Restore manager state
        self.config_name = manager_state.get("config_name", self.config_name)
        self.round_counter = manager_state.get("round_counter", 0)
        self.domain_history = manager_state.get("domain_history", {})
        self.aggregate_history = manager_state.get("aggregate_history", [])
        self.golden_ratio_A = manager_state.get("golden_ratio_A", 0.1)
        self.golden_ratio_B = manager_state.get("golden_ratio_B", -1.0)
        self.golden_ratio_fitted = manager_state.get("golden_ratio_fitted", False)
        self._plateau_count = manager_state.get("plateau_count", 0)
        self._overperform_count = manager_state.get("overperform_count", 0)
        self._last_aggregate_score = manager_state.get("last_aggregate_score", 0.0)
        self._last_growth_dim = manager_state.get("last_growth_dim", "n_layers")

        # Check if model size changed — rebuild if needed
        current_size = self.gen_model.get_size_config()
        saved_size = manager_state.get("current_size", current_size)
        needs_rebuild = False
        for key in ("d_model", "n_layers", "n_kv_heads", "intermediate_size",
                     "nlrq_rank"):
            if saved_size.get(key) != current_size.get(key):
                needs_rebuild = True
                break

        if needs_rebuild:
            # Rebuild with saved dimensions
            overrides = {}
            for key in ("d_model", "n_layers", "n_heads", "n_kv_heads",
                        "intermediate_size", "vocab_size", "max_seq_len",
                        "nlrq_rank"):
                if key in saved_size:
                    overrides[key] = saved_size[key]
            self.gen_model = LLMGenModel(
                config_name=self.config_name,
                device=str(self.device),
                **overrides,
            )

        # Load model weights
        current_state = self.gen_model.model.state_dict()
        loaded = 0
        with torch.no_grad():
            for name, param in current_state.items():
                if name in weights:
                    saved = torch.from_numpy(weights[name])
                    if saved.shape == param.shape:
                        param.copy_(saved.to(param.device, dtype=param.dtype))
                        loaded += 1

        print(f"  [GenModelManager] Loaded state: {loaded} params, "
              f"{self.round_counter} rounds, "
              f"params={self.gen_model.param_count():,}")
        return True

    # ── Utilities ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a summary of the current manager state."""
        return {
            "rounds": self.round_counter,
            "param_count": self.gen_model.param_count(),
            "current_size": self.gen_model.get_size_config(),
            "aggregate_performance": self.get_current_performance(),
            "golden_ratio": self.get_golden_ratio_curve(),
            "plateau_count": self._plateau_count,
            "overperform_count": self._overperform_count,
            "should_grow": self.should_grow(),
            "should_shrink": self.should_shrink(),
            "domains_tracked": list(self.domain_history.keys()),
        }

    def __repr__(self) -> str:
        return (f"GenModelManager(rounds={self.round_counter}, "
                f"params={self.gen_model.param_count():,}, "
                f"perf={self.get_current_performance():.3f})")

    # ── Wipe / Reset ──────────────────────────────────────────────────

    def wipe_knowledge(self) -> dict:
        """Wipe all gen model knowledge and reset to a fresh state.

        Deletes:
          - gen_model_state row (id=1): saved weights + manager state
          - all gen_models rows: gen model checkpoints
          - all gen_model_performance rows: performance history

        Resets in-memory state to initial values and re-initializes the
        gen model to a fresh model with the default config.

        Returns:
            Dict with deletion counts:
            {"gen_model_state_deleted": bool,
             "gen_models_deleted": int,
             "gen_model_performance_deleted": int}
        """
        state_deleted = False
        gen_models_deleted = 0
        perf_deleted = 0

        # ── Wipe DB tables ────────────────────────────────────────────
        if self.db is not None:
            c = self.db.conn.cursor()

            # gen_model_state (single row, id=1)
            c.execute("DELETE FROM gen_model_state WHERE id = 1")
            state_deleted = c.rowcount > 0

            # gen_models (all checkpoints)
            c.execute("SELECT COUNT(*) FROM gen_models")
            gen_models_deleted = c.fetchone()[0]
            c.execute("DELETE FROM gen_models")

            # gen_model_performance (all history)
            c.execute("SELECT COUNT(*) FROM gen_model_performance")
            perf_deleted = c.fetchone()[0]
            c.execute("DELETE FROM gen_model_performance")

            self.db.conn.commit()

        # ── Reset in-memory state ─────────────────────────────────────
        self.round_counter = 0
        self.domain_history = {}
        self.aggregate_history = []
        self.golden_ratio_A = 0.1
        self.golden_ratio_B = -1.0
        self.golden_ratio_fitted = False
        self._plateau_count = 0
        self._overperform_count = 0
        self._last_aggregate_score = 0.0
        self._last_growth_dim = "n_layers"

        # ── Re-initialize gen model to fresh default config ───────────
        del self.gen_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.gen_model = LLMGenModel(
            config_name=self.config_name,
            device=str(self.device),
            tokenizer_path=self.tokenizer_path,
        )

        print(f"  [GenModelManager] Wiped knowledge: "
              f"state_deleted={state_deleted}, "
              f"gen_models_deleted={gen_models_deleted}, "
              f"perf_deleted={perf_deleted}")

        return {
            "gen_model_state_deleted": state_deleted,
            "gen_models_deleted": gen_models_deleted,
            "gen_model_performance_deleted": perf_deleted,
        }


# ── Helper Functions ───────────────────────────────────────────────────


def _interpolate_2d(old: torch.Tensor, new_shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear interpolate a 2D weight tensor to a new shape.

    Uses F.interpolate for smooth resampling. Handles both grow and shrink.

    Args:
        old: (out_features, in_features) weight tensor.
        new_shape: (new_out, new_in) target shape.

    Returns:
        (new_out, new_in) interpolated weight tensor.
    """
    old_out, old_in = old.shape
    new_out, new_in = new_shape

    # Add batch/channel dims for F.interpolate: (1, 1, out, in) -> (1, 1, new_out, new_in)
    x = old.float().unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(new_out, new_in), mode="bilinear",
                      align_corners=False)
    return x.squeeze(0).squeeze(0).to(old.dtype)


def _interpolate_1d(old: torch.Tensor, new_len: int) -> torch.Tensor:
    """Linear interpolate a 1D tensor to a new length.

    Args:
        old: (old_len,) tensor.
        new_len: Target length.

    Returns:
        (new_len,) interpolated tensor.
    """
    old_len = old.shape[0]
    if old_len == new_len:
        return old.clone()
    # Use linear interpolation via F.interpolate
    x = old.float().unsqueeze(0).unsqueeze(0)  # (1, 1, old_len)
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return x.squeeze(0).squeeze(0).to(old.dtype)


class _CPUAdamW:
    """CPU-state AdamW optimizer for VRAM-constrained training.

    Keeps optimizer state (momentum, variance) on CPU while model parameters
    stay on GPU. Gradients are moved to CPU for the optimizer step, then
    updated parameters are moved back to GPU.

    This enables training models that would OOM with standard AdamW (which
    stores 2x param count in GPU VRAM for optimizer state).

    For very small models (like gen_model_tiny), standard AdamW on GPU is
    fine. This is mainly for when the model grows larger.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.params = [p for p in params if p.requires_grad]
        # Optimizer state on CPU: (momentum, variance) per param
        self.m = [torch.zeros_like(p, device="cpu") for p in self.params]
        self.v = [torch.zeros_like(p, device="cpu") for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        self.t += 1
        bias_c1 = 1 - self.betas[0] ** self.t
        bias_c2 = 1 - self.betas[1] ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # Move gradient to CPU for update
            grad = p.grad.detach().cpu()

            # Update moments
            self.m[i].mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
            self.v[i].mul_(self.betas[1]).addcmul_(
                grad, grad, value=1 - self.betas[1])

            # Bias correction
            m_hat = self.m[i] / bias_c1
            v_hat = self.v[i] / bias_c2

            # Compute update
            update = m_hat / (v_hat.sqrt() + self.eps)
            if self.weight_decay > 0:
                # Decoupled weight decay
                update = update + self.weight_decay * p.detach().cpu()

            # Apply update to parameter (on its original device)
            p.data.add_(-self.lr * update.to(p.device, dtype=p.dtype))

    def state_dict(self):
        return {
            "t": self.t,
            "m": [m.clone() for m in self.m],
            "v": [v.clone() for v in self.v],
        }

    def load_state_dict(self, state):
        self.t = state["t"]
        self.m = [m.clone() for m in state["m"]]
        self.v = [v.clone() for v in state["v"]]


# ── Standalone Wipe Function ───────────────────────────────────────────


def wipe_gen_model_knowledge(db_path: str = "forge_evolve.db") -> dict:
    """Nuclear wipe of all gen model knowledge directly from the DB.

    Opens the SQLite DB directly (no GenModelManager instance needed) and
    deletes all rows from:
      - gen_model_state (saved weights + manager state)
      - gen_models (gen model checkpoints)
      - gen_model_performance (performance history)

    This is a destructive, irreversible operation. Use when you want to
    completely reset the gen model subsystem without instantiating a
    GenModelManager (e.g. before a fresh evolution run).

    Args:
        db_path: Path to the ForgeEvolve SQLite database file.

    Returns:
        Dict with deletion results:
        {"gen_model_state_deleted": bool,
         "gen_models_deleted": int,
         "gen_model_performance_deleted": int}
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    state_deleted = False
    gen_models_deleted = 0
    perf_deleted = 0

    try:
        # gen_model_state (single row, id=1) — table may not exist yet
        try:
            c.execute("DELETE FROM gen_model_state WHERE id = 1")
            state_deleted = c.rowcount > 0
        except sqlite3.OperationalError:
            pass  # table doesn't exist — nothing to wipe

        # gen_models (all checkpoints) — table may not exist yet
        try:
            c.execute("SELECT COUNT(*) FROM gen_models")
            gen_models_deleted = c.fetchone()[0]
            c.execute("DELETE FROM gen_models")
        except sqlite3.OperationalError:
            pass

        # gen_model_performance (all history) — table may not exist yet
        try:
            c.execute("SELECT COUNT(*) FROM gen_model_performance")
            perf_deleted = c.fetchone()[0]
            c.execute("DELETE FROM gen_model_performance")
        except sqlite3.OperationalError:
            pass

        conn.commit()
    finally:
        conn.close()

    print(f"  [wipe_gen_model_knowledge] Wiped {db_path}: "
          f"state_deleted={state_deleted}, "
          f"gen_models_deleted={gen_models_deleted}, "
          f"perf_deleted={perf_deleted}")

    return {
        "gen_model_state_deleted": state_deleted,
        "gen_models_deleted": gen_models_deleted,
        "gen_model_performance_deleted": perf_deleted,
    }
