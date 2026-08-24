"""RPO (Root-token Policy Optimization) for Adaptive Thinking.

Implements the RPO algorithm from ACL 2026 (Kim et al.):
  "Efficiently Learning To Reason or Not to Reason: Root-token Policy
   Optimization for Adaptive Thinking"

Core idea: Train ONLY the first generated token (root token) to decide
whether to think (emit <think> tag) or not (answer directly). This is
the pivotal branching point — the rest of the generation follows from
this single decision.

Key advantages over full GRPO for adaptive thinking:
  - 2% of training compute (only 1 token's gradient vs hundreds)
  - Minimal VRAM (no full-sequence backprop)
  - Difficulty-aware: model learns to think on hard problems, skip on easy ones
  - ~50% token reduction at inference with accuracy maintained/improved

Algorithm:
  1. For each prompt, generate G completions: some with thinking, some without
  2. Score each completion with verifiable reward
  3. Compute GRPO-style group-relative advantages
  4. Forward pass on prompt only, get logits at the root position
  5. Compute loss ONLY on the root token's log-prob, weighted by advantage
  6. Backprop — only the root token's gradient flows through the model

Usage:
  from research.training.runners.rpo_train import RPOTrainer, RPOConfig
  trainer = RPOTrainer(model, tokenizer, config=RPOConfig())
  trainer.train_step(prompts, completions, rewards, think_flags)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any

from research.training.training_utils import oom_guard


@dataclass
class RPOConfig:
    """RPO hyperparameters (Root-token Policy Optimization)."""
    learning_rate: float = 1e-5       # lower than GRPO — only root token updates
    group_size: int = 4               # G completions per prompt (mix think/no-think)
    think_ratio: float = 0.5          # fraction of completions that use thinking
    clip_range: float = 0.2           # PPO-style clipping on root token ratio
    max_grad_norm: float = 1.0
    max_seq_len: int = 512            # for prompt tokenization
    grad_accum_steps: int = 2
    # Entropy bonus on root token to maintain exploration of think/no-think
    entropy_coefficient: float = 0.01
    # Temperature for generating completions during rollout
    generation_temperature: float = 0.8
    # Whether to use median baseline (MC-GRPO style)
    use_median_baseline: bool = True
    # Device
    device: str = "cuda"
    # Optimizer type
    optimizer: str = "adamw"  # "adamw" or "cpu_offload"


class RPOTrainer:
    """Root-token Policy Optimization trainer.

    Trains the model to adaptively decide whether to think or not based on
    problem difficulty. Only the root token (first generated token) receives
    gradient updates — the rest of the model is frozen during RPO training.

    This is dramatically cheaper than full GRPO:
    - 1 token backprop vs ~200-500 tokens
    - No reference model needed (no KL)
    - No full-sequence forward pass for gradient computation
    """

    def __init__(self, model, tokenizer, config: RPOConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RPOConfig()
        self.device = self.config.device

        # Optimizer (only trainable params)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if trainable:
            if self.config.optimizer == "cpu_offload":
                from research.training.optim.hybrid_offload import CPUAdamW
                self.optimizer = CPUAdamW(
                    trainable, lr=self.config.learning_rate, weight_decay=0.01)
                print("  [RPO] Using CPUAdamW (optimizer states on CPU)")
            else:
                self.optimizer = torch.optim.AdamW(
                    trainable, lr=self.config.learning_rate, weight_decay=0.01)
        else:
            self.optimizer = None

        self.total_steps = 0
        self._think_token_id = None  # auto-detected on first step

    def _detect_think_token(self) -> int:
        """Detect the thinking token ID for this tokenizer.

        LFM2.5 uses Qwen-style format. The think token is typically the
        first token of <think> or a similar reasoning marker.
        """
        if self._think_token_id is not None:
            return self._think_token_id

        # Try common thinking markers
        candidates = ["<think>", "<thinking>", "Thinking", "Let me think"]
        for marker in candidates:
            ids = self.tokenizer(marker, add_special_tokens=False).input_ids
            if ids:
                self._think_token_id = ids[0]
                print(f"  [RPO] Think token: '{marker}' -> id {self._think_token_id}")
                return self._think_token_id

        # Fallback: use a newline or space token as the "no-think" marker
        # and pick an arbitrary token as "think"
        self._think_token_id = self.tokenizer("\n", add_special_tokens=False).input_ids[0]
        print(f"  [RPO] Fallback think token id: {self._think_token_id}")
        return self._think_token_id

    def _compute_advantages(self, rewards: list[float]) -> list[float]:
        """GRPO-style group-relative advantages."""
        if not rewards:
            return []

        if self.config.use_median_baseline and len(rewards) >= 2:
            sorted_r = sorted(rewards)
            n = len(sorted_r)
            if n % 2 == 1:
                baseline = sorted_r[n // 2]
            else:
                baseline = (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
        else:
            baseline = sum(rewards) / len(rewards)

        if len(rewards) > 1:
            variance = sum((r - baseline) ** 2 for r in rewards) / len(rewards)
            std = math.sqrt(variance + 1e-8)
        else:
            std = 1.0

        return [(r - baseline) / (std + 1e-8) for r in rewards]

    def train_step(
        self,
        prompts: list[str],
        completions: list[list[str]],
        rewards: list[list[float]],
        think_flags: list[list[bool]] | None = None,
    ) -> dict:
        """One RPO training step.

        Args:
            prompts: list of N prompts
            completions: N x G completions per prompt
            rewards: N x G verifiable rewards (1.0 correct, 0.0 incorrect)
            think_flags: N x G booleans — whether each completion used thinking.
                         If None, auto-detected from completion content.

        Returns:
            stats dict with loss, reward, think_rate, etc.
        """
        if not self.optimizer:
            return {"error": "no trainable parameters"}

        think_token_id = self._detect_think_token()
        self.model.train()

        total_loss = torch.zeros(1, device=self.device)
        total_reward = 0.0
        total_think = 0
        total_n = 0
        n_updates = 0
        accum_count = 0

        for prompt_idx, (prompt, comps, rews) in enumerate(
                zip(prompts, completions, rewards)):
            if not comps or len(comps) < 2:
                continue

            # Auto-detect think flags if not provided
            if think_flags is None:
                flags = [c.strip().startswith(("<think>", "<thinking>", "Let me"))
                         for c in comps]
            else:
                flags = think_flags[prompt_idx] if prompt_idx < len(think_flags) else []

            # Compute group-relative advantages
            advantages = self._compute_advantages(rews)

            for comp_idx, (completion, reward, advantage, did_think) in enumerate(
                    zip(comps, rews, advantages, flags)):
                if abs(advantage) < 1e-6:
                    continue

                # Tokenize prompt only — we only need the root position logits
                enc = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=self.config.max_seq_len,
                    add_special_tokens=False)
                input_ids = enc.input_ids.to(self.device)
                prompt_len = input_ids.shape[1]

                if prompt_len == 0:
                    continue

                with oom_guard(str(self.device), label="rpo_fwd") as safe:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16,
                        enabled=("cuda" in str(self.device)),
                    ):
                        # Forward pass on prompt only — get logits at last position
                        # (the root token position)
                        logits, _ = self.model(input_ids)

                    if safe.skipped:
                        continue

                    # Root token logits: last position of the prompt
                    root_logits = logits[0, -1, :].float()  # (V,)
                    root_log_probs = F.log_softmax(root_logits, dim=-1)

                    # The target root token: think token if did_think, else
                    # the first token of the completion
                    if did_think:
                        target_token = think_token_id
                    else:
                        # First token of the non-thinking completion
                        comp_ids = self.tokenizer(
                            completion, add_special_tokens=False).input_ids
                        if not comp_ids:
                            continue
                        target_token = comp_ids[0]

                    # Log-prob of the target root token
                    target_log_prob = root_log_probs[target_token]

                    # PPO ratio (root token only — old log-prob ≈ current at start)
                    ratio = torch.exp(target_log_prob - target_log_prob.detach())

                    # Clipped policy gradient loss (on root token only!)
                    clipped_ratio = ratio.clamp(
                        1 - self.config.clip_range, 1 + self.config.clip_range)
                    pg_loss = -torch.min(
                        ratio * advantage, clipped_ratio * advantage
                    )

                    # Entropy bonus on root token (encourages exploration of
                    # think vs no-think)
                    entropy = -(root_log_probs.exp() * root_log_probs).sum()
                    entropy_bonus = self.config.entropy_coefficient * entropy

                    # Total loss = pg_loss - entropy_bonus (minimize)
                    loss = (pg_loss - entropy_bonus) / self.config.grad_accum_steps
                    loss.backward()

                if safe.skipped:
                    # Zero any partial gradients from the failed forward/backward
                    # to prevent corruption of the next accumulation window.
                    self.optimizer.zero_grad()
                    continue
                accum_count += 1

                total_loss += loss.detach() * self.config.grad_accum_steps
                total_reward += reward
                total_think += int(did_think)
                total_n += 1
                n_updates += 1

                # Gradient step
                if accum_count >= self.config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0

        # Final step
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.eval()
        self.total_steps += 1

        think_rate = total_think / max(total_n, 1)
        mean_reward = total_reward / max(n_updates, 1)

        return {
            "n_updates": n_updates,
            "mean_loss": total_loss.item() / max(n_updates, 1),
            "mean_reward": mean_reward,
            "think_rate": think_rate,
            "total_steps": self.total_steps,
        }
