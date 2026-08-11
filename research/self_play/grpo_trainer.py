"""GRPO (Group Relative Policy Optimization) trainer for self-play RL.

Implements the GRPO algorithm from DeepSeek-R1, adapted for small models (1-3B)
with MC-GRPO median baseline for stability with small group sizes.

Research basis:
- DeepSeek-R1: GRPO with verifiable rewards, KL penalty against reference model
- MC-GRPO: median baseline for G=2-4 (reduces gap to <1% vs G=8)
- AVSPO: Advantage Collapse Rate monitoring (alert if >0.3)
- "Survive or Collapse" (arxiv 2605.22217): data gate is binding constraint,
  not reward design. Executor verification (test_passed=True) IS the strict gate.

Hyperparameters (small model optimal, 1-3B):
  learning_rate: 5e-6
  kl_coefficient: 0.02
  clip_range: 0.2
  group_size: 4 (use MC-GRPO median baseline)
  temperature: 0.8
  max_grad_norm: 1.0

Reward design:
  solver_reward = 1.0 if test_passed else 0.0  (binary from executor = the gate)
  proposer_reward = 1.0 - |success_rate - 0.5|  (AZR learnability)
  Invalid task penalty: -0.1

Usage:
    from research.self_play.grpo_trainer import GRPOTrainer
    trainer = GRPOTrainer(model, tokenizer, ref_model, device="cuda")
    stats = trainer.train_step(prompts, generate_fn, verify_fn)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """GRPO hyperparameters optimized for 1-3B models."""
    learning_rate: float = 5e-6
    kl_coefficient: float = 0.02       # β in KL(π || π_ref)
    clip_range: float = 0.2            # ε for PPO-style ratio clipping
    group_size: int = 4                # G completions per prompt (MC-GRPO)
    temperature: float = 0.8           # generation temperature
    max_grad_norm: float = 1.0
    max_seq_len: int = 512             # max sequence length for training
    use_median_baseline: bool = True   # MC-GRPO: median instead of mean
    invalid_penalty: float = -0.1      # penalty for invalid/unverifiable tasks
    grad_accum_steps: int = 2          # gradient accumulation


@dataclass
class GRPOStats:
    """Tracks GRPO training metrics for monitoring."""
    total_steps: int = 0
    total_rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    kl_divergences: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    advantage_collapse_count: int = 0  # ACR numerator
    # Advantage Collapse Rate (ACR): fraction of groups where all advantages ≈ 0
    # Alert if ACR > 0.3 (AVSPO paper)

    @property
    def mean_reward(self) -> float:
        return sum(self.total_rewards) / max(len(self.total_rewards), 1)

    @property
    def mean_advantage(self) -> float:
        return sum(self.advantages) / max(len(self.advantages), 1)

    @property
    def mean_kl(self) -> float:
        return sum(self.kl_divergences) / max(len(self.kl_divergences), 1)

    @property
    def advantage_collapse_rate(self) -> float:
        """ACR = fraction of steps with collapsed advantages.
        Alert if > 0.3 (AVSPO)."""
        if self.total_steps == 0:
            return 0.0
        return self.advantage_collapse_count / self.total_steps


class GRPOTrainer:
    """GRPO trainer for self-play RL with verifiable rewards.

    The executor-based verification (test_passed=True) serves as the strict
    data gate (ε=0). This is the binding constraint on stability — NOT reward
    design ("Survive or Collapse" paper). Never train on unverified solutions.
    """

    def __init__(self, model, tokenizer, ref_model,
                 device: str = "cuda",
                 config: GRPOConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model  # frozen reference model for KL penalty
        self.device = device
        self.config = config or GRPOConfig()
        self.stats = GRPOStats()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizer (only trainable params — assumes LoRA or similar)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if trainable:
            self.optimizer = torch.optim.AdamW(
                trainable, lr=self.config.learning_rate, weight_decay=0.01)
        else:
            self.optimizer = None

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """Compute GRPO group-relative advantages.

        MC-GRPO: use median baseline for robustness with small groups.
        advantage = (reward - baseline) / (group_std + eps)
        """
        if not rewards:
            return []

        if self.config.use_median_baseline and len(rewards) >= 2:
            # MC-GRPO: median is more robust to outliers than mean
            sorted_r = sorted(rewards)
            n = len(sorted_r)
            if n % 2 == 1:
                baseline = sorted_r[n // 2]
            else:
                baseline = (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
        else:
            baseline = sum(rewards) / len(rewards)

        # Group std (population std for small groups)
        if len(rewards) > 1:
            variance = sum((r - baseline) ** 2 for r in rewards) / len(rewards)
            std = math.sqrt(variance + 1e-8)
        else:
            std = 1.0

        advantages = [(r - baseline) / (std + 1e-8) for r in rewards]

        # Track advantage collapse (all advantages ≈ 0)
        if all(abs(a) < 0.01 for a in advantages):
            self.stats.advantage_collapse_count += 1

        return advantages

    def compute_kl_penalty(self, input_ids: torch.Tensor,
                           current_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL(π_current || π_ref) per token.

        KL = Σ p_current * log(p_current / p_ref)
        Approximated per-token, averaged over sequence.
        """
        with torch.no_grad():
            ref_logits, _ = self.ref_model(input_ids)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_probs = ref_log_probs.exp()

        curr_log_probs = F.log_softmax(current_logits, dim=-1)
        curr_probs = curr_log_probs.exp()

        # KL divergence per token: sum over vocab
        kl_per_token = (curr_probs * (curr_log_probs - ref_log_probs)).sum(dim=-1)
        return kl_per_token.mean()

    def train_step(self, prompts: list[str],
                   completions: list[list[str]],
                   rewards: list[list[float]],
                   prompt_token_lens: list[int] | None = None) -> dict:
        """One GRPO training step.

        Args:
            prompts: list of prompt strings (B prompts)
            completions: list of lists — completions[g] for each prompt (G per prompt)
            rewards: list of lists — rewards[g] for each completion (binary from executor)
            prompt_token_lens: optional pre-computed prompt lengths

        Returns:
            stats dict with loss, reward, KL, ACR
        """
        if not self.optimizer:
            return {"error": "no trainable parameters"}

        self.model.train()
        total_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        n_updates = 0
        accum_count = 0

        for prompt_idx, (prompt, comps, rews) in enumerate(zip(prompts, completions, rewards)):
            if not comps or len(comps) < 2:
                continue

            # Compute group-relative advantages
            advantages = self.compute_advantages(rews)
            self.stats.advantages.extend(advantages)
            self.stats.total_rewards.extend(rews)

            for comp_idx, (completion, reward, advantage) in enumerate(
                    zip(comps, rews, advantages)):
                # Skip zero-advantage samples (no gradient signal)
                if abs(advantage) < 1e-6:
                    continue

                # Tokenize full sequence
                full_text = prompt + completion
                enc = self.tokenizer(full_text, return_tensors="pt",
                                     truncation=True, max_length=self.config.max_seq_len)
                input_ids = enc.input_ids.to(self.device)

                # Find prompt boundary
                if prompt_token_lens:
                    prompt_len = prompt_token_lens[prompt_idx]
                else:
                    prompt_len = self.tokenizer(
                        prompt, return_tensors="pt").input_ids.shape[1]

                if input_ids.shape[1] <= prompt_len + 1:
                    continue  # completion too short

                # Forward pass (current policy)
                logits, _ = self.model(input_ids)
                solution_logits = logits[0, prompt_len - 1:-1, :]
                solution_targets = input_ids[0, prompt_len:].long()

                if solution_logits.shape[0] == 0:
                    continue

                # Compute log probabilities for the generated tokens
                log_probs = F.log_softmax(solution_logits, dim=-1)
                token_log_probs = log_probs.gather(1, solution_targets.unsqueeze(0)).squeeze(0)

                # Compute old log probs (for PPO ratio) — use current as "old" for first pass
                # In a full implementation, this would be from the rollout model
                with torch.no_grad():
                    old_log_probs = token_log_probs.detach()

                # PPO ratio
                ratio = (token_log_probs - old_log_probs).exp()

                # Clipped policy gradient loss
                clipped_ratio = ratio.clamp(1 - self.config.clip_range,
                                            1 + self.config.clip_range)
                pg_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

                # KL penalty
                kl = self.compute_kl_penalty(input_ids, logits)
                kl_loss = self.config.kl_coefficient * kl

                # Total loss
                loss = (pg_loss + kl_loss) / self.config.grad_accum_steps
                loss.backward()
                accum_count += 1

                total_loss += loss.item() * self.config.grad_accum_steps
                total_kl += kl.item()
                total_reward += reward
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

        # Update stats
        self.stats.total_steps += 1
        self.stats.policy_losses.append(total_loss / max(n_updates, 1))
        self.stats.kl_divergences.append(total_kl / max(n_updates, 1))

        acr = self.stats.advantage_collapse_rate
        acr_alert = " [WARNING: ACR>0.3]" if acr > 0.3 else ""

        return {
            "n_updates": n_updates,
            "mean_loss": total_loss / max(n_updates, 1),
            "mean_reward": total_reward / max(n_updates, 1),
            "mean_kl": total_kl / max(n_updates, 1),
            "advantage_collapse_rate": acr,
            "acr_alert": acr_alert,
        }

    def get_stats(self) -> dict:
        """Return cumulative training stats for monitoring."""
        return {
            "total_steps": self.stats.total_steps,
            "mean_reward": self.stats.mean_reward,
            "mean_advantage": self.stats.mean_advantage,
            "mean_kl": self.stats.mean_kl,
            "mean_policy_loss": (
                sum(self.stats.policy_losses) / max(len(self.stats.policy_losses), 1)
            ),
            "advantage_collapse_rate": self.stats.advantage_collapse_rate,
            "acr_alert": self.stats.advantage_collapse_rate > 0.3,
        }
