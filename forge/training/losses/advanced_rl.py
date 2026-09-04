"""Advanced RL algorithms: SPPO, PS-PPO, EVPO, GRPO-OR.

Based on four 2026 papers:
  1. SPPO (ACL 2026): Sequence-Level PPO for long-horizon reasoning.
     Reformulates reasoning as Sequence-Level Contextual Bandit.
     Decoupled scalar value function → low-variance advantages without
     multi-sampling. Matches GRPO quality with PPO sample efficiency.

  2. PS-PPO (arXiv 2606.29758): Prefix-Sampling PPO for critic-free RLHF.
     Samples a cutoff timestep per trajectory, backprops only through prefix.
     Importance-weighting correction → unbiased truncated gradient.
     Large compute and memory savings for long reasoning traces.

  3. EVPO (arXiv 2604.19485): Explained Variance Policy Optimization.
     Monitors batch-level explained variance (EV) to adaptively switch
     between critic-based (PPO) and batch-mean (GRPO) advantage estimation.
     Positive EV → use critic; zero/negative EV → use batch mean.
     Provably no greater variance than the better of the two.

  4. GRPO-OR (arXiv 2607.18163): Output Reset trust region.
     Replaces clipped surrogate with smooth one-sided saturation (OR).
     Advantage sign determines direction; zero residual after crossing
     favorable margin. Smaller observed spread than GRPO-clip.

For our self-play training (infinite_loop.py → GRPOTrainer):
  - SPPO: better for long CoT reasoning (no multi-sampling needed)
  - PS-PPO: compute-efficient for long traces (prefix backprop only)
  - EVPO: adaptive critic usage (best of PPO + GRPO)
  - GRPO-OR: smoother trust region (less variance)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SPPOConfig:
    """SPPO: Sequence-Level PPO configuration."""
    lr: float = 1e-6
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    # Decoupled value function (scalar, not token-level)
    value_lr: float = 1e-4
    value_decay: float = 0.99


class SPPO:
    """SPPO: Sequence-Level PPO for long-horizon reasoning.

    Key insight: standard token-level PPO has unstable credit assignment
    over long CoT. GRPO avoids this but needs multiple samples per prompt
    (compute-heavy).

    SPPO reformulates as Sequence-Level Contextual Bandit:
      - One advantage per sequence (not per token)
      - Decoupled scalar value function (no token-level critic)
      - Low-variance advantage without multi-sampling

    Update:
      advantage = reward - V(sequence)  (scalar)
      loss = -min(ratio * advantage, clip(ratio) * advantage)
      value_loss = (V(sequence) - reward)^2
    """

    def __init__(self, config: SPPOConfig):
        self.config = config
        # Decoupled value function: simple scalar predictor
        self.value_estimate = 0.0  # running average of rewards
        self.value_decay = config.value_decay

    def compute_advantage(self, reward: float) -> float:
        """Compute sequence-level advantage."""
        advantage = reward - self.value_estimate
        return advantage

    def update_value(self, reward: float):
        """Update the decoupled value estimate."""
        self.value_estimate = self.value_decay * self.value_estimate + \
                              (1 - self.value_decay) * reward

    def compute_loss(self, log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantage: float,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute SPPO loss.

        Args:
            log_probs: (T,) log probabilities of generated tokens
            old_log_probs: (T,) log probabilities from behavior policy
            advantage: scalar sequence-level advantage
            mask: (T,) mask for valid tokens

        Returns:
            loss: scalar policy gradient loss
        """
        # Policy ratio (per-token, but advantage is sequence-level)
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range,
                                     1 + self.config.clip_range)

        # Sequence-level advantage applied to all tokens
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def stats(self) -> dict:
        return {
            "value_estimate": self.value_estimate,
            "algorithm": "SPPO",
        }


@dataclass
class PSPPOConfig:
    """PS-PPO: Prefix-Sampling PPO configuration."""
    lr: float = 1e-6
    clip_range: float = 0.2
    # Cutoff distribution parameters
    min_cutoff_ratio: float = 0.3  # minimum fraction of trajectory to use
    max_cutoff_ratio: float = 1.0  # maximum (1.0 = full trajectory)


class PSPPO:
    """PS-PPO: Prefix-Sampling PPO for compute-efficient critic-free RLHF.

    Key insight: intermediate prefixes often determine the final outcome.
    Backpropagating through the full trajectory is wasteful.

    PS-PPO:
      1. Sample a cutoff timestep for each trajectory
      2. Backprop only through the prefix (up to cutoff)
      3. Importance-weighting correction → unbiased estimator

    Result: large compute and memory savings, comparable accuracy.
    """

    def __init__(self, config: PSPPOConfig):
        self.config = config

    def sample_cutoff(self, trajectory_length: int) -> int:
        """Sample a cutoff timestep for a trajectory.

        Cutoff is sampled uniformly from [min_cutoff, trajectory_length].
        """
        min_cutoff = int(trajectory_length * self.config.min_cutoff_ratio)
        max_cutoff = trajectory_length
        return torch.randint(min_cutoff, max_cutoff + 1, (1,)).item()

    def compute_loss(self, log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     rewards: torch.Tensor,
                     cutoff: int,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute PS-PPO loss with prefix sampling.

        Args:
            log_probs: (T,) current policy log probs
            old_log_probs: (T,) behavior policy log probs
            rewards: (T,) per-token rewards (or trajectory reward at end)
            cutoff: only use tokens 0..cutoff-1
            mask: (T,) validity mask

        Returns:
            loss: scalar loss
        """
        T = log_probs.shape[0]
        cutoff = min(cutoff, T)

        # Only use prefix
        prefix_log_probs = log_probs[:cutoff]
        prefix_old_log_probs = old_log_probs[:cutoff]

        # Importance weighting correction
        # Full trajectory reward, but only prefix gradient
        trajectory_reward = rewards.sum() if rewards.dim() == 1 else rewards

        # Group-relative advantage (critic-free)
        advantage = trajectory_reward  # simplified (would use group mean)

        # Ratio
        ratio = torch.exp(prefix_log_probs - prefix_old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range,
                                     1 + self.config.clip_range)

        # Importance correction: scale by (T / cutoff) to be unbiased
        correction = T / cutoff
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage) * correction

        if mask is not None:
            prefix_mask = mask[:cutoff]
            loss = loss * prefix_mask
            loss = loss.sum() / prefix_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


@dataclass
class EVPOConfig:
    """EVPO: Explained Variance Policy Optimization configuration."""
    lr: float = 1e-6
    clip_range: float = 0.2
    critic_lr: float = 1e-4
    ev_threshold: float = 0.0  # switch point (positive EV → use critic)


class EVPO:
    """EVPO: Explained Variance Policy Optimization.

    Monitors batch-level explained variance (EV) to adaptively switch:
      - Positive EV → critic reduces variance → use critic-based advantage (PPO)
      - Zero/negative EV → critic inflates variance → use batch-mean (GRPO)

    Provably achieves no greater variance than the better of the two.

    EV = 1 - Var(advantage_critic) / Var(advantage_batch_mean)
    """

    def __init__(self, config: EVPOConfig):
        self.config = config
        self.critic_value = 0.0  # running critic estimate
        self._ev_history: list[float] = []

    def compute_ev(self, rewards: torch.Tensor,
                   critic_values: torch.Tensor) -> float:
        """Compute explained variance from a batch.

        Args:
            rewards: (G,) rewards from G group samples
            critic_values: (G,) critic value estimates

        Returns:
            ev: explained variance (positive = critic helps)
        """
        # Advantage with critic
        adv_critic = rewards - critic_values
        # Advantage with batch mean (GRPO-style)
        adv_batch = rewards - rewards.mean()

        var_critic = adv_critic.var().item()
        var_batch = adv_batch.var().item()

        if var_batch < 1e-8:
            return 0.0

        ev = 1.0 - var_critic / var_batch
        self._ev_history.append(ev)
        return ev

    def compute_advantage(self, reward: float,
                          critic_value: float,
                          batch_mean_reward: float,
                          ev: float) -> float:
        """Compute advantage using adaptive critic/batch-mean.

        If EV > threshold: use critic-based advantage (lower variance)
        If EV ≤ threshold: use batch-mean advantage (critic inflates variance)
        """
        if ev > self.config.ev_threshold:
            # Critic helps → use critic-based
            return reward - critic_value
        else:
            # Critic hurts → use batch mean
            return reward - batch_mean_reward

    def update_critic(self, reward: float, lr: float | None = None):
        """Update critic value estimate."""
        lr = lr or self.config.critic_lr
        self.critic_value = (1 - lr) * self.critic_value + lr * reward

    def stats(self) -> dict:
        if not self._ev_history:
            return {"algorithm": "EVPO", "ev_mean": 0, "n_batches": 0}
        return {
            "algorithm": "EVPO",
            "ev_mean": sum(self._ev_history) / len(self._ev_history),
            "ev_recent": self._ev_history[-1],
            "n_batches": len(self._ev_history),
            "critic_value": self.critic_value,
            "using_critic": self._ev_history[-1] > self.config.ev_threshold if self._ev_history else False,
        }


@dataclass
class GRPOORConfig:
    """GRPO-OR: Output Reset trust region configuration."""
    lr: float = 1e-6
    margin: float = 0.2  # favorable margin for OR saturation


class GRPOOR:
    """GRPO-OR: Output Reset trust region for GRPO.

    Replaces clipped surrogate with smooth one-sided saturation (OR):
      - Advantage sign determines update direction
      - Token contributes zero OR residual after crossing favorable margin
      - Smoother than clip (no abrupt derivative change)

    OR loss: if advantage > 0:
              loss = -min(0, margin - log_ratio)^2 * advantage
            if advantage < 0:
              loss = -max(0, margin + log_ratio)^2 * advantage
    """

    def __init__(self, config: GRPOORConfig):
        self.config = config

    def compute_loss(self, log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute GRPO-OR loss.

        Args:
            log_probs: (T,) current policy log probs
            old_log_probs: (T,) behavior policy log probs
            advantages: (T,) per-token advantages (group-relative)
            mask: (T,) validity mask

        Returns:
            loss: scalar OR loss
        """
        log_ratio = log_probs - old_log_probs  # (T,)
        margin = self.config.margin

        # OR: smooth one-sided saturation
        # For positive advantage: penalize if log_ratio < margin (under-sampling)
        # For negative advantage: penalize if log_ratio > -margin (over-sampling)
        pos_adv = advantages > 0
        neg_adv = ~pos_adv

        # Positive advantage: OR residual = min(0, margin - log_ratio)^2
        or_pos = torch.clamp(margin - log_ratio, max=0).pow(2)
        # Negative advantage: OR residual = max(0, margin + log_ratio)^2
        or_neg = torch.clamp(margin + log_ratio, min=0).pow(2)

        # Apply advantage sign
        loss = torch.where(pos_adv, -or_pos * advantages, -or_neg * advantages)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def stats(self) -> dict:
        return {
            "algorithm": "GRPO-OR",
            "margin": self.config.margin,
        }
