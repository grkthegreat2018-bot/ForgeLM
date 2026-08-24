"""Trainer: REINFORCE updates for batched generators.

REINFORCE: generators that produced high-scoring candidates get their
weights nudged to produce more similar outputs. Low-scoring generators
get nudged away.

CUDA: updates run on GPU. Single Adam optimizer over the batched weight
tensor. Gradient clipping prevents REINFORCE explosions. Optimizer state
for mutated generators is reset to avoid stale momentum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GeneratorTrainer:
    """REINFORCE trainer for the batched generator population."""

    def __init__(self, batched_gen, lr: float = 1e-3,
                 baseline_decay: float = 0.9, device: torch.device = None,
                 max_grad_norm: float = 1.0,
                 pushaway_strength: float = 0.3):
        self.batched_gen = batched_gen
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.max_grad_norm = max_grad_norm
        self.pushaway_strength = pushaway_strength
        # Single optimizer over all batched weights
        self.optimizer = torch.optim.Adam(batched_gen.parameters(), lr=lr)
        # Track which generator slots were mutated → reset their optimizer state
        self._mutated_slots: set[int] = set()

    def notify_mutation(self, gen_idx: int):
        """Tell the trainer that generator gen_idx was mutated.
        Its Adam momentum/velocity buffers will be reset before next step.
        """
        self._mutated_slots.add(gen_idx)

    def _reset_optimizer_state(self, gen_idx: int):
        """Reset Adam momentum/velocity for a single generator slot.
        Without this, mutated generators get updates based on stale momentum
        from the previous weights that no longer exist.
        """
        # Adam state is stored in optimizer.state[param][key]
        # For batched weights (N, in, out), we zero the slice at gen_idx
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if gen_idx < p.shape[0]:
                    state = self.optimizer.state.get(p, {})
                    if "step" in state:
                        # Zero the exp_avg and exp_avg_sq for this generator
                        state["exp_avg"][gen_idx].zero_()
                        state["exp_avg_sq"][gen_idx].zero_()

    def update(self, candidates: list[dict], scores: list[float],
               context: torch.Tensor):
        """REINFORCE update for generators that produced each candidate.

        Args:
            candidates: list of {params, generator_idx, noise}
            scores: evaluation scores (higher = better)
            context: current context vector (on device)
        """
        if not scores:
            return

        # Reset optimizer state for mutated generators
        for idx in self._mutated_slots:
            self._reset_optimizer_state(idx)
        self._mutated_slots.clear()

        # Update baseline
        avg_score = float(np.mean(scores))
        self.baseline = (self.baseline_decay * self.baseline +
                         (1 - self.baseline_decay) * avg_score)

        # Group by generator
        gen_updates: dict[int, list[tuple]] = {}
        for cand, score in zip(candidates, scores):
            idx = cand.get("generator_idx", -1)
            if idx < 0:
                continue
            gen_updates.setdefault(idx, []).append((cand, score))

        # Batched REINFORCE: collect all losses, do one backward pass
        self.batched_gen.train()
        losses = []

        for gen_idx, pairs in gen_updates.items():
            for cand, score in pairs:
                if "noise" not in cand or "params" not in cand:
                    continue

                noise = cand["noise"].to(self.device)
                target_params = cand["params"].to(self.device)

                # Recompute output with gradient for this generator
                output = self.batched_gen.forward_single(noise, context, gen_idx)

                # REINFORCE: reward = score - baseline
                reward = score - self.baseline

                # Pull toward target if reward > 0, push away if < 0
                # Asymmetric: push-away is weaker to avoid destabilizing
                if reward > 0:
                    loss = nn.functional.mse_loss(output, target_params)
                else:
                    loss = -nn.functional.mse_loss(output, target_params) * self.pushaway_strength

                losses.append(loss)

        if not losses:
            return

        total_loss = torch.stack(losses).sum()

        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping — REINFORCE can produce large gradients
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.batched_gen.parameters(), self.max_grad_norm
            )

        self.optimizer.step()
