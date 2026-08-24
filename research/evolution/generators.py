"""Tiny neural generators — produce candidate configs from noise + context.

Each generator is a small MLP (3 layers, ~50K params).
Input: noise vector (diversity) + context vector (current best config).
Output: candidate config (continuous params, discretized by domain).

Population of N generators explores different regions of config space.
REINFORCE updates push generators toward producing high-scoring configs.
Bottom 20% die each generation; top 20% spawn mutated children.

CUDA: All generators are batched into a single (N, ...) weight tensor.
Forward pass for all 500 generators runs as ONE batched matmul on GPU.
This turns 500 sequential CPU forward passes into 1 GPU batched op.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratorConfig:
    noise_dim: int = 16           # randomness input
    context_dim: int = 32         # current best config encoding
    hidden_dim: int = 64          # MLP hidden size
    output_dim: int = 8           # number of parameters to generate
    n_generators: int = 1000      # population size
    mutation_rate: float = 0.1    # weight noise when cloning
    init_scale: float = 0.3       # weight init scale


class BatchedGenerator(nn.Module):
    """All N generators batched into a single module for GPU efficiency.

    Instead of N separate nn.Linear layers, we store weights as:
      W0: (N, in_dim, hidden_dim)
      b0: (N, hidden_dim)
      W1: (N, hidden_dim, hidden_dim)
      b1: (N, hidden_dim)
      W2: (N, hidden_dim, out_dim)
      b2: (N, out_dim)

    Forward pass for all N generators with different noise:
      input: (N, noise_dim + context_dim) — each generator gets its own noise
      output: (N, out_dim) via batched matmul

    This is ~100x faster than looping over N separate modules on CPU.
    """

    def __init__(self, cfg: GeneratorConfig, device: torch.device = None):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        n = cfg.n_generators
        in_dim = cfg.noise_dim + cfg.context_dim
        h = cfg.hidden_dim
        out = cfg.output_dim

        # Batched weights: (N, in, out) for each layer
        self.W0 = nn.Parameter(torch.randn(n, in_dim, h, device=self.device) * 0.3)
        self.b0 = nn.Parameter(torch.zeros(n, h, device=self.device))
        self.W1 = nn.Parameter(torch.randn(n, h, h, device=self.device) * 0.3)
        self.b1 = nn.Parameter(torch.zeros(n, h, device=self.device))
        self.W2 = nn.Parameter(torch.randn(n, h, out, device=self.device) * 0.3)
        self.b2 = nn.Parameter(torch.zeros(n, out, device=self.device))

        # Per-generator fitness tracking (not a parameter)
        self.register_buffer("fitness_ema", torch.zeros(n, device=self.device))
        self.register_buffer("age", torch.zeros(n, device=self.device))

    def forward(self, noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Batched forward pass for all N generators.

        Args:
            noise: (N, noise_dim) — different noise per generator
            context: (context_dim,) — shared context (current best)

        Returns:
            (N, output_dim) outputs in [0, 1] (sigmoid)
        """
        # Expand context to (N, context_dim)
        n = self.cfg.n_generators
        ctx = context.unsqueeze(0).expand(n, -1)  # (N, context_dim)
        x = torch.cat([noise, ctx], dim=-1)        # (N, in_dim)

        # Batched matmul: (N, in) @ (N, in, h) → (N, h)
        h0 = torch.bmm(x.unsqueeze(1), self.W0).squeeze(1) + self.b0
        h0 = torch.tanh(h0)
        h1 = torch.bmm(h0.unsqueeze(1), self.W1).squeeze(1) + self.b1
        h1 = torch.tanh(h1)
        out = torch.bmm(h1.unsqueeze(1), self.W2).squeeze(1) + self.b2
        return torch.sigmoid(out)  # (N, output_dim) in [0, 1]

    def forward_single(self, noise: torch.Tensor, context: torch.Tensor,
                       gen_idx: int) -> torch.Tensor:
        """Forward pass for a single generator (for REINFORCE updates)."""
        ctx = context.expand(1, -1)  # (1, context_dim)
        x = torch.cat([noise.unsqueeze(0), ctx], dim=-1)  # (1, in_dim)

        w0 = self.W0[gen_idx:gen_idx+1]  # (1, in, h)
        b0 = self.b0[gen_idx:gen_idx+1]
        w1 = self.W1[gen_idx:gen_idx+1]
        b1 = self.b1[gen_idx:gen_idx+1]
        w2 = self.W2[gen_idx:gen_idx+1]
        b2 = self.b2[gen_idx:gen_idx+1]

        h0 = torch.bmm(x.unsqueeze(1), w0).squeeze(1) + b0
        h0 = torch.tanh(h0)
        h1 = torch.bmm(h0.unsqueeze(1), w1).squeeze(1) + b1
        h1 = torch.tanh(h1)
        out = torch.bmm(h1.unsqueeze(1), w2).squeeze(1) + b2
        return torch.sigmoid(out).squeeze(0)  # (output_dim,)

    def mutate_generator(self, idx: int, parent_idx: int, rate: float | None = None):
        """Copy parent's weights into slot idx, then add mutation noise."""
        rate = rate or self.cfg.mutation_rate
        with torch.no_grad():
            self.W0[idx] = self.W0[parent_idx] + torch.randn_like(self.W0[idx]) * rate
            self.b0[idx] = self.b0[parent_idx] + torch.randn_like(self.b0[idx]) * rate
            self.W1[idx] = self.W1[parent_idx] + torch.randn_like(self.W1[idx]) * rate
            self.b1[idx] = self.b1[parent_idx] + torch.randn_like(self.b1[idx]) * rate
            self.W2[idx] = self.W2[parent_idx] + torch.randn_like(self.W2[idx]) * rate
            self.b2[idx] = self.b2[parent_idx] + torch.randn_like(self.b2[idx]) * rate
            self.fitness_ema[idx] = self.fitness_ema[parent_idx] * 0.5
            self.age[idx] = 0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TemplateGenerator:
    """Non-neural generator: exhaustive/random sampling of discrete params.

    Used alongside neural generators to ensure full coverage of small
    discrete search spaces (e.g., block_size in {16, 32, 64, 128}).
    No learning — just systematic or random enumeration.
    """

    def __init__(self, choices: dict[str, list], seed: int = 42):
        self.choices = choices
        self.rng = np.random.RandomState(seed)
        self._keys = list(choices.keys())
        self._indices = {k: 0 for k in self._keys}
        self._exhausted = False
        self._score = 0.0
        self._fitness_ema = 0.0

    def sample(self, n: int) -> list[dict]:
        """Sample n configs. Mixes exhaustive (first) with random (rest)."""
        results = []
        for _ in range(n):
            if not self._exhausted:
                config = {}
                for k in self._keys:
                    idx = self._indices[k]
                    config[k] = self.choices[k][idx]
                results.append(config)
                for k in reversed(self._keys):
                    self._indices[k] += 1
                    if self._indices[k] < len(self.choices[k]):
                        break
                    self._indices[k] = 0
                else:
                    self._exhausted = True
            else:
                config = {k: self.rng.choice(v) for k, v in self.choices.items()}
                results.append(config)
        return results

    def reset(self):
        self._indices = {k: 0 for k in self._keys}
        self._exhausted = False


class GeneratorPopulation:
    """Manages batched generators + optional template generator.

    All N neural generators are stored in a single BatchedGenerator module
    for GPU-efficient batched forward passes.
    """

    def __init__(self, cfg: GeneratorConfig, template: TemplateGenerator | None = None,
                 device: torch.device = None):
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.batched_gen = BatchedGenerator(cfg, device=self.device).to(self.device)
        self.template = template
        self.context = torch.zeros(cfg.context_dim, device=self.device)
        self.generation = 0
        self.noise_buffer = None  # stored for REINFORCE

        total_params = self.batched_gen.n_params()
        mem_mb = total_params * 4 / 1e6
        print(f"[Generators] {cfg.n_generators} batched generators on {self.device}, "
              f"{total_params/1e6:.1f}M total params, {mem_mb:.1f} MB")

    def generate(self, n_per_gen: int = 1) -> list[dict[str, Any]]:
        """Generate candidates from all generators in one batched forward pass."""
        n = self.cfg.n_generators
        total = n * n_per_gen

        # Single batched noise tensor for all generators
        noise = torch.randn(total, self.cfg.noise_dim, device=self.device)
        # Repeat context for all
        ctx = self.context

        # One batched forward pass — all generators at once
        with torch.no_grad():
            outputs = self.batched_gen(noise, ctx)  # (total, output_dim)

        candidates = []
        for i in range(total):
            gen_idx = i // n_per_gen
            candidates.append({
                "params": outputs[i].detach(),
                "generator_idx": gen_idx,
                "noise": noise[i].detach(),
            })

        # Store noise for REINFORCE
        self.noise_buffer = noise.detach()

        # Add template samples if present
        if self.template and not self.template._exhausted:
            template_configs = self.template.sample(200)
            for tc in template_configs:
                candidates.append({"params": None, "template_config": tc,
                                   "generator_idx": -1})

        return candidates

    def evolve(self, scores: list[float], generator_indices: list[int]) -> list[int]:
        """Evolve population: kill worst, clone+mutate best, update fitness.
        Returns list of generator indices that were mutated (for optimizer reset).

        Every 10 generations, injects a diversity burst: random noise added to
        30% of generators to prevent convergence and maintain exploration.
        """
        self.generation += 1

        # Update fitness EMA per generator
        gen_scores: dict[int, list[float]] = {}
        for score, idx in zip(scores, generator_indices):
            if idx >= 0:
                gen_scores.setdefault(idx, []).append(score)

        with torch.no_grad():
            for idx, sc_list in gen_scores.items():
                avg = sum(sc_list) / len(sc_list)
                self.batched_gen.fitness_ema[idx] = \
                    0.7 * self.batched_gen.fitness_ema[idx] + 0.3 * avg
                self.batched_gen.age[idx] += 1

        # Sort by fitness
        fitness = self.batched_gen.fitness_ema.cpu().numpy()
        ranked = np.argsort(fitness)[::-1]  # descending
        n_kill = len(ranked) // 5
        n_clone = len(ranked) // 5

        bottom_indices = ranked[-n_kill:]
        top_indices = ranked[:n_clone]

        mutated = []
        for i, bad_idx in enumerate(bottom_indices):
            parent_idx = top_indices[i % n_clone]
            self.batched_gen.mutate_generator(int(bad_idx), int(parent_idx))
            mutated.append(int(bad_idx))

        # Diversity burst every 10 generations: inject noise into 30% of
        # mid-tier generators to prevent population convergence
        if self.generation % 10 == 0:
            n_burst = len(ranked) // 3
            mid_indices = ranked[n_clone:n_clone + n_burst]
            with torch.no_grad():
                for idx in mid_indices:
                    self.batched_gen.W0[idx] += torch.randn_like(
                        self.batched_gen.W0[idx]) * 0.15
                    self.batched_gen.W1[idx] += torch.randn_like(
                        self.batched_gen.W1[idx]) * 0.15
                    self.batched_gen.W2[idx] += torch.randn_like(
                        self.batched_gen.W2[idx]) * 0.15
                    mutated.append(int(idx))

        return mutated

    def update_context(self, best_params: torch.Tensor):
        """Update context vector with current best config."""
        ctx = torch.zeros(self.cfg.context_dim, device=self.device)
        n = min(best_params.shape[0], self.cfg.context_dim)
        ctx[:n] = best_params[:n].to(self.device)
        self.context = ctx
