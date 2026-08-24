"""Synthetic test domain: known function with multiple optima.

Used to validate that ForgeEvolve finds optima faster than random/grid search.
The function has a global optimum + several local optima, with a deceptive
gradient (greedy hill-climbing gets stuck).

Function: f(x) = sum of shifted Rastrigin + cosine terms
  - 8 continuous params in [0, 1]
  - Global optimum at x = [0.7, 0.3, 0.5, 0.8, 0.2, 0.6, 0.4, 0.9]
  - Multiple local optima (Rastrigin landscape)
  - Score = -f(x) (we maximize, so lower f = better)

Behavioral dims: mean(x), std(x) → MAP-Elites explores diverse param patterns.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Any
from . import BaseDomain


class SyntheticDomain(BaseDomain):
    """Synthetic optimization landscape for testing ForgeEvolve."""

    def __init__(self, dim: int = 8, seed: int = 42):
        self.dim = dim
        rng = np.random.RandomState(seed)
        # Random global optimum
        self.optimum = rng.rand(dim)
        # Known optimal score
        self.optimal_score = self._evaluate_raw(self.optimum)["score"]

    def name(self) -> str:
        return "synthetic"

    def output_dim(self) -> int:
        return self.dim

    def _evaluate_raw(self, x: np.ndarray) -> dict:
        """Rastrigin-like function with shifted optimum."""
        # Shift so optimum is at self.optimum
        shifted = x - self.optimum
        # Rastrigin: f = 10*D + sum(x^2 - 10*cos(2*pi*x))
        # Scaled to [0,1] domain
        rastrigin = 10 * self.dim + np.sum(
            (shifted * 5) ** 2 - 10 * np.cos(2 * np.pi * shifted * 5)
        )
        # Add a deceptive cosine term
        deceptive = 2 * np.sum(np.cos(3 * np.pi * x))
        # Score: higher = better. Optimal = ~0 rastrigin + max deceptive
        score = -(rastrigin + deceptive) + 50  # shift to positive
        return {
            "score": float(score),
            "behavioral": (float(np.mean(x)), float(np.std(x))),
            "metadata": {"rastrigin": float(rastrigin), "deceptive": float(deceptive)},
        }

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        x = params.detach().cpu().numpy()
        x = np.clip(x, 0, 1)
        return {"x": x}

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        if "x" in config:
            return torch.tensor(config["x"], dtype=torch.float32)
        elif "grid_2" in config:
            return torch.full((self.dim,), float(config["grid_2"]))
        else:
            return torch.zeros(self.dim)

    def evaluate(self, config: dict[str, Any]) -> dict:
        if "x" in config:
            x = config["x"]
        elif "grid_2" in config:
            # Template generator: single value, expand to dim
            val = config["grid_2"]
            x = np.full(self.dim, val, dtype=np.float32)
        else:
            # Fallback: extract any numeric values
            x = np.array([v for v in config.values() if isinstance(v, (int, float))])
            if len(x) < self.dim:
                x = np.pad(x, (0, self.dim - len(x)))
            x = x[:self.dim]
        return self._evaluate_raw(x)

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [
            ("param_mean", 10, 0.0, 1.0),
            ("param_std", 10, 0.0, 0.5),
        ]

    def discrete_choices(self) -> dict[str, list] | None:
        # Add coarse grid as template
        return {
            "grid_2": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
