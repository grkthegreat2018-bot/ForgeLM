"""OP-MIX: On-Policy data mixing via low-rank adapters.

Based on "Always Learning, Always Mixing: Efficient and Simple Data Mixing
All The Time" (arXiv 2605.15220).

Key insight: data mixing is an online decision problem that recurs
throughout training. OP-MIX simulates candidate data mixtures by
interpolating between low-rank adapters trained on different data sources,
eliminating separate proxy models.

How it works:
  1. Train small LoRA adapters on each data source (cheap, parallel)
  2. Simulate a candidate mixture by interpolating adapter weights
  3. Evaluate the interpolated model on a validation set
  4. Select the best mixture and apply it to the main training run

Results: consistently finds near-optimal mixtures using a fraction of
the compute of proxy-model approaches. Works across pretraining,
continual midtraining, and continual instruction tuning.

For our self-play training:
  - Multiple data sources: self-play generated, curated, tool-use, code
  - Current: fixed mixing ratio throughout training
  - OP-MIX: dynamically adjust mixing ratio based on model's learning dynamics
  - Better data efficiency, less manual tuning

This implementation provides:
  1. OPMixAdapter: low-rank adapter for a data source
  2. OPMixSimulator: simulates mixtures via adapter interpolation
  3. OPMixOptimizer: finds the best mixture online
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DataSource:
    """A training data source with its adapter."""
    name: str
    adapter: Optional[nn.Module] = None
    weight: float = 1.0  # current mixing weight
    n_samples: int = 0
    val_loss: float = float('inf')


class OPMixAdapter(nn.Module):
    """Low-rank LoRA adapter for a data source.

    Small adapter trained on a single data source. Used to simulate
    mixtures by interpolating adapter weights.
    """

    def __init__(self, d_model: int, rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        # LoRA: down-project, up-project
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

        # Initialize: down = random, up = zero (lossless start)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class OPMixSimulator:
    """Simulates data mixtures by interpolating adapter weights.

    Given N adapters trained on N data sources, simulates a mixture with
    weights (w1, ..., wN) by creating a combined adapter:
      combined_up = sum(wi * adapter_i.up)
      combined_down = mean(adapter_i.down)  # or weighted mean

    This is much cheaper than training a new model on the mixture.
    """

    def __init__(self, adapters: dict[str, OPMixAdapter]):
        self.adapters = adapters

    def simulate(self, weights: dict[str, float]) -> OPMixAdapter:
        """Create a combined adapter from weighted interpolation.

        Args:
            weights: {source_name: weight} — mixing weights

        Returns:
            combined: interpolated adapter
        """
        # Normalize weights
        total = sum(weights.values())
        if total == 0:
            weights = {k: 1.0 / len(weights) for k in weights}
        else:
            weights = {k: v / total for k, v in weights.items()}

        # Get first adapter for shape info
        first_adapter = next(iter(self.adapters.values()))
        combined = OPMixAdapter(first_adapter.d_model, first_adapter.rank)
        combined = combined.to(next(first_adapter.parameters()).device)

        # Interpolate weights
        combined_up = torch.zeros_like(combined.up.weight)
        combined_down = torch.zeros_like(combined.down.weight)

        for name, adapter in self.adapters.items():
            w = weights.get(name, 0.0)
            combined_up += w * adapter.up.weight
            combined_down += w * adapter.down.weight

        combined.up.weight.data = combined_up
        combined.down.weight.data = combined_down

        return combined

    def evaluate_mixture(self, weights: dict[str, float],
                         model: nn.Module, val_loader,
                         criterion) -> float:
        """Evaluate a simulated mixture on validation data.

        Applies the interpolated adapter to the model and computes
        validation loss. Returns the loss (lower is better).
        """
        combined = self.simulate(weights)

        # Temporarily apply adapter to model
        original_forwards = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] == combined.d_model:
                original_forwards[name] = module.forward
                combined_copy = copy.deepcopy(combined).to(module.weight.device)

                def make_forward(orig, adapter):
                    def fwd(x):
                        return orig(adapter(x))
                    return fwd

                module.forward = make_forward(original_forwards[name], combined_copy)

        # Evaluate
        total_loss = 0.0
        n_batches = 0
        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(next(model.parameters()).device)
                target_ids = batch['target_ids'].to(next(model.parameters()).device)
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_loss += loss.item()
                n_batches += 1
                if n_batches >= 10:  # quick evaluation
                    break

        # Restore original forwards
        for name, module in model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]

        return total_loss / max(n_batches, 1)


class OPMixOptimizer:
    """Finds the best data mixture online.

    Uses grid search or Bayesian optimization over mixing weights,
    evaluating each candidate via the simulator (cheap — no retraining).
    """

    def __init__(self, sources: list[str], simulator: OPMixSimulator):
        self.sources = sources
        self.simulator = simulator
        self._best_weights: dict[str, float] = {}
        self._best_loss: float = float('inf')
        self._history: list[dict] = []

    def search(self, model: nn.Module, val_loader, criterion,
               n_candidates: int = 20) -> dict[str, float]:
        """Search for the best mixing weights.

        Args:
            model: the base model (adapters applied temporarily)
            val_loader: validation data loader
            criterion: loss function
            n_candidates: number of candidate mixtures to evaluate

        Returns:
            best_weights: {source: weight}
        """
        # Generate candidate mixtures
        candidates = self._generate_candidates(n_candidates)

        best_weights = None
        best_loss = float('inf')

        for weights in candidates:
            loss = self.simulator.evaluate_mixture(weights, model, val_loader, criterion)

            self._history.append({
                'weights': weights.copy(),
                'loss': loss,
            })

            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()

        self._best_weights = best_weights or {s: 1.0 / len(self.sources) for s in self.sources}
        self._best_loss = best_loss

        print(f"  [OP-MIX] Best mixture: {self._best_weights} (loss={best_loss:.4f})")
        return self._best_weights

    def _generate_candidates(self, n: int) -> list[dict[str, float]]:
        """Generate candidate mixing weight combinations."""
        candidates = []

        # Uniform mixture
        candidates.append({s: 1.0 / len(self.sources) for s in self.sources})

        # Single-source mixtures
        for s in self.sources:
            w = {src: 0.0 for src in self.sources}
            w[s] = 1.0
            candidates.append(w)

        # Random mixtures
        for _ in range(n - len(candidates)):
            weights = torch.softmax(torch.randn(len(self.sources)), dim=0)
            w = {s: weights[i].item() for i, s in enumerate(self.sources)}
            candidates.append(w)

        return candidates

    def get_best(self) -> tuple[dict[str, float], float]:
        return self._best_weights, self._best_loss

    def stats(self) -> dict:
        return {
            "sources": self.sources,
            "n_evaluated": len(self._history),
            "best_loss": self._best_loss,
            "best_weights": self._best_weights,
        }
