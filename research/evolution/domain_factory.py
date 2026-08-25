"""Dynamic domain factory — auto-spawns refinement domains from converged parents.

When a domain converges (no improvement for N generations), the factory:
1. Takes the best config found
2. Creates a "refinement" domain that narrows the search space around it
   (each parameter range is ±20% of the original range, centered on the best)
3. Registers it dynamically so the engine can run it next

This removes the 57-domain limit: the system dynamically creates as many
refinement domains as needed, enabling infinite-depth search without
plateauing or overfitting.

Architecture:
  - RefinementDomain wraps a parent domain, overriding decode() to map
    [0,1] params into a narrow range around the parent's best config
  - The factory tracks which domains have converged and spawns children
  - Children inherit the parent's evaluate() but with narrowed param ranges
  - After a child converges, it can spawn its own children (recursive refinement)

Usage:
  from research.evolution.domain_factory import DomainFactory
  factory = DomainFactory()
  # After a domain converges:
  child = factory.spawn_refinement(parent_domain, best_config, depth=1)
  factory.register(child)  # adds to DOMAINS registry dynamically
"""
from __future__ import annotations

import torch
from typing import Any
from research.evolution.domains import BaseDomain, DOMAINS


class RefinementDomain(BaseDomain):
    """A narrowed search domain centered on a parent's best config.

    Wraps a parent domain but maps [0,1] generator outputs into a narrow
    range around the best-known config. This enables deep refinement of
    promising regions without re-searching the entire space.

    The narrowing factor controls how tight the search is:
      - depth=0: full range (same as parent)
      - depth=1: ±20% of full range around best
      - depth=2: ±4% of full range around best (0.2²)
      - depth=3: ±0.8% (essentially fine-tuning)
    """

    def __init__(self, parent: BaseDomain, best_config: dict[str, Any],
                 best_params: torch.Tensor, depth: int = 1,
                 narrowing: float = 0.2):
        # Don't call super().__init__() — it may fail if parent didn't.
        # Instead, copy device from parent defensively.
        self._device = getattr(parent, '_device', None)
        if self._device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parent = parent
        self.best_config = best_config
        self.best_params = best_params
        self.depth = depth
        self.narrowing = narrowing ** depth  # exponential narrowing

    def name(self) -> str:
        return f"{self.parent.name()}_refine_d{self.depth}"

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def output_dim(self) -> int:
        return self.parent.output_dim()

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        """Map [0,1] params → config, but in a narrowed range around best.

        param_range = [best - narrowing/2, best + narrowing/2]
        decoded = best - narrowing/2 + params * narrowing
        """
        p = params.detach().cpu().numpy()
        best = self.best_params.detach().cpu().numpy()
        # Narrowed params: map [0,1] → [best-narrow/2, best+narrow/2]
        narrowed = best - self.narrowing / 2 + p * self.narrowing
        narrowed = narrowed.clip(0, 1)  # stay in valid range
        return self.parent.decode(torch.tensor(narrowed, device=self.device))

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        """Encode config → [0,1] params (inverse of decode)."""
        full_params = self.parent.encode(config)
        # Map from full range back to narrowed [0,1] space
        best = self.best_params.to(full_params.device)
        narrowed = (full_params - best + self.narrowing / 2) / self.narrowing
        return narrowed.clamp(0, 1)

    def evaluate(self, config: dict[str, Any]) -> dict:
        return self.parent.evaluate(config)

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        # Inherit parent's behavioral dims but narrow the ranges
        dims = self.parent.behavioral_dims()
        # Keep same grid structure — the archive will fill with refined entries
        return dims

    def discrete_choices(self) -> dict[str, list] | None:
        return self.parent.discrete_choices()

    def seed_configs(self) -> list[dict[str, Any]]:
        """Seed with the best config + small perturbations."""
        seeds = [self.best_config]
        # Add perturbed versions for diversity
        for _ in range(4):
            p = self.best_params + torch.randn_like(self.best_params) * 0.05
            p = p.clamp(0, 1)
            seeds.append(self.parent.decode(p))
        return seeds

    def to_cpu(self) -> "RefinementDomain":
        return RefinementDomain(
            parent=self.parent.to_cpu(),
            best_config=self.best_config,
            best_params=self.best_params.cpu(),
            depth=self.depth,
            narrowing=self.narrowing,
        )


class DomainFactory:
    """Factory that dynamically creates and registers refinement domains.

    Tracks parent domains that have converged and spawns child refinement
    domains. Children can themselves spawn grandchildren, enabling
    recursive deepening of the search.

    The factory caps the total number of active domains to prevent
    unbounded growth (default: 200).
    """

    def __init__(self, max_domains: int = 200, max_depth: int = 5,
                 narrowing: float = 0.2):
        self.max_domains = max_domains
        self.max_depth = max_depth
        self.narrowing = narrowing
        self.spawned: dict[str, RefinementDomain] = {}
        self.lineage: dict[str, str] = {}  # child_name → parent_name

    def spawn_refinement(self, parent: BaseDomain,
                         best_config: dict[str, Any],
                         best_params: torch.Tensor,
                         depth: int = 1) -> RefinementDomain | None:
        """Create a refinement domain from a converged parent.

        Returns None if max_depth or max_domains is reached.
        """
        if depth > self.max_depth:
            return None
        if len(self.spawned) >= self.max_domains:
            return None

        child = RefinementDomain(
            parent=parent,
            best_config=best_config,
            best_params=best_params,
            depth=depth,
            narrowing=self.narrowing,
        )
        self.spawned[child.name()] = child
        self.lineage[child.name()] = parent.name()
        return child

    def register(self, domain: RefinementDomain):
        """Dynamically register a domain instance in the global DOMAINS dict.

        Since RefinementDomain instances are all the same class but with
        different configs, we store the instance itself wrapped in a
        lambda factory so DOMAINS[name]() returns the pre-configured instance.
        """
        # Store a factory that returns the pre-configured instance
        instance = domain
        DOMAINS[domain.name()] = lambda: instance

    def get_spawned_names(self) -> list[str]:
        """Return names of all spawned refinement domains."""
        return list(self.spawned.keys())

    def get_lineage(self, name: str) -> list[str]:
        """Trace the lineage of a domain back to its root parent."""
        chain = [name]
        current = name
        while current in self.lineage:
            current = self.lineage[current]
            chain.append(current)
        return chain

    def should_spawn(self, domain_name: str, convergence_count: int,
                     patience: int = 5) -> bool:
        """Check if a domain has converged enough to warrant refinement."""
        return convergence_count >= patience and domain_name not in self.spawned
