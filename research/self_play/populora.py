"""PopuLoRA — Population-based asymmetric self-play with LoRA adapters.

PopuLoRA is a population-based self-play framework where teachers and
students are specialized LoRA adapters on a shared frozen base. Cross-
evaluation between sub-populations replaces self-calibration, preventing
the single-agent degeneration where the model generates easy problems
it can reliably solve.

Key mechanics:
  1. Population: instead of 1 expert per topic, maintain N LoRA adapters
     per topic on a shared frozen base model.
  2. Teachers (high-quality adapters) propose problems.
  3. Students (random subset) solve them.
  4. Cross-evaluation: teacher's problems are evaluated against ALL
     students, not just the proposing teacher's paired student.
  5. Evolution operators:
     - Mutation: add noise to LoRA weights (exploration).
     - Crossover: merge two adapters via SLERP (spherical linear
       interpolation) for quality recombination.
     - Selection: top-k adapters by cross-eval fitness survive.

This creates a co-evolutionary arms race that prevents overfitting and
encourages diverse problem generation.

Usage:
    from research.self_play.populora import PopuLoRA, PopulationConfig

    pop = PopuLoRA(base_model, tokenizer, device="cuda",
                  config=PopulationConfig(population_size=4))
    results = pop.evolve_round(topic="python_algorithms",
                              generate_fn=generate, verify_fn=verify)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class PopulationConfig:
    """PopuLoRA population configuration."""
    population_size: int = 4       # N adapters per topic
    lora_rank: int = 8             # LoRA rank for each adapter
    lora_alpha: int = 16           # LoRA alpha
    mutation_std: float = 0.02     # noise std for mutation
    crossover_ratio: float = 0.5   # fraction of population from crossover
    elite_fraction: float = 0.25   # top fraction that survives selection
    min_fitness: float = 0.1       # minimum fitness to survive
    # Adapter storage: "vram" (keep all in VRAM) or "disk" (offload via AirMoE).
    storage: str = "disk"


@dataclass
class AdapterFitness:
    """Fitness record for one adapter."""
    adapter_id: str
    topic: str
    problems_proposed: int = 0
    problems_solved_by_others: int = 0  # how many other adapters solved my problems
    problems_solved: int = 0  # how many of others' problems I solved
    difficulty_score: float = 0.0  # 1 - (solve_rate by others)
    competence_score: float = 0.0  # solve_rate on others' problems
    fitness: float = 0.0

    def compute_fitness(self):
        """Compute composite fitness: difficulty * competence."""
        if self.problems_proposed > 0:
            self.difficulty_score = 1.0 - (self.problems_solved_by_others /
                                           max(self.problems_proposed, 1))
        if self.problems_solved > 0:
            self.competence_score = self.problems_solved / max(
                self.problems_solved + 1, 1)
        # Clamp to [0, 1] to avoid negative/complex fitness.
        self.difficulty_score = max(0.0, min(1.0, self.difficulty_score))
        self.competence_score = max(0.0, min(1.0, self.competence_score))
        # Fitness = geometric mean of difficulty and competence.
        product = self.difficulty_score * self.competence_score
        self.fitness = product ** 0.5 if product >= 0 else 0.0
        return self.fitness


class LoRAAdapter:
    """A single LoRA adapter in the population.

    Stores LoRA A/B matrices for each target layer. Can be applied to
    and removed from the base model.
    """

    def __init__(self, adapter_id: str, target_layers: list[str],
                 d_model: int, rank: int = 8, alpha: int = 16):
        self.adapter_id = adapter_id
        self.target_layers = target_layers
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: A (d_model -> rank), B (rank -> d_model)
        # Init: A = random, B = zero (standard LoRA init = identity at start).
        self.weights = {}
        for name in target_layers:
            self.weights[name] = {
                "A": torch.randn(rank, d_model) * 0.01,
                "B": torch.zeros(d_model, rank),
            }

    def apply_to(self, model: nn.Module):
        """Apply this adapter's LoRA deltas to the model's linear layers."""
        for name, module in model.named_modules():
            # Match by module name or module.weight name.
            key = name if name in self.weights else f"{name}.weight"
            if key in self.weights and isinstance(module, nn.Linear):
                w = self.weights[key]
                # delta = alpha/rank * B @ A
                delta = self.scaling * (w["B"] @ w["A"])
                # Add to weight (in-place).
                module.weight.data += delta.to(module.weight.device,
                                               module.weight.dtype)

    def remove_from(self, model: nn.Module):
        """Remove this adapter's LoRA deltas from the model."""
        for name, module in model.named_modules():
            key = name if name in self.weights else f"{name}.weight"
            if key in self.weights and isinstance(module, nn.Linear):
                w = self.weights[key]
                delta = self.scaling * (w["B"] @ w["A"])
                module.weight.data -= delta.to(module.weight.device,
                                               module.weight.dtype)

    def mutate(self, std: float = 0.02):
        """Create a mutated copy of this adapter."""
        child = LoRAAdapter(
            f"{self.adapter_id}_mut_{random.randint(0, 9999)}",
            self.target_layers, self.d_model, self.rank, self.alpha)
        for name in self.weights:
            child.weights[name]["A"] = self.weights[name]["A"] + \
                torch.randn_like(self.weights[name]["A"]) * std
            child.weights[name]["B"] = self.weights[name]["B"] + \
                torch.randn_like(self.weights[name]["B"]) * std
        return child

    @staticmethod
    def slerp(a: "LoRAAdapter", b: "LoRAAdapter", t: float = 0.5) -> "LoRAAdapter":
        """Spherical linear interpolation between two adapters (crossover).

        SLERP interpolates along the great circle on the unit hypersphere,
        preserving the norm structure of both parents.

        Args:
            a: first parent.
            b: second parent.
            t: interpolation factor (0=a, 1=b, 0.5=midpoint).

        Returns:
            New adapter with SLERP'd weights.
        """
        child = LoRAAdapter(
            f"{a.adapter_id}_x_{b.adapter_id}_{random.randint(0, 9999)}",
            a.target_layers, a.d_model, a.rank, a.alpha)

        for name in a.weights:
            for key in ("A", "B"):
                wa = a.weights[name][key].flatten()
                wb = b.weights[name][key].flatten()
                # Normalize for SLERP.
                na = wa.norm() + 1e-8
                nb = wb.norm() + 1e-8
                ua = wa / na
                ub = wb / nb
                # Dot product (cosine similarity).
                dot = (ua * ub).sum().clamp(-1.0, 1.0)
                # SLERP formula.
                if dot > 0.9995:
                    # Nearly parallel — linear interpolation.
                    result = (1 - t) * wa + t * wb
                else:
                    theta = torch.arccos(dot)
                    sin_theta = torch.sin(theta)
                    result = (torch.sin((1 - t) * theta) / sin_theta) * wa + \
                             (torch.sin(t * theta) / sin_theta) * wb
                child.weights[name][key] = result.reshape(
                    a.weights[name][key].shape)

        return child


class PopuLoRA:
    """Population-based asymmetric self-play with LoRA adapters.

    Maintains a population of LoRA adapters per topic, evolves them via
    mutation/crossover/selection, and uses cross-evaluation to compute
    fitness (preventing single-agent degeneration).
    """

    def __init__(self, base_model: nn.Module, tokenizer,
                 device: str = "cuda",
                 config: PopulationConfig | None = None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or PopulationConfig()
        self.populations: dict[str, list[LoRAAdapter]] = {}
        self.fitness_history: dict[str, list[AdapterFitness]] = {}

    def init_population(self, topic: str, target_layers: list[str],
                        d_model: int):
        """Initialize a population of adapters for a topic.

        Args:
            topic: the topic/domain name.
            target_layers: list of module names to apply LoRA to.
            d_model: model dimension.
        """
        adapters = []
        for i in range(self.config.population_size):
            adapter = LoRAAdapter(
                f"{topic}_adapter_{i}", target_layers, d_model,
                rank=self.config.lora_rank, alpha=self.config.lora_alpha)
            adapters.append(adapter)
        self.populations[topic] = adapters
        self.fitness_history[topic] = []

    def evolve_round(self, topic: str,
                     generate_fn: Callable,
                     verify_fn: Callable,
                     n_problems: int = 4) -> dict:
        """Run one evolution round for a topic.

        1. Each adapter proposes problems (as teacher).
        2. All other adapters attempt to solve them (as students).
        3. Cross-evaluation computes fitness.
        4. Selection + mutation + crossover produces next generation.

        Args:
            topic: the topic to evolve.
            generate_fn: function(adapter, topic, n) -> list of problems.
            verify_fn: function(problem, solution) -> bool (verified?).
            n_problems: problems per adapter per round.

        Returns:
            Stats dict with fitness scores and evolution metrics.
        """
        if topic not in self.populations:
            raise ValueError(f"Topic '{topic}' not initialized")

        adapters = self.populations[topic]
        n = len(adapters)

        # Phase 1: Each adapter proposes problems.
        all_problems = {}  # adapter_id -> list of problems
        for adapter in adapters:
            problems = generate_fn(adapter, topic, n_problems)
            all_problems[adapter.adapter_id] = problems

        # Phase 2: Cross-evaluation — each adapter tries all others' problems.
        fitness = {a.adapter_id: AdapterFitness(a.adapter_id, topic)
                    for a in adapters}

        for teacher in adapters:
            teacher_problems = all_problems[teacher.adapter_id]
            fitness[teacher.adapter_id].problems_proposed = len(teacher_problems)

            for student in adapters:
                if student.adapter_id == teacher.adapter_id:
                    continue  # skip self-evaluation

                # Apply student adapter to base model.
                student.apply_to(self.base_model)
                try:
                    for problem in teacher_problems:
                        solution = generate_fn(student, topic, 1,
                                              prompt=problem)
                        if verify_fn(problem, solution):
                            fitness[student.adapter_id].problems_solved += 1
                            fitness[teacher.adapter_id].problems_solved_by_others += 1
                finally:
                    # Always remove adapter (even on error).
                    student.remove_from(self.base_model)

        # Phase 3: Compute fitness.
        for f in fitness.values():
            f.compute_fitness()
        self.fitness_history[topic].extend(list(fitness.values()))

        # Phase 4: Selection + evolution.
        sorted_adapters = sorted(adapters,
            key=lambda a: fitness[a.adapter_id].fitness, reverse=True)
        n_elite = max(1, int(n * self.config.elite_fraction))
        elites = sorted_adapters[:n_elite]

        # Crossover: create children from elite pairs.
        n_crossover = int(n * self.config.crossover_ratio)
        children = []
        for _ in range(n_crossover):
            if len(elites) >= 2:
                a, b = random.sample(elites, 2)
                child = LoRAAdapter.slerp(a, b)
                children.append(child)

        # Mutation: mutate elites to create explorers.
        n_mutate = n - n_elite - len(children)
        mutated = []
        for _ in range(n_mutate):
            parent = random.choice(elites)
            mutated.append(parent.mutate(self.config.mutation_std))

        # Next generation.
        self.populations[topic] = elites + children + mutated

        return {
            "topic": topic,
            "generation_size": len(self.populations[topic]),
            "n_elites": n_elite,
            "n_crossover": len(children),
            "n_mutated": len(mutated),
            "mean_fitness": sum(f.fitness for f in fitness.values()) / n,
            "max_fitness": max(f.fitness for f in fitness.values()),
            "fitness": {aid: f.fitness for aid, f in fitness.items()},
        }

    def get_best_adapter(self, topic: str) -> LoRAAdapter | None:
        """Get the highest-fitness adapter for a topic."""
        if topic not in self.populations or not self.populations[topic]:
            return None
        history = self.fitness_history.get(topic, [])
        if not history:
            return self.populations[topic][0]
        best_id = max(history, key=lambda f: f.fitness).adapter_id
        for adapter in self.populations[topic]:
            if adapter.adapter_id == best_id:
                return adapter
        return self.populations[topic][0]

    def get_population_stats(self, topic: str) -> dict:
        """Get statistics about a topic's population."""
        if topic not in self.populations:
            return {"topic": topic, "initialized": False}
        history = self.fitness_history.get(topic, [])
        if not history:
            return {"topic": topic, "initialized": True,
                    "population_size": len(self.populations[topic]),
                    "generations": 0}
        recent = history[-len(self.populations[topic]):]
        return {
            "topic": topic,
            "initialized": True,
            "population_size": len(self.populations[topic]),
            "generations": len(history) // len(self.populations[topic]),
            "mean_fitness": sum(f.fitness for f in recent) / len(recent),
            "max_fitness": max(f.fitness for f in recent),
            "mean_difficulty": sum(f.difficulty_score for f in recent) / len(recent),
            "mean_competence": sum(f.competence_score for f in recent) / len(recent),
        }
