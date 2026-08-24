"""MAP-Elites archive: stores best solution per behavioral dimension.

Quality-diversity: instead of keeping only the single best, we keep
a grid of best solutions across behavioral dimensions. This ensures
diverse discoveries (not just local optima).

Example for quant domain:
  behavioral_dim_1 = memory_usage (bins: 0.5x, 1x, 2x, 4x compression)
  behavioral_dim_2 = error_level (bins: <5%, 5-10%, 10-20%, >20%)
  Each cell stores the best config found for that (compression, error) combo.
"""
from __future__ import annotations

import collections
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ArchiveEntry:
    config: dict[str, Any]     # the candidate config
    score: float               # evaluation score (higher = better)
    behavioral: tuple           # behavioral descriptor values
    generation: int             # when it was found
    metadata: dict = field(default_factory=dict)


class MapElitesArchive:
    """Grid-based MAP-Elites archive.

    Args:
        dims: list of (name, n_bins, min_val, max_val) per behavioral dimension
    """

    def __init__(self, dims: list[tuple[str, int, float, float]],
                 max_history: int = 5000):
        self.dims = dims
        self.dim_names = [d[0] for d in dims]
        self.n_bins = [d[1] for d in dims]
        self.ranges = [(d[2], d[3]) for d in dims]
        self.grid_shape = tuple(self.n_bins)
        self.grid: dict[tuple, ArchiveEntry | None] = {}
        self._init_grid()
        self.n_filled = 0
        self.best_score = -float("inf")
        self.best_entry: Optional[ArchiveEntry] = None
        self.max_history = max_history  # cap to prevent OOM on long runs
        self.history = collections.deque(maxlen=self.max_history)
        self._pareto_cache: list[ArchiveEntry] | None = None  # cached Pareto front

    def _init_grid(self):
        for idx in np.ndindex(*self.grid_shape):
            self.grid[idx] = None

    def _to_bin(self, behavioral: tuple) -> tuple:
        """Map behavioral values to grid indices."""
        idx = []
        for i, val in enumerate(behavioral):
            lo, hi = self.ranges[i]
            n = self.n_bins[i]
            # Clamp + bin
            clamped = max(lo, min(hi - 1e-9, val))
            bin_idx = int((clamped - lo) / (hi - lo) * n)
            bin_idx = max(0, min(n - 1, bin_idx))
            idx.append(bin_idx)
        return tuple(idx)

    def add(self, config: dict, score: float, behavioral: tuple,
            generation: int, metadata: dict | None = None) -> bool:
        """Try to add a candidate to the archive.

        Returns True if it was accepted (new cell or better than existing).
        """
        entry = ArchiveEntry(
            config=config, score=score, behavioral=behavioral,
            generation=generation, metadata=metadata or {},
        )
        self.history.append(entry)

        idx = self._to_bin(behavioral)
        existing = self.grid[idx]

        if existing is None or score > existing.score:
            was_empty = existing is None
            self.grid[idx] = entry
            if was_empty:
                self.n_filled += 1
            if score > self.best_score:
                self.best_score = score
                self.best_entry = entry
            self._pareto_cache = None  # invalidate cache
            return True
        return False

    def sample_elite(self) -> Optional[ArchiveEntry]:
        """Sample a random non-empty cell's entry (for crossover/mutation)."""
        filled = [v for v in self.grid.values() if v is not None]
        if not filled:
            return None
        return np.random.choice(filled)

    def sample_elite_ucb(self, current_gen: int = 0,
                         c: float = 1.0) -> Optional[ArchiveEntry]:
        """UCB-based parent selection (Monte Carlo Elites).

        From "Monte Carlo Elites" (Gaier 2021): treat parent selection as
        a multi-armed bandit. Each archive cell is an arm. UCB selects
        cells that are either high-quality or under-explored.

        UCB = normalized_score + c * sqrt(ln(total_pulls) / cell_pulls)

        Args:
            current_gen: current generation (for age weighting)
            c: exploration constant (higher = more exploration)
        """
        filled_cells = [(idx, v) for idx, v in self.grid.items() if v is not None]
        if not filled_cells:
            return None

        # Track pull counts per cell (lazy init)
        if not hasattr(self, "_cell_pulls"):
            self._cell_pulls = {idx: 0 for idx, _ in filled_cells}
        for idx, _ in filled_cells:
            if idx not in self._cell_pulls:
                self._cell_pulls[idx] = 0

        total_pulls = sum(self._cell_pulls.values()) + 1
        scores = np.array([v.score for _, v in filled_cells])
        score_min, score_max = scores.min(), scores.max()
        score_range = max(score_max - score_min, 1e-8)

        ucb_values = []
        for idx, entry in filled_cells:
            n_pulls = self._cell_pulls[idx] + 1
            norm_score = (entry.score - score_min) / score_range
            exploration = c * np.sqrt(np.log(total_pulls) / n_pulls)
            ucb_values.append(norm_score + exploration)

        ucb_values = np.array(ucb_values)
        # Sample proportional to UCB (softmax with temperature)
        probs = np.exp(ucb_values - ucb_values.max())
        probs = probs / probs.sum()
        chosen_idx = np.random.choice(len(filled_cells), p=probs)

        # Increment pull count
        cell_idx = filled_cells[chosen_idx][0]
        self._cell_pulls[cell_idx] += 1

        return filled_cells[chosen_idx][1]

    def get_pareto_front(self) -> list[ArchiveEntry]:
        """Compute Pareto front from archive entries.

        Uses behavioral dimensions as objectives (e.g., compression vs error).
        A solution is Pareto-optimal if no other solution dominates it
        (is better in all dimensions).

        For (compression, error): compression should be maximized,
        error should be minimized. We handle this by inverting error.
        """
        # Use cached Pareto front if valid (invalidated on add())
        if self._pareto_cache is not None:
            return self._pareto_cache

        filled = [v for v in self.grid.values() if v is not None]
        if not filled:
            self._pareto_cache = []
            return []

        # Determine which dims to maximize vs minimize
        # Convention: first dim = compression (maximize), second = error (minimize)
        pareto = []
        for i, entry_i in enumerate(filled):
            beh_i = entry_i.behavioral
            dominated = False
            for j, entry_j in enumerate(filled):
                if i == j:
                    continue
                beh_j = entry_j.behavioral
                # Check if j dominates i
                # j dominates i if: j is >= in compression AND <= in error
                # AND strictly better in at least one
                if (beh_j[0] >= beh_i[0] and beh_j[1] <= beh_i[1] and
                    (beh_j[0] > beh_i[0] or beh_j[1] < beh_i[1])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(entry_i)

        self._pareto_cache = pareto
        return pareto

    def get_all_elites(self) -> list[ArchiveEntry]:
        """Return all non-empty cell entries."""
        return [v for v in self.grid.values() if v is not None]

    def coverage(self) -> float:
        """Fraction of grid cells filled."""
        total = np.prod(self.grid_shape)
        return self.n_filled / total

    def summary(self) -> str:
        pareto = self.get_pareto_front()
        return (f"Archive: {self.n_filled}/{int(np.prod(self.grid_shape))} cells "
                f"({self.coverage()*100:.1f}%), best={self.best_score:.4f}, "
                f"pareto_front={len(pareto)}")
