"""Novelty search module — behavioral diversity prevents plateaus.

Based on Lehman & Stanley's novelty search (GECCO 2008-2011):
  "Abandoning the objective: Novelty search and the problem of measuring
   progress" — search for behavioral novelty instead of (or alongside)
   fitness. This avoids deceptive local optima.

Combined with "Composite Novelty Pulsation" (GECCO 2020):
  Alternate between novelty-driven exploration and quality-driven exploitation.
  This prevents both plateauing (novelty phase) and overfitting (quality phase).

Implementation:
  - Novelty metric: average distance to k-nearest archived configs in
    behavioral space. High novelty = far from everything seen before.
  - Pulsation: cycles between novelty (explore) and quality (exploit) phases.
    Phase length adapts based on improvement rate.
  - When quality plateaus, extend novelty phase to find new regions.
  - When novelty finds a promising region, switch to quality to exploit it.

Integration with MAP-Elites:
  - During novelty phase: accept configs into archive based on novelty score
    (not just quality). This fills cells that are behaviorally distant.
  - During quality phase: standard MAP-Elites (accept if better score in cell).
  - The archive tracks both quality and novelty for each entry.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class NoveltyConfig:
    """Configuration for novelty search integration."""
    enabled: bool = True
    # Novelty metric: k-nearest-neighbor distance in behavioral space
    k_neighbors: int = 5
    # Pulsation: cycle between novelty and quality phases
    pulse_novelty_len: int = 3   # generations of novelty search
    pulse_quality_len: int = 5   # generations of quality search
    # Adaptive: extend novelty phase when quality plateaus
    adaptive_pulsation: bool = True
    max_novelty_len: int = 10    # cap on extended novelty phase
    # Novelty bonus added to surrogate prediction during novelty phase
    novelty_bonus: float = 1.5
    # Minimum novelty threshold to accept into archive (prevents noise)
    min_novelty: float = 0.05


class NoveltySearch:
    """Novelty search + pulsation for the ForgeEvolve engine.

    Tracks behavioral diversity and alternates between novelty-driven
    exploration and quality-driven exploitation to prevent plateaus.
    """

    def __init__(self, cfg: NoveltyConfig):
        self.cfg = cfg
        self.phase = "quality"  # "novelty" or "quality"
        self.phase_counter = 0
        self.generation = 0
        # Track behavioral vectors of all archived configs for novelty computation
        self.behavioral_archive: list[np.ndarray] = []
        # Track improvement rate for adaptive pulsation
        self._improvement_history: list[float] = []
        self._current_novelty_len: int = cfg.pulse_novelty_len

    def update_phase(self, best_score: float) -> str:
        """Update the current phase (novelty/quality) based on pulsation cycle.

        Called at the start of each generation. Returns the current phase.
        """
        self.generation += 1
        self.phase_counter += 1

        # Track improvement
        self._improvement_history.append(best_score)
        if len(self._improvement_history) > 20:
            self._improvement_history.pop(0)

        # Check if we should switch phases
        if self.phase == "quality" and self.phase_counter >= self.cfg.pulse_quality_len:
            # Switch to novelty phase
            self.phase = "novelty"
            self.phase_counter = 0
            # Adaptive: if quality was plateauing, extend novelty phase
            if self.cfg.adaptive_pulsation and len(self._improvement_history) >= 5:
                recent = self._improvement_history[-5:]
                improvement = recent[-1] - recent[0]
                if improvement < 0.01:  # plateau threshold
                    self._current_novelty_len = min(
                        self.cfg.max_novelty_len,
                        self.cfg.pulse_novelty_len * 2)
                else:
                    self._current_novelty_len = self.cfg.pulse_novelty_len

        elif self.phase == "novelty" and self.phase_counter >= self._current_novelty_len:
            # Switch to quality phase
            self.phase = "quality"
            self.phase_counter = 0

        return self.phase

    def compute_novelty(self, behavioral: np.ndarray | list[float],
                        archive_behaviors: list[np.ndarray] | None = None) -> float:
        """Compute novelty score for a behavioral vector.

        Novelty = average distance to k-nearest neighbors in behavioral space.
        High novelty = far from everything seen before = worth exploring.
        """
        if not self.cfg.enabled:
            return 0.0

        pool = archive_behaviors if archive_behaviors is not None else self.behavioral_archive
        if len(pool) == 0:
            return 1.0  # everything is novel when archive is empty

        vec = np.array(behavioral, dtype=np.float32)
        distances = [np.linalg.norm(vec - np.array(b, dtype=np.float32)) for b in pool]
        distances.sort()
        k = min(self.cfg.k_neighbors, len(distances))
        novelty = sum(distances[:k]) / k
        return float(novelty)

    def add_behavioral(self, behavioral: np.ndarray | list[float]):
        """Add a behavioral vector to the novelty archive."""
        self.behavioral_archive.append(np.array(behavioral, dtype=np.float32))
        # Cap archive size to prevent O(n) novelty computation from being too slow
        if len(self.behavioral_archive) > 2000:
            self.behavioral_archive = self.behavioral_archive[-2000:]

    def get_acquisition_bonus(self) -> float:
        """Return the bonus to add to surrogate predictions.

        During novelty phase: return novelty_bonus (encourages exploration).
        During quality phase: return 0 (pure exploitation).
        """
        if not self.cfg.enabled:
            return 0.0
        return self.cfg.novelty_bonus if self.phase == "novelty" else 0.0

    def should_accept(self, score: float, novelty: float,
                      cell_score: float | None) -> bool:
        """Decide whether to accept a config into the archive.

        During quality phase: accept if score > cell_score (standard MAP-Elites).
        During novelty phase: accept if novelty > min_novelty OR score > cell_score.
        """
        if not self.cfg.enabled:
            return cell_score is None or score > cell_score

        if self.phase == "novelty":
            # Novelty phase: accept novel configs even if not highest score
            if novelty > self.cfg.min_novelty:
                return True
            # Also accept if it's a new best for the cell
            return cell_score is None or score > cell_score
        else:
            # Quality phase: standard MAP-Elites
            return cell_score is None or score > cell_score

    def status(self) -> str:
        """Return a status string for logging."""
        return f"phase={self.phase}({self.phase_counter}), archive={len(self.behavioral_archive)}"
