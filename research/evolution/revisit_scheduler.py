"""Domain revisit scheduler — re-queues stale domains after code changes.

Based on "Dynamic Quality-Diversity Search" (arXiv 2404.05769):
  When the environment changes (code is updated, new training data, new
  hardware), previously-converged domains may have new optima. The scheduler
  re-evaluates stale domains to check if recent changes unlocked improvements.

How it works:
  1. Tracks the last-run timestamp + best score for each domain
  2. Monitors source files for changes (git diff or mtime)
  3. When a source file related to a domain changes, marks that domain as
     "stale" — its past convergence may no longer hold
  4. Stale domains are re-queued with higher priority
  5. After re-running, if the domain found new improvements, it's marked
     "fresh". If not, it goes back to "converged" with a longer cooldown.

This breaks the plateau cycle: a domain that converged 50 loops ago might
find new improvements after we've trained the model, changed config defaults,
or added new architecture keys.

Priority formula:
  priority = base_priority * staleness_factor * code_change_factor
  staleness_factor = 1 + (loops_since_last_run / 10)
  code_change_factor = 2.0 if related code changed, 1.0 otherwise
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DomainRunRecord:
    """Tracks when a domain was last run and its results."""
    domain_name: str
    last_run_loop: int = 0
    last_best_score: float = float('-inf')
    last_run_time: float = 0.0  # unix timestamp
    consecutive_converged: int = 0  # how many times in a row it converged
    related_files: list[str] = field(default_factory=list)
    file_mtimes: dict[str, float] = field(default_factory=dict)


class DomainRevisitScheduler:
    """Schedules domain re-runs based on staleness + code changes.

    Usage:
        scheduler = DomainRevisitScheduler()
        scheduler.register("QuantDomain", related_files=["research/inference/quant/"])
        # After each loop:
        stale = scheduler.get_stale_domains(current_loop=10)
        # stale = domains that should be re-run
    """

    def __init__(self, project_root: str = ".", cooldown_loops: int = 5,
                 max_consecutive_converged: int = 3):
        self.root = Path(project_root)
        self.cooldown_loops = cooldown_loops  # min loops between re-runs
        self.max_consecutive_converged = max_consecutive_converged
        self.records: dict[str, DomainRunRecord] = {}
        self._domain_file_map: dict[str, list[str]] = {}

    def register(self, domain_name: str, related_files: list[str] | None = None):
        """Register a domain with its related source files.

        related_files: list of file paths or directories that, when changed,
        indicate the domain's search space may have shifted.
        """
        record = DomainRunRecord(domain_name=domain_name)
        if related_files:
            record.related_files = related_files
            # Record current mtimes
            for f in related_files:
                record.file_mtimes[f] = self._get_mtime(f)
        self.records[domain_name] = record
        self._domain_file_map[domain_name] = related_files or []

    def update_after_run(self, domain_name: str, loop: int, best_score: float,
                         converged: bool = False):
        """Update a domain's record after it has been run."""
        if domain_name not in self.records:
            self.register(domain_name)
        record = self.records[domain_name]
        record.last_run_loop = loop
        record.last_best_score = best_score
        record.last_run_time = time.time()
        if converged:
            record.consecutive_converged += 1
        else:
            record.consecutive_converged = 0
        # Update file mtimes
        for f in record.related_files:
            record.file_mtimes[f] = self._get_mtime(f)

    def get_stale_domains(self, current_loop: int) -> list[tuple[str, float]]:
        """Return list of (domain_name, priority) for domains that should be re-run.

        A domain is stale if:
        1. It hasn't been run in >= cooldown_loops loops, AND
        2. Either its related code changed OR it has converged many times
           (maybe the environment shifted enough to unlock new optima)
        """
        stale = []
        for name, record in self.records.items():
            loops_since = current_loop - record.last_run_loop
            if loops_since < self.cooldown_loops:
                continue

            # Check if related code changed
            code_changed = self._check_code_changes(record)

            # Calculate priority
            staleness_factor = 1.0 + (loops_since / 10.0)
            code_change_factor = 2.0 if code_changed else 1.0
            converged_factor = 1.0 + record.consecutive_converged * 0.3

            # Skip domains that have converged too many times without code changes
            if (record.consecutive_converged >= self.max_consecutive_converged
                    and not code_changed):
                continue

            priority = staleness_factor * code_change_factor * converged_factor
            stale.append((name, priority))

        # Sort by priority (highest first)
        stale.sort(key=lambda x: -x[1])
        return stale

    def _check_code_changes(self, record: DomainRunRecord) -> bool:
        """Check if any related files have been modified since last run."""
        for f in record.related_files:
            current_mtime = self._get_mtime(f)
            last_mtime = record.file_mtimes.get(f, 0)
            if current_mtime > last_mtime:
                return True
        return False

    def _get_mtime(self, path: str) -> float:
        """Get modification time of a file or directory (recursively)."""
        full_path = self.root / path
        if not full_path.exists():
            return 0.0
        if full_path.is_file():
            return full_path.stat().st_mtime
        # Directory: get the max mtime of all .py files
        max_mtime = 0.0
        for py_file in full_path.rglob("*.py"):
            mtime = py_file.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
        return max_mtime

    def get_status(self) -> dict:
        """Return a status summary of all tracked domains."""
        return {
            "total_domains": len(self.records),
            "converged": sum(1 for r in self.records.values()
                             if r.consecutive_converged > 0),
            "stale_candidates": len(self.get_stale_domains(0)),
        }
