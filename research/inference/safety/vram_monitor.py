"""VRAM pressure monitor for proactive OOM resilience.

R34-5: Graduated degradation before OOM, not reactive after.
Monitors torch.cuda.memory_allocated() and triggers degradation at
multiple pressure levels:
  - 70%: switch to compressed KV (s4r)
  - 80%: switch to 4-bit KV (snapkv_4bit)
  - 90%: switch to CPU offload
  - 95%: reduce max_new_tokens / context_limit

VRAM budget: negligible — stores only threshold config and history.
"""
from __future__ import annotations

import time
from typing import Optional, Callable


class VRAMPressureMonitor:
    """Monitors GPU VRAM usage and triggers graduated degradation.

    The monitor is called periodically (e.g. between generation rounds
    or during long prefill). It checks torch.cuda.memory_allocated()
    against configurable thresholds and calls registered degradation
    callbacks when pressure crosses each level.
    """

    LEVEL_NORMAL = 0
    LEVEL_COMPRESSED = 1
    LEVEL_AGGRESSIVE = 2
    LEVEL_OFFLOAD = 3
    LEVEL_CRITICAL = 4

    def __init__(
        self,
        budget_bytes: int = 12 * 1024**3,
        threshold_compressed: float = 0.70,
        threshold_aggressive: float = 0.80,
        threshold_offload: float = 0.90,
        threshold_critical: float = 0.95,
        check_interval_s: float = 2.0,
    ):
        self.budget = budget_bytes
        self.thresholds = {
            self.LEVEL_COMPRESSED: threshold_compressed,
            self.LEVEL_AGGRESSIVE: threshold_aggressive,
            self.LEVEL_OFFLOAD: threshold_offload,
            self.LEVEL_CRITICAL: threshold_critical,
        }
        self.check_interval = check_interval_s
        self._last_check = 0.0
        self._current_level = self.LEVEL_NORMAL
        self._callbacks: dict[int, Callable] = {}
        self._history: list[tuple[float, float]] = []
        self._max_history = 100

    def register_callback(self, level: int, callback: Callable):
        """Register a degradation callback for a pressure level."""
        self._callbacks[level] = callback

    def get_pressure(self) -> float:
        """Returns current VRAM pressure as fraction [0, 1]."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            allocated = torch.cuda.memory_allocated()
            return min(1.0, allocated / self.budget)
        except Exception:
            return 0.0

    def get_level(self, pressure: float | None = None) -> int:
        """Map pressure fraction to degradation level."""
        if pressure is None:
            pressure = self.get_pressure()
        level = self.LEVEL_NORMAL
        for lv, thresh in sorted(self.thresholds.items()):
            if pressure >= thresh:
                level = lv
        return level

    def maybe_degrade(self, force: bool = False) -> int:
        """Check pressure and trigger degradation if needed.

        Returns the current level. Calls the registered callback if
        the level increased since the last check.
        """
        now = time.monotonic()
        if not force and (now - self._last_check) < self.check_interval:
            return self._current_level
        self._last_check = now

        pressure = self.get_pressure()
        new_level = self.get_level(pressure)

        self._history.append((now, pressure))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if new_level > self._current_level:
            cb = self._callbacks.get(new_level)
            if cb:
                try:
                    cb(pressure, new_level)
                except Exception:
                    pass
        self._current_level = new_level
        return new_level

    def reset(self):
        """Reset to normal level (e.g. after model reload)."""
        self._current_level = self.LEVEL_NORMAL
        self._history.clear()

    def stats(self) -> dict:
        """Return current monitor statistics."""
        pressure = self.get_pressure()
        return {
            "pressure": pressure,
            "level": self.get_level(pressure),
            "level_name": self.level_name(self._current_level),
            "budget_gb": self.budget / 1e9,
            "allocated_gb": pressure * self.budget / 1e9,
            "history_len": len(self._history),
            "avg_pressure": (
                sum(p for _, p in self._history) / len(self._history)
                if self._history else 0.0
            ),
            "max_pressure": (
                max(p for _, p in self._history)
                if self._history else 0.0
            ),
        }

    @staticmethod
    def level_name(level: int) -> str:
        names = {
            VRAMPressureMonitor.LEVEL_NORMAL: "normal",
            VRAMPressureMonitor.LEVEL_COMPRESSED: "compressed",
            VRAMPressureMonitor.LEVEL_AGGRESSIVE: "aggressive",
            VRAMPressureMonitor.LEVEL_OFFLOAD: "offload",
            VRAMPressureMonitor.LEVEL_CRITICAL: "critical",
        }
        return names.get(level, "unknown")
