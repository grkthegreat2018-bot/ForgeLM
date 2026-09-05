"""GPU / compute monitoring via torch.cuda.

Falls back to zeroed stats when CUDA is unavailable so the GUI still renders.

R37: ``cached_snapshot()`` returns the most recent stats produced by a
background poller thread (``GpuPoller``), so the UI thread never blocks
on ``import torch`` (1-3 s CUDA runtime init) or CUDA API queries.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GpuStats:
    available: bool = False
    device_name: str = "—"
    index: int = 0
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    vram_pct: float = 0.0
    compute_pct: float = 0.0          # placeholder; torch doesn't expose utilization
    temperature_c: int = 0
    power_w: float = 0.0
    power_limit_w: float = 0.0
    cuda_version: str = ""

    @property
    def vram_label(self) -> str:
        if not self.available:
            return "CUDA unavailable"
        return f"{self.vram_allocated_gb:.2f} / {self.vram_total_gb:.2f} GB"


class GpuMonitor:
    """Wraps torch.cuda queries. Safe to call when CUDA is missing."""

    def __init__(self) -> None:
        self._torch = None
        self._torch_loaded = False
        # R37: thread-safe cache populated by GpuPoller (background thread).
        # UI threads read this via cached_snapshot() to avoid importing torch
        # on the event loop (1-3 s CUDA init freeze).
        self._cache_lock = threading.Lock()
        self._cached: GpuStats = GpuStats()

    def _ensure_torch(self):
        """Lazily import torch on first use — importing torch at construction
        time adds 1-3s to GUI startup (CUDA runtime init). Deferring to the
        first snapshot() call means the window appears before torch loads."""
        if self._torch_loaded:
            return self._torch
        self._torch_loaded = True
        try:
            import torch  # noqa
            self._torch = torch
        except Exception as e:
            logger.warning("torch unavailable — GPU monitoring disabled: %s", e)
            self._torch = None
        return self._torch

    @property
    def available(self) -> bool:
        t = self._ensure_torch()
        return bool(t and t.cuda.is_available())

    def cached_snapshot(self) -> GpuStats:
        """Return the most recent GpuStats from the background poller.

        Never blocks — returns a zeroed GpuStats if no poll has completed
        yet (e.g. during the first second after launch before torch has
        finished importing in the background thread).
        """
        with self._cache_lock:
            return self._cached

    def _set_cached(self, stats: GpuStats) -> None:
        """Called by GpuPoller to update the cache (thread-safe)."""
        with self._cache_lock:
            self._cached = stats

    def snapshot(self) -> GpuStats:
        t = self._ensure_torch()
        if not t or not t.cuda.is_available():
            return GpuStats()
        try:
            idx = t.cuda.current_device()
            name = t.cuda.get_device_name(idx)
            total = t.cuda.get_device_properties(idx).total_memory / 1e9
            alloc = t.cuda.memory_allocated(idx) / 1e9
            reserved = t.cuda.memory_reserved(idx) / 1e9
            free = max(0.0, total - alloc)
            pct = (alloc / total * 100.0) if total > 0 else 0.0
            cuda_ver = t.version.cuda or ""
            return GpuStats(
                available=True,
                device_name=name,
                index=idx,
                vram_allocated_gb=alloc,
                vram_reserved_gb=reserved,
                vram_total_gb=total,
                vram_free_gb=free,
                vram_pct=pct,
                cuda_version=str(cuda_ver),
            )
        except Exception as e:
            logger.warning("GPU stats query failed: %s", e)
            return GpuStats()

    def reset_peak(self) -> None:
        t = self._ensure_torch()
        if t and t.cuda.is_available():
            try:
                t.cuda.reset_peak_memory_stats()
            except Exception as e:
                logger.warning("reset_peak_memory_stats failed: %s", e)


class GpuPoller:
    """Background GPU stats poller using a plain daemon thread.

    Imports torch + queries CUDA in the background, updating the
    ``GpuMonitor`` cache so the UI event loop never blocks on CUDA init
    (1-3 s on first import).  The poller is a daemon thread — it stops
    automatically when the process exits, or call ``stop()`` for a clean
    shutdown.
    """

    def __init__(self, gpu: GpuMonitor, interval_s: float = 2.0) -> None:
        self._gpu = gpu
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="GpuPoller", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout_s)
        self._thread = None

    def _run(self) -> None:
        import time as _time
        while not self._stop.is_set():
            try:
                stats = self._gpu.snapshot()
                self._gpu._set_cached(stats)
            except Exception as e:
                logger.debug("GpuPoller snapshot error: %s", e)
            self._stop.wait(self._interval_s)
