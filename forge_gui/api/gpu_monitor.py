"""GPU / compute monitoring via torch.cuda.

Falls back to zeroed stats when CUDA is unavailable so the GUI still renders.
"""
from __future__ import annotations

import logging
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
        try:
            import torch  # noqa
            self._torch = torch
        except Exception as e:
            logger.warning("torch unavailable — GPU monitoring disabled: %s", e)
            self._torch = None

    @property
    def available(self) -> bool:
        return bool(self._torch and self._torch.cuda.is_available())

    def snapshot(self) -> GpuStats:
        t = self._torch
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
        t = self._torch
        if t and t.cuda.is_available():
            try:
                t.cuda.reset_peak_memory_stats()
            except Exception as e:
                logger.warning("reset_peak_memory_stats failed: %s", e)
