"""PCIe bandwidth profiler for bandwidth-adaptive CPU-GPU offload.

Inspired by FreeToken (arXiv:2608.16157): the q* policy splits work between
PCIe transfer (cache fill) and CPU execution based on two *measured*
bandwidths:
  - B_P: pinned-host → GPU transfer bandwidth (PCIe)
  - B_H: host-side processing bandwidth (CPU DRAM read + compute)

FreeToken profiles these at deployment. We do the same, but with a novel
twist: BandwidthPredictor tracks the bandwidth *trend* over training steps
and can pre-emptively adjust the offload policy before OOM occurs (see
BandwidthPredictor below).

Usage:
  from research.runtime.bandwidth_profiler import BandwidthProfiler
  profiler = BandwidthProfiler(device="cuda")
  bp, bh = profiler.profile()          # one-shot measurement
  bp, bh = profiler.get_cached()       # cached result (no re-measure)

  # Adaptive: re-profile when VRAM pressure changes
  bp, bh = profiler.profile_if_stale(max_age_steps=100)
"""
from __future__ import annotations

import time
import threading
import torch
from dataclasses import dataclass, field


@dataclass
class BandwidthResult:
    """Measured bandwidths and derived q* parameters."""
    b_p: float = 0.0  # PCIe transfer bandwidth (GB/s) — GPU→CPU or CPU→GPU
    b_h: float = 0.0  # Host processing bandwidth (GB/s) — CPU read + compute
    q_star_ratio: float = 0.0  # b_p / b_h — fraction of misses to cache-fill
    measured_at: float = 0.0  # time.perf_counter() timestamp
    transfer_ms_per_gb: float = 0.0  # ms per GB transferred
    profiled: bool = False

    def q_star(self, m: int) -> int:
        """Compute q* = number of items to cache-fill out of m misses.

        q* = m * B_P / B_H  (FreeToken Eq. 4)
        Always at least 1 (keep cache warming).
        """
        if m <= 0:
            return 0
        q = int(round(m * self.q_star_ratio))
        return max(1, min(q, m))


class BandwidthProfiler:
    """Profile PCIe transfer and host processing bandwidths.

    Measures:
      1. B_P: Time a large GPU→CPU pinned transfer, compute GB/s.
      2. B_H: Time a large CPU tensor operation (read + compute), compute GB/s.

    The results are cached and only re-measured when explicitly requested
    or when VRAM conditions change significantly.
    """

    def __init__(self, device: str = "cuda",
                 probe_size_mb: int = 512,
                 warmup: int = 2,
                 trials: int = 5):
        self.device = torch.device(device)
        self.probe_bytes = probe_size_mb * 1024 * 1024
        self.warmup = warmup
        self.trials = trials
        self._result: BandwidthResult | None = None
        self._lock = threading.Lock()

    def profile(self) -> tuple[float, float]:
        """Run a fresh bandwidth measurement. Returns (B_P, B_H) in GB/s."""
        with self._lock:
            b_p = self._measure_transfer_bandwidth()
            b_h = self._measure_host_bandwidth()
            ratio = b_p / b_h if b_h > 0 else 1.0
            self._result = BandwidthResult(
                b_p=b_p, b_h=b_h, q_star_ratio=ratio,
                measured_at=time.perf_counter(),
                transfer_ms_per_gb=1000.0 / b_p if b_p > 0 else 0.0,
                profiled=True,
            )
            return b_p, b_h

    def get_cached(self) -> BandwidthResult:
        """Return cached result, or profile if none exists."""
        if self._result is None:
            self.profile()
        assert self._result is not None
        return self._result

    def profile_if_stale(self, max_age_seconds: float = 300.0) -> BandwidthResult:
        """Re-profile if the cached result is older than max_age_seconds."""
        r = self._result
        if r is None or (time.perf_counter() - r.measured_at) > max_age_seconds:
            self.profile()
        return self.get_cached()

    def _measure_transfer_bandwidth(self) -> float:
        """Measure GPU→CPU pinned transfer bandwidth (B_P) in GB/s.

        Uses a large probe tensor to amortize launch overhead. Measures
        both directions and takes the average (training needs both).
        """
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return 25.0  # conservative default for PCIe 4.0 x16

        # Allocate probe on GPU and pinned CPU buffer
        n_elements = self.probe_bytes // 4  # float32
        gpu_tensor = torch.randn(n_elements, device=self.device, dtype=torch.float32)
        cpu_pinned = torch.empty(n_elements, dtype=torch.float32).pin_memory()

        # Warmup
        for _ in range(self.warmup):
            cpu_pinned.copy_(gpu_tensor, non_blocking=True)
            torch.cuda.synchronize()
        for _ in range(self.warmup):
            gpu_tensor.copy_(cpu_pinned, non_blocking=True)
            torch.cuda.synchronize()

        # Measure GPU→CPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.trials):
            cpu_pinned.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.synchronize()
        g2c_time = (time.perf_counter() - t0) / self.trials

        # Measure CPU→GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.trials):
            gpu_tensor.copy_(cpu_pinned, non_blocking=True)
        torch.cuda.synchronize()
        c2g_time = (time.perf_counter() - t0) / self.trials

        bytes_transferred = self.probe_bytes
        b_p_g2c = bytes_transferred / g2c_time / 1e9  # GB/s
        b_p_c2g = bytes_transferred / c2g_time / 1e9
        b_p = (b_p_g2c + b_p_c2g) / 2  # average both directions

        del gpu_tensor, cpu_pinned
        torch.cuda.empty_cache()
        return b_p

    def _measure_host_bandwidth(self) -> float:
        """Measure host-side processing bandwidth (B_H) in GB/s.

        Simulates CPU optimizer work: read grad + read master + read/write
        exp_avg + read/write exp_avg_sq. This is ~5x tensor size in bytes
        read/written per element (grad fp32 + master fp32 + m fp32 + v fp32).
        """
        n_elements = self.probe_bytes // 4  # float32
        grad = torch.randn(n_elements, dtype=torch.float32)
        master = torch.randn(n_elements, dtype=torch.float32)
        exp_avg = torch.zeros(n_elements, dtype=torch.float32)
        exp_avg_sq = torch.zeros(n_elements, dtype=torch.float32)

        # Warmup
        for _ in range(self.warmup):
            exp_avg.mul_(0.9).add_(grad, alpha=0.1)
            exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
            master.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(1e-8), value=-1e-3)

        # Reset for measurement
        exp_avg.zero_()
        exp_avg_sq.zero_()

        # Measure: AdamW-style update (5 tensor passes: grad, master, m, v, m again)
        torch.cuda.synchronize()  # ensure no GPU interference
        t0 = time.perf_counter()
        for _ in range(self.trials):
            exp_avg.mul_(0.9).add_(grad, alpha=0.1)
            exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
            master.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(1e-8), value=-1e-3)
        elapsed = (time.perf_counter() - t0) / self.trials

        # Bytes touched: grad(4B) + master(4B) + m(4B read+write=8B) + v(4B read+write=8B)
        # = 28 bytes per element, but effective is ~5x tensor size for bandwidth estimate
        bytes_processed = n_elements * 4 * 5  # 5 tensor passes
        b_h = bytes_processed / elapsed / 1e9  # GB/s

        del grad, master, exp_avg, exp_avg_sq
        return b_h


class BandwidthPredictor:
    """Novel: Track bandwidth trend over training steps and predict OOM.

    FreeToken profiles bandwidth once at deployment. We go further: track
    the bandwidth *trend* over training steps. If PCIe bandwidth is
    degrading (e.g., due to thermal throttling or VRAM fragmentation
    causing slower transfers), pre-emptively adjust the offload policy
    before an OOM occurs.

    The predictor maintains a sliding window of (step, bandwidth, vram)
    samples and uses linear regression to project when VRAM will hit the
    limit. If the projected time-to-OOM is < threshold, it signals the
    optimizer to increase offload (move more states to CPU).
    """

    def __init__(self, window_size: int = 50, vram_limit_gb: float = 11.0,
                 preempt_threshold_steps: int = 20):
        self.window_size = window_size
        self.vram_limit_gb = vram_limit_gb
        self.preempt_threshold_steps = preempt_threshold_steps
        self._samples: list[tuple[int, float, float]] = []  # (step, b_p, vram_gb)
        self._lock = threading.Lock()

    def record(self, step: int, b_p: float, vram_gb: float):
        """Record a bandwidth + VRAM sample."""
        with self._lock:
            self._samples.append((step, b_p, vram_gb))
            if len(self._samples) > self.window_size:
                self._samples.pop(0)

    def predict_vram_at(self, future_step: int) -> float | None:
        """Predict VRAM usage at a future step using linear regression.

        Returns None if not enough samples.
        """
        with self._lock:
            if len(self._samples) < 5:
                return None
            steps = [s[0] for s in self._samples]
            vrams = [s[2] for s in self._samples]
            n = len(steps)
            sum_x = sum(steps)
            sum_y = sum(vrams)
            sum_xy = sum(s * v for s, v in zip(steps, vrams))
            sum_x2 = sum(s * s for s in steps)
            denom = n * sum_x2 - sum_x * sum_x
            if denom == 0:
                return None
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            return slope * future_step + intercept

    def should_preempt_offload(self, current_step: int) -> bool:
        """Check if we should increase offload to avoid projected OOM.

        Returns True if predicted VRAM at (current_step + threshold) exceeds
        the limit, or if bandwidth is degrading (suggesting pressure).
        """
        predicted = self.predict_vram_at(current_step + self.preempt_threshold_steps)
        if predicted is not None and predicted > self.vram_limit_gb:
            return True

        # Also check bandwidth degradation: if recent B_P < 80% of early B_P
        with self._lock:
            if len(self._samples) < 10:
                return False
            early_bp = sum(s[1] for s in self._samples[:5]) / 5
            recent_bp = sum(s[1] for s in self._samples[-5:]) / 5
            if early_bp > 0 and recent_bp < early_bp * 0.8:
                return True

        return False

    def stats(self) -> dict:
        """Return current predictor statistics for logging."""
        with self._lock:
            if not self._samples:
                return {"samples": 0}
            recent = self._samples[-1]
            return {
                "samples": len(self._samples),
                "last_step": recent[0],
                "last_bp": f"{recent[1]:.1f} GB/s",
                "last_vram": f"{recent[2]:.2f} GB",
                "limit": f"{self.vram_limit_gb:.1f} GB",
            }


# Global singleton (profiling is expensive, share results)
_global_profiler: BandwidthProfiler | None = None
_global_lock = threading.Lock()


def get_bandwidth_profiler(device: str = "cuda") -> BandwidthProfiler:
    """Get or create the global bandwidth profiler singleton."""
    global _global_profiler
    with _global_lock:
        if _global_profiler is None:
            _global_profiler = BandwidthProfiler(device=device)
        return _global_profiler
