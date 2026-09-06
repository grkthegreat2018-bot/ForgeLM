"""Built-in diagnostics for ForgeEngine and trainers.

Eliminates the need for one-off profiling/log-reading scripts by providing:
  - EngineProfiler: per-layer timing, bottleneck identification, memory tracking
  - EventLog: structured ring-buffer log (replaces scattered print() calls)
  - OutputHistory: ring-buffer of past generation outputs with metadata
  - HealthReport: combined health check (stats + bottleneck + memory + warnings)

These are mixed into ForgeEngine via composition (self.profiler, self.events,
self.outputs) and into SFTTrainer via the same EventLog/OutputHistory classes.

Usage in ForgeEngine:
    engine = ForgeEngine(model, tok)
    engine.bottleneck(prompt="test", max_new_tokens=32)  # per-layer timing
    engine.read_log(n=20)                                 # recent events
    engine.read_output(n=5)                               # recent generations
    engine.diagnose()                                     # full health report

Usage in SFTTrainer:
    trainer = SFTTrainer(...)
    trainer.read_log(n=50)        # training events
    trainer.read_output(n=10)     # sample generations from eval
    trainer.bottleneck(n_steps=5) # per-phase training timing
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch


# ── EventLog ──────────────────────────────────────────────────────────────

@dataclass
class Event:
    timestamp: float
    level: str          # "info", "warn", "error", "profile"
    source: str         # "engine", "trainer", "kv_cache", etc.
    message: str
    data: dict = field(default_factory=dict)


class EventLog:
    """Thread-safe ring-buffer log of structured events.

    Replaces scattered print() calls with a queryable event store.
    Access via read_log() or to_list().
    """

    def __init__(self, capacity: int = 500):
        self._events: deque[Event] = deque(maxlen=capacity)

    def log(self, message: str, level: str = "info",
            source: str = "engine", **data):
        self._events.append(Event(
            timestamp=time.time(),
            level=level,
            source=source,
            message=message,
            data=data,
        ))

    def warn(self, message: str, source: str = "engine", **data):
        self.log(message, level="warn", source=source, **data)

    def error(self, message: str, source: str = "engine", **data):
        self.log(message, level="error", source=source, **data)

    def profile(self, message: str, source: str = "engine", **data):
        self.log(message, level="profile", source=source, **data)

    def read_log(self, n: int = 50, level: str | None = None,
                 source: str | None = None) -> list[dict]:
        """Read recent events as dicts (newest last). Optional level/source filter."""
        events = list(self._events)
        if level:
            events = [e for e in events if e.level == level]
        if source:
            events = [e for e in events if e.source == source]
        events = events[-n:] if n > 0 else events
        return [
            {
                "time": _format_ts(e.timestamp),
                "level": e.level,
                "source": e.source,
                "message": e.message,
                **e.data,
            }
            for e in events
        ]

    def to_list(self) -> list[dict]:
        return self.read_log(n=0)

    def clear(self):
        self._events.clear()

    def __len__(self):
        return len(self._events)


def _format_ts(ts: float) -> str:
    """Format a unix timestamp as HH:MM:SS.mmm."""
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.") + f"{int((ts % 1) * 1000):03d}"


# ── OutputHistory ─────────────────────────────────────────────────────────

@dataclass
class OutputRecord:
    timestamp: float
    prompt: str
    output: str
    tokens_generated: int
    time_ms: float
    tokens_per_sec: float
    temperature: float
    metadata: dict = field(default_factory=dict)


class OutputHistory:
    """Ring-buffer of past generation outputs with metadata.

    Access via read_output() to retrieve recent generations without
    re-running the model.
    """

    def __init__(self, capacity: int = 100):
        self._records: deque[OutputRecord] = deque(maxlen=capacity)

    def record(self, prompt: str, output: str, tokens_generated: int,
               time_ms: float, temperature: float = 0.0, **metadata):
        tps = tokens_generated / (time_ms / 1000) if time_ms > 0 else 0
        self._records.append(OutputRecord(
            timestamp=time.time(),
            prompt=prompt,
            output=output,
            tokens_generated=tokens_generated,
            time_ms=time_ms,
            tokens_per_sec=tps,
            temperature=temperature,
            metadata=metadata,
        ))

    def read_output(self, n: int = 10) -> list[dict]:
        """Read recent generation outputs as dicts (newest last)."""
        records = list(self._records)[-n:] if n > 0 else list(self._records)
        return [
            {
                "time": _format_ts(r.timestamp),
                "prompt": r.prompt[:200],
                "output": r.output[:500],
                "tokens": r.tokens_generated,
                "time_ms": round(r.time_ms, 1),
                "tok_s": round(r.tokens_per_sec, 1),
                "temperature": r.temperature,
                **r.metadata,
            }
            for r in records
        ]

    def __len__(self):
        return len(self._records)


# ── EngineProfiler ────────────────────────────────────────────────────────

class EngineProfiler:
    """Per-layer timing profiler for ForgeEngine.

    Runs a short generation and records wall-clock time per transformer block,
    identifying the slowest layers (bottlenecks). Uses forward hooks that
    are removed after profiling completes.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._hooks: list = []

    def _find_blocks(self) -> list:
        """Locate the transformer block ModuleList on the model."""
        for attr in ("blocks", "layers", "transformer", "h"):
            mod = getattr(self.model, attr, None)
            if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
                return list(mod)
        return []

    def profile_generate(self, input_ids: torch.Tensor,
                         max_new_tokens: int = 16) -> dict:
        """Profile a single generation pass, returning per-layer timings.

        Returns dict with:
          - per_layer_ms: list of {index, type, time_ms} per block
          - bottlenecks: top-5 slowest layers
          - total_ms: total generation time
          - tokens: number of tokens generated
          - tok_s: tokens per second
        """
        blocks = self._find_blocks()
        if not blocks:
            return {"error": "No transformer blocks found on model"}

        timings: list[float] = [0.0] * len(blocks)
        counts: list[int] = [0] * len(blocks)

        def make_pre_hook(idx: int):
            def pre_hook(module, args):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                module._profile_start = time.perf_counter()
            return pre_hook

        def make_post_hook(idx: int):
            def post_hook(module, args, kwargs, output):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t = time.perf_counter()
                start = getattr(module, "_profile_start", None)
                if start is not None:
                    timings[idx] += (t - start) * 1000
                    counts[idx] += 1
                    module._profile_start = None
            return post_hook

        # Register both pre- and post-forward hooks per block
        for i, block in enumerate(blocks):
            pre = block.register_forward_pre_hook(make_pre_hook(i))
            post = block.register_forward_hook(make_post_hook(i), with_kwargs=True)
            self._hooks.append(pre)
            self._hooks.append(post)

        try:
            # Run generation
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()

            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    self.model(input_ids if _ == 0 else input_ids[:, -1:],
                               use_cache=True)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            total_ms = (time.perf_counter() - t0) * 1000
        finally:
            # Always remove hooks
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

        # Build results
        per_layer = []
        for i, (t_ms, cnt) in enumerate(zip(timings, counts)):
            layer_type = type(blocks[i]).__name__
            per_layer.append({
                "index": i,
                "type": layer_type,
                "time_ms": round(t_ms, 2),
                "calls": cnt,
                "avg_ms": round(t_ms / cnt, 2) if cnt > 0 else 0,
            })

        # Sort by time to find bottlenecks
        sorted_layers = sorted(per_layer, key=lambda x: x["time_ms"], reverse=True)
        bottlenecks = sorted_layers[:5]

        total_layer_ms = sum(t["time_ms"] for t in per_layer)
        non_layer_ms = total_ms - total_layer_ms

        return {
            "per_layer_ms": per_layer,
            "bottlenecks": bottlenecks,
            "total_ms": round(total_ms, 2),
            "layer_ms": round(total_layer_ms, 2),
            "non_layer_ms": round(non_layer_ms, 2),
            "tokens": max_new_tokens,
            "tok_s": round(max_new_tokens / (total_ms / 1000), 1) if total_ms > 0 else 0,
            "n_layers": len(blocks),
        }


# ── HealthReport ──────────────────────────────────────────────────────────

def build_health_report(engine) -> dict:
    """Combined health check for a ForgeEngine.

    Aggregates stats() + memory + active features + warnings into a single
    report. Does NOT run generation (non-invasive).
    """
    report: dict[str, Any] = {
        "status": "healthy",
        "warnings": [],
        "timestamp": _format_ts(time.time()),
    }

    # Engine stats
    try:
        stats = engine.stats()
        report["stats"] = stats
    except Exception as e:
        report["warnings"].append(f"stats() failed: {e}")
        report["status"] = "degraded"

    # VRAM
    if engine.device.type == "cuda":
        try:
            vram = engine.vram_usage()
            report["vram"] = vram
            if vram.get("percent", 0) > 90:
                report["warnings"].append(
                    f"VRAM at {vram['percent']:.0f}% — risk of OOM"
                )
                report["status"] = "warning"
        except Exception as e:
            report["warnings"].append(f"vram_usage() failed: {e}")

    # Awake check
    if hasattr(engine, "_awake") and not engine._awake:
        report["warnings"].append("Engine is asleep — call wake() before generation")
        report["status"] = "warning"

    # KV cache
    if engine.kv_cache is None:
        report["warnings"].append("No KV cache strategy active — using default")
    else:
        try:
            info = engine.kv_cache.info()
            report["kv_cache"] = info
        except Exception:
            pass

    # Recent errors from event log
    if hasattr(engine, "events"):
        errors = engine.events.read_log(n=10, level="error")
        if errors:
            report["warnings"].append(
                f"{len(errors)} recent error(s) in event log"
            )
            if report["status"] == "healthy":
                report["status"] = "warning"
            report["recent_errors"] = errors

    # Recent warnings from event log
    if hasattr(engine, "events"):
        warns = engine.events.read_log(n=10, level="warn")
        if warns:
            report["recent_warnings"] = warns

    return report


# ── BandwidthProfiler ──────────────────────────────────────────────────────

class BandwidthProfiler:
    """Profile PCIe bandwidth and CPU-GPU transfer speeds.

    FreeToken-style bandwidth-adaptive CPU-GPU co-execution requires
    knowing the actual transfer speed to optimally split work.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._cpu_to_gpu_gbps: float | None = None
        self._gpu_to_cpu_gbps: float | None = None
        self._cpu_compute_gflops: float | None = None
        self._gpu_compute_gflops: float | None = None

    def benchmark_transfers(self, size_mb: int = 64) -> dict:
        """Benchmark CPU↔GPU transfer bandwidth."""
        import time
        if self.device.type != "cuda":
            return {"cpu_to_gpu_gbps": 0, "gpu_to_cpu_gbps": 0}
        n_bytes = size_mb * 1024 * 1024
        n_elements = n_bytes // 2  # float16
        # CPU → GPU
        cpu_tensor = torch.randn(n_elements, dtype=torch.float16, pin_memory=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        self._cpu_to_gpu_gbps = (n_bytes / 1e9) / dt
        # GPU → CPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            _ = gpu_tensor.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        self._gpu_to_cpu_gbps = (n_bytes / 1e9) / dt
        del cpu_tensor, gpu_tensor
        torch.cuda.empty_cache()
        return {
            "cpu_to_gpu_gbps": self._cpu_to_gpu_gbps,
            "gpu_to_cpu_gbps": self._gpu_to_cpu_gbps,
        }

    def benchmark_compute(self, n: int = 4096) -> dict:
        """Benchmark CPU and GPU matmul throughput."""
        import time
        # GPU compute
        if self.device.type == "cuda":
            a = torch.randn(n, n, dtype=torch.float16, device=self.device)
            b = torch.randn(n, n, dtype=torch.float16, device=self.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                c = a @ b
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            self._gpu_compute_gflops = (2 * n**3 / 1e9) / dt
            del a, b, c
            torch.cuda.empty_cache()
        # CPU compute
        a = torch.randn(n, n, dtype=torch.float32)
        b = torch.randn(n, n, dtype=torch.float32)
        t0 = time.perf_counter()
        for _ in range(2):
            c = a @ b
        dt = (time.perf_counter() - t0) / 2
        self._cpu_compute_gflops = (2 * n**3 / 1e9) / dt
        del a, b, c
        return {
            "gpu_compute_gflops": self._gpu_compute_gflops or 0,
            "cpu_compute_gflops": self._cpu_compute_gflops,
        }

    def optimal_split(self, total_flops: float, transfer_bytes: float) -> dict:
        """Compute optimal CPU/GPU work split.

        Returns {gpu_fraction, cpu_fraction, estimated_speedup}.
        """
        gpu_gflops = self._gpu_compute_gflops or 0
        cpu_gflops = self._cpu_compute_gflops or 0
        transfer_gbps = self._cpu_to_gpu_gbps or 0
        if gpu_gflops == 0:
            return {"gpu_fraction": 0, "cpu_fraction": 1, "estimated_speedup": 1.0}
        if cpu_gflops == 0 or transfer_gbps == 0:
            return {"gpu_fraction": 1, "cpu_fraction": 0, "estimated_speedup": 1.0}
        # Simple model: find split that equalizes time
        # GPU time = (f * total_flops) / (gpu_gflops * 1e9)
        # CPU time = ((1-f) * total_flops) / (cpu_gflops * 1e9) + transfer_bytes / (transfer_gbps * 1e9)
        # Equalize: solve for f
        best_f = 1.0
        best_time = float('inf')
        for f in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
            gpu_time = (f * total_flops) / (gpu_gflops * 1e9)
            cpu_time = ((1 - f) * total_flops) / (cpu_gflops * 1e9)
            transfer_time = (transfer_bytes * (1 - f)) / (transfer_gbps * 1e9)
            total_time = max(gpu_time, cpu_time + transfer_time)
            if total_time < best_time:
                best_time = total_time
                best_f = f
        # Speedup vs GPU-only
        gpu_only_time = total_flops / (gpu_gflops * 1e9)
        speedup = gpu_only_time / best_time if best_time > 0 else 1.0
        return {
            "gpu_fraction": best_f,
            "cpu_fraction": 1 - best_f,
            "estimated_speedup": speedup,
        }

    def profile_all(self) -> dict:
        """Run all benchmarks and return results."""
        return {
            "transfers": self.benchmark_transfers(),
            "compute": self.benchmark_compute(),
        }
