"""Lightweight task status/logger for the ForgeAI web task monitor.

Each task gets a directory under research/tasks/<task_id>/ containing:
- status.json: live metadata and metrics
- log.txt: full stdout-style log
- outputs.jsonl: generation outputs from eval (one JSON per line)

Built-in diagnostics (read_log, read_output, bottleneck) eliminate the need
for one-off log-reading/profiling scripts. These are available on every
TaskLogger / task_scope instance.
"""
import atexit
import json
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime, timezone
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def _now():
    return datetime.now(UTC).isoformat()


def _ts_short():
    return datetime.now(UTC).strftime("%H:%M:%S")


class TaskLogger:
    """Task logger with built-in diagnostics.

    In addition to writing status.json + log.txt to disk, keeps in-memory
    ring buffers of structured events and generation outputs for queryable
    access via read_log() and read_output().
    """

    def __init__(self, name, task_id=None, tasks_dir=None,
                 log_capacity: int = 1000, output_capacity: int = 200):
        self.name = name
        self.pid = os.getpid()
        self.tasks_dir = Path(tasks_dir) if tasks_dir else TASKS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.task_id = task_id or f"{name}_{timestamp}_{self.pid}"
        self.task_dir = self.tasks_dir / self.task_id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.task_dir / "status.json"
        self.log_file = self.task_dir / "log.txt"
        self.outputs_file = self.task_dir / "outputs.jsonl"
        self._lock = threading.Lock()

        # In-memory ring buffers for queryable diagnostics.
        self._events: deque = deque(maxlen=log_capacity)
        self._outputs: deque = deque(maxlen=output_capacity)

        # Per-phase timing for bottleneck profiling.
        self._phase_timings: dict[str, list[float]] = {}

        # Persistent file handles for append-mode files (avoid open/close per call).
        # status_file is opened per-write (mode "w" overwrites) but throttled by time.
        self._log_handle = None
        self._outputs_handle = None
        self._write_count = 0
        self._flush_every = 64  # flush every N writes to append handles
        self._last_status_write = 0.0  # for time-based status throttle
        self._status_throttle_s = 1.0  # only write status if >1s since last write

        # Open persistent append handles
        try:
            self._log_handle = open(self.log_file, "a", encoding="utf-8")
            self._outputs_handle = open(self.outputs_file, "a", encoding="utf-8")
        except Exception as e:
            import warnings
            warnings.warn(f"task_logger: failed to open persistent handles: {e}",
                          RuntimeWarning, stacklevel=2)

        # Register atexit cleanup for persistent file handles
        atexit.register(self._atexit_close)

        self.status = {
            "id": self.task_id,
            "name": self.name,
            "pid": self.pid,
            "status": "running",
            "started_at": _now(),
            "updated_at": _now(),
            "message": "Task started",
            "metrics": {},
            "progress": {},
        }
        self._write_status()

    def _write_status(self, force: bool = False):
        """Write status.json (overwritten each time, time-throttled).

        Args:
            force: if True, bypass the time-based throttle (used by finish/update).
        """
        now = time.monotonic()
        if not force and (now - self._last_status_write) < self._status_throttle_s:
            return  # throttled — skip this write
        self._last_status_write = now
        with self._lock:
            try:
                with open(self.status_file, "w", encoding="utf-8") as f:
                    json.dump(self.status, f, indent=2)
            except Exception as e:
                import warnings
                warnings.warn(f"task_logger: failed to write status: {e}",
                              RuntimeWarning, stacklevel=2)

    def _write_log(self, text):
        with self._lock:
            try:
                if self._log_handle is not None:
                    self._log_handle.write(text + "\n")
                    self._write_count += 1
                    if self._write_count % self._flush_every == 0:
                        self._log_handle.flush()
                else:
                    # Fallback: open per-call if persistent handle failed
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
            except Exception as e:
                import warnings
                warnings.warn(f"task_logger: failed to write log: {e}",
                              RuntimeWarning, stacklevel=2)

    def log(self, message, level: str = "info", **data):
        """Log a message to disk + in-memory event buffer.

        Args:
            level: "info", "warn", "error", "profile"
            **data: extra structured fields stored in the event buffer
        """
        ts = _ts_short()
        line = f"[{ts}] {message}"
        self.status["message"] = message
        self.status["updated_at"] = _now()
        self._write_log(line)
        self._write_status()
        self._events.append({
            "time": ts,
            "level": level,
            "message": message,
            **data,
        })

    def warn(self, message, **data):
        self.log(message, level="warn", **data)

    def error(self, message, **data):
        self.log(message, level="error", **data)

    def update(self, status=None, metrics=None, progress=None):
        if status:
            self.status["status"] = status
        if metrics:
            self.status["metrics"].update(metrics)
        if progress:
            self.status["progress"].update(progress)
        self.status["updated_at"] = _now()
        self._write_status(force=True)

    def record_output(self, prompt: str, output: str, **metadata):
        """Record a generation output (from eval/sample) to disk + buffer."""
        entry = {"time": _ts_short(), "prompt": prompt[:200],
                 "output": output[:500], **metadata}
        self._outputs.append(entry)
        with self._lock:
            try:
                if self._outputs_handle is not None:
                    self._outputs_handle.write(
                        json.dumps(entry, ensure_ascii=False) + "\n")
                    self._write_count += 1
                    if self._write_count % self._flush_every == 0:
                        self._outputs_handle.flush()
                else:
                    with open(self.outputs_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def time_phase(self, phase: str):
        """Context manager for timing a training phase (bottleneck profiling).

        Usage:
            with log.time_phase("forward"):
                loss = model(...)
            # log._phase_timings["forward"] now has the elapsed time
        """
        return _PhaseTimer(self, phase)

    def flush(self):
        """Flush all buffered writes to disk."""
        with self._lock:
            if self._log_handle is not None:
                try:
                    self._log_handle.flush()
                except Exception:
                    pass
            if self._outputs_handle is not None:
                try:
                    self._outputs_handle.flush()
                except Exception:
                    pass

    def _atexit_close(self):
        """Close persistent file handles (called at interpreter exit)."""
        self.flush()
        with self._lock:
            for attr in ("_log_handle", "_outputs_handle"):
                handle = getattr(self, attr, None)
                if handle is not None and not handle.closed:
                    try:
                        handle.close()
                    except Exception:
                        pass
                setattr(self, attr, None)

    def __del__(self):
        """Close persistent file handles on garbage collection."""
        try:
            self._atexit_close()
        except Exception:
            pass

    def finish(self, status="completed", message=None):
        self.status["status"] = status
        self.status["updated_at"] = _now()
        if message:
            self.status["message"] = message
            self._write_log(f"[{_ts_short()}] {message}")
        self._write_status(force=True)
        self.flush()

    # ── Built-in diagnostics (replaces one-off scripts) ──────────────────

    def read_log(self, n: int = 50, level: str | None = None) -> list[dict]:
        """Read recent log events as structured dicts (newest last).

        Replaces log-tailing scripts:
            log.read_log(n=20, level="error")  # recent errors
            log.read_log(n=100)                 # last 100 events
        """
        events = list(self._events)
        if level:
            events = [e for e in events if e.get("level") == level]
        return events[-n:] if n > 0 else events

    def read_output(self, n: int = 10) -> list[dict]:
        """Read recent generation outputs (from eval/sample).

        Replaces output-capture scripts:
            log.read_output(n=5)  # last 5 eval generations
        """
        outputs = list(self._outputs)
        return outputs[-n:] if n > 0 else outputs

    def bottleneck(self) -> dict:
        """Report per-phase timing from time_phase() calls.

        Shows which training phases (forward, backward, optimizer, data_load)
        are slowest. Call after training steps that used time_phase():
            report = log.bottleneck()
            print(report["bottlenecks"])
        """
        per_phase = []
        for phase, times in self._phase_timings.items():
            if not times:
                continue
            total = sum(times)
            per_phase.append({
                "phase": phase,
                "total_ms": round(total * 1000, 2),
                "calls": len(times),
                "avg_ms": round(total * 1000 / len(times), 2),
            })
        per_phase.sort(key=lambda x: x["total_ms"], reverse=True)
        total_ms = sum(p["total_ms"] for p in per_phase)
        return {
            "per_phase": per_phase,
            "bottlenecks": per_phase[:5],
            "total_ms": round(total_ms, 2),
            "n_phases": len(per_phase),
        }

    def diagnose(self) -> dict:
        """Health report for this training task.

        Combines status + metrics + recent errors + bottleneck summary:
            report = log.diagnose()
        """
        errors = [e for e in self._events if e.get("level") == "error"]
        warns = [e for e in self._events if e.get("level") == "warn"]
        status = self.status["status"]
        report_status = "healthy"
        warnings = []
        if errors:
            warnings.append(f"{len(errors)} error(s) logged")
            report_status = "warning" if report_status == "healthy" else report_status
        if status == "failed":
            report_status = "failed"
        elif status == "running" and warns:
            report_status = "warning"
            warnings.append(f"{len(warns)} warning(s) logged")
        return {
            "status": report_status,
            "task_status": status,
            "task_id": self.task_id,
            "name": self.name,
            "metrics": self.status.get("metrics", {}),
            "progress": self.status.get("progress", {}),
            "warnings": warnings,
            "recent_errors": errors[-5:],
            "recent_warnings": warns[-5:],
            "bottleneck": self.bottleneck() if self._phase_timings else None,
        }


class _PhaseTimer:
    """Context manager for timing a named training phase."""

    def __init__(self, logger: TaskLogger, phase: str):
        self.logger = logger
        self.phase = phase
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        self.logger._phase_timings.setdefault(self.phase, []).append(elapsed)


# Convenience context manager for graceful shutdown.
class task_scope:
    def __init__(self, name, task_id=None):
        self.logger = TaskLogger(name, task_id=task_id)

    def __enter__(self):
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.finish("failed", f"Failed: {exc_val}")
        else:
            self.logger.finish("completed", "Task finished")
