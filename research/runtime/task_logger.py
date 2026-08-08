"""Lightweight task status/logger for the ForgeAI web task monitor.

Each task gets a directory under research/tasks/<task_id>/ containing:
- status.json: live metadata and metrics
- log.txt: full stdout-style log
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def _now():
    return datetime.now(timezone.utc).isoformat()


class TaskLogger:
    def __init__(self, name, task_id=None, tasks_dir=None):
        self.name = name
        self.pid = os.getpid()
        self.tasks_dir = Path(tasks_dir) if tasks_dir else TASKS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.task_id = task_id or f"{name}_{timestamp}_{self.pid}"
        self.task_dir = self.tasks_dir / self.task_id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.task_dir / "status.json"
        self.log_file = self.task_dir / "log.txt"
        self._lock = threading.Lock()

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

    def _write_status(self):
        with self._lock:
            try:
                with open(self.status_file, "w", encoding="utf-8") as f:
                    json.dump(self.status, f, indent=2)
            except Exception:
                pass

    def _write_log(self, text):
        with self._lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except Exception:
                pass

    def log(self, message):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.status["message"] = message
        self.status["updated_at"] = _now()
        self._write_log(line)
        self._write_status()

    def update(self, status=None, metrics=None, progress=None):
        if status:
            self.status["status"] = status
        if metrics:
            self.status["metrics"].update(metrics)
        if progress:
            self.status["progress"].update(progress)
        self.status["updated_at"] = _now()
        self._write_status()

    def finish(self, status="completed", message=None):
        self.status["status"] = status
        self.status["updated_at"] = _now()
        if message:
            self.status["message"] = message
            self._write_log(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}")
        self._write_status()


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
