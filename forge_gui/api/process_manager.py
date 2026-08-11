"""Process manager — spawns subprocesses, captures stdout/stderr, emits Qt signals.

Each managed process runs as a QThread that reads the subprocess pipe
line-by-line and emits signals. The ProcessManager tracks all live processes
and provides a unified feed for the Tasks page.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from .status_reader import project_root

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Metadata for a launched process."""
    id: str
    name: str
    command: list[str]
    pid: int = 0
    status: str = "starting"  # starting | running | done | crashed | killed
    started_at: float = 0.0
    ended_at: float = 0.0
    exit_code: int | None = None
    log_path: Path | None = None
    lines: list[str] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        end = self.ended_at if self.ended_at else time.time()
        return max(0.0, end - self.started_at)

    @property
    def is_live(self) -> bool:
        return self.status in ("starting", "running")


class _ProcessWorker(QThread):
    """Runs a subprocess, reads stdout+stderr line-by-line, emits signals."""

    line = Signal(str, str)    # task_id, line_text
    status = Signal(str, str)  # task_id, status_string
    finished = Signal(str, int)  # task_id, exit_code

    def __init__(self, task_id: str, cmd: list[str], cwd: str,
                 env: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.task_id = task_id
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self._proc: os.subprocess | None = None  # type: ignore
        self._killed = False

    def kill_proc(self) -> None:
        self._killed = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception as e:
                logger.warning("terminate failed: %s", e)

    def run(self) -> None:
        import subprocess
        try:
            self.status.emit(self.task_id, "running")
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=self.env or os.environ.copy(),
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                line = raw.rstrip("\r\n")
                if line:
                    self.line.emit(self.task_id, line)
                if self._killed:
                    break
            self._proc.wait()
            code = self._proc.returncode
            if self._killed:
                self.status.emit(self.task_id, "killed")
            elif code == 0:
                self.status.emit(self.task_id, "done")
            else:
                self.status.emit(self.task_id, "crashed")
            self.finished.emit(self.task_id, code)
        except Exception as e:
            self.line.emit(self.task_id, f"[PROCESS ERROR] {type(e).__name__}: {e}")
            self.status.emit(self.task_id, "crashed")
            self.finished.emit(self.task_id, -1)


class ProcessManager(QObject):
    """Singleton-ish manager tracking all launched processes.

    Emits signals that the Launch + Tasks pages connect to for live updates.
    """
    line = Signal(str, str)    # task_id, line
    status_changed = Signal(str, str)  # task_id, status
    task_added = Signal(str)   # task_id
    task_removed = Signal(str) # task_id

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tasks: dict[str, TaskInfo] = {}
        self._workers: dict[str, _ProcessWorker] = {}
        self._counter = 0

    def launch(self, name: str, cmd: list[str],
               cwd: str | None = None) -> str:
        """Spawn a subprocess, return the task_id."""
        self._counter += 1
        task_id = f"task_{self._counter:04d}"
        root = project_root()
        work_dir = cwd or str(root)

        # Write logs to research/tasks/<task_id>/
        task_dir = root / "research" / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_dir / "log.txt"

        info = TaskInfo(
            id=task_id, name=name, command=cmd,
            started_at=time.time(), log_path=log_path,
        )
        self.tasks[task_id] = info

        worker = _ProcessWorker(task_id, cmd, work_dir, parent=self)
        worker.line.connect(self._on_line)
        worker.status.connect(self._on_status)
        worker.finished.connect(self._on_finished)
        self._workers[task_id] = worker
        worker.start()

        self.task_added.emit(task_id)
        return task_id

    def kill(self, task_id: str) -> None:
        w = self._workers.get(task_id)
        if w:
            w.kill_proc()

    def remove(self, task_id: str) -> None:
        """Remove a finished task from tracking."""
        if task_id in self.tasks and not self.tasks[task_id].is_live:
            del self.tasks[task_id]
            self.task_removed.emit(task_id)

    def live_tasks(self) -> list[TaskInfo]:
        return [t for t in self.tasks.values() if t.is_live]

    def all_tasks(self) -> list[TaskInfo]:
        return list(self.tasks.values())

    def _on_line(self, task_id: str, line: str) -> None:
        info = self.tasks.get(task_id)
        if info:
            info.lines.append(line)
            # Persist to log file
            if info.log_path:
                try:
                    with open(info.log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
        self.line.emit(task_id, line)

    def _on_status(self, task_id: str, status: str) -> None:
        info = self.tasks.get(task_id)
        if info:
            info.status = status
            if status in ("done", "crashed", "killed"):
                info.ended_at = time.time()
            # Update PID once running
            w = self._workers.get(task_id)
            if w and w._proc and info.pid == 0:
                info.pid = w._proc.pid
        self.status_changed.emit(task_id, status)

    def _on_finished(self, task_id: str, exit_code: int) -> None:
        info = self.tasks.get(task_id)
        if info:
            info.exit_code = exit_code
        # Clean up worker reference but keep task info
        self._workers.pop(task_id, None)


# ── Preset process templates ──────────────────────────────────────────────

@dataclass
class ProcessPreset:
    """A launchable process template with editable arguments."""
    name: str
    script: str
    description: str
    args: list[str] = field(default_factory=list)
    arg_defaults: dict[str, str] = field(default_factory=dict)


def get_presets() -> list[ProcessPreset]:
    """Return all available process presets."""
    root = project_root()
    venv_python = str(root / "venv" / "Scripts" / "python.exe")
    if not Path(venv_python).is_file():
        venv_python = "python"

    return [
        ProcessPreset(
            name="Train Expert (Supervised)",
            script="scripts/train_expert.py",
            description="Fine-tune an AirMoE expert on external data (math, science, etc.)",
            arg_defaults={"--topic": "python_algorithms", "--data": "", "--epochs": "3"},
        ),
        ProcessPreset(
            name="Train Expert (Self-Play)",
            script="scripts/train_expert.py",
            description="Self-play mode — model generates + verifies code solutions",
            arg_defaults={"--topic": "python_algorithms", "--mode": "selfplay", "--epochs": "3"},
        ),
        ProcessPreset(
            name="Train DSpark Head",
            script="scripts/train_dspark.py",
            description="Train speculative decoding head (4-token prediction) on ForgeLM V2",
            arg_defaults={"--epochs": "3", "--lr": "1e-4", "--batch-size": "2"},
        ),
        ProcessPreset(
            name="Ablation Benchmark",
            script="scripts/ablation_benchmark.py",
            description="Run ablation benchmark suite across model configurations",
            arg_defaults={},
        ),
        ProcessPreset(
            name="Download HF Datasets",
            script="scripts/download_hf_datasets.py",
            description="Download HuggingFace datasets for training",
            arg_defaults={},
        ),
        ProcessPreset(
            name="Extract Vocab Packs",
            script="scripts/extract_vocab_packs.py",
            description="Extract vocabulary packs from tokenizer",
            arg_defaults={},
        ),
        ProcessPreset(
            name="Inject HF Data",
            script="scripts/inject_hf_data.py",
            description="Inject HuggingFace training data into the pipeline",
            arg_defaults={},
        ),
    ]


def build_command(preset: ProcessPreset, arg_overrides: dict[str, str]) -> list[str]:
    """Build the full command list from a preset + arg overrides."""
    root = project_root()
    venv_python = str(root / "venv" / "Scripts" / "python.exe")
    if not Path(venv_python).is_file():
        venv_python = "python"
    cmd = [venv_python, str(root / preset.script)]
    # Merge defaults with overrides
    merged = {**preset.arg_defaults, **arg_overrides}
    for flag, val in merged.items():
        if val.strip():
            cmd.extend([flag, val.strip()])
    return cmd
