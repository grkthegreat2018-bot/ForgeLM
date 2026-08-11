"""Tails log files for the Logs page.

Watches the project logs/ directory plus per-run log.txt files. Keeps a
rolling buffer and supports level/source filtering and substring search.
"""
from __future__ import annotations

import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .status_reader import project_root

logger = logging.getLogger(__name__)

LEVEL_RE = re.compile(r"\b(ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE|CRITICAL|FATAL)\b", re.I)


@dataclass
class LogLine:
    source: str
    text: str
    level: str = "INFO"

    def fmt(self) -> str:
        return f"[{self.level:<5}] {self.source}: {self.text}"


@dataclass
class LogSource:
    name: str
    path: Path
    offset: int = 0
    active: bool = True


class LogTailer:
    """Polls multiple log files, appends new lines to a shared ring buffer."""

    def __init__(self, max_lines: int = 4000) -> None:
        self.max_lines = max_lines
        self.buffer: deque[LogLine] = deque(maxlen=max_lines)
        self.sources: dict[str, LogSource] = {}

    def add_source(self, name: str, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            return
        self.sources[name] = LogSource(name=name, path=path, offset=path.stat().st_size)

    def discover(self) -> list[str]:
        """Auto-discover *.log under logs/ and research/checkpoints/**/log.txt."""
        root = project_root()
        found: list[str] = []
        logs_dir = root / "logs"
        if logs_dir.is_dir():
            for p in sorted(logs_dir.glob("*.log")):
                self.add_source(p.stem, p)
                found.append(p.stem)
        ckpt = root / "research" / "checkpoints"
        if ckpt.is_dir():
            for p in sorted(ckpt.rglob("log.txt")):
                name = f"run:{p.parent.name}"
                self.add_source(name, p)
                found.append(name)
        tasks = root / "research" / "tasks"
        if tasks.is_dir():
            for p in sorted(tasks.rglob("log.txt")):
                name = f"task:{p.parent.name}"
                self.add_source(name, p)
                found.append(name)
        return found

    def poll(self) -> list[LogLine]:
        """Read new bytes from every source; return only the freshly appended lines."""
        new_lines: list[LogLine] = []
        for src in list(self.sources.values()):
            try:
                size = src.path.stat().st_size
            except Exception as e:
                logger.warning("log source %s stat failed, deactivating: %s", src.name, e)
                src.active = False
                continue
            if size < src.offset:
                # Truncated / rotated — restart from 0.
                src.offset = 0
            if size == src.offset:
                continue
            try:
                with open(src.path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(src.offset)
                    chunk = f.read(size - src.offset)
                src.offset = size
            except Exception as e:
                logger.warning("log source %s read failed: %s", src.name, e)
                continue
            for raw in chunk.splitlines():
                if not raw.strip():
                    continue
                m = LEVEL_RE.search(raw)
                level = (m.group(0).upper() if m else "INFO")
                line = LogLine(source=src.name, text=raw, level=level)
                self.buffer.append(line)
                new_lines.append(line)
        return new_lines

    def filtered(
        self,
        query: str = "",
        levels: Optional[set[str]] = None,
        sources: Optional[set[str]] = None,
        limit: int = 2000,
    ) -> list[LogLine]:
        q = query.lower()
        out: list[LogLine] = []
        for ln in reversed(self.buffer):
            if levels and ln.level not in levels:
                continue
            if sources and ln.source not in sources:
                continue
            if q and q not in ln.text.lower():
                continue
            out.append(ln)
            if len(out) >= limit:
                break
        out.reverse()
        return out

    def clear(self) -> None:
        self.buffer.clear()
