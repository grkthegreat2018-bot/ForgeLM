"""Incremental events.jsonl tailer for the Self-Play live page.

Tails events.jsonl files under research/checkpoints/** and provides:
  - poll(): new events since last call (oldest-first)
  - all_events(): bounded rolling buffer (deque maxlen ~2000)
  - latest_status(): cheap status.json read with mtime cache

Robust to truncation/rotation (resets offset when file shrinks) and partial
last lines (keeps trailing partial bytes until a newline arrives). Designed
to be called every ~500ms from the UI thread without blocking.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .status_reader import project_root

logger = logging.getLogger(__name__)


class _EventSource:
    __slots__ = ("path", "offset", "partial", "mtime")

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.partial = b""
        self.mtime = 0.0


class EventsReader:
    """Tails events.jsonl + reads status.json for the Self-Play page."""

    def __init__(self, search_dirs: Optional[list[Path]] = None,
                 max_events: int = 2000,
                 rescan_interval: float = 5.0) -> None:
        self._search_dirs = search_dirs or [project_root() / "research" / "checkpoints"]
        self._sources: dict[str, _EventSource] = {}
        self._buffer: deque[dict] = deque(maxlen=max_events)
        self._last_rescan = 0.0
        self._rescan_interval = rescan_interval
        # status.json cache
        self._status_path: Optional[Path] = None
        self._status_mtime = 0.0
        self._status_cache: Optional[dict] = None

    def _discover(self) -> None:
        """Find events.jsonl files under search dirs (called at most every rescan_interval).

        On first discovery of a file, back-fills the last ~200 events so the
        GUI shows recent history immediately instead of a blank feed.
        """
        for d in self._search_dirs:
            if not d.is_dir():
                continue
            for p in d.rglob("events.jsonl"):
                key = str(p)
                if key not in self._sources:
                    try:
                        st = p.stat()
                        src = _EventSource(p)
                        # Back-fill: read last ~50KB of the file to populate
                        # recent history (roughly 200-500 events at ~200B each).
                        backfill = min(st.st_size, 50_000)
                        src.offset = st.st_size - backfill
                        src.mtime = st.st_mtime
                        self._sources[key] = src
                    except OSError:
                        pass

    def poll(self) -> list[dict]:
        """Read new events from all sources. Returns oldest-first list."""
        now = time.time()
        if now - self._last_rescan > self._rescan_interval:
            self._last_rescan = now
            self._discover()

        new_events: list[dict] = []
        for src in list(self._sources.values()):
            try:
                st = src.path.stat()
            except OSError:
                continue
            size = st.st_size
            if size < src.offset:
                # Truncated / rotated — restart from 0
                src.offset = 0
                src.partial = b""
            if size == src.offset:
                continue
            try:
                with open(src.path, "rb") as f:
                    f.seek(src.offset)
                    chunk = f.read(size - src.offset)
                src.offset = size
            except OSError as e:
                logger.debug("events read failed %s: %s", src.path, e)
                continue

            data = src.partial + chunk
            lines = data.split(b"\n")
            src.partial = lines.pop()  # last element may be partial (no trailing \n)
            for raw in lines:
                if not raw.strip():
                    continue
                try:
                    ev = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self._buffer.append(ev)
                new_events.append(ev)

        return new_events

    def all_events(self) -> list[dict]:
        """Return the bounded rolling buffer (oldest-first, up to maxlen)."""
        return list(self._buffer)

    def latest_status(self) -> Optional[dict]:
        """Cheap read of the newest status.json under checkpoints/self_play/."""
        # Find the status file (prefer self_play/ subdir)
        candidates: list[Path] = []
        for d in self._search_dirs:
            if not d.is_dir():
                continue
            sp = d / "self_play" / "status.json"
            if sp.is_file():
                candidates.append(sp)
            else:
                for p in d.rglob("status.json"):
                    if "self_play" in p.parent.name.lower():
                        candidates.append(p)

        if not candidates:
            self._status_path = None
            self._status_cache = None
            return None

        # Pick the most recently modified
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            mtime = best.stat().st_mtime
        except OSError:
            return self._status_cache

        if best == self._status_path and mtime == self._status_mtime and self._status_cache is not None:
            return self._status_cache

        try:
            with open(best, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._status_path = best
                self._status_mtime = mtime
                self._status_cache = data
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("status read failed %s: %s", best, e)

        return self._status_cache

    def heartbeat_age(self) -> Optional[float]:
        """Seconds since the last heartbeat.json write (None if no heartbeat file)."""
        if self._status_path is None:
            self.latest_status()
        if self._status_path is None:
            return None
        hb = self._status_path.with_name("heartbeat.json")
        if not hb.is_file():
            return None
        try:
            with open(hb, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = float(data.get("ts", 0))
            return time.time() - ts
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    def heartbeat_stalled(self) -> Optional[bool]:
        """Check if the training loop has stalled (progress-coupled heartbeat).

        Returns True if the heartbeat writer detected no training-loop
        progress within its stall threshold (the heartbeat.json contains
        a "stalled": true flag). Returns False if the heartbeat is fresh
        and progress is being made. Returns None if no heartbeat file
        exists or it can't be read.
        """
        if self._status_path is None:
            self.latest_status()
        if self._status_path is None:
            return None
        hb = self._status_path.with_name("heartbeat.json")
        if not hb.is_file():
            return None
        try:
            with open(hb, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("stalled", False))
        except (json.JSONDecodeError, OSError, ValueError):
            return None
