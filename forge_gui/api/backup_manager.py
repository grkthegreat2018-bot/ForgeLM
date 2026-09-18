"""Backup manager — automatic project snapshots for agentic mode.

Features:
- **Change detection**: every 60 seconds, scans the project for modified files
  (by mtime + content hash). If the number of changed files exceeds a threshold,
  a new backup is created.
- **ZIP compression**: all backups are stored as .zip for maximum space efficiency.
- **Naming**: {project_name}_{date}_{time}.zip (e.g. ForgeAI_20260902_143022.zip)
- **Load backup**: agent can call `load_backup` tool → user confirmation dialog
  → if confirmed, wipes current project and restores the unzipped backup.
- **Agent freeze**: while a confirmation dialog is open, the agent loop is
  fully frozen (cannot execute any tools or generate) — prevents bypass.

Backups are stored in `data/backups/` relative to the project root.
Only triggers in agentic mode (not chat mode).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ._signal import SimpleSignal

logger = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────────

CHECK_INTERVAL_S = 60       # check every 60 seconds
MIN_CHANGED_FILES = 3       # need at least 3 changed files to trigger backup
MAX_BACKUPS = 20            # keep at most 20 backups (oldest auto-deleted)
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".pytest_cache", ".ruff_cache", "data", ".devin",
             "research/checkpoints", "data/backups"}
SKIP_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
             ".safetensors", ".pt", ".bin", ".gguf", ".zip"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB — skip larger files in hash check


# ── backup manager ──────────────────────────────────────────────────────

class BackupManager:
    """Manages automatic project backups during agentic mode.

    Signals (SimpleSignal):
        backup_created(path): emitted when a new backup ZIP is saved.
        backup_loaded(path): emitted when a backup is restored.
        agent_freeze(): emitted when a confirmation prompt opens —
            the agent loop MUST stop all execution until agent_unfreeze.
        agent_unfreeze(): emitted when the prompt closes.
    """

    def __init__(self, project_root: Path,
                 confirm_restore: Callable[[str], bool] | None = None) -> None:
        self.backup_created = SimpleSignal()
        self.backup_loaded = SimpleSignal()
        self.agent_freeze = SimpleSignal()
        self.agent_unfreeze = SimpleSignal()
        self.project_root = Path(project_root).resolve()
        # approval callback: (backup_path) -> True to proceed.
        # Default denies — callers must wire a real approval channel.
        self.confirm_restore = confirm_restore or (lambda _p: False)
        self.backup_dir = self.project_root / "data" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._timer: threading.Timer | None = None
        self._file_hashes: dict[str, str] = {}
        self._active = False
        self._frozen = False
        self._project_name = self.project_root.name
        # NOTE: initial snapshot deferred to start() — hashing every file in
        # the project at construction time blocked GUI startup for several
        # seconds. start() (agentic mode only) calls _snapshot_hashes().

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def start(self) -> None:
        """Start automatic backup monitoring (agentic mode only)."""
        self._active = True
        self._snapshot_hashes()
        self._schedule_next()
        logger.info("BackupManager started for project: %s", self._project_name)

    def stop(self) -> None:
        """Stop automatic backup monitoring."""
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("BackupManager stopped")

    def _schedule_next(self) -> None:
        t = threading.Timer(CHECK_INTERVAL_S, self._tick)
        t.daemon = True
        t.start()
        self._timer = t

    def _tick(self) -> None:
        self._check_and_backup()
        if self._active:
            self._schedule_next()

    # ── change detection ──────────────────────────────────────────────
    def _snapshot_hashes(self) -> None:
        """Take a snapshot of all file hashes for change detection."""
        self._file_hashes = {}
        for path in self._walk_files():
            try:
                self._file_hashes[str(path)] = self._hash_file(path)
            except OSError:
                continue

    def _walk_files(self) -> list[Path]:
        """Walk project files, skipping dirs and large/binary files."""
        results = []
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                ext = p.suffix.lower()
                if ext in SKIP_EXTS:
                    continue
                try:
                    if p.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                results.append(p)
        return results

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Fast file hash (first 4KB + size, not full file)."""
        size = path.stat().st_size
        h = hashlib.md5()
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(4096))
        return h.hexdigest()

    def _count_changes(self) -> int:
        """Count how many files changed since last snapshot."""
        current = {}
        for path in self._walk_files():
            try:
                current[str(path)] = self._hash_file(path)
            except OSError:
                continue
        changes = 0
        for k, v in current.items():
            if self._file_hashes.get(k) != v:
                changes += 1
        # also count deleted files
        deleted = len(set(self._file_hashes) - set(current))
        changes += deleted
        self._file_hashes = current
        return changes

    def _check_and_backup(self) -> None:
        """Timer callback: check for changes and create backup if needed."""
        if not self._active or self._frozen:
            return
        try:
            changes = self._count_changes()
            if changes >= MIN_CHANGED_FILES:
                logger.info("BackupManager: %d files changed, creating backup", changes)
                self.create_backup()
        except Exception as e:
            logger.warning("BackupManager check failed: %s", e)

    # ── backup creation ───────────────────────────────────────────────
    def create_backup(self) -> str | None:
        """Create a ZIP backup of the project. Returns the backup path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self._project_name}_{timestamp}.zip"
        zip_path = self.backup_dir / name
        try:
            self._zip_project(zip_path)
            self._prune_old_backups()
            logger.info("Backup created: %s", zip_path)
            self.backup_created.emit(str(zip_path))
            return str(zip_path)
        except Exception as e:
            logger.error("Backup creation failed: %s", e)
            return None

    def _zip_project(self, zip_path: Path) -> None:
        """Zip the project directory into zip_path."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for path in self._walk_files():
                arcname = path.relative_to(self.project_root)
                try:
                    zf.write(path, arcname)
                except OSError:
                    continue

    def _prune_old_backups(self) -> None:
        """Delete oldest backups if we exceed MAX_BACKUPS."""
        backups = sorted(self.backup_dir.glob("*.zip"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[MAX_BACKUPS:]:
            try:
                old.unlink()
                logger.info("Pruned old backup: %s", old.name)
            except OSError:
                pass

    # ── backup listing ────────────────────────────────────────────────
    def list_backups(self) -> list[dict]:
        """List all available backups, newest first."""
        backups = []
        for p in sorted(self.backup_dir.glob("*.zip"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            st = p.stat()
            backups.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(st.st_size / (1024 * 1024), 1),
                "date": datetime.fromtimestamp(st.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"),
            })
        return backups

    # ── backup restore (with user confirmation) ───────────────────────
    def request_restore(self, backup_path: str) -> bool:
        """Request to restore a backup via the ``confirm_restore`` callback.

        FREEZES the agent during the confirmation. If confirmed:
        1. Wipes current project files
        2. Unzips the backup into the project
        3. Returns True

        If declined, returns False.
        The agent is unfrozen before returning.
        """
        self._frozen = True
        self.agent_freeze.emit()
        try:
            if not self.confirm_restore(backup_path):
                logger.info("Backup restore declined: %s", backup_path)
                return False

            # perform restore
            logger.info("Restoring backup: %s", backup_path)
            self._restore_backup(backup_path)
            self.backup_loaded.emit(backup_path)
            return True
        finally:
            self._frozen = False
            self.agent_unfreeze.emit()

    def _restore_backup(self, zip_path: str) -> None:
        """Wipe project and restore from ZIP."""
        zip_p = Path(zip_path)
        if not zip_p.is_file():
            raise FileNotFoundError(f"backup not found: {zip_path}")

        # wipe current project files (skip backup dir, .git, venv)
        wipe_skip = SKIP_DIRS | {"data/backups"}
        for dirpath, dirnames, filenames in os.walk(self.project_root, topdown=False):
            dirnames[:] = [d for d in dirnames if d not in wipe_skip]
            for fn in filenames:
                p = Path(dirpath) / fn
                ext = p.suffix.lower()
                if ext in SKIP_EXTS:
                    continue
                try:
                    p.unlink()
                except OSError:
                    continue
            # remove empty dirs (but not the root or backup dir)
            if Path(dirpath) != self.project_root:
                try:
                    if not any(Path(dirpath).iterdir()):
                        Path(dirpath).rmdir()
                except OSError:
                    continue

        # extract backup
        with zipfile.ZipFile(zip_p, "r") as zf:
            zf.extractall(self.project_root)
        logger.info("Backup restored: %s", zip_path)


# ── backup tool definitions (for the tool harness) ─────────────────────

def backup_tool_defs() -> list[dict]:
    """Tool definitions for backup operations."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_backups",
                "description": (
                    "List all available project backups with dates and sizes. "
                    "Use to show the user what backups are available before "
                    "calling load_backup."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_backup",
                "description": (
                    "Manually trigger a project backup. Creates a ZIP snapshot "
                    "of the current project state. Use before risky operations."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_backup",
                "description": (
                    "Restore a project backup. REQUIRES USER CONFIRMATION — "
                    "a dialog will appear and the agent will be FROZEN until "
                    "the user responds. If confirmed, the current project is "
                    "wiped and replaced with the backup. Cannot be undone."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "backup_path": {
                            "type": "string",
                            "description": (
                                "Path to the backup ZIP file (from list_backups)"),
                        },
                    },
                    "required": ["backup_path"],
                },
            },
        },
    ]
