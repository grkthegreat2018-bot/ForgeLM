"""Library install manager — prompts user to approve pip installs.

When the agent's script needs a new Python library:
1. Check if it's in the auto-accept allowlist → install silently
2. If not, prompt the user with a dialog:
   - Show library name + description
   - "Install" / "Deny" buttons
   - "Save as allowed" checkbox → adds to allowlist for future projects
3. If approved, pip install into the project venv
4. If denied, return error to the agent

The allowlist is persisted to `data/library_allowlist.json` and shared
across all projects. Common safe libraries are pre-populated.

The agent is FROZEN during the approval dialog (same pattern as backup
restore) to prevent bypass attempts.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .status_reader import project_root

logger = logging.getLogger(__name__)

# Pre-populated auto-accept list (common, safe, well-known libraries)
_DEFAULT_ALLOWLIST = [
    "numpy", "scipy", "pandas", "matplotlib", "seaborn",
    "requests", "httpx", "aiohttp",
    "pytest", "pytest-asyncio", "pytest-cov",
    "pydantic", "pydantic-core",
    "rich", "click", "typer",
    "tqdm", "colorama",
    "PyYAML", "toml", "tomli",
    "Pillow", "opencv-python",
    "scikit-learn", "scikit-image",
    "beautifulsoup4", "lxml",
    "python-dateutil", "pytz",
    "tabulate", "texttable",
    "jieba", "nltk",
    "transformers", "tokenizers", "datasets",
    "accelerate", "safetensors",
    "psutil", "GPUtil",
]


class LibraryInstallManager(QObject):
    """Manages library install approvals and the allowlist.

    Signals:
        install_started(package): pip install started
        install_completed(package, success): pip install finished
        agent_freeze(): approval dialog opened — agent must stop
        agent_unfreeze(): dialog closed — agent can resume
        allowlist_updated(package): package added to allowlist
    """

    install_started = Signal(str)
    install_completed = Signal(str, bool)
    agent_freeze = Signal()
    agent_unfreeze = Signal()
    allowlist_updated = Signal(str)

    def __init__(self, venv_python: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._venv_python = venv_python or sys.executable
        self._frozen = False
        self._allowlist_path = project_root() / "data" / "library_allowlist.json"
        self._allowlist: set[str] = set()
        self._load_allowlist()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def venv_python(self) -> str:
        return self._venv_python

    @venv_python.setter
    def venv_python(self, path: str) -> None:
        self._venv_python = path

    # ── allowlist persistence ─────────────────────────────────────────
    def _load_allowlist(self) -> None:
        """Load the allowlist from disk, merging with defaults."""
        loaded = set(_DEFAULT_ALLOWLIST)
        if self._allowlist_path.is_file():
            try:
                data = json.loads(self._allowlist_path.read_text(encoding="utf-8"))
                loaded.update(data.get("allowed", []))
            except (json.JSONDecodeError, OSError):
                pass
        self._allowlist = loaded

    def _save_allowlist(self) -> None:
        """Save the allowlist to disk."""
        self._allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"allowed": sorted(self._allowlist)}
        self._allowlist_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8")

    def get_allowlist(self) -> list[str]:
        """Get the current allowlist (sorted)."""
        return sorted(self._allowlist)

    def add_to_allowlist(self, package: str) -> None:
        """Add a package to the allowlist and save."""
        self._allowlist.add(package)
        self._save_allowlist()
        self.allowlist_updated.emit(package)
        logger.info("Added to allowlist: %s", package)

    def is_allowed(self, package: str) -> bool:
        """Check if a package is in the auto-accept allowlist."""
        # normalize: strip version specifiers
        base = package.split("=")[0].split(">")[0].split("<")[0].split("[")[0].strip()
        return base in self._allowlist

    # ── install flow ──────────────────────────────────────────────────
    def request_install(self, package: str,
                        parent: Optional[QWidget] = None) -> dict:
        """Request to install a package. Shows dialog if not in allowlist.

        Returns dict with:
            {"installed": True/False, "package": ..., "auto_approved": bool}
            or {"error": "..."} if denied or failed.
        """
        # check allowlist first
        if self.is_allowed(package):
            return self._do_install(package, auto_approved=True)

        # need user approval — freeze agent
        self._frozen = True
        self.agent_freeze.emit()
        try:
            approved, save_allowed = self._show_approval_dialog(package, parent)
            if not approved:
                return {"error": f"install denied by user: {package}",
                        "denied": True}
            if save_allowed:
                self.add_to_allowlist(package)
            return self._do_install(package, auto_approved=False)
        finally:
            self._frozen = False
            self.agent_unfreeze.emit()

    def _show_approval_dialog(self, package: str,
                              parent: Optional[QWidget]) -> tuple[bool, bool]:
        """Show the approval dialog. Returns (approved, save_to_allowlist)."""
        dialog = QDialog(parent or self._parent_widget)
        dialog.setWindowTitle("Library Install Request")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        # message
        msg = QLabel(
            f"<b>The agent wants to install a new Python library:</b><br><br>"
            f"<code>pip install {package}</code><br><br>"
            f"This will be installed into the project venv.<br>"
            f"If you trust this library, click Install.")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        # save checkbox
        save_cb = QCheckBox(
            "Save as allowed (auto-approve for future projects)")
        layout.addWidget(save_cb)

        # buttons
        btn_row = QHBoxLayout()
        btn_install = QPushButton("Install")
        btn_deny = QPushButton("Deny")
        btn_install.setObjectName("rateGood")
        btn_deny.setObjectName("rateBad")
        btn_row.addStretch(1)
        btn_row.addWidget(btn_deny)
        btn_row.addWidget(btn_install)
        layout.addLayout(btn_row)

        btn_install.clicked.connect(lambda: dialog.accept())
        btn_deny.clicked.connect(lambda: dialog.reject())

        result = dialog.exec()
        approved = (result == QDialog.DialogCode.Accepted)
        return approved, save_cb.isChecked()

    def _do_install(self, package: str, auto_approved: bool) -> dict:
        """Run pip install for the package."""
        self.install_started.emit(package)
        try:
            cmd = [self._venv_python, "-m", "pip", "install", package]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            success = proc.returncode == 0
            self.install_completed.emit(package, success)
            if success:
                return {"installed": True, "package": package,
                        "auto_approved": auto_approved,
                        "stdout": proc.stdout[-2000:]}
            else:
                return {"error": f"pip install failed (exit {proc.returncode})",
                        "package": package,
                        "stderr": proc.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            self.install_completed.emit(package, False)
            return {"error": "pip install timeout (300s)", "package": package}
        except FileNotFoundError:
            self.install_completed.emit(package, False)
            return {"error": "pip not found", "package": package}

    def install_multiple(self, packages: list[str]) -> list[dict]:
        """Install multiple packages. Each goes through the approval flow."""
        results = []
        for pkg in packages:
            results.append(self.request_install(pkg))
        return results


# ── tool definitions ────────────────────────────────────────────────────

def library_tool_defs() -> list[dict]:
    """Tool definitions for library installation."""
    return [
        {
            "type": "function",
            "function": {
                "name": "install_library",
                "description": (
                    "Install a Python library into the project venv. "
                    "If the library is in the auto-accept allowlist, it "
                    "installs silently. Otherwise, the user is prompted "
                    "for approval. The agent is frozen during approval."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "package": {
                            "type": "string",
                            "description": (
                                "Package name (optionally with version, "
                                "e.g. 'requests' or 'requests>=2.28')."),
                        },
                    },
                    "required": ["package"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_library",
                "description": (
                    "Check if a Python library is installed in the venv. "
                    "Returns the version if installed, or not_installed."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "package": {
                            "type": "string",
                            "description": "Package name to check.",
                        },
                    },
                    "required": ["package"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_allowed_libraries",
                "description": (
                    "List all libraries in the auto-accept allowlist. "
                    "These install without user prompting."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
