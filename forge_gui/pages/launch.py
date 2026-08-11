"""Launch page — boot processes via GUI with preset scripts + custom commands.

Shows preset process templates (train_expert, train_dspark, etc.) with
editable arguments, a custom command input, and a list of running/finished
tasks with kill buttons. Process output is shown live in the Tasks page.
"""
from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..api.process_manager import ProcessManager, build_command, get_presets
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label, status_tag

logger = logging.getLogger(__name__)


class LaunchPage(QWidget):
    def __init__(self, proc_mgr: ProcessManager,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mgr = proc_mgr
        self._presets = get_presets()
        self._arg_edits: dict[str, dict[str, QLineEdit]] = {}  # preset_name -> {flag: edit}
        self._task_rows: dict[str, QWidget] = {}

        # ---- summary cards ----
        self._c_running = MetricCard("Running", "0")
        self._c_done = MetricCard("Completed", "0")
        self._c_crashed = MetricCard("Failed", "0")
        self._c_total = MetricCard("Total launched", "0")
        cards = card_grid([self._c_running, self._c_done, self._c_crashed,
                           self._c_total], cols=4)

        # ---- preset selector ----
        preset_card = QFrame(); preset_card.setObjectName("card")
        pl = QVBoxLayout(preset_card); pl.setContentsMargins(16, 14, 16, 16)
        pl.setSpacing(10)
        pl.addWidget(section_label("PRESET SCRIPTS"))

        sel_row = QHBoxLayout(); sel_row.setSpacing(10)
        sel_row.addWidget(QLabel("Script"))
        self._preset_combo = QComboBox(); self._preset_combo.setMinimumWidth(320)
        for p in self._presets:
            self._preset_combo.addItem(p.name, p.name)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        sel_row.addWidget(self._preset_combo, 1)
        pl.addLayout(sel_row)

        self._preset_desc = QLabel("")
        self._preset_desc.setObjectName("cardBody")
        self._preset_desc.setWordWrap(True)
        pl.addWidget(self._preset_desc)

        # Arg editor (rebuilt on preset change)
        self._arg_host = QVBoxLayout(); self._arg_host.setSpacing(8)
        pl.addLayout(self._arg_host)

        self._launch_preset_btn = QPushButton("Launch")
        self._launch_preset_btn.setObjectName("primary")
        self._launch_preset_btn.clicked.connect(self._launch_preset)
        pl.addWidget(self._launch_preset_btn)

        # ---- custom command ----
        custom_card = QFrame(); custom_card.setObjectName("card")
        cl = QVBoxLayout(custom_card); cl.setContentsMargins(16, 14, 16, 16)
        cl.setSpacing(10)
        cl.addWidget(section_label("CUSTOM COMMAND"))

        cl.addWidget(QLabel("Enter a shell command (runs in project root):"))
        self._custom_cmd = QLineEdit()
        self._custom_cmd.setPlaceholderText(
            "e.g.  venv\\Scripts\\python.exe scripts/train_expert.py --topic math --data data.json"
        )
        cl.addWidget(self._custom_cmd)

        self._launch_custom_btn = QPushButton("Run command")
        self._launch_custom_btn.setObjectName("primary")
        self._launch_custom_btn.clicked.connect(self._launch_custom)
        cl.addWidget(self._launch_custom_btn)

        # ---- running tasks ----
        tasks_card = QFrame(); tasks_card.setObjectName("card")
        tl = QVBoxLayout(tasks_card); tl.setContentsMargins(16, 14, 16, 16)
        tl.setSpacing(10)
        thead = QHBoxLayout()
        thead.addWidget(section_label("LAUNCHED TASKS"))
        thead.addStretch(1)
        self._clear_finished_btn = QPushButton("Clear finished")
        self._clear_finished_btn.clicked.connect(self._clear_finished)
        thead.addWidget(self._clear_finished_btn)
        tl.addLayout(thead)

        self._tasks_host = QVBoxLayout(); self._tasks_host.setSpacing(6)
        tl.addLayout(self._tasks_host)
        self._tasks_empty = QLabel("No tasks launched yet.")
        self._tasks_empty.setObjectName("cardEmpty")
        self._tasks_host.addWidget(self._tasks_empty)

        self._host = page_container(cards, preset_card, custom_card, tasks_card)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._host)

        # Connect manager signals
        self._mgr.task_added.connect(self._on_task_added)
        self._mgr.task_removed.connect(self._on_task_removed)
        self._mgr.status_changed.connect(self._on_task_status)

        self._on_preset_changed(0)

    def _on_preset_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._presets):
            return
        preset = self._presets[idx]
        self._preset_desc.setText(preset.description)

        # Clear old arg edits
        while self._arg_host.count():
            item = self._arg_host.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._arg_edits[preset.name] = {}
        if not preset.arg_defaults:
            lbl = QLabel("No configurable arguments for this script.")
            lbl.setObjectName("cardEmpty")
            self._arg_host.addWidget(lbl)
            return

        for flag, default in preset.arg_defaults.items():
            row = QHBoxLayout(); row.setSpacing(10)
            lbl = QLabel(flag); lbl.setMinimumWidth(100)
            lbl.setStyleSheet("color:#8b96a8; font-family:'Cascadia Code',monospace; font-size:12px;")
            row.addWidget(lbl)
            edit = QLineEdit(default)
            edit.setMinimumWidth(200)
            edit.setStyleSheet("font-family:'Cascadia Code',monospace; font-size:12px;")
            row.addWidget(edit, 1)
            host = QWidget(); host.setLayout(row)
            self._arg_host.addWidget(host)
            self._arg_edits[preset.name][flag] = edit

    def _launch_preset(self) -> None:
        idx = self._preset_combo.currentIndex()
        if idx < 0 or idx >= len(self._presets):
            return
        preset = self._presets[idx]
        overrides = {}
        edits = self._arg_edits.get(preset.name, {})
        for flag, edit in edits.items():
            overrides[flag] = edit.text().strip()
        cmd = build_command(preset, overrides)
        self._mgr.launch(preset.name, cmd)

    def _launch_custom(self) -> None:
        cmd_text = self._custom_cmd.text().strip()
        if not cmd_text:
            return
        # Split respecting quotes
        import shlex
        try:
            parts = shlex.split(cmd_text, posix=True)
        except ValueError as e:
            QMessageBox.warning(self, "Parse error", f"Could not parse command:\n{e}")
            return
        if not parts:
            return
        name = parts[-1].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        self._mgr.launch(f"custom: {name}", parts)

    def _on_task_added(self, task_id: str) -> None:
        if self._tasks_empty is not None:
            try:
                self._tasks_empty.setVisible(False)
            except RuntimeError:
                # C++ object already deleted by _rebuild_task_list
                self._tasks_empty = None
        self._rebuild_task_list()

    def _on_task_removed(self, task_id: str) -> None:
        self._rebuild_task_list()

    def _on_task_status(self, task_id: str, status: str) -> None:
        self._rebuild_task_list()

    def _clear_finished(self) -> None:
        for tid in list(self._mgr.tasks.keys()):
            info = self._mgr.tasks[tid]
            if not info.is_live:
                self._mgr.remove(tid)
        self._rebuild_task_list()

    def _rebuild_task_list(self) -> None:
        # Clear
        self._tasks_empty = None  # invalidate ref before deleting widgets
        while self._tasks_host.count():
            item = self._tasks_host.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        tasks = self._mgr.all_tasks()
        if not tasks:
            self._tasks_empty = QLabel("No tasks launched yet.")
            self._tasks_empty.setObjectName("cardEmpty")
            self._tasks_host.addWidget(self._tasks_empty)
            self._update_summary()
            return

        for info in tasks:
            row = self._build_task_row(info)
            self._tasks_host.addWidget(row)
        self._update_summary()

    def _build_task_row(self, info) -> QWidget:
        host = QFrame(); host.setObjectName("cardAlt")
        h = QHBoxLayout(host); h.setContentsMargins(14, 10, 14, 10); h.setSpacing(12)

        kind = "ok" if info.is_live else ("warn" if info.status == "done" else "err")
        tag = status_tag(info.status, kind)
        h.addWidget(tag)

        name = QLabel(info.name)
        name.setStyleSheet("color:#e6edf3; font-weight:600; font-size:13px;")
        h.addWidget(name)

        h.addStretch(1)

        elapsed = f"{info.elapsed_s:.0f}s"
        if info.pid:
            elapsed = f"PID {info.pid} · {elapsed}"
        meta = QLabel(elapsed)
        meta.setStyleSheet("color:#8b96a8; font-size:12px;")
        h.addWidget(meta)

        if info.is_live:
            kill_btn = QPushButton("Kill")
            kill_btn.setObjectName("danger")
            kill_btn.setFixedWidth(70)
            kill_btn.clicked.connect(lambda checked=False, tid=info.id: self._mgr.kill(tid))
            h.addWidget(kill_btn)
        else:
            rm_btn = QPushButton("×")
            rm_btn.setFixedWidth(30)
            rm_btn.clicked.connect(lambda checked=False, tid=info.id: self._mgr.remove(tid))
            h.addWidget(rm_btn)

        return host

    def _update_summary(self) -> None:
        tasks = self._mgr.all_tasks()
        running = sum(1 for t in tasks if t.is_live)
        done = sum(1 for t in tasks if t.status == "done")
        crashed = sum(1 for t in tasks if t.status in ("crashed", "killed"))
        self._c_running.set_value(str(running))
        self._c_done.set_value(str(done))
        self._c_crashed.set_value(str(crashed))
        self._c_total.set_value(str(len(tasks)))

    def refresh(self) -> None:
        self._update_summary()
