"""Tasks page — unified live feed of all running tasks with per-task detail.

Shows a combined real-time stdout/stderr feed from all GUI-launched processes,
plus a task selector to view individual task output. Also picks up
status.json-based runs discovered by StatusReader.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..api.process_manager import ProcessManager
from ..api.status_reader import StatusReader
from ..widgets.log_view import LogView
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label

logger = logging.getLogger(__name__)


class TasksPage(QWidget):
    def __init__(self, proc_mgr: ProcessManager,
                 status_reader: StatusReader,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mgr = proc_mgr
        self._status = status_reader
        self._selected_task: str | None = None
        self._all_feed: list[str] = []  # unified feed buffer

        # ---- summary cards ----
        self._c_live = MetricCard("Live processes", "0")
        self._c_runs = MetricCard("Training runs", "0")
        self._c_lines = MetricCard("Feed lines", "0")
        self._c_tasks_total = MetricCard("Tasks tracked", "0")
        cards = card_grid([self._c_live, self._c_runs, self._c_lines,
                           self._c_tasks_total], cols=4)

        # ---- task selector ----
        sel_row = QHBoxLayout(); sel_row.setSpacing(10)
        sel_row.addWidget(section_label("VIEW"))
        self._task_combo = QComboBox(); self._task_combo.setMinimumWidth(360)
        self._task_combo.addItem("● ALL (unified feed)", "__all__")
        self._task_combo.currentIndexChanged.connect(self._on_select)
        sel_row.addWidget(self._task_combo, 1)
        sel_row.addStretch(1)
        self._clear_btn = QPushButton("Clear feed")
        self._clear_btn.clicked.connect(self._clear_feed)
        sel_row.addWidget(self._clear_btn)
        sel_host = QWidget(); sel_host.setLayout(sel_row)

        # ---- live feed ----
        self._feed = LogView()

        # ---- task detail panel ----
        detail_card = QFrame(); detail_card.setObjectName("card")
        dl = QVBoxLayout(detail_card); dl.setContentsMargins(16, 14, 16, 14)
        dl.setSpacing(8)
        dl.addWidget(section_label("TASK DETAILS"))
        self._detail_name = QLabel("Select a task to view details.")
        self._detail_name.setObjectName("cardBody")
        self._detail_name.setWordWrap(True)
        dl.addWidget(self._detail_name)

        self._detail_cmd = QLabel("")
        self._detail_cmd.setObjectName("mono")
        self._detail_cmd.setWordWrap(True)
        self._detail_cmd.setStyleSheet(
            "color:#8b96a8; font-size:11px; font-family:'Cascadia Code',monospace;"
        )
        dl.addWidget(self._detail_cmd)

        self._detail_stats = QLabel("")
        self._detail_stats.setObjectName("cardBody")
        dl.addWidget(self._detail_stats)

        # ---- splitter: feed + detail ----
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._feed)
        splitter.addWidget(detail_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._host = page_container(cards, sel_host, splitter)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._host)

        # Connect manager signals
        self._mgr.line.connect(self._on_line)
        self._mgr.task_added.connect(self._on_task_added)
        self._mgr.task_removed.connect(self._on_task_removed)
        self._mgr.status_changed.connect(self._on_status_changed)

    def _on_line(self, task_id: str, line: str) -> None:
        info = self._mgr.tasks.get(task_id)
        prefix = info.name if info else task_id
        entry = f"[{prefix}] {line}"
        self._all_feed.append(entry)
        if len(self._all_feed) > 5000:
            self._all_feed = self._all_feed[-4000:]
        # Only re-render if viewing all or this specific task
        if self._selected_task in (None, "__all__", task_id):
            self._feed.append_line(entry)

    def _on_task_added(self, task_id: str) -> None:
        self._rebuild_combo()

    def _on_task_removed(self, task_id: str) -> None:
        if self._selected_task == task_id:
            self._task_combo.setCurrentIndex(0)
        self._rebuild_combo()

    def _on_status_changed(self, task_id: str, status: str) -> None:
        self._rebuild_combo()
        self._update_detail()

    def _rebuild_combo(self) -> None:
        cur = self._task_combo.currentData()
        self._task_combo.blockSignals(True)
        self._task_combo.clear()
        self._task_combo.addItem("● ALL (unified feed)", "__all__")
        for info in self._mgr.all_tasks():
            label = f"{'▶' if info.is_live else '■'} {info.name} ({info.status})"
            self._task_combo.addItem(label, info.id)
        # Restore selection
        if cur:
            idx = self._task_combo.findData(cur)
            if idx >= 0:
                self._task_combo.setCurrentIndex(idx)
        self._task_combo.blockSignals(False)

    def _on_select(self, idx: int) -> None:
        data = self._task_combo.currentData()
        self._selected_task = data
        self._render_feed()
        self._update_detail()

    def _render_feed(self) -> None:
        if self._selected_task is None or self._selected_task == "__all__":
            self._feed.replace_all(self._all_feed)
        else:
            info = self._mgr.tasks.get(self._selected_task)
            if info:
                prefix = info.name
                lines = [f"[{prefix}] {ln}" for ln in info.lines]
                self._feed.replace_all(lines)
            else:
                self._feed.replace_all([])

    def _update_detail(self) -> None:
        if self._selected_task is None or self._selected_task == "__all__":
            self._detail_name.setText("Select a specific task above to view details.")
            self._detail_cmd.setText("")
            self._detail_stats.setText("")
            return
        info = self._mgr.tasks.get(self._selected_task)
        if not info:
            self._detail_name.setText("Task not found.")
            return
        self._detail_name.setText(
            f"{info.name}  —  status: {info.status}\n"
            f"PID: {info.pid or '—'}  ·  elapsed: {info.elapsed_s:.0f}s"
        )
        self._detail_cmd.setText(" ".join(info.command))
        if info.exit_code is not None:
            self._detail_stats.setText(f"Exit code: {info.exit_code}")
        else:
            self._detail_stats.setText(f"Log lines captured: {len(info.lines)}")

    def _clear_feed(self) -> None:
        self._all_feed.clear()
        self._feed.replace_all([])

    def refresh(self) -> None:
        # Update summary cards
        tasks = self._mgr.all_tasks()
        live = sum(1 for t in tasks if t.is_live)
        runs = self._status.snapshot()
        active_runs = sum(1 for r in runs if r.is_live)
        self._c_live.set_value(str(live))
        self._c_runs.set_value(str(active_runs))
        self._c_lines.set_value(str(len(self._all_feed)))
        self._c_tasks_total.set_value(str(len(tasks)))
        self._rebuild_combo()
        self._update_detail()
