"""Logs page — multi-source tailed log console with level/source filters + search."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from ..api.log_tailer import LogTailer
from ..theme import Palette
from ..widgets.log_view import LogView
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label


LEVELS = ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"]


class LogsPage(QWidget):
    def __init__(self, tailer: LogTailer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tailer = tailer
        self._levels: set[str] = set(LEVELS)
        self._sources: set[str] = set()
        self._query = ""

        # ---- summary cards ----
        self._c_total = MetricCard("Total lines", "0")
        self._c_err = MetricCard("Errors", "0", spark_color=Palette.err)
        self._c_warn = MetricCard("Warnings", "0", spark_color=Palette.warn)
        self._c_src = MetricCard("Sources", "0")
        cards = card_grid([self._c_total, self._c_err, self._c_warn, self._c_src], cols=4)

        # ---- filter bar ----
        filt = QFrame(); filt.setObjectName("card")
        fl = QVBoxLayout(filt); fl.setContentsMargins(16,14,16,14); fl.setSpacing(10)
        fl.addWidget(section_label("FILTERS"))

        lvl_row = QHBoxLayout(); lvl_row.setSpacing(14)
        lvl_row.addWidget(QLabel("Levels"))
        self._lvl_checks: dict[str, QCheckBox] = {}
        for lv in LEVELS:
            cb = QCheckBox(lv); cb.setChecked(True)
            cb.stateChanged.connect(lambda _=False, x=lv: self._on_level(x))
            self._lvl_checks[lv] = cb
            lvl_row.addWidget(cb)
        lvl_row.addStretch(1)
        fl.addLayout(lvl_row)

        src_row = QHBoxLayout(); src_row.setSpacing(10)
        src_row.addWidget(QLabel("Source"))
        self._src_combo = QComboBox(); self._src_combo.setMinimumWidth(220)
        self._src_combo.addItem("(all)", "")
        src_row.addWidget(self._src_combo)
        src_row.addStretch(1)
        self._search = QLineEdit(); self._search.setPlaceholderText("Search log text…")
        self._search.textChanged.connect(self._on_search)
        src_row.addWidget(self._search)
        self._clear_btn = QPushButton("Clear buffer")
        src_row.addWidget(self._clear_btn)
        fl.addLayout(src_row)

        # ---- log view ----
        self._view = LogView()

        self._host = page_container(cards, filt, self._view)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(self._host)

        self._src_combo.currentIndexChanged.connect(self._on_source)
        self._clear_btn.clicked.connect(self._tailer.clear)

    def _on_level(self, lv: str) -> None:
        cb = self._lvl_checks.get(lv)
        if cb:
            if cb.isChecked():
                self._levels.add(lv)
            else:
                self._levels.discard(lv)
        self._render()

    def _on_source(self, _idx: int) -> None:
        src = self._src_combo.currentData() or ""
        self._sources = {src} if src else set()
        self._render()

    def _on_search(self, text: str) -> None:
        self._query = text.strip()
        self._render()

    def _render(self) -> None:
        lines = self._tailer.filtered(
            query=self._query, levels=self._levels,
            sources=self._sources or None, limit=3000,
        )
        self._view.replace_all([ln.fmt() for ln in lines])

    def refresh(self) -> None:
        # discover sources on first refresh
        if self._src_combo.count() <= 1:
            found = self._tailer.discover()
            for s in found:
                self._src_combo.addItem(s, s)
        new = self._tailer.poll()
        # update counts
        total = len(self._tailer.buffer)
        errs = sum(1 for ln in self._tailer.buffer if ln.level in ("ERROR", "FATAL", "CRITICAL"))
        warns = sum(1 for ln in self._tailer.buffer if ln.level in ("WARN", "WARNING"))
        self._c_total.set_value(str(total))
        self._c_err.set_value(str(errs)); self._c_err.push_spark(errs)
        self._c_warn.set_value(str(warns)); self._c_warn.push_spark(warns)
        self._c_src.set_value(str(len(self._tailer.sources)))
        if new:
            self._render()
