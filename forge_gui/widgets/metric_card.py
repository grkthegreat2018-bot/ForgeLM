"""MetricCard — a small panel showing a title, big value, optional delta + sparkline."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
                               QWidget)

from ..theme import Palette
from .chart import LiveLineChart


class MetricCard(QFrame):
    """Compact KPI card: title, big value, unit, delta, optional sparkline."""

    def __init__(self, title: str, value: str = "—", unit: str = "",
                 delta: Optional[str] = None, delta_up: bool = True,
                 spark_color: Optional[str] = None,
                 spark_window: int = 80, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._spark: Optional[LiveLineChart] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("cardTitle")
        lay.addWidget(title_lbl)

        row = QHBoxLayout(); row.setContentsMargins(16, 0, 16, 14); row.setSpacing(8)
        self._value_lbl = QLabel(value)
        self._value_lbl.setObjectName("cardValue")
        row.addWidget(self._value_lbl, 1)
        if unit:
            u = QLabel(unit); u.setObjectName("cardUnit")
            row.addWidget(u, 0, Qt.AlignmentFlag.AlignBottom)
        if delta is not None:
            self._delta_lbl = QLabel(delta)
            self._delta_lbl.setObjectName("cardDeltaUp" if delta_up else "cardDeltaDown")
            row.addWidget(self._delta_lbl, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(row)

        if spark_color:
            self._spark = LiveLineChart(
                series=[("v", spark_color)], window=spark_window,
                height=46, min_y=None, max_y=None,
            )
            self._spark._show_grid = False
            lay.addWidget(self._spark)

    def set_value(self, value: str, unit: str = "",
                  delta: Optional[str] = None, delta_up: bool = True) -> None:
        self._value_lbl.setText(value)
        if delta is not None and hasattr(self, "_delta_lbl"):
            self._delta_lbl.setText(delta)
            self._delta_lbl.setObjectName("cardDeltaUp" if delta_up else "cardDeltaDown")
            self._delta_lbl.style().unpolish(self._delta_lbl)
            self._delta_lbl.style().polish(self._delta_lbl)
        elif delta is None and hasattr(self, "_delta_lbl"):
            self._delta_lbl.setText("")

    def push_spark(self, v: float) -> None:
        if self._spark:
            self._spark.push("v", v)
