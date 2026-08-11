"""Custom QPainter live line chart — heart-monitor style scrolling plot.

Supports multiple named series with rolling windows, gradient fills, grid,
and a current-value readout. No external chart dep.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QLinearGradient, QBrush
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import Palette


class LiveLineChart(QWidget):
    """Scrolling multi-series line chart with gradient area fill."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                 series: Optional[list[tuple[str, str]]] = None,
                 window: int = 240, min_y: Optional[float] = None,
                 max_y: Optional[float] = None, y_label: str = "",
                 height: int = 160) -> None:
        super().__init__(parent)
        # series: list of (name, color_hex)
        self._series_defs = series or [("value", Palette.chart_loss)]
        self._window = window
        self._data: dict[str, deque] = {
            name: deque(maxlen=window) for name, _ in self._series_defs
        }
        self._colors: dict[str, str] = {n: c for n, c in self._series_defs}
        self._min_y = min_y
        self._max_y = max_y
        self._y_label = y_label
        self._fixed_height = height
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._show_grid = True

    # ---- public API ----
    def push(self, name: str, value: float) -> None:
        if name not in self._data:
            self._data[name] = deque(maxlen=self._window)
        self._data[name].append(float(value))
        self.update()

    def push_multi(self, values: dict[str, float]) -> None:
        for k, v in values.items():
            self.push(k, v)
        self.update()

    def clear(self) -> None:
        for d in self._data.values():
            d.clear()
        self.update()

    def set_series(self, series: list[tuple[str, str]]) -> None:
        self._series_defs = series
        self._colors = {n: c for n, c in series}
        for n, _ in series:
            self._data.setdefault(n, deque(maxlen=self._window))
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(400, self._fixed_height)

    # ---- painting ----
    def _y_range(self) -> tuple[float, float]:
        vals: list[float] = []
        for d in self._data.values():
            vals.extend(d)
        if not vals:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = min(vals), max(vals)
            if abs(hi - lo) < 1e-9:
                hi = lo + 1.0
            pad = (hi - lo) * 0.12
            lo -= pad
            hi += pad
        if self._min_y is not None:
            lo = self._min_y
        if self._max_y is not None:
            hi = self._max_y
        return lo, hi

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        bg = QColor(Palette.panel)
        p.fillRect(self.rect(), bg)
        # subtle border
        p.setPen(QPen(QColor(Palette.border), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 10, 10)

        pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 22
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)

        lo, hi = self._y_range()
        span = max(1e-9, hi - lo)

        # grid + y labels
        if self._show_grid:
            grid_pen = QPen(QColor(Palette.border), 1)
            grid_pen.setStyle(Qt.PenStyle.DotLine)
            p.setPen(grid_pen)
            f = QFont(); f.setPointSize(8); p.setFont(f)
            p.setPen(QColor(Palette.text_faint))
            for i in range(5):
                yy = pad_t + plot_h * i / 4
                p.setPen(QPen(QColor(Palette.border), 1, Qt.PenStyle.DotLine))
                p.drawLine(pad_l, int(yy), pad_l + plot_w, int(yy))
                val = hi - span * i / 4
                p.setPen(QColor(Palette.text_faint))
                p.drawText(2, int(yy) + 3, 38, 14, Qt.AlignmentFlag.AlignRight,
                           f"{val:.3g}")
            # x baseline
            p.setPen(QPen(QColor(Palette.border), 1))
            p.drawLine(pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h)

        # series
        for name, color_hex in self._series_defs:
            d = self._data.get(name)
            if not d or len(d) < 2:
                continue
            color = QColor(color_hex)
            n = len(d)
            pts = []
            for i, v in enumerate(d):
                x = pad_l + plot_w * i / max(1, self._window - 1)
                yv = pad_t + plot_h * (1 - (v - lo) / span)
                yv = max(pad_t, min(pad_t + plot_h, yv))
                pts.append((x, yv))
            # area fill
            grad = QLinearGradient(0, pad_t, 0, pad_t + plot_h)
            c0 = QColor(color); c0.setAlpha(90)
            c1 = QColor(color); c1.setAlpha(0)
            grad.setColorAt(0, c0); grad.setColorAt(1, c1)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            fill_poly = pts + [(pts[-1][0], pad_t + plot_h), (pts[0][0], pad_t + plot_h)]
            from PySide6.QtGui import QPolygonF
            from PySide6.QtCore import QPointF
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in fill_poly]))
            # line
            pen = QPen(color); pen.setWidthF(1.8)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
            # current dot
            lx, ly = pts[-1]
            p.setBrush(color); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(lx, ly), 3.0, 3.0)

        # y label
        if self._y_label:
            p.setPen(QColor(Palette.text_dim))
            f = QFont(); f.setPointSize(8); p.setFont(f)
            p.drawText(pad_l, h - 4, self._y_label)
        p.end()
