"""Circular gauge widget — used for VRAM %, compute load, progress."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSize, QPointF, QRectF
from PySide6.QtGui import (QColor, QFont, QLinearGradient, QPainter, QPen,
                           QConicalGradient, QBrush)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import Palette


class CircularGauge(QWidget):
    """Arc gauge with a gradient sweep, center % label, and caption."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                 caption: str = "", unit: str = "%",
                 min_val: float = 0.0, max_val: float = 100.0,
                 size: int = 140, start_angle: float = 225.0,
                 span_angle: float = 270.0,
                 color_a: str = Palette.grad_a, color_b: str = Palette.grad_b,
                 danger_color: str = Palette.err) -> None:
        super().__init__(parent)
        self._caption = caption
        self._unit = unit
        self._min = min_val
        self._max = max_val
        self._value = 0.0
        self._size = size
        self._start = start_angle
        self._span = span_angle
        self._color_a = color_a
        self._color_b = color_b
        self._danger = danger_color
        self._danger_threshold = 0.85  # fraction of range
        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_value(self, v: float) -> None:
        self._value = max(self._min, min(self._max, float(v)))
        self.update()

    def set_caption(self, c: str) -> None:
        self._caption = c
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._size, self._size)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 12
        arc_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        # background track
        p.setPen(QPen(QColor(Palette.panel_alt), 10, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawArc(arc_rect, int(self._start * 16), int(-self._span * 16))
        # value arc
        frac = (self._value - self._min) / max(1e-9, self._max - self._min)
        if frac > 0:
            sweep = -self._span * frac
            color = QColor(self._danger if frac > self._danger_threshold else self._color_a)
            if frac <= self._danger_threshold:
                grad = QConicalGradient(cx, cy, self._start)
                grad.setColorAt(0.0, QColor(self._color_a))
                grad.setColorAt(1.0, QColor(self._color_b))
                pen = QPen(QBrush(grad), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            else:
                pen = QPen(color, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawArc(arc_rect, int(self._start * 16), int(sweep * 16))
        # center label
        p.setPen(QColor(Palette.text))
        f = QFont(); f.setPointSize(max(1, self._size // 9)); f.setBold(True); p.setFont(f)
        if self._unit == "%":
            txt = f"{frac*100:.0f}%"
        else:
            txt = f"{self._value:.2f}"
        p.drawText(self.rect().adjusted(0, -6, 0, 0), Qt.AlignmentFlag.AlignCenter, txt)
        # caption
        p.setPen(QColor(Palette.text_dim))
        f2 = QFont(); f2.setPointSize(8); p.setFont(f2)
        p.drawText(self.rect().adjusted(0, self._size // 3, 0, 0),
                   Qt.AlignmentFlag.AlignCenter, self._caption)
        p.end()
