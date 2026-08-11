"""Shared helpers for building pages."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout,
                               QScrollArea, QVBoxLayout, QWidget)

from ..theme import Palette


def make_card(title: str = "", value: str = "", unit: str = "") -> QFrame:
    card = QFrame(); card.setObjectName("card")
    lay = QVBoxLayout(card); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
    if title:
        t = QLabel(title.upper()); t.setObjectName("cardTitle")
        lay.addWidget(t)
    if value:
        row = QHBoxLayout(); row.setContentsMargins(16, 0, 16, 14); row.setSpacing(8)
        v = QLabel(value); v.setObjectName("cardValue")
        row.addWidget(v, 1)
        if unit:
            u = QLabel(unit); u.setObjectName("cardUnit")
            row.addWidget(u, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(row)
    return card


def section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper()); lbl.setObjectName("sectionHeader")
    return lbl


def card_grid(cards: list[QWidget], cols: int = 3, parent: Optional[QWidget] = None) -> QWidget:
    """Wrap cards in a responsive grid layout inside a container."""
    container = QWidget(); container.setObjectName("root")
    grid = QGridLayout(container)
    grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(14)
    for i, c in enumerate(cards):
        r, col = divmod(i, cols)
        grid.addWidget(c, r, col)
    # make columns stretch equally
    for col in range(cols):
        grid.setColumnStretch(col, 1)
    return container


def scroll_wrap(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(inner)
    sa.setObjectName("root")
    return sa


def page_container(*sections: QWidget, spacing: int = 18) -> QScrollArea:
    """Stack sections vertically inside a scroll area with consistent margins."""
    host = QWidget(); host.setObjectName("root")
    lay = QVBoxLayout(host); lay.setContentsMargins(28, 22, 28, 28); lay.setSpacing(spacing)
    lay.setAlignment(Qt.AlignmentFlag.AlignTop)
    for s in sections:
        lay.addWidget(s)
    return scroll_wrap(host)


def status_tag(text: str, kind: str = "idle") -> QLabel:
    """kind: ok | warn | err | idle"""
    lbl = QLabel(text.upper())
    lbl.setObjectName(f"tag{kind.capitalize()}")
    return lbl
