"""NavSidebar — vertical nav rail with brand + selectable buttons + status footer."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPainter
from PySide6.QtWidgets import (QFrame, QLabel, QPushButton, QVBoxLayout, QWidget)


class NavButton(QPushButton):
    """Flat selectable nav button with an [active] property for QSS styling."""

    def __init__(self, text: str, icon: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setText(("  " + icon + "  " if icon else "    ") + text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", False)

    def set_active(self, on: bool) -> None:
        self.setProperty("active", on)
        self.setChecked(on)
        self.style().unpolish(self)
        self.style().polish(self)


class NavSidebar(QFrame):
    """Left navigation rail. Emits page_changed(index) on selection."""

    page_changed = Signal(int)

    def __init__(self, items: list[tuple[str, str]], parent: Optional[QWidget] = None) -> None:
        # items: list of (label, icon_glyph)
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(232)
        self._buttons: list[NavButton] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 22, 12, 14); lay.setSpacing(4)

        brand = QLabel("ForgeAI"); brand.setObjectName("brand")
        sub = QLabel("CONTROL CENTER"); sub.setObjectName("brandSub")
        lay.addWidget(brand); lay.addWidget(sub)
        lay.addSpacing(22)

        for i, (label, icon) in enumerate(items):
            b = NavButton(label, icon)
            b.clicked.connect(lambda _=False, idx=i: self._select(idx))
            self._buttons.append(b)
            lay.addWidget(b)
        lay.addStretch(1)

        self._status = QLabel("● idle")
        self._status.setObjectName("statusLabel")
        lay.addWidget(self._status)
        lay.addSpacing(4)
        self._gpu = QLabel("GPU —")
        self._gpu.setObjectName("statusLabel")
        lay.addWidget(self._gpu)

        if self._buttons:
            self._buttons[0].set_active(True)

    def _select(self, idx: int) -> None:
        for i, b in enumerate(self._buttons):
            b.set_active(i == idx)
        self.page_changed.emit(idx)

    def select_page(self, idx: int) -> None:
        """Public programmatic page selection (e.g. keyboard shortcuts)."""
        if 0 <= idx < len(self._buttons):
            self._select(idx)

    def set_status(self, text: str, color: str = "#5a6577") -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-size: 11px; padding: 6px 16px;")

    def set_gpu(self, text: str) -> None:
        self._gpu.setText(text)
