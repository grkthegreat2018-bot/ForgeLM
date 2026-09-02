"""NavSidebar — vertical nav rail with brand + grouped sections + status footer.

Supports optional section headers to group related pages. Pass items as
either (label, icon) for a plain button, or ("__section__", "Section Title")
to insert a non-selectable section header.

The sidebar is collapsible — click the collapse button (or press Ctrl+B)
to toggle between full width (232px) and collapsed (48px, icons only).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QToolButton, QVBoxLayout, QWidget)

from ..api.status_reader import project_root

_FULL_WIDTH = 232
_COLLAPSED_WIDTH = 48


class NavButton(QPushButton):
    """Flat selectable nav button with an [active] property for QSS styling."""

    def __init__(self, text: str, icon: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_text = ("  " + icon + "  " if icon else "    ") + text
        self._icon = icon
        self._collapsed_text = "  " + icon + "  " if icon else "    "
        self.setText(self._full_text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", False)
        self._collapsed = False

    def set_active(self, on: bool) -> None:
        self.setProperty("active", on)
        self.setChecked(on)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.setText(self._collapsed_text if collapsed else self._full_text)
        # hide text in collapsed mode (show only icon)
        if collapsed:
            self.setToolTip(self._full_text.strip())
        else:
            self.setToolTip("")


class NavSidebar(QFrame):
    """Left navigation rail. Emits page_changed(index) on selection.

    Items list entries:
      (label, icon)            — a selectable page button
      ("__section__", title)   — a non-selectable section header
    Page indices count only selectable buttons (section headers are skipped).

    Collapsible: call toggle_collapse() or click the collapse button.
    """

    page_changed = Signal(int)
    collapsed_changed = Signal(bool)

    def __init__(self, items: list[tuple[str, str]],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._full_width = _FULL_WIDTH
        self._collapsed_width = _COLLAPSED_WIDTH
        self.setFixedWidth(self._full_width)
        self._collapsed = False
        self._buttons: list[NavButton] = []
        self._section_labels: list[QLabel] = []
        self._page_indices: list[int] = []   # button row → logical page idx
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 22, 12, 14); lay.setSpacing(4)

        # ── brand row + collapse button ──
        brand_row = QHBoxLayout(); brand_row.setSpacing(10)
        self._logo = QLabel(); self._logo.setObjectName("brand")
        icon_path = project_root() / "ForgeAI_Icon.png"
        if icon_path.is_file():
            pm = QPixmap(str(icon_path))
            if not pm.isNull():
                self._logo.setPixmap(pm.scaled(
                    40, 40, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self._logo.setFixedSize(40, 40)
        if self._logo.pixmap() is None or self._logo.pixmap().isNull():
            self._logo.setText("⚒")
        self._brand_col = QVBoxLayout(); self._brand_col.setSpacing(0)
        self._brand_col.setContentsMargins(0, 0, 0, 0)
        brand = QLabel("ForgeAI"); brand.setObjectName("brand")
        sub = QLabel("CONTROL CENTER"); sub.setObjectName("brandSub")
        self._brand_col.addWidget(brand); self._brand_col.addWidget(sub)
        brand_row.addWidget(self._logo, 0, Qt.AlignmentFlag.AlignTop)
        brand_row.addLayout(self._brand_col)
        brand_row.addStretch(1)
        # collapse toggle button
        self._collapse_btn = QToolButton()
        self._collapse_btn.setText("\u25C0")
        self._collapse_btn.setToolTip("Collapse sidebar (Ctrl+B)")
        self._collapse_btn.setObjectName("collapseBtn")
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.clicked.connect(self.toggle_collapse)
        brand_row.addWidget(self._collapse_btn, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(brand_row)
        lay.addSpacing(22)

        page_idx = 0
        for label, icon in items:
            if label == "__section__":
                if page_idx > 0:
                    lay.addSpacing(10)
                hdr = QLabel(icon.upper())
                hdr.setObjectName("navSection")
                lay.addWidget(hdr)
                self._section_labels.append(hdr)
                lay.addSpacing(2)
                continue
            b = NavButton(label, icon)
            b.clicked.connect(lambda _=False, idx=page_idx: self._select(idx))
            self._buttons.append(b)
            self._page_indices.append(page_idx)
            lay.addWidget(b)
            page_idx += 1

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
            b.set_active(self._page_indices[i] == idx)
        self.page_changed.emit(idx)

    def select_page(self, idx: int) -> None:
        """Public programmatic page selection (e.g. keyboard shortcuts)."""
        for i, pi in enumerate(self._page_indices):
            if pi == idx:
                self._select(idx)
                return

    def set_status(self, text: str, color: str = "#5a6577") -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-size: 11px; padding: 6px 16px;")

    def set_gpu(self, text: str) -> None:
        self._gpu.setText(text)

    # ── collapse / expand ─────────────────────────────────────────────
    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle_collapse(self) -> None:
        """Toggle between full and collapsed width."""
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Set the sidebar to collapsed or expanded state."""
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        if collapsed:
            self.setFixedWidth(self._collapsed_width)
            self._collapse_btn.setText("\u25B6")
            self._collapse_btn.setToolTip("Expand sidebar (Ctrl+B)")
            # hide text labels, show only icons
            for b in self._buttons:
                b.set_collapsed(True)
            for hdr in self._section_labels:
                hdr.setVisible(False)
            # hide brand text, keep logo
            for i in range(self._brand_col.count()):
                w = self._brand_col.itemAt(i).widget()
                if w:
                    w.setVisible(False)
            self._status.setVisible(False)
            self._gpu.setVisible(False)
            # adjust margins for icon-only mode
            self.layout().setContentsMargins(8, 22, 4, 14)
        else:
            self.setFixedWidth(self._full_width)
            self._collapse_btn.setText("\u25C0")
            self._collapse_btn.setToolTip("Collapse sidebar (Ctrl+B)")
            for b in self._buttons:
                b.set_collapsed(False)
            for hdr in self._section_labels:
                hdr.setVisible(True)
            for i in range(self._brand_col.count()):
                w = self._brand_col.itemAt(i).widget()
                if w:
                    w.setVisible(True)
            self._status.setVisible(True)
            self._gpu.setVisible(True)
            self.layout().setContentsMargins(16, 22, 12, 14)
        self.collapsed_changed.emit(collapsed)
