"""NavSidebar — vertical nav rail with brand + grouped sections + status footer.

Supports optional section headers to group related pages. Pass items as
either (label, icon) for a plain button, or ("__section__", "Section Title")
to insert a non-selectable section header.

Features:
- **Collapsible**: click the collapse button (or Ctrl+B) to toggle between
  full width (232px) and collapsed (48px, icons only).
- **Scrollable**: nav buttons live inside a QScrollArea so they don't get
  squished on short windows.
- **Search filter**: a search box at the top filters buttons by label
  (hidden when collapsed).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QToolButton,
                               QVBoxLayout, QWidget)

from ..api.status_reader import project_root

_FULL_WIDTH = 232
_COLLAPSED_WIDTH = 48


class NavButton(QPushButton):
    """Flat selectable nav button with an [active] property for QSS styling."""

    def __init__(self, text: str, icon: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = text
        self._full_text = ("  " + icon + "  " if icon else "    ") + text
        self._icon = icon
        self._collapsed_text = "  " + icon + "  " if icon else "    "
        self.setText(self._full_text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", False)
        self._collapsed = False

    @property
    def label(self) -> str:
        return self._label

    def set_active(self, on: bool) -> None:
        self.setProperty("active", on)
        self.setChecked(on)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.setText(self._collapsed_text if collapsed else self._full_text)
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
    Searchable: type in the search box to filter buttons by label.
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
        # map: section label → list of button indices that belong to it
        self._section_children: dict[int, list[int]] = {}
        self._page_indices: list[int] = []   # button row → logical page idx

        # ── main layout: brand (fixed) + search (fixed) + scroll (flex) + status (fixed) ──
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── brand row + collapse button (fixed top) ──
        brand_container = QFrame()
        brand_container.setObjectName("sidebarBrand")
        brand_lay = QVBoxLayout(brand_container)
        brand_lay.setContentsMargins(16, 22, 12, 8)
        brand_lay.setSpacing(6)

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
            self._logo.setText("\u2692")
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
        brand_lay.addLayout(brand_row)

        # ── search box (fixed, below brand) ──
        self._search = QLineEdit()
        self._search.setObjectName("navSearch")
        self._search.setPlaceholderText("Search pages...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter_buttons)
        brand_lay.addWidget(self._search)

        outer.addWidget(brand_container)

        # ── scrollable nav area (flex) ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setObjectName("navScroll")
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_content.setObjectName("navScrollContent")
        self._nav_lay = QVBoxLayout(scroll_content)
        self._nav_lay.setContentsMargins(16, 8, 12, 8)
        self._nav_lay.setSpacing(4)

        # populate nav items
        current_section_idx = -1
        page_idx = 0
        for label, icon in items:
            if label == "__section__":
                if page_idx > 0:
                    self._nav_lay.addSpacing(10)
                hdr = QLabel(icon.upper())
                hdr.setObjectName("navSection")
                self._nav_lay.addWidget(hdr)
                self._section_labels.append(hdr)
                current_section_idx = len(self._section_labels) - 1
                self._section_children[current_section_idx] = []
                self._nav_lay.addSpacing(2)
                continue
            b = NavButton(label, icon)
            b.clicked.connect(lambda _=False, idx=page_idx: self._select(idx))
            self._buttons.append(b)
            self._page_indices.append(page_idx)
            self._nav_lay.addWidget(b)
            if current_section_idx >= 0:
                self._section_children[current_section_idx].append(
                    len(self._buttons) - 1)
            page_idx += 1

        self._nav_lay.addStretch(1)
        self._scroll.setWidget(scroll_content)
        outer.addWidget(self._scroll, 1)

        # ── status footer (fixed bottom) ──
        footer = QFrame()
        footer.setObjectName("sidebarFooter")
        footer_lay = QVBoxLayout(footer)
        footer_lay.setContentsMargins(16, 4, 12, 14)
        footer_lay.setSpacing(4)
        self._status = QLabel("\u25CF idle")
        self._status.setObjectName("statusLabel")
        footer_lay.addWidget(self._status)
        self._gpu = QLabel("GPU \u2014")
        self._gpu.setObjectName("statusLabel")
        footer_lay.addWidget(self._gpu)
        outer.addWidget(footer)

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

    # ── search filter ─────────────────────────────────────────────────
    def _filter_buttons(self, text: str) -> None:
        """Filter nav buttons by search text. Hides non-matching buttons
        and their section headers if all children are hidden."""
        if not text.strip():
            # show everything
            for b in self._buttons:
                b.setVisible(True)
            for hdr in self._section_labels:
                hdr.setVisible(True)
            return

        query = text.lower().strip()
        for b in self._buttons:
            b.setVisible(query in b.label.lower())

        # hide section headers if all their children are hidden
        for sec_idx, child_indices in self._section_children.items():
            any_visible = any(
                i < len(self._buttons) and self._buttons[i].isVisible()
                for i in child_indices)
            self._section_labels[sec_idx].setVisible(any_visible)

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
            # hide search box in collapsed mode
            self._search.setVisible(False)
            self._status.setVisible(False)
            self._gpu.setVisible(False)
            # adjust margins for icon-only mode
            self._nav_lay.setContentsMargins(8, 8, 4, 8)
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
            self._search.setVisible(True)
            self._search.clear()
            self._status.setVisible(True)
            self._gpu.setVisible(True)
            self._nav_lay.setContentsMargins(16, 8, 12, 8)
        self.collapsed_changed.emit(collapsed)
