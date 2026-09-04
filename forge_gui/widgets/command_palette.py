"""R36-3: Command Palette — Ctrl+K global search across pages and actions."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)


class CommandPalette(QDialog):
    """VS Code-style command palette for quick navigation and actions.

    Searches across:
    - All GUI pages (navigate)
    - Quick actions (toggle theme, font size, refresh)
    """

    page_selected = Signal(int)

    def __init__(self, parent, index_to_name: dict[int, str]):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setObjectName("commandPalette")
        self._index_to_name = index_to_name
        self._all_items: list[tuple[str, int, str]] = []  # (label, page_idx, category)
        self._build_items()
        self._setup_ui()

    def _build_items(self) -> None:
        """Build the searchable item list."""
        # Pages
        for idx, name in sorted(self._index_to_name.items()):
            self._all_items.append((f"Go to {name}", idx, "Navigate"))
        # Actions (page_idx = -1 means it's an action, not navigation)
        self._all_items.append(("Toggle Theme (Light/Dark)", -1, "Action:theme"))
        self._all_items.append(("Font Size: Small", -1, "Action:font:small"))
        self._all_items.append(("Font Size: Medium", -1, "Action:font:medium"))
        self._all_items.append(("Font Size: Large", -1, "Action:font:large"))
        self._all_items.append(("Font Size: Extra Large", -1, "Action:font:xl"))
        self._all_items.append(("Refresh Current Page", -1, "Action:refresh"))

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search pages and actions…")
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._activate_selected)
        layout.addWidget(self.search)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._activate_item)
        self.list.itemActivated.connect(self._activate_item)
        layout.addWidget(self.list)

        self._populate(self._all_items)
        self.search.setFocus()

    def _filter(self, text: str) -> None:
        text = text.lower().strip()
        if not text:
            self._populate(self._all_items)
            return
        filtered = [
            item for item in self._all_items
            if text in item[0].lower()
        ]
        self._populate(filtered)

    def _populate(self, items: list[tuple[str, int, str]]) -> None:
        self.list.clear()
        for label, idx, category in items:
            li = QListWidgetItem(f"{label}")
            li.setData(Qt.ItemDataRole.UserRole, (idx, category))
            self.list.addItem(li)
        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _activate_selected(self) -> None:
        item = self.list.currentItem()
        if item:
            self._activate_item(item)

    def _activate_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return
        idx, category = data
        if idx >= 0:
            self.page_selected.emit(idx)
        elif category.startswith("Action:"):
            self._execute_action(category)
        self.accept()

    def _execute_action(self, action: str) -> None:
        from ..theme import ThemeManager
        parts = action.split(":")
        if "theme" in parts:
            ThemeManager.toggle_theme()
        elif "font" in parts:
            size = parts[-1] if len(parts) > 2 else "medium"
            size_map = {"small": "small", "medium": "medium",
                        "large": "large", "xl": "xl"}
            ThemeManager.set_font_size(size_map.get(size, "medium"))
        elif "refresh" in parts:
            parent = self.parent()
            if parent and hasattr(parent, "_force_refresh"):
                parent._force_refresh()
