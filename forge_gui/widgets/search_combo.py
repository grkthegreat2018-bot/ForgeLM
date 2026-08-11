"""SearchableComboBox — QComboBox with filter-as-you-type.

A drop-in replacement for QComboBox that makes the popup filter items
based on the typed text. Supports both display text and user data.

Usage:
    combo = SearchableComboBox()
    combo.addItem("my_model.safetensors", "path/to/model")
    combo.addItem("another.pt", "path/to/another")
    # User types "my" → popup shows only "my_model.safetensors"
"""
from __future__ import annotations


from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter


class SearchableComboBox(QComboBox):
    """QComboBox with case-insensitive fuzzy filtering on the popup."""

    def __init__(self, parent: Qt | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        # Completer for live filtering
        self._completer = QCompleter(self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setModel(self.model())
        self.setCompleter(self._completer)

        # Clear placeholder text; show hint instead
        self.lineEdit().setPlaceholderText("Type to search…")
        self.lineEdit().setClearButtonEnabled(True)

        # When user selects from popup, clear the edit text and show the item
        self._completer.activated.connect(self._on_completer_activated)

        # Prevent the edit text from being treated as a new item
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_completer_activated(self, text: str) -> None:
        """Find the item matching the completer text and select it."""
        for i in range(self.count()):
            if self.itemText(i) == text:
                self.setCurrentIndex(i)
                self.lineEdit().clear()
                return

    def _on_index_changed(self, idx: int) -> None:
        """Clear the search text when a real item is selected."""
        if idx >= 0:
            self.lineEdit().clear()

    def currentSearchData(self):
        """Return the data of the currently selected item, or None."""
        idx = self.currentIndex()
        if idx < 0:
            return None
        return self.itemData(idx)
