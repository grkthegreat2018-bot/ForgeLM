"""EmptyState — a reusable placeholder shown when a list/collection is empty.

Displays a centered icon glyph, title, description, and an optional action
button. Uses the shared ``Palette`` tokens so it matches the dark theme.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..theme import Palette


class EmptyState(QFrame):
    """Centered empty-collection placeholder: icon + title + description + action.

    Parameters
    ----------
    title : str
        Bold heading line (e.g. "No checkpoints found").
    description : str
        Secondary help text shown beneath the title.
    icon : str, optional
        A single emoji/unicode glyph rendered large above the title.
        Defaults to "📭".
    action_text : str, optional
        If given, a primary-styled button is shown with this label.
    on_action : callable, optional
        Callback fired when the action button is clicked.  Alternatively
        connect to the ``action_triggered`` signal.
    """

    action_triggered = Signal()

    def __init__(self, title: str, description: str = "",
                 icon: str = "📭", action_text: Optional[str] = None,
                 on_action: Optional[Callable[[], None]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 40, 32, 40)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # icon
        self._icon_lbl = QLabel(icon)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet(
            f"font-size: 40px; color: {Palette.text_faint}; "
            f"background: transparent; border: none;")
        lay.addWidget(self._icon_lbl)

        # title
        self._title_lbl = QLabel(title)
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {Palette.text}; "
            f"background: transparent; border: none;")
        lay.addWidget(self._title_lbl)

        # description
        self._desc_lbl = QLabel(description)
        self._desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setStyleSheet(
            f"font-size: 12px; color: {Palette.text_dim}; "
            f"background: transparent; border: none;")
        lay.addWidget(self._desc_lbl)

        # optional action button
        self._action_btn: Optional[QPushButton] = None
        if action_text:
            row = QHBoxLayout()
            row.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._action_btn = QPushButton(action_text)
            self._action_btn.setObjectName("primary")
            self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action_btn.clicked.connect(self.action_triggered.emit)
            if on_action is not None:
                self.action_triggered.connect(on_action)
            row.addWidget(self._action_btn)
            lay.addLayout(row)

    # ── setters ──────────────────────────────────────────────────────
    def set_title(self, title: str) -> None:
        self._title_lbl.setText(title)

    def set_description(self, description: str) -> None:
        self._desc_lbl.setText(description)

    def set_icon(self, icon: str) -> None:
        self._icon_lbl.setText(icon)

    @property
    def action_button(self) -> Optional[QPushButton]:
        return self._action_btn
