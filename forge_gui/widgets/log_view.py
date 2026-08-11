"""LogView — read-only tailed log console with level coloring + search highlight."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import (QColor, QFont, QSyntaxHighlighter, QTextCharFormat,
                           QTextDocument)
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from ..theme import Palette

LEVEL_COLORS = {
    "ERROR": Palette.err, "ERR": Palette.err, "FATAL": Palette.err,
    "CRITICAL": Palette.err, "WARN": Palette.warn, "WARNING": Palette.warn,
    "INFO": Palette.text, "DEBUG": Palette.text_dim, "TRACE": Palette.text_faint,
}


class _LevelHighlighter(QSyntaxHighlighter):
    def __init__(self, doc: QTextDocument) -> None:
        super().__init__(doc)
        self._fmts = {lvl: QTextCharFormat() for lvl in LEVEL_COLORS}
        for lvl, col in LEVEL_COLORS.items():
            self._fmts[lvl].setForeground(QColor(col))
            self._fmts[lvl].setFontWeight(QFont.Weight.Bold if lvl in
                                          ("ERROR", "FATAL", "CRITICAL", "WARN",
                                           "WARNING") else QFont.Weight.Normal)

    def highlightBlock(self, text: str) -> None:
        upper = text.upper()
        for lvl, fmt in self._fmts.items():
            idx = upper.find(lvl)
            if idx >= 0:
                self.setFormat(idx, len(lvl), fmt)
                return


class LogView(QPlainTextEdit):
    """Append-only log console with level coloring and a cap on buffer size."""

    def __init__(self, parent: Optional[QWidget] = None, max_blocks: int = 4000) -> None:
        super().__init__(parent)
        self.setObjectName("logView")
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_blocks)
        self._highlight = _LevelHighlighter(self.document())
        f = QFont("Cascadia Mono"); f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(10); self.setFont(f)

    def append_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        self.appendPlainText("\n".join(lines))

    def append_line(self, line: str) -> None:
        self.appendPlainText(line)

    def replace_all(self, lines: list[str]) -> None:
        self.setPlainText("\n".join(lines))
