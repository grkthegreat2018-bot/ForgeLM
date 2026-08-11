"""Color-coded log panel — shared by Compute and Self-Play pages.

Color scheme (per user spec):
  Green  = correct, valid, finished
  Yellow = stall, waiting, retry, light warning, fallback
  Orange = error, fail, gen garbled, failed fallback
  Red    = crash, major error, OOM
  White  = normal, info, basic, general (default)
  Bold   = phase, event marker, state log, important
  Purple = unexpected, instability, NaN, failsafe, investigate
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCharFormat, QTextCursor, QColor
from PySide6.QtWidgets import QPlainTextEdit, QWidget

from ..theme import Palette


# Log levels mapped to colors
LEVEL_COLORS = {
    "ok":      "#4caf50",   # green
    "info":    "#c8d0dc",   # white-ish (default text)
    "warn":    "#ffc107",   # yellow
    "error":   "#ff9800",   # orange
    "crash":   "#f44336",   # red
    "phase":   "#64b5f6",   # blue-bold (phase markers)
    "invest":  "#ab47bc",   # purple
}

# Font weight for bold events
BOLD = "font-weight:bold;"
LARGE = "font-size:13px;"


class ColorLogWidget(QPlainTextEdit):
    """A log panel that appends color-coded HTML lines.

    Usage:
        log = ColorLogWidget(max_lines=500)
        log.append_line("Task completed", level="ok")
        log.append_line("VRAM OOM at 16 tokens", level="crash")
        log.append_line("[PHASE] Epoch 1/3", level="phase", bold=True)
    """

    def __init__(self, max_lines: int = 500, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        self.setObjectName("cardBody")
        f = QFont("Cascadia Mono"); f.setStyleHint(QFont.StyleHint.Monospace); f.setPointSize(10)
        self.setFont(f)
        # Dark background for contrast
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: #1a1e26; color: {LEVEL_COLORS['info']}; "
            f"border: none; padding: 8px; }}")
        self._line_count = 0

    def append_line(self, text: str, level: str = "info",
                    bold: bool = False, large: bool = False) -> None:
        """Append a color-coded line to the log.

        Args:
            text: the log message (plain text, HTML-escaped automatically)
            level: one of ok/info/warn/error/crash/phase/invest
            bold: render in bold (for phase markers, state changes)
            large: render in larger font (for important events)
        """
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
        # Escape HTML special chars in the text
        safe = (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        style = f"color:{color};"
        if bold:
            style += BOLD
        if large:
            style += LARGE
        html = f'<div style="{style}">{safe}</div>'
        self.appendHtml(html)
        # Auto-scroll to bottom
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._line_count += 1

    def clear_log(self) -> None:
        """Clear all log content."""
        self.clear()
        self._line_count = 0
