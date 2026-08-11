"""Chat page — migrated chat UI talking to a local OpenAI-compatible endpoint.

Mirrors the existing web/app.js chat experience: prompt input, streaming
response render, model selector, temperature, system prompt. Uses the
ForgeAI backend proxy at /v1 (same as scripts/launch.py) when available,
otherwise shows an offline notice.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout,
                               QWidget)

from ..theme import Palette
from ._base import page_container, section_label

logger = logging.getLogger(__name__)


class _Worker(QThread):
    """Streams a chat completion in the background.

    Supports cooperative cancellation: call cancel(), which the streaming
    loop observes between chunks; terminate() is only a last resort.
    """

    chunk = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, base, model, temp, messages):
        super().__init__()
        self.base, self.model, self.temp, self.messages = base, model, temp, messages
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation of the streaming loop."""
        self._cancelled = True

    def run(self):
        try:
            url = self.base.rstrip("/") + "/chat/completions"
            payload = json.dumps({
                "model": self.model, "temperature": self.temp,
                "stream": True, "messages": self.messages,
            }).encode()
            req = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                buf = b""
                for raw in iter(lambda: resp.read(1024), b""):
                    if self._cancelled:
                        break
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._emit(line.decode(errors="ignore"))
                if buf and not self._cancelled:
                    self._emit(buf.decode(errors="ignore"))
            self.done.emit("")
        except Exception as e:
            if self._cancelled:
                self.done.emit("")
            else:
                logger.warning("chat request failed: %s", e)
                self.failed.emit(f"{type(e).__name__}: {e}")

    def _emit(self, line: str):
        line = line.strip()
        if not line or not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                self.chunk.emit(delta)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.warning("skipping malformed stream chunk: %s (%r)", e, data[:120])


class _MessageBubble(QFrame):
    def __init__(self, role: str, text: str) -> None:
        super().__init__()
        is_user = role == "user"
        self.setObjectName("cardAlt" if is_user else "card")
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 10, 14, 10)
        who = QLabel(("YOU" if is_user else "FORGEAI"))
        who.setStyleSheet(f"color:{Palette.accent if is_user else Palette.chart_kl};"
                          f" font-size:10px; font-weight:700; letter-spacing:1px;")
        lay.addWidget(who)
        body = QLabel(text); body.setWordWrap(True)
        body.setStyleSheet("color:#e6edf3; font-size:13px;")
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(body)


class ChatPage(QWidget):
    def __init__(self, parent: Optional[QWidget] = None,
                 base_url: str = "http://localhost:8080/v1") -> None:
        super().__init__(parent)
        self._base = base_url.rstrip("/")
        self._messages: list[dict] = []
        self._system = ""

        # ---- settings strip ----
        cfg = QFrame(); cfg.setObjectName("card")
        cl = QHBoxLayout(cfg); cl.setContentsMargins(16,12,16,12); cl.setSpacing(12)
        cl.addWidget(section_label("MODEL"))
        self._model = QLineEdit("llama3"); self._model.setFixedWidth(160)
        cl.addWidget(self._model)
        cl.addWidget(QLabel("Temp"))
        self._temp = QDoubleSpinBox(); self._temp.setRange(0, 2); self._temp.setValue(0.7)
        self._temp.setSingleStep(0.05); self._temp.setFixedWidth(80)
        cl.addWidget(self._temp)
        cl.addWidget(QLabel("Endpoint"))
        self._endpoint = QLineEdit(self._base); self._endpoint.setMinimumWidth(220)
        cl.addWidget(self._endpoint, 1)
        self._fetch_btn = QPushButton("Fetch models")
        cl.addWidget(self._fetch_btn)
        self._clear_btn = QPushButton("Clear")
        cl.addWidget(self._clear_btn)

        # ---- conversation scroll ----
        self._conv_host = QWidget(); self._conv_host.setObjectName("root")
        self._conv_lay = QVBoxLayout(self._conv_host)
        self._conv_lay.setContentsMargins(0, 0, 0, 0); self._conv_lay.setSpacing(12)
        self._conv_lay.addStretch(1)
        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame); self._scroll.setObjectName("root")
        self._scroll.setWidget(self._conv_host)

        # ---- composer ----
        comp = QFrame(); comp.setObjectName("card")
        cpl = QVBoxLayout(comp); cpl.setContentsMargins(12,12,12,12); cpl.setSpacing(8)
        self._input = QPlainTextEdit(); self._input.setPlaceholderText(
            "Message ForgeAI…  (Enter to send, Shift+Enter for newline)")
        self._input.setFixedHeight(80)
        cpl.addWidget(self._input)
        btn_row = QHBoxLayout()
        self._send_btn = QPushButton("Send"); self._send_btn.setObjectName("primary")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._send_btn); btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        self._status = QLabel("Idle"); self._status.setStyleSheet("color:#5a6577; font-size:11px;")
        btn_row.addWidget(self._status)
        cpl.addLayout(btn_row)

        self._host = page_container(cfg, self._scroll, comp)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(self._host)

        self._send_btn.clicked.connect(self._send)
        self._stop_btn.clicked.connect(self._stop)
        self._clear_btn.clicked.connect(self._clear)
        self._fetch_btn.clicked.connect(self._fetch_models)
        self._input.installEventFilter(self)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self._input and ev.type() == QEvent.Type.KeyPress:
            k: QKeyEvent = ev
            if k.key() == Qt.Key.Key_Return and not (k.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._send()
                return True
        return super().eventFilter(obj, ev)

    def _add_bubble(self, role: str, text: str) -> _MessageBubble:
        b = _MessageBubble(role, text)
        # insert before the trailing stretch
        self._conv_lay.insertWidget(self._conv_lay.count() - 1, b)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return b

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._add_bubble("user", text)
        self._messages.append({"role": "user", "content": text})
        self._send_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._status.setText("Sending…")
        # run request in a worker to keep UI responsive
        self._worker = _Worker(self._endpoint.text() or self._base,
                               self._model.text(), self._temp.value(),
                               list(self._messages))
        self._assistant_bubble_text = ""
        self._assistant_bubble = self._add_bubble("assistant", "…")
        self._worker.chunk.connect(self._on_chunk)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_chunk(self, delta: str) -> None:
        self._assistant_bubble_text += delta
        body = self._assistant_bubble.findChild(QLabel)
        if body:
            body.setText(self._assistant_bubble_text)
        self._scroll_to_bottom()

    def _on_done(self, _final: str) -> None:
        self._messages.append({"role": "assistant", "content": self._assistant_bubble_text})
        self._send_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._status.setText("done")

    def _on_fail(self, err: str) -> None:
        body = self._assistant_bubble.findChild(QLabel)
        if body:
            body.setText(f"(error) {err}\n\nIs the ForgeAI server running? "
                         f"Start it with: python scripts/launch.py")
        self._send_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._status.setText(f"error: {err}")

    def _stop(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.isRunning():
            # cooperative cancellation first; terminate only as a last resort
            worker.cancel()
            if not worker.wait(2000):
                logger.warning("chat worker did not stop in 2s — terminating")
                worker.terminate()
                worker.wait(1000)
        self._send_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._status.setText("stopped")

    def _clear(self) -> None:
        self._messages.clear()
        while self._conv_lay.count() > 1:
            it = self._conv_lay.takeAt(0)
            w = it.widget()
            if w: w.deleteLater()

    def _fetch_models(self) -> None:
        try:
            url = (self._endpoint.text() or self._base).rstrip("/") + "/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            names = [m.get("id", "?") for m in data.get("data", [])]
            self._status.setText(f"fetched {len(names)} models: {', '.join(names[:4])}")
        except Exception as e:
            self._status.setText(f"fetch failed: {e}")

    def refresh(self) -> None:
        pass
