"""Chat Studio — LM-Studio-style local chat with quality ratings → SFT data.

Three-pane layout:
  ┌──────────┬────────────────────────────┬───────────────┐
  │ conv list│   streaming chat bubbles   │ inference     │
  │ + export │   (rate 👍/👎 per reply)   │ settings      │
  └──────────┴────────────────────────────┴───────────────┘

Two inference sources:
  • Local ForgeEngine — the resident engine from EngineRuntime (streaming)
  • HTTP endpoint — any OpenAI-compatible /v1 server (e.g. forge_server.py)

Every assistant reply can be rated good/bad; good-rated turns export as
sft_train-compatible JSONL (ChatStore.export_training_data).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.request
from typing import Optional

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTextEdit,
                               QToolButton, QVBoxLayout, QWidget)

from ..api.chat_store import RATING_BAD, RATING_GOOD, ChatStore
from ..api.engine_runtime import EngineRuntime
from ..api.master_prompt import generate_master_prompt, get_default_prompt_for_config
from ..theme import Palette
from ._base import section_label

logger = logging.getLogger(__name__)


# ── workers ────────────────────────────────────────────────────────────

class _EngineChatWorker(QThread):
    """Streams a chat completion from the resident ForgeEngine."""

    chunk = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, runtime: EngineRuntime, messages: list[dict],
                 max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int,
                 repetition_penalty: float = 1.05, parent=None) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self.messages = messages
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from research.self_play.discovery.qwen_adapter import (
                qwen_render_messages,
            )
            rendered = qwen_render_messages(self.messages, add_generation_prompt=True)
            parts: list[str] = []
            with self._runtime.acquire(timeout_s=30.0) as engine:
                for tok in engine.generate_stream(
                        rendered, max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty):
                    if self._cancelled:
                        break
                    parts.append(tok)
                    self.chunk.emit(tok)
            self.done.emit("".join(parts))
        except RuntimeError as e:
            # engine busy / not loaded — non-fatal, report cleanly
            self.failed.emit(str(e))
        except Exception as e:
            logger.warning("engine chat failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _EngineChatToolWorker(QThread):
    """Chat worker with tool calling support.

    Runs an agentic loop: generate → parse tool calls → execute → feed back.
    Streams the final text response. Tool calls are emitted as signals
    so the UI can show tool-use cards.
    """

    chunk = Signal(str)
    done = Signal(str)
    failed = Signal(str)
    tool_call_made = Signal(dict)      # {name, arguments}
    tool_call_result = Signal(dict)    # sandbox record

    def __init__(self, runtime: EngineRuntime, messages: list[dict],
                 tool_harness, max_new_tokens: int, temperature: float,
                 top_p: float, top_k: int,
                 repetition_penalty: float = 1.05,
                 max_rounds: int = 6, parent=None) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self.messages = messages
        self.tool_harness = tool_harness
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_rounds = max_rounds
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from research.self_play.discovery.qwen_adapter import (
                qwen_parse_tool_calls, qwen_render_messages,
            )
            import json as _json

            # use reduced tool set for chat (keeps prompt within KV cache)
            defs = self.tool_harness.chat_tool_defs()
            messages = list(self.messages)
            final_text = ""

            for round_idx in range(self.max_rounds):
                if self._cancelled:
                    break

                rendered = qwen_render_messages(
                    messages, tools=defs, add_generation_prompt=True)

                with self._runtime.acquire(timeout_s=30.0) as engine:
                    raw = engine.generate(
                        rendered, max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        skip_special_tokens=False)

                if self._cancelled:
                    break

                tool_calls, content = qwen_parse_tool_calls(raw)
                final_text = content or ""
                messages.append({"role": "assistant", "content": content or ""})

                if not tool_calls:
                    # no tools — stream the final text to UI
                    self.chunk.emit(final_text)
                    break

                # execute tool calls
                for tc in tool_calls:
                    if self._cancelled:
                        break
                    call = tc if isinstance(tc, dict) else {"name": str(tc)}
                    self.tool_call_made.emit(call)
                    results = self.tool_harness.execute_calls([call])
                    rec = results[0] if results else {
                        "ok": False, "result": {"error": "no result"}}
                    self.tool_call_result.emit(rec)
                    # feed result back to model
                    messages.append({
                        "role": "tool",
                        "name": call.get("name", "tool"),
                        "content": _json.dumps(rec.get("result", rec),
                                                ensure_ascii=False),
                    })
            else:
                # ran out of rounds — emit whatever we have
                if final_text:
                    self.chunk.emit(final_text)

            self.done.emit(final_text)
        except RuntimeError as e:
            self.failed.emit(str(e))
        except Exception as e:
            logger.warning("chat tool worker failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _EndpointWorker(QThread):
    """Streams a chat completion from an OpenAI-compatible endpoint."""

    chunk = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, base, model, temp, messages, parent=None) -> None:
        super().__init__(parent)
        self.base, self.model, self.temp, self.messages = base, model, temp, messages
        self._cancelled = False

    def cancel(self) -> None:
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


# ── bubbles ────────────────────────────────────────────────────────────

class _ToolBlock(QFrame):
    """Collapsible tool-use block (like OpenAI/Anthropic platforms)."""

    def __init__(self, tool_name: str = "", tool_args: str = "",
                 tool_result: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("toolBlock")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        header = QHBoxLayout(); header.setSpacing(6)
        self._toggle = QToolButton()
        self._toggle.setText("\u25B6")
        self._toggle.setCheckable(True)
        self._toggle.setObjectName("toolToggle")
        self._label = QLabel(f"Tool: {tool_name}" if tool_name else "Tool use")
        self._label.setObjectName("toolLabel")
        header.addWidget(self._toggle)
        header.addWidget(self._label)
        header.addStretch(1)
        lay.addLayout(header)
        self._detail = QTextEdit()
        self._detail.setObjectName("toolDetail")
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(120)
        detail_text = ""
        if tool_args:
            detail_text += f"Args: {tool_args}\n"
        if tool_result:
            detail_text += f"Result: {tool_result}"
        self._detail.setPlainText(detail_text)
        self._detail.setVisible(False)
        lay.addWidget(self._detail)
        self._toggle.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        self._toggle.setText("\u25BC" if checked else "\u25B6")
        self._detail.setVisible(checked)

    def set_result(self, result: str) -> None:
        self._detail.setPlainText(self._detail.toPlainText() + f"\nResult: {result}")


class _ThinkingBlock(QFrame):
    """Collapsible thinking/reasoning block (like Claude/OpenAI o1)."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("thinkingBlock")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        header = QHBoxLayout(); header.setSpacing(6)
        self._toggle = QToolButton()
        self._toggle.setText("\u25B6")
        self._toggle.setCheckable(True)
        self._toggle.setObjectName("toolToggle")
        self._label = QLabel("Thinking\u2026")
        self._label.setObjectName("thinkingLabel")
        header.addWidget(self._toggle)
        header.addWidget(self._label)
        header.addStretch(1)
        lay.addLayout(header)
        self._body = QTextEdit()
        self._body.setObjectName("thinkingBody")
        self._body.setReadOnly(True)
        self._body.setPlainText(text)
        self._body.setMaximumHeight(200)
        self._body.setVisible(False)
        lay.addWidget(self._body)
        self._toggle.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        self._toggle.setText("\u25BC" if checked else "\u25B6")
        self._body.setVisible(checked)

    def append_text(self, text: str) -> None:
        self._body.setPlainText(self._body.toPlainText() + text)

    def finalize(self) -> None:
        self._label.setText("Thinking")


class _ToolBlock(QFrame):
    """Collapsible tool-call/result block (like Claude/other platforms)."""

    def __init__(self, tool_name: str, args_str: str = "",
                 result_str: str = "", status: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("toolBlock")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        # header row: tool name + status + toggle
        header = QHBoxLayout()
        if result_str:
            label_text = f"\U0001f9ea {tool_name} \u2192 {status}"
        else:
            label_text = f"\U0001f527 {tool_name}"
        self._label = QLabel(label_text)
        self._label.setObjectName("toolLabel")
        header.addWidget(self._label)
        header.addStretch(1)

        # toggle button to expand/collapse detail
        self._toggle = QToolButton()
        self._toggle.setText("\u25BC")
        self._toggle.setObjectName("toolToggle")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.clicked.connect(self._toggle_detail)
        header.addWidget(self._toggle)
        lay.addLayout(header)

        # detail area (args or result)
        self._detail = QTextEdit()
        self._detail.setObjectName("toolDetail")
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(120)
        detail_text = result_str if result_str else args_str
        self._detail.setPlainText(detail_text)
        self._detail.setVisible(False)
        lay.addWidget(self._detail)

    def _toggle_detail(self) -> None:
        self._detail.setVisible(self._toggle.isChecked())
        self._toggle.setText("\u25C0" if self._toggle.isChecked() else "\u25BC")


class _MessageBubble(QFrame):
    """One chat message with model-name label, optional image, tool/thinking blocks."""

    rated = Signal(int, str)   # msg_index, rating ('good'|'bad')

    def __init__(self, role: str, text: str, msg_index: int = -1,
                 rating: Optional[str] = None,
                 model_name: str = "",
                 image_path: str = "") -> None:
        super().__init__()
        self.role = role
        self.msg_index = msg_index
        self.setObjectName("bubbleUser" if role == "user" else
                           "bubbleSystem" if role == "system" else "bubbleAssistant")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)
        # Label: show model name for assistant, "User" for user, "System" for system
        if role == "assistant":
            label_text = model_name or "ForgeAI"
        elif role == "user":
            label_text = "User"
        else:
            label_text = "System"
        who = QLabel(label_text)
        who.setObjectName("bubbleWho")
        who.setStyleSheet(
            f"color:{Palette.accent if role == 'user' else Palette.chart_kl};")
        lay.addWidget(who)
        # Image (if provided — user-uploaded images in V2 Pro mode)
        if image_path:
            img_label = QLabel()
            pix = QPixmap(image_path)
            if not pix.isNull():
                if pix.width() > 300:
                    pix = pix.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation)
                img_label.setPixmap(pix)
            else:
                img_label.setText(f"[image: {os.path.basename(image_path)}]")
            img_label.setObjectName("bubbleImage")
            lay.addWidget(img_label)
        # Body text — use QTextEdit for selectable + scrollable long text
        body = QTextEdit(text)
        body.setObjectName("bubbleBody")
        body.setReadOnly(True)
        body.setMaximumHeight(400)
        body.setPlaceholderText("")
        lay.addWidget(body)
        # Tool/thinking blocks container (added dynamically by the page)
        self._blocks_lay = QVBoxLayout()
        self._blocks_lay.setSpacing(4)
        lay.addLayout(self._blocks_lay)
        if role == "assistant" and msg_index >= 0:
            row = QHBoxLayout(); row.setSpacing(6)
            self._good = QPushButton("good")
            self._good.setObjectName("rateGood")
            self._good.setCheckable(True)
            self._bad = QPushButton("bad")
            self._bad.setObjectName("rateBad")
            self._bad.setCheckable(True)
            self._good.clicked.connect(lambda: self._rate(RATING_GOOD))
            self._bad.clicked.connect(lambda: self._rate(RATING_BAD))
            hint = QLabel("good replies become training data")
            hint.setObjectName("chatMeta")
            row.addWidget(self._good); row.addWidget(self._bad)
            row.addStretch(1); row.addWidget(hint)
            lay.addLayout(row)
            self.set_rating(rating)

    def add_tool_block(self, tool_name: str, tool_args: str = "",
                       tool_result: str = "") -> _ToolBlock:
        block = _ToolBlock(tool_name, tool_args, tool_result)
        self._blocks_lay.addWidget(block)
        return block

    def add_thinking_block(self, text: str = "") -> _ThinkingBlock:
        block = _ThinkingBlock(text)
        self._blocks_lay.addWidget(block)
        return block

    def add_tool_block(self, tool_name: str, args_str: str) -> None:
        """Add a collapsible tool-call block to the bubble."""
        block = _ToolBlock(tool_name, args_str)
        self._blocks_lay.addWidget(block)

    def add_tool_result(self, tool_name: str, status: str,
                        result_str: str) -> None:
        """Add a tool result block to the bubble."""
        block = _ToolBlock(tool_name, "", result_str, status)
        self._blocks_lay.addWidget(block)

    def _rate(self, which: str) -> None:
        self.rated.emit(self.msg_index, which)

    def set_rating(self, rating: Optional[str]) -> None:
        self._good.setChecked(rating == RATING_GOOD)
        self._bad.setChecked(rating == RATING_BAD)

    def set_text(self, text: str) -> None:
        for child in self.findChildren(QTextEdit):
            if child.objectName() == "bubbleBody":
                child.setPlainText(text)
                break


# ── page ───────────────────────────────────────────────────────────────

class ChatPage(QWidget):
    def __init__(self, store: Optional[ChatStore] = None,
                 runtime: Optional[EngineRuntime] = None,
                 models_index=None, lorebook=None, lora_harness=None,
                 tool_harness=None,
                 base_url: str = "http://localhost:8080/v1",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store or ChatStore()
        self.runtime = runtime
        self.models_index = models_index
        self.lorebook = lorebook
        self.lora_harness = lora_harness
        self.tool_harness = tool_harness
        self._base = base_url.rstrip("/")
        self._worker: Optional[QThread] = None
        self._conv_id: Optional[str] = None
        self._pending_msg_idx = -1
        self._assistant_text = ""
        self._assistant_bubble: Optional[_MessageBubble] = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("pages")
        outer.addWidget(splitter)

        # ── left: conversations ───────────────────────────────────────
        left = QFrame(); left.setObjectName("root")
        ll = QVBoxLayout(left); ll.setContentsMargins(16, 18, 8, 12)
        ll.setSpacing(8)
        ll.addWidget(section_label("Conversations"))
        self._conv_list = QListWidget()
        self._conv_list.setObjectName("convList")
        self._conv_list.currentRowChanged.connect(self._on_conv_selected)
        ll.addWidget(self._conv_list, 1)
        new_btn = QPushButton("+ New chat")
        new_btn.setObjectName("primary")
        new_btn.clicked.connect(self._new_chat)
        ll.addWidget(new_btn)
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self._delete_chat)
        ll.addWidget(del_btn)
        ll.addSpacing(8)
        ll.addWidget(section_label("Training data"))
        self._export_info = QLabel("rate replies 👍 → export as SFT data")
        self._export_info.setObjectName("chatMeta")
        self._export_info.setWordWrap(True)
        ll.addWidget(self._export_info)
        export_btn = QPushButton("Export rated → JSONL")
        export_btn.clicked.connect(self._export)
        ll.addWidget(export_btn)
        splitter.addWidget(left)

        # ── center: conversation ──────────────────────────────────────
        center = QFrame(); center.setObjectName("root")
        cl = QVBoxLayout(center); cl.setContentsMargins(8, 18, 8, 12)
        cl.setSpacing(8)
        self._conv_host = QWidget(); self._conv_host.setObjectName("root")
        self._conv_lay = QVBoxLayout(self._conv_host)
        self._conv_lay.setContentsMargins(0, 0, 0, 0)
        self._conv_lay.setSpacing(12)
        self._conv_lay.addStretch(1)
        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setObjectName("root")
        self._scroll.setWidget(self._conv_host)
        cl.addWidget(self._scroll, 1)

        comp = QFrame(); comp.setObjectName("card")
        cpl = QVBoxLayout(comp); cpl.setContentsMargins(12, 12, 12, 12)
        cpl.setSpacing(8)
        self._input = QPlainTextEdit()
        self._input.setPlaceholderText(
            "Message ForgeAI…  (Enter to send, Shift+Enter for newline)")
        self._input.setFixedHeight(76)
        cpl.addWidget(self._input)
        btn_row = QHBoxLayout()
        self._image_btn = QPushButton("+ Image")
        self._image_btn.setToolTip("Attach an image (V2 Pro only — requires multimodal model)")
        self._image_btn.clicked.connect(self._on_attach_image)
        btn_row.addWidget(self._image_btn)
        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("primary")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._send_btn); btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        self._status = QLabel("Idle")
        self._status.setObjectName("chatMeta")
        btn_row.addWidget(self._status)
        cpl.addLayout(btn_row)
        cl.addWidget(comp)
        splitter.addWidget(center)

        # ── right: inference settings ─────────────────────────────────
        right = QFrame(); right.setObjectName("root")
        rl = QVBoxLayout(right); rl.setContentsMargins(8, 18, 16, 12)
        rl.setSpacing(8)
        rl.addWidget(section_label("Inference"))
        self._source = QComboBox()
        self._source.addItems(["Local ForgeEngine", "HTTP endpoint"])
        rl.addWidget(self._source)

        self._model_label = section_label("Checkpoint")
        rl.addWidget(self._model_label)
        self._model = QComboBox()
        # Defer model scan to after window show — _reload_models() triggers
        # models_index.models() which scans research/checkpoints/.
        QTimer.singleShot(0, self._reload_models)
        rl.addWidget(self._model)

        self._endpoint_label = section_label("Endpoint URL")
        rl.addWidget(self._endpoint_label)
        self._endpoint = QLineEdit(self._base)
        rl.addWidget(self._endpoint)
        self._endpoint_model_label = section_label("Endpoint model")
        rl.addWidget(self._endpoint_model_label)
        self._endpoint_model = QLineEdit("forgelm-v10")
        rl.addWidget(self._endpoint_model)

        rl.addWidget(section_label("System prompt"))
        self._use_master = QCheckBox("Auto master prompt (model identity)")
        self._use_master.setToolTip(
            "Prepend a master system prompt with the model's name, architecture, "
            "capabilities, and guidelines. Adapts per model (V2 Light = text-only, "
            "V2 Pro = multimodal).")
        self._use_master.setChecked(True)
        rl.addWidget(self._use_master)
        self._system = QPlainTextEdit()
        self._system.setPlaceholderText("Custom system prompt (added on top of master)…")
        self._system.setFixedHeight(90)
        rl.addWidget(self._system)

        rl.addWidget(section_label("Sampling"))
        grid = QHBoxLayout()
        grid.addWidget(QLabel("Temp"))
        self._temp = QDoubleSpinBox()
        self._temp.setRange(0.0, 2.0); self._temp.setSingleStep(0.05)
        self._temp.setValue(0.7)
        grid.addWidget(self._temp)
        grid.addWidget(QLabel("Top-p"))
        self._top_p = QDoubleSpinBox()
        self._top_p.setRange(0.0, 1.0); self._top_p.setSingleStep(0.05)
        self._top_p.setValue(0.95)
        grid.addWidget(self._top_p)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Top-k"))
        self._top_k = QSpinBox()
        self._top_k.setRange(1, 200); self._top_k.setValue(80)
        row2.addWidget(self._top_k)
        row2.addWidget(QLabel("Max tok"))
        self._max_tok = QSpinBox()
        self._max_tok.setRange(16, 8192); self._max_tok.setValue(512)
        row2.addWidget(self._max_tok)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Rep pen"))
        self._rep_pen = QDoubleSpinBox()
        self._rep_pen.setRange(1.0, 2.0); self._rep_pen.setSingleStep(0.01)
        self._rep_pen.setValue(1.05)
        row3.addWidget(self._rep_pen)
        row3.addStretch(1)
        self._thinking = QCheckBox("Thinking mode")
        self._thinking.setToolTip(
            "Prepend a chain-of-thought instruction to the system prompt — "
            "the model reasons step-by-step before answering.")
        row3.addWidget(self._thinking)
        rl.addLayout(grid); rl.addLayout(row2); rl.addLayout(row3)

        # tool mode toggle
        self._tools_enabled = QCheckBox("Enable tools (memory + LoRA)")
        self._tools_enabled.setToolTip(
            "When enabled, the model can use tools: remember/recall_memory/"
            "forget for long-term memory, and load_lora/unload_lora/list_loras "
            "for skill specialization. Read-only file tools (list_dir, "
            "read_file, grep_project) are also available.")
        self._tools_enabled.setChecked(True)
        rl.addWidget(self._tools_enabled)

        # LoRA mode indicator
        self._lora_status = QLabel("LoRA: auto (chat_assist)")
        self._lora_status.setObjectName("chatMeta")
        self._lora_status.setStyleSheet("color:#8b96a8; font-size:11px;")
        rl.addWidget(self._lora_status)

        self._engine_state = QLabel("engine: —")
        self._engine_state.setObjectName("chatMeta")
        rl.addWidget(self._engine_state)
        rl.addStretch(1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 640, 280])

        # signals
        self._send_btn.clicked.connect(self._send)
        self._stop_btn.clicked.connect(self._stop)
        self._input.installEventFilter(self)
        self._source.currentIndexChanged.connect(self._on_source_changed)
        self._on_source_changed()

        if runtime is not None:
            runtime.state_changed.connect(self._on_engine_state)
            self._on_engine_state(runtime.state)
        self._reload_conversations()

    # ── conversations ─────────────────────────────────────────────────
    def _reload_conversations(self) -> None:
        self._conv_list.blockSignals(True)
        self._conv_list.clear()
        for conv in self.store.conversations:
            good, bad = ChatStore.count_ratings(conv)
            label = f"{conv['title']}\n{len(conv['messages'])} msgs"
            if good or bad:
                label += f"   ★{good} ✗{bad}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, conv["id"])
            self._conv_list.addItem(item)
        self._conv_list.blockSignals(False)
        if self.store.conversations and self._conv_id is None:
            self._conv_list.setCurrentRow(0)
        elif self._conv_id is not None:
            for i in range(self._conv_list.count()):
                if (self._conv_list.item(i).data(Qt.ItemDataRole.UserRole)
                        == self._conv_id):
                    self._conv_list.setCurrentRow(i)
                    break

    def _on_conv_selected(self, row: int) -> None:
        item = self._conv_list.item(row)
        if item is None:
            self._conv_id = None
            return
        self._conv_id = item.data(Qt.ItemDataRole.UserRole)
        self._render_conversation()

    def _new_chat(self) -> None:
        model = (self._model.currentData() or "") if self._model.count() else ""
        conv = self.store.create(model=model)
        self._conv_id = conv["id"]
        self._reload_conversations()
        self._input.setFocus()

    def _delete_chat(self) -> None:
        if self._conv_id is not None:
            self.store.delete(self._conv_id)
            self._conv_id = None
            self._reload_conversations()
            self._render_conversation()

    def _conv(self) -> Optional[dict]:
        return self.store.get(self._conv_id) if self._conv_id else None

    # ── rendering ─────────────────────────────────────────────────────
    def _render_conversation(self) -> None:
        while self._conv_lay.count() > 1:
            it = self._conv_lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        conv = self._conv()
        if conv is None:
            return
        model_name = self._get_model_name()
        for i, m in enumerate(conv["messages"]):
            if m.get("role") == "system":
                continue
            bubble = _MessageBubble(
                m["role"], m.get("content", ""),
                msg_index=i if m["role"] == "assistant" else -1,
                rating=m.get("rating"),
                model_name=model_name if m["role"] == "assistant" else "",
                image_path=m.get("image", ""))
            bubble.rated.connect(self._on_rated)
            self._conv_lay.insertWidget(self._conv_lay.count() - 1, bubble)
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _add_bubble(self, role: str, text: str,
                    msg_index: int = -1,
                    rating: Optional[str] = None,
                    image_path: str = "") -> _MessageBubble:
        model_name = self._get_model_name() if role == "assistant" else ""
        b = _MessageBubble(role, text, msg_index, rating,
                           model_name=model_name, image_path=image_path)
        b.rated.connect(self._on_rated)
        self._conv_lay.insertWidget(self._conv_lay.count() - 1, b)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return b

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ── ratings ───────────────────────────────────────────────────────
    def _on_rated(self, msg_index: int, which: str) -> None:
        if self._conv_id is None:
            return
        new = self.store.rate_message(self._conv_id, msg_index, which)
        # refresh checked states on all bubbles of this conversation
        conv = self._conv()
        if conv is None:
            return
        for bubble in self._bubbles():
            if bubble.msg_index >= 0:
                m = conv["messages"][bubble.msg_index]
                bubble.set_rating(m.get("rating"))
        good, bad = ChatStore.count_ratings(conv)
        self._export_info.setText(
            f"{good} good / {bad} bad in this chat · good turns → SFT data")
        self._reload_conversations()
        self._status.setText(f"rated {new}" if new else "rating cleared")

    def _bubbles(self) -> list[_MessageBubble]:
        out = []
        for i in range(self._conv_lay.count()):
            w = self._conv_lay.itemAt(i).widget()
            if isinstance(w, _MessageBubble):
                out.append(w)
        return out

    # ── export ────────────────────────────────────────────────────────
    def _export(self) -> None:
        try:
            path, n = self.store.export_training_data()
            self._export_info.setText(f"exported {n} examples →\n{path}")
            self._status.setText(f"exported {n} examples")
        except Exception as e:
            self._status.setText(f"export failed: {e}")

    # ── sending / streaming ───────────────────────────────────────────
    def eventFilter(self, obj, ev):
        if obj is self._input and ev.type() == QEvent.Type.KeyPress:
            k: QKeyEvent = ev
            if (k.key() == Qt.Key.Key_Return
                    and not (k.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._send()
                return True
        return super().eventFilter(obj, ev)

    def _get_model_name(self) -> str:
        """Get the display name of the currently selected/loaded model."""
        if self.runtime is not None and self.runtime.is_ready():
            info = self.runtime.info
            cfg_name = info.get("config_name", "")
            if cfg_name:
                from ..api.master_prompt import _detect_model_name
                return _detect_model_name(cfg_name)
        if self._model.count() > 0:
            name = self._model.currentText()
            if "V2_Pro" in name or "v2_pro" in name:
                return "ForgeLM V2 Pro"
            if "V2_Light" in name or "v2_light" in name:
                return "ForgeLM V2 Light"
        return "ForgeAI"

    def _is_pro_model(self) -> bool:
        """Check if the loaded model is V2 Pro (multimodal)."""
        if self.runtime is not None and self.runtime.is_ready():
            info = self.runtime.info
            cfg_name = info.get("config_name", "")
            return "v2_pro" in cfg_name or "v11" in cfg_name
        name = self._get_model_name()
        return "Pro" in name

    def _build_master_prompt(self) -> str:
        """Build the master system prompt for the current model."""
        parts = []
        if self._use_master.isChecked():
            try:
                if self.runtime is not None and self.runtime.is_ready():
                    info = self.runtime.info
                    cfg_name = info.get("config_name", "")
                    if cfg_name:
                        from research.config import get_config
                        cfg = get_config(cfg_name)
                        parts.append(generate_master_prompt(
                            cfg, cfg_name,
                            tools_enabled=self._tools_enabled.isChecked(),
                            thinking_enabled=self._thinking.isChecked()))
                else:
                    name = self._model.currentText() if self._model.count() else ""
                    cfg_name = "forgelm_v2_light"
                    if "Pro" in name or "pro" in name.lower():
                        cfg_name = "forgelm_v2_pro"
                    parts.append(get_default_prompt_for_config(
                        cfg_name,
                        tools_enabled=self._tools_enabled.isChecked(),
                        thinking_enabled=self._thinking.isChecked()))
            except Exception as e:
                logger.warning("master prompt generation failed: %s", e)
        custom = self._system.toPlainText().strip()
        if custom:
            parts.append(custom)
        return "\n\n".join(parts)

    def _history_for_request(self) -> list[dict]:
        conv = self._conv()
        msgs: list[dict] = []
        sys_prompt = self._build_master_prompt()
        # inject lorebook memory (hybrid: constant + keyword-triggered)
        if self.lorebook is not None:
            recent = []
            if conv:
                recent = [{"role": m["role"], "content": m["content"]}
                          for m in conv["messages"]
                          if m.get("role") in ("user", "assistant")]
            lore = self.lorebook.inject(recent)
            if lore:
                sys_prompt = lore + "\n\n" + sys_prompt if sys_prompt else lore
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        if conv:
            msgs.extend({"role": m["role"], "content": m["content"]}
                        for m in conv["messages"]
                        if m.get("role") in ("user", "assistant"))
        return msgs

    def _switch_lora_mode(self, mode: str) -> None:
        """Switch the LoRA harness to a mode (chat/agent/self_play)."""
        if self.lora_harness is not None:
            self.lora_harness.set_mode(mode)
            current = self.lora_harness.current_adapter
            if current:
                from pathlib import Path
                self._lora_status.setText("LoRA: " + Path(current).name)
            else:
                self._lora_status.setText("LoRA: auto (" + mode + ")")

    def _on_attach_image(self) -> None:
        """Open file dialog to attach an image."""
        if not self._is_pro_model():
            self._status.setText("Image upload requires V2 Pro (multimodal model)")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if path:
            self._pending_image = path
            self._status.setText(f"Image attached: {os.path.basename(path)}")

    def _send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text or self._worker is not None:
            return
        if self._conv_id is None:
            self._new_chat()
            if self._conv_id is None:
                return
        # check for pending image attachment
        image_path = getattr(self, "_pending_image", "")
        self._pending_image = ""
        self._input.clear()
        self._add_bubble("user", text, image_path=image_path)
        self.store.append_message(self._conv_id, "user", text, image=image_path)
        self._reload_conversations()
        self._send_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._status.setText("Sending…")
        self._assistant_text = ""
        self._assistant_bubble = self._add_bubble("assistant", "…")
        # add thinking block if thinking mode is on
        if self._thinking.isChecked():
            self._thinking_block = self._assistant_bubble.add_thinking_block("")

        history = self._history_for_request()
        if self._source.currentIndex() == 0:
            if self.runtime is None or not self.runtime.is_ready():
                self._on_fail("Local engine not loaded — open the Engine page "
                              "and click Load (fast load ≈ 15-30s), or switch "
                              "to HTTP endpoint.")
                return
            if self._tools_enabled.isChecked() and self.tool_harness is not None:
                # use tool-aware worker that can call tools
                self._worker = _EngineChatToolWorker(
                    self.runtime, history, self.tool_harness,
                    self._max_tok.value(), self._temp.value(),
                    self._top_p.value(), self._top_k.value(),
                    self._rep_pen.value())
                self._worker.tool_call_made.connect(self._on_tool_call)
                self._worker.tool_call_result.connect(self._on_tool_result)
            else:
                self._worker = _EngineChatWorker(
                    self.runtime, history, self._max_tok.value(),
                    self._temp.value(), self._top_p.value(), self._top_k.value(),
                    self._rep_pen.value())
        else:
            self._worker = _EndpointWorker(
                self._endpoint.text() or self._base,
                self._endpoint_model.text(), self._temp.value(), history)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_chunk(self, delta: str) -> None:
        self._assistant_text += delta
        if self._assistant_bubble is not None:
            display_text = self._assistant_text
            thinking_block = getattr(self, "_thinking_block", None)
            if thinking_block is not None and "<think>" in display_text:
                # parse thinking tags — show in thinking block, not body
                think_match = re.search(r"<think>(.*?)(</think>|$)",
                                        display_text, re.DOTALL)
                if think_match:
                    thinking_content = think_match.group(1).strip()
                    thinking_block._body.setPlainText(thinking_content)
                    if "</think>" in display_text:
                        thinking_block.finalize()
                        after = display_text.split("</think>")[-1].strip()
                        self._assistant_bubble.set_text(after)
                    else:
                        self._assistant_bubble.set_text("")
                else:
                    self._assistant_bubble.set_text(display_text)
            else:
                self._assistant_bubble.set_text(display_text)
        self._scroll_to_bottom()

    def _on_done(self, _final: str) -> None:
        self._finalize_reply()

    def _on_fail(self, err: str) -> None:
        if self._assistant_bubble is not None:
            self._assistant_bubble.set_text(f"(error) {err}")
        self._send_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._worker = None
        self._status.setText(f"error: {err[:120]}")

    def _on_tool_call(self, call: dict) -> None:
        """Show a tool call card in the assistant bubble."""
        if self._assistant_bubble is not None:
            name = call.get("name", "tool")
            args = call.get("arguments", {})
            import json as _json
            args_str = _json.dumps(args, ensure_ascii=False)[:200]
            self._assistant_bubble.add_tool_block(name, args_str)
        self._status.setText(f"tool: {call.get('name', '?')}…")

    def _on_tool_result(self, rec: dict) -> None:
        """Show tool result in the assistant bubble."""
        if self._assistant_bubble is not None:
            name = rec.get("name", "tool")
            ok = rec.get("ok", False)
            result = rec.get("result", {})
            import json as _json
            result_str = _json.dumps(result, ensure_ascii=False)[:300]
            status = "ok" if ok else "error"
            self._assistant_bubble.add_tool_result(name, status, result_str)
        self._status.setText("tool done")

    def _finalize_reply(self) -> None:
        if self._conv_id is not None and self._assistant_text:
            idx = self.store.append_message(
                self._conv_id, "assistant", self._assistant_text)
            if self._assistant_bubble is not None:
                self._assistant_bubble.msg_index = idx
        self._reload_conversations()
        self._send_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._worker = None
        self._status.setText("done · rate the reply 👍/👎")

    def _stop(self) -> None:
        w = self._worker
        if w is not None and w.isRunning():
            w.cancel()
            if not w.wait(2000):
                logger.warning("chat worker did not stop in 2s — terminating")
                w.terminate(); w.wait(1000)
        self._finalize_reply()
        self._status.setText("stopped")

    # ── settings ──────────────────────────────────────────────────────
    def _reload_models(self) -> None:
        self._model.clear()
        if self.models_index is not None:
            try:
                for m in self.models_index.models():
                    if "lora" in m.name.lower():
                        continue
                    self._model.addItem(m.name, m.path)
            except Exception as e:
                logger.warning("model list failed: %s", e)
        if self._model.count() == 0:
            self._model.addItem("ForgeLM_V2_Light.safetensors",
                                "research/checkpoints/ForgeLM_V2_Light.safetensors")

    def _on_source_changed(self) -> None:
        local = self._source.currentIndex() == 0
        self._model.setVisible(local)
        self._model_label.setVisible(local)
        self._endpoint.setVisible(not local)
        self._endpoint_label.setVisible(not local)
        self._endpoint_model.setVisible(not local)
        self._endpoint_model_label.setVisible(not local)

    def _on_engine_state(self, state: str) -> None:
        color = {"ready": Palette.ok, "loading": Palette.warn,
                 "error": Palette.err}.get(state, Palette.text_faint)
        extra = ""
        if self.runtime is not None and state == "ready":
            info = self.runtime.info
            extra = f" · {info.get('config_name', '')}"
            self._select_resident_model(info.get("checkpoint", ""))
        self._engine_state.setText(f"engine: {state}{extra}")
        self._engine_state.setStyleSheet(f"color: {color};")

    def _select_resident_model(self, checkpoint: str) -> None:
        """Point the model combo at the checkpoint that is actually loaded."""
        if not checkpoint:
            return
        for i in range(self._model.count()):
            data = self._model.itemData(i) or ""
            if data and (data == checkpoint
                         or checkpoint.endswith(str(data).replace("/", "\\"))
                         or checkpoint.endswith(str(data))):
                if self._model.currentIndex() != i:
                    self._model.blockSignals(True)
                    self._model.setCurrentIndex(i)
                    self._model.blockSignals(False)
                return

    def refresh(self) -> None:
        # always reflect the live runtime state (covers missed signals)
        if self.runtime is not None:
            self._on_engine_state(self.runtime.state)
        # set LoRA harness to chat mode on first refresh
        if self.lora_harness is not None and self.lora_harness.mode != "chat":
            self._switch_lora_mode("chat")
