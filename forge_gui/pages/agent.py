"""Agent page — agentic coding platform over the resident ForgeEngine.

Layout:
  ┌──────────────────────────────────────────────┬──────────────┐
  │  live trace (rounds → tool calls → results)  │ task config  │
  ├──────────────────────────────────────────────┴──────────────┤
  │  task composer (workspace, prompt, run/stop, approval bar)  │
  └─────────────────────────────────────────────────────────────┘

The runner (AgentRunner) streams each round: assistant text, tool-call
cards with arguments, sandbox results. Side-effecting tools can require
one-click approval. Run history is kept per session.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..api.agent_runner import DEFAULT_SYSTEM, AgentRunner
from ..api.engine_runtime import EngineRuntime
from ..theme import Palette
from ._base import section_label

logger = logging.getLogger(__name__)

TOOL_NAMES = ["list_dir", "read_file", "write_file", "append_file",
              "delete_file", "run_python", "run_cmd", "grep_project"]


class _ToolCard(QFrame):
    """Tool call card: name + args + result (result filled in later)."""

    def __init__(self, call: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolCard")
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        head = QHBoxLayout()
        name = QLabel(f"🔧 {call.get('name', '?')}()")
        name.setObjectName("toolName")
        head.addWidget(name)
        self._state = QLabel("running…")
        self._state.setObjectName("toolPending")
        head.addStretch(1); head.addWidget(self._state)
        lay.addLayout(head)
        args = call.get("arguments") or {}
        args_edit = QPlainTextEdit()
        args_edit.setObjectName("toolArgs")
        args_edit.setReadOnly(True)
        try:
            args_edit.setPlainText(json.dumps(args, ensure_ascii=False, indent=1))
        except (TypeError, ValueError):
            args_edit.setPlainText(str(args))
        args_edit.setFixedHeight(min(110, 24 + 14 * max(1, len(args))))
        lay.addWidget(args_edit)
        self._result = QPlainTextEdit()
        self._result.setObjectName("toolResult")
        self._result.setReadOnly(True)
        self._result.setVisible(False)
        lay.addWidget(self._result)

    def set_result(self, rec: dict) -> None:
        ok = rec.get("ok", False)
        self._state.setText("ok" if ok else "error")
        self._state.setObjectName("toolOk" if ok else "toolErr")
        self._state.setStyleSheet(
            f"color: {Palette.ok if ok else Palette.err};")
        res = rec.get("result", {})
        try:
            text = json.dumps(res, ensure_ascii=False, indent=1)
        except (TypeError, ValueError):
            text = str(res)
        self._result.setPlainText(text[:4000])
        self._result.setFixedHeight(min(180, 24 + 13 * max(1, text.count("\n") + 1)))
        self._result.setVisible(True)


class _RoundBlock(QFrame):
    """One agent round: header + assistant text + tool cards."""

    def __init__(self, round_idx: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("agentRound")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(12, 10, 12, 10)
        self.lay.setSpacing(8)
        head = QLabel(f"ROUND {round_idx + 1}")
        head.setObjectName("agentRoundHead")
        self.lay.addWidget(head)
        self.text = QLabel("")
        self.text.setObjectName("bubbleBody")
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lay.addWidget(self.text)

    def add_tool_card(self, card: _ToolCard) -> None:
        self.lay.addWidget(card)


class AgentPage(QWidget):
    def __init__(self, runtime: EngineRuntime,
                 tool_harness=None, lorebook=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.tool_harness = tool_harness
        self.lorebook = lorebook
        self._runner: Optional[AgentRunner] = None
        self._rounds: dict[str, _RoundBlock] = {}
        self._n_runs = 0

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("pages")
        outer.addWidget(splitter)

        # ── trace area ────────────────────────────────────────────────
        self._trace_host = QWidget(); self._trace_host.setObjectName("root")
        self._trace_lay = QVBoxLayout(self._trace_host)
        self._trace_lay.setContentsMargins(20, 16, 20, 8)
        self._trace_lay.setSpacing(10)
        self._trace_lay.addStretch(1)
        self._trace = QScrollArea(); self._trace.setWidgetResizable(True)
        self._trace.setFrameShape(QFrame.Shape.NoFrame)
        self._trace.setObjectName("root")
        self._trace.setWidget(self._trace_host)
        splitter.addWidget(self._trace)

        # ── bottom: composer + config ─────────────────────────────────
        bottom = QWidget(); bottom.setObjectName("root")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(20, 8, 20, 14)
        bl.setSpacing(12)

        # task column
        task_col = QVBoxLayout(); task_col.setSpacing(6)
        ws_row = QHBoxLayout()
        ws_row.addWidget(section_label("Workspace"))
        self._workspace = QLineEdit(str(self._default_workspace()))
        self._workspace.setMinimumWidth(280)
        browse = QPushButton("…")
        browse.setFixedWidth(36)
        browse.clicked.connect(self._pick_workspace)
        ws_row.addWidget(self._workspace, 1); ws_row.addWidget(browse)
        task_col.addLayout(ws_row)
        self._task = QPlainTextEdit()
        self._task.setPlaceholderText(
            "Describe the coding task… e.g. 'fix the failing test in tests/'")
        self._task.setFixedHeight(72)
        task_col.addWidget(self._task)
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶ Run agent")
        self._run_btn.setObjectName("primary")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn); btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        self._status = QLabel("Idle · engine must be loaded (Engine Console)")
        self._status.setObjectName("chatMeta")
        btn_row.addWidget(self._status)
        task_col.addLayout(btn_row)
        bl.addLayout(task_col, 1)

        # config column
        cfg = QFrame(); cfg.setObjectName("card")
        cg = QVBoxLayout(cfg); cg.setContentsMargins(14, 12, 14, 12)
        cg.setSpacing(6)
        cg.addWidget(section_label("Agent config"))
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Rounds"))
        self._rounds = QSpinBox()
        self._rounds.setRange(1, 32); self._rounds.setValue(8)
        row1.addWidget(self._rounds)
        row1.addWidget(QLabel("Max tok"))
        self._max_tok = QSpinBox()
        self._max_tok.setRange(64, 4096); self._max_tok.setValue(512)
        row1.addWidget(self._max_tok)
        cg.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Temp"))
        self._temp = QDoubleSpinBox()
        self._temp.setRange(0.0, 2.0); self._temp.setSingleStep(0.05)
        self._temp.setValue(0.2)
        row2.addWidget(self._temp)
        row2.addWidget(QLabel("Rep pen"))
        self._rep_pen = QDoubleSpinBox()
        self._rep_pen.setRange(1.0, 2.0); self._rep_pen.setSingleStep(0.01)
        self._rep_pen.setValue(1.05)
        row2.addWidget(self._rep_pen)
        cg.addLayout(row2)
        self._approval = QCheckBox("Approve writes & commands")
        self._approval.setChecked(True)
        cg.addWidget(self._approval)
        self._thinking = QCheckBox("Thinking mode")
        self._thinking.setToolTip(
            "Prepend a chain-of-thought instruction to the system prompt — "
            "the agent reasons step-by-step before acting.")
        cg.addWidget(self._thinking)
        cg.addWidget(section_label("Tools"))
        self._tool_checks: dict[str, QCheckBox] = {}
        tool_grid = QVBoxLayout(); tool_grid.setSpacing(2)
        for i, t in enumerate(TOOL_NAMES):
            cb = QCheckBox(t)
            cb.setChecked(t not in ("delete_file",))
            self._tool_checks[t] = cb
            tool_grid.addWidget(cb)
        cg.addLayout(tool_grid)
        bl.addWidget(cfg)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0)
        splitter.setSizes([560, 300])

        self._run_btn.clicked.connect(self._run)
        self._stop_btn.clicked.connect(self._stop)
        if self.runtime is not None:
            self.runtime.state_changed.connect(self._on_engine_state)
            self._on_engine_state(self.runtime.state)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _default_workspace() -> str:
        try:
            return os.path.join(str(__import__("pathlib").Path.home()), "Projects")
        except Exception:
            return os.getcwd()

    def _pick_workspace(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select workspace",
                                             self._workspace.text() or ".")
        if d:
            self._workspace.setText(d)

    def _on_engine_state(self, state: str) -> None:
        if state == "ready":
            self._status.setText("Engine ready — describe a task and run")
        elif state == "loading":
            self._status.setText("Engine loading…")
        else:
            self._status.setText("Idle · engine must be loaded (Engine Console)")

    # ── run / stop ────────────────────────────────────────────────────
    def _run(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            return
        task = self._task.toPlainText().strip()
        if not task:
            self._status.setText("describe a task first")
            return
        if self.runtime is None or not self.runtime.is_ready():
            QMessageBox.information(
                self, "Engine not loaded",
                "Load a checkpoint in the Engine Console first\n"
                "(the agent runs the resident ForgeEngine).")
            return
        workspace = self._workspace.text().strip() or "."
        tools = [t for t, cb in self._tool_checks.items() if cb.isChecked()]
        sys_prompt = DEFAULT_SYSTEM
        if self._thinking.isChecked():
            _open, _close = "<think>", "<" + "/think>"
            sys_prompt = ("Think step-by-step before acting. Show your "
                          "reasoning inside " + _open + "..." + _close +
                          ", then act.\n\n" + DEFAULT_SYSTEM)
        self._n_runs += 1
        self._clear_trace()
        self._status.setText("running…")
        self._run_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._runner = AgentRunner(
            self.runtime, task, workspace,
            system_prompt=sys_prompt,
            max_rounds=self._rounds.value(),
            max_new_tokens=self._max_tok.value(),
            temperature=self._temp.value(),
            repetition_penalty=self._rep_pen.value(),
            enabled_tools=tools,
            require_approval=self._approval.isChecked(),
            tool_harness=self.tool_harness,
        )
        self._runner.round_started.connect(self._on_round)
        self._runner.text.connect(self._on_text)
        self._runner.tool_call.connect(self._on_tool_call)
        self._runner.tool_result.connect(self._on_tool_result)
        self._runner.approval_requested.connect(self._on_approval)
        self._runner.finished_ok.connect(self._on_finished)
        self._runner.failed.connect(self._on_failed)
        self._runner.start()

    def _stop(self) -> None:
        r = self._runner
        if r is not None and r.isRunning():
            r.cancel()
            self._status.setText("stopping…")

    # ── runner slots ──────────────────────────────────────────────────
    def _on_round(self, round_idx: str) -> None:
        block = _RoundBlock(int(round_idx))
        self._rounds[round_idx] = block
        self._trace_lay.insertWidget(self._trace_lay.count() - 1, block)
        self._scroll_bottom()

    def _on_text(self, round_idx: str, content: str) -> None:
        block = self._rounds.get(round_idx)
        if block is not None:
            block.text.setText(content or "(no text)")
            self._scroll_bottom()

    def _on_tool_call(self, round_idx: str, call: dict) -> None:
        block = self._rounds.get(round_idx)
        if block is None:
            self._on_round(round_idx)
            block = self._rounds[round_idx]
        card = _ToolCard(call)
        block.add_tool_card(card)
        self._scroll_bottom()

    def _on_tool_result(self, round_idx: str, rec: dict) -> None:
        block = self._rounds.get(round_idx)
        if block is None:
            return
        cards = block.findChildren(_ToolCard)
        if cards:
            cards[-1].set_result(rec)
        self._scroll_bottom()

    def _on_approval(self, round_idx: str, call: dict) -> None:
        r = self._runner
        if r is None:
            return
        ret = QMessageBox.question(
            self, "Agent approval",
            f"Agent wants to execute:\n\n{call.get('name')}\n"
            f"{json.dumps(call.get('arguments', {}), indent=1)[:800]}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        r.respond_approval(ret == QMessageBox.StandardButton.Yes)

    def _on_finished(self, result: dict) -> None:
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        summary = (result.get("content") or "").strip()
        n_calls = len(result.get("tool_calls", []))
        tail = (f"finished · {result.get('rounds', 0)} rounds · "
                f"{n_calls} tool calls · {result.get('elapsed_s', 0)}s")
        if result.get("cancelled"):
            tail += " · cancelled"
        self._status.setText(tail)
        if summary:
            final = QLabel(summary)
            final.setObjectName("bubbleBody")
            final.setWordWrap(True)
            self._trace_lay.insertWidget(self._trace_lay.count() - 1, final)
            self._scroll_bottom()

    def _on_failed(self, err: str) -> None:
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._status.setText(f"failed: {err[:140]}")

    # ── trace helpers ─────────────────────────────────────────────────
    def _clear_trace(self) -> None:
        self._rounds.clear()
        while self._trace_lay.count() > 1:
            it = self._trace_lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

    def _scroll_bottom(self) -> None:
        bar = self._trace.verticalScrollBar()
        bar.setValue(bar.maximum())

    def refresh(self) -> None:
        # switch LoRA harness to agent mode
        if self.tool_harness is not None and hasattr(self.tool_harness, "lora_harness"):
            lh = self.tool_harness.lora_harness
            if lh is not None and lh.mode != "agent":
                lh.set_mode("agent")
