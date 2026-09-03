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
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QToolButton, QVBoxLayout,
                               QWidget)

from ..api.agent_runner import (APPROVAL_ALL, APPROVAL_DESTRUCTIVE,
                                APPROVAL_NONE, DEFAULT_SYSTEM, AgentRunner)
from ..api.engine_runtime import EngineRuntime
from ..theme import Palette
from ._base import section_label

logger = logging.getLogger(__name__)

# ── tool categories ──────────────────────────────────────────────────────
# Tools grouped by function for the config panel. Each category has a
# "select all" toggle and individual checkboxes. Defaults are chosen so
# the agent can explore + edit + run code without popping up dialogs
# for every basic operation.

TOOL_CATEGORIES: dict[str, dict] = {
    "File I/O": {
        "tools": ["list_dir", "read_file", "write_file", "append_file",
                  "create_file", "rename_file", "delete_file", "file_info",
                  "dir_tree", "create_dir"],
        "default_off": ["delete_file", "rename_file", "dir_tree",
                        "file_info", "create_dir"],
    },
    "Code Intelligence": {
        "tools": ["grep_project", "find_references", "find_definitions",
                  "find_todos", "line_count", "syntax_check"],
        "default_off": ["find_references", "find_definitions",
                        "find_todos", "line_count", "syntax_check"],
    },
    "Precise Edits": {
        "tools": ["search_replace", "project_search_replace", "undo_edit"],
        "default_off": ["project_search_replace", "undo_edit"],
    },
    "Execution": {
        "tools": ["run_python", "run_cmd", "run_tests"],
        "default_off": ["run_cmd", "run_tests"],
    },
    "Git": {
        "tools": ["git_status", "git_diff", "git_log", "git_revert",
                  "git_branch", "git_stash"],
        "default_off": ["git_revert", "git_branch", "git_stash",
                        "git_log", "git_diff"],
    },
}

# Tools from the harness that are NOT coding tools (memory, LoRA, MCP, etc.)
# These are always available if the harness provides them — they don't
# need checkboxes since they're not workspace-file operations.
HARNESS_EXTRA_TOOLS = {
    "remember", "recall_memory", "forget",
    "load_lora", "unload_lora", "list_loras",
    "get_time", "set_timer", "check_timer", "cancel_timer", "list_timers",
    "check_library", "list_allowed_libraries",
    "list_backups", "create_backup",
    "spawn_sub_agent", "spawn_sub_agents", "check_sub_agent",
    "wait_sub_agents", "list_sub_agents",
    # web (read-only GET — always available for real-time research)
    "web_search", "web_fetch", "wikipedia_search", "arxiv_search",
}


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
        self.round_idx = round_idx
        self.raw_output = ""
        self.parsed_text = ""
        self.rendered_prompt = ""
        self.tool_log: list[dict] = []  # [{name, arguments, result}]
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

    def set_raw(self, raw: str) -> None:
        self.raw_output = raw or ""

    def set_prompt(self, prompt: str) -> None:
        self.rendered_prompt = prompt or ""

    def set_parsed_text(self, content: str) -> None:
        self.parsed_text = content or ""
        self.text.setText(content or "(no text)")

    def add_tool_card(self, card: _ToolCard) -> None:
        self.lay.addWidget(card)

    def log_tool_call(self, call: dict) -> None:
        self.tool_log.append({
            "name": call.get("name", "?"),
            "arguments": call.get("arguments") or call.get("args") or {},
            "result": None,
        })

    def log_tool_result(self, rec: dict) -> None:
        if self.tool_log:
            self.tool_log[-1]["result"] = rec

    def to_log_text(self) -> str:
        """Serialize this round to plain text for clipboard/debugging."""
        lines = [f"--- Round {self.round_idx + 1} ---"]
        lines.append("RENDERED PROMPT (first 2000 chars):")
        lines.append((self.rendered_prompt or "(none)")[:2000])
        lines.append("")
        lines.append("RAW OUTPUT:")
        lines.append(self.raw_output or "(empty)")
        lines.append("")
        lines.append("PARSED TEXT:")
        lines.append(self.parsed_text or "(none)")
        for i, tl in enumerate(self.tool_log):
            lines.append("")
            lines.append(f"TOOL CALL {i+1}: {tl['name']}({json.dumps(tl['arguments'], ensure_ascii=False)})")
            res = tl.get("result")
            if res is not None:
                try:
                    res_text = json.dumps(res, ensure_ascii=False, indent=1)
                except (TypeError, ValueError):
                    res_text = str(res)
                lines.append(f"RESULT: {res_text[:4000]}")
            else:
                lines.append("RESULT: (pending)")
        return "\n".join(lines)


class AgentPage(QWidget):
    def __init__(self, runtime: EngineRuntime,
                 tool_harness=None, lorebook=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.tool_harness = tool_harness
        self.lorebook = lorebook
        self._runner: Optional[AgentRunner] = None
        self._round_blocks: dict[int, _RoundBlock] = {}
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
        self._copy_btn = QPushButton("Copy log")
        self._copy_btn.setToolTip(
            "Copy the full agent trace (raw model output + parsed text + "
            "tool calls + results) to clipboard for debugging.")
        btn_row.addWidget(self._run_btn); btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addStretch(1)
        self._status = QLabel("Idle · engine must be loaded (Engine Console)")
        self._status.setObjectName("chatMeta")
        btn_row.addWidget(self._status)
        task_col.addLayout(btn_row)
        bl.addLayout(task_col, 1)

        # config column — wrapped in a scroll area so the tool list
        # doesn't get squished when there are many categories.
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
        # Approval mode: 3-state combo instead of single checkbox
        cg.addWidget(section_label("Approval"))
        self._approval_mode = QComboBox()
        self._approval_mode.addItem("Destructive only", APPROVAL_DESTRUCTIVE)
        self._approval_mode.addItem("All side-effects", APPROVAL_ALL)
        self._approval_mode.addItem("None (auto)", APPROVAL_NONE)
        self._approval_mode.setToolTip(
            "Controls when the agent asks for permission before acting.\n"
            "Destructive only: prompts for delete/revert/branch ops.\n"
            "All side-effects: prompts for every write/command.\n"
            "None: agent acts autonomously.")
        cg.addWidget(self._approval_mode)
        self._thinking = QCheckBox("Thinking mode")
        self._thinking.setToolTip(
            "Prepend a chain-of-thought instruction to the system prompt — "
            "the agent reasons step-by-step before acting.")
        cg.addWidget(self._thinking)
        cg.addWidget(section_label("Tools"))
        self._tool_checks: dict[str, QCheckBox] = {}
        self._cat_checks: dict[str, QToolButton] = {}
        for cat_name, cat_def in TOOL_CATEGORIES.items():
            cat_row = QHBoxLayout()
            cat_toggle = QToolButton()
            cat_toggle.setText(cat_name)
            cat_toggle.setCheckable(True)
            cat_toggle.setChecked(True)
            cat_toggle.setObjectName("categoryToggle")
            self._cat_checks[cat_name] = cat_toggle
            cat_row.addWidget(cat_toggle)
            cat_row.addStretch(1)
            cg.addLayout(cat_row)
            tool_grid = QVBoxLayout(); tool_grid.setSpacing(2)
            tool_grid.setContentsMargins(20, 0, 0, 0)
            for t in cat_def["tools"]:
                cb = QCheckBox(t)
                cb.setChecked(t not in cat_def.get("default_off", []))
                self._tool_checks[t] = cb
                tool_grid.addWidget(cb)
            cg.addLayout(tool_grid)
            # wire category toggle to select/deselect all in category
            cat_toggle.toggled.connect(
                lambda checked, gn=cat_name: self._on_cat_toggle(gn, checked))
        # Wrap config in a scroll area so it doesn't get squished.
        cfg_scroll = QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cfg_scroll.setObjectName("root")
        cfg_scroll.setWidget(cfg)
        cfg_scroll.setFixedWidth(300)
        bl.addWidget(cfg_scroll)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0)
        splitter.setSizes([560, 300])

        self._run_btn.clicked.connect(self._run)
        self._stop_btn.clicked.connect(self._stop)
        self._copy_btn.clicked.connect(self._copy_log)
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

    def _on_cat_toggle(self, cat_name: str, checked: bool) -> None:
        cat = TOOL_CATEGORIES.get(cat_name)
        if cat is None:
            return
        for t in cat["tools"]:
            cb = self._tool_checks.get(t)
            if cb is not None:
                cb.setChecked(checked)

    def _on_engine_state(self, state: str) -> None:
        if state == "ready":
            self._status.setText("Engine ready — describe a task and run")
        elif state == "loading":
            self._status.setText("Engine loading…")
        else:
            self._status.setText("Idle · engine must be loaded (Engine Console)")

    def _get_enabled_tools(self) -> Optional[list[str]]:
        """Get the list of enabled coding tools + always-on harness tools."""
        tools = [t for t, cb in self._tool_checks.items() if cb.isChecked()]
        # Always include harness extra tools (memory, LoRA, time, etc.)
        # if the harness provides them — they don't need checkboxes.
        if self.tool_harness is not None:
            try:
                all_defs = self.tool_harness.tool_defs()
                for d in all_defs:
                    name = d["function"]["name"]
                    if name in HARNESS_EXTRA_TOOLS:
                        tools.append(name)
            except Exception:
                pass
        return tools

    def _build_system_prompt(self, enabled_tools: list[str]) -> str:
        """Build a system prompt. Tool definitions are appended by the
        qwen_render_messages renderer, so we don't list them here to
        avoid redundancy and save KV cache tokens."""
        prompt = DEFAULT_SYSTEM
        # If web tools are available, add a research hint so the model
        # knows to use them for online research tasks.
        web_tools = {"web_search", "web_fetch", "wikipedia_search",
                     "arxiv_search"}
        if web_tools & set(enabled_tools):
            prompt += (
                "\nYou have web tools available. For any task that needs "
                "online research, current information, or looking up papers, "
                "call web_search FIRST with a relevant query.")
        if self._thinking.isChecked():
            prompt = (
                "Think step-by-step before acting. Show your reasoning, "
                "then use tools to complete the task.\n\n" + prompt
            )
        return prompt

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
        tools = self._get_enabled_tools()
        sys_prompt = self._build_system_prompt(tools or [])
        approval = self._approval_mode.currentData()
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
            approval_mode=approval,
            tool_harness=self.tool_harness,
        )
        self._runner.round_started.connect(self._on_round)
        self._runner.text.connect(self._on_text)
        self._runner.raw_output.connect(self._on_raw)
        self._runner.prompt_rendered.connect(self._on_prompt)
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
    def _on_round(self, round_idx: int) -> None:
        block = _RoundBlock(round_idx)
        self._round_blocks[round_idx] = block
        self._trace_lay.insertWidget(self._trace_lay.count() - 1, block)
        self._scroll_bottom()

    def _on_text(self, round_idx: int, content: str) -> None:
        block = self._round_blocks.get(round_idx)
        if block is not None:
            block.set_parsed_text(content)
            self._scroll_bottom()

    def _on_raw(self, round_idx: int, raw: str) -> None:
        block = self._round_blocks.get(round_idx)
        if block is not None:
            block.set_raw(raw)

    def _on_prompt(self, round_idx: int, prompt: str) -> None:
        block = self._round_blocks.get(round_idx)
        if block is not None:
            block.set_prompt(prompt)

    def _on_tool_call(self, round_idx: int, call: dict) -> None:
        block = self._round_blocks.get(round_idx)
        if block is None:
            self._on_round(round_idx)
            block = self._round_blocks[round_idx]
        block.log_tool_call(call)
        card = _ToolCard(call)
        block.add_tool_card(card)
        self._scroll_bottom()

    def _on_tool_result(self, round_idx: int, rec: dict) -> None:
        block = self._round_blocks.get(round_idx)
        if block is None:
            return
        block.log_tool_result(rec)
        cards = block.findChildren(_ToolCard)
        if cards:
            cards[-1].set_result(rec)
        self._scroll_bottom()

    def _on_approval(self, round_idx: int, call: dict) -> None:
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
    def _copy_log(self) -> None:
        """Serialize the full agent trace to clipboard for debugging."""
        from PySide6.QtWidgets import QApplication
        parts = [
            "=== ForgeAI Agent Log ===",
            f"Task: {self._task.toPlainText().strip()[:500]}",
            f"Workspace: {self._workspace.text().strip()}",
            f"Rounds: {self._rounds.value()} | Max tok: {self._max_tok.value()} "
            f"| Temp: {self._temp.value()} | Rep pen: {self._rep_pen.value()}",
            "",
        ]
        if not self._round_blocks:
            parts.append("(no rounds — agent has not been run yet)")
        for idx in sorted(self._round_blocks):
            block = self._round_blocks[idx]
            parts.append(block.to_log_text())
            parts.append("")
        text = "\n".join(parts)
        QApplication.clipboard().setText(text)
        self._status.setText(f"log copied ({len(text)} chars)")

    def _clear_trace(self) -> None:
        self._round_blocks.clear()
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
