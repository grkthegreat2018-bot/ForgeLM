"""AgentRunner — background agentic coding loop over the resident engine.

Implements the tool round-trip that ``ForgeEngine.generate_with_tools``
uses internally, but emits Qt signals per step so the Agent page can render
a live trace (assistant text → tool call cards → results → next round).

Tools execute via a :class:`ToolHarness` which combines coding tools,
memory tools (lorebook), and LoRA tools with safety checking. Optional
approval mode pauses before side-effecting tools (write/append/delete/
run_*) until the UI calls ``respond_approval``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal

from .agent_tools import tool_results_to_text
from .engine_runtime import EngineRuntime

logger = logging.getLogger(__name__)

SIDE_EFFECT_TOOLS = {"write_file", "append_file", "delete_file",
                     "run_python", "run_cmd"}

DEFAULT_SYSTEM = (
    "You are Forge Agent, an expert coding agent working inside the user's "
    "workspace. Solve the task step by step using the provided tools "
    "(list_dir, read_file, write_file, run_python, run_cmd, grep_project, "
    "remember, recall_memory, forget, load_lora, unload_lora, list_loras). "
    "Prefer small, verifiable changes: read before writing, run the code to "
    "check your work, and keep the final answer concise. Use 'remember' to "
    "save important facts to long-term memory. Use 'load_lora' to specialize "
    "the model for a task (e.g. load_lora(category='coding') before writing "
    "code). When the task is complete, reply with a short summary and stop "
    "calling tools."
)


class AgentRunner(QThread):
    """Runs one agent task to completion (or cancellation)."""

    round_started = Signal(int)
    text = Signal(str, str)              # round_idx, assistant content
    tool_call = Signal(int, dict)        # round_idx, {name, arguments}
    tool_result = Signal(int, dict)      # round_idx, sandbox record
    approval_requested = Signal(int, dict)  # round_idx, call dict
    finished_ok = Signal(dict)           # final result
    failed = Signal(str)

    def __init__(self, runtime: EngineRuntime, task: str, workspace: str,
                 system_prompt: str = DEFAULT_SYSTEM,
                 max_rounds: int = 8, max_new_tokens: int = 512,
                 temperature: float = 0.2, top_p: float = 0.95, top_k: int = 80,
                 repetition_penalty: float = 1.05,
                 enabled_tools: Optional[list[str]] = None,
                 require_approval: bool = False,
                 history: Optional[list[dict]] = None,
                 tool_harness=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self.task = task
        self.workspace = workspace
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.enabled_tools = enabled_tools  # None = all
        self.require_approval = require_approval
        self.history = list(history or [])  # prior conversation (chat hand-off)
        self.tool_harness = tool_harness    # ToolHarness or None (auto-create)
        self._cancel = False
        self._approval = threading.Event()
        self._approval_granted = False

    # ── control ───────────────────────────────────────────────────────
    def cancel(self) -> None:
        self._cancel = True
        self._approval_granted = False
        self._approval.set()

    def respond_approval(self, granted: bool) -> None:
        self._approval_granted = granted
        self._approval.set()

    # ── main loop ─────────────────────────────────────────────────────
    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            result = self._loop()
            result["elapsed_s"] = round(time.perf_counter() - t0, 2)
            self.finished_ok.emit(result)
        except Exception as e:
            logger.warning("agent loop failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _loop(self) -> dict:
        from research.self_play.discovery.qwen_adapter import (  # type: ignore
            qwen_parse_tool_calls, qwen_render_messages,
        )
        from .tool_harness import ToolHarness

        # Use provided harness or create one
        harness = self.tool_harness
        if harness is None:
            harness = ToolHarness(self.workspace, enable_safety=True)

        defs = harness.tool_defs()
        if self.enabled_tools is not None:
            allow = set(self.enabled_tools)
            defs = [d for d in defs if d["function"]["name"] in allow]

        messages: list[dict] = list(self.history)
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.task})

        all_calls: list[dict] = []
        content = ""
        rounds = 0
        for round_idx in range(self.max_rounds):
            if self._cancel:
                break
            rounds = round_idx + 1
            self.round_started.emit(round_idx)

            rendered = qwen_render_messages(
                messages, tools=defs, add_generation_prompt=True)
            with self._runtime.acquire() as engine:
                raw = engine.generate(
                    rendered, max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature, top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty)
            if self._cancel:
                break

            tool_calls, content = qwen_parse_tool_calls(raw)
            self.text.emit(str(round_idx), content or "")
            messages.append({"role": "assistant", "content": content or ""})

            if not tool_calls:
                break

            for tc in tool_calls:
                if self._cancel:
                    break
                call = tc if isinstance(tc, dict) else {"name": str(tc)}
                self.tool_call.emit(str(round_idx), call)
                if not self._may_execute(call):
                    results = [{"ok": False,
                                "result": {"error": "blocked by policy"}}]
                else:
                    if self.require_approval and self._needs_approval(call):
                        self._approval.clear()
                        self.approval_requested.emit(str(round_idx), call)
                        self._approval.wait()
                        if self._cancel or not self._approval_granted:
                            results = [{"ok": False,
                                        "result": {"error": "denied by user"}}]
                            rec = results[0]
                            self.tool_result.emit(str(round_idx), rec)
                            messages.append({
                                "role": "tool",
                                "name": call.get("name", "tool"),
                                "content": json.dumps(rec["result"]),
                            })
                            continue
                    results = harness.execute_calls([call])
                rec = results[0]
                all_calls.append(call)
                self.tool_result.emit(str(round_idx), rec)
                messages.append({
                    "role": "tool",
                    "name": call.get("name", "tool"),
                    "content": tool_results_to_text(rec),
                })

                # check for safety termination
                if rec.get("result", {}).get("terminated"):
                    self._cancel = True
                    break

        return {
            "content": content or "",
            "rounds": rounds,
            "tool_calls": all_calls,
            "sandbox": harness.summary(),
            "cancelled": self._cancel,
            "safety": harness.strikes.summary(),
        }

    # ── policy ────────────────────────────────────────────────────────
    def _may_execute(self, call: dict) -> bool:
        if self.enabled_tools is None:
            return True
        return call.get("name") in set(self.enabled_tools)

    @staticmethod
    def _needs_approval(call: dict) -> bool:
        return call.get("name") in SIDE_EFFECT_TOOLS
