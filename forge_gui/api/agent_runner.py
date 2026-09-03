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

# Tools that modify files or execute code — may require approval.
SIDE_EFFECT_TOOLS = frozenset({
    "write_file", "append_file", "delete_file",
    "run_python", "run_cmd",
    "search_replace", "create_file", "git_revert",
    "rename_file", "project_search_replace", "undo_edit",
    "git_branch", "git_stash", "run_tests",
})

# Destructive tools that can cause data loss — always prompt in
# "destructive" approval mode.
DESTRUCTIVE_TOOLS = frozenset({
    "delete_file", "git_revert", "project_search_replace",
    "git_branch", "git_stash",
})

# Approval modes
APPROVAL_NONE = "none"
APPROVAL_DESTRUCTIVE = "destructive"
APPROVAL_ALL = "all"

DEFAULT_SYSTEM = (
    "You are Forge Agent, an expert coding and research agent. You MUST use "
    "tools to complete the task — never just write explanations. Call a tool "
    "every turn until done, then give a short summary.\n"
    "For research tasks: start with web_search to find information online, "
    "then web_fetch to read full pages. Use wikipedia_search for factual "
    "background and arxiv_search for academic papers.\n"
    "For coding tasks: start with list_dir and read_file to explore the "
    "workspace, then use write_file or search_replace to make changes.\n"
    "Format: <|tool_call_start|>[tool_name(arg='value')]<|tool_call_end|>"
)

# How many times to retry when the model produces no tool call, feeding
# back an error message each time (matches discovery loop pattern).
NO_TOOL_RETRIES = 2


class AgentRunner(QThread):
    """Runs one agent task to completion (or cancellation)."""

    round_started = Signal(int)
    text = Signal(int, str)              # round_idx, assistant content
    raw_output = Signal(int, str)        # round_idx, raw model output (pre-parse)
    prompt_rendered = Signal(int, str)   # round_idx, full rendered prompt
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
                 approval_mode: str = APPROVAL_DESTRUCTIVE,
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
        self.approval_mode = approval_mode
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
            TOOL_CALL_START, TOOL_CALL_END,
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

        # Grammar constraints enforce JSON tool-call format, but the model
        # was trained on Pythonic format [func(arg='val')]. JSON grammar
        # would break Pythonic parsing, so we skip grammar constraints.
        # The model already knows the format from training.
        grammar_proc = None

        for round_idx in range(self.max_rounds):
            if self._cancel:
                break
            rounds = round_idx + 1
            self.round_started.emit(round_idx)

            rendered = qwen_render_messages(
                messages, tools=defs, add_generation_prompt=True)
            self.prompt_rendered.emit(round_idx, rendered)

            # Generate with grammar-constrained decoding if available.
            # generate_raw supports logits_processor; generate() does not.
            with self._runtime.acquire() as engine:
                if grammar_proc is not None and hasattr(engine, "generate_raw"):
                    raw = engine.generate_raw(
                        rendered, max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        logits_processor=grammar_proc,
                        skip_special_tokens=False)
                else:
                    raw = engine.generate(
                        rendered, max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature, top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        skip_special_tokens=False)
            if self._cancel:
                break

            tool_calls, content = qwen_parse_tool_calls(raw)
            self.raw_output.emit(round_idx, raw)
            self.text.emit(round_idx, content or "")
            messages.append({"role": "assistant", "content": content or ""})

            if not tool_calls:
                # Retry: feed back an error telling the model to use tools,
                # then try again (up to NO_TOOL_RETRIES times per round).
                # This matches the discovery loop pattern.
                retried = False
                for retry in range(NO_TOOL_RETRIES):
                    if self._cancel:
                        break
                    err_msg = (
                        f"No tool call found in your response. You MUST use "
                        f"tools to complete the task. To call a tool, output: "
                        f"{TOOL_CALL_START}\n{{\"name\": \"tool_name\", "
                        f"\"arguments\": {{...}}}}\n{TOOL_CALL_END}\n"
                        f"Available tools: {', '.join(d['function']['name'] for d in defs)}. "
                        f"Try again — start by calling list_dir or read_file "
                        f"to explore the workspace."
                    )
                    messages.append({"role": "tool", "name": "system",
                                     "content": err_msg})
                    rendered = qwen_render_messages(
                        messages, tools=defs, add_generation_prompt=True)
                    self.prompt_rendered.emit(round_idx, rendered)
                    with self._runtime.acquire() as engine:
                        if grammar_proc is not None and hasattr(engine, "generate_raw"):
                            raw = engine.generate_raw(
                                rendered, max_new_tokens=self.max_new_tokens,
                                temperature=self.temperature, top_p=self.top_p,
                                top_k=self.top_k,
                                repetition_penalty=self.repetition_penalty,
                                logits_processor=grammar_proc,
                                skip_special_tokens=False)
                        else:
                            raw = engine.generate(
                                rendered, max_new_tokens=self.max_new_tokens,
                                temperature=self.temperature, top_p=self.top_p,
                                top_k=self.top_k,
                                repetition_penalty=self.repetition_penalty,
                                skip_special_tokens=False)
                    if self._cancel:
                        break
                    tool_calls, content = qwen_parse_tool_calls(raw)
                    self.raw_output.emit(round_idx, raw)
                    self.text.emit(round_idx, content or "")
                    messages.append({"role": "assistant", "content": content or ""})
                    if tool_calls:
                        retried = True
                        break
                if not retried:
                    break

            for tc in tool_calls:
                if self._cancel:
                    break
                call = tc if isinstance(tc, dict) else {"name": str(tc)}
                self.tool_call.emit(round_idx, call)
                if not self._may_execute(call):
                    results = [{"ok": False,
                                "result": {"error": "blocked by policy"}}]
                else:
                    if self._needs_approval(call):
                        self._approval.clear()
                        self.approval_requested.emit(round_idx, call)
                        self._approval.wait()
                        if self._cancel or not self._approval_granted:
                            results = [{"ok": False,
                                        "result": {"error": "denied by user"}}]
                            rec = results[0]
                            self.tool_result.emit(round_idx, rec)
                            messages.append({
                                "role": "tool",
                                "name": call.get("name", "tool"),
                                "content": json.dumps(rec["result"]),
                            })
                            continue
                    results = harness.execute_calls([call])
                rec = results[0]
                all_calls.append(call)
                self.tool_result.emit(round_idx, rec)
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

    def _needs_approval(self, call: dict) -> bool:
        name = call.get("name", "")
        if self.approval_mode == APPROVAL_NONE:
            return False
        if self.approval_mode == APPROVAL_ALL:
            return name in SIDE_EFFECT_TOOLS
        # destructive only
        return name in DESTRUCTIVE_TOOLS
