"""SPOC/MIRROR — Agent self-correction & rollback for tool-use errors.

SPOC: Spontaneous self-correction in a single pass. The model detects
its own tool-call errors and corrects them without external feedback.
Source: arXiv 2506.06923

MIRROR: Intra + inter reflection. Structured reflection phases that
review tool results, identify failures, and roll back to a safe state
before retrying with a corrected approach.
Source: arXiv 2505.20670

VRAM budget: negligible — stores only conversation snapshots (text).
No model parameters or GPU buffers.
"""
from __future__ import annotations

import json
import copy
from typing import Any


class RollbackPoint:
    """A snapshot of the agent conversation state for rollback."""

    __slots__ = ("messages", "tool_calls", "tool_results", "round_idx")

    def __init__(self, messages, tool_calls, tool_results, round_idx):
        self.messages = copy.deepcopy(messages)
        self.tool_calls = list(tool_calls)
        self.tool_results = list(tool_results)
        self.round_idx = round_idx


class SPOCCorrector:
    """SPOC — Spontaneous self-correction for tool-use errors.

    Detects tool-call errors by inspecting tool results for error keys.
    When an error is detected, injects a reflection prompt that asks the
    model to identify what went wrong and retry with a corrected call.
    """

    ERROR_KEYS = ("error", "exception", "traceback", "failed", "invalid")
    MAX_CORRECTIONS = 2

    def __init__(self, max_corrections: int = 2):
        self.max_corrections = max_corrections

    def detect_error(self, result: dict) -> tuple[bool, str | None]:
        """Check if a tool result indicates an error."""
        if not isinstance(result, dict):
            return False, None
        for key in self.ERROR_KEYS:
            if key in result:
                val = result[key]
                if val:
                    return True, str(val)
        if "ok" in result and result["ok"] is False:
            return True, result.get("error", "Tool returned ok=False")
        return False, None

    def build_reflection_prompt(self, tool_call: dict,
                                error_msg: str) -> str:
        """Build a reflection prompt for the model to self-correct."""
        return (
            f"The tool call to '{tool_call.get('name', 'unknown')}' "
            f"failed with error: {error_msg}\n"
            f"Arguments used: {json.dumps(tool_call.get('arguments', {}))}\n\n"
            f"Reflect on what went wrong and retry with corrected arguments. "
            f"Do not repeat the same mistake."
        )


class MIRRORCorrector:
    """MIRROR — Intra + inter reflection with rollback.

    Intra-round reflection: within a single tool round, review all tool
    results and identify failures. If failures detected, roll back to
    the state before the failed round and retry with a reflection.

    Inter-round reflection: across multiple rounds, track error patterns.
    If the same tool fails repeatedly, escalate to a different approach.
    """

    def __init__(self, max_rollbacks: int = 2, error_window: int = 3):
        self.max_rollbacks = max_rollbacks
        self.error_window = error_window
        self._rollback_stack: list[RollbackPoint] = []
        self._error_history: list[str] = []

    def checkpoint(self, messages, tool_calls, tool_results, round_idx):
        """Save a rollback point before a risky tool round."""
        self._rollback_stack.append(
            RollbackPoint(messages, tool_calls, tool_results, round_idx))

    def rollback(self) -> RollbackPoint | None:
        """Roll back to the most recent checkpoint."""
        if self._rollback_stack:
            return self._rollback_stack.pop()
        return None

    def record_error(self, tool_name: str, error_msg: str):
        """Track error patterns across rounds."""
        self._error_history.append(f"{tool_name}: {error_msg}")

    def should_escalate(self, tool_name: str) -> bool:
        """Check if a tool has failed too many times (inter-round)."""
        count = sum(1 for e in self._error_history if e.startswith(tool_name))
        return count >= self.error_window

    def build_inter_reflection(self) -> str:
        """Build an inter-round reflection summarizing error patterns."""
        if not self._error_history:
            return ""
        recent = self._error_history[-self.error_window:]
        return (
            "Recent errors across rounds:\n"
            + "\n".join(f"  - {e}" for e in recent)
            + "\n\nConsider a different approach to avoid repeating these errors."
        )


class AgentSelfCorrector:
    """Unified SPOC + MIRROR self-correction for generate_with_tools.

    Wraps the tool execution loop with:
    1. Error detection (SPOC)
    2. Rollback + reflection (MIRROR)
    3. Error pattern tracking (MIRROR inter-round)
    """

    def __init__(self, max_corrections: int = 2, max_rollbacks: int = 2):
        self.spoc = SPOCCorrector(max_corrections)
        self.mirror = MIRRORCorrector(max_rollbacks)
        self.corrections_made = 0
        self.rollbacks_made = 0

    def check_results(self, tool_calls, tool_results
                      ) -> list[tuple[int, dict, str]]:
        """Check tool results for errors. Returns list of (idx, call, error)."""
        errors = []
        for i, (call, result) in enumerate(zip(tool_calls, tool_results)):
            is_error, msg = self.spoc.detect_error(result)
            if is_error:
                errors.append((i, call, msg or "Unknown error"))
                self.mirror.record_error(call.get("name", "unknown"), msg or "")
        return errors

    def maybe_correct(self, messages, tool_calls, tool_results,
                      round_idx) -> str | None:
        """Attempt self-correction for failed tool calls.

        Returns a reflection prompt to inject, or None if no correction
        needed. Also handles rollback if too many errors in a round.
        """
        errors = self.check_results(tool_calls, tool_results)
        if not errors:
            return None

        if self.corrections_made >= self.spoc.max_corrections:
            rb = self.mirror.rollback()
            if rb is not None and self.rollbacks_made < self.mirror.max_rollbacks:
                self.rollbacks_made += 1
                self.corrections_made = 0
                return (
                    "Rolling back to a previous state due to repeated errors. "
                    + self.mirror.build_inter_reflection()
                )
            return None

        self.corrections_made += 1
        first_error = errors[0]
        reflection = self.spoc.build_reflection_prompt(
            first_error[1], first_error[2])
        if self.mirror.should_escalate(first_error[1].get("name", "")):
            reflection += "\n\n" + self.mirror.build_inter_reflection()
        return reflection

    def checkpoint(self, messages, tool_calls, tool_results, round_idx):
        self.mirror.checkpoint(messages, tool_calls, tool_results, round_idx)

    def stats(self) -> dict:
        return {
            "corrections_made": self.corrections_made,
            "rollbacks_made": self.rollbacks_made,
            "error_history": list(self.mirror._error_history),
        }
