"""Unified tool harness — combines coding, memory, LoRA, and MCP tools with safety.

This is the single tool dispatch layer used by both the Chat page and the
Agent page. It wraps:

- **ToolSandbox** (agent_tools.py) — coding tools (read/write/grep/run)
- **MemoryTools** (lorebook.py) — remember/recall/forget
- **LoraTools** (this module) — load_lora/unload_lora/list_loras
- **MCPManager** (mcp_client.py) — external MCP server tools
- **SafetyChecker** (safety_checker.py) — validates every edit before commit

Safety integration:
- Every ``write_file`` / ``append_file`` is checked with ``check_edit()``
  *before* the write happens. If unsafe, the edit is cancelled and the
  model is warned.
- Every ``run_cmd`` / ``run_python`` is checked with ``check_command()``
  / ``check_ast()`` before execution.
- The ``StrikeTracker`` accumulates violations. After 3 strikes, the
  harness returns a termination signal and the agent loop should stop.

The harness also supports read-only mode (for chat without side effects)
and full mode (for the agentic loop).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .agent_tools import ToolSandbox, tool_results_to_text
from .backup_manager import BackupManager, backup_tool_defs
from .library_install import LibraryInstallManager, library_tool_defs
from .lorebook import Lorebook, MemoryTools, memory_tool_defs
from .lora_training_trigger import LoraTrainingTrigger, lora_training_tool_defs
from .safety_checker import StrikeTracker, check_ast, check_command, check_edit
from .status_reader import project_root
from .sub_agent import SubAgentManager, sub_agent_tool_defs
from .time_manager import TimeManager, time_tool_defs
from .web_tools import WebTools, web_tool_defs

logger = logging.getLogger(__name__)

# ── LoRA tool definitions ───────────────────────────────────────────────

_LORA_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "load_lora",
            "description": (
                "Hot-load a LoRA adapter onto the resident engine to "
                "specialize it for a task. Use when you need coding, math, "
                "or agentic skill enhancement. The adapter stays loaded "
                "until unloaded or replaced."),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Skill category to load (coding, math, "
                            "reasoning, tool_use, agentic, chat_assist, "
                            "self_play, vision). The best adapter for "
                            "this category will be auto-selected."),
                        "enum": ["coding", "math", "reasoning", "tool_use",
                                 "agentic", "chat_assist", "self_play",
                                 "vision"],
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unload_lora",
            "description": (
                "Unload the current LoRA adapter from the resident engine, "
                "reverting to the base model."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_loras",
            "description": (
                "List all available LoRA adapters grouped by skill category, "
                "showing which is currently loaded."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── unified harness ─────────────────────────────────────────────────────

class ToolHarness:
    """Unified tool dispatch with safety checking and strike tracking.

    Args:
        workspace: Root directory for file operations (sandbox jail).
        lorebook: Lorebook instance for memory tools (None to disable).
        lora_harness: LoraHarness instance for LoRA tools (None to disable).
        read_only: If True, side-effecting tools (write/delete/run) are
            disabled. Memory tools still work. Used for chat mode without
            full agent capabilities.
        enable_safety: If True (default), all edits are safety-checked.
    """

    def __init__(self, workspace: str,
                 lorebook: Optional[Lorebook] = None,
                 lora_harness=None,
                 mcp_manager=None,
                 lora_training=None,
                 backup_manager: Optional[BackupManager] = None,
                 sub_agent_manager: Optional[SubAgentManager] = None,
                 time_manager: Optional[TimeManager] = None,
                 library_manager: Optional[LibraryInstallManager] = None,
                 web_tools: Optional[WebTools] = None,
                 read_only: bool = False,
                 enable_safety: bool = True) -> None:
        self.sandbox = ToolSandbox(workspace)
        self.lorebook = lorebook
        self.lora_harness = lora_harness
        self.mcp_manager = mcp_manager
        self.lora_training = lora_training
        self.backup_manager = backup_manager
        self.sub_agent_manager = sub_agent_manager
        self.time_manager = time_manager
        self.library_manager = library_manager
        self.web_tools = web_tools
        self.read_only = read_only
        self.enable_safety = enable_safety
        self.strikes = StrikeTracker()

        # memory tools wrapper
        self._memory = MemoryTools(lorebook) if lorebook else None

        # side-effecting tool names (blocked in read-only mode)
        self._side_effect_tools = {
            "write_file", "append_file", "delete_file",
            "run_python", "run_cmd",
            "search_replace", "create_file", "git_revert",
            "rename_file", "project_search_replace", "undo_edit",
            "git_branch", "git_stash", "run_tests",
        }

        # cache of MCP tool names for dispatch
        self._mcp_tool_names: set[str] = set()
        if mcp_manager is not None:
            self._refresh_mcp_tools()

    def _refresh_mcp_tools(self) -> None:
        """Refresh the cache of MCP tool names."""
        if self.mcp_manager is not None:
            self._mcp_tool_names = {t.name for t in self.mcp_manager.all_tools()}

    # ── tool definitions ──────────────────────────────────────────────
    def tool_defs(self) -> list[dict]:
        """All available tool definitions for the model."""
        from .agent_tools import tool_defs as coding_defs
        defs = coding_defs()
        if self._memory is not None:
            defs.extend(memory_tool_defs())
        if self.lora_harness is not None:
            defs.extend(_LORA_TOOL_DEFS)
        if self.mcp_manager is not None:
            self._refresh_mcp_tools()
            defs.extend(self.mcp_manager.tool_defs())
        if self.lora_training is not None:
            defs.extend(lora_training_tool_defs())
        if self.backup_manager is not None and not self.read_only:
            defs.extend(backup_tool_defs())
        if self.sub_agent_manager is not None and not self.read_only:
            defs.extend(sub_agent_tool_defs())
        if self.time_manager is not None:
            defs.extend(time_tool_defs())
        if self.library_manager is not None and not self.read_only:
            defs.extend(library_tool_defs())
        if self.web_tools is not None:
            defs.extend(web_tool_defs())
        # filter out side-effect tools in read-only mode
        if self.read_only:
            defs = [d for d in defs
                    if d["function"]["name"] not in self._side_effect_tools]
        return defs

    def chat_tool_defs(self) -> list[dict]:
        """Return a reduced set of tool definitions suitable for chat mode.

        Chat mode has a limited KV cache (2048 tokens on 12GB VRAM), so
        we only expose the most commonly useful tools to keep the system
        prompt short. The full set is available in agent mode.
        """
        # tools that are useful in chat and have short descriptions
        chat_tools = {
            # memory
            "remember", "recall_memory", "forget",
            # LoRA
            "load_lora", "unload_lora", "list_loras",
            # read-only file tools
            "list_dir", "read_file", "grep_project", "file_info",
            # time
            "get_time", "set_timer", "check_timer", "cancel_timer", "list_timers",
            # library check (read-only)
            "check_library", "list_allowed_libraries",
            # web (read-only GET — safe for chat)
            "web_search", "web_fetch", "wikipedia_search", "arxiv_search",
        }
        defs = self.tool_defs()
        return [d for d in defs
                if d["function"]["name"] in chat_tools]

    # ── dispatch ──────────────────────────────────────────────────────
    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call with safety checking.

        Returns a record dict: {"name", "args", "ok", "elapsed_s", "result"}.
        If the safety check fails, the edit is cancelled and the result
        contains the safety verdict.
        """
        import time as _time
        t0 = _time.perf_counter()

        # read-only check
        if self.read_only and name in self._side_effect_tools:
            result = {"error": f"tool '{name}' not available in read-only mode"}
            return self._record(name, args, result, False, t0)

        # safety check for write operations
        if self.enable_safety and name in ("write_file", "append_file", "create_file"):
            verdict = check_edit(
                args.get("content", ""), args.get("path", ""),
                len(args.get("content", "").encode("utf-8")))
            if not verdict:
                self.strikes.record(verdict, f"{name}({args.get('path', '')})")
                result = {
                    "error": f"SAFETY: {verdict.reason}",
                    "safety_layer": verdict.layer,
                    "safety_severity": verdict.severity,
                    "suggestions": verdict.suggestions,
                    "strikes": self.strikes.count,
                    "terminated": self.strikes.terminated,
                }
                return self._record(name, args, result, False, t0)

        # safety check for search_replace (check the new_text content)
        if self.enable_safety and name == "search_replace":
            verdict = check_edit(
                args.get("new_text", ""), args.get("path", ""),
                len(args.get("new_text", "").encode("utf-8")))
            if not verdict:
                self.strikes.record(verdict, f"search_replace({args.get('path', '')})")
                result = {
                    "error": f"SAFETY: {verdict.reason}",
                    "safety_layer": verdict.layer,
                    "strikes": self.strikes.count,
                    "terminated": self.strikes.terminated,
                }
                return self._record(name, args, result, False, t0)

        # safety check for project_search_replace
        if self.enable_safety and name == "project_search_replace":
            from .safety_checker import check_path_safety
            verdict = check_edit(args.get("new_text", ""), "snippet.txt",
                                 len(args.get("new_text", "").encode("utf-8")))
            if not verdict:
                self.strikes.record(verdict, "project_search_replace")
                result = {
                    "error": f"SAFETY: {verdict.reason}",
                    "safety_layer": verdict.layer,
                    "strikes": self.strikes.count,
                    "terminated": self.strikes.terminated,
                }
                return self._record(name, args, result, False, t0)

        # safety check for commands
        if self.enable_safety and name == "run_cmd":
            verdict = check_command(args.get("command", ""))
            if not verdict:
                self.strikes.record(verdict, f"run_cmd({args.get('command', '')[:60]})")
                result = {
                    "error": f"SAFETY: {verdict.reason}",
                    "safety_layer": verdict.layer,
                    "strikes": self.strikes.count,
                    "terminated": self.strikes.terminated,
                }
                return self._record(name, args, result, False, t0)

        # safety check for python code
        if self.enable_safety and name == "run_python":
            verdict = check_ast(args.get("code", ""), "snippet.py")
            if not verdict:
                self.strikes.record(verdict, "run_python")
                result = {
                    "error": f"SAFETY: {verdict.reason}",
                    "safety_layer": verdict.layer,
                    "strikes": self.strikes.count,
                    "terminated": self.strikes.terminated,
                }
                return self._record(name, args, result, False, t0)

        # check termination
        if self.strikes.terminated:
            result = {
                "error": "SAFETY: loop terminated due to repeated violations",
                "flagged_areas": self.strikes.flagged_areas,
                "strikes": self.strikes.count,
            }
            return self._record(name, args, result, False, t0)

        # dispatch to the right tool set
        try:
            if name in ("remember", "recall_memory", "forget"):
                if self._memory is None:
                    result = {"error": "memory tools not available"}
                else:
                    rec = self._memory.execute(name, args)
                    return self._record(name, args, rec.get("result", {}),
                                        rec.get("ok", False), t0)
            elif name in ("load_lora", "unload_lora", "list_loras"):
                result = self._lora_tool(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            elif self.mcp_manager is not None and name in self._mcp_tool_names:
                # MCP tool dispatch
                result = self.mcp_manager.call_tool(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            elif name in ("train_lora", "check_training"):
                if self.lora_training is None:
                    result = {"error": "LoRA training tools not available"}
                    return self._record(name, args, result, False, t0)
                rec = self.lora_training.execute(name, args)
                return self._record(name, args, rec.get("result", {}),
                                    rec.get("ok", False), t0)
            elif name in ("list_backups", "create_backup", "load_backup"):
                result = self._backup_tool(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            elif name in ("spawn_sub_agent", "spawn_sub_agents",
                          "check_sub_agent", "wait_sub_agents",
                          "list_sub_agents"):
                result = self._sub_agent_tool(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            elif name in ("get_time", "set_timer", "set_alarm",
                          "check_timer", "cancel_timer", "list_timers"):
                result = self._time_tool(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            elif name in ("install_library", "check_library",
                          "list_allowed_libraries"):
                result = self._library_tool(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            elif self.web_tools is not None and name in WebTools.NAMES:
                result = self.web_tools.execute(name, args)
                ok = not (isinstance(result, dict) and "error" in result)
                return self._record(name, args, result, ok, t0)
            else:
                # coding tools via sandbox
                rec = self.sandbox.execute(name, args)
                return rec
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
            return self._record(name, args, result, False, t0)

    def execute_calls(self, calls: list[dict]) -> list[dict]:
        return [self.execute(c.get("name", ""),
                             c.get("arguments") or c.get("args") or {})
                for c in calls]

    # ── LoRA tools ────────────────────────────────────────────────────
    def _lora_tool(self, name: str, args: dict) -> dict:
        if self.lora_harness is None:
            return {"error": "LoRA tools not available (no lora_harness)"}
        if name == "list_loras":
            by_cat = self.lora_harness.adapters_by_category()
            current = self.lora_harness.current_adapter
            out = {"current": current, "categories": {}}
            for cat, entries in by_cat.items():
                if entries:
                    out["categories"][cat] = [
                        {"name": e.name, "rank": e.rank,
                         "size": e.size_label, "path": e.path}
                        for e in entries
                    ]
            return out
        elif name == "load_lora":
            category = args.get("category", "")
            if not category:
                return {"error": "category required"}
            from .lora_store import scan_lora_adapters
            entries = scan_lora_adapters()
            matching = [e for e in entries if e.category == category
                        and not e.header_error]
            if not matching:
                return {"error": f"no adapters found for category '{category}'"}
            matching.sort(key=lambda e: e.modified, reverse=True)
            best = matching[0]
            self.lora_harness._load(best.path)
            return {"ok": True, "result": {
                "loaded": best.name, "category": category,
                "rank": best.rank, "path": best.path}}
        elif name == "unload_lora":
            if self.lora_harness._current is None:
                return {"error": "no LoRA currently loaded"}
            # trigger unload via manager
            from .lora_store import LoraManager
            # the harness has a reference to the manager
            self.lora_harness._mgr.unload_from_engine(self.lora_harness._runtime)
            return {"ok": True, "result": {"unloaded": True}}
        return {"error": f"unknown lora tool: {name}"}

    # ── backup tools ──────────────────────────────────────────────────
    def _backup_tool(self, name: str, args: dict) -> dict:
        if self.backup_manager is None:
            return {"error": "backup tools not available"}
        if name == "list_backups":
            return {"backups": self.backup_manager.list_backups()}
        elif name == "create_backup":
            path = self.backup_manager.create_backup()
            if path:
                return {"created": True, "path": path}
            return {"error": "backup creation failed"}
        elif name == "load_backup":
            backup_path = args.get("backup_path", "")
            if not backup_path:
                return {"error": "backup_path required"}
            # this shows a confirmation dialog and freezes the agent
            ok = self.backup_manager.request_restore(backup_path)
            if ok:
                return {"restored": True, "path": backup_path}
            return {"error": "restore declined by user or failed"}
        return {"error": f"unknown backup tool: {name}"}

    # ── sub-agent tools ───────────────────────────────────────────────
    def _sub_agent_tool(self, name: str, args: dict) -> dict:
        if self.sub_agent_manager is None:
            return {"error": "sub-agent tools not available"}
        if name == "spawn_sub_agent":
            task_id = self.sub_agent_manager.spawn(
                prompt=args.get("prompt", ""),
                system_prompt=args.get("system_prompt", ""),
                temperature=args.get("temperature", 0.7),
                max_tokens=args.get("max_tokens", 512))
            return {"task_id": task_id, "status": "spawned"}
        elif name == "spawn_sub_agents":
            tasks = args.get("tasks", [])
            if not tasks:
                return {"error": "tasks list required"}
            ids = self.sub_agent_manager.spawn_batch(tasks)
            return {"task_ids": ids, "count": len(ids)}
        elif name == "check_sub_agent":
            task_id = args.get("task_id", "")
            task = self.sub_agent_manager.get_result(task_id)
            if task is None:
                return {"error": f"task not found: {task_id}"}
            return {"task_id": task.task_id, "status": task.status,
                    "result": task.result, "error": task.error,
                    "elapsed_s": round(task.elapsed_s, 2)}
        elif name == "wait_sub_agents":
            tasks = self.sub_agent_manager.wait_all()
            return {"tasks": [
                {"task_id": t.task_id, "status": t.status,
                 "result": t.result, "error": t.error}
                for t in tasks.values()]}
        elif name == "list_sub_agents":
            return {"tasks": self.sub_agent_manager.list_tasks()}
        return {"error": f"unknown sub-agent tool: {name}"}

    # ── time tools ────────────────────────────────────────────────────
    def _time_tool(self, name: str, args: dict) -> dict:
        if self.time_manager is None:
            return {"error": "time tools not available"}
        if name == "get_time":
            return self.time_manager.get_time()
        elif name == "set_timer":
            tid = self.time_manager.set_timer(
                seconds=args.get("seconds", 0),
                label=args.get("label", ""),
                message=args.get("message", ""),
                repeat=args.get("repeat", False),
                on_process_exit=args.get("on_process_exit", ""),
                on_user_prompt=args.get("on_user_prompt", False))
            return {"timer_id": tid, "set": True}
        elif name == "set_alarm":
            tid = self.time_manager.set_alarm(
                time_str=args.get("time", ""),
                label=args.get("label", ""),
                message=args.get("message", ""),
                repeat=args.get("repeat", False),
                on_process_exit=args.get("on_process_exit", ""),
                on_user_prompt=args.get("on_user_prompt", False))
            if not tid:
                return {"error": "invalid time format (use HH:MM or HH:MM AM/PM)"}
            return {"timer_id": tid, "set": True}
        elif name == "check_timer":
            result = self.time_manager.check_timer(args.get("timer_id", ""))
            if result is None:
                return {"error": f"timer not found: {args.get('timer_id', '')}"}
            return result
        elif name == "cancel_timer":
            ok = self.time_manager.cancel_timer(args.get("timer_id", ""))
            if not ok:
                return {"error": f"timer not found: {args.get('timer_id', '')}"}
            return {"cancelled": True}
        elif name == "list_timers":
            return {"timers": self.time_manager.list_timers()}
        return {"error": f"unknown time tool: {name}"}

    # ── library tools ─────────────────────────────────────────────────
    def _library_tool(self, name: str, args: dict) -> dict:
        if self.library_manager is None:
            return {"error": "library tools not available"}
        if name == "install_library":
            return self.library_manager.request_install(args.get("package", ""))
        elif name == "check_library":
            import importlib
            pkg = args.get("package", "").split("=")[0].split(">")[0].split("<")[0].strip()
            pkg_norm = pkg.replace("-", "_")
            try:
                mod = importlib.import_module(pkg_norm)
                ver = getattr(mod, "__version__", "installed")
                return {"package": pkg, "installed": True, "version": ver}
            except ImportError:
                return {"package": pkg, "installed": False}
        elif name == "list_allowed_libraries":
            return {"allowed": self.library_manager.get_allowlist()}
        return {"error": f"unknown library tool: {name}"}

    # ── helpers ───────────────────────────────────────────────────────
    def _record(self, name: str, args: dict, result: dict,
                ok: bool, t0: float) -> dict:
        import time as _time
        rec = {"name": name, "args": args, "ok": ok,
               "elapsed_s": round(_time.perf_counter() - t0, 3),
               "result": result}
        self.sandbox.calls.append(rec)
        return rec

    def summary(self) -> dict:
        s = self.sandbox.summary()
        s["read_only"] = self.read_only
        s["strikes"] = self.strikes.summary()
        if self.lorebook:
            s["memory"] = self.lorebook.stats()
        if self.mcp_manager is not None:
            s["mcp"] = self.mcp_manager.status()
        return s

    def reset_strikes(self) -> None:
        self.strikes.reset()
