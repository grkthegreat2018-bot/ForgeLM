"""Service registry — constructs and wires all backend services.

Mirrors the MainWindow backend graph from forge_gui/app.py: eager
services (gpu, status, models index, procs, engine runtime) plus lazy
stores (chat, lorebook, mcp, web tools, lora training, tool harness).
"""
from __future__ import annotations

import asyncio
import logging

from forge_gui.api.chat_store import ChatStore
from forge_gui.api.events_reader import EventsReader
from forge_gui.api.log_tailer import LogTailer
from forge_gui.api.lorebook import Lorebook
from forge_gui.api.master_prompt import (
    get_default_prompt_for_config, generate_master_prompt)
from forge_gui.api.mcp_client import MCPManager
from forge_gui.api.models_index import ModelsIndex
from forge_gui.api.status_reader import StatusReader, project_root
from forge_gui.api.web_tools import WebTools

from .hub import EventHub
from .services.agent_loop import AgentService
from .services.boot import BootService
from .services.engine_rt import EngineService
from .services.gpu import GpuMonitor
from .services.lora import LoraService
from .services.procs import ProcessService
from .services.support import (
    ApprovalChannel, BackupService, LibraryService, LoraHarnessAdapter,
    SubAgentService, TimeService)

logger = logging.getLogger(__name__)


class Services:
    def __init__(self) -> None:
        self.hub = EventHub()
        self.approvals = ApprovalChannel(self.hub)

        # eager
        self.gpu = GpuMonitor(interval_s=2.0)
        self.status_reader = StatusReader()
        self.models_index = ModelsIndex()
        self.log_tailer = LogTailer()
        self.events = EventsReader()
        self.procs = ProcessService(self.hub)
        self.engine = EngineService(self.hub)
        self.lora = LoraService(self.hub, self.engine)
        self.boot = BootService(self.hub, self.engine)
        self.agent = AgentService(self.hub, self.engine)
        self.sub_agents = SubAgentService(self.hub, self.engine)
        self.timers = TimeService(self.hub)
        self.backups = BackupService(self.hub, self.approvals)
        self.library = LibraryService(self.hub, self.approvals)

        # lazy
        self._chat_store: ChatStore | None = None
        self._lorebook: Lorebook | None = None
        self._mcp: MCPManager | None = None
        self._web_tools: WebTools | None = None
        self._lora_training = None
        self._lora_adapter: LoraHarnessAdapter | None = None

    # ── lazy stores ───────────────────────────────────────────────────
    @property
    def chat_store(self) -> ChatStore:
        if self._chat_store is None:
            self._chat_store = ChatStore()
        return self._chat_store

    @property
    def lorebook(self) -> Lorebook:
        if self._lorebook is None:
            self._lorebook = Lorebook()
        return self._lorebook

    @property
    def mcp(self) -> MCPManager:
        if self._mcp is None:
            self._mcp = MCPManager()
        return self._mcp

    @property
    def web_tools(self) -> WebTools:
        if self._web_tools is None:
            self._web_tools = WebTools(enabled=True)
        return self._web_tools

    @property
    def lora_training(self):
        if self._lora_training is None:
            from forge_gui.api.lora_training_trigger import (
                LoraTrainingTrigger)
            self._lora_training = LoraTrainingTrigger(
                proc_mgr=self.procs, chat_store=self.chat_store,
                checkpoint="research/checkpoints/ForgeLM_V2.safetensors")
        return self._lora_training

    def lora_adapter(self, loop) -> LoraHarnessAdapter:
        if self._lora_adapter is None:
            self._lora_adapter = LoraHarnessAdapter(
                self.lora, self.engine, loop)
        return self._lora_adapter

    # ── tool harness factory ──────────────────────────────────────────
    def make_harness(self, workspace: str | None = None,
                     read_only: bool = False):
        from forge_gui.api.tool_harness import ToolHarness
        loop = asyncio.get_event_loop()
        return ToolHarness(
            workspace=workspace or str(project_root()),
            lorebook=self.lorebook,
            lora_harness=self.lora_adapter(loop),
            mcp_manager=self.mcp,
            lora_training=self.lora_training,
            backup_manager=self.backups,
            sub_agent_manager=self.sub_agents,
            time_manager=self.timers,
            library_manager=self.library,
            web_tools=self.web_tools,
            read_only=read_only,
            enable_safety=True,
        )

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.hub.bind_loop(loop)
        self.procs.bind_loop(loop)
        self.engine.bind_loop(loop)
        self.agent.bind_loop(loop)
        self.gpu.start(self.hub)
        self.agent.set_harness_factory(
            lambda workspace=None: self.make_harness(workspace))

    async def stop(self) -> None:
        self.backups.stop()
        self.timers.shutdown()
        self.sub_agents.shutdown()
        await self.gpu.stop()
        await self.procs.shutdown()
        await self.engine.shutdown()


services = Services()
