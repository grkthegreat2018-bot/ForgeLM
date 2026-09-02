"""ForgeAI GUI application shell — QMainWindow with sidebar nav + stacked pages."""
from __future__ import annotations

import logging
import sys
import time

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .api.backup_manager import BackupManager
from .api.chat_store import ChatStore
from .api.engine_runtime import EngineRuntime
from .api.gpu_monitor import GpuMonitor
from .api.library_install import LibraryInstallManager
from .api.log_tailer import LogTailer
from .api.lorebook import Lorebook
from .api.lora_store import LoraHarness, LoraManager
from .api.mcp_client import MCPManager
from .api.models_index import ModelsIndex
from .api.process_manager import ProcessManager
from .api.status_reader import StatusReader, project_root
from .api.sub_agent import SubAgentManager
from .api.time_manager import TimeManager
from .api.tool_harness import ToolHarness
from .api.lora_training_trigger import LoraTrainingTrigger
from .pages.agent import AgentPage
from .pages.chat import ChatPage
from .pages.compute import ComputePage
from .pages.dashboard import DashboardPage
from .pages.engine import EnginePage
from .pages.finetune import FineTunePage
from .pages.generations import GenerationsPage
from .pages.launch import LaunchPage
from .pages.logs import LogsPage
from .pages.lora import LoraPage
from .pages.models import ModelsPage
from .pages.selfplay import SelfPlayPage
from .pages.tasks import TasksPage
from .pages.training import TrainingPage
from .theme import Palette, apply_theme
from .widgets.sidebar import NavSidebar

logger = logging.getLogger(__name__)

# Grouped sidebar: ("__section__", title) inserts a non-selectable header.
# Page indices count only selectable buttons (headers are skipped).
PAGES = [
    ("__section__", "Workspace"),
    ("Dashboard", "◎"),
    ("Chat Studio", "✉"),
    ("Agent", "⌘"),
    ("Generations", "✦"),
    ("__section__", "Engine & Models"),
    ("Engine", "⚙"),
    ("Models", "❖"),
    ("LoRA", "◆"),
    ("__section__", "Train"),
    ("Fine-Tune", "⚒"),
    ("Self-Play", "⚡"),
    ("Training Live", "📈"),
    ("__section__", "System"),
    ("Launch", "▶"),
    ("Tasks", "☰"),
    ("Compute", "◈"),
    ("Logs", "≡"),
]

# Map page name → logical index (excluding section headers)
_PAGE_INDEX = {}
_INDEX_TO_NAME = {}
_idx = 0
for _label, _icon in PAGES:
    if _label != "__section__":
        _PAGE_INDEX[_label] = _idx
        _INDEX_TO_NAME[_idx] = _label
        _idx += 1
_NUM_PAGES = _idx


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ForgeAI")
        self.resize(1320, 860)
        self.setMinimumSize(1080, 680)
        self.setObjectName("root")
        self._set_window_icon()

        # ---- shared backends ----
        self.gpu = GpuMonitor()
        self.status_reader = StatusReader()
        self.models_index = ModelsIndex()
        self.log_tailer = LogTailer()
        self.proc_mgr = ProcessManager(self)
        self.chat_store = ChatStore()
        self.engine_runtime = EngineRuntime(self)
        self.lora_mgr = LoraManager(self)
        self.lorebook = Lorebook()
        self.lora_harness = LoraHarness(self.lora_mgr, self.engine_runtime, self)
        self.mcp_manager = MCPManager()
        self.lora_training = LoraTrainingTrigger(
            proc_mgr=self.proc_mgr, chat_store=self.chat_store,
            checkpoint="research/checkpoints/ForgeLM_V2_Light.safetensors")
        self.backup_manager = BackupManager(project_root(), parent=self)
        self.sub_agent_manager = SubAgentManager(self.engine_runtime, parent=self)
        self.time_manager = TimeManager(parent=self)
        self.library_manager = LibraryInstallManager(parent=self)
        # shared tool harness for agent mode (full access)
        self.tool_harness = ToolHarness(
            workspace=str(project_root()),
            lorebook=self.lorebook,
            lora_harness=self.lora_harness,
            mcp_manager=self.mcp_manager,
            lora_training=self.lora_training,
            backup_manager=self.backup_manager,
            sub_agent_manager=self.sub_agent_manager,
            time_manager=self.time_manager,
            library_manager=self.library_manager,
            read_only=False,
            enable_safety=True,
        )

        # ---- sidebar ----
        self.sidebar = NavSidebar(PAGES)
        self.sidebar.page_changed.connect(self._on_page_changed)

        # ---- pages ----
        # Order MUST match PAGES (excluding section headers).
        self.pages = QStackedWidget(); self.pages.setObjectName("pages")
        self.page_dashboard = DashboardPage(self.gpu, self.status_reader, self.models_index)
        self.page_chat = ChatPage(self.chat_store, self.engine_runtime,
                                  self.models_index, self.lorebook,
                                  self.lora_harness, self.tool_harness)
        self.page_agent = AgentPage(self.engine_runtime, self.tool_harness,
                                    self.lorebook)
        self.page_generations = GenerationsPage(self.engine_runtime, self.models_index)
        self.page_engine = EnginePage(self.engine_runtime, self.models_index)
        self.page_models = ModelsPage(self.models_index, self.engine_runtime)
        self.page_lora = LoraPage(self.engine_runtime, self.lora_mgr,
                                  self.models_index)
        self.page_finetune = FineTunePage(self.chat_store, self.proc_mgr)
        self.page_selfplay = SelfPlayPage(self.status_reader, self.proc_mgr)
        self.page_training = TrainingPage(self.status_reader)
        self.page_launch = LaunchPage(self.proc_mgr)
        self.page_tasks = TasksPage(self.proc_mgr, self.status_reader)
        self.page_compute = ComputePage(self.gpu)
        self.page_logs = LogsPage(self.log_tailer)
        for p in (self.page_dashboard, self.page_chat, self.page_agent,
                  self.page_generations, self.page_engine, self.page_models,
                  self.page_lora, self.page_finetune, self.page_selfplay,
                  self.page_training, self.page_launch, self.page_tasks,
                  self.page_compute, self.page_logs):
            self.pages.addWidget(p)

        # cross-page navigation signals
        self.page_models.request_open.connect(self._navigate_to)
        self.page_lora.request_open.connect(
            lambda idx: self._navigate_to(_PAGE_INDEX["Fine-Tune"])
            if idx < 0 else self._navigate_to(idx))

        # ---- topbar ----
        topbar = QFrame(); topbar.setObjectName("topbar"); topbar.setFixedHeight(64)
        tb = QHBoxLayout(topbar); tb.setContentsMargins(24, 0, 24, 0); tb.setSpacing(12)
        title_col = QVBoxLayout(); title_col.setSpacing(0); title_col.setContentsMargins(0,0,0,0)
        self.page_title = QLabel("Dashboard"); self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel("System overview"); self.page_subtitle.setObjectName("pageSubtitle")
        title_col.addWidget(self.page_title); title_col.addWidget(self.page_subtitle)
        tb.addLayout(title_col); tb.addStretch(1)
        self.live_dot = QLabel("● LIVE"); self.live_dot.setObjectName("liveDot")
        tb.addWidget(self.live_dot)
        self.clock = QLabel("--:--:--"); self.clock.setObjectName("clock")
        tb.addWidget(self.clock)

        # ---- layout ----
        central = QWidget(); central.setObjectName("root")
        cl = QVBoxLayout(central); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        cl.addWidget(topbar)
        body = QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)
        body.addWidget(self.sidebar)
        body.addWidget(self.pages, 1)
        cl.addLayout(body)
        self.setCentralWidget(central)

        # ---- refresh timers ----
        # Fast timer (500ms): refresh ONLY the visible page for responsive
        # live updates (self-play event feed, charts). Slow timer (2000ms):
        # GPU snapshot + sidebar status + background pages (every 4th slow
        # tick = ~8s) to avoid unnecessary work on hidden pages.
        self._tick = 0
        self._fast_timer = QTimer(self); self._fast_timer.setInterval(500)
        self._fast_timer.timeout.connect(self._refresh_fast)
        self._fast_timer.start()
        self._slow_timer = QTimer(self); self._slow_timer.setInterval(2000)
        self._slow_timer.timeout.connect(self._refresh_slow)
        self._slow_timer.start()

        # ---- keyboard shortcuts ----
        self._setup_shortcuts()

        # ---- persisted UI state ----
        self._settings = QSettings("ForgeAI", "ForgeGUI")
        geo = self._settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        try:
            last_page = int(self._settings.value("lastPage", 0) or 0)
        except (TypeError, ValueError):
            last_page = 0
        self.sidebar.select_page(last_page)

        self._refresh_slow()
        self._refresh_fast()

    def _setup_shortcuts(self) -> None:
        # Ctrl+1..N switch pages (N = number of selectable pages)
        for i in range(_NUM_PAGES):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.sidebar.select_page(idx))
        # Ctrl+R force refresh of the current page
        sc_refresh = QShortcut(QKeySequence("Ctrl+R"), self)
        sc_refresh.activated.connect(self._force_refresh)
        # Ctrl+B toggle sidebar collapse
        sc_sidebar = QShortcut(QKeySequence("Ctrl+B"), self)
        sc_sidebar.activated.connect(self.sidebar.toggle_collapse)

    def _force_refresh(self) -> None:
        page = self.pages.currentWidget()
        if hasattr(page, "refresh"):
            try:
                page.refresh()
            except Exception as e:
                logger.warning("force refresh failed on %s: %s",
                               type(page).__name__, e, exc_info=True)

    def closeEvent(self, event) -> None:
        # wait for an in-flight engine load so the QThread is never
        # destroyed while still running
        try:
            self.engine_runtime.shutdown()
        except Exception as e:
            logger.warning("engine shutdown on close failed: %s", e)
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("lastPage", self.pages.currentIndex())
        super().closeEvent(event)

    def _set_window_icon(self) -> None:
        icon_path = project_root() / "ForgeAI_Icon.png"
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _on_page_changed(self, idx: int) -> None:
        self.pages.setCurrentIndex(idx)
        name = _INDEX_TO_NAME.get(idx, "")
        self.page_title.setText(name)
        self.page_subtitle.setText(_SUBTITLES.get(name, ""))
        if hasattr(self, "_settings"):
            self._settings.setValue("lastPage", idx)
        # immediate refresh on page switch
        page = self.pages.widget(idx)
        if hasattr(page, "refresh"):
            page.refresh()

    def _navigate_to(self, idx: int) -> None:
        """Programmatic page switch (from cross-page signals)."""
        if 0 <= idx < self.pages.count():
            self.sidebar.select_page(idx)

    def _refresh_fast(self) -> None:
        """Fast tick (500ms): refresh only the visible page for live updates."""
        cur = self.pages.currentWidget()
        if hasattr(cur, "refresh"):
            try:
                cur.refresh()
            except Exception as e:
                logger.warning("fast refresh error on %s: %s",
                               type(cur).__name__, e, exc_info=True)

    def _refresh_slow(self) -> None:
        """Slow tick (2000ms): clock, GPU, sidebar, background pages."""
        self._tick += 1
        self.clock.setText(time.strftime("%H:%M:%S"))
        # Refresh background pages every 4th slow tick (~8s)
        cur = self.pages.currentWidget()
        if self._tick % 4 == 0:
            for i in range(self.pages.count()):
                page = self.pages.widget(i)
                if page is cur or not hasattr(page, "refresh"):
                    continue
                try:
                    page.refresh()
                except Exception as e:
                    logger.warning("bg refresh error on page %d (%s): %s",
                                   i, type(page).__name__, e, exc_info=True)
        # sidebar status
        gs = self.gpu.snapshot()
        if gs.available:
            self.sidebar.set_gpu(f"GPU {gs.vram_pct:.0f}% · {gs.vram_allocated_gb:.1f}GB")
            self.sidebar.set_status("● live", Palette.ok)
        else:
            self.sidebar.set_gpu("GPU offline")
            self.sidebar.set_status("● idle", Palette.text_faint)


_SUBTITLES = {
    "Dashboard": "System overview · live GPU, runs, recent activity",
    "Chat Studio": "Chat with the resident engine or any endpoint · rate replies → SFT data",
    "Agent": "Agentic coding loop · sandboxed tools · approval-gated writes",
    "Engine": "Resident ForgeEngine · Activation Studio · 60+ features · live stats",
    "Models": "Checkpoint browser + row actions + boot & test + registered configs",
    "LoRA": "Adapter library · hot-load / swap / merge · train new adapters",
    "Fine-Tune": "Full trainer params · LoRA / full FT · adapter-only save",
    "Self-Play": "Live self-play event feed · per-task progress · ETA · charts",
    "Training Live": "Real-time loss / lr / step / self-play metrics",
    "Generations": "Live token-by-token model generation stream",
    "Launch": "Boot training / self-play / benchmark processes via GUI",
    "Tasks": "Unified live feed of all running tasks with per-task detail",
    "Compute": "GPU topology · VRAM allocator · runtime info",
    "Logs": "Multi-source tailed log console with filters",
}


def run() -> int:
    _setup_logging()
    _force_utf8_stdio()
    app = QApplication(sys.argv)
    app.setApplicationName("ForgeAI Control Center")
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


def _force_utf8_stdio() -> None:
    """Windows consoles default to cp1252 — engine prints (→, ·) raise
    UnicodeEncodeError and silently skip warmup. Force UTF-8 everywhere."""
    import os
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _setup_logging() -> None:
    """Log to logs/gui.log + console so GUI issues are diagnosable.

    Engine output arrives via print() (not logging), so stdio is also
    teed into the same file — see _tee_stdio.
    """
    import logging.handlers
    log_dir = project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "gui.log", maxBytes=2_000_000, backupCount=3,
        encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # torch noise stays out of the GUI log
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _tee_stdio(log_dir / "gui.log")


class _Tee:
    """Write-through stream copy: console + gui.log (engine print() output)."""

    def __init__(self, original, log_file) -> None:
        self._original = original
        self._log = log_file

    def write(self, msg: str) -> None:
        try:
            self._original.write(msg)
        except Exception:
            pass
        try:
            self._log.write(msg)
            self._log.flush()
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):  # pass through encoding etc.
        return getattr(self._original, name)


def _tee_stdio(log_path) -> None:
    import sys
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


if __name__ == "__main__":
    raise SystemExit(run())
