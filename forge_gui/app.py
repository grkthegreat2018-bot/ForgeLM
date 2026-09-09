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
    QSplashScreen,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .api.backup_manager import BackupManager
from .api.chat_store import ChatStore
from .api.engine_runtime import EngineRuntime
from .api.gpu_monitor import GpuMonitor, GpuPoller
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
from .api.web_tools import WebTools
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
from .theme import Palette, ThemeManager, apply_theme
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
        # R37: Only backends needed by the Dashboard (the first visible
        # page) or core infrastructure are constructed eagerly.  All
        # chat/agent/LoRA/MCP/backups/sub-agent/time/library/web backends
        # are lazy-constructed on first access via properties below —
        # this shaves several hundred ms of disk I/O + object creation
        # off boot time since most users never visit those pages in a
        # given session.
        self.gpu = GpuMonitor()
        self.status_reader = StatusReader()
        self.models_index = ModelsIndex()
        self.log_tailer = LogTailer()
        self.proc_mgr = ProcessManager(self)
        self.engine_runtime = EngineRuntime(self)
        # Lazy backends — None until first property access
        self._chat_store: ChatStore | None = None
        self._lora_mgr: LoraManager | None = None
        self._lorebook: Lorebook | None = None
        self._lora_harness: LoraHarness | None = None
        self._mcp_manager: MCPManager | None = None
        self._lora_training: LoraTrainingTrigger | None = None
        self._backup_manager: BackupManager | None = None
        self._sub_agent_manager: SubAgentManager | None = None
        self._time_manager: TimeManager | None = None
        self._library_manager: LibraryInstallManager | None = None
        self._web_tools: WebTools | None = None
        self._tool_harness: ToolHarness | None = None

        # R37: Background GPU poller — imports torch + queries CUDA in a
        # daemon thread so the UI event loop never blocks on the 1-3 s
        # CUDA runtime init.  Pages call gpu.cached_snapshot() instead of
        # gpu.snapshot() to read the latest cached stats.
        self._gpu_poller = GpuPoller(self.gpu, interval_s=2.0)
        self._gpu_poller.start()

        # ---- sidebar ----
        self.sidebar = NavSidebar(PAGES)
        self.sidebar.page_changed.connect(self._on_page_changed)

        # ---- pages (lazy construction) ----
        # R35-1: Pages are constructed on first visit, not at startup.
        # This reduces boot time and memory — most users don't visit all 14 pages.
        self.pages = QStackedWidget(); self.pages.setObjectName("pages")
        self._page_factories = self._build_page_factories()
        self._page_cache: dict[int, QWidget] = {}
        # Pre-construct only the Dashboard (first visible page)
        self._page_cache[0] = self._page_factories[0]()
        self.pages.addWidget(self._page_cache[0])
        # Add placeholder widgets for remaining pages (replaced on first visit)
        for i in range(1, len(self._page_factories)):
            ph = QWidget()  # lightweight placeholder
            self.pages.addWidget(ph)

        # cross-page navigation signals (connected on first page construction)

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
        # R36-1: Theme toggle button
        from PySide6.QtWidgets import QPushButton
        self.theme_btn = QPushButton("◐"); self.theme_btn.setFixedSize(32, 32)
        self.theme_btn.setToolTip("Toggle light/dark theme")
        self.theme_btn.clicked.connect(self._toggle_theme)
        tb.addWidget(self.theme_btn)
        # R36-4: Font size button
        self.font_btn = QPushButton("A"); self.font_btn.setFixedSize(32, 32)
        self.font_btn.setToolTip("Cycle font size (S/M/L/XL)")
        self.font_btn.clicked.connect(self._cycle_font_size)
        tb.addWidget(self.font_btn)

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
        # R37: Timers are NOT started here — they are started by
        # _start_timers() via a singleShot(800ms) so the window paints
        # fully before any refresh work runs.  This eliminates the
        # "window appears then freezes" effect caused by the first
        # refresh tick firing before the window manager has finished
        # compositing.
        self._tick = 0
        self._fast_timer = QTimer(self); self._fast_timer.setInterval(500)
        self._fast_timer.timeout.connect(self._refresh_fast)
        self._slow_timer = QTimer(self); self._slow_timer.setInterval(2000)
        self._slow_timer.timeout.connect(self._refresh_slow)
        QTimer.singleShot(800, self._start_timers)

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
        # Set the visible page immediately WITHOUT triggering refresh —
        # setCurrentIndex + title update are instant, so the window paints
        # the correct page on win.show() without waiting for GPU/CUDA init
        # or disk scans. The sidebar button state + first refresh are
        # deferred to the event loop (fires after win.show() but before the
        # 500ms fast timer), so the heavy refresh() runs on a visible window
        # instead of blocking the window from appearing.
        self.pages.setCurrentIndex(last_page)
        _name = _INDEX_TO_NAME.get(last_page, "")
        self.page_title.setText(_name)
        self.page_subtitle.setText(_SUBTITLES.get(_name, ""))
        QTimer.singleShot(0, lambda: self.sidebar.select_page(last_page))

        # R36-2: First-run onboarding (deferred so window appears first)
        QTimer.singleShot(500, self._maybe_show_onboarding)

        # NOTE: no eager _refresh_slow()/_refresh_fast() here — the timers
        # (started above) fire on the next event-loop tick after win.show(),
        # so the window appears before GPU/CUDA init or page scans run.

    # ---- lazy backend properties (R37) ----
    # Constructed on first access — defers disk I/O + object creation
    # for backends only needed by chat/agent/LoRA/etc pages.
    @property
    def chat_store(self) -> ChatStore:
        if self._chat_store is None:
            self._chat_store = ChatStore()
        return self._chat_store

    @property
    def lora_mgr(self) -> LoraManager:
        if self._lora_mgr is None:
            self._lora_mgr = LoraManager(self)
        return self._lora_mgr

    @property
    def lorebook(self) -> Lorebook:
        if self._lorebook is None:
            self._lorebook = Lorebook()
        return self._lorebook

    @property
    def lora_harness(self) -> LoraHarness:
        if self._lora_harness is None:
            self._lora_harness = LoraHarness(self.lora_mgr,
                                             self.engine_runtime, self)
        return self._lora_harness

    @property
    def mcp_manager(self) -> MCPManager:
        if self._mcp_manager is None:
            self._mcp_manager = MCPManager()
        return self._mcp_manager

    @property
    def lora_training(self) -> LoraTrainingTrigger:
        if self._lora_training is None:
            self._lora_training = LoraTrainingTrigger(
                proc_mgr=self.proc_mgr, chat_store=self.chat_store,
                checkpoint="research/checkpoints/ForgeLM_V2.safetensors")
        return self._lora_training

    @property
    def backup_manager(self) -> BackupManager:
        if self._backup_manager is None:
            self._backup_manager = BackupManager(project_root(), parent=self)
        return self._backup_manager

    @property
    def sub_agent_manager(self) -> SubAgentManager:
        if self._sub_agent_manager is None:
            self._sub_agent_manager = SubAgentManager(
                self.engine_runtime, parent=self)
        return self._sub_agent_manager

    @property
    def time_manager(self) -> TimeManager:
        if self._time_manager is None:
            self._time_manager = TimeManager(parent=self)
        return self._time_manager

    @property
    def library_manager(self) -> LibraryInstallManager:
        if self._library_manager is None:
            self._library_manager = LibraryInstallManager(parent=self)
        return self._library_manager

    @property
    def web_tools(self) -> WebTools:
        if self._web_tools is None:
            self._web_tools = WebTools(enabled=True)
        return self._web_tools

    @property
    def tool_harness(self) -> ToolHarness:
        if self._tool_harness is None:
            self._tool_harness = ToolHarness(
                workspace=str(project_root()),
                lorebook=self.lorebook,
                lora_harness=self.lora_harness,
                mcp_manager=self.mcp_manager,
                lora_training=self.lora_training,
                backup_manager=self.backup_manager,
                sub_agent_manager=self.sub_agent_manager,
                time_manager=self.time_manager,
                library_manager=self.library_manager,
                web_tools=self.web_tools,
                read_only=False,
                enable_safety=True,
            )
        return self._tool_harness

    def _start_timers(self) -> None:
        """R37: Start refresh timers after the window has been visible for
        ~800 ms so the first paint is never interrupted by a refresh tick."""
        self._fast_timer.start()
        self._slow_timer.start()

    def _maybe_show_onboarding(self) -> None:
        """R36-2: Show onboarding dialog on first run."""
        try:
            from .widgets.onboarding import maybe_show_onboarding
            maybe_show_onboarding(self)
        except Exception as e:
            logger.debug("onboarding skipped: %s", e)

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
        # R36-3: Ctrl+K command palette
        sc_palette = QShortcut(QKeySequence("Ctrl+K"), self)
        sc_palette.activated.connect(self._open_command_palette)
        # R36-1: Ctrl+Shift+T toggle theme
        sc_theme = QShortcut(QKeySequence("Ctrl+Shift+T"), self)
        sc_theme.activated.connect(self._toggle_theme)
        # R36-4: Ctrl+Shift+F cycle font size
        sc_font = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
        sc_font.activated.connect(self._cycle_font_size)

    def _toggle_theme(self) -> None:
        """R36-1: Toggle between dark and light themes."""
        new_theme = ThemeManager.toggle_theme()
        self.theme_btn.setText("◑" if new_theme == "light" else "◐")

    def _cycle_font_size(self) -> None:
        """R36-4: Cycle through font sizes S→M→L→XL→S."""
        sizes = list(ThemeManager.FONT_SIZES.keys())
        current = ThemeManager.current_font_size()
        idx = sizes.index(current) if current in sizes else 1
        next_size = sizes[(idx + 1) % len(sizes)]
        ThemeManager.set_font_size(next_size)
        self.font_btn.setText(f"A{next_size[0].upper()}")

    def _open_command_palette(self) -> None:
        """R36-3: Open the command palette dialog."""
        from .widgets.command_palette import CommandPalette
        palette = CommandPalette(self, _INDEX_TO_NAME)
        palette.page_selected.connect(self._navigate_to)
        palette.exec()

    def _force_refresh(self) -> None:
        page = self.pages.currentWidget()
        if hasattr(page, "refresh"):
            try:
                page.refresh()
            except Exception as e:
                logger.warning("force refresh failed on %s: %s",
                               type(page).__name__, e, exc_info=True)

    def closeEvent(self, event) -> None:
        # R37: stop the background GPU poller (daemon thread, but clean
        # shutdown avoids a brief join delay on exit).
        try:
            self._gpu_poller.stop(timeout_s=2.0)
        except Exception:
            pass
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
        self._ensure_page_constructed(idx)
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

    def _ensure_page_constructed(self, idx: int) -> None:
        """R35-1: Lazily construct a page on first visit."""
        if idx in self._page_cache:
            return
        if idx < 0 or idx >= len(self._page_factories):
            return
        page = self._page_factories[idx]()
        self._page_cache[idx] = page
        # Replace the placeholder widget in the stacked widget
        old_widget = self.pages.widget(idx)
        self.pages.removeWidget(old_widget)
        old_widget.deleteLater()
        self.pages.insertWidget(idx, page)
        # Connect cross-page navigation signals
        if hasattr(page, "request_open"):
            page.request_open.connect(self._navigate_to)
        if idx == _PAGE_INDEX.get("LoRA", -1):
            page.request_open.connect(
                lambda i: self._navigate_to(_PAGE_INDEX["Fine-Tune"])
                if i < 0 else self._navigate_to(i))

    def _build_page_factories(self) -> list:
        """R35-1: Factory closures for lazy page construction."""
        return [
            lambda: DashboardPage(self.gpu, self.status_reader, self.models_index),
            lambda: ChatPage(self.chat_store, self.engine_runtime,
                             self.models_index, self.lorebook,
                             self.lora_harness, self.tool_harness),
            lambda: AgentPage(self.engine_runtime, self.tool_harness,
                              self.lorebook),
            lambda: GenerationsPage(self.engine_runtime, self.models_index),
            lambda: EnginePage(self.engine_runtime, self.models_index),
            lambda: ModelsPage(self.models_index, self.engine_runtime),
            lambda: LoraPage(self.engine_runtime, self.lora_mgr,
                             self.models_index),
            lambda: FineTunePage(self.chat_store, self.proc_mgr),
            lambda: SelfPlayPage(self.status_reader, self.proc_mgr),
            lambda: TrainingPage(self.status_reader),
            lambda: LaunchPage(self.proc_mgr),
            lambda: TasksPage(self.proc_mgr, self.status_reader),
            lambda: ComputePage(self.gpu),
            lambda: LogsPage(self.log_tailer),
        ]

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
        # sidebar status — R37: use cached snapshot (populated by
        # background GpuPoller) so the UI thread never blocks on torch.
        gs = self.gpu.cached_snapshot()
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
    _install_crash_handlers()
    _force_utf8_stdio()
    app = QApplication(sys.argv)
    app.setApplicationName("ForgeAI Control Center")
    apply_theme(app)

    # R35-2: Splash screen during startup
    splash = QSplashScreen()
    splash_msg = "Loading ForgeAI…"
    splash.setStyleSheet(
        "QSplashScreen { background: #1a1a2e; color: #e0e0e0; }"
        "QLabel { color: #e0e0e0; font-size: 14px; padding: 20px; }"
    )
    splash.showMessage(splash_msg, 0x84)  # AlignBottom | AlignHCenter
    splash.show()
    app.processEvents()

    win = MainWindow()
    win.show()
    splash.finish(win)
    return app.exec()


def _install_crash_handlers() -> None:
    """Leave a trace when the process dies unexpectedly.

    The GUI previously died with exit code 1 and no output (native crash in
    a CUDA/Qt worker). faulthandler captures segfaults; sys/threading
    excepthooks capture unhandled exceptions in Qt slots and QThreads —
    all appended to logs/crash.log.
    """
    import faulthandler
    import threading
    import traceback

    log_dir = project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    global _crash_log
    _crash_log = open(log_dir / "crash.log", "a", encoding="utf-8",
                      buffering=1)
    faulthandler.enable(_crash_log)

    def _log_exc(header: str, exc) -> None:
        try:
            _crash_log.write(f"\n{header}\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__,
                                      file=_crash_log)
            _crash_log.flush()
        except Exception:
            pass

    def _sys_hook(t, exc, tb):
        _log_exc(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] unhandled "
                 f"exception in main thread ({getattr(t, '__name__', t)}):",
                 exc)
        sys.__excepthook__(t, exc, tb)

    def _thread_hook(args):
        _log_exc(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] unhandled "
                 f"exception in thread {args.thread.name}:", args.exc)
        print(f"Unhandled exception in thread {args.thread.name}: "
              f"{args.exc!r}", file=sys.stderr)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


_crash_log = None


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
