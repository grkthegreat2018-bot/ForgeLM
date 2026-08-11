"""ForgeAI GUI application shell — QMainWindow with sidebar nav + stacked pages."""
from __future__ import annotations

import logging
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

from .api.gpu_monitor import GpuMonitor
from .api.log_tailer import LogTailer
from .api.models_index import ModelsIndex
from .api.process_manager import ProcessManager
from .api.status_reader import StatusReader, project_root
from .pages.chat import ChatPage
from .pages.compute import ComputePage
from .pages.dashboard import DashboardPage
from .pages.generations import GenerationsPage
from .pages.launch import LaunchPage
from .pages.logs import LogsPage
from .pages.models import ModelsPage
from .pages.selfplay import SelfPlayPage
from .pages.tasks import TasksPage
from .pages.training import TrainingPage
from .theme import Palette, apply_theme
from .widgets.sidebar import NavSidebar

logger = logging.getLogger(__name__)

PAGES = [
    ("Dashboard", "◎"),
    ("Self-Play", "⚡"),
    ("Training Live", "📈"),
    ("Generations", "✦"),
    ("Models", "❖"),
    ("Launch", "▶"),
    ("Tasks", "☰"),
    ("Compute", "◈"),
    ("Logs", "≡"),
    ("Chat", "✉"),
]


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

        # ---- sidebar ----
        self.sidebar = NavSidebar(PAGES)
        self.sidebar.page_changed.connect(self._on_page_changed)

        # ---- pages ----
        self.pages = QStackedWidget(); self.pages.setObjectName("pages")
        self.page_dashboard = DashboardPage(self.gpu, self.status_reader, self.models_index)
        self.page_selfplay = SelfPlayPage(self.status_reader, self.proc_mgr)
        self.page_training = TrainingPage(self.status_reader)
        self.page_generations = GenerationsPage(self.models_index)
        self.page_models = ModelsPage(self.models_index)
        self.page_launch = LaunchPage(self.proc_mgr)
        self.page_tasks = TasksPage(self.proc_mgr, self.status_reader)
        self.page_compute = ComputePage(self.gpu)
        self.page_logs = LogsPage(self.log_tailer)
        self.page_chat = ChatPage()
        for p in (self.page_dashboard, self.page_selfplay, self.page_training,
                  self.page_generations, self.page_models, self.page_launch,
                  self.page_tasks, self.page_compute, self.page_logs,
                  self.page_chat):
            self.pages.addWidget(p)

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
        # Ctrl+1..9 switch pages
        for i in range(len(PAGES)):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.sidebar.select_page(idx))
        # Ctrl+R force refresh of the current page
        sc_refresh = QShortcut(QKeySequence("Ctrl+R"), self)
        sc_refresh.activated.connect(self._force_refresh)

    def _force_refresh(self) -> None:
        page = self.pages.currentWidget()
        if hasattr(page, "refresh"):
            try:
                page.refresh()
            except Exception as e:
                logger.warning("force refresh failed on %s: %s",
                               type(page).__name__, e, exc_info=True)

    def closeEvent(self, event) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("lastPage", self.pages.currentIndex())
        super().closeEvent(event)

    def _set_window_icon(self) -> None:
        icon_path = project_root() / "ForgeAI_Icon.png"
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _on_page_changed(self, idx: int) -> None:
        self.pages.setCurrentIndex(idx)
        name = PAGES[idx][0]
        self.page_title.setText(name)
        self.page_subtitle.setText(_SUBTITLES.get(name, ""))
        if hasattr(self, "_settings"):
            self._settings.setValue("lastPage", idx)
        # immediate refresh on page switch
        page = self.pages.widget(idx)
        if hasattr(page, "refresh"):
            page.refresh()

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
    "Self-Play": "Live self-play event feed · per-task progress · ETA · charts",
    "Training Live": "Real-time loss / lr / step / self-play metrics",
    "Generations": "Live token-by-token model generation stream",
    "Models": "Checkpoint browser + boot & test + registered configs",
    "Launch": "Boot training / self-play / benchmark processes via GUI",
    "Tasks": "Unified live feed of all running tasks with per-task detail",
    "Compute": "GPU topology · VRAM allocator · runtime info",
    "Logs": "Multi-source tailed log console with filters",
    "Chat": "Local LLM chat (OpenAI-compatible endpoint)",
}


def run() -> int:
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("ForgeAI Control Center")
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
