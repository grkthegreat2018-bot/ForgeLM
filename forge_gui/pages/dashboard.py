"""Dashboard page — top-level overview: GPU gauges, active runs, recent activity."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..api.gpu_monitor import GpuMonitor
from ..api.models_index import ModelsIndex
from ..api.status_reader import StatusReader
from ..theme import Palette
from ..widgets.gauge import CircularGauge
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label, status_tag


class DashboardPage(QWidget):
    request_open = Signal(int)  # ask app to switch page index

    def __init__(self, gpu: GpuMonitor, status_reader: StatusReader,
                 models_index: ModelsIndex, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gpu = gpu
        self._status = status_reader
        self._models = models_index

        # ---- top: GPU gauges row ----
        self._vram_gauge = CircularGauge(caption="VRAM", unit="%", size=150)
        self._compute_gauge = CircularGauge(caption="COMPUTE", unit="%", size=150,
                                            color_a=Palette.chart_div, color_b=Palette.chart_kl)
        self._disk_gauge = CircularGauge(caption="CHECKPOINTS", unit="%",
                                         max_val=100, size=150,
                                         color_a=Palette.ok, color_b=Palette.chart_div)
        gauges_row = QHBoxLayout(); gauges_row.setSpacing(28)
        gauges_row.addStretch(1)
        for g in (self._vram_gauge, self._compute_gauge, self._disk_gauge):
            gauges_row.addWidget(g, 0, Qt.AlignmentFlag.AlignCenter)
        gauges_row.addStretch(1)
        gauges_host = QWidget(); gauges_host.setLayout(gauges_row)

        # ---- KPI cards ----
        self._card_runs = MetricCard("Active Runs", "0", spark_color=Palette.accent)
        self._card_loss = MetricCard("Latest Loss", "—", spark_color=Palette.chart_loss)
        self._card_tps = MetricCard("Tokens / s", "—", spark_color=Palette.chart_reward)
        self._card_models = MetricCard("Checkpoints", "0")
        self._card_vram = MetricCard("VRAM Used", "—", unit="GB",
                                     spark_color=Palette.chart_lr)
        self._card_uptime = MetricCard("Session", "00:00")
        cards = card_grid([self._card_runs, self._card_loss, self._card_tps,
                           self._card_models, self._card_vram, self._card_uptime], cols=3)

        # ---- active runs list ----
        runs_card = QFrame(); runs_card.setObjectName("card")
        rc_lay = QVBoxLayout(runs_card); rc_lay.setContentsMargins(16, 14, 16, 16)
        rc_lay.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(section_label("Active Training Runs"))
        head.addStretch(1)
        self._runs_count = QLabel("0"); self._runs_count.setObjectName("cardUnit")
        head.addWidget(self._runs_count)
        rc_lay.addLayout(head)
        self._runs_host = QVBoxLayout(); self._runs_host.setSpacing(8)
        rc_lay.addLayout(self._runs_host)
        self._runs_card = runs_card

        # ---- recent activity ----
        self._activity = QLabel("No recent activity.")
        self._activity.setObjectName("cardBody")
        self._activity.setWordWrap(True)
        act_card = QFrame(); act_card.setObjectName("card")
        a_lay = QVBoxLayout(act_card); a_lay.setContentsMargins(0, 0, 0, 0)
        act_head = QLabel("RECENT ACTIVITY"); act_head.setObjectName("cardTitle")
        a_lay.addWidget(act_head)
        a_lay.addWidget(self._activity)

        sections = [gauges_host, cards, runs_card, act_card]
        self._host = page_container(*sections)

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._host)

        self._activity_log: list[str] = []
        self._start_ts = 0.0

    def refresh(self) -> None:
        import time
        if self._start_ts == 0.0:
            self._start_ts = time.time()
        gs = self._gpu.snapshot()
        self._vram_gauge.set_value(gs.vram_pct)
        self._compute_gauge.set_value(min(100.0, gs.vram_pct * 1.1))  # proxy
        self._card_vram.set_value(f"{gs.vram_allocated_gb:.2f}", unit="GB")
        self._card_vram.push_spark(gs.vram_allocated_gb)
        # checkpoints disk usage proxy: count * 5% capped
        models = self._models.models()
        self._card_models.set_value(str(len(models)))
        self._disk_gauge.set_value(min(100.0, len(models) * 8))
        # runs
        runs = self._status.snapshot()
        active = [r for r in runs if r.is_live]
        self._card_runs.set_value(str(len(active)))
        self._runs_count.setText(f"{len(active)} live / {len(runs)} total")
        self._rebuild_runs(active[:6])
        if runs:
            latest = runs[0]
            self._card_loss.set_value(f"{latest.loss:.4f}")
            self._card_loss.push_spark(latest.loss)
        # session uptime
        up = int(time.time() - self._start_ts)
        self._card_uptime.set_value(f"{up//60:02d}:{up%60:02d}")
        # activity
        for r in active:
            line = f"[{r.status}] {r.name} — step {r.step}/{r.max_steps} loss {r.loss:.4f}"
            if line not in self._activity_log[:20]:
                self._activity_log.insert(0, line)
        self._activity_log = self._activity_log[:20]
        self._activity.setText("\n".join(self._activity_log) or "No recent activity.")

    def _rebuild_runs(self, runs) -> None:
        # clear
        while self._runs_host.count():
            it = self._runs_host.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        if not runs:
            empty = QLabel("No active runs. Start training with --status-file to see it here.")
            empty.setObjectName("cardEmpty")
            self._runs_host.addWidget(empty)
            return
        for r in runs:
            row = self._build_run_row(r)
            self._runs_host.addWidget(row)

    def _build_run_row(self, r) -> QWidget:
        host = QFrame(); host.setObjectName("cardAlt")
        h = QHBoxLayout(host); h.setContentsMargins(14, 10, 14, 10); h.setSpacing(12)
        tag = status_tag(r.status, "ok" if r.is_live else ("warn" if r.status == "done" else "idle"))
        h.addWidget(tag)
        name = QLabel(r.name); name.setStyleSheet("color:#e6edf3; font-weight:600; font-size:13px;")
        h.addWidget(name)
        h.addStretch(1)
        meta = QLabel(f"step {r.step}/{r.max_steps}  ·  loss {r.loss:.4f}  ·  {r.heartbeat_age_s:.0f}s ago")
        meta.setStyleSheet("color:#8b96a8; font-size:12px;")
        h.addWidget(meta)
        bar = QProgressBar(); bar.setFixedWidth(160); bar.setRange(0, 100)
        bar.setValue(int(r.progress_pct)); bar.setTextVisible(True)
        h.addWidget(bar)
        return host
