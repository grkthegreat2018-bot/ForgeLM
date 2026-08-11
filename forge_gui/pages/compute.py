"""Compute page — GPU topology, VRAM allocator, kernel/compile status, executor."""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..api.gpu_monitor import GpuMonitor
from ..theme import Palette
from ..widgets.chart import LiveLineChart
from ..widgets.color_log import ColorLogWidget
from ..widgets.gauge import CircularGauge
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label


class ComputePage(QWidget):
    def __init__(self, gpu: GpuMonitor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gpu = gpu
        self._prev_vram_pct = 0.0
        self._prev_free_gb = 0.0
        self._log_lines = 0

        # ---- gauges ----
        self._g_alloc = CircularGauge(caption="ALLOCATED", size=160)
        self._g_reserved = CircularGauge(caption="RESERVED", size=160,
                                         color_a=Palette.chart_lr, color_b=Palette.warn)
        self._g_free = CircularGauge(caption="FREE", size=160,
                                     color_a=Palette.ok, color_b=Palette.chart_div)
        grow = QHBoxLayout(); grow.setSpacing(28); grow.addStretch(1)
        for g in (self._g_alloc, self._g_reserved, self._g_free):
            grow.addWidget(g, 0, Qt.AlignmentFlag.AlignCenter)
        grow.addStretch(1)
        g_host = QWidget(); g_host.setLayout(grow)

        # ---- KPI cards ----
        self._c_name = MetricCard("Device", "—")
        self._c_cuda = MetricCard("CUDA", "—")
        self._c_total = MetricCard("VRAM total", "—", unit="GB")
        self._c_alloc = MetricCard("Allocated", "—", unit="GB", spark_color=Palette.accent)
        self._c_reserved = MetricCard("Reserved", "—", unit="GB", spark_color=Palette.chart_lr)
        self._c_free = MetricCard("Free", "—", unit="GB", spark_color=Palette.ok)
        cards = card_grid([self._c_name, self._c_cuda, self._c_total,
                           self._c_alloc, self._c_reserved, self._c_free], cols=3)

        # ---- VRAM bar ----
        bar_card = QFrame(); bar_card.setObjectName("card")
        bl = QVBoxLayout(bar_card); bl.setContentsMargins(16,14,16,16); bl.setSpacing(8)
        bl.addWidget(section_label("VRAM ALLOCATOR"))
        self._bar = QProgressBar(); self._bar.setRange(0, 1000); self._bar.setTextVisible(True)
        self._bar.setFixedHeight(22)
        bl.addWidget(self._bar)
        legend = QHBoxLayout(); legend.setSpacing(20)
        for lbl, col in [("Allocated", Palette.accent), ("Reserved", Palette.chart_lr),
                         ("Free", Palette.ok)]:
            dot = QLabel("●"); dot.setStyleSheet(f"color:{col}; font-size:14px;")
            t = QLabel(lbl); t.setStyleSheet("color:#8b96a8; font-size:11px;")
            legend.addWidget(dot); legend.addWidget(t)
        legend.addStretch(1)
        bl.addLayout(legend)

        # ---- live VRAM chart ----
        self._chart = LiveLineChart(
            series=[("alloc", Palette.accent), ("reserved", Palette.chart_lr)],
            window=180, y_label="GB", height=180)
        chart_card = QFrame(); chart_card.setObjectName("card")
        cl = QVBoxLayout(chart_card); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        ch = QLabel("VRAM TIMELINE"); ch.setObjectName("cardTitle")
        cl.addWidget(ch); cl.addWidget(self._chart)

        # ---- torch info panel ----
        self._torch_info = QLabel("—")
        self._torch_info.setObjectName("cardBody"); self._torch_info.setWordWrap(True)
        info_card = QFrame(); info_card.setObjectName("card")
        il = QVBoxLayout(info_card); il.setContentsMargins(0,0,0,0)
        ih = QLabel("RUNTIME INFO"); ih.setObjectName("cardTitle")
        il.addWidget(ih); il.addWidget(self._torch_info)

        # ---- compute event log ----
        self._log = ColorLogWidget(max_lines=200)
        log_card = QFrame(); log_card.setObjectName("card")
        ll = QVBoxLayout(log_card); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(0)
        lh = QLabel("COMPUTE LOG"); lh.setObjectName("cardTitle")
        ll.addWidget(lh); ll.addWidget(self._log)

        self._host = page_container(g_host, cards, bar_card, chart_card, info_card, log_card)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(self._host)

    def refresh(self) -> None:
        gs = self._gpu.snapshot()
        if not gs.available:
            self._c_name.set_value("CUDA unavailable")
            self._c_cuda.set_value("—"); self._c_total.set_value("—", unit="GB")
            self._c_alloc.set_value("—", unit="GB"); self._c_reserved.set_value("—", unit="GB")
            self._c_free.set_value("—", unit="GB")
            self._g_alloc.set_value(0); self._g_reserved.set_value(0); self._g_free.set_value(0)
            self._bar.setValue(0); self._bar.setFormat("CUDA unavailable")
            self._torch_info.setText("CUDA not available — running on CPU.")
            if self._log_lines == 0:
                self._log.append_line("CUDA unavailable — running on CPU.", level="crash", bold=True)
                self._log_lines += 1
            return

        # Log significant VRAM changes
        self._log_vram_events(gs)

        self._c_name.set_value(gs.device_name)
        self._c_cuda.set_value(gs.cuda_version or "—")
        self._c_total.set_value(f"{gs.vram_total_gb:.2f}", unit="GB")
        self._c_alloc.set_value(f"{gs.vram_allocated_gb:.2f}", unit="GB")
        self._c_alloc.push_spark(gs.vram_allocated_gb)
        self._c_reserved.set_value(f"{gs.vram_reserved_gb:.2f}", unit="GB")
        self._c_reserved.push_spark(gs.vram_reserved_gb)
        self._c_free.set_value(f"{gs.vram_free_gb:.2f}", unit="GB")
        self._g_alloc.set_value(gs.vram_pct)
        reserved_pct = (gs.vram_reserved_gb / max(1e-9, gs.vram_total_gb)) * 100.0
        self._g_reserved.set_value(reserved_pct)
        free_pct = 100.0 - reserved_pct
        self._g_free.set_value(free_pct)
        self._bar.setValue(int(reserved_pct * 10))
        self._bar.setFormat(f"{gs.vram_reserved_gb:.2f} / {gs.vram_total_gb:.2f} GB  ({reserved_pct:.1f}%)")
        self._chart.push_multi({"alloc": gs.vram_allocated_gb,
                                "reserved": gs.vram_reserved_gb})
        info = [
            f"device:        {gs.device_name}  (cuda:{gs.index})",
            f"cuda version:  {gs.cuda_version or '—'}",
            f"allocated:     {gs.vram_allocated_gb:.3f} GB",
            f"reserved:      {gs.vram_reserved_gb:.3f} GB",
            f"free:          {gs.vram_free_gb:.3f} GB",
            f"total:         {gs.vram_total_gb:.3f} GB",
        ]
        try:
            import torch
            info.append(f"torch:         {torch.__version__}")
            if torch.cuda.is_available():
                info.append(f"compute cap:   {torch.cuda.get_device_capability()}")
                info.append(f"streams:       {torch.cuda.Stream().query() and 'ok'}")
        except Exception:
            pass
        self._torch_info.setText("\n".join(info))

    def _log_vram_events(self, gs) -> None:
        """Log significant VRAM changes with color coding.

        Only logs when there's a meaningful change (>5% delta or threshold crossings)
        to avoid flooding the log on every 500ms refresh tick.
        """
        tstr = time.strftime("%H:%M:%S")
        vram_pct = gs.vram_pct
        free_gb = gs.vram_free_gb
        delta_pct = abs(vram_pct - self._prev_vram_pct)

        # Only log on significant changes or threshold crossings
        should_log = False
        level = "info"
        msg = ""

        if free_gb < 0.5 and self._prev_free_gb >= 0.5:
            # Crossed into critical free VRAM territory
            should_log = True
            level = "crash"
            msg = f"[{tstr}] VRAM CRITICAL: only {free_gb:.2f} GB free ({vram_pct:.1f}% used)"
        elif free_gb < 1.0 and self._prev_free_gb >= 1.0:
            should_log = True
            level = "error"
            msg = f"[{tstr}] VRAM LOW: {free_gb:.2f} GB free ({vram_pct:.1f}% used)"
        elif delta_pct > 5.0:
            should_log = True
            if vram_pct > self._prev_vram_pct:
                level = "warn"
                msg = f"[{tstr}] VRAM +{delta_pct:.1f}% → {vram_pct:.1f}% ({gs.vram_allocated_gb:.2f} GB alloc)"
            else:
                level = "ok"
                msg = f"[{tstr}] VRAM -{delta_pct:.1f}% → {vram_pct:.1f}% ({gs.vram_allocated_gb:.2f} GB alloc)"
        elif self._log_lines == 0:
            # First log line — report initial state
            should_log = True
            level = "info"
            msg = f"[{tstr}] GPU ready: {gs.device_name} | {gs.vram_allocated_gb:.2f}/{gs.vram_total_gb:.2f} GB | {free_gb:.2f} GB free"

        if should_log:
            self._log.append_line(msg, level=level, bold=(level in ("crash", "phase")))
            self._log_lines += 1

        self._prev_vram_pct = vram_pct
        self._prev_free_gb = free_gb
