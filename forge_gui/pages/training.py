"""Training Live page — real-time loss/lr/step charts + self-play monitor metrics."""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from ..api.status_reader import RunSnapshot, StatusReader
from ..theme import Palette
from ..widgets.chart import LiveLineChart
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label, status_tag

logger = logging.getLogger(__name__)


class TrainingPage(QWidget):
    def __init__(self, status_reader: StatusReader, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._status = status_reader
        self._runs: list[RunSnapshot] = []
        self._selected_id: Optional[str] = None
        self._loss_hist: list[float] = []
        self._lr_hist: list[float] = []

        # ---- run selector ----
        sel_row = QHBoxLayout(); sel_row.setSpacing(10)
        sel_row.addWidget(section_label("RUN"))
        self._combo = QComboBox(); self._combo.setMinimumWidth(320)
        self._combo.currentIndexChanged.connect(self._on_select)
        sel_row.addWidget(self._combo)
        sel_row.addStretch(1)
        self._refresh_btn = QPushButton("Refresh runs")
        sel_row.addWidget(self._refresh_btn)
        self._stop_btn = QPushButton("Stop run"); self._stop_btn.setObjectName("danger")
        sel_row.addWidget(self._stop_btn)
        sel_host = QWidget(); sel_host.setLayout(sel_row)

        # ---- KPI cards ----
        self._c_step = MetricCard("Step", "0 / 0")
        self._c_loss = MetricCard("Loss", "—", spark_color=Palette.chart_loss, spark_window=160)
        self._c_lr = MetricCard("Learning rate", "—", spark_color=Palette.chart_lr, spark_window=160)
        self._c_vram = MetricCard("VRAM", "—", unit="GB", spark_color=Palette.chart_div)
        self._c_eta = MetricCard("Progress", "0%")
        self._c_age = MetricCard("Heartbeat", "—")
        cards = card_grid([self._c_step, self._c_loss, self._c_lr,
                           self._c_vram, self._c_eta, self._c_age], cols=3)

        # ---- charts ----
        self._loss_chart = LiveLineChart(
            series=[("loss", Palette.chart_loss)], window=240,
            y_label="loss", height=200)
        self._lr_chart = LiveLineChart(
            series=[("lr", Palette.chart_lr)], window=240,
            y_label="lr", height=160)
        self._reward_chart = LiveLineChart(
            series=[("reward", Palette.chart_reward), ("intrinsic", Palette.chart_kl),
                    ("grounded", Palette.chart_div)],
            window=240, y_label="reward", height=180)
        self._acr_chart = LiveLineChart(
            series=[("acr", Palette.err), ("diversity", Palette.chart_div),
                    ("kl", Palette.chart_kl)],
            window=240, y_label="self-play", height=180)

        # wrap charts in cards
        loss_card = self._chart_card("LOSS CURVE", self._loss_chart)
        lr_card = self._chart_card("LEARNING RATE", self._lr_chart)
        reward_card = self._chart_card("SELF-PLAY REWARDS", self._reward_chart)
        acr_card = self._chart_card("ACR · DIVERSITY · KL", self._acr_chart)
        charts_grid = card_grid([loss_card, lr_card, reward_card, acr_card], cols=2)

        # ---- self-play metrics section (hidden until metrics are present) ----
        self._sp_io = MetricCard("IO accuracy", "—")
        self._sp_buf = MetricCard("Replay buffer", "—")
        self._sp_quality = MetricCard("Data quality kept", "—")
        self._sp_alerts = MetricCard("Monitor alerts", "0")
        sp_grid = card_grid([self._sp_io, self._sp_buf, self._sp_quality,
                             self._sp_alerts], cols=4)
        self._sp_alert_msg = QLabel(""); self._sp_alert_msg.setObjectName("cardBody")
        self._sp_alert_msg.setWordWrap(True)
        self._sp_section = QFrame(); self._sp_section.setObjectName("card")
        sp_lay = QVBoxLayout(self._sp_section); sp_lay.setContentsMargins(0,0,0,10)
        sph = QLabel("SELF-PLAY METRICS"); sph.setObjectName("cardTitle")
        sp_lay.addWidget(sph)
        sp_inner = QWidget(); sp_inner_l = QVBoxLayout(sp_inner)
        sp_inner_l.setContentsMargins(0,0,0,0); sp_inner_l.setSpacing(0)
        sp_inner_l.addWidget(sp_grid); sp_inner_l.addWidget(self._sp_alert_msg)
        sp_lay.addWidget(sp_inner)
        self._sp_section.setVisible(False)

        # ---- extra metrics table ----
        self._extra = QLabel("No extra metrics.")
        self._extra.setObjectName("cardBody")
        self._extra.setWordWrap(True)
        extra_card = QFrame(); extra_card.setObjectName("card")
        e_lay = QVBoxLayout(extra_card); e_lay.setContentsMargins(0,0,0,0)
        eh = QLabel("RUN METADATA"); eh.setObjectName("cardTitle")
        e_lay.addWidget(eh); e_lay.addWidget(self._extra)

        self._host = page_container(sel_host, cards, charts_grid,
                                    self._sp_section, extra_card)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(self._host)

        self._refresh_btn.clicked.connect(self._reload_runs)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

    def _on_stop_clicked(self) -> None:
        run = self._selected_run()
        if run is None:
            QMessageBox.information(self, "Stop run", "No run selected.")
            return
        ans = QMessageBox.question(
            self, "Stop run",
            f"Request cooperative stop for run '{run.name}'?\n\n"
            "This writes a STOP_REQUESTED sentinel next to the run's status "
            "file; the trainer shuts down at its next checkpoint poll.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return
        if self._status.request_stop(run):
            logger.info("stop requested for run %s", run.id)
            self._extra.setText(f"Stop requested for '{run.name}' — sentinel written.")
        else:
            logger.warning("stop sentinel write failed for run %s", run.id)
            QMessageBox.warning(self, "Stop run",
                                "Failed to write the stop sentinel — see logs.")

    def _selected_run(self) -> Optional[RunSnapshot]:
        for r in self._runs:
            if r.id == self._selected_id:
                return r
        return None

    def _chart_card(self, title: str, chart: LiveLineChart) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay = QVBoxLayout(card); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        t = QLabel(title); t.setObjectName("cardTitle")
        lay.addWidget(t)
        lay.addWidget(chart)
        return card

    def _reload_runs(self) -> None:
        self._runs = self._status.snapshot()
        self._combo.blockSignals(True)
        self._combo.clear()
        for r in self._runs:
            self._combo.addItem(f"{r.name}  ({r.status})", r.id)
        self._combo.blockSignals(False)
        if self._runs:
            # keep selection if still present
            idx = 0
            if self._selected_id:
                for i, r in enumerate(self._runs):
                    if r.id == self._selected_id:
                        idx = i; break
            self._combo.setCurrentIndex(idx)
            self._on_select(idx)
        else:
            self._reset_view()

    def _on_select(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._runs):
            return
        r = self._runs[idx]
        self._selected_id = r.id
        self._loss_hist.clear(); self._lr_hist.clear()
        self._loss_chart.clear(); self._lr_chart.clear()
        self._reward_chart.clear(); self._acr_chart.clear()
        self._update_view(r)

    def _reset_view(self) -> None:
        self._c_step.set_value("0 / 0"); self._c_loss.set_value("—")
        self._c_lr.set_value("—"); self._c_vram.set_value("—", unit="GB")
        self._c_eta.set_value("0%"); self._c_age.set_value("—")
        self._extra.setText("No runs discovered.")

    def _update_view(self, r: RunSnapshot) -> None:
        self._c_step.set_value(f"{r.step} / {r.max_steps}")
        self._c_loss.set_value(f"{r.loss:.4f}")
        self._c_loss.push_spark(r.loss)
        self._c_lr.set_value(f"{r.lr:.2e}")
        self._c_lr.push_spark(r.lr)
        self._c_vram.set_value(f"{r.vram_gb:.2f}", unit="GB")
        self._c_eta.set_value(f"{r.progress_pct:.1f}%")
        self._c_age.set_value(f"{r.heartbeat_age_s:.0f}s ago")
        self._loss_chart.push("loss", r.loss)
        self._lr_chart.push("lr", r.lr)
        # self-play metrics from extra
        sp = r.extra
        if "mean_reward" in sp:
            self._reward_chart.push_multi({
                "reward": float(sp.get("mean_reward", 0.0)),
                "intrinsic": float(sp.get("intrinsic_reward", 0.0)),
                "grounded": float(sp.get("grounded_reward", 0.0)),
            })
        if "advantage_collapse_rate" in sp or "diversity_score" in sp:
            self._acr_chart.push_multi({
                "acr": float(sp.get("advantage_collapse_rate", 0.0)),
                "diversity": float(sp.get("diversity_score", 0.0)),
                "kl": float(sp.get("kl_divergence", 0.0)),
            })
        self._update_selfplay_section(sp)
        # metadata
        lines = [f"status_file: {r.status_file}", f"method: {r.method or '—'}"]
        for k, v in list(r.extra.items())[:14]:
            lines.append(f"{k}: {v}")
        self._extra.setText("\n".join(lines))

    def _update_selfplay_section(self, sp: dict) -> None:
        """Render self-play scalar metrics + monitor alerts; hidden when absent."""
        alerts = sp.get("monitor_alerts")
        has_any = any(k in sp for k in ("io_accuracy", "replay_buffer_size",
                                        "data_quality_kept")) or bool(alerts)
        self._sp_section.setVisible(bool(has_any))
        if not has_any:
            return
        if "io_accuracy" in sp:
            try:
                v = float(sp["io_accuracy"])
                self._sp_io.set_value(f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.3f}")
            except (TypeError, ValueError):
                self._sp_io.set_value(str(sp["io_accuracy"]))
        if "replay_buffer_size" in sp:
            self._sp_buf.set_value(str(sp["replay_buffer_size"]))
        if "data_quality_kept" in sp:
            self._sp_quality.set_value(str(sp["data_quality_kept"]))
        if isinstance(alerts, list) and alerts:
            self._sp_alerts.set_value(str(len(alerts)))
            latest = alerts[-1]
            msg = (latest.get("message") or str(latest)) if isinstance(latest, dict) \
                else str(latest)
            self._sp_alert_msg.setText(f"latest alert: {msg}")
        else:
            self._sp_alerts.set_value("0")
            self._sp_alert_msg.setText("")

    def refresh(self) -> None:
        # refresh selected run from latest snapshot
        if not self._selected_id:
            if not self._runs:
                self._reload_runs()
            return
        # cheap refresh: re-snapshot and find
        runs = self._status.snapshot()
        self._runs = runs
        for r in runs:
            if r.id == self._selected_id:
                self._update_view(r)
                return
        # selection vanished
        self._selected_id = None
        self._reload_runs()
