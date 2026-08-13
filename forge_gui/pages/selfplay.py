"""Self-Play Live page — real-time event feed + KPIs + charts for self-play training.

Consumes EventsReader (events.jsonl tail) + StatusReader for run discovery.
Displays per-task progress, throughput, ETA, success rate, and a live event
feed — replacing the old per-epoch-only status view that looked frozen for
4+ minutes between epoch writes.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ..api.events_reader import EventsReader
from ..api.process_manager import ProcessManager
from ..api.status_reader import StatusReader
from ..theme import Palette
from ..widgets.chart import LiveLineChart
from ..widgets.color_log import ColorLogWidget
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label, status_tag

logger = logging.getLogger(__name__)

# Topics available for self-play training (must match DOMAIN_ARCHETYPES keys)
_SELFPLAY_TOPICS = [
    "python_algorithms", "python_math", "python_strings",
    "python_general", "python_oop", "python_file_io",
    "math_arithmetic", "all_topics",
]


class SelfPlayPage(QWidget):
    """Live self-play monitoring with inline start/stop controls."""

    def __init__(self, status_reader: StatusReader,
                 proc_mgr: ProcessManager,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._status = status_reader
        self._proc_mgr = proc_mgr
        self._events = EventsReader()
        self._success_hist: list[float] = []
        self._quality_hist: list[float] = []
        self._gen_ms_hist: list[float] = []
        self._exec_ms_hist: list[float] = []
        self._tpm_hist: list[float] = []
        self._last_event_count = 0
        self._selfplay_task_id: Optional[str] = None  # track our launched process

        # ---- control bar (always visible — start/stop self-play from here) ----
        ctrl = QFrame(); ctrl.setObjectName("card")
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(10)
        ch = QHBoxLayout(); ch.setSpacing(12)
        ch.addWidget(section_label("SELF-PLAY CONTROL"))
        ch.addStretch(1)
        cl.addLayout(ch)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Topic:"))
        self._topic_combo = QComboBox(); self._topic_combo.setMinimumWidth(200)
        for t in _SELFPLAY_TOPICS:
            self._topic_combo.addItem(t)
        row.addWidget(self._topic_combo)
        row.addWidget(QLabel("Epochs:"))
        self._epochs_spin = QSpinBox(); self._epochs_spin.setRange(1, 50); self._epochs_spin.setValue(3)
        row.addWidget(self._epochs_spin)
        row.addWidget(QLabel("Tasks:"))
        self._tasks_spin = QSpinBox(); self._tasks_spin.setRange(5, 500); self._tasks_spin.setValue(50)
        self._tasks_spin.setSingleStep(10)
        row.addWidget(self._tasks_spin)
        row.addStretch(1)
        self._start_btn = QPushButton("Start Self-Play")
        self._start_btn.setObjectName("primary")
        self._start_btn.clicked.connect(self._on_start)
        row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self._stop_btn)
        cl.addLayout(row)
        self._ctrl_status = QLabel("")
        self._ctrl_status.setObjectName("cardBody")
        cl.addWidget(self._ctrl_status)

        # ---- status banner ----
        banner = QFrame(); banner.setObjectName("card")
        bl = QVBoxLayout(banner); bl.setContentsMargins(16, 14, 16, 14); bl.setSpacing(6)
        brow = QHBoxLayout(); brow.setSpacing(12)
        self._phase_tag = status_tag("IDLE", "idle")
        brow.addWidget(self._phase_tag)
        self._phase_detail = QLabel("No self-play run active")
        self._phase_detail.setObjectName("cardBody")
        brow.addWidget(self._phase_detail, 1)
        self._topic_lbl = QLabel("")
        self._topic_lbl.setObjectName("mono")
        brow.addWidget(self._topic_lbl)
        bl.addLayout(brow)
        brow2 = QHBoxLayout(); brow2.setSpacing(12)
        self._current_task = QLabel("—")
        self._current_task.setObjectName("mono")
        brow2.addWidget(QLabel("Current:")); brow2.addWidget(self._current_task, 1)
        self._hb_age = QLabel("heartbeat: —")
        self._hb_age.setObjectName("mono")
        brow2.addWidget(self._hb_age)
        bl.addLayout(brow2)

        # ---- KPI cards ----
        self._c_epoch = MetricCard("Epoch", "0 / 0")
        self._c_tasks = MetricCard("Tasks", "0 / 0")
        self._c_success = MetricCard("Success rate", "—")
        self._c_tpm = MetricCard("Tasks/min", "—")
        self._c_tps = MetricCard("Gen tok/s", "—")
        self._c_eta = MetricCard("ETA", "—")
        self._c_replay = MetricCard("Replay buffer", "—")
        self._c_io = MetricCard("IO accuracy", "—")
        self._kpi_cards = [self._c_epoch, self._c_tasks, self._c_success,
                           self._c_tpm, self._c_tps, self._c_eta,
                           self._c_replay, self._c_io]
        kpi = card_grid([self._c_epoch, self._c_tasks, self._c_success,
                         self._c_tpm, self._c_tps, self._c_eta,
                         self._c_replay, self._c_io], cols=4)

        # ---- progress bar ----
        self._progress = QProgressBar(); self._progress.setRange(0, 100); self._progress.setValue(0)

        # ---- charts ----
        self._success_chart = LiveLineChart(
            series=[("success", Palette.chart_reward), ("quality", Palette.chart_div)],
            window=240, y_label="rate", height=180)
        self._timing_chart = LiveLineChart(
            series=[("gen_ms", Palette.chart_lr), ("exec_ms", Palette.chart_loss)],
            window=240, y_label="ms", height=180)
        self._tpm_chart = LiveLineChart(
            series=[("tpm", Palette.accent2)], window=240, y_label="tasks/min", height=160)
        success_card = self._chart_card("SUCCESS RATE · QUALITY", self._success_chart)
        timing_card = self._chart_card("GEN · EXEC TIMING (ms)", self._timing_chart)
        tpm_card = self._chart_card("THROUGHPUT (tasks/min)", self._tpm_chart)
        charts = card_grid([success_card, timing_card, tpm_card], cols=2)

        # ---- live event feed ----
        feed_card = QFrame(); feed_card.setObjectName("card")
        fl = QVBoxLayout(feed_card); fl.setContentsMargins(0, 0, 0, 0); fl.setSpacing(0)
        fh = QLabel("LIVE EVENT FEED"); fh.setObjectName("cardTitle")
        fl.addWidget(fh)
        self._feed = ColorLogWidget(max_lines=500)
        fl.addWidget(self._feed)

        # ---- monitor alerts ----
        self._alert_lbl = QLabel("No alerts.")
        self._alert_lbl.setObjectName("cardBody")
        self._alert_lbl.setWordWrap(True)
        alert_card = QFrame(); alert_card.setObjectName("card")
        al = QVBoxLayout(alert_card); al.setContentsMargins(0, 0, 0, 0)
        ah = QLabel("MONITOR ALERTS"); ah.setObjectName("cardTitle")
        al.addWidget(ah); al.addWidget(self._alert_lbl)

        # ---- empty state (shown when no run + no events; control bar stays visible) ----
        self._empty = QLabel("Press 'Start Self-Play' above to begin training.\nMetrics, charts, and the event feed will appear here once training starts.")
        self._empty.setObjectName("cardEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_card = QFrame(); empty_card.setObjectName("card")
        el = QVBoxLayout(empty_card); el.setContentsMargins(24, 40, 24, 40)
        el.addWidget(self._empty)
        self._empty_card = empty_card

        # ---- layout ----
        self._host = page_container(ctrl, empty_card, banner, kpi, self._progress,
                                    charts, feed_card, alert_card, spacing=14)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.addWidget(self._host)

        # Hide monitoring sections until we detect a run; control bar always visible
        self._content_widgets = [banner, kpi, self._progress, charts, feed_card, alert_card]
        self._show_empty(True)

    def _chart_card(self, title: str, chart: LiveLineChart) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        lay = QVBoxLayout(card); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        t = QLabel(title); t.setObjectName("cardTitle")
        lay.addWidget(t); lay.addWidget(chart)
        return card

    def _show_empty(self, empty: bool) -> None:
        self._empty_card.setVisible(empty)
        for w in self._content_widgets:
            w.setVisible(not empty)

    def _reset_display(self) -> None:
        """Clear feed, charts, KPIs, and history buffers for a fresh run."""
        self._feed.clear_log()
        self._success_hist.clear()
        self._quality_hist.clear()
        self._gen_ms_hist.clear()
        self._exec_ms_hist.clear()
        self._tpm_hist.clear()
        self._success_chart.clear()
        self._timing_chart.clear()
        # Reset KPI cards
        for card in self._kpi_cards:
            card.set_value("--")
        # Reset progress bar
        self._progress.setValue(0)
        # Reset events reader so it re-discovers and back-fills fresh events
        self._events = EventsReader()
        self._last_event_count = 0

    # ── start/stop controls ───────────────────────────────────────────
    def _on_start(self) -> None:
        """Launch the self-play training process via ProcessManager."""
        topic = self._topic_combo.currentText()
        epochs = str(self._epochs_spin.value())
        n_tasks = str(self._tasks_spin.value())
        from ..api.status_reader import project_root
        from pathlib import Path
        root = project_root()
        venv_py = str(root / "venv" / "Scripts" / "python.exe")
        if not Path(venv_py).is_file():
            venv_py = "python"

        # Kill any existing self-play process before starting a new one
        self._on_stop()

        # Clear old status/event/stop files so the new run starts fresh
        sp_dir = root / "research" / "checkpoints" / "self_play"
        for fname in ("events.jsonl", "status.json", "heartbeat.json",
                      "STOP_REQUESTED", "stop_requested"):
            f = sp_dir / fname
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass

        cmd = [venv_py, "-u", "-m", "research.training.self_play_expert_training",
               "--topics", topic,
               "--epochs", epochs,
               "--n-tasks", n_tasks,
               "--use-curriculum",
               "--batch-size", "8",
               "--verify-workers", "4"]
        task_id = self._proc_mgr.launch("Self-Play Training", cmd)
        self._selfplay_task_id = task_id
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        logger.info("self-play launched as task %s", task_id)

        # Clear old state so the feed/charts start fresh for the new run
        self._reset_display()
        # Show a launching state (model load takes ~15s before events flow)
        self._show_empty(False)
        self._feed.append_line("[Launching] Loading model + warmup... events will appear here once training starts.", level="warn", bold=True)
        self._ctrl_status.setText(f"Launching: topic={topic}, epochs={epochs}, tasks={n_tasks}/epoch (loading model...)")

    def _on_stop(self) -> None:
        """Stop the running self-play process + write stop sentinel."""
        # Kill the process if we launched it from here
        if self._selfplay_task_id and self._selfplay_task_id in self._proc_mgr.tasks:
            info = self._proc_mgr.tasks[self._selfplay_task_id]
            if info.is_live:
                self._proc_mgr.kill(self._selfplay_task_id)
                self._ctrl_status.setText("Stop signal sent — process terminating...")
                logger.info("stop requested for task %s", self._selfplay_task_id)
        # Also kill ANY self-play process in the process manager (may have been
        # launched from the Launch page or a previous GUI session)
        for tid, info in list(self._proc_mgr.tasks.items()):
            if info.is_live and "self" in info.name.lower() and "play" in info.name.lower():
                self._proc_mgr.kill(tid)
                logger.info("killed stray self-play task %s", tid)
        # Write the cooperative stop sentinel as a backup
        try:
            snaps = self._status.snapshot()
            for s in snaps:
                if "self_play" in s.name.lower() and s.status == "running":
                    self._status.request_stop(s)
                    break
        except Exception as e:
            logger.debug("stop sentinel write failed: %s", e)
        self._selfplay_task_id = None
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)

    def _update_ctrl_state(self) -> None:
        """Enable/disable start/stop based on process + status liveness."""
        # Check if our launched process is still alive
        if self._selfplay_task_id:
            info = self._proc_mgr.tasks.get(self._selfplay_task_id)
            if info and not info.is_live:
                # Process finished
                self._selfplay_task_id = None
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)
                if info.status == "done":
                    self._ctrl_status.setText("Training completed.")
                elif info.status == "crashed":
                    self._ctrl_status.setText(f"Training crashed (exit code {info.exit_code}).")
                elif info.status == "killed":
                    self._ctrl_status.setText("Training stopped.")
        # Check if a self-play run is active (from status.json)
        status = self._events.latest_status()
        if status:
            run_status = status.get("status", "")
            # Check heartbeat staleness: if the heartbeat is >60s old, the
            # run is dead even if status.json still says "running" (e.g.
            # the process was killed without writing a final status).
            hb_age = self._events.heartbeat_age()
            is_stale = hb_age is not None and hb_age > 60.0
            # Progress-coupled stall detection: the heartbeat thread may be
            # alive (fresh hb_age) but the training loop itself may be hung.
            # The heartbeat writer sets "stalled": true when no progress is
            # detected within its stall threshold.
            hb_stalled = self._events.heartbeat_stalled()
            is_hung = hb_stalled is True and not is_stale
            if run_status == "running" and not is_stale:
                if not self._selfplay_task_id:
                    # A run is active but we didn't launch it (e.g. from Launch page)
                    self._stop_btn.setEnabled(True)
                    self._start_btn.setEnabled(False)
                if is_hung and not self._selfplay_task_id:
                    self._ctrl_status.setText(
                        "Run appears hung (heartbeat alive but no training progress). "
                        "Click Stop to terminate.")
            elif run_status == "running" and is_stale:
                # Stale run — treat as dead, allow starting a new one
                if not self._selfplay_task_id:
                    self._stop_btn.setEnabled(False)
                    self._start_btn.setEnabled(True)
                    self._ctrl_status.setText("Previous run appears dead (stale heartbeat). Ready for new run.")
            elif run_status == "done":
                if not self._selfplay_task_id:
                    self._stop_btn.setEnabled(False)
                    self._start_btn.setEnabled(True)
            elif run_status in ("error", "crashed"):
                if not self._selfplay_task_id:
                    self._stop_btn.setEnabled(False)
                    self._start_btn.setEnabled(True)
                    self._ctrl_status.setText("Previous run crashed. Ready for new run.")

    def _format_event(self, ev: dict) -> tuple[str, str, bool]:
        """Format an event dict into (text, level, bold) for color-coded display.

        Levels: ok=green, info=white, warn=yellow, error=orange, crash=red,
                phase=blue-bold, invest=purple
        """
        kind = ev.get("kind", "?")
        ts = ev.get("ts", 0)
        tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "--:--:--"

        if kind == "phase":
            return (f"[PHASE] {tstr} → {ev.get('phase','')} {ev.get('detail','')}",
                    "phase", True)

        if kind == "curriculum":
            parse_fail = ev.get("parse_failed", 0)
            level = "warn" if parse_fail > 10 else "info"
            return (f"[CURR]  {tstr} proposed={ev.get('proposed',0)} "
                    f"validated={ev.get('validated',0)} parse_fail={parse_fail}",
                    level, False)

        if kind == "task_start":
            return (f"[START] {tstr} #{ev.get('idx',0)}/{ev.get('total',0)} "
                    f"{ev.get('task','')[:60]}", "info", False)

        if kind == "round":
            success = ev.get("success", False)
            quality = ev.get("quality", 0.0)
            error = ev.get("error", "")
            if success:
                level = "ok"
                status = "OK"
            elif "OOM" in error or "out of memory" in error.lower():
                level = "crash"
                status = "OOM"
            elif "garbled" in error.lower() or quality == 0.0:
                level = "error"
                status = "FAIL"
            else:
                level = "error"
                status = "FAIL"
            return (f"[{status}]  {tstr} #{ev.get('task_idx',0)} R{ev.get('round',0)} "
                    f"q={quality:.2f} {ev.get('gen_ms',0):.0f}ms+{ev.get('exec_ms',0):.0f}ms"
                    + (f" err={error[:40]}" if error else ""),
                    level, False)

        if kind == "task_done":
            success = ev.get("success", False)
            level = "ok" if success else "error"
            status = "DONE-OK" if success else "DONE-FAIL"
            return (f"[{status}] {tstr} #{ev.get('task_idx',0)} "
                    f"rounds={ev.get('rounds_used',0)} q={ev.get('best_quality',0):.2f}",
                    level, False)

        if kind == "epoch_done":
            train_rate = ev.get("train_rate", 0.0)
            val_rate = ev.get("val_rate", 0.0)
            loss = ev.get("loss", 0.0)
            # Purple if NaN/instability, green if good rates, yellow if 0%
            if loss != loss or loss > 100:  # NaN check
                level = "invest"
            elif train_rate > 0.3:
                level = "ok"
            elif train_rate == 0.0:
                level = "warn"
            else:
                level = "info"
            return (f"[EPOCH] {tstr} {ev.get('epoch',0)}/{ev.get('n_epochs',0)} "
                    f"train={train_rate:.1%} val={val_rate:.1%} loss={loss:.4f}",
                    level, True)

        if kind == "alert":
            alert_level = ev.get("level", "")
            if alert_level in ("critical", "crash", "oom"):
                level = "crash"
            elif alert_level in ("error", "fail"):
                level = "error"
            elif alert_level in ("nan", "instability", "investigate"):
                level = "invest"
            elif alert_level in ("warn", "warning"):
                level = "warn"
            else:
                level = "warn"
            return (f"[ALERT] {tstr} [{alert_level}] {ev.get('msg','')[:80]}",
                    level, True)

        if kind == "done":
            return (f"[DONE]  {tstr} {ev.get('reason','')}", "ok", True)

        return (f"[{kind.upper()}] {tstr} {str(ev)[:80]}", "info", False)

    def refresh(self) -> None:
        """Called by the app timer (~1s). Non-blocking, fast."""
        try:
            self._do_refresh()
        except Exception as e:
            logger.warning("SelfPlayPage refresh error: %s", e, exc_info=True)

    def _do_refresh(self) -> None:
        # Update start/stop button states
        self._update_ctrl_state()
        # Poll new events
        new_evts = self._events.poll()
        status = self._events.latest_status()
        hb_age = self._events.heartbeat_age()
        hb_stalled = self._events.heartbeat_stalled()

        # No status file and no events → empty or launching state
        if status is None and not self._events.all_events():
            if self._selfplay_task_id:
                # We just launched — show launching state, not empty
                self._show_empty(False)
            else:
                self._show_empty(True)
            return

        self._show_empty(False)

        # Clear the launching message when first events/status arrive
        if (new_evts or status) and self._feed._line_count <= 1:
            self._feed.clear_log()

        # Append new events to feed (color-coded)
        if new_evts:
            for ev in new_evts:
                text, level, bold = self._format_event(ev)
                self._feed.append_line(text, level=level, bold=bold)

        # Update charts from new task_done + round events
        for ev in new_evts:
            if ev.get("kind") == "task_done":
                success = 1.0 if ev.get("success") else 0.0
                self._success_hist.append(success)
                self._quality_hist.append(ev.get("best_quality", 0.0))
                self._success_chart.push("success", success)
                self._success_chart.push("quality", ev.get("best_quality", 0.0))
            elif ev.get("kind") == "round":
                self._gen_ms_hist.append(ev.get("gen_ms", 0.0))
                self._exec_ms_hist.append(ev.get("exec_ms", 0.0))
                self._timing_chart.push("gen_ms", ev.get("gen_ms", 0.0))
                self._timing_chart.push("exec_ms", ev.get("exec_ms", 0.0))

        # Update from status.json
        if status:
            phase = status.get("phase", "unknown")
            phase_detail = status.get("phase_detail", "")
            topic = status.get("topic", "")
            run_status = status.get("status", "idle")

            # Phase tag color
            if run_status == "done":
                self._phase_tag.setText("DONE")
                self._phase_tag.setObjectName("tagOk")
            elif run_status == "error":
                self._phase_tag.setText("ERROR")
                self._phase_tag.setObjectName("tagErr")
            elif hb_stalled is True and hb_age is not None and hb_age < 60:
                # Heartbeat thread alive but training loop hung
                self._phase_tag.setText(phase.upper() + " (STALLED)")
                self._phase_tag.setObjectName("tagWarn")
            elif hb_age is not None and hb_age < 10:
                self._phase_tag.setText(phase.upper())
                self._phase_tag.setObjectName("tagOk")
            elif hb_age is not None and hb_age < 60:
                self._phase_tag.setText(phase.upper() + " (STALE)")
                self._phase_tag.setObjectName("tagWarn")
            else:
                self._phase_tag.setText(phase.upper())
                self._phase_tag.setObjectName("tagIdle")
            self._phase_tag.style().unpolish(self._phase_tag)
            self._phase_tag.style().polish(self._phase_tag)

            self._phase_detail.setText(phase_detail)
            self._topic_lbl.setText(f"topic: {topic}" if topic else "")
            self._current_task.setText(status.get("current_task", "—")[:80])

            # Heartbeat age
            if hb_age is not None:
                self._hb_age.setText(f"heartbeat: {hb_age:.0f}s ago")
            else:
                self._hb_age.setText("heartbeat: —")

            # KPI cards
            epoch = status.get("step", 0)
            n_epochs = status.get("max_steps", 0)
            self._c_epoch.set_value(f"{epoch} / {n_epochs}")

            tasks_done = status.get("tasks_done", 0)
            tasks_total = status.get("tasks_total", 0)
            self._c_tasks.set_value(f"{tasks_done} / {tasks_total}")

            successes = status.get("epoch_successes", 0)
            if tasks_done > 0:
                self._c_success.set_value(f"{successes / tasks_done:.0%}")
            else:
                self._c_success.set_value("—")

            tpm = status.get("tasks_per_min", 0.0)
            self._c_tpm.set_value(f"{tpm:.1f}" if tpm else "—")
            if tpm:
                self._tpm_chart.push("tpm", tpm)

            tok_s = status.get("gen_tok_s", 0.0)
            self._c_tps.set_value(f"{tok_s:.0f}" if tok_s else "—")

            eta = status.get("eta_s")
            if eta is not None and eta > 0:
                mins, secs = divmod(int(eta), 60)
                self._c_eta.set_value(f"{mins}:{secs:02d}")
            else:
                self._c_eta.set_value("—")

            replay = status.get("replay_buffer_size", 0)
            self._c_replay.set_value(str(replay) if replay else "—")

            io_acc = status.get("io_accuracy", 0.0)
            self._c_io.set_value(f"{io_acc:.0%}" if io_acc else "—")

            # Progress bar
            if tasks_total > 0:
                self._progress.setValue(int(100 * tasks_done / tasks_total))
            elif n_epochs > 0:
                self._progress.setValue(int(100 * epoch / n_epochs))
            else:
                self._progress.setValue(0)

            # Alerts
            alerts = status.get("monitor_alerts", [])
            if alerts:
                lines = []
                for a in alerts[-5:]:
                    level = a.get("level", "info")
                    msg = a.get("msg", "")
                    metric = a.get("metric", "")
                    val = a.get("value", "")
                    lines.append(f"[{level.upper()}] {metric}={val} — {msg}")
                self._alert_lbl.setText("\n".join(lines))
            else:
                self._alert_lbl.setText("No alerts.")
