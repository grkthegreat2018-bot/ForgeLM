"""Generations page — live token-by-token model generation stream + history."""
from __future__ import annotations


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..api.generation import GenerationWorker, list_checkpoints
from ..api.models_index import ModelsIndex
from ..theme import Palette
from ..widgets.metric_card import MetricCard
from ..widgets.search_combo import SearchableComboBox
from ._base import card_grid, page_container, section_label


class GenerationsPage(QWidget):
    def __init__(self, models_index: ModelsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._models = models_index
        self._worker: GenerationWorker | None = None
        self._history: list[tuple[str, str, float]] = []  # (prompt, output, tps)

        # ---- controls ----
        ctrl = QFrame(); ctrl.setObjectName("card")
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(16,14,16,16); cl.setSpacing(10)
        cl.addWidget(section_label("GENERATION CONTROLS"))

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Checkpoint"))
        self._ckpt_combo = SearchableComboBox(); self._ckpt_combo.setMinimumWidth(280)
        ckpt_row.addWidget(self._ckpt_combo, 1)
        ckpt_row.addWidget(QLabel("Config"))
        self._cfg_combo = QComboBox(); self._cfg_combo.setMinimumWidth(180)
        ckpt_row.addWidget(self._cfg_combo, 1)
        cl.addLayout(ckpt_row)

        param_row = QHBoxLayout()
        for label, w in [("Temp", self._spin("temp", 0.7, 0.0, 2.0, 0.05)),
                         ("Top-K", self._spin("topk", 50, 0, 1000, 1, True)),
                         ("Top-P", self._spin("topp", 0.95, 0.0, 1.0, 0.01)),
                         ("Max new", self._spin("max", 128, 1, 4096, 1, True))]:
            param_row.addWidget(QLabel(label)); param_row.addWidget(w)
        param_row.addStretch(1)
        cl.addLayout(param_row)

        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Enter prompt…  e.g.  def fibonacci(n):")
        self._prompt.setFixedHeight(70)
        cl.addWidget(self._prompt)

        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Generate"); self._gen_btn.setObjectName("primary")
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._gen_btn); btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        self._status_lbl = QLabel("Idle"); self._status_lbl.setObjectName("mono")
        self._status_lbl.setStyleSheet("color:#8b96a8; font-size:12px;")
        btn_row.addWidget(self._status_lbl)
        cl.addLayout(btn_row)

        # ---- output + stats ----
        self._output = QPlainTextEdit(); self._output.setObjectName("logView")
        self._output.setReadOnly(True)
        self._c_tps = MetricCard("Tokens / s", "—", spark_color=Palette.chart_reward)
        self._c_tokens = MetricCard("Tokens", "0", spark_color=Palette.accent)
        self._c_runs = MetricCard("Generations", "0")
        stats = card_grid([self._c_tps, self._c_tokens, self._c_runs], cols=3)

        # ---- history ----
        hist_card = QFrame(); hist_card.setObjectName("card")
        hl = QVBoxLayout(hist_card); hl.setContentsMargins(16,14,16,16); hl.setSpacing(8)
        hl.addWidget(section_label("GENERATION HISTORY"))
        self._hist = QLabel("No generations yet.")
        self._hist.setObjectName("mono"); self._hist.setWordWrap(True)
        self._hist.setStyleSheet("color:#8b96a8; font-size:12px;")
        hl.addWidget(self._hist)

        self._host = page_container(ctrl, stats, self._output, hist_card)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(self._host)

        self._gen_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        self._reload()

    def _spin(self, name, default, lo, hi, step, integer=False):
        sb = QSpinBox() if integer else QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(default)
        sb.setFixedWidth(90)
        setattr(self, f"_{name}", sb)
        return sb

    def _reload(self) -> None:
        self._ckpt_combo.clear()
        for rel, name in list_checkpoints():
            self._ckpt_combo.addItem(name, rel)
        self._cfg_combo.clear()
        for c in self._models.configs():
            self._cfg_combo.addItem(c.name, c.name)

    def _start(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        prompt = self._prompt.toPlainText().strip() or "def fibonacci(n):"
        ckpt = self._ckpt_combo.currentSearchData() or ""
        cfg = self._cfg_combo.currentData() or ""
        self._output.clear()
        self._status_lbl.setText("Starting…")
        self._gen_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._worker = GenerationWorker(
            prompt=prompt, checkpoint=ckpt, config_name=cfg,
            max_new_tokens=self._max.value(),
            temperature=self._temp.value(),
            top_k=self._topk.value(),
            top_p=self._topp.value(),
            parent=self,
        )
        self._worker.token.connect(self._on_token)
        self._worker.status.connect(self._on_status)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._status_lbl.setText("Stopping…")

    def _on_token(self, tok: str) -> None:
        self._output.moveCursor(Qt.MoveCursor.End)
        self._output.insertPlainText(tok)

    def _on_status(self, s: str) -> None:
        self._status_lbl.setText(s)

    def _on_error(self, e: str) -> None:
        self._status_lbl.setText(f"error: {e}")
        self._gen_btn.setEnabled(True); self._stop_btn.setEnabled(False)

    def _on_done(self, full: str, tps: float) -> None:
        self._status_lbl.setText(f"done · {tps:.1f} tok/s")
        self._gen_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._c_tps.set_value(f"{tps:.1f}"); self._c_tps.push_spark(tps)
        ntok = len(full.split())
        self._c_tokens.set_value(str(ntok)); self._c_tokens.push_spark(ntok)
        prompt = self._prompt.toPlainText().strip() or "(default)"
        self._history.insert(0, (prompt[:60], full[:200].replace("\n", " ⏎ "), tps))
        self._history = self._history[:12]
        self._c_runs.set_value(str(len(self._history)))
        lines = [f"▸ {p}  ({t:.1f} tok/s)\n  {out}\n" for p, out, t in self._history]
        self._hist.setText("\n".join(lines))

    def refresh(self) -> None:
        # nothing to poll; generation is event-driven
        pass
