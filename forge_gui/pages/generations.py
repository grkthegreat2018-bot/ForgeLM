"""Generations page — live token-by-token model generation stream + history.

Supports four generation modes:
  - Standard: streaming token-by-token (generate_stream)
  - Adaptive: adaptive thinking — root token decides think vs direct (generate_adaptive)
  - Batch: multi-prompt batched forward pass (generate_batch)
  - Raw: raw decode with min_p/min_k sampling (generate_raw)
"""
from __future__ import annotations


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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

from ..api.generation import (
    AdaptiveGenWorker,
    BatchGenWorker,
    GenerationWorker,
    RawGenWorker,
)
from ..api.models_index import ModelsIndex
from ..theme import Palette
from ..widgets.metric_card import MetricCard
from ._base import card_grid, page_container, section_label


class GenerationsPage(QWidget):
    def __init__(self, runtime, models_index: ModelsIndex,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self._models = models_index
        self._worker: GenerationWorker | None = None
        self._history: list[tuple[str, str, float]] = []  # (prompt, output, tps)

        # ---- controls ----
        ctrl = QFrame(); ctrl.setObjectName("card")
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(16,14,16,16); cl.setSpacing(10)
        cl.addWidget(section_label("GENERATION CONTROLS"))

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Model"))
        self._resident_lbl = QLabel("no model resident — load one on the Engine page")
        self._resident_lbl.setObjectName("kvVal")
        ckpt_row.addWidget(self._resident_lbl, 1)
        cl.addLayout(ckpt_row)

        # mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self._mode = QComboBox()
        self._mode.addItem("Standard (stream)", "standard")
        self._mode.addItem("Adaptive thinking", "adaptive")
        self._mode.addItem("Batch (multi-prompt)", "batch")
        self._mode.addItem("Raw (min_p/min_k)", "raw")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode)
        mode_row.addStretch(1)
        cl.addLayout(mode_row)

        param_row = QHBoxLayout()
        for label, w in [("Temp", self._spin("temp", 0.7, 0.0, 2.0, 0.05)),
                         ("Top-K", self._spin("topk", 50, 0, 1000, 1, True)),
                         ("Top-P", self._spin("topp", 0.95, 0.0, 1.0, 0.01)),
                         ("Max new", self._spin("max", 128, 1, 4096, 1, True))]:
            param_row.addWidget(QLabel(label)); param_row.addWidget(w)
        param_row.addStretch(1)
        cl.addLayout(param_row)

        # adaptive-specific params (hidden by default)
        self._adaptive_row = QHBoxLayout()
        self._adaptive_row.addWidget(QLabel("Think tok"))
        self._spin("think_max", 512, 16, 8192, 16, True)
        self._adaptive_row.addWidget(self._think_max)
        self._adaptive_row.addWidget(QLabel("No-think tok"))
        self._spin("no_think_max", 256, 16, 4096, 16, True)
        self._adaptive_row.addWidget(self._no_think_max)
        self._adaptive_row.addStretch(1)
        self._adaptive_widget = QWidget()
        self._adaptive_widget.setLayout(self._adaptive_row)
        self._adaptive_widget.setVisible(False)
        cl.addWidget(self._adaptive_widget)

        # raw-specific params (hidden by default)
        self._raw_row = QHBoxLayout()
        self._raw_row.addWidget(QLabel("Min-P"))
        self._spin("minp", 0.0, 0.0, 1.0, 0.01)
        self._raw_row.addWidget(self._minp)
        self._raw_row.addWidget(QLabel("Min-K"))
        self._spin("mink", 0.0, 0.0, 1000.0, 1.0)
        self._raw_row.addWidget(self._mink)
        self._raw_row.addWidget(QLabel("Rep penalty"))
        self._spin("rep_pen", 1.05, 1.0, 2.0, 0.01)
        self._raw_row.addWidget(self._rep_pen)
        self._raw_skip = QCheckBox("Skip special tokens")
        self._raw_row.addWidget(self._raw_skip)
        self._raw_row.addStretch(1)
        self._raw_widget = QWidget()
        self._raw_widget.setLayout(self._raw_row)
        self._raw_widget.setVisible(False)
        cl.addWidget(self._raw_widget)

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
        # show what the shared runtime currently holds
        if self._runtime is not None and self._runtime.is_ready():
            info = self._runtime.info
            import os
            name = os.path.basename(str(info.get("checkpoint", "?")))
            self._resident_lbl.setText(
                f"resident: {name} ({info.get('config_name', '?')})")
        elif self._runtime is not None and self._runtime.state == "loading":
            self._resident_lbl.setText("model loading…")
        else:
            self._resident_lbl.setText(
                "no model resident — load one on the Engine page")

    def _on_mode_changed(self) -> None:
        mode = self._mode.currentData()
        self._adaptive_widget.setVisible(mode == "adaptive")
        self._raw_widget.setVisible(mode == "raw")
        if mode == "batch":
            self._prompt.setPlaceholderText(
                "One prompt per line — each line is a separate prompt for batch generation.")
        else:
            self._prompt.setPlaceholderText("Enter prompt…  e.g.  def fibonacci(n):")

    def _start(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        mode = self._mode.currentData()
        self._output.clear()
        self._status_lbl.setText("Starting…")
        self._gen_btn.setEnabled(False); self._stop_btn.setEnabled(False)

        if mode == "adaptive":
            self._start_adaptive()
        elif mode == "batch":
            self._start_batch()
        elif mode == "raw":
            self._start_raw()
        else:
            self._start_standard()

    def _start_standard(self) -> None:
        prompt = self._prompt.toPlainText().strip() or "def fibonacci(n):"
        self._stop_btn.setEnabled(True)
        self._worker = GenerationWorker(
            runtime=self._runtime, prompt=prompt,
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

    def _start_adaptive(self) -> None:
        prompt = self._prompt.toPlainText().strip() or "def fibonacci(n):"
        self._worker = AdaptiveGenWorker(
            runtime=self._runtime, prompt=prompt,
            think_max_tokens=self._think_max.value(),
            no_think_max_tokens=self._no_think_max.value(),
            temperature=self._temp.value(),
            top_p=self._topp.value(),
            top_k=self._topk.value(),
            parent=self,
        )
        self._worker.status.connect(self._on_status)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_adaptive_done)
        self._worker.start()

    def _start_batch(self) -> None:
        text = self._prompt.toPlainText().strip()
        prompts = [p.strip() for p in text.split("\n") if p.strip()]
        if not prompts:
            prompts = ["def fibonacci(n):"]
        self._worker = BatchGenWorker(
            runtime=self._runtime, prompts=prompts,
            max_new_tokens=self._max.value(),
            temperature=self._temp.value(),
            top_p=self._topp.value(),
            top_k=self._topk.value(),
            parent=self,
        )
        self._worker.status.connect(self._on_status)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_batch_done)
        self._worker.start()

    def _start_raw(self) -> None:
        prompt = self._prompt.toPlainText().strip() or "def fibonacci(n):"
        self._worker = RawGenWorker(
            runtime=self._runtime, prompt=prompt,
            max_new_tokens=self._max.value(),
            temperature=self._temp.value(),
            top_p=self._topp.value(),
            top_k=self._topk.value(),
            repetition_penalty=self._rep_pen.value(),
            min_p=self._minp.value(),
            min_k=self._mink.value(),
            skip_special_tokens=self._raw_skip.isChecked(),
            parent=self,
        )
        self._worker.status.connect(self._on_status)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_adaptive_done(self, text: str, did_think: bool, tps: float) -> None:
        mode_label = "thinking" if did_think else "direct"
        self._status_lbl.setText(f"done · {tps:.1f} tok/s · {mode_label}")
        self._gen_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._output.setPlainText(text)
        self._c_tps.set_value(f"{tps:.1f}"); self._c_tps.push_spark(tps)
        ntok = len(text.split())
        self._c_tokens.set_value(str(ntok)); self._c_tokens.push_spark(ntok)
        prompt = self._prompt.toPlainText().strip() or "(default)"
        self._history.insert(0, (f"[{mode_label}] {prompt[:55]}",
                                 text[:200].replace("\n", " ⏎ "), tps))
        self._history = self._history[:12]
        self._c_runs.set_value(str(len(self._history)))
        lines = [f"▸ {p}  ({t:.1f} tok/s)\n  {out}\n" for p, out, t in self._history]
        self._hist.setText("\n".join(lines))

    def _on_batch_done(self, results: list, tps: float) -> None:
        self._status_lbl.setText(f"done · {tps:.1f} tok/s · {len(results)} prompts")
        self._gen_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        output_text = "\n---\n".join(
            f"[{i}] {r[:300]}" for i, r in enumerate(results))
        self._output.setPlainText(output_text)
        self._c_tps.set_value(f"{tps:.1f}"); self._c_tps.push_spark(tps)
        total_words = sum(len(r.split()) for r in results)
        self._c_tokens.set_value(str(total_words)); self._c_tokens.push_spark(total_words)
        prompt = f"batch({len(results)})"
        self._history.insert(0, (prompt, output_text[:200].replace("\n", " ⏎ "), tps))
        self._history = self._history[:12]
        self._c_runs.set_value(str(len(self._history)))
        lines = [f"▸ {p}  ({t:.1f} tok/s)\n  {out}\n" for p, out, t in self._history]
        self._hist.setText("\n".join(lines))

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
        # keep the resident-model label current (event-driven page otherwise)
        self._reload()
