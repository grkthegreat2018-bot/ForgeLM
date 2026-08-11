"""Models page — checkpoint browser + registered config registry."""
from __future__ import annotations


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..api.model_boot import ModelBootWorker
from ..api.models_index import ModelsIndex, _human_bytes
from ..widgets.metric_card import MetricCard
from ..widgets.search_combo import SearchableComboBox
from ._base import card_grid, page_container, section_label


class ModelsPage(QWidget):
    def __init__(self, models_index: ModelsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._models = models_index
        self._boot_worker: ModelBootWorker | None = None
        self._all_models: list = []

        # ---- summary cards ----
        self._c_count = MetricCard("Checkpoints", "0")
        self._c_total = MetricCard("Total size", "—")
        self._c_safet = MetricCard("Safetensors", "0")
        self._c_configs = MetricCard("Registered configs", "0")
        cards = card_grid([self._c_count, self._c_total, self._c_safet, self._c_configs],
                          cols=4)

        # ---- checkpoints table ----
        ck_card = QFrame(); ck_card.setObjectName("card")
        cl = QVBoxLayout(ck_card); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        head = QHBoxLayout(); head.setContentsMargins(16,14,8,8)
        head.addWidget(section_label("CHECKPOINTS"))
        head.addStretch(1)
        self._search = QLineEdit(); self._search.setPlaceholderText("Filter checkpoints…")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedWidth(220)
        self._search.textChanged.connect(self._filter_table)
        head.addWidget(self._search)
        self._refresh_btn = QPushButton("Refresh")
        head.addWidget(self._refresh_btn)
        cl.addLayout(head)
        self._ck_table = QTableWidget(0, 5)
        self._ck_table.setHorizontalHeaderLabels(["Name", "Path", "Size", "Config", "Modified"])
        hh = self._ck_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._ck_table.verticalHeader().setVisible(False)
        self._ck_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._ck_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._ck_table.setAlternatingRowColors(True)
        self._ck_table.setObjectName("dataTable")
        cl.addWidget(self._ck_table)

        # ---- configs table ----
        cfg_card = QFrame(); cfg_card.setObjectName("card")
        gl = QVBoxLayout(cfg_card); gl.setContentsMargins(0,0,0,0); gl.setSpacing(0)
        ch = QLabel("REGISTERED MODEL CONFIGS"); ch.setObjectName("cardTitle")
        ch.setContentsMargins(16,14,16,8)
        gl.addWidget(ch)
        self._cfg_table = QTableWidget(0, 8)
        self._cfg_table.setHorizontalHeaderLabels(
            ["Name", "d_model", "Layers", "Heads", "KV heads", "Vocab",
             "Attn / FFN", "~Params"])
        for i in range(8):
            self._cfg_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.Stretch if i == 0 else QHeaderView.ResizeMode.ResizeToContents)
        self._cfg_table.verticalHeader().setVisible(False)
        self._cfg_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._cfg_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._cfg_table.setAlternatingRowColors(True)
        self._cfg_table.setObjectName("dataTable")
        gl.addWidget(self._cfg_table)

        # ---- boot & test panel ----
        boot_card = QFrame(); boot_card.setObjectName("card")
        bl = QVBoxLayout(boot_card); bl.setContentsMargins(16, 14, 16, 16); bl.setSpacing(10)
        bl.addWidget(section_label("BOOT & TEST MODEL"))
        br = QHBoxLayout(); br.setSpacing(10)
        br.addWidget(QLabel("Checkpoint"))
        self._boot_ckpt = SearchableComboBox(); self._boot_ckpt.setMinimumWidth(280)
        br.addWidget(self._boot_ckpt, 2)
        br.addWidget(QLabel("Config"))
        self._boot_cfg = QComboBox(); self._boot_cfg.setMinimumWidth(180)
        br.addWidget(self._boot_cfg, 1)
        bl.addLayout(br)
        pr = QHBoxLayout(); pr.setSpacing(10)
        pr.addWidget(QLabel("Test prompt"))
        self._boot_prompt = QLineEdit("def fibonacci(n):")
        pr.addWidget(self._boot_prompt, 1)
        pr.addWidget(QLabel("Max tokens"))
        self._boot_max = QLineEdit("64"); self._boot_max.setFixedWidth(60)
        pr.addWidget(self._boot_max)
        bl.addLayout(pr)
        bbtn = QHBoxLayout()
        self._boot_btn = QPushButton("Boot & Test"); self._boot_btn.setObjectName("primary")
        self._boot_stop_btn = QPushButton("Stop"); self._boot_stop_btn.setObjectName("danger")
        self._boot_stop_btn.setEnabled(False)
        bbtn.addWidget(self._boot_btn); bbtn.addWidget(self._boot_stop_btn)
        bbtn.addStretch(1)
        self._boot_status = QLabel("Idle")
        self._boot_status.setStyleSheet("color:#8b96a8; font-size:12px;")
        bbtn.addWidget(self._boot_status)
        bl.addLayout(bbtn)
        self._boot_output = QPlainTextEdit(); self._boot_output.setObjectName("logView")
        self._boot_output.setReadOnly(True); self._boot_output.setFixedHeight(160)
        bl.addWidget(self._boot_output)

        self._host = page_container(cards, ck_card, cfg_card, boot_card)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(self._host)
        self._refresh_btn.clicked.connect(self.refresh)
        self._boot_btn.clicked.connect(self._boot_test)
        self._boot_stop_btn.clicked.connect(self._boot_stop)

    def refresh(self) -> None:
        models = self._models.models()
        self._all_models = models
        configs = self._models.configs()
        total = sum(m.size_bytes for m in models)
        self._c_count.set_value(str(len(models)))
        self._c_total.set_value(_human_bytes(total))
        self._c_safet.set_value(str(sum(1 for m in models if m.is_safetensors)))
        self._c_configs.set_value(str(len(configs)))
        # checkpoints table
        self._populate_table(models)
        # configs table
        self._cfg_table.setRowCount(len(configs))
        for i, c in enumerate(configs):
            row = [c.name, str(c.d_model), str(c.n_layers), str(c.n_heads),
                   str(c.n_kv_heads or "—"), str(c.vocab_size),
                   f"{c.attn_type} / {c.ffn_type}", c.params_label]
            for j, txt in enumerate(row):
                it = QTableWidgetItem(txt)
                if j != 0 and j != 6:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
                self._cfg_table.setItem(i, j, it)
        # boot combos
        self._boot_ckpt.clear()
        for m in models:
            self._boot_ckpt.addItem(m.name, m.path)
        self._boot_cfg.clear()
        for c in configs:
            self._boot_cfg.addItem(c.name, c.name)

    def _populate_table(self, models: list) -> None:
        self._ck_table.setRowCount(len(models))
        for i, m in enumerate(models):
            items = [m.name, m.path, m.size_label, m.config_name or "—",
                     _mtime(m.modified)]
            for j, txt in enumerate(items):
                it = QTableWidgetItem(txt)
                if j in (2, 3, 4):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
                self._ck_table.setItem(i, j, it)

    def _filter_table(self, text: str) -> None:
        q = text.strip().lower()
        if not q:
            self._populate_table(self._all_models)
            return
        filtered = [m for m in self._all_models
                    if q in m.name.lower() or q in m.path.lower()
                    or q in (m.config_name or "").lower()]
        self._populate_table(filtered)

    def _boot_test(self) -> None:
        if self._boot_worker and self._boot_worker.isRunning():
            return
        ckpt = self._boot_ckpt.currentSearchData() or ""
        cfg = self._boot_cfg.currentData() or ""
        if not ckpt or not cfg:
            self._boot_status.setText("Select a checkpoint and config first.")
            return
        prompt = self._boot_prompt.text().strip() or "def fibonacci(n):"
        try:
            max_tok = int(self._boot_max.text().strip() or "64")
        except ValueError:
            max_tok = 64
        self._boot_output.clear()
        self._boot_status.setText("Starting…")
        self._boot_btn.setEnabled(False); self._boot_stop_btn.setEnabled(True)
        self._boot_worker = ModelBootWorker(
            checkpoint=ckpt, config_name=cfg, test_prompt=prompt,
            max_new_tokens=max_tok, parent=self,
        )
        self._boot_worker.status.connect(self._boot_on_status)
        self._boot_worker.output.connect(self._boot_on_output)
        self._boot_worker.result.connect(self._boot_on_result)
        self._boot_worker.error.connect(self._boot_on_error)
        self._boot_worker.start()

    def _boot_stop(self) -> None:
        if self._boot_worker:
            self._boot_worker.cancel()
            self._boot_status.setText("Stopping…")

    def _boot_on_status(self, s: str) -> None:
        self._boot_status.setText(s)

    def _boot_on_output(self, text: str) -> None:
        self._boot_output.appendPlainText(text)

    def _boot_on_result(self, d: dict) -> None:
        self._boot_status.setText(
            f"Done · {d['tps']:.1f} tok/s · {d['tokens']} tokens · {d['time_s']:.2f}s"
        )
        self._boot_btn.setEnabled(True); self._boot_stop_btn.setEnabled(False)

    def _boot_on_error(self, e: str) -> None:
        self._boot_status.setText(f"Error: {e}")
        self._boot_output.appendPlainText(f"[ERROR] {e}")
        self._boot_btn.setEnabled(True); self._boot_stop_btn.setEnabled(False)


def _mtime(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
