"""LoRA Manager — adapter library + hot-swap + merge studio.

Left: every ``*lora*.safetensors`` adapter under research/checkpoints/
(with rank / params / dtype parsed straight from the safetensors header,
no torch needed). Selecting one auto-fills rank.

Right top: hot-swap on the resident engine — load an adapter (rank, alpha,
target modules), unload, live info.

Right bottom: merge an adapter into a base checkpoint entirely on CPU
(never touches the 12GB VRAM budget) → standalone merged model.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHeaderView, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..api.engine_runtime import EngineRuntime
from ..api.lora_store import LoraManager, scan_lora_adapters
from ..api.models_index import ModelsIndex
from ..api.status_reader import project_root
from ..widgets.search_combo import SearchableComboBox
from ._base import section_label

logger = logging.getLogger(__name__)

TARGET_LABELS = {
    "default": "Default (FFN + attention)",
    "ffn": "FFN only (w_gate / w_up / w_down)",
    "attention": "Attention only (q / k / v / out / in)",
}


def _kv_row(key: str) -> tuple[QFrame, QLabel]:
    row = QFrame(); row.setObjectName("kvRow")
    h = QHBoxLayout(row); h.setContentsMargins(12, 8, 12, 8); h.setSpacing(12)
    k = QLabel(key); k.setObjectName("kvKey"); k.setMinimumWidth(110)
    v = QLabel("—"); v.setObjectName("kvVal"); v.setWordWrap(True)
    v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    h.addWidget(k, 0); h.addWidget(v, 1)
    return row, v


class LoraPage(QWidget):
    request_open = Signal(int)   # ask app to switch page (to Fine-Tune)

    def __init__(self, runtime: EngineRuntime, lora_mgr: LoraManager,
                 models_index: ModelsIndex,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.mgr = lora_mgr
        self.models_index = models_index
        self._entries: list = []

        outer = QHBoxLayout(self); outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(16)

        # ── left: adapter library ────────────────────────────────────
        left = QFrame(); left.setObjectName("card")
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(0)
        head = QHBoxLayout(); head.setContentsMargins(16, 14, 8, 8)
        head.addWidget(section_label("Adapter library"))
        head.addStretch(1)
        self._refresh_btn = QPushButton("Rescan")
        head.addWidget(self._refresh_btn)
        self._train_btn = QPushButton("Train new LoRA →")
        head.addWidget(self._train_btn)
        ll.addLayout(head)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Rank", "Params", "Size", "Modified"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 5):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setObjectName("dataTable")
        self._table.itemSelectionChanged.connect(self._on_select)
        ll.addWidget(self._table, 1)
        self._hint = QLabel("Adapters are *lora*.safetensors files — saved by "
                            "self-play epochs, discovery runs, or Fine-Tune "
                            "with 'save adapter'.")
        self._hint.setObjectName("cardBody"); self._hint.setWordWrap(True)
        self._hint.setContentsMargins(16, 10, 16, 14)
        ll.addWidget(self._hint)
        outer.addWidget(left, 5)

        # ── right column ─────────────────────────────────────────────
        right_col = QVBoxLayout(); right_col.setSpacing(16)

        # hot-swap card
        swap_card = QFrame(); swap_card.setObjectName("card")
        sl = QVBoxLayout(swap_card); sl.setContentsMargins(16, 14, 16, 16)
        sl.setSpacing(10)
        sl.addWidget(section_label("Hot-swap on resident engine"))
        self._swap_info: dict[str, QLabel] = {}
        for key in ("Engine", "Adapter", "Rank / alpha", "Params"):
            row, val = _kv_row(key)
            self._swap_info[key] = val
            sl.addWidget(row)
        grid = QWidget(); gl = QHBoxLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0); gl.setSpacing(10)
        gl.addWidget(QLabel("Rank"))
        self._rank = QSpinBox(); self._rank.setRange(1, 256); self._rank.setValue(32)
        gl.addWidget(self._rank)
        gl.addWidget(QLabel("Alpha"))
        self._alpha = QSpinBox(); self._alpha.setRange(0, 512); self._alpha.setValue(64)
        self._alpha.setToolTip("0 = auto (rank × 2)")
        gl.addWidget(self._alpha)
        gl.addWidget(QLabel("Targets"))
        self._targets = QComboBox()
        for key, label in TARGET_LABELS.items():
            self._targets.addItem(label, key)
        gl.addWidget(self._targets, 1)
        sl.addWidget(grid)
        btns = QHBoxLayout(); btns.setSpacing(10)
        self._load_btn = QPushButton("⏏ Load adapter")
        self._load_btn.setObjectName("primary")
        self._unload_btn = QPushButton("Unload")
        self._unload_btn.setObjectName("danger")
        btns.addWidget(self._load_btn); btns.addWidget(self._unload_btn)
        btns.addStretch(1)
        sl.addLayout(btns)
        self._swap_status = QLabel("Select an adapter on the left, then Load "
                                   "(requires a resident engine).")
        self._swap_status.setObjectName("chatMeta"); self._swap_status.setWordWrap(True)
        sl.addWidget(self._swap_status)
        right_col.addWidget(swap_card, 1)

        # merge card
        merge_card = QFrame(); merge_card.setObjectName("card")
        ml = QVBoxLayout(merge_card); ml.setContentsMargins(16, 14, 16, 16)
        ml.setSpacing(10)
        ml.addWidget(section_label("Merge adapter → base checkpoint (CPU)"))
        br = QHBoxLayout(); br.setSpacing(10)
        br.addWidget(QLabel("Base"))
        self._base = SearchableComboBox()
        br.addWidget(self._base, 1)
        br.addWidget(QLabel("Config"))
        self._config = QComboBox(); self._config.setEditable(True)
        br.addWidget(self._config, 1)
        ml.addLayout(br)
        orow = QHBoxLayout(); orow.setSpacing(10)
        orow.addWidget(QLabel("Output"))
        self._out = QLineEdit("")
        orow.addWidget(self._out, 1)
        self._pick_out_btn = QPushButton("…")
        self._pick_out_btn.setFixedWidth(32)
        orow.addWidget(self._pick_out_btn)
        ml.addLayout(orow)
        self._merge_btn = QPushButton("⚒ Merge (runs on CPU — GPU untouched)")
        self._merge_btn.setObjectName("primary")
        ml.addWidget(self._merge_btn)
        self._merge_status = QLabel("Merging folds adapter weights into the base "
                                    "model and writes a standalone checkpoint.")
        self._merge_status.setObjectName("chatMeta"); self._merge_status.setWordWrap(True)
        ml.addWidget(self._merge_status)
        right_col.addWidget(merge_card, 1)
        outer.addLayout(right_col, 4)

        # signals
        self._refresh_btn.clicked.connect(self.refresh)
        self._train_btn.clicked.connect(lambda: self.request_open.emit(-1))
        self._load_btn.clicked.connect(self._load_on_engine)
        self._unload_btn.clicked.connect(self._unload)
        self._merge_btn.clicked.connect(self._merge)
        self._pick_out_btn.clicked.connect(self._pick_out)
        self.mgr.busy_changed.connect(self._on_busy)
        self.mgr.status.connect(self._on_mgr_status)
        self.mgr.lora_loaded.connect(self._on_loaded)
        self.mgr.lora_unloaded.connect(self._on_unloaded)
        self.mgr.merge_done.connect(self._on_merged)
        self.mgr.failed.connect(self._on_failed)
        self.runtime.state_changed.connect(self._on_engine_state)

        # Defer refresh to after window show — scan_lora_adapters() + model
        # index scan are heavy and block GUI startup if run in __init__.
        QTimer.singleShot(0, self.refresh)

    # ── library ───────────────────────────────────────────────────────
    def refresh(self) -> None:
        self._entries = scan_lora_adapters()
        t = self._table
        t.setRowCount(len(self._entries))
        import datetime
        for i, e in enumerate(self._entries):
            items = [
                e.name,
                str(e.rank) if e.rank else "?",
                f"{e.n_params/1e6:.1f}M" if e.n_params else "—",
                e.size_label,
                datetime.datetime.fromtimestamp(e.modified).strftime("%Y-%m-%d %H:%M")
                if e.modified else "—",
            ]
            for j, txt in enumerate(items):
                it = QTableWidgetItem(txt)
                if j:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignVCenter
                                        | Qt.AlignmentFlag.AlignRight)
                if j == 0:
                    it.setToolTip(f"{e.path}\nbase hint: {e.base_hint or '?'}"
                                  + (f"\ndtype: {e.dtype}" if e.dtype else ""))
                t.setItem(i, j, it)
        if not self._entries:
            t.setRowCount(1)
            placeholder = QTableWidgetItem("No LoRA adapters found — train one "
                                           "in Fine-Tune with 'save adapter'")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            t.setItem(0, 0, placeholder)
        # populate base/config combos for merge
        self._base.clear()
        configs = ["forgelm_v2_light"]
        if self.models_index is not None:
            try:
                configs = [c.name for c in self.models_index.configs()] or configs
                for m in self.models_index.models():
                    if "lora" not in m.name.lower() and m.is_safetensors:
                        self._base.addItem(m.name, m.path)
            except Exception as e:
                logger.warning("lora page model list failed: %s", e)
        self._config.clear(); self._config.addItems(configs)
        self._on_engine_state(self.runtime.state)

    def _selected_entry(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if 0 <= idx < len(self._entries):
            return self._entries[idx]
        return None

    def _on_select(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        if e.rank:
            self._rank.setValue(e.rank)
            if not self._alpha.value():
                self._alpha.setValue(e.rank * 2)
        self._swap_status.setText(
            f"Selected {e.name} (rank {e.rank or '?'}) — ready to load.")
        if not self._out.text().strip():
            stem = (e.base_hint or "merged") + ".merged"
            self._out.setText(f"research/checkpoints/{stem}.safetensors")

    # ── hot-swap ──────────────────────────────────────────────────────
    def _load_on_engine(self) -> None:
        e = self._selected_entry()
        if e is None:
            self._swap_status.setText("select an adapter in the library first")
            return
        if not self.runtime.is_ready():
            self._swap_status.setText(
                "no resident engine — load a base model on the Engine page first")
            return
        alpha = self._alpha.value() or None
        self.mgr.load_on_engine(self.runtime, e.path, self._rank.value(),
                                alpha, self._targets.currentData() or "default")

    def _unload(self) -> None:
        if not self.runtime.is_ready():
            self._swap_status.setText("no resident engine")
            return
        self.mgr.unload_from_engine(self.runtime)

    def _on_engine_state(self, state: str) -> None:
        if state == "ready":
            info = self.runtime.info
            self._swap_info["Engine"].setText(
                f"{info.get('config_name', '?')} · ready")
            self.mgr.refresh_info(self.runtime)
        else:
            self._swap_info["Engine"].setText(f"engine {state}")
            for k in ("Adapter", "Rank / alpha", "Params"):
                self._swap_info[k].setText("—")

    def _on_loaded(self, info: dict) -> None:
        path = str(info.get("path", "?"))
        self._swap_info["Adapter"].setText(path.replace("\\", "/").rsplit("/", 1)[-1])
        self._swap_info["Rank / alpha"].setText(
            f"r={info.get('rank', '?')} · α={info.get('alpha', '?')}")
        n = info.get("n_params", 0)
        self._swap_info["Params"].setText(
            f"{n/1e6:.1f}M across {info.get('n_adapters', '?')} adapters")
        self._swap_status.setText("adapter hot-loaded ✓ — chat/agent now use it")

    def _on_unloaded(self) -> None:
        for k in ("Adapter", "Rank / alpha", "Params"):
            self._swap_info[k].setText("—")
        self._swap_status.setText("adapter unloaded — base model behavior restored")

    # ── merge ─────────────────────────────────────────────────────────
    def _pick_out(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = str(project_root() / "research" / "checkpoints")
        path, _ = QFileDialog.getSaveFileName(
            self, "Merged checkpoint path", start,
            "Safetensors (*.safetensors)")
        if path:
            self._out.setText(path)

    def _merge(self) -> None:
        e = self._selected_entry()
        if e is None:
            self._merge_status.setText("select an adapter in the library first")
            return
        base = self._base.currentSearchData() or self._base.currentText()
        if not base:
            self._merge_status.setText("pick a base checkpoint")
            return
        out = self._out.text().strip()
        if not out:
            self._merge_status.setText("set an output path")
            return
        if not out.endswith(".safetensors"):
            out += ".safetensors"
        rank = e.rank or self._rank.value()
        alpha = self._alpha.value() or None
        cfg = self._config.currentText().strip() or "forgelm_v2_light"
        adapter = e.path
        if not _isabs(adapter):
            adapter = str(project_root() / adapter)
        self._merge_status.setText(f"merging {e.name} into {base} on CPU…")
        self.mgr.merge(base, cfg, adapter, rank, alpha, out)

    def _on_merged(self, out_path: str) -> None:
        self._merge_status.setText(
            f"merged ✓ → {out_path}\n(it now appears on the Models page after a "
            f"rescan)")
        self.refresh()

    # ── manager events ────────────────────────────────────────────────
    def _on_busy(self, busy: bool) -> None:
        for b in (self._load_btn, self._unload_btn, self._merge_btn):
            b.setEnabled(not busy)

    def _on_mgr_status(self, msg: str) -> None:
        # route to whichever section started it (merge overrides swap text)
        if "merg" in msg.lower() or "saving" in msg.lower() or "base" in msg.lower():
            self._merge_status.setText(msg)
        else:
            self._swap_status.setText(msg)

    def _on_failed(self, err: str) -> None:
        self._swap_status.setText(f"error: {err}")
        self._merge_status.setText(f"error: {err}")


def _isabs(p: str) -> bool:
    import os
    return os.path.isabs(p)
