"""Fine-Tune Studio — dataset builder + full sft_train launcher.

Left: dataset sources — rated chat exports (data/sft/*.jsonl) and any other
JSONL under data/. Multi-select with example counts.

Right: the full trainer surface, mirroring every important flag of
research/training/runners/sft_train.py — base model, LoRA vs full FT
(with adapter-only save for the LoRA Manager), schedule, optimizer, loss
function, curriculum, EMA/validation — plus a live command preview so
what you see is exactly what launches. Runs land on the Tasks page with
live logs via ProcessManager.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from ..api.chat_store import ChatStore
from ..api.models_index import ModelsIndex
from ..api.process_manager import ProcessManager
from ..api.status_reader import project_root
from ._base import section_label

logger = logging.getLogger(__name__)

OPTIMIZERS = ["muon_sf", "muon_sf_plain", "muon", "fused", "lion",
              "flash_adamw", "flash_lion", "sf_normuon", "amuse", "mona",
              "bnb", "forge", "cpu_offload", "badam", "fira_nlrq"]
LOSSES = ["ce", "focal", "label_smoothing", "lovasz", "dynamic_focal",
          "mixture"]
CURRICULA = ["none", "vanilla", "pacing", "interleaved", "warmup"]


def _row(label: str, widget: QWidget, stretch_label: bool = False):
    row = QHBoxLayout(); row.setSpacing(10)
    l = QLabel(label); l.setMinimumWidth(120)
    row.addWidget(l, 1 if stretch_label else 0)
    row.addWidget(widget, 0 if stretch_label else 1)
    return row


def _spin(lo: float, hi: float, val: float, dec: int = 0,
          step: float = 1.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setDecimals(dec)
    s.setSingleStep(step)
    return s


class FineTunePage(QWidget):
    def __init__(self, store: ChatStore, proc_mgr: ProcessManager,
                 models_index: ModelsIndex | None = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store
        self.proc_mgr = proc_mgr
        self.models_index = models_index

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_datasets_pane())
        splitter.addWidget(self._build_config_pane())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter)
        self._reload_models()
        self.refresh()

    def _reload_models(self) -> None:
        """Populate base-model combos once (NOT in refresh() — refresh runs
        every 500ms on the visible page and must not reset selections)."""
        self._config.clear()
        names = ["forgelm_v2_light"]
        if self.models_index is not None:
            try:
                names = [c.name for c in self.models_index.configs()] or names
            except Exception:
                pass
        self._config.addItems(names)
        self._ckpt.clear()
        self._ckpt.addItem("research/checkpoints/ForgeLM_V2_Light.safetensors",
                           "research/checkpoints/ForgeLM_V2_Light.safetensors")
        if self.models_index is not None:
            try:
                for m in self.models_index.models():
                    if "lora" not in m.name.lower() and m.is_safetensors:
                        self._ckpt.addItem(m.name, m.path)
            except Exception:
                pass
        self._ckpt.setCurrentIndex(0)

    # ── left: datasets ────────────────────────────────────────────────
    def _build_datasets_pane(self) -> QWidget:
        left = QFrame(); left.setObjectName("card")
        ll = QVBoxLayout(left); ll.setContentsMargins(16, 14, 16, 16)
        ll.setSpacing(10)
        ll.addWidget(section_label("Datasets"))
        self._ds_list = QListWidget()
        self._ds_list.setObjectName("datasetList")
        self._ds_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._ds_list.itemSelectionChanged.connect(self._update_info)
        ll.addWidget(self._ds_list, 1)
        self._ds_info = QLabel("select one or more JSONL datasets")
        self._ds_info.setObjectName("chatMeta")
        self._ds_info.setWordWrap(True)
        ll.addWidget(self._ds_info)
        export_row = QHBoxLayout(); export_row.setSpacing(8)
        export_btn = QPushButton("★ Export rated chats")
        export_btn.setToolTip("Good-rated Chat Studio turns → sft_train JSONL")
        export_btn.clicked.connect(self._export_rated)
        refresh_btn = QPushButton("Rescan")
        refresh_btn.clicked.connect(self.refresh)
        export_row.addWidget(export_btn); export_row.addWidget(refresh_btn)
        export_row.addStretch(1)
        ll.addLayout(export_row)
        return left

    # ── right: trainer config ─────────────────────────────────────────
    def _build_config_pane(self) -> QWidget:
        host = QWidget()
        rl = QVBoxLayout(host); rl.setContentsMargins(16, 14, 16, 16)
        rl.setSpacing(12)
        rl.addWidget(section_label("Trainer (sft_train.py)"))

        # base model card
        base_card = QFrame(); base_card.setObjectName("cardAlt")
        bl = QVBoxLayout(base_card); bl.setContentsMargins(14, 12, 14, 14)
        bl.setSpacing(8)
        br0 = QHBoxLayout(); br0.setSpacing(10)
        br0.addWidget(QLabel("Config"))
        self._config = QComboBox(); self._config.setEditable(True)
        br0.addWidget(self._config, 1)
        br0.addWidget(QLabel("Checkpoint"))
        self._ckpt = QComboBox()
        br0.addWidget(self._ckpt, 1)
        bl.addLayout(br0)

        # mode card
        mode_card = QFrame(); mode_card.setObjectName("cardAlt")
        ml = QVBoxLayout(mode_card); ml.setContentsMargins(14, 12, 14, 14)
        ml.setSpacing(8)
        mode_row = QHBoxLayout(); mode_row.setSpacing(10)
        self._use_lora = QComboBox()
        self._use_lora.addItems(["LoRA (recommended)", "Full fine-tune"])
        self._use_lora.currentIndexChanged.connect(self._preview)
        mode_row.addWidget(QLabel("Mode"), 0)
        mode_row.addWidget(self._use_lora, 1)
        ml.addLayout(mode_row)
        lora_row = QHBoxLayout(); lora_row.setSpacing(10)
        lora_row.addWidget(QLabel("Rank"))
        self._lora_r = _spin(1, 256, 32); self._lora_r.valueChanged.connect(lambda _v: self._preview())
        lora_row.addWidget(self._lora_r)
        lora_row.addWidget(QLabel("Alpha"))
        self._lora_alpha = _spin(1, 512, 64); self._lora_alpha.valueChanged.connect(lambda _v: self._preview())
        lora_row.addWidget(self._lora_alpha)
        lora_row.addStretch(1)
        ml.addLayout(lora_row)
        self._save_adapter = QCheckBox("Save LoRA adapter only (hot-loadable in LoRA Manager)")
        self._save_adapter.setChecked(True)
        self._save_adapter.setToolTip(
            "Adds --save-lora-adapter: writes <save>.lora.safetensors with just "
            "the trained LoRA tensors. Hot-load / swap it on the resident "
            "engine, or merge it later — no full checkpoint needed.")
        self._save_adapter.toggled.connect(self._preview)
        ml.addWidget(self._save_adapter)
        self._bitnet = QCheckBox("BitNet everywhere (ternary QAT, manual LoRA)")
        self._bitnet.setChecked(True)
        self._bitnet.setToolTip("--bitnet-everywhere: all nn.Linear → BitNet b1.58 "
                                "QAT. Auto-selects the BitNet-compatible manual "
                                "LoRA path. Validated: 2.39x vs AdamW, 6.32GB VRAM.")
        self._bitnet.toggled.connect(self._preview)
        ml.addWidget(self._bitnet)
        bl.addWidget(mode_card)
        rl.addWidget(base_card)
        rl.addWidget(mode_card)

        # schedule card
        sched_card = QFrame(); sched_card.setObjectName("cardAlt")
        sl = QVBoxLayout(sched_card); sl.setContentsMargins(14, 12, 14, 14)
        sl.setSpacing(8)
        self._cfg_rows: dict[str, QDoubleSpinBox] = {}
        for key, label, lo, hi, val, dec, step in [
            ("max_steps", "Max steps", 1, 100000, 500, 0, 10),
            ("lr", "Learning rate", 1e-7, 1e-2, 5e-5, 7, 1e-5),
            ("min_lr", "Min LR", 0.0, 1e-2, 5e-6, 7, 1e-6),
            ("warmup", "Warmup steps", 0, 1000, 20, 0, 5),
            ("batch_size", "Batch size", 1, 64, 1, 0, 1),
            ("grad_accum", "Grad accum", 1, 64, 5, 0, 1),
            ("seq_len", "Seq length", 128, 32768, 1024, 0, 128),
        ]:
            s = _spin(lo, hi, val, dec, step)
            s.valueChanged.connect(lambda _v: self._preview())
            self._cfg_rows[key] = s
            sl.addLayout(_row(label, s))
        rl.addWidget(sched_card)

        # optimizer + loss card
        adv_card = QFrame(); adv_card.setObjectName("cardAlt")
        al = QVBoxLayout(adv_card); al.setContentsMargins(14, 12, 14, 14)
        al.setSpacing(8)
        self._optimizer = QComboBox(); self._optimizer.addItems(OPTIMIZERS)
        self._optimizer.setCurrentText("muon_sf")
        self._optimizer.setToolTip(
            "cpu_offload = CPUAdamW-style offload (mixed CPU/GPU, safest for "
            "12GB); badam = block-wise ADAM; muon_sf = Muon+SF (fastest).")
        self._optimizer.currentIndexChanged.connect(lambda _i: self._preview())
        al.addLayout(_row("Optimizer", self._optimizer))
        self._loss = QComboBox(); self._loss.addItems(LOSSES)
        self._loss.currentIndexChanged.connect(lambda _i: self._preview())
        self._loss.setToolTip("focal = hard-token focus; label_smoothing = "
                              "anti-overconfidence; mixture = all of the above.")
        al.addLayout(_row("Loss", self._loss))
        self._entropy_alpha = _spin(0.0, 2.0, 0.5, 2, 0.1)
        self._entropy_alpha.setToolTip("Token-entropy weighting (0 disables)")
        self._entropy_alpha.valueChanged.connect(lambda _v: self._preview())
        al.addLayout(_row("Entropy α", self._entropy_alpha))
        self._curriculum = QComboBox(); self._curriculum.addItems(CURRICULA)
        self._curriculum.currentIndexChanged.connect(lambda _i: self._preview())
        self._curriculum.setToolTip("Easy→hard data ordering: 18-45% fewer steps "
                                    "to baseline.")
        al.addLayout(_row("Curriculum", self._curriculum))
        wd_row = QHBoxLayout(); wd_row.setSpacing(10)
        wd_row.addWidget(QLabel("Weight decay"))
        self._wd = _spin(0.0, 1.0, 0.01, 3, 0.01)
        self._wd.valueChanged.connect(lambda _v: self._preview())
        wd_row.addWidget(self._wd)
        wd_row.addWidget(QLabel("Grad clip"))
        self._clip = _spin(0.0, 10.0, 1.0, 2, 0.1)
        self._clip.valueChanged.connect(lambda _v: self._preview())
        wd_row.addWidget(self._clip)
        wd_row.addStretch(1)
        al.addLayout(wd_row)
        rl.addWidget(adv_card)

        # extras card
        extras = QFrame(); extras.setObjectName("cardAlt")
        el = QVBoxLayout(extras); el.setContentsMargins(14, 12, 14, 14)
        el.setSpacing(8)
        ex_row = QHBoxLayout(); ex_row.setSpacing(16)
        self._ema = QCheckBox("EMA")
        self._ema.setToolTip("Exponential moving average weights (smoother model)")
        self._ema.toggled.connect(self._preview)
        self._augment = QCheckBox("Augment")
        self._augment.setToolTip("Token noise + FIM augmentation (anti-overfit)")
        self._augment.toggled.connect(self._preview)
        self._synpro = QCheckBox("SYNPRO")
        self._synpro.setToolTip("Synthetic rephrase/reformat (3.7-5.2x tokens)")
        self._synpro.toggled.connect(self._preview)
        for cb in (self._ema, self._augment, self._synpro):
            ex_row.addWidget(cb)
        ex_row.addStretch(1)
        ex_row.addWidget(QLabel("Validate every"))
        self._val_every = QSpinBox(); self._val_every.setRange(0, 1000)
        self._val_every.setValue(0)
        self._val_every.setToolTip("0 = off; runs a held-out val split")
        self._val_every.valueChanged.connect(lambda _v: self._preview())
        ex_row.addWidget(self._val_every)
        el.addLayout(ex_row)
        rl.addWidget(extras)

        # output card
        out_card = QFrame(); out_card.setObjectName("cardAlt")
        ol = QVBoxLayout(out_card); ol.setContentsMargins(14, 12, 14, 14)
        ol.setSpacing(8)
        save_row = QHBoxLayout(); save_row.setSpacing(10)
        save_row.addWidget(QLabel("Save to"))
        self._save = QLineEdit("research/checkpoints/ForgeLM_V2_Light.sft.safetensors")
        save_row.addWidget(self._save, 1)
        ol.addLayout(save_row)
        rl.addWidget(out_card)

        # preview + launch
        self._preview_lbl = QLabel("")
        self._preview_lbl.setObjectName("mono")
        self._preview_lbl.setWordWrap(True)
        self._preview_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self._preview_lbl)
        launch_btn = QPushButton("⚡ Launch fine-tune")
        launch_btn.setObjectName("primary")
        launch_btn.clicked.connect(self._launch)
        rl.addWidget(launch_btn)
        self._status = QLabel("idle")
        self._status.setObjectName("chatMeta")
        self._status.setWordWrap(True)
        rl.addWidget(self._status)
        rl.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(host)
        scroll.setObjectName("root")
        return scroll

    # ── datasets ──────────────────────────────────────────────────────
    def refresh(self) -> None:
        selected = self._selected_paths()
        self._ds_list.clear()
        seen = set()
        for ex in self.store.list_exports():
            self._add_ds_item(f"★ {ex['name']}  ({ex['examples']} ex)",
                              ex["path"], seen)
        root = project_root() / "data"
        if root.is_dir():
            for p in sorted(root.rglob("*.jsonl")):
                if str(p) in seen:
                    continue
                try:
                    n = sum(1 for line in p.read_text(encoding="utf-8")
                            .splitlines() if line.strip())
                except OSError:
                    n = 0
                self._add_ds_item(f"{p.name}  ({n} ex)",
                                  str(p), seen)
        for i in range(self._ds_list.count()):
            item = self._ds_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in selected:
                item.setSelected(True)
        self._update_info()

    def _add_ds_item(self, label: str, path: str, seen: set) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._ds_list.addItem(item)
        seen.add(path)

    def _selected_paths(self) -> list[str]:
        return [i.data(Qt.ItemDataRole.UserRole)
                for i in self._ds_list.selectedItems()]

    def _update_info(self) -> None:
        paths = self._selected_paths()
        if paths:
            self._ds_info.setText(
                f"{len(paths)} dataset(s) selected:\n"
                + "\n".join(Path(p).name for p in paths[:6]))
        else:
            self._ds_info.setText("select one or more JSONL datasets")
        self._preview()

    def _export_rated(self) -> None:
        try:
            path, n = self.store.export_training_data()
            self._status.setText(f"exported {n} examples → {path}")
            self.refresh()
        except Exception as e:
            self._status.setText(f"export failed: {e}")

    # ── command build / preview / launch ──────────────────────────────
    def _build_cmd(self) -> list[str] | None:
        paths = self._selected_paths()
        if not paths:
            return None
        root = project_root()
        venv_python = root / "venv" / "Scripts" / "python.exe"
        cmd = [str(venv_python if venv_python.is_file() else "python"),
               str(root / "research" / "training" / "runners" / "sft_train.py"),
               "--data", *paths,
               "--config", self._config.currentText().strip() or "forgelm_v2_light",
               "--checkpoint", self._ckpt.currentData()
               or "research/checkpoints/ForgeLM_V2_Light.safetensors",
               "--max-steps", str(int(self._cfg_rows["max_steps"].value())),
               "--lr", f"{self._cfg_rows['lr'].value():.7g}",
               "--min-lr", f"{self._cfg_rows['min_lr'].value():.7g}",
               "--batch-size", str(int(self._cfg_rows["batch_size"].value())),
               "--grad-accum", str(int(self._cfg_rows["grad_accum"].value())),
               "--seq-len", str(int(self._cfg_rows["seq_len"].value())),
               "--warmup-steps", str(int(self._cfg_rows["warmup"].value())),
               "--weight-decay", f"{self._wd.value():.3g}",
               "--grad-clip", f"{self._clip.value():.3g}",
               "--optimizer", self._optimizer.currentText(),
               "--loss-function", self._loss.currentText(),
               "--entropy-alpha", f"{self._entropy_alpha.value():.2g}",
               "--curriculum", self._curriculum.currentText(),
               "--save", self._save.text().strip(),
               ]
        if self._use_lora.currentIndex() == 0:
            cmd += ["--lora",
                    "--lora-r", str(int(self._lora_r.value())),
                    "--lora-alpha", str(int(self._lora_alpha.value()))]
            if self._save_adapter.isChecked():
                cmd.append("--save-lora-adapter")
        else:
            cmd.append("--no-lora")
        if self._bitnet.isChecked():
            cmd.append("--bitnet-everywhere")
        else:
            cmd.append("--no-bitnet-everywhere")
        if self._ema.isChecked():
            cmd.append("--ema")
        if self._augment.isChecked():
            cmd.append("--augment")
        if self._synpro.isChecked():
            cmd.append("--synpro")
        if self._val_every.value() > 0:
            cmd += ["--val-every", str(self._val_every.value())]
        return cmd

    def _preview(self) -> None:
        if not hasattr(self, "_preview_lbl"):
            return
        cmd = self._build_cmd()
        if cmd is None:
            self._preview_lbl.setText("Select datasets to build the command.")
            return
        self._preview_lbl.setText("  " + " ".join(cmd))

    def _launch(self) -> None:
        cmd = self._build_cmd()
        if cmd is None:
            QMessageBox.information(self, "No dataset",
                                    "Select at least one dataset on the left "
                                    "(or export rated chats first).")
            return
        paths = self._selected_paths()
        name = f"fine-tune · {Path(paths[0]).name}"
        try:
            task_id = self.proc_mgr.launch(name, cmd)
            self._status.setText(f"launched task {task_id} — live logs on the "
                                 f"Tasks page")
            if self._use_lora.currentIndex() == 0 and self._save_adapter.isChecked():
                adapter = Path(self._save.text().strip()).with_suffix(
                    ".lora.safetensors")
                self._status.setText(
                    f"launched {task_id} · adapter will be saved to {adapter.name} "
                    f"— hot-load it on the LoRA page after training")
        except Exception as e:
            logger.warning("launch failed: %s", e, exc_info=True)
            self._status.setText(f"launch failed: {e}")
