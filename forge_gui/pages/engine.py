"""Engine Studio — the control surface for the resident ForgeEngine.

Three tabs:
  1. Load & Power  — checkpoint + config + activation mode, load/unload,
                    sleep (L1 CPU offload / L2 discard) / wake, live VRAM.
  2. Activation    — EVERY ForgeEngine feature (mirrors ActivationConfig):
                    presets, core strategies (KV cache / decoding /
                    quantization / acceleration with per-option help),
                    numeric knobs and all 45+ feature flags grouped by
                    category. Apply live (no reload) or at next load.
  3. Stats & Tools — live stats, LoRA state, benchmark / bottleneck /
                    diagnose, engine log, crash recovery, reset stats.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QProgressBar, QRadioButton, QScrollArea,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..api.activation_catalog import (FIELDS, PRESETS, FieldSpec,
                                       active_diff, default_config,
                                       fields_by_category, normalize_value,
                                       preset_config, validate)
from ..api.engine_runtime import EngineRuntime
from ..theme import Palette
from ..widgets.search_combo import SearchableComboBox
from ._base import section_label

logger = logging.getLogger(__name__)


# ── workers ─────────────────────────────────────────────────────────────

class _MaintWorker(QThread):
    """Runs an engine maintenance action (benchmark/diagnose/…) off-UI."""

    output = Signal(str)
    done = Signal(str)

    def __init__(self, runtime: EngineRuntime, action: str,
                 prompt: str = "The quick brown fox", max_new_tokens: int = 64,
                 n_runs: int = 3, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.action = action
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.n_runs = n_runs

    def run(self) -> None:
        try:
            with self.runtime.acquire() as engine:
                if self.action == "benchmark":
                    rep = engine.benchmark(prompt=self.prompt,
                                           max_new_tokens=self.max_new_tokens,
                                           n_runs=self.n_runs)
                    self.output.emit(_fmt_bench(rep))
                elif self.action == "bottleneck":
                    rep = engine.bottleneck(prompt=self.prompt,
                                            max_new_tokens=min(
                                                16, self.max_new_tokens))
                    lines = [f"  {b['type']}#{b['index']}: {b['time_ms']:.1f} ms"
                             for b in rep.get("bottlenecks", [])[:8]]
                    self.output.emit(
                        f"tok/s: {rep.get('tok_s', rep.get('tokens_per_sec', '?'))}\n"
                        f"slowest blocks:\n" + "\n".join(lines))
                elif self.action == "diagnose":
                    rep = engine.diagnose()
                    self.output.emit(_fmt_report(rep))
                elif self.action == "read_log":
                    entries = engine.read_log(n=40)
                    if not entries:
                        self.output.emit("(engine log is empty)")
                    else:
                        self.output.emit("\n".join(
                            f"[{e.get('level', 'info')}] {e.get('source', '?')}: "
                            f"{e.get('message', e)}" for e in entries))
                elif self.action == "recover":
                    rep = engine.recover()
                    self.output.emit(_fmt_report(rep) if rep
                                     else "no crash recovery state found")
                elif self.action == "clear_recovery":
                    engine.clear_recovery()
                    self.output.emit("recovery state cleared")
                elif self.action == "reset_stats":
                    engine.reset_stats()
                    self.output.emit("stats reset")
                else:
                    self.output.emit(f"unknown action {self.action}")
            self.done.emit(self.action)
        except Exception as e:
            logger.warning("maintenance action %s failed: %s", self.action, e)
            self.output.emit(f"error: {type(e).__name__}: {e}")
            self.done.emit(self.action)


def _fmt_bench(rep: Any) -> str:
    if isinstance(rep, dict):
        keys = ["tokens_per_sec", "tok_s", "latency_ms", "tokens",
                "generated_tokens", "runs", "prompt_toks", "gen_toks",
                "prefill_s", "decode_s"]
        vals = {k: rep.get(k) for k in keys if rep.get(k) is not None}
        if vals:
            return "\n".join(f"{k}: {v}" for k, v in vals.items())
    return str(rep)[:2000]


def _fmt_report(rep: Any) -> str:
    if isinstance(rep, dict):
        out = []
        for k, v in rep.items():
            s = str(v)
            if len(s) > 160:
                s = s[:160] + "…"
            out.append(f"{k}: {s}")
        return "\n".join(out) if out else "(empty report)"
    return str(rep)[:2000]


def _kv_row(key: str) -> tuple[QFrame, QLabel]:
    row = QFrame(); row.setObjectName("kvRow")
    h = QHBoxLayout(row); h.setContentsMargins(12, 8, 12, 8); h.setSpacing(12)
    k = QLabel(key); k.setObjectName("kvKey"); k.setMinimumWidth(110)
    v = QLabel("—"); v.setObjectName("kvVal"); v.setWordWrap(True)
    v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    h.addWidget(k, 0); h.addWidget(v, 1)
    return row, v


# ── page ────────────────────────────────────────────────────────────────

class EnginePage(QWidget):
    def __init__(self, runtime: EngineRuntime, models_index=None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.models_index = models_index
        self._maint: Optional[_MaintWorker] = None
        self._preset_group: dict[str, QPushButton] = {}
        self._combo_specs: dict[str, tuple[QComboBox, QLabel]] = {}
        self._spin_fields: dict[str, QWidget] = {}
        self._flag_checks: dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_load_tab(), "Load & Power")
        self._tabs.addTab(self._build_activation_tab(), "Activation")
        self._tabs.addTab(self._build_tools_tab(), "Stats & Tools")
        outer.addWidget(self._tabs)

        # elapsed-time ticker while loading
        self._load_t0 = 0.0
        self._elapsed = QTimer(self)
        self._elapsed.setInterval(500)
        self._elapsed.timeout.connect(self._tick_elapsed)

        self.runtime.state_changed.connect(self._on_state)
        self.runtime.progress.connect(self._on_progress)
        self.runtime.ready.connect(lambda info: self._on_state("ready"))
        self.runtime.failed.connect(self._on_failed)
        self.runtime.reactivating.connect(
            lambda: self._apply_status.setText("applying features to resident engine…"))
        self.runtime.reactivated.connect(self._on_reactivated)
        self._on_state(self.runtime.state)

    # ── tab 1: load & power ──────────────────────────────────────────
    def _build_load_tab(self) -> QWidget:
        host = QWidget(); col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18); col.setSpacing(14)

        model_card = QFrame(); model_card.setObjectName("card")
        ml = QVBoxLayout(model_card); ml.setContentsMargins(16, 14, 16, 16)
        ml.setSpacing(10)
        ml.addWidget(section_label("Model"))
        self._ckpt = SearchableComboBox()
        self._reload_checkpoints()
        ml.addWidget(self._ckpt)
        cfg_row = QHBoxLayout(); cfg_row.setSpacing(10)
        cfg_row.addWidget(QLabel("Config"))
        self._config = QComboBox()
        self._config.setEditable(True)
        self._reload_configs()
        cfg_row.addStretch(1); cfg_row.addWidget(self._config, 1)
        ml.addLayout(cfg_row)

        # activation mode
        ml.addWidget(section_label("Activation mode"))
        self._mode_auto = QRadioButton("Auto — engine-chosen optimal preset "
                                       "(VRAM/KeyStack aware)")
        self._mode_manual = QRadioButton("Manual — use the Activation tab config")
        self._mode_auto.setChecked(True)
        self._mode_auto.toggled.connect(self._on_mode_changed)
        ml.addWidget(self._mode_auto); ml.addWidget(self._mode_manual)
        self._fast_load = QCheckBox("Fast load (skip torch.compile)")
        self._fast_load.setChecked(True)
        self._fast_load.setToolTip(
            "ON: ~15-45s load, eager kernels — reliable, ~5.4GB VRAM.\n"
            "OFF: +2-3 min compile, 1.3-2x faster decode — BLOCKED: "
            "IRIFP4Linear weight cache is incompatible with CUDA-graph "
            "memory reuse (see AGENTS.md).")
        ml.addWidget(self._fast_load)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self._load_btn = QPushButton("⏏ Load")
        self._load_btn.setObjectName("primary")
        self._unload_btn = QPushButton("Unload")
        self._unload_btn.setObjectName("danger")
        btn_row.addWidget(self._load_btn); btn_row.addWidget(self._unload_btn)
        btn_row.addStretch(1)
        ml.addLayout(btn_row)

        self._load_bar = QProgressBar()
        self._load_bar.setRange(0, 0)
        self._load_bar.setVisible(False)
        self._load_bar.setFixedHeight(14)
        ml.addWidget(self._load_bar)
        self._state_lbl = QLabel("state: idle")
        self._state_lbl.setObjectName("engineState")
        ml.addWidget(self._state_lbl)
        self._info_lbl = QLabel("no engine resident")
        self._info_lbl.setObjectName("kvVal")
        self._info_lbl.setWordWrap(True)
        ml.addWidget(self._info_lbl)
        col.addWidget(model_card)

        power_card = QFrame(); power_card.setObjectName("card")
        pl = QVBoxLayout(power_card); pl.setContentsMargins(16, 14, 16, 16)
        pl.setSpacing(10)
        pl.addWidget(section_label("Power (VRAM release without unload)"))
        row = QHBoxLayout(); row.setSpacing(10)
        self._sleep1_btn = QPushButton("Sleep L1 · offload to CPU")
        self._sleep2_btn = QPushButton("Sleep L2 · discard (wake = reload)")
        self._wake_btn = QPushButton("Wake")
        for b in (self._sleep1_btn, self._sleep2_btn, self._wake_btn):
            row.addWidget(b)
        pl.addLayout(row)
        self._power_lbl = QLabel("")
        self._power_lbl.setObjectName("chatMeta")
        pl.addWidget(self._power_lbl)
        col.addWidget(power_card)

        vram_card = QFrame(); vram_card.setObjectName("card")
        vl = QVBoxLayout(vram_card); vl.setContentsMargins(16, 14, 16, 16)
        vl.setSpacing(10)
        vl.addWidget(section_label("VRAM"))
        self._vram_bar = QProgressBar(); self._vram_bar.setRange(0, 100)
        self._vram_bar.setFixedHeight(18)
        vl.addWidget(self._vram_bar)
        self._vram_lbl = QLabel("—")
        self._vram_lbl.setObjectName("kvVal")
        vl.addWidget(self._vram_lbl)
        col.addWidget(vram_card)
        col.addStretch(1)

        # signals
        self._load_btn.clicked.connect(self._load)
        self._unload_btn.clicked.connect(self.runtime.unload)
        self._sleep1_btn.clicked.connect(lambda: self._sleep(1))
        self._sleep2_btn.clicked.connect(lambda: self._sleep(2))
        self._wake_btn.clicked.connect(self._wake)
        return _scroll(host)

    # ── tab 2: activation studio ─────────────────────────────────────
    def _build_activation_tab(self) -> QWidget:
        host = QWidget(); col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18); col.setSpacing(14)

        # presets
        preset_card = QFrame(); preset_card.setObjectName("card")
        pl = QVBoxLayout(preset_card); pl.setContentsMargins(16, 14, 16, 16)
        pl.setSpacing(10)
        pl.addWidget(section_label("Presets"))
        chips = QHBoxLayout(); chips.setSpacing(8)
        for p in PRESETS:
            chip = QPushButton(p.label)
            chip.setObjectName("presetChip")
            chip.setCheckable(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(p.description)
            chip.clicked.connect(lambda _=False, name=p.name: self._apply_preset(name))
            self._preset_group[p.name] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        pl.addLayout(chips)
        self._preset_desc = QLabel("Pick a preset to fill every option below, "
                                   "then fine-tune individual features.")
        self._preset_desc.setObjectName("cardBody")
        self._preset_desc.setWordWrap(True)
        pl.addWidget(self._preset_desc)
        row = QHBoxLayout(); row.setSpacing(10)
        self._match_btn = QPushButton("Copy resident engine's config")
        self._match_btn.clicked.connect(self._copy_resident)
        row.addWidget(self._match_btn); row.addStretch(1)
        pl.addLayout(row)
        col.addWidget(preset_card)

        # every category → one card
        for cat, fields in fields_by_category():
            card = QFrame(); card.setObjectName("card")
            cl = QVBoxLayout(card); cl.setContentsMargins(16, 14, 16, 16)
            cl.setSpacing(10)
            cl.addWidget(section_label(cat.upper()))
            grid_host = QWidget()
            grid = QGridLayout(grid_host)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(18); grid.setVerticalSpacing(8)
            g_row, g_col = 0, 0
            n_flags = 0
            for f in fields:
                if f.kind == "combo":
                    cl.addLayout(self._build_combo_row(f))
                elif f.kind in ("int", "opt_int", "opt_float"):
                    cl.addLayout(self._build_spin_row(f))
                else:  # bool flag → 3-column checkbox grid
                    cb = _flag_check(f)
                    cb.toggled.connect(lambda _on: self._update_diff())
                    self._flag_checks[f.name] = cb
                    grid.addWidget(cb, g_row, g_col)
                    n_flags += 1
                    g_col += 1
                    if g_col == 3:
                        g_col = 0; g_row += 1
            if n_flags:
                for c in range(3):
                    grid.setColumnStretch(c, 1)
                cl.addWidget(grid_host)
            col.addWidget(card)

        # apply card
        apply_card = QFrame(); apply_card.setObjectName("card")
        al = QVBoxLayout(apply_card); al.setContentsMargins(16, 14, 16, 16)
        al.setSpacing(10)
        al.addWidget(section_label("Apply"))
        self._diff_lbl = QLabel("")
        self._diff_lbl.setObjectName("cardBody")
        self._diff_lbl.setWordWrap(True)
        al.addWidget(self._diff_lbl)
        row = QHBoxLayout(); row.setSpacing(10)
        self._apply_live_btn = QPushButton("⚡ Apply to resident engine")
        self._apply_live_btn.setObjectName("primary")
        self._apply_live_btn.setToolTip(
            "Re-activates runtime strategies on the loaded engine WITHOUT "
            "reloading weights — swap KV cache / quantization / decoding live.")
        self._apply_live_btn.clicked.connect(self._apply_live)
        row.addWidget(self._apply_live_btn)
        row.addStretch(1)
        al.addLayout(row)
        self._apply_status = QLabel("Changes are also used at next load when "
                                    "Manual mode is selected on the Load tab.")
        self._apply_status.setObjectName("chatMeta")
        self._apply_status.setWordWrap(True)
        al.addWidget(self._apply_status)
        col.addWidget(apply_card)
        col.addStretch(1)

        self._fill_form(default_config())
        return _scroll(host)

    def _build_combo_row(self, f: FieldSpec):
        row = QHBoxLayout(); row.setSpacing(10)
        lbl = QLabel(f.label); lbl.setMinimumWidth(150)
        lbl.setToolTip(f.tooltip)
        row.addWidget(lbl)
        combo = QComboBox()
        for o in f.options:
            combo.addItem(o.label, o.value)
            if o.tip:
                combo.setItemData(combo.count() - 1, o.tip, Qt.ItemDataRole.ToolTipRole)
        combo.currentIndexChanged.connect(
            lambda _i, c=combo, f=f: self._on_combo_changed(c, f))
        row.addWidget(combo, 1)
        desc = QLabel(""); desc.setObjectName("chatMeta"); desc.setWordWrap(True)
        row.addWidget(desc, 2)
        self._combo_specs[f.name] = (combo, desc)
        self._on_combo_changed(combo, f)
        return row

    def _on_combo_changed(self, combo: QComboBox, f: FieldSpec) -> None:
        idx = combo.currentIndex()
        spec = f.options[idx] if 0 <= idx < len(f.options) else None
        desc = self._combo_specs.get(f.name, (None, None))[1]
        if desc is not None and spec is not None:
            desc.setText(spec.tip or spec.label)
        self._update_diff()

    def _build_spin_row(self, f: FieldSpec):
        row = QHBoxLayout(); row.setSpacing(10)
        lbl = QLabel(f.label); lbl.setMinimumWidth(150)
        lbl.setToolTip(f.tooltip)
        row.addWidget(lbl)
        if f.kind == "opt_float":
            s = QDoubleSpinBox()
            s.setRange(f.lo, f.hi); s.setDecimals(f.decimals)
            s.setSingleStep(f.step)
            s.setValue(float(f.default) if f.default is not None else 0.0)
        elif f.kind == "opt_int":
            s = QSpinBox()
            s.setRange(int(f.lo), int(f.hi)); s.setSingleStep(int(f.step))
            s.setValue(int(f.default) if f.default is not None else 0)
        else:
            s = QSpinBox()
            s.setRange(int(f.lo), int(f.hi)); s.setSingleStep(int(f.step))
            s.setValue(int(f.default or 0))
        if f.suffix:
            s.setSuffix(f.suffix)
        s.setToolTip(f.tooltip + ("\n(0 = off / model default)"
                                  if f.default is None else ""))
        s.valueChanged.connect(lambda _v: self._update_diff())
        row.addStretch(1); row.addWidget(s)
        self._spin_fields[f.name] = s
        return row

    # ── tab 3: stats & tools ─────────────────────────────────────────
    def _build_tools_tab(self) -> QWidget:
        host = QWidget(); col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18); col.setSpacing(14)

        stats_card = QFrame(); stats_card.setObjectName("card")
        sl = QVBoxLayout(stats_card); sl.setContentsMargins(16, 14, 16, 16)
        sl.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(section_label("Engine stats"))
        head.addStretch(1)
        self._stats_refresh_btn = QPushButton("Refresh")
        self._stats_refresh_btn.clicked.connect(self._poll_stats)
        head.addWidget(self._stats_refresh_btn)
        sl.addLayout(head)
        self._stat_rows: dict[str, QLabel] = {}
        for key in ("Resident", "Generations", "Decoding", "Quantization",
                    "KV cache", "Acceleration", "LoRA", "VRAM", "Power"):
            row, val = _kv_row(key)
            self._stat_rows[key] = val
            sl.addWidget(row)
        col.addWidget(stats_card)

        maint_card = QFrame(); maint_card.setObjectName("card")
        ml2 = QVBoxLayout(maint_card); ml2.setContentsMargins(16, 14, 16, 16)
        ml2.setSpacing(10)
        ml2.addWidget(section_label("Maintenance"))
        pr = QHBoxLayout(); pr.setSpacing(10)
        pr.addWidget(QLabel("Prompt"))
        self._bench_prompt = QLineEdit("The quick brown fox")
        pr.addWidget(self._bench_prompt, 1)
        ml2.addLayout(pr)
        nr = QHBoxLayout(); nr.setSpacing(10)
        nr.addWidget(QLabel("Tokens"))
        self._bench_tok = QSpinBox(); self._bench_tok.setRange(8, 2048)
        self._bench_tok.setValue(64)
        nr.addWidget(self._bench_tok)
        nr.addWidget(QLabel("Runs"))
        self._bench_runs = QSpinBox(); self._bench_runs.setRange(1, 10)
        self._bench_runs.setValue(3)
        nr.addWidget(self._bench_runs)
        nr.addStretch(1)
        ml2.addLayout(nr)
        act = QHBoxLayout(); act.setSpacing(8)
        for label, action in (("Benchmark", "benchmark"),
                              ("Bottleneck", "bottleneck"),
                              ("Diagnose", "diagnose"),
                              ("Engine log", "read_log")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, a=action: self._maintain(a))
            act.addWidget(b)
        act.addStretch(1)
        ml2.addLayout(act)
        self._maint_out = QPlainTextEdit()
        self._maint_out.setObjectName("logView")
        self._maint_out.setReadOnly(True)
        self._maint_out.setFixedHeight(180)
        ml2.addWidget(self._maint_out)
        col.addWidget(maint_card)

        rec_card = QFrame(); rec_card.setObjectName("card")
        rl = QVBoxLayout(rec_card); rl.setContentsMargins(16, 14, 16, 16)
        rl.setSpacing(10)
        rl.addWidget(section_label("Crash recovery & reset"))
        row = QHBoxLayout(); row.setSpacing(8)
        self._recover_btn = QPushButton("Inspect recovery state")
        self._recover_btn.clicked.connect(lambda: self._maintain("recover"))
        self._clear_rec_btn = QPushButton("Clear")
        self._clear_rec_btn.clicked.connect(lambda: self._maintain("clear_recovery"))
        self._reset_stats_btn = QPushButton("Reset stats")
        self._reset_stats_btn.clicked.connect(lambda: self._maintain("reset_stats"))
        for b in (self._recover_btn, self._clear_rec_btn, self._reset_stats_btn):
            row.addWidget(b)
        row.addStretch(1)
        rl.addLayout(row)
        self._rec_lbl = QLabel("If a generation crashed mid-flight, ForgeEngine "
                               "keeps a recovery snapshot — inspect it here.")
        self._rec_lbl.setObjectName("chatMeta"); self._rec_lbl.setWordWrap(True)
        rl.addWidget(self._rec_lbl)
        col.addWidget(rec_card)
        col.addStretch(1)
        return _scroll(host)

    # ── checkpoints / configs ──────────────────────────────────────────
    def _reload_checkpoints(self) -> None:
        self._ckpt.clear()
        added = False
        if self.models_index is not None:
            try:
                for m in self.models_index.models():
                    # LoRA adapters can't boot as base models
                    if "lora" in m.name.lower():
                        continue
                    if m.is_safetensors:
                        self._ckpt.addItem(m.name, m.path)
                        added = True
            except Exception as e:
                logger.warning("checkpoint list failed: %s", e)
        if not added:
            self._ckpt.addItem("ForgeLM_V2_Light.safetensors",
                               "research/checkpoints/ForgeLM_V2_Light.safetensors")

    def _reload_configs(self) -> None:
        self._config.clear()
        names = ["forgelm_v2_light"]
        if self.models_index is not None:
            try:
                names = [c.name for c in self.models_index.configs()] or names
            except Exception as e:
                logger.warning("config list failed: %s", e)
        self._config.addItems(names)

    def refresh(self) -> None:
        self._poll_stats()
        self._update_vram()

    # ── load / unload / power ─────────────────────────────────────────
    def _load(self) -> None:
        path = self._ckpt.currentSearchData() or self._ckpt.currentData() \
            or self._ckpt.currentText()
        cfg = self._config.currentText().strip() or "forgelm_v2_light"
        if self._mode_manual.isChecked():
            activation = self._collect_config()
            errors = validate(activation)
            if errors:
                QMessageBox.warning(self, "Invalid activation config",
                                    "\n".join(errors))
                return
            self.runtime.load(path, cfg, activation=activation)
        else:
            self.runtime.load(path, cfg,
                              use_compile=not self._fast_load.isChecked())

    def _on_mode_changed(self) -> None:
        manual = self._mode_manual.isChecked()
        self._fast_load.setEnabled(not manual)
        if manual:
            self._fast_load.setToolTip(
                "Manual mode: torch.compile follows the Activation tab's "
                "'torch.compile' flag.")
        else:
            self._fast_load.setToolTip(
                "ON: ~15-45s load, eager kernels — reliable, ~5.4GB VRAM.\n"
                "OFF: +2-3 min compile, 1.3-2x faster decode — BLOCKED: "
                "IRIFP4Linear weight cache is incompatible with CUDA-graph "
                "memory reuse (see AGENTS.md).")

    def _tick_elapsed(self) -> None:
        if self.runtime.state == "loading":
            dt = max(0, int(time.perf_counter() - self._load_t0))
            self._state_lbl.setText(f"state: loading · {dt}s elapsed")

    def _sleep(self, level: int) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._power_lbl.setText("no engine")
            return
        try:
            eng.sleep(level=level)
            self._power_lbl.setText(
                "sleeping (L1: weights in CPU RAM, wake ~2-3s)"
                if level == 1 else
                "sleeping (L2: weights discarded, wake reloads checkpoint)")
            self._poll_stats(); self._update_vram()
        except Exception as e:
            self._power_lbl.setText(f"sleep failed: {e}")

    def _wake(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._power_lbl.setText("no engine")
            return
        try:
            eng.wake()
            self._power_lbl.setText("awake")
            self._poll_stats(); self._update_vram()
        except Exception as e:
            self._power_lbl.setText(f"wake failed: {e}")

    # ── state ─────────────────────────────────────────────────────────
    def _on_state(self, state: str) -> None:
        color = {"ready": Palette.ok, "loading": Palette.warn,
                 "error": Palette.err}.get(state, Palette.text_faint)
        self._state_lbl.setText(f"state: {state}")
        self._state_lbl.setStyleSheet(f"color: {color};")
        loading = state == "loading"
        self._load_btn.setEnabled(not loading)
        self._load_btn.setText("⏳ Loading…" if loading else "⏏ Load")
        self._load_bar.setVisible(loading)
        self._ckpt.setEnabled(not loading)
        self._config.setEnabled(not loading)
        if loading:
            self._load_t0 = time.perf_counter()
            self._elapsed.start()
        else:
            self._elapsed.stop()
        if state == "ready":
            info = self.runtime.info
            self._info_lbl.setText(
                f"{info.get('checkpoint', '?')}\n"
                f"config: {info.get('config_name', '?')} · "
                f"device: {info.get('device', '?')} · "
                f"dtype: {info.get('dtype', '?')} · "
                f"loaded in {info.get('load_s', '?')}s"
                + (" · compiled" if info.get("use_compile") else " · fast load"))
            self._select_resident(info.get("checkpoint", ""))
            act = info.get("activation") or {}
            if act:
                self._mark_matching_preset(act)
            self._poll_stats(); self._update_vram()
        elif state == "idle":
            self._info_lbl.setText("no engine resident")
            self._clear_stats()

    def _select_resident(self, checkpoint: str) -> None:
        """Highlight the resident checkpoint in the combo."""
        if not checkpoint:
            return
        want = str(checkpoint).replace("\\", "/").lower()
        for i in range(self._ckpt.count()):
            data = str(self._ckpt.itemData(i) or "").replace("\\", "/").lower()
            if data and (data == want or want.endswith(data)):
                self._ckpt.setCurrentIndex(i)
                return

    def _on_progress(self, msg: str) -> None:
        self._info_lbl.setText(msg)

    def _on_failed(self, err: str) -> None:
        self._info_lbl.setText(f"load failed: {err}")

    # ── activation studio logic ───────────────────────────────────────
    def _apply_preset(self, name: str) -> None:
        cfg = preset_config(name)
        if cfg is None:
            return
        # preserve the compile preference across preset switches unless the
        # preset itself specifies it
        p = next((x for x in PRESETS if x.name == name), None)
        if p is not None and "use_compile" not in p.config:
            cur = self._collect_config().get("use_compile")
            cfg["use_compile"] = bool(cur)
        self._fill_form(cfg)
        for n, chip in self._preset_group.items():
            chip.setChecked(n == name)
        desc = next((x for x in PRESETS if x.name == name), None)
        if desc is not None:
            self._preset_desc.setText(desc.description)
        self._update_diff()

    def _copy_resident(self) -> None:
        act = self.runtime.info.get("activation")
        if not act:
            self._apply_status.setText("no resident engine config to copy")
            return
        # only fields we know how to render
        cfg = {f.name: act.get(f.name, f.default) for f in FIELDS}
        self._fill_form(cfg)
        for chip in self._preset_group.values():
            chip.setChecked(False)
        self._preset_desc.setText("Copied the resident engine's active config.")
        self._update_diff()

    def _fill_form(self, cfg: dict) -> None:
        for f in FIELDS:
            if f.kind == "combo":
                combo = self._combo_specs[f.name][0]
                want = cfg.get(f.name)
                idx = combo.findData(str(want) if want is not None else "none")
                if idx < 0:
                    idx = combo.findData("none")
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif f.kind in ("int", "opt_int", "opt_float"):
                s = self._spin_fields.get(f.name)
                if s is None:
                    continue
                v = cfg.get(f.name)
                if v is None:
                    v = 0
                if isinstance(s, QDoubleSpinBox):
                    s.setValue(float(v))
                else:
                    s.setValue(int(v))
            else:
                cb = self._flag_checks.get(f.name)
                if cb is not None:
                    cb.setChecked(bool(cfg.get(f.name, False)))

    def _collect_config(self) -> dict:
        cfg: dict = {}
        for f in FIELDS:
            if f.kind == "combo":
                combo = self._combo_specs[f.name][0]
                cfg[f.name] = normalize_value(f, combo.currentData())
            elif f.kind in ("opt_int", "opt_float"):
                s = self._spin_fields.get(f.name)
                if s is None:
                    cfg[f.name] = f.default
                    continue
                v = s.value()
                # 0 / min = "off" (None) for optional numerics
                cfg[f.name] = None if not v else normalize_value(f, v)
            elif f.kind == "int":
                s = self._spin_fields.get(f.name)
                cfg[f.name] = int(s.value()) if s is not None else f.default
            else:
                cb = self._flag_checks.get(f.name)
                cfg[f.name] = bool(cb.isChecked()) if cb is not None else f.default
        return cfg

    def _mark_matching_preset(self, active: dict) -> None:
        for p in PRESETS:
            cfg = preset_config(p.name)
            if cfg and all(active.get(k) == v for k, v in cfg.items()
                           if k in active):
                for n, chip in self._preset_group.items():
                    chip.setChecked(n == p.name)
                return
        for chip in self._preset_group.values():
            chip.setChecked(False)

    def _update_diff(self) -> None:
        if not hasattr(self, "_diff_lbl"):
            return  # form still under construction
        active = self.runtime.info.get("activation") or {}
        diff = active_diff(active, self._collect_config())
        if not active:
            self._diff_lbl.setText("No resident engine — this config will be "
                                   "used at load (Manual mode).")
        elif not diff:
            self._diff_lbl.setText("Resident engine already matches this config.")
        else:
            self._diff_lbl.setText("Changes vs resident engine:\n  "
                                   + "\n  ".join(diff[:14]))

    def _apply_live(self) -> None:
        if not self.runtime.is_ready():
            self._apply_status.setText(
                "no resident engine — select Manual mode on the Load tab and "
                "this config will be used at load.")
            return
        cfg = self._collect_config()
        errors = validate(cfg)
        if errors:
            QMessageBox.warning(self, "Invalid activation config",
                                "\n".join(errors))
            return
        self._apply_live_btn.setEnabled(False)
        self.runtime.reactivate(cfg)

    def _on_reactivated(self, active: dict) -> None:
        self._apply_live_btn.setEnabled(True)
        self._apply_status.setText("features applied ✓")
        self._fill_form({f.name: active.get(f.name, f.default) for f in FIELDS})
        self._mark_matching_preset(active)
        self._update_diff()
        self._poll_stats(); self._update_vram()

    # ── stats / maintenance ───────────────────────────────────────────
    def _clear_stats(self) -> None:
        for val in self._stat_rows.values():
            val.setText("—")
        self._vram_bar.setValue(0)
        self._vram_lbl.setText("—")

    def _poll_stats(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        try:
            s = eng.stats()
            vram = s.get("vram") or {}
            self._stat_rows["Resident"].setText(
                f"{self.runtime.info.get('config_name', '?')} · "
                f"{self.runtime.info.get('dtype', '?')}")
            self._stat_rows["Generations"].setText(
                f"{s.get('generation_count', 0)} runs · "
                f"{s.get('total_tokens_generated', 0)} tokens")
            self._stat_rows["Decoding"].setText(str(s.get("decoding", "?")))
            self._stat_rows["Quantization"].setText(
                str(s.get("quantization") or "bf16 (none)"))
            self._stat_rows["KV cache"].setText(_short(s.get("kv_cache")))
            self._stat_rows["Acceleration"].setText(
                str(s.get("acceleration") or "none"))
            lora = eng.lora_info()
            self._stat_rows["LoRA"].setText(
                f"{_base_name(lora['path'])} · rank {lora['rank']} · "
                f"{lora.get('n_params', 0)/1e6:.1f}M params"
                if lora else "none loaded")
            if vram:
                self._stat_rows["VRAM"].setText(
                    f"{vram.get('used_gb', 0):.2f} / {vram.get('total_gb', 0):.2f} GB"
                    f" ({vram.get('percent', 0):.0f}%) · weights "
                    f"{vram.get('model_weights_gb', 0):.2f} GB")
            awake = True
            try:
                awake = eng.is_awake
            except Exception:
                pass
            self._stat_rows["Power"].setText("awake" if awake else "sleeping")
        except Exception as e:
            logger.warning("stats poll failed: %s", e)

    def _update_vram(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._vram_bar.setValue(0)
            self._vram_lbl.setText("no engine — GPU free for training/launches")
            return
        try:
            v = eng.vram_usage()
            pct = max(0.0, min(100.0, float(v.get("percent", 0))))
            self._vram_bar.setValue(int(pct))
            self._vram_bar.setFormat(f"{pct:.0f}%")
            self._vram_lbl.setText(
                f"{v.get('used_gb', 0):.2f} GB used · {v.get('free_gb', 0):.2f} GB"
                f" free · weights {v.get('model_weights_gb', 0):.2f} GB")
        except Exception as e:
            self._vram_lbl.setText(f"vram read failed: {e}")

    def _maintain(self, action: str) -> None:
        if self._maint is not None and self._maint.isRunning():
            return
        if not self.runtime.is_ready():
            self._maint_out.setPlainText("load an engine first")
            return
        self._maint_out.setPlainText(f"running {action}…")
        self._maint = _MaintWorker(
            self.runtime, action, prompt=self._bench_prompt.text().strip()
            or "The quick brown fox",
            max_new_tokens=self._bench_tok.value(),
            n_runs=self._bench_runs.value(), parent=self)
        self._maint.output.connect(self._maint_out.setPlainText)
        self._maint.done.connect(lambda _a: self._poll_stats())
        self._maint.start()


def _flag_check(f: FieldSpec) -> QCheckBox:
    cb = QCheckBox(f.label)
    cb.setToolTip(f.tooltip)
    cb.setCursor(Qt.CursorShape.PointingHandCursor)
    return cb


def _base_name(p: str) -> str:
    return str(p).replace("\\", "/").rsplit("/", 1)[-1]


def _short(v: Any, limit: int = 300) -> str:
    s = str(v)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(inner)
    sa.setObjectName("root")
    return sa
