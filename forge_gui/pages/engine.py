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
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QHeaderView,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QProgressBar,
                               QRadioButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)

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
                elif self.action == "read_output":
                    outs = engine.read_output(n=10)
                    if not outs:
                        self.output.emit("(no recent generation outputs)")
                    else:
                        lines = []
                        for o in outs:
                            ts = o.get("timestamp", o.get("time", "?"))
                            tps = o.get("tokens_per_sec", o.get("tok_s", "?"))
                            toks = o.get("tokens", o.get("n_tokens", "?"))
                            preview = str(o.get("text", o.get("output", "")))[:120]
                            lines.append(f"[{ts}] {toks} tok · {tps} tok/s: {preview}")
                        self.output.emit("\n".join(lines))
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


class _GenericWorker(QThread):
    """Runs an arbitrary callable on the resident engine off-UI.

    Used for long-running operations (merge, evolve, library_optimize,
    generate_batch, generate_adaptive) that would freeze the UI if run
    synchronously. The callable receives the engine and returns a result
    that is emitted via ``finished_ok``.
    """

    output = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, runtime: EngineRuntime, fn, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.fn = fn

    def run(self) -> None:
        try:
            with self.runtime.acquire(timeout_s=300.0) as engine:
                self.output.emit("running…")
                result = self.fn(engine)
            self.finished_ok.emit(result)
        except Exception as e:
            logger.warning("generic worker failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self.done.emit()


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
        self._generic: Optional[_GenericWorker] = None
        self._preset_group: dict[str, QPushButton] = {}
        self._combo_specs: dict[str, tuple[QComboBox, QLabel]] = {}
        self._spin_fields: dict[str, QWidget] = {}
        self._flag_checks: dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_load_tab(), "Load & Power")
        self._tabs.addTab(self._build_activation_tab(), "Activation")
        self._tabs.addTab(self._build_tools_tab(), "Stats & Tools")
        self._tabs.addTab(self._build_library_tab(), "Library")
        self._tabs.addTab(self._build_sessions_tab(), "Sessions")
        self._tabs.addTab(self._build_merge_tab(), "Merge Studio")
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
        # Defer checkpoint/config scans to after window show — these trigger
        # models_index.models()/configs() which scan research/checkpoints/.
        QTimer.singleShot(0, self._reload_checkpoints)
        ml.addWidget(self._ckpt)
        cfg_row = QHBoxLayout(); cfg_row.setSpacing(10)
        cfg_row.addWidget(QLabel("Config"))
        self._config = QComboBox()
        self._config.setEditable(True)
        QTimer.singleShot(0, self._reload_configs)
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
        self._apply_optimal_btn = QPushButton("★ Apply optimal preset")
        self._apply_optimal_btn.setToolTip(
            "Activate the engine's built-in optimal preset (rotorquant KV, "
            "Triton conv, prefix cache, fused QK-Norm+RoPE, chunked prefill, "
            "seq split) — overrides the form with the tuned defaults.")
        self._apply_optimal_btn.clicked.connect(self._apply_optimal)
        row.addWidget(self._apply_optimal_btn)
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

    # ── tab 4: library ───────────────────────────────────────────────
    def _build_library_tab(self) -> QWidget:
        host = QWidget(); col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18); col.setSpacing(14)

        # ── config card: enable + budget ─────────────────────────────
        cfg_card = QFrame(); cfg_card.setObjectName("card")
        cl = QVBoxLayout(cfg_card); cl.setContentsMargins(16, 14, 16, 16)
        cl.setSpacing(10)
        cl.addWidget(section_label("Library configuration"))
        row = QHBoxLayout(); row.setSpacing(10)
        self._lib_enabled = QCheckBox("Lorebook injection enabled")
        self._lib_enabled.setToolTip("Globally enable/disable library lorebook "
                                     "injection into prompts.")
        self._lib_enabled.toggled.connect(self._lib_set_enabled)
        row.addWidget(self._lib_enabled)
        row.addWidget(QLabel("Injection budget"))
        self._lib_budget = QSpinBox(); self._lib_budget.setRange(0, 32768)
        self._lib_budget.setSuffix(" tok")
        self._lib_budget.setToolTip("Max tokens injected per request.")
        row.addWidget(self._lib_budget)
        self._lib_budget_btn = QPushButton("Set")
        self._lib_budget_btn.clicked.connect(self._lib_set_budget)
        row.addWidget(self._lib_budget_btn)
        row.addStretch(1)
        cl.addLayout(row)
        col.addWidget(cfg_card)

        # ── save card ────────────────────────────────────────────────
        save_card = QFrame(); save_card.setObjectName("card")
        sl = QVBoxLayout(save_card); sl.setContentsMargins(16, 14, 16, 16)
        sl.setSpacing(10)
        sl.addWidget(section_label("Save entry"))
        self._lib_content = QTextEdit()
        self._lib_content.setPlaceholderText("Content to store in the library…")
        self._lib_content.setFixedHeight(80)
        sl.addWidget(self._lib_content)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Category"))
        self._lib_category = QComboBox()
        for cat in ("custom", "failure", "win", "research", "common_data"):
            self._lib_category.addItem(cat, cat)
        row.addWidget(self._lib_category)
        row.addWidget(QLabel("Priority"))
        self._lib_priority = QSpinBox(); self._lib_priority.setRange(0, 100)
        row.addWidget(self._lib_priority)
        row.addStretch(1)
        sl.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Tags"))
        self._lib_tags = QLineEdit(); self._lib_tags.setPlaceholderText("comma-separated")
        row.addWidget(self._lib_tags, 1)
        sl.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Description"))
        self._lib_desc = QLineEdit()
        row.addWidget(self._lib_desc, 1)
        sl.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Triggers"))
        self._lib_triggers = QLineEdit(); self._lib_triggers.setPlaceholderText("comma-separated keywords")
        row.addWidget(self._lib_triggers, 1)
        sl.addLayout(row)
        self._lib_save_btn = QPushButton("⏏ Save entry")
        self._lib_save_btn.setObjectName("primary")
        self._lib_save_btn.clicked.connect(self._lib_save)
        sl.addWidget(self._lib_save_btn)
        self._lib_save_status = QLabel("")
        self._lib_save_status.setObjectName("chatMeta")
        sl.addWidget(self._lib_save_status)
        col.addWidget(save_card)

        # ── search + lookup card ─────────────────────────────────────
        search_card = QFrame(); search_card.setObjectName("card")
        shl = QVBoxLayout(search_card); shl.setContentsMargins(16, 14, 16, 16)
        shl.setSpacing(10)
        shl.addWidget(section_label("Search & lookup"))
        row = QHBoxLayout(); row.setSpacing(10)
        self._lib_search = QLineEdit(); self._lib_search.setPlaceholderText("full-text query…")
        row.addWidget(self._lib_search, 1)
        self._lib_search_btn = QPushButton("Search")
        self._lib_search_btn.clicked.connect(self._lib_do_search)
        row.addWidget(self._lib_search_btn)
        shl.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Lookup tags"))
        self._lib_lookup_tags = QLineEdit(); self._lib_lookup_tags.setPlaceholderText("comma-separated tags")
        row.addWidget(self._lib_lookup_tags, 1)
        row.addWidget(QLabel("Category"))
        self._lib_lookup_cat = QComboBox()
        self._lib_lookup_cat.addItem("(any)", "")
        for cat in ("custom", "failure", "win", "research", "common_data"):
            self._lib_lookup_cat.addItem(cat, cat)
        row.addWidget(self._lib_lookup_cat)
        self._lib_lookup_btn = QPushButton("Lookup")
        self._lib_lookup_btn.clicked.connect(self._lib_do_lookup)
        row.addWidget(self._lib_lookup_btn)
        shl.addLayout(row)
        col.addWidget(search_card)

        # ── entries table ────────────────────────────────────────────
        list_card = QFrame(); list_card.setObjectName("card")
        ll = QVBoxLayout(list_card); ll.setContentsMargins(16, 14, 16, 16)
        ll.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(section_label("Entries"))
        head.addStretch(1)
        self._lib_refresh_btn = QPushButton("Refresh")
        self._lib_refresh_btn.clicked.connect(self._lib_refresh_list)
        head.addWidget(self._lib_refresh_btn)
        self._lib_optimize_btn = QPushButton("Optimize")
        self._lib_optimize_btn.clicked.connect(self._lib_optimize)
        head.addWidget(self._lib_optimize_btn)
        ll.addLayout(head)
        self._lib_table = QTableWidget(0, 5)
        self._lib_table.setHorizontalHeaderLabels(
            ["ID", "Category", "Description", "Tags", "Tokens"])
        hh = self._lib_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._lib_table.verticalHeader().setVisible(False)
        self._lib_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._lib_table.setAlternatingRowColors(True)
        self._lib_table.setObjectName("dataTable")
        ll.addWidget(self._lib_table)
        col.addWidget(list_card)

        # ── stats card ───────────────────────────────────────────────
        stats_card = QFrame(); stats_card.setObjectName("card")
        stl = QVBoxLayout(stats_card); stl.setContentsMargins(16, 14, 16, 16)
        stl.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(section_label("Library stats"))
        head.addStretch(1)
        self._lib_stats_btn = QPushButton("Refresh")
        self._lib_stats_btn.clicked.connect(self._lib_refresh_stats)
        head.addWidget(self._lib_stats_btn)
        stl.addLayout(head)
        self._lib_stats_lbl = QLabel("—")
        self._lib_stats_lbl.setObjectName("kvVal")
        self._lib_stats_lbl.setWordWrap(True)
        stl.addWidget(self._lib_stats_lbl)
        col.addWidget(stats_card)
        col.addStretch(1)
        return _scroll(host)

    def _lib_set_enabled(self, on: bool) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        try:
            eng.library_set_enabled(on)
        except Exception as e:
            self._lib_save_status.setText(f"toggle failed: {e}")

    def _lib_set_budget(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._lib_save_status.setText("no engine")
            return
        try:
            eng.library_set_budget(self._lib_budget.value())
            self._lib_save_status.setText(f"budget set to {self._lib_budget.value()} tok")
        except Exception as e:
            self._lib_save_status.setText(f"budget failed: {e}")

    def _lib_save(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._lib_save_status.setText("no engine — load a model first")
            return
        content = self._lib_content.toPlainText().strip()
        if not content:
            self._lib_save_status.setText("content is empty")
            return
        tags = [t.strip() for t in self._lib_tags.text().split(",") if t.strip()]
        triggers = [t.strip() for t in self._lib_triggers.text().split(",") if t.strip()]
        try:
            eid = eng.library_save(
                content=content, category=self._lib_category.currentData(),
                tags=tags, description=self._lib_desc.text().strip(),
                triggers=triggers, priority=self._lib_priority.value())
            self._lib_save_status.setText(f"saved ✓ entry_id={eid}")
            self._lib_content.clear(); self._lib_tags.clear()
            self._lib_desc.clear(); self._lib_triggers.clear()
            self._lib_refresh_list()
        except Exception as e:
            self._lib_save_status.setText(f"save failed: {e}")

    def _lib_do_search(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        q = self._lib_search.text().strip()
        if not q:
            return
        try:
            results = eng.library_search(q, limit=50)
            self._lib_fill_table(results)
        except Exception as e:
            self._lib_save_status.setText(f"search failed: {e}")

    def _lib_do_lookup(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        tags = [t.strip() for t in self._lib_lookup_tags.text().split(",") if t.strip()]
        cat = self._lib_lookup_cat.currentData() or None
        kwargs = {}
        if tags:
            kwargs["tags"] = tags
        if cat:
            kwargs["category"] = cat
        try:
            results = eng.library_lookup(**kwargs)
            self._lib_fill_table(results)
        except Exception as e:
            self._lib_save_status.setText(f"lookup failed: {e}")

    def _lib_fill_table(self, entries: list) -> None:
        t = self._lib_table
        t.setRowCount(len(entries))
        for i, e in enumerate(entries):
            d = e.to_dict() if hasattr(e, "to_dict") else e
            cells = [str(d.get("id", "?")), str(d.get("category", "?")),
                     str(d.get("description", ""))[:80],
                     ", ".join(d.get("tags", [])),
                     str(d.get("token_count", 0))]
            for j, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if j in (0, 1, 4):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignVCenter
                                        | Qt.AlignmentFlag.AlignRight)
                t.setItem(i, j, it)

    def _lib_refresh_list(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        try:
            entries = eng.library.list_entries(limit=200)
            self._lib_fill_table(entries)
        except Exception as e:
            self._lib_save_status.setText(f"refresh failed: {e}")

    def _lib_optimize(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._lib_save_status.setText("no engine")
            return
        self._lib_save_status.setText("optimizing…")
        def _fn(e):
            return e.library_optimize()
        self._run_generic(_fn, self._lib_on_optimize)

    def _lib_on_optimize(self, result) -> None:
        self._lib_save_status.setText(f"optimized ✓ {_fmt_report(result)}")
        self._lib_refresh_list(); self._lib_refresh_stats()

    def _lib_refresh_stats(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._lib_stats_lbl.setText("—")
            return
        try:
            s = eng.library_stats()
            parts = [f"{s.get('total_entries', 0)} entries",
                     f"{s.get('total_tokens', 0)} tokens",
                     f"budget {s.get('injection_budget', 0)} tok"]
            by_cat = s.get("by_category", {})
            if by_cat:
                parts.append(" · ".join(f"{k}:{v}" for k, v in by_cat.items()))
            self._lib_stats_lbl.setText("  ·  ".join(parts))
        except Exception as e:
            self._lib_stats_lbl.setText(f"stats failed: {e}")

    def _run_generic(self, fn, on_ok=None) -> None:
        """Run a long callable on the resident engine off-UI."""
        if not self.runtime.is_ready():
            return
        if self._generic is not None and self._generic.isRunning():
            return
        self._generic = _GenericWorker(self.runtime, fn, parent=self)
        if on_ok is not None:
            self._generic.finished_ok.connect(on_ok)
        self._generic.failed.connect(
            lambda e: self._maint_out.setPlainText(f"error: {e}"))
        self._generic.start()

    # ── tab 5: sessions ──────────────────────────────────────────────
    def _build_sessions_tab(self) -> QWidget:
        host = QWidget(); col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18); col.setSpacing(14)

        # ── begin session ────────────────────────────────────────────
        begin_card = QFrame(); begin_card.setObjectName("card")
        bl = QVBoxLayout(begin_card); bl.setContentsMargins(16, 14, 16, 16)
        bl.setSpacing(10)
        bl.addWidget(section_label("Begin session"))
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Session ID"))
        self._sess_id = QLineEdit(); self._sess_id.setPlaceholderText("unique session name")
        row.addWidget(self._sess_id, 1)
        row.addWidget(QLabel("TTL (s)"))
        self._sess_ttl = QSpinBox(); self._sess_ttl.setRange(0, 86400)
        self._sess_ttl.setSpecialValueText("no TTL")
        row.addWidget(self._sess_ttl)
        self._sess_begin_btn = QPushButton("⏏ Begin")
        self._sess_begin_btn.setObjectName("primary")
        self._sess_begin_btn.clicked.connect(self._sess_begin)
        row.addWidget(self._sess_begin_btn)
        bl.addLayout(row)
        col.addWidget(begin_card)

        # ── continue session ─────────────────────────────────────────
        cont_card = QFrame(); cont_card.setObjectName("card")
        ctl = QVBoxLayout(cont_card); ctl.setContentsMargins(16, 14, 16, 16)
        ctl.setSpacing(10)
        ctl.addWidget(section_label("Continue session"))
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Session"))
        self._sess_pick = QComboBox()
        row.addWidget(self._sess_pick, 1)
        self._sess_refresh_btn = QPushButton("↻")
        self._sess_refresh_btn.setFixedWidth(32)
        self._sess_refresh_btn.clicked.connect(self._sess_refresh_pick)
        row.addWidget(self._sess_refresh_btn)
        ctl.addLayout(row)
        self._sess_prompt = QPlainTextEdit()
        self._sess_prompt.setPlaceholderText("Prompt to continue the session with…")
        self._sess_prompt.setFixedHeight(70)
        ctl.addWidget(self._sess_prompt)
        row = QHBoxLayout(); row.setSpacing(10)
        for label, w in [("Max tok", self._spin_widget("sess_max", 128, 1, 4096)),
                         ("Temp", self._dspin_widget("sess_temp", 0.0, 0.0, 2.0, 0.05)),
                         ("Top-P", self._dspin_widget("sess_topp", 1.0, 0.0, 1.0, 0.01))]:
            row.addWidget(QLabel(label)); row.addWidget(w)
        row.addStretch(1)
        ctl.addLayout(row)
        self._sess_continue_btn = QPushButton("Generate")
        self._sess_continue_btn.clicked.connect(self._sess_continue)
        ctl.addWidget(self._sess_continue_btn)
        self._sess_output = QPlainTextEdit()
        self._sess_output.setObjectName("logView")
        self._sess_output.setReadOnly(True)
        self._sess_output.setFixedHeight(120)
        ctl.addWidget(self._sess_output)
        col.addWidget(cont_card)

        # ── pin / unpin / end ────────────────────────────────────────
        mgmt_card = QFrame(); mgmt_card.setObjectName("card")
        ml = QVBoxLayout(mgmt_card); ml.setContentsMargins(16, 14, 16, 16)
        ml.setSpacing(10)
        ml.addWidget(section_label("Pin / unpin / end"))
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Session"))
        self._sess_mgmt_pick = QComboBox()
        row.addWidget(self._sess_mgmt_pick, 1)
        row.addWidget(QLabel("Pin TTL (s)"))
        self._sess_pin_ttl = QSpinBox(); self._sess_pin_ttl.setRange(0, 86400)
        self._sess_pin_ttl.setSpecialValueText("no TTL")
        row.addWidget(self._sess_pin_ttl)
        self._sess_pin_btn = QPushButton("Pin")
        self._sess_pin_btn.clicked.connect(self._sess_pin)
        self._sess_unpin_btn = QPushButton("Unpin")
        self._sess_unpin_btn.clicked.connect(self._sess_unpin)
        self._sess_end_btn = QPushButton("End")
        self._sess_end_btn.setObjectName("danger")
        self._sess_end_btn.clicked.connect(self._sess_end)
        for b in (self._sess_pin_btn, self._sess_unpin_btn, self._sess_end_btn):
            row.addWidget(b)
        ml.addLayout(row)
        col.addWidget(mgmt_card)

        # ── session stats ────────────────────────────────────────────
        stats_card = QFrame(); stats_card.setObjectName("card")
        ssl = QVBoxLayout(stats_card); ssl.setContentsMargins(16, 14, 16, 16)
        ssl.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(section_label("Session stats"))
        head.addStretch(1)
        self._sess_stats_btn = QPushButton("Refresh")
        self._sess_stats_btn.clicked.connect(self._sess_refresh_stats)
        head.addWidget(self._sess_stats_btn)
        ssl.addLayout(head)
        self._sess_stats_lbl = QLabel("—")
        self._sess_stats_lbl.setObjectName("kvVal")
        self._sess_stats_lbl.setWordWrap(True)
        ssl.addWidget(self._sess_stats_lbl)
        col.addWidget(stats_card)
        col.addStretch(1)
        return _scroll(host)

    def _spin_widget(self, attr: str, val, lo, hi, step=1) -> QSpinBox:
        s = QSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setValue(val)
        setattr(self, f"_{attr}", s)
        return s

    def _dspin_widget(self, attr: str, val, lo, hi, step, dec=2) -> QDoubleSpinBox:
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setDecimals(dec); s.setValue(val)
        setattr(self, f"_{attr}", s)
        return s

    def _sess_begin(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._sess_output.setPlainText("no engine")
            return
        sid = self._sess_id.text().strip()
        if not sid:
            self._sess_output.setPlainText("enter a session ID")
            return
        ttl = self._sess_ttl.value() or None
        try:
            eng.begin_session(sid, ttl=ttl)
            self._sess_output.setPlainText(f"session '{sid}' begun ✓")
            self._sess_refresh_pick()
        except Exception as e:
            self._sess_output.setPlainText(f"begin failed: {e}")

    def _sess_continue(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._sess_output.setPlainText("no engine")
            return
        sid = self._sess_pick.currentText().strip()
        if not sid:
            self._sess_output.setPlainText("select a session first")
            return
        prompt = self._sess_prompt.toPlainText().strip()
        if not prompt:
            self._sess_output.setPlainText("enter a prompt")
            return
        self._sess_output.setPlainText("generating…")
        def _fn(e):
            return e.continue_session(
                sid, prompt, max_new_tokens=self._sess_max.value(),
                temperature=self._sess_temp.value(),
                top_p=self._sess_topp.value())
        self._run_generic(
            _fn,
            on_ok=lambda text: (self._sess_output.setPlainText(str(text)),
                                 self._sess_prompt.clear()))

    def _sess_pin(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        sid = self._sess_mgmt_pick.currentText().strip()
        if not sid:
            return
        ttl = self._sess_pin_ttl.value() or None
        try:
            eng.pin_session(sid, ttl=ttl)
            self._sess_output.setPlainText(f"pinned '{sid}' ✓")
        except Exception as e:
            self._sess_output.setPlainText(f"pin failed: {e}")

    def _sess_unpin(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        sid = self._sess_mgmt_pick.currentText().strip()
        if not sid:
            return
        try:
            eng.unpin_session(sid)
            self._sess_output.setPlainText(f"unpinned '{sid}' ✓")
        except Exception as e:
            self._sess_output.setPlainText(f"unpin failed: {e}")

    def _sess_end(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        sid = self._sess_mgmt_pick.currentText().strip()
        if not sid:
            return
        try:
            eng.end_session(sid)
            self._sess_output.setPlainText(f"ended '{sid}' ✓")
            self._sess_refresh_pick()
        except Exception as e:
            self._sess_output.setPlainText(f"end failed: {e}")

    def _sess_refresh_pick(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            return
        try:
            s = eng.session_stats()
            ids = list(s.get("sessions", {}).keys()) if isinstance(s.get("sessions"), dict) else []
            for combo in (self._sess_pick, self._sess_mgmt_pick):
                cur = combo.currentText()
                combo.clear(); combo.addItems(ids)
                if cur in ids:
                    combo.setCurrentText(cur)
        except Exception:
            pass

    def _sess_refresh_stats(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._sess_stats_lbl.setText("—")
            return
        try:
            s = eng.session_stats()
            self._sess_stats_lbl.setText(_fmt_report(s))
        except Exception as e:
            self._sess_stats_lbl.setText(f"stats failed: {e}")

    # ── tab 6: merge studio ──────────────────────────────────────────
    def _build_merge_tab(self) -> QWidget:
        host = QWidget(); col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18); col.setSpacing(14)

        # ── one-shot merge ───────────────────────────────────────────
        merge_card = QFrame(); merge_card.setObjectName("card")
        ml = QVBoxLayout(merge_card); ml.setContentsMargins(16, 14, 16, 16)
        ml.setSpacing(10)
        ml.addWidget(section_label("Merge checkpoints (one-shot)"))
        ml.addWidget(QLabel("Parents (select 2+ for crossover, 1+ for mutation):"))
        self._merge_parents = QTableWidget(0, 2)
        self._merge_parents.setHorizontalHeaderLabels(["", "Checkpoint"])
        hh = self._merge_parents.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._merge_parents.verticalHeader().setVisible(False)
        self._merge_parents.setObjectName("dataTable")
        QTimer.singleShot(0, self._merge_fill_parents)
        ml.addWidget(self._merge_parents)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Method"))
        self._merge_method = QComboBox()
        for m in ("blockwise_crossover", "block_random_crossover",
                  "uniform_crossover", "gaussian_mutation",
                  "quant_perturb", "block_swap",
                  "slerp", "linear", "ties", "dare", "svd", "task_arith"):
            self._merge_method.addItem(m, m)
        row.addWidget(self._merge_method, 1)
        ml.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Output"))
        self._merge_out = QLineEdit()
        self._merge_out.setPlaceholderText("data/merged/<method>_<ts>.safetensors")
        row.addWidget(self._merge_out, 1)
        self._merge_pick_btn = QPushButton("…")
        self._merge_pick_btn.setFixedWidth(32)
        self._merge_pick_btn.clicked.connect(self._merge_pick_out)
        row.addWidget(self._merge_pick_btn)
        ml.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        self._merge_load_result = QCheckBox("Hot-swap merged weights into engine after save")
        row.addWidget(self._merge_load_result)
        row.addStretch(1)
        ml.addLayout(row)
        self._merge_btn = QPushButton("⚒ Merge (CPU — GPU untouched)")
        self._merge_btn.setObjectName("primary")
        self._merge_btn.clicked.connect(self._merge_run)
        ml.addWidget(self._merge_btn)
        col.addWidget(merge_card)

        # ── evolutionary merge ───────────────────────────────────────
        evolve_card = QFrame(); evolve_card.setObjectName("card")
        el = QVBoxLayout(evolve_card); el.setContentsMargins(16, 14, 16, 16)
        el.setSpacing(10)
        el.addWidget(section_label("Evolutionary merge (GENOME)"))
        row = QHBoxLayout(); row.setSpacing(10)
        for label, w in [("Generations", self._spin_widget("evo_gen", 5, 1, 50)),
                         ("Pop size", self._spin_widget("evo_pop", 8, 2, 32)),
                         ("Elitism", self._spin_widget("evo_elite", 1, 0, 10))]:
            row.addWidget(QLabel(label)); row.addWidget(w)
        row.addStretch(1)
        el.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Crossover"))
        self._evo_crossover = QComboBox()
        for m in ("blockwise", "block_random", "uniform"):
            self._evo_crossover.addItem(m, m)
        row.addWidget(self._evo_crossover)
        row.addWidget(QLabel("Mutation"))
        self._evo_mutation = QComboBox()
        for m in ("gaussian", "quant_perturb", "block_swap"):
            self._evo_mutation.addItem(m, m)
        row.addWidget(self._evo_mutation)
        row.addWidget(QLabel("Mut rate"))
        self._evo_mut_rate = QDoubleSpinBox()
        self._evo_mut_rate.setRange(0.0, 1.0); self._evo_mut_rate.setSingleStep(0.1)
        self._evo_mut_rate.setValue(0.5)
        row.addWidget(self._evo_mut_rate)
        row.addStretch(1)
        el.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Bench prompt"))
        self._evo_bench_prompt = QLineEdit("The quick brown fox jumps over the lazy dog.")
        row.addWidget(self._evo_bench_prompt, 1)
        row.addWidget(QLabel("Bench tok"))
        self._evo_bench_tok = QSpinBox(); self._evo_bench_tok.setRange(8, 512)
        self._evo_bench_tok.setValue(32)
        row.addWidget(self._evo_bench_tok)
        el.addLayout(row)
        self._evo_btn = QPushButton("⚒ Evolve (uses engine as fitness evaluator)")
        self._evo_btn.setObjectName("primary")
        self._evo_btn.clicked.connect(self._evolve_run)
        el.addWidget(self._evo_btn)
        col.addWidget(evolve_card)

        # ── output ───────────────────────────────────────────────────
        out_card = QFrame(); out_card.setObjectName("card")
        ol = QVBoxLayout(out_card); ol.setContentsMargins(16, 14, 16, 16)
        ol.setSpacing(10)
        ol.addWidget(section_label("Output"))
        self._merge_output = QPlainTextEdit()
        self._merge_output.setObjectName("logView")
        self._merge_output.setReadOnly(True)
        self._merge_output.setFixedHeight(140)
        ol.addWidget(self._merge_output)
        col.addWidget(out_card)
        col.addStretch(1)
        return _scroll(host)

    def _merge_fill_parents(self) -> None:
        t = self._merge_parents
        paths: list[tuple[str, str]] = []
        if self.models_index is not None:
            try:
                for m in self.models_index.models():
                    if m.is_safetensors and "lora" not in m.name.lower():
                        paths.append((m.name, m.path))
            except Exception:
                pass
        if not paths:
            paths.append(("ForgeLM_V2_Light.safetensors",
                          "research/checkpoints/ForgeLM_V2_Light.safetensors"))
        t.setRowCount(len(paths))
        for i, (name, path) in enumerate(paths):
            cb = QCheckBox()
            cb_widget = QWidget(); cb_l = QHBoxLayout(cb_widget)
            cb_l.setContentsMargins(0, 0, 0, 0); cb_l.addWidget(cb)
            cb_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setCellWidget(i, 0, cb_widget)
            t.setItem(i, 1, QTableWidgetItem(f"{name}  ({path})"))

    def _merge_selected_parents(self) -> list[str]:
        t = self._merge_parents
        root = None
        try:
            from ..api.status_reader import project_root as _pr
            root = _pr()
        except Exception:
            pass
        parents = []
        for i in range(t.rowCount()):
            cw = t.cellWidget(i, 0)
            cb = cw.findChild(QCheckBox) if cw else None
            if cb and cb.isChecked():
                item = t.item(i, 1)
                if item is None:
                    continue
                text = item.text()
                # extract path from "name  (path)"
                if "(" in text and text.endswith(")"):
                    path = text[text.rfind("(") + 1:text.rfind(")")]
                else:
                    path = text
                if root and not _isabs_str(path):
                    path = str(root / path)
                parents.append(path)
        return parents

    def _merge_pick_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Merged checkpoint output", "",
            "Safetensors (*.safetensors)")
        if path:
            self._merge_out.setText(path)

    def _merge_run(self) -> None:
        parents = self._merge_selected_parents()
        if not parents:
            self._merge_output.setPlainText("select at least 1 parent")
            return
        method = self._merge_method.currentData()
        out_path = self._merge_out.text().strip() or None
        load_result = self._merge_load_result.isChecked()
        cfg = self._config.currentText().strip() or None
        self._merge_output.setPlainText(f"merging {len(parents)} parents via {method}…")
        def _fn(e):
            return e.merge_checkpoints(
                parents=parents, method=method, out_path=out_path,
                config_name=cfg, load_result=load_result)
        self._run_generic(_fn, self._merge_on_done)

    def _merge_on_done(self, path) -> None:
        self._merge_output.setPlainText(f"merged ✓ → {path}")
        self._poll_stats()

    def _evolve_run(self) -> None:
        parents = self._merge_selected_parents()
        if len(parents) < 2:
            self._merge_output.setPlainText("evolve needs 2+ parents")
            return
        self._merge_output.setPlainText(
            f"evolving {len(parents)} parents, "
            f"{self._evo_gen.value()} generations…")
        def _fn(e):
            return e.evolve_merge(
                parents=parents,
                n_generations=self._evo_gen.value(),
                population_size=self._evo_pop.value(),
                crossover=self._evo_crossover.currentData(),
                mutation=self._evo_mutation.currentData(),
                mutation_rate=self._evo_mut_rate.value(),
                elitism=self._evo_elite.value(),
                benchmark_prompt=self._evo_bench_prompt.text().strip(),
                benchmark_tokens=self._evo_bench_tok.value())
        self._run_generic(_fn, self._evolve_on_done)

    def _evolve_on_done(self, result) -> None:
        if isinstance(result, dict):
            best = result.get("best_path", "?")
            gens = result.get("generations_completed", "?")
            self._merge_output.setPlainText(
                f"evolution complete ✓ {gens} generations\n"
                f"best checkpoint: {best}")
        else:
            self._merge_output.setPlainText(f"evolution complete ✓ {result}")
        self._poll_stats()

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
                              ("Engine log", "read_log"),
                              ("Recent outputs", "read_output")):
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

        # CacheBlend runtime controls
        blend_card = QFrame(); blend_card.setObjectName("card")
        bl = QVBoxLayout(blend_card); bl.setContentsMargins(16, 14, 16, 16)
        bl.setSpacing(10)
        bl.addWidget(section_label("CacheBlend (non-prefix KV reuse)"))
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Chunk size"))
        self._blend_chunk = QSpinBox(); self._blend_chunk.setRange(32, 4096)
        self._blend_chunk.setValue(256); self._blend_chunk.setSingleStep(32)
        row.addWidget(self._blend_chunk)
        row.addWidget(QLabel("Max chunks"))
        self._blend_max = QSpinBox(); self._blend_max.setRange(1, 4096)
        self._blend_max.setValue(512)
        row.addWidget(self._blend_max)
        self._blend_enable_btn = QPushButton("Enable")
        self._blend_enable_btn.clicked.connect(self._blend_enable)
        row.addWidget(self._blend_enable_btn)
        row.addStretch(1)
        bl.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(QLabel("Register chunk text"))
        self._blend_text = QLineEdit()
        self._blend_text.setPlaceholderText("text to pre-compute KV for reuse…")
        row.addWidget(self._blend_text, 1)
        self._blend_register_btn = QPushButton("Register")
        self._blend_register_btn.clicked.connect(self._blend_register)
        row.addWidget(self._blend_register_btn)
        bl.addLayout(row)
        self._blend_status = QLabel("CacheBlend enables non-prefix KV reuse — "
                                    "pre-compute chunks for instant retrieval.")
        self._blend_status.setObjectName("chatMeta"); self._blend_status.setWordWrap(True)
        bl.addWidget(self._blend_status)
        col.addWidget(blend_card)

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

    def _apply_optimal(self) -> None:
        """Activate the engine's built-in optimal preset on the resident engine."""
        if not self.runtime.is_ready():
            self._apply_status.setText(
                "no resident engine — load a model first")
            return
        self._apply_status.setText("applying optimal preset…")
        def _fn(e):
            e.activate_optimal()
            ac = getattr(e, "active_config", None)
            return ac.to_dict() if ac is not None else {}
        self._run_generic(_fn, self._on_optimal_applied)

    def _on_optimal_applied(self, active: dict) -> None:
        self._apply_status.setText("optimal preset applied ✓")
        self._fill_form({f.name: active.get(f.name, f.default) for f in FIELDS})
        self._mark_matching_preset(active)
        self._update_diff()
        self._poll_stats(); self._update_vram()

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

    def _blend_enable(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._blend_status.setText("no engine")
            return
        try:
            eng.enable_cache_blend(
                chunk_size=self._blend_chunk.value(),
                max_chunks=self._blend_max.value())
            self._blend_status.setText("CacheBlend enabled ✓")
        except Exception as e:
            self._blend_status.setText(f"enable failed: {e}")

    def _blend_register(self) -> None:
        eng = self.runtime.try_engine()
        if eng is None:
            self._blend_status.setText("no engine")
            return
        text = self._blend_text.text().strip()
        if not text:
            return
        def _fn(e):
            return e.register_blend_chunk(text)
        self._run_generic(
            _fn,
            on_ok=lambda n: (self._blend_status.setText(f"registered chunk #{n} ✓"),
                             self._blend_text.clear()))

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


def _isabs_str(p: str) -> bool:
    import os
    return os.path.isabs(p)


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
