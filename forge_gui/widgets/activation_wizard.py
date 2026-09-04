"""One-Click Quantize & Activation Preset Wizard (R36-5).

A 3-step dialog that walks the user from a high-level *use-case* (Chat,
Coding, Agent, Long-Context, Max-Speed) to a fully-resolved
``ActivationConfig``-compatible kwargs dict that the caller can pass
straight to ``engine.activate(**kwargs)`` / ``runtime.reactivate(cfg)``.

The use-case → activation mapping lives in :func:`use_case_config` which
is **pure python** (no Qt imports) so it can be unit-tested on any
machine without a running ``QApplication``.

Wizard steps:
  1. Select use-case (radio buttons)
  2. Auto-selected settings (read-only summary of the resolved config)
  3. Final summary + "Apply" button

On completion (``dialog.exec() == QDialog.Accepted``) the resolved kwargs
dict is available via ``dialog.result_kwargs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from PySide6.QtWidgets import (QButtonGroup, QDialog, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton,
                               QStackedWidget, QVBoxLayout, QWidget)


def _section_label(text: str) -> QLabel:
    """Small section header (matches pages._base.section_label styling)."""
    lbl = QLabel(text.upper())
    lbl.setObjectName("sectionHeader")
    return lbl

# ── use-case → activation mapping (pure python, no Qt) ─────────────────


@dataclass(frozen=True)
class UseCase:
    """A selectable use-case preset."""
    key: str
    label: str
    description: str
    config: dict[str, Any]


# The auto-selection logic from the R36-5 spec.
USE_CASES: tuple[UseCase, ...] = (
    UseCase(
        "chat", "Chat",
        "Balanced conversational quality — 8-bit weights/activations, "
        "SnapKV compression, standard decode.",
        {"quantize": "w8a8", "kv_cache": "snapkv", "decoding": "standard",
         "warmup": True},
    ),
    UseCase(
        "coding", "Coding",
        "Code generation — INT4 weights, S4R 4-bit KV, speculative "
        "decoding for repetitive token patterns.",
        {"quantize": "int4", "kv_cache": "s4r", "decoding": "speculative",
         "warmup": True},
    ),
    UseCase(
        "agent", "Agent",
        "Tool-calling agent — 8-bit weights, dense KV (reliable context), "
        "torch.compile for fast step loops.",
        {"quantize": "w8a8", "kv_cache": "standard", "decoding": "standard",
         "use_compile": True, "warmup": True},
    ),
    UseCase(
        "long_context", "Long-Context",
        "Long-context reading — NVFP4 weights, CPU-offloaded KV (32GB "
        "system RAM), standard decode.",
        {"quantize": "nvfp4", "kv_cache": "cpu_offload",
         "decoding": "standard", "warmup": True},
    ),
    UseCase(
        "max_speed", "Max-Speed",
        "Maximum throughput — FP8 weights, RotorQuant 4-bit KV, Medusa "
        "multi-head decode, torch.compile.",
        {"quantize": "fp8", "kv_cache": "rotorquant", "decoding": "medusa",
         "use_compile": True, "warmup": True},
    ),
)

USE_CASE_BY_KEY: dict[str, UseCase] = {u.key: u for u in USE_CASES}


def use_case_config(key: str) -> dict[str, Any]:
    """Return the activation kwargs dict for a use-case key.

    Raises ``KeyError`` for an unknown use-case. The returned dict is a
    fresh copy so callers may mutate it freely.
    """
    uc = USE_CASE_BY_KEY[key]
    return dict(uc.config)


def use_case_labels() -> list[tuple[str, str, str]]:
    """``[(key, label, description), …]`` for building UIs without Qt."""
    return [(u.key, u.label, u.description) for u in USE_CASES]


# ── pretty-printing helper (pure python) ───────────────────────────────


def format_config_summary(cfg: dict[str, Any]) -> str:
    """Render an activation kwargs dict as a readable multi-line summary."""
    order = ["quantize", "kv_cache", "decoding", "use_compile", "warmup"]
    lines = []
    for k in order:
        if k in cfg:
            v = cfg[k]
            if v is None:
                v = "none"
            lines.append(f"  {k:<14} {v}")
    # any remaining keys not in the canonical order
    for k, v in cfg.items():
        if k in order:
            continue
        if v is None:
            v = "none"
        lines.append(f"  {k:<14} {v}")
    return "\n".join(lines)


# ── wizard dialog ──────────────────────────────────────────────────────


class ActivationWizard(QDialog):
    """3-step one-click quantize & activation preset wizard.

    Usage::

        wiz = ActivationWizard(parent)
        if wiz.exec() == QDialog.DialogCode.Accepted:
            runtime.reactivate(wiz.result_kwargs)
    """

    def __init__(self, parent: Optional[QWidget] = None,
                 default_key: str = "chat") -> None:
        super().__init__(parent)
        self.setWindowTitle("Activation Wizard")
        self.setMinimumWidth(480)
        self.result_kwargs: dict[str, Any] = {}

        self._selected_key: str = default_key

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_step1())
        self._stack.addWidget(self._build_step2())
        self._stack.addWidget(self._build_step3())
        outer.addWidget(self._stack, 1)

        # nav bar
        nav = QFrame()
        nav.setObjectName("cardAlt")
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(16, 10, 16, 10)
        nl.setSpacing(10)
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn = QPushButton("Next")
        self._next_btn.setObjectName("primary")
        self._next_btn.clicked.connect(self._go_next)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("primary")
        self._apply_btn.clicked.connect(self._apply)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        nl.addWidget(self._cancel_btn)
        nl.addStretch(1)
        nl.addWidget(self._back_btn)
        nl.addWidget(self._next_btn)
        nl.addWidget(self._apply_btn)
        outer.addWidget(nav)

        self._update_nav()
        self._refresh_step2()

    # ── step 1: use-case selection ──────────────────────────────────
    def _build_step1(self) -> QWidget:
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18)
        col.setSpacing(14)
        col.addWidget(_section_label("Step 1 · Select use-case"))
        intro = QLabel("Choose how you plan to use the model. The wizard "
                       "will auto-select the best quantization, KV cache "
                       "and decoding strategy for that workload.")
        intro.setObjectName("cardBody")
        intro.setWordWrap(True)
        col.addWidget(intro)

        self._use_group = QButtonGroup(self)
        self._use_group.setExclusive(True)
        self._use_radios: dict[str, QRadioButton] = {}
        for uc in USE_CASES:
            card = QFrame()
            card.setObjectName("card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 10, 14, 10)
            cl.setSpacing(4)
            rb = QRadioButton(uc.label)
            rb.setChecked(uc.key == self._selected_key)
            rb.toggled.connect(lambda on, k=uc.key: self._on_use_changed(k, on))
            self._use_group.addButton(rb)
            self._use_radios[uc.key] = rb
            cl.addWidget(rb)
            desc = QLabel(uc.description)
            desc.setObjectName("chatMeta")
            desc.setWordWrap(True)
            cl.addWidget(desc)
            col.addWidget(card)
        col.addStretch(1)
        return host

    # ── step 2: auto-selected settings ──────────────────────────────
    def _build_step2(self) -> QWidget:
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18)
        col.setSpacing(14)
        col.addWidget(_section_label("Step 2 · Auto-selected settings"))
        self._step2_use_lbl = QLabel("")
        self._step2_use_lbl.setObjectName("kvVal")
        col.addWidget(self._step2_use_lbl)
        self._step2_cfg_lbl = QLabel("")
        self._step2_cfg_lbl.setObjectName("mono")
        self._step2_cfg_lbl.setWordWrap(True)
        col.addWidget(self._step2_cfg_lbl)
        col.addStretch(1)
        return host

    # ── step 3: summary + apply ─────────────────────────────────────
    def _build_step3(self) -> QWidget:
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(22, 18, 22, 18)
        col.setSpacing(14)
        col.addWidget(_section_label("Step 3 · Summary"))
        self._step3_use_lbl = QLabel("")
        self._step3_use_lbl.setObjectName("kvVal")
        col.addWidget(self._step3_use_lbl)
        self._step3_cfg_lbl = QLabel("")
        self._step3_cfg_lbl.setObjectName("mono")
        self._step3_cfg_lbl.setWordWrap(True)
        col.addWidget(self._step3_cfg_lbl)
        note = QLabel("Click Apply to send these settings to the resident "
                      "engine (live re-activation, no reload).")
        note.setObjectName("chatMeta")
        note.setWordWrap(True)
        col.addWidget(note)
        col.addStretch(1)
        return host

    # ── navigation ──────────────────────────────────────────────────
    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        self._back_btn.setVisible(idx > 0)
        self._next_btn.setVisible(idx < 2)
        self._apply_btn.setVisible(idx == 2)
        self._next_btn.setEnabled(idx == 0 or True)

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx < 2:
            self._refresh_step2()
            self._stack.setCurrentIndex(idx + 1)
            self._update_nav()

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_nav()

    # ── selection / refresh ─────────────────────────────────────────
    def _on_use_changed(self, key: str, on: bool) -> None:
        if on:
            self._selected_key = key
            self._refresh_step2()

    def _refresh_step2(self) -> None:
        uc = USE_CASE_BY_KEY.get(self._selected_key)
        if uc is None:
            return
        cfg = use_case_config(self._selected_key)
        summary = format_config_summary(cfg)
        header = f"Use-case: {uc.label}"
        self._step2_use_lbl.setText(header)
        self._step2_cfg_lbl.setText(summary)
        self._step3_use_lbl.setText(header)
        self._step3_cfg_lbl.setText(summary)

    # ── apply ───────────────────────────────────────────────────────
    def _apply(self) -> None:
        self.result_kwargs = use_case_config(self._selected_key)
        self.accept()
