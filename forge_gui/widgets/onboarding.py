"""First-run onboarding dialog for ForgeAI.

Appears on first launch (detected via the QSettings key ``onboarded`` being
absent).  Shows a welcome message, a "trust this agent?" checkbox for agent
mode, a quick model-path / download hint, and a "Get started" button that
persists ``onboarded = True`` to ``QSettings("ForgeAI", "ForgeGUI")``.

Usage (from ``app.py`` — NOT modified here, integration is done in parallel)::

    from .widgets.onboarding import maybe_show_onboarding
    maybe_show_onboarding(self)   # self = QMainWindow
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..theme import Palette

# QSettings org / app names — must match app.py (MainWindow uses these).
_SETTINGS_ORG = "ForgeAI"
_SETTINGS_APP = "ForgeGUI"
_ONBOARDED_KEY = "onboarded"
_TRUST_KEY = "agent_trusted"
_MODEL_PATH_KEY = "onboard_model_path"


class OnboardingDialog(QDialog):
    """First-run welcome + quick-setup dialog.

    Signals
    -------
    finished_onboarding(bool trusted, str model_path)
        Emitted when the user clicks "Get started".  ``trusted`` is the
        agent-trust checkbox state; ``model_path`` is the entered path
        (may be empty).
    """

    finished_onboarding = Signal(bool, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to ForgeAI")
        self.setMinimumWidth(480)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)
        self.setStyleSheet(
            f"QDialog {{ background: {Palette.bg}; }}"
            f"QLabel {{ background: transparent; color: {Palette.text}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(14)

        # ── welcome header ───────────────────────────────────────────
        icon_lbl = QLabel("⚒")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(
            f"font-size: 36px; color: {Palette.accent}; background: transparent;")
        root.addWidget(icon_lbl)

        title = QLabel("Welcome to ForgeAI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Palette.text};")
        root.addWidget(title)

        welcome = QLabel(
            "ForgeAI is a local-first LLM studio — train, fine-tune, chat, "
            "and run agentic coding tasks with your own models on your own "
            "GPU.\n\nThis quick setup gets you running in under a minute.")
        welcome.setWordWrap(True)
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet(
            f"font-size: 12px; color: {Palette.text_dim};")
        root.addWidget(welcome)

        # separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {Palette.border}; border: none;")
        root.addWidget(sep)

        # ── trust this agent? ────────────────────────────────────────
        trust_card = QFrame()
        trust_card.setStyleSheet(
            f"QFrame {{ background: {Palette.panel}; "
            f"border: 1px solid {Palette.border}; border-radius: 8px; }}")
        tl = QVBoxLayout(trust_card)
        tl.setContentsMargins(16, 12, 16, 12)
        tl.setSpacing(6)
        trust_head = QLabel("Agent mode")
        trust_head.setStyleSheet(
            f"font-size: 11px; font-weight: 700; letter-spacing: 1px; "
            f"color: {Palette.text_dim};")
        tl.addWidget(trust_head)
        self._trust_cb = QCheckBox(
            "Trust this agent — allow autonomous tool use without per-step approval")
        self._trust_cb.setToolTip(
            "When checked, the Agent page runs tools (file edits, commands) "
            "autonomously.  Uncheck to require approval before each action.")
        self._trust_cb.setChecked(False)
        tl.addWidget(self._trust_cb)
        trust_hint = QLabel(
            "You can change this later on the Agent page.")
        trust_hint.setStyleSheet(
            f"font-size: 11px; color: {Palette.text_faint};")
        tl.addWidget(trust_hint)
        root.addWidget(trust_card)

        # ── quick model setup ────────────────────────────────────────
        model_card = QFrame()
        model_card.setStyleSheet(
            f"QFrame {{ background: {Palette.panel}; "
            f"border: 1px solid {Palette.border}; border-radius: 8px; }}")
        ml = QVBoxLayout(model_card)
        ml.setContentsMargins(16, 12, 16, 12)
        ml.setSpacing(6)
        model_head = QLabel("Quick model setup")
        model_head.setStyleSheet(
            f"font-size: 11px; font-weight: 700; letter-spacing: 1px; "
            f"color: {Palette.text_dim};")
        ml.addWidget(model_head)
        ml.addWidget(QLabel("Checkpoint path (optional — skip to browse later):"))
        row = QHBoxLayout(); row.setSpacing(8)
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText(
            "research/checkpoints/ForgeLM_V2_Light.safetensors")
        row.addWidget(self._model_path, 1)
        self._browse_btn = QPushButton("Browse…")
        row.addWidget(self._browse_btn)
        ml.addLayout(row)
        hint = QLabel(
            "No checkpoint yet? Train one from the Fine-Tune page or download "
            "a compatible safetensors file into research/checkpoints/.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {Palette.text_faint};")
        ml.addWidget(hint)
        root.addWidget(model_card)

        # ── get started button ───────────────────────────────────────
        root.addStretch(1)
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._start_btn = QPushButton("Get started")
        self._start_btn.setObjectName("primary")
        self._start_btn.setMinimumWidth(160)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self._on_get_started)
        btn_row.addWidget(self._start_btn)
        root.addLayout(btn_row)

    # ── public accessors (for tests / app wiring) ────────────────────
    @property
    def trust_checkbox(self) -> QCheckBox:
        return self._trust_cb

    @property
    def model_path_edit(self) -> QLineEdit:
        return self._model_path

    @property
    def start_button(self) -> QPushButton:
        return self._start_btn

    @property
    def browse_button(self) -> QPushButton:
        return self._browse_btn

    def is_trusted(self) -> bool:
        return self._trust_cb.isChecked()

    def model_path(self) -> str:
        return self._model_path.text().strip()

    # ── internals ────────────────────────────────────────────────────
    def _on_get_started(self) -> None:
        trusted = self.is_trusted()
        path = self.model_path()
        # persist to QSettings
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue(_ONBOARDED_KEY, True)
        s.setValue(_TRUST_KEY, trusted)
        if path:
            s.setValue(_MODEL_PATH_KEY, path)
        self.finished_onboarding.emit(trusted, path)
        self.accept()


# ── helpers ──────────────────────────────────────────────────────────────

def is_onboarded() -> bool:
    """Return ``True`` if the user has already completed onboarding."""
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return bool(s.value(_ONBOARDED_KEY, False))


def mark_onboarded(trusted: bool = False, model_path: str = "") -> None:
    """Persist the onboarding-complete flag (and optional extras) to QSettings."""
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(_ONBOARDED_KEY, True)
    s.setValue(_TRUST_KEY, trusted)
    if model_path:
        s.setValue(_MODEL_PATH_KEY, model_path)


def maybe_show_onboarding(parent: Optional[QWidget] = None,
                          force: bool = False) -> Optional[OnboardingDialog]:
    """Show the onboarding dialog if the user hasn't been onboarded yet.

    Returns the dialog instance if it was shown (and accepted), ``None`` if
    onboarding was skipped because the user is already onboarded.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget (typically the QMainWindow).
    force : bool
        If ``True``, show the dialog even if already onboarded (useful for
        a "re-run setup" menu action).
    """
    if not force and is_onboarded():
        return None
    dlg = OnboardingDialog(parent)
    dlg.exec()
    return dlg
