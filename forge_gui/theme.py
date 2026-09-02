"""Dark, sleek QSS theme + shared color palette for ForgeAI GUI."""
from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


class Palette:
    """Central color tokens (hex strings + QColor)."""

    bg = "#090c12"
    bg_alt = "#0e121b"
    panel = "#161b28"
    panel_alt = "#1d2432"
    border = "#232b3c"
    border_hi = "#313c55"
    text = "#e8edf5"
    text_dim = "#93a0b4"
    text_faint = "#5d687c"
    accent = "#8aa3ff"        # primary blue
    accent_hi = "#aebdff"
    accent_dim = "#40508a"
    accent2 = "#3ad9c9"       # complementary teal/cyan
    ok = "#3fb950"
    warn = "#d29922"
    err = "#f85149"
    grad_a = "#8aa3ff"
    grad_b = "#a06bff"
    card_hover = "#3b4a6b"    # card border highlight on hover
    chart_grid = "#1c2333"    # subtle chart grid lines
    chart_loss = "#f85149"
    chart_lr = "#d29922"
    chart_reward = "#3fb950"
    chart_kl = "#a06bff"
    chart_div = "#8aa3ff"

    @classmethod
    def qcolor(cls, name: str) -> QColor:
        return QColor(getattr(cls, name))


def _palette_tokens() -> dict[str, str]:
    """Hex-token dict for QSS templating (all str Palette attributes)."""
    return {k: v for k, v in vars(Palette).items()
            if isinstance(v, str) and v.startswith("#")}


QSS = Template(r"""
QWidget#root { background: $bg; }

QMainWindow, QWidget { background: $bg; color: $text; font-size: 13px; }

/* ── Sidebar ─────────────────────────────────────────── */
QWidget#sidebar { background: $bg_alt; border-right: 1px solid $border; }
QLabel#brand { color: $text; font-size: 18px; font-weight: 700;
    letter-spacing: 1px; padding: 4px 0 2px 0; }
QLabel#brandSub { color: $text_faint; font-size: 10px; letter-spacing: 3px;
    font-weight: 600; }
NavButton {
    text-align: left; padding: 10px 16px; border: none; border-radius: 8px;
    border-left: 3px solid transparent;
    color: $text_dim; font-size: 13px; font-weight: 500;
}
NavButton:hover {
    background: $panel_alt; color: $text; border-left: 3px solid $accent_dim;
}
NavButton[active="true"] {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 $panel_alt, stop:1 $panel);
    color: $accent_hi; font-weight: 600;
    border-left: 3px solid $accent;
}
QLabel#statusLabel { color: $text_faint; font-size: 11px; padding: 6px 16px; }
QLabel#navSection { color: $text_faint; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; padding: 8px 16px 2px 16px; }
QFrame#sidebarSep { background: $border; max-height: 1px; border: none; }
QToolButton#collapseBtn { background: transparent; border: none;
    color: $text_faint; font-size: 11px; padding: 4px;
    border-radius: 4px; max-width: 24px; max-height: 24px; }
QToolButton#collapseBtn:hover { background: $surface_hover; color: $text; }

/* ── Topbar ──────────────────────────────────────────── */
QFrame#topbar { background: $bg; border-bottom: 1px solid $border; }
QLabel#pageTitle { color: $text; font-size: 18px; font-weight: 700; }
QLabel#pageSubtitle { color: $text_faint; font-size: 12px; }
QLabel#liveDot { color: $ok; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
QLabel#clock {
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace;
    color: $text_dim; font-size: 12px;
}

/* ── Pages / scroll ──────────────────────────────────── */
QScrollArea, QStackedWidget#pages { background: $bg; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px 1px; }
QScrollBar::handle:vertical { background: $border_hi; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: $accent_dim; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 1px 2px; }
QScrollBar::handle:horizontal { background: $border_hi; border-radius: 4px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: $accent_dim; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Splitters ───────────────────────────────────────── */
QSplitter::handle { background: $border; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QSplitter::handle:hover { background: $accent_dim; }

/* ── Cards / panels ──────────────────────────────────── */
QFrame#card {
    background: $panel; border: 1px solid $border; border-radius: 8px;
}
QFrame#card:hover { border-color: $card_hover; }
QFrame#cardAlt {
    background: $panel_alt; border: 1px solid $border_hi; border-radius: 8px;
}
QFrame#cardAlt:hover { border-color: $card_hover; }
QLabel#cardTitle { color: $text_dim; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; padding: 14px 16px 0 16px; }
QLabel#cardValue { color: $text; font-size: 24px; font-weight: 700;
    padding: 2px 16px 14px 16px; }
QLabel#cardValueSmall { color: $text; font-size: 18px; font-weight: 600;
    padding: 2px 16px 14px 16px; }
QLabel#cardUnit { color: $text_faint; font-size: 12px; font-weight: 400; }
QLabel#cardDeltaUp { color: $ok; font-size: 11px; font-weight: 600; }
QLabel#cardDeltaDown { color: $err; font-size: 11px; font-weight: 600; }
QLabel#sectionHeader { color: $text_dim; font-size: 10px; font-weight: 700;
    letter-spacing: 3px; padding: 4px 0; }

/* ── Inputs / buttons ────────────────────────────────── */
QPushButton {
    background: #1c2230; color: #e6edf3; border: 1px solid #2e3850;
    border-radius: 8px; padding: 8px 16px; font-size: 13px; font-weight: 500;
}
QPushButton:hover { background: #232a3a; border-color: #3a4a7a; }
QPushButton:pressed { background: #161b27; }
QPushButton:disabled { color: #5a6577; border-color: #232a3a; }
QPushButton#primary {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c9cff, stop:1 #a06bff);
    color: #0b0e14; border: none; font-weight: 700;
}
QPushButton#primary:hover { background: #9fb4ff; }
QPushButton#primary:disabled { background: #2e3850; color: #5a6577; }
QPushButton#danger { color: #f85149; border-color: #3a2a2a; }
QPushButton#danger:hover { background: #2a1a1a; }

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #11151f; color: #e6edf3; border: 1px solid #232a3a;
    border-radius: 8px; padding: 8px 10px; font-size: 13px;
    selection-background-color: #3a4a7a;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7c9cff; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #161b27; color: #e6edf3; border: 1px solid #2e3850;
    selection-background-color: #2e3850; outline: 0;
}
QComboBox::down-arrow { image: none; width: 0; height: 0; }

QProgressBar {
    background: #11151f; border: 1px solid #232a3a; border-radius: 6px;
    text-align: center; color: #e6edf3; font-size: 11px; height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c9cff, stop:1 #a06bff);
    border-radius: 5px;
}

QGroupBox {
    background: #161b27; border: 1px solid #232a3a; border-radius: 10px;
    margin-top: 14px; padding: 14px 12px 12px 12px; color: #8b96a8;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
}
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; }

QLabel#mono { font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace; }

/* Card body text (mono info blocks) and empty-state placeholder */
QLabel#cardBody {
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace;
    color: #8b96a8; font-size: 12px; padding: 14px 16px;
}
QLabel#cardEmpty { color: #5a6577; font-size: 12px; padding: 8px 0; }

/* Data tables (checkpoints / configs) */
QTableWidget#dataTable {
    background: #161b27; alternate-background-color: #1c2230;
    border: none; border-radius: 0 0 12px 12px; color: #e6edf3;
    gridline-color: #232a3a;
}
QHeaderView::section {
    background: #11151f; color: #8b96a8; padding: 8px;
    border: none; border-bottom: 1px solid #232a3a; font-weight: 600;
}
QTableWidget#dataTable::item { padding: 6px 8px; }
QTableWidget#dataTable::item:selected { background: #2e3850; }
QPlainTextEdit#logView, QTextEdit#logView {
    background: #0b0e14; color: #c8d2e0; border: 1px solid #232a3a;
    border-radius: 8px; font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace;
    font-size: 12px; padding: 8px;
}
QLabel#tagOk { color: #3fb950; font-size: 11px; font-weight: 700; }
QLabel#tagWarn { color: #d29922; font-size: 11px; font-weight: 700; }
QLabel#tagErr { color: #f85149; font-size: 11px; font-weight: 700; }
QLabel#tagIdle { color: #5a6577; font-size: 11px; font-weight: 700; }

QToolTip { background: #1c2230; color: #e6edf3; border: 1px solid #2e3850;
    border-radius: 6px; padding: 6px 8px; font-size: 12px; }

/* ── Chat studio ─────────────────────────────────────── */
QFrame#bubbleUser { background: #1d2740; border: 1px solid #33436e;
    border-radius: 12px 12px 4px 12px; }
QFrame#bubbleAssistant { background: #161b28; border: 1px solid #232b3c;
    border-radius: 12px 12px 12px 4px; }
QFrame#bubbleSystem { background: #141a26; border: 1px dashed #2e3850;
    border-radius: 10px; }
QLabel#bubbleWho { font-size: 10px; font-weight: 700; letter-spacing: 1px;
    color: #93a0b4; }
QLabel#bubbleBody, QTextEdit#bubbleBody {
    background: transparent; color: #e8edf5; font-size: 13px; border: none; }
QTextEdit#bubbleBody {
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace;
    font-size: 12px; }
QPushButton#rateGood { border: 1px solid #233a2a; color: #3fb950;
    border-radius: 6px; padding: 2px 8px; font-size: 12px; font-weight: 700; }
QPushButton#rateGood:checked { background: #16321e; border-color: #3fb950; }
QPushButton#rateBad { border: 1px solid #3a2a2a; color: #f85149;
    border-radius: 6px; padding: 2px 8px; font-size: 12px; font-weight: 700; }
QPushButton#rateBad:checked { background: #321616; border-color: #f85149; }
QLabel#chatMeta { color: #5d687c; font-size: 10px; }

/* ── Chat: tool + thinking blocks ────────────────────── */
QFrame#toolBlock { background: #111620; border: 1px solid #2a3450;
    border-radius: 8px; }
QFrame#thinkingBlock { background: #0f1420; border: 1px solid #3a2a5a;
    border-radius: 8px; }
QLabel#toolLabel { color: #58a6ff; font-size: 11px; font-weight: 600; }
QLabel#thinkingLabel { color: #bc8cff; font-size: 11px; font-weight: 600;
    font-style: italic; }
QToolButton#toolToggle { background: transparent; border: none;
    color: #8b96a8; font-size: 10px; padding: 0; }
QTextEdit#toolDetail { background: #0a0e16; color: #b1bac4;
    font-family: 'Cascadia Mono','Consolas',monospace; font-size: 11px;
    border: 1px solid #1e2638; border-radius: 4px; padding: 4px; }
QTextEdit#thinkingBody { background: #0a0e16; color: #c9b3e6;
    font-family: 'Cascadia Mono','Consolas',monospace; font-size: 11px;
    border: 1px solid #2a1e3e; border-radius: 4px; padding: 4px; }
QLabel#bubbleImage { background: transparent; border: 1px solid #2e3850;
    border-radius: 8px; padding: 4px; }

/* ── Agent trace ─────────────────────────────────────── */
QFrame#agentRound { background: #12161f; border: 1px solid #232b3c;
    border-radius: 10px; }
QLabel#agentRoundHead { color: #a06bff; font-size: 11px; font-weight: 700;
    letter-spacing: 1px; }
QFrame#toolCard { background: #171d2a; border: 1px solid #2a3450;
    border-radius: 8px; }
QLabel#toolName { color: #3ad9c9; font-size: 11px; font-weight: 700;
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace; }
QLabel#toolArgs, QTextEdit#toolArgs {
    background: #0b0e14; color: #9fb4ff; border: none; border-radius: 6px;
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace;
    font-size: 11px; padding: 6px; }
QTextEdit#toolResult {
    background: #0b0e14; color: #8b96a8; border: none; border-radius: 6px;
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace;
    font-size: 11px; padding: 6px; }
QLabel#toolOk { color: #3fb950; font-size: 10px; font-weight: 700; }
QLabel#toolErr { color: #f85149; font-size: 10px; font-weight: 700; }
QLabel#toolPending { color: #d29922; font-size: 10px; font-weight: 700; }

/* ── Engine console / fine-tune studio ───────────────── */
QLabel#engineState { font-size: 12px; font-weight: 700; letter-spacing: 1px; }
QFrame#kvRow { background: #12161f; border: 1px solid #232b3c;
    border-radius: 8px; }
QLabel#kvKey { color: #93a0b4; font-size: 11px; font-weight: 600; }
QLabel#kvVal { color: #e8edf5; font-size: 12px;
    font-family: 'Cascadia Mono','JetBrains Mono','Consolas',monospace; }
QListWidget#datasetList, QListWidget#convList {
    background: #11151f; border: 1px solid #232a3a; border-radius: 8px;
    color: #e6edf3; font-size: 12px; padding: 4px; outline: 0; }
QListWidget#datasetList::item, QListWidget#convList::item {
    padding: 6px 8px; border-radius: 6px; }
QListWidget#datasetList::item:selected, QListWidget#convList::item:selected {
    background: #2e3850; color: #e8edf5; }
QListWidget#datasetList::item:hover, QListWidget#convList::item:hover {
    background: #1d2432; }
QCheckBox { color: #93a0b4; font-size: 12px; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid #2e3850; background: #11151f; }
QCheckBox::indicator:checked { background: #7c9cff;
    border-color: #7c9cff; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }
QTabWidget::pane { border: 1px solid #232b3c; border-radius: 8px;
    background: #0e121b; }
QTabBar::tab { background: transparent; color: #93a0b4; padding: 8px 14px;
    border: none; border-bottom: 2px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: #aebdff; border-bottom: 2px solid #8aa3ff; }
QTabBar::tab:hover { color: #e8edf5; }
""")


def apply_theme(app: "QApplication") -> None:
    """Apply the ForgeAI dark theme to the application."""
    app.setStyleSheet(QSS.substitute(**_palette_tokens()))
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, Palette.qcolor("bg"))
    pal.setColor(QPalette.ColorRole.Base, Palette.qcolor("bg_alt"))
    pal.setColor(QPalette.ColorRole.Text, Palette.qcolor("text"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, Palette.qcolor("text_faint"))
    pal.setColor(QPalette.ColorRole.Button, Palette.qcolor("panel_alt"))
    pal.setColor(QPalette.ColorRole.ButtonText, Palette.qcolor("text"))
    pal.setColor(QPalette.ColorRole.Highlight, Palette.qcolor("accent_dim"))
    pal.setColor(QPalette.ColorRole.HighlightedText, Palette.qcolor("text"))
    app.setPalette(pal)
    f = QFont()
    fam = QFontDatabase.families()
    for preferred in ("Inter", "Segoe UI Variable", "Segoe UI", "SF Pro Text", "Roboto"):
        if preferred in fam:
            f.setFamily(preferred)
            break
    else:
        f.setFamilies(["Inter", "Segoe UI Variable", "Segoe UI", "SF Pro Text", "Roboto"])
    f.setPointSize(10)
    app.setFont(f)
