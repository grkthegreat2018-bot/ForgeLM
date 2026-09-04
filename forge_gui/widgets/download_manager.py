"""R36-6: Model Download Manager — HuggingFace Hub downloader widget.

Provides:
- Pure VRAM-budget helpers (``effective_size_gb``, ``vram_warning``) that can
  be unit-tested without a QApplication.
- :class:`DownloadThread` — a QThread that runs ``huggingface_hub``'s
  ``snapshot_download`` (or a mock when the library is absent) and emits
  progress / completion / error signals.
- :class:`DownloadManager` — a QWidget with a search field, model result list
  (size + estimated post-quant VRAM), per-model Download button, progress bar,
  and a quantization selector (4 / 8 / 16 bit). VRAM warnings are colored
  using :class:`forge_gui.theme.Palette` (red / yellow / green).

The widget can be embedded in a page or shown as a dialog. It does not modify
``app.py`` or ``theme.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..theme import Palette

logger = logging.getLogger(__name__)

# ── Hardware constants ────────────────────────────────────────────────────
RTX_5070_VRAM_GB = 12.0
# Leave ~1 GB headroom for activations / KV cache before we call it "won't fit".
VRAM_RED_THRESHOLD_GB = 11.0
VRAM_YELLOW_THRESHOLD_GB = 8.0

QUANT_OPTIONS = (4, 8, 16)  # bit widths offered in the UI


# ── Pure VRAM-budget logic (no Qt dependency) ─────────────────────────────
def effective_size_gb(raw_size_gb: float, quant_bits: int) -> float:
    """Post-quantization model footprint in GB.

    Formula: ``raw_size_gb * (quant_bits / 16)`` — a 16-bit (bf16/fp16)
    checkpoint at ``quant_bits=4`` shrinks to ~25% of its raw size.
    """
    if quant_bits <= 0:
        raise ValueError("quant_bits must be positive")
    return raw_size_gb * (quant_bits / 16.0)


@dataclass(frozen=True)
class VramWarning:
    """Result of a VRAM-budget check."""

    level: str          # "red" | "yellow" | "green"
    message: str
    effective_gb: float


def vram_warning(raw_size_gb: float, quant_bits: int,
                 vram_gb: float = RTX_5070_VRAM_GB) -> VramWarning:
    """Classify a model's post-quant footprint against the VRAM budget.

    - effective > 11 GB  → red    "May not fit in 12GB VRAM"
    - effective > 8 GB   → yellow "Tight fit, consider lower quantization"
    - effective <= 8 GB  → green  "Fits comfortably"
    """
    eff = effective_size_gb(raw_size_gb, quant_bits)
    red_limit = min(VRAM_RED_THRESHOLD_GB, vram_gb - 1.0)
    if eff > red_limit:
        return VramWarning("red",
                           f"May not fit in {vram_gb:.0f}GB VRAM "
                           f"({eff:.1f}GB needed)", eff)
    if eff > VRAM_YELLOW_THRESHOLD_GB:
        return VramWarning("yellow",
                           "Tight fit, consider lower quantization "
                           f"({eff:.1f}GB)", eff)
    return VramWarning("green", f"Fits comfortably ({eff:.1f}GB)", eff)


def warning_color(level: str) -> str:
    """Map a warning level to a Palette hex token."""
    if level == "red":
        return Palette.err
    if level == "yellow":
        return Palette.warn
    return Palette.ok


# ── Model search result (pure data) ───────────────────────────────────────
@dataclass
class ModelSearchResult:
    """A single HuggingFace model match returned by the search."""

    repo_id: str
    raw_size_gb: float
    description: str = ""


# A small built-in catalog used when HuggingFace search is unavailable /
# offline. Maps a friendly name to an approximate raw (fp16) size in GB.
_FALLBACK_CATALOG: tuple[ModelSearchResult, ...] = (
    ModelSearchResult("Qwen/Qwen2.5-0.5B-Instruct", 1.0, "0.5B params, fp16"),
    ModelSearchResult("Qwen/Qwen2.5-1.5B-Instruct", 3.0, "1.5B params, fp16"),
    ModelSearchResult("Qwen/Qwen2.5-3B-Instruct", 6.0, "3B params, fp16"),
    ModelSearchResult("Qwen/Qwen2.5-7B-Instruct", 14.0, "7B params, fp16"),
    ModelSearchResult("meta-llama/Llama-3.2-1B-Instruct", 2.0, "1B params, fp16"),
    ModelSearchResult("meta-llama/Llama-3.2-3B-Instruct", 6.0, "3B params, fp16"),
    ModelSearchResult("microsoft/Phi-3.5-mini-instruct", 7.5, "3.8B params, fp16"),
    ModelSearchResult("HuggingFaceTB/SmolLM2-1.7B-Instruct", 3.4, "1.7B params, fp16"),
)


def search_models(query: str,
                  catalog: Optional[tuple[ModelSearchResult, ...]] = None
                  ) -> list[ModelSearchResult]:
    """Filter the catalog by substring match on repo_id (case-insensitive).

    Uses the fallback catalog by default so the widget is usable offline.
    When ``huggingface_hub`` is installed the widget may swap in a live
    search; this pure helper keeps the logic testable without network.
    """
    cat = catalog if catalog is not None else _FALLBACK_CATALOG
    q = (query or "").strip().lower()
    if not q:
        return list(cat)
    return [m for m in cat if q in m.repo_id.lower()]


# ── Download thread ───────────────────────────────────────────────────────
class DownloadThread(QThread):
    """Background HuggingFace snapshot download.

    Emits:
        progress(int)   — 0..100 percent
        status(str)     — human-readable status line
        done(str)       — local path on success
        error(str)      — error message on failure
    """

    progress = Signal(int)
    status = Signal(str)
    done = Signal(str)
    error = Signal(str)

    def __init__(self, repo_id: str, target_dir: str,
                 repo_type: str = "model", parent=None) -> None:
        super().__init__(parent)
        self.repo_id = repo_id
        self.target_dir = target_dir
        self.repo_type = repo_type
        self._cancel = False
        self._last_pct = -1

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: C901
        try:
            self.status.emit(f"Resolving {self.repo_id}…")
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                snapshot_download = None  # type: ignore[assignment]

            if snapshot_download is None:
                # Mock download: emit a smooth 0→100 progress sweep so the
                # UI is demonstrable without the library / network.
                self.status.emit(f"[mock] downloading {self.repo_id}…")
                import time
                steps = 20
                for i in range(1, steps + 1):
                    if self._cancel:
                        self.status.emit("Cancelled")
                        return
                    pct = int(i * 100 / steps)
                    if pct != self._last_pct:
                        self._last_pct = pct
                        self.progress.emit(pct)
                    time.sleep(0.02)
                self.done.emit(self.target_dir)
                return

            def _on_progress(p: int) -> None:
                if self._cancel:
                    raise RuntimeError("cancelled")
                if p != self._last_pct:
                    self._last_pct = p
                    self.progress.emit(p)

            self.status.emit(f"Downloading {self.repo_id}…")
            local_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                local_dir=self.target_dir,
                token=None,
                max_workers=4,
            )
            self.progress.emit(100)
            self.done.emit(str(local_path))
        except RuntimeError as e:
            if "cancel" in str(e).lower():
                self.status.emit("Cancelled")
                return
            self.error.emit(str(e))
        except Exception as e:
            logger.warning("download failed: %s", e, exc_info=True)
            self.error.emit(f"{type(e).__name__}: {e}")


# ── Download manager widget ───────────────────────────────────────────────
class DownloadManager(QWidget):
    """Search HuggingFace models, preview post-quant VRAM, download."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[DownloadThread] = None
        self._results: list[ModelSearchResult] = []
        self._build_ui()
        self._refresh_results()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("MODEL DOWNLOAD MANAGER")
        title.setObjectName("sectionHeader")
        root.addWidget(title)

        # Search row
        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search HuggingFace models…")
        self._search.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self._search, 1)

        self._quant = QComboBox()
        for b in QUANT_OPTIONS:
            self._quant.addItem(f"{b}-bit", b)
        self._quant.setCurrentIndex(0)  # 4-bit default
        self._quant.currentIndexChanged.connect(self._refresh_results)
        search_row.addWidget(QLabel("Quant:"))
        search_row.addWidget(self._quant)
        root.addLayout(search_row)

        # Results list
        self._results_box = QVBoxLayout()
        self._results_box.setSpacing(6)
        root.addLayout(self._results_box, 1)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("cardEmpty")
        root.addWidget(self._status_lbl)

        root.addStretch(1)

    # ── Behavior ──────────────────────────────────────────────────────────
    def _current_quant_bits(self) -> int:
        return self._quant.currentData()

    def _on_search_changed(self, _text: str) -> None:
        self._refresh_results()

    def _refresh_results(self) -> None:
        # Clear existing rows
        while self._results_box.count():
            item = self._results_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        query = self._search.text()
        self._results = search_models(query)
        bits = self._current_quant_bits()
        for res in self._results:
            row = self._make_result_row(res, bits)
            self._results_box.addWidget(row)

    def _make_result_row(self, res: ModelSearchResult, bits: int) -> QWidget:
        frame = QFrame()
        frame.setObjectName("card")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        info = QVBoxLayout()
        info.setSpacing(2)
        name_lbl = QLabel(res.repo_id)
        name_lbl.setStyleSheet(f"color: {Palette.text}; font-weight: 600;")
        info.addWidget(name_lbl)

        warn = vram_warning(res.raw_size_gb, bits)
        eff = effective_size_gb(res.raw_size_gb, bits)
        detail = (f"raw {res.raw_size_gb:.1f}GB · {bits}-bit → {eff:.1f}GB · "
                  f"{warn.message}")
        detail_lbl = QLabel(detail)
        detail_lbl.setStyleSheet(f"color: {warning_color(warn.level)}; "
                                 f"font-size: 11px;")
        info.addWidget(detail_lbl)
        lay.addLayout(info, 1)

        dl_btn = QPushButton("Download")
        dl_btn.setObjectName("primary")
        dl_btn.clicked.connect(lambda _=False, r=res: self._start_download(r))
        lay.addWidget(dl_btn)
        return frame

    def _start_download(self, res: ModelSearchResult) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._status_lbl.setText("A download is already in progress.")
            return
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_lbl.setText(f"Starting download: {res.repo_id}")
        self._thread = DownloadThread(
            repo_id=res.repo_id,
            target_dir=f"models/{res.repo_id.split('/')[-1]}",
            parent=self,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.status.connect(self._status_lbl.setText)
        self._thread.done.connect(self._on_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, pct: int) -> None:
        self._progress.setValue(pct)

    def _on_done(self, path: str) -> None:
        self._status_lbl.setText(f"Downloaded to {path}")
        self._progress.setValue(100)

    def _on_error(self, msg: str) -> None:
        self._status_lbl.setText(f"Error: {msg}")
        self._status_lbl.setStyleSheet(f"color: {Palette.err};")

    def cancel_download(self) -> None:
        if self._thread is not None:
            self._thread.cancel()
