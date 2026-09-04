"""Unit tests for R36-6 Model Download Manager (forge_gui.widgets.download_manager).

Logic tests (VRAM budget, effective size, search filtering, progress tracking)
run WITHOUT a QApplication. Widget-construction tests are skipped if PySide6
is not importable. ``huggingface_hub`` is never required — the download thread
falls back to a mock sweep.
"""
from __future__ import annotations

import importlib
import math
import sys
from unittest.mock import patch

import pytest

# Import the pure-logic helpers directly — they do not require Qt at import
# time because the module only references PySide6 at class-definition scope,
# which is lazy enough for CPython to import the functions. If PySide6 is
# truly absent, skip the module-level import.
PySide6 = importlib.util.find_spec("PySide6")
if PySide6 is None:  # pragma: no cover - environment guard
    pytest.skip("PySide6 not installed", allow_module_level=True)

from forge_gui.widgets.download_manager import (  # noqa: E402
    DownloadManager,
    DownloadThread,
    ModelSearchResult,
    QUANT_OPTIONS,
    RTX_5070_VRAM_GB,
    VRAM_RED_THRESHOLD_GB,
    VRAM_YELLOW_THRESHOLD_GB,
    effective_size_gb,
    search_models,
    vram_warning,
    warning_color,
)
from forge_gui.theme import Palette  # noqa: E402


# ── effective_size_gb ─────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,bits,expected", [
    (16.0, 16, 16.0),
    (16.0, 8, 8.0),
    (16.0, 4, 4.0),
    (14.0, 4, 3.5),
    (6.0, 8, 3.0),
    (1.0, 4, 0.25),
])
def test_effective_size_gb(raw, bits, expected):
    assert math.isclose(effective_size_gb(raw, bits), expected, rel_tol=1e-9)


def test_effective_size_gb_zero_bits_raises():
    with pytest.raises(ValueError):
        effective_size_gb(10.0, 0)


def test_effective_size_gb_negative_bits_raises():
    with pytest.raises(ValueError):
        effective_size_gb(10.0, -4)


# ── vram_warning: all three levels ────────────────────────────────────────
def test_vram_warning_green():
    # 6GB raw @ 4-bit → 1.5GB effective → green
    w = vram_warning(6.0, 4)
    assert w.level == "green"
    assert "comfortably" in w.message.lower()
    assert math.isclose(w.effective_gb, 1.5, rel_tol=1e-9)


def test_vram_warning_yellow():
    # 18GB raw @ 8-bit → 9GB effective → yellow (>8, <=11)
    w = vram_warning(18.0, 8)
    assert w.level == "yellow"
    assert "tight" in w.message.lower()
    assert math.isclose(w.effective_gb, 9.0, rel_tol=1e-9)


def test_vram_warning_red():
    # 48GB raw @ 4-bit → 12GB effective → red (>11)
    w = vram_warning(48.0, 4)
    assert w.level == "red"
    assert "not fit" in w.message.lower() or "may not" in w.message.lower()
    assert math.isclose(w.effective_gb, 12.0, rel_tol=1e-9)


def test_vram_warning_boundary_yellow_to_green():
    # exactly 8GB effective → green (<=8)
    w = vram_warning(16.0, 8)
    assert w.level == "green"


def test_vram_warning_boundary_just_above_yellow():
    # 8.01GB effective → yellow
    w = vram_warning(16.02, 8)
    assert w.level == "yellow"


def test_vram_warning_boundary_red():
    # exactly 11GB effective → yellow (not >11)
    w = vram_warning(22.0, 8)
    assert math.isclose(w.effective_gb, 11.0, rel_tol=1e-9)
    assert w.level == "yellow"
    # just above 11 → red
    w2 = vram_warning(22.02, 8)
    assert w2.level == "red"


def test_vram_warning_custom_vram_budget():
    # 24GB card: red limit becomes 23
    w = vram_warning(46.0, 8, vram_gb=24.0)  # 23GB effective
    assert w.level == "red"


def test_vram_warning_message_contains_effective_size():
    w = vram_warning(10.0, 4)
    assert "2.5" in w.message


# ── warning_color ─────────────────────────────────────────────────────────
def test_warning_color_red():
    assert warning_color("red") == Palette.err


def test_warning_color_yellow():
    assert warning_color("yellow") == Palette.warn


def test_warning_color_green():
    assert warning_color("green") == Palette.ok


def test_warning_color_unknown_defaults_green():
    assert warning_color("bogus") == Palette.ok


# ── search_models ─────────────────────────────────────────────────────────
def test_search_models_empty_query_returns_all():
    cat = (ModelSearchResult("a/b", 1.0), ModelSearchResult("c/d", 2.0))
    assert len(search_models("", cat)) == 2


def test_search_models_substring_filter():
    cat = (
        ModelSearchResult("Qwen/Qwen2.5-0.5B", 1.0),
        ModelSearchResult("meta-llama/Llama-3", 6.0),
        ModelSearchResult("Qwen/Qwen2.5-7B", 14.0),
    )
    out = search_models("qwen", cat)
    assert {m.repo_id for m in out} == {"Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B"}


def test_search_models_case_insensitive():
    cat = (ModelSearchResult("Foo/Bar", 1.0),)
    assert search_models("FOO", cat) == list(cat)


def test_search_models_no_match():
    cat = (ModelSearchResult("Foo/Bar", 1.0),)
    assert search_models("zzz", cat) == []


def test_search_models_default_catalog_nonempty():
    assert len(search_models("")) > 0


# ── DownloadThread progress tracking (no QApplication needed for signals) ─
def _make_thread():
    return DownloadThread("Test/Model", "models/Test-Model")


def test_download_thread_attributes():
    t = _make_thread()
    assert t.repo_id == "Test/Model"
    assert t.target_dir == "models/Test-Model"
    assert t._last_pct == -1
    assert t._cancel is False


def test_download_thread_cancel_flag():
    t = _make_thread()
    t.cancel()
    assert t._cancel is True


def test_download_thread_mock_progress_sweep():
    """Run the mock download (no huggingface_hub) and track emitted progress.

    We bypass QThread.start()/wait() (which need a running event loop) and
    call run() directly, capturing progress signals via a slot list.
    """
    t = _make_thread()
    seen: list[int] = []
    t.progress.connect(lambda p: seen.append(p))
    done_paths: list[str] = []
    t.done.connect(lambda p: done_paths.append(p))
    errs: list[str] = []
    t.error.connect(lambda e: errs.append(e))

    # Force the mock path: make snapshot_download unimportable.
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        t.run()

    assert errs == []
    assert done_paths == ["models/Test-Model"]
    assert seen, "progress should have been emitted at least once"
    assert seen[-1] == 100
    # progress is monotonic non-decreasing
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))


def test_download_thread_cancel_mid_mock():
    t = _make_thread()
    seen: list[int] = []
    t.progress.connect(lambda p: seen.append(p))
    done_paths: list[str] = []
    t.done.connect(lambda p: done_paths.append(p))

    # Cancel immediately so the first loop iteration bails out.
    t.cancel()
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        t.run()
    assert done_paths == []
    # Either no progress or some progress then cancelled — never reaches 100.
    if seen:
        assert seen[-1] < 100


# ── DownloadManager widget construction (needs PySide6 + QApplication) ────
def _need_qapp():
    """Return a QApplication singleton, creating one if necessary."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    return app


def test_download_manager_construction():
    app = _need_qapp()
    app  # silence unused
    dm = DownloadManager()
    assert dm._search is not None
    assert dm._quant.count() == len(QUANT_OPTIONS)
    # default quant is the first option (4-bit)
    assert dm._current_quant_bits() == QUANT_OPTIONS[0]
    # results populated from default catalog
    assert len(dm._results) > 0
    # progress bar hidden until a download starts
    assert not dm._progress.isVisible()


def test_download_manager_search_filters_results():
    app = _need_qapp()
    dm = DownloadManager()
    full = len(dm._results)
    dm._search.setText("qwen")
    filtered = len(dm._results)
    assert filtered <= full
    assert all("qwen" in r.repo_id.lower() for r in dm._results)


def test_download_manager_quant_change_updates_warning_text():
    app = _need_qapp()
    dm = DownloadManager()
    # Find a known large model row and verify the detail label color changes
    # with quantization. We just assert refresh doesn't crash and results
    # rebuild.
    dm._quant.setCurrentIndex(2)  # 16-bit
    assert dm._current_quant_bits() == 16
    dm._quant.setCurrentIndex(0)  # back to 4-bit
    assert dm._current_quant_bits() == 4


# ── constants sanity ──────────────────────────────────────────────────────
def test_constants():
    assert RTX_5070_VRAM_GB == 12.0
    assert VRAM_RED_THRESHOLD_GB == 11.0
    assert VRAM_YELLOW_THRESHOLD_GB == 8.0
    assert QUANT_OPTIONS == (4, 8, 16)
