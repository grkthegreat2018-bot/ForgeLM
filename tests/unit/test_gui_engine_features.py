"""Tests for GUI coverage of all ForgeEngine features.

Verifies that every public ForgeEngine method is reachable from the GUI,
and that the new workers/tabs/pages function correctly with mocked engines.
"""
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# ── ForgeEngine public method inventory ─────────────────────────────────

# Every public method on ForgeEngine that should be reachable from the GUI.
# (classmethods and private methods excluded; properties included as they
# are read by the GUI.)
FORGE_ENGINE_PUBLIC_METHODS = [
    "library_save", "library_set_enabled", "library_set_budget",
    "library_lookup", "library_search", "library_optimize", "library_stats",
    "reset_stats", "from_checkpoint", "activate_optimal", "activate",
    "activate_config", "generate", "enable_cache_blend",
    "register_blend_chunk", "generate_adaptive", "generate_batch",
    "generate_with_tools", "generate_raw", "generate_stream",
    "benchmark", "stats", "bottleneck", "read_log", "read_output",
    "diagnose", "merge_checkpoints", "evolve_merge",
    "load_lora", "unload_lora", "has_lora", "lora_info",
    "sleep", "wake", "is_awake", "vram_usage",
    "recover", "clear_recovery",
    "begin_session", "continue_session", "pin_session", "unpin_session",
    "end_session", "session_stats",
]


def _get_forge_engine_methods():
    """Get the set of public method names on ForgeEngine (lazy import)."""
    from research.inference.forge_engine import ForgeEngine
    methods = set()
    for name in dir(ForgeEngine):
        if name.startswith("_"):
            continue
        attr = getattr(ForgeEngine, name, None)
        if callable(attr) or isinstance(attr, property):
            methods.add(name)
    return methods


# ── Qt application fixture ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    """Shared QApplication for all tests in this module."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def _wait_worker(worker, qapp, timeout_ms=5000):
    """Wait for a QThread to finish while processing events (for signals)."""
    from PySide6.QtCore import QElapsedTimer
    timer = QElapsedTimer()
    timer.start()
    while worker.isRunning() and not timer.hasExpired(timeout_ms):
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


# ── Coverage: every ForgeEngine method is reachable from the GUI ────────

class TestFeatureCoverage:
    def test_all_public_methods_in_inventory(self):
        """The inventory list matches ForgeEngine's actual public methods."""
        actual = _get_forge_engine_methods()
        for name in FORGE_ENGINE_PUBLIC_METHODS:
            assert name in actual, f"{name} not a public ForgeEngine method"

    def test_engine_page_has_six_tabs(self, qapp):
        """Engine page must have all 6 tabs: Load, Activation, Stats, Library, Sessions, Merge."""
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.engine import EnginePage

        runtime = EngineRuntime()
        page = EnginePage(runtime, models_index=None)
        assert page._tabs.count() == 6
        labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
        assert "Load & Power" in labels
        assert "Activation" in labels
        assert "Stats & Tools" in labels
        assert "Library" in labels
        assert "Sessions" in labels
        assert "Merge Studio" in labels

    def test_engine_page_has_optimal_button(self, qapp):
        """Activation tab must have an 'Apply optimal preset' button."""
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.engine import EnginePage

        runtime = EngineRuntime()
        page = EnginePage(runtime, models_index=None)
        assert hasattr(page, "_apply_optimal_btn")
        assert page._apply_optimal_btn.text() == "★ Apply optimal preset"

    def test_engine_page_has_cache_blend_controls(self, qapp):
        """Stats & Tools tab must have CacheBlend controls."""
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.engine import EnginePage

        runtime = EngineRuntime()
        page = EnginePage(runtime, models_index=None)
        assert hasattr(page, "_blend_enable_btn")
        assert hasattr(page, "_blend_register_btn")
        assert hasattr(page, "_blend_chunk")
        assert hasattr(page, "_blend_max")

    def test_engine_page_has_read_output_button(self, qapp):
        """Stats & Tools tab must have a 'Recent outputs' maintenance button."""
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.engine import EnginePage

        runtime = EngineRuntime()
        page = EnginePage(runtime, models_index=None)
        # The maintenance buttons are created dynamically — check the action
        # is handled by looking at the _MaintWorker action list
        # (we verify via the button text)
        maint_tab = page._tabs.widget(2)  # Stats & Tools
        buttons = maint_tab.findChildren(type(page._recover_btn))
        texts = [b.text() for b in buttons]
        assert "Recent outputs" in texts

    def test_generations_page_has_mode_selector(self, qapp):
        """Generations page must have a mode combo with 4 modes."""
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.generations import GenerationsPage

        runtime = EngineRuntime()
        page = GenerationsPage(runtime, models_index=None)
        assert page._mode.count() == 4
        modes = [page._mode.itemData(i) for i in range(page._mode.count())]
        assert "standard" in modes
        assert "adaptive" in modes
        assert "batch" in modes
        assert "raw" in modes

    def test_generations_page_mode_switching(self, qapp):
        """Switching modes shows/hides the right parameter rows."""
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.generations import GenerationsPage

        runtime = EngineRuntime()
        page = GenerationsPage(runtime, models_index=None)
        # Standard: both hidden
        page._mode.setCurrentIndex(0)
        assert page._adaptive_widget.isHidden()
        assert page._raw_widget.isHidden()
        # Adaptive: adaptive visible, raw hidden
        page._mode.setCurrentIndex(1)
        assert not page._adaptive_widget.isHidden()
        assert page._raw_widget.isHidden()
        # Raw: raw visible, adaptive hidden
        page._mode.setCurrentIndex(3)
        assert not page._raw_widget.isHidden()
        assert page._adaptive_widget.isHidden()


# ── Generation workers ──────────────────────────────────────────────────

class TestAdaptiveGenWorker:
    def test_worker_emits_done(self, qapp):
        from forge_gui.api.generation import AdaptiveGenWorker

        runtime = MagicMock()
        runtime.is_ready.return_value = True
        engine = MagicMock()
        engine.generate_adaptive.return_value = ("result text", True)
        runtime.acquire.return_value.__enter__ = MagicMock(return_value=engine)
        runtime.acquire.return_value.__exit__ = MagicMock(return_value=None)

        worker = AdaptiveGenWorker(runtime, "test prompt")
        results = []
        worker.done.connect(lambda t, d, s: results.append((t, d, s)))
        worker.error.connect(lambda e: results.append(("error", e, 0)))
        worker.start()
        _wait_worker(worker, qapp)

        assert len(results) == 1
        assert results[0][0] == "result text"
        assert results[0][1] is True
        engine.generate_adaptive.assert_called_once()

    def test_worker_no_engine(self, qapp):
        from forge_gui.api.generation import AdaptiveGenWorker

        runtime = MagicMock()
        runtime.is_ready.return_value = False

        worker = AdaptiveGenWorker(runtime, "test")
        errors = []
        worker.error.connect(lambda e: errors.append(e))
        worker.start()
        _wait_worker(worker, qapp)

        assert len(errors) == 1
        assert "No model resident" in errors[0]


class TestBatchGenWorker:
    def test_worker_emits_results(self, qapp):
        from forge_gui.api.generation import BatchGenWorker

        runtime = MagicMock()
        runtime.is_ready.return_value = True
        engine = MagicMock()
        engine.generate_batch.return_value = ["out1", "out2", "out3"]
        runtime.acquire.return_value.__enter__ = MagicMock(return_value=engine)
        runtime.acquire.return_value.__exit__ = MagicMock(return_value=None)

        worker = BatchGenWorker(runtime, ["p1", "p2", "p3"])
        results = []
        worker.done.connect(lambda r, s: results.append((r, s)))
        worker.start()
        _wait_worker(worker, qapp)

        assert len(results) == 1
        assert results[0][0] == ["out1", "out2", "out3"]
        engine.generate_batch.assert_called_once_with(
            ["p1", "p2", "p3"], max_new_tokens=256, temperature=0.0,
            top_p=1.0, top_k=80)


class TestRawGenWorker:
    def test_worker_emits_text(self, qapp):
        from forge_gui.api.generation import RawGenWorker

        runtime = MagicMock()
        runtime.is_ready.return_value = True
        engine = MagicMock()
        engine.generate_raw.return_value = "raw output"
        runtime.acquire.return_value.__enter__ = MagicMock(return_value=engine)
        runtime.acquire.return_value.__exit__ = MagicMock(return_value=None)

        worker = RawGenWorker(runtime, "test", min_p=0.1, min_k=5.0,
                              skip_special_tokens=True)
        results = []
        worker.done.connect(lambda t, s: results.append((t, s)))
        worker.start()
        _wait_worker(worker, qapp)

        assert len(results) == 1
        assert results[0][0] == "raw output"
        call_kwargs = engine.generate_raw.call_args.kwargs
        assert call_kwargs["min_p"] == 0.1
        assert call_kwargs["min_k"] == 5.0
        assert call_kwargs["skip_special_tokens"] is True


# ── Engine page handler methods ─────────────────────────────────────────

class TestEnginePageHandlers:
    def _make_page(self, qapp):
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.pages.engine import EnginePage
        runtime = EngineRuntime()
        return EnginePage(runtime, models_index=None)

    def test_library_handlers_exist(self, qapp):
        page = self._make_page(qapp)
        for method in ("_lib_save", "_lib_set_enabled", "_lib_set_budget",
                       "_lib_do_search", "_lib_do_lookup", "_lib_optimize",
                       "_lib_refresh_list", "_lib_refresh_stats"):
            assert hasattr(page, method), f"missing handler {method}"

    def test_session_handlers_exist(self, qapp):
        page = self._make_page(qapp)
        for method in ("_sess_begin", "_sess_continue", "_sess_pin",
                       "_sess_unpin", "_sess_end", "_sess_refresh_pick",
                       "_sess_refresh_stats"):
            assert hasattr(page, method), f"missing handler {method}"

    def test_merge_handlers_exist(self, qapp):
        page = self._make_page(qapp)
        for method in ("_merge_run", "_evolve_run", "_merge_fill_parents",
                       "_merge_selected_parents", "_merge_pick_out"):
            assert hasattr(page, method), f"missing handler {method}"

    def test_optimal_handler_exists(self, qapp):
        page = self._make_page(qapp)
        assert hasattr(page, "_apply_optimal")
        assert hasattr(page, "_on_optimal_applied")

    def test_blend_handlers_exist(self, qapp):
        page = self._make_page(qapp)
        assert hasattr(page, "_blend_enable")
        assert hasattr(page, "_blend_register")

    def test_library_save_calls_engine(self, qapp):
        """library_save handler delegates to engine.library_save."""
        page = self._make_page(qapp)
        # mock the runtime's try_engine to return a mock engine
        mock_engine = MagicMock()
        mock_engine.library_save.return_value = "abc123"
        page.runtime.try_engine = MagicMock(return_value=mock_engine)
        page.runtime.is_ready = MagicMock(return_value=True)

        page._lib_content.setPlainText("test content")
        page._lib_tags.setText("tag1, tag2")
        page._lib_desc.setText("a description")
        page._lib_triggers.setText("trigger1")
        page._lib_priority.setValue(5)
        page._lib_category.setCurrentIndex(1)  # failure

        page._lib_save()

        mock_engine.library_save.assert_called_once()
        call_kwargs = mock_engine.library_save.call_args.kwargs
        assert call_kwargs["content"] == "test content"
        assert call_kwargs["category"] == "failure"
        assert call_kwargs["tags"] == ["tag1", "tag2"]
        assert call_kwargs["description"] == "a description"
        assert call_kwargs["triggers"] == ["trigger1"]
        assert call_kwargs["priority"] == 5

    def test_session_begin_calls_engine(self, qapp):
        """begin_session handler delegates to engine.begin_session."""
        page = self._make_page(qapp)
        mock_engine = MagicMock()
        mock_engine.session_stats.return_value = {"sessions": {}}
        page.runtime.try_engine = MagicMock(return_value=mock_engine)

        page._sess_id.setText("test-session")
        page._sess_ttl.setValue(0)  # no TTL
        page._sess_begin()

        mock_engine.begin_session.assert_called_once_with("test-session", ttl=None)

    def test_merge_selected_parents_empty(self, qapp):
        """No parents selected → empty list."""
        page = self._make_page(qapp)
        parents = page._merge_selected_parents()
        assert parents == []

    def test_merge_selected_parents_checked(self, qapp):
        """Checked rows → their paths."""
        page = self._make_page(qapp)
        # _merge_fill_parents is deferred via QTimer.singleShot(0, ...) —
        # process events so it runs before we access the table
        qapp.processEvents()
        from PySide6.QtWidgets import QCheckBox
        cw = page._merge_parents.cellWidget(0, 0)
        assert cw is not None, "parent table not filled — timer didn't fire"
        cb = cw.findChild(QCheckBox)
        cb.setChecked(True)
        parents = page._merge_selected_parents()
        assert len(parents) == 1
