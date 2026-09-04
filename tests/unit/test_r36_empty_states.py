"""Tests for R36-2: Empty-State Guidance + First-Run Onboarding.

Covers:
  - EmptyState widget construction (title, description, action button)
  - OnboardingDialog construction (trust checkbox, model path, start button)
  - First-run detection via QSettings "onboarded" key
  - Empty state shown when the checkpoint list is empty (ModelsPage)

Widget tests that require PySide6 use ``pytest.importorskip`` and a shared
QApplication fixture so they are skipped in headless environments without
PySide6 installed.  Pure-Python logic (QSettings helpers) is tested with
``unittest.mock`` so no running QApplication is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── PySide6 availability ────────────────────────────────────────────────
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from forge_gui.widgets.empty_state import EmptyState  # noqa: E402
from forge_gui.widgets.onboarding import (  # noqa: E402
    OnboardingDialog,
    is_onboarded,
    mark_onboarded,
    maybe_show_onboarding,
    _ONBOARDED_KEY,
    _SETTINGS_ORG,
    _SETTINGS_APP,
)


# ── shared QApplication fixture ─────────────────────────────────────────
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── EmptyState ──────────────────────────────────────────────────────────

class TestEmptyState:
    def test_construction_with_action_button(self, qapp):
        es = EmptyState(
            title="No checkpoints found",
            description="Add one to get started.",
            icon="📦",
            action_text="Browse",
        )
        assert es._title_lbl.text() == "No checkpoints found"
        assert es._desc_lbl.text() == "Add one to get started."
        assert es._icon_lbl.text() == "📦"
        assert es.action_button is not None
        assert es.action_button.text() == "Browse"

    def test_construction_without_action_button(self, qapp):
        es = EmptyState(title="Empty", description="Nothing here")
        assert es.action_button is None

    def test_action_triggered_signal(self, qapp):
        es = EmptyState(title="t", action_text="Go")
        received = []
        es.action_triggered.connect(lambda: received.append(True))
        es.action_button.click()
        assert received == [True]

    def test_on_action_callback(self, qapp):
        calls = []
        es = EmptyState(title="t", action_text="Go", on_action=lambda: calls.append(1))
        es.action_button.click()
        assert calls == [1]

    def test_setters(self, qapp):
        es = EmptyState(title="old", description="old desc", icon="x")
        es.set_title("new title")
        es.set_description("new desc")
        es.set_icon("🎯")
        assert es._title_lbl.text() == "new title"
        assert es._desc_lbl.text() == "new desc"
        assert es._icon_lbl.text() == "🎯"


# ── OnboardingDialog ────────────────────────────────────────────────────

class TestOnboardingDialog:
    def test_construction(self, qapp):
        dlg = OnboardingDialog()
        assert dlg.windowTitle() == "Welcome to ForgeAI"
        # trust checkbox defaults unchecked
        assert dlg.trust_checkbox.isChecked() is False
        # model path edit is empty
        assert dlg.model_path() == ""
        # start button exists
        assert dlg.start_button.text() == "Get started"
        # browse button exists
        assert dlg.browse_button is not None

    def test_is_trusted(self, qapp):
        dlg = OnboardingDialog()
        dlg.trust_checkbox.setChecked(True)
        assert dlg.is_trusted() is True
        dlg.trust_checkbox.setChecked(False)
        assert dlg.is_trusted() is False

    def test_model_path(self, qapp):
        dlg = OnboardingDialog()
        dlg.model_path_edit.setText("research/checkpoints/model.safetensors")
        assert dlg.model_path() == "research/checkpoints/model.safetensors"

    def test_finished_onboarding_signal(self, qapp):
        dlg = OnboardingDialog()
        dlg.trust_checkbox.setChecked(True)
        dlg.model_path_edit.setText("/path/to/ckpt")
        received = []
        dlg.finished_onboarding.connect(
            lambda trusted, path: received.append((trusted, path)))
        # patch QSettings so we don't pollute real settings
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            MockQS.return_value = mock_inst
            dlg._on_get_started()
        assert received == [(True, "/path/to/ckpt")]
        # QSettings.setValue called with onboarded=True
        calls = {args[0]: args[1] for args, _ in mock_inst.setValue.call_args_list}
        assert calls.get(_ONBOARDED_KEY) is True


# ── First-run detection (QSettings helpers) ─────────────────────────────

class TestFirstRunDetection:
    def test_is_onboarded_false_when_key_absent(self):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            mock_inst.value.return_value = False
            MockQS.return_value = mock_inst
            assert is_onboarded() is False
            mock_inst.value.assert_called_once_with(_ONBOARDED_KEY, False)

    def test_is_onboarded_true_when_key_present(self):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            mock_inst.value.return_value = True
            MockQS.return_value = mock_inst
            assert is_onboarded() is True

    def test_mark_onboarded_sets_key(self):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            MockQS.return_value = mock_inst
            mark_onboarded(trusted=True, model_path="/ckpt")
            calls = {args[0]: args[1] for args, _ in mock_inst.setValue.call_args_list}
            assert calls[_ONBOARDED_KEY] is True
            assert calls["agent_trusted"] is True
            assert calls["onboard_model_path"] == "/ckpt"

    def test_mark_onboarded_without_model_path(self):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            MockQS.return_value = mock_inst
            mark_onboarded()
            # onboarded key always set; model path only set when non-empty
            keys_set = [c.args[0] for c in mock_inst.setValue.call_args_list]
            assert _ONBOARDED_KEY in keys_set
            assert "onboard_model_path" not in keys_set

    def test_maybe_show_onboarding_skips_when_already_onboarded(self):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            mock_inst.value.return_value = True
            MockQS.return_value = mock_inst
            result = maybe_show_onboarding()
            assert result is None

    def test_maybe_show_onboarding_shows_when_not_onboarded(self, qapp):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            mock_inst.value.return_value = False
            MockQS.return_value = mock_inst
            # patch the dialog exec so it doesn't block
            with patch.object(OnboardingDialog, "exec", return_value=0):
                result = maybe_show_onboarding()
            assert result is not None
            assert isinstance(result, OnboardingDialog)

    def test_maybe_show_onboarding_force(self, qapp):
        with patch("forge_gui.widgets.onboarding.QSettings") as MockQS:
            mock_inst = MagicMock()
            mock_inst.value.return_value = True  # already onboarded
            MockQS.return_value = mock_inst
            with patch.object(OnboardingDialog, "exec", return_value=0):
                result = maybe_show_onboarding(force=True)
            assert result is not None


# ── Empty state integration in ModelsPage ───────────────────────────────

class TestModelsPageEmptyState:
    def test_empty_state_shown_when_no_checkpoints(self, qapp, tmp_path):
        """When models_index returns no models, the empty state is visible."""
        from forge_gui.pages.models import ModelsPage
        from forge_gui.api.models_index import ModelsIndex

        # ModelsIndex with an empty models() result
        mi = MagicMock(spec=ModelsIndex)
        mi.models.return_value = []
        mi.configs.return_value = []

        page = ModelsPage(models_index=mi, runtime=None)
        page.refresh()

        # the stacked widget should show the empty-state page
        assert page._ck_stack.currentIndex() == 1
        assert page._ck_stack.currentWidget() is page._ck_empty
        assert page._ck_table.rowCount() == 0

    def test_table_shown_when_checkpoints_exist(self, qapp, tmp_path):
        """When models_index returns models, the table is visible."""
        from forge_gui.pages.models import ModelsPage
        from forge_gui.api.models_index import ModelsIndex

        mi = MagicMock(spec=ModelsIndex)
        mock_model = MagicMock()
        mock_model.name = "model.safetensors"
        mock_model.ext = ".safetensors"
        mock_model.path = "research/checkpoints/model.safetensors"
        mock_model.size_label = "1.0 GB"
        mock_model.size_bytes = 1073741824
        mock_model.config_name = "forgelm_v2_light"
        mock_model.modified = 1700000000.0
        mock_model.is_safetensors = True
        mi.models.return_value = [mock_model]
        mi.configs.return_value = []

        page = ModelsPage(models_index=mi, runtime=None)
        page.refresh()

        assert page._ck_stack.currentIndex() == 0
        assert page._ck_stack.currentWidget() is page._ck_table
        assert page._ck_table.rowCount() == 1


# ── Empty state integration in EnginePage ───────────────────────────────

class TestEnginePageEmptyState:
    def test_empty_state_shown_when_no_checkpoints(self, qapp):
        """When no safetensors checkpoints exist, the empty state is visible."""
        from forge_gui.pages.engine import EnginePage
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.api.models_index import ModelsIndex

        runtime = MagicMock(spec=EngineRuntime)
        runtime.state = "idle"
        runtime.info = {}

        mi = MagicMock(spec=ModelsIndex)
        mi.models.return_value = []
        mi.configs.return_value = []

        page = EnginePage(runtime=runtime, models_index=mi)
        page._reload_checkpoints()

        assert not page._ckpt_empty.isHidden()
        assert page._ckpt.isHidden()

    def test_combo_shown_when_checkpoints_exist(self, qapp):
        """When checkpoints exist, the combo is visible and empty state hidden."""
        from forge_gui.pages.engine import EnginePage
        from forge_gui.api.engine_runtime import EngineRuntime
        from forge_gui.api.models_index import ModelsIndex

        runtime = MagicMock(spec=EngineRuntime)
        runtime.state = "idle"
        runtime.info = {}

        mi = MagicMock(spec=ModelsIndex)
        mock_model = MagicMock()
        mock_model.name = "model.safetensors"
        mock_model.path = "research/checkpoints/model.safetensors"
        mock_model.is_safetensors = True
        mi.models.return_value = [mock_model]
        mi.configs.return_value = []

        page = EnginePage(runtime=runtime, models_index=mi)
        page._reload_checkpoints()

        assert page._ckpt_empty.isHidden()
        assert not page._ckpt.isHidden()
