"""Unit tests for R36-5: One-Click Quantize & Activation Preset Wizard.

Covers:
  - use-case → activation kwargs mapping (all 5 use-cases) — pure python,
    no QApplication required.
  - wizard construction (skipped if PySide6 not available).
  - Apply produces a valid ActivationConfig.
"""
from __future__ import annotations

import pytest

from forge_gui.widgets import activation_wizard as wiz


# ── use-case → activation mapping ──────────────────────────────────────

@pytest.mark.parametrize("key,expected", [
    ("chat", {"quantize": "w8a8", "kv_cache": "snapkv",
              "decoding": "standard", "warmup": True}),
    ("coding", {"quantize": "int4", "kv_cache": "s4r",
                "decoding": "speculative", "warmup": True}),
    ("agent", {"quantize": "w8a8", "kv_cache": "standard",
               "decoding": "standard", "use_compile": True, "warmup": True}),
    ("long_context", {"quantize": "nvfp4", "kv_cache": "cpu_offload",
                      "decoding": "standard", "warmup": True}),
    ("max_speed", {"quantize": "fp8", "kv_cache": "rotorquant",
                   "decoding": "medusa", "use_compile": True, "warmup": True}),
])
def test_use_case_config_mapping(key, expected):
    cfg = wiz.use_case_config(key)
    assert cfg == expected


def test_use_case_config_returns_copy():
    a = wiz.use_case_config("chat")
    a["quantize"] = "mutated"
    b = wiz.use_case_config("chat")
    assert b["quantize"] == "w8a8", "use_case_config must return a fresh copy"


def test_use_case_config_unknown_key():
    with pytest.raises(KeyError):
        wiz.use_case_config("bogus")


def test_all_five_use_cases_present():
    keys = {u.key for u in wiz.USE_CASES}
    assert keys == {"chat", "coding", "agent", "long_context", "max_speed"}


def test_use_case_labels_shape():
    labels = wiz.use_case_labels()
    assert len(labels) == 5
    for key, label, desc in labels:
        assert key and label and desc


def test_use_case_configs_only_valid_activation_fields():
    """Every key in each use-case config must be a real ActivationConfig field."""
    from dataclasses import fields as dc_fields
    from forge.engine.activation import ActivationConfig

    valid = {f.name for f in dc_fields(ActivationConfig)}
    for uc in wiz.USE_CASES:
        bad = set(uc.config) - valid
        assert not bad, f"use-case {uc.key} has unknown fields: {bad}"


def test_use_case_configs_valid_options():
    """Combo values must be in the activation_catalog option sets."""
    from forge_gui.api import activation_catalog as cat

    option_sets = {}
    for f in cat.FIELDS:
        if f.kind == "combo":
            option_sets[f.name] = {o.value for o in f.options} | {None}
    for uc in wiz.USE_CASES:
        for k, v in uc.config.items():
            if k in option_sets:
                assert v in option_sets[k], \
                    f"use-case {uc.key}: {k}={v!r} not a valid option"


# ── format_config_summary ──────────────────────────────────────────────

def test_format_config_summary_lists_all_keys():
    cfg = {"quantize": "w8a8", "kv_cache": "snapkv", "decoding": "standard"}
    s = wiz.format_config_summary(cfg)
    assert "quantize" in s and "w8a8" in s
    assert "kv_cache" in s and "snapkv" in s
    assert "decoding" in s and "standard" in s


def test_format_config_summary_handles_none():
    cfg = {"quantize": None, "kv_cache": "snapkv"}
    s = wiz.format_config_summary(cfg)
    assert "none" in s


# ── Apply produces valid ActivationConfig ──────────────────────────────

@pytest.mark.parametrize("key", [
    "chat", "coding", "agent", "long_context", "max_speed",
])
def test_apply_produces_valid_activation_config(key):
    """Each use-case config must round-trip into an ActivationConfig."""
    from forge.engine.activation import ActivationConfig

    cfg = wiz.use_case_config(key)
    ac = ActivationConfig.from_kwargs(**cfg)
    # the values carried through must match
    for k, v in cfg.items():
        assert getattr(ac, k) == v, f"{key}: {k} mismatch ({getattr(ac, k)} != {v})"


@pytest.mark.parametrize("key", [
    "chat", "coding", "agent", "long_context", "max_speed",
])
def test_apply_config_passes_catalog_validation(key):
    """Wizard configs must pass the catalog validator (no bad options)."""
    from forge_gui.api import activation_catalog as cat

    base = cat.default_config()
    base.update(wiz.use_case_config(key))
    errors = cat.validate(base)
    assert not errors, f"{key} config has validation errors: {errors}"


# ── wizard construction (needs PySide6 / QApplication) ─────────────────

try:
    import PySide6  # noqa: F401
    _HAS_PYSIDE = True
except Exception:
    _HAS_PYSIDE = False

pyside_skip = pytest.mark.skipif(
    not _HAS_PYSIDE, reason="PySide6 not available")


@pyside_skip
def test_wizard_construction():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    dialog = wiz.ActivationWizard()
    assert dialog._stack.count() == 3
    # default selection is chat
    assert dialog._selected_key == "chat"
    # radio buttons exist for all use-cases
    assert set(dialog._use_radios) == {
        "chat", "coding", "agent", "long_context", "max_speed"}


@pyside_skip
def test_wizard_apply_sets_result_kwargs():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    dialog = wiz.ActivationWizard(default_key="coding")
    dialog._apply()
    assert dialog.result_kwargs == wiz.use_case_config("coding")


@pyside_skip
def test_wizard_use_case_change_updates_summary():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    dialog = wiz.ActivationWizard(default_key="chat")
    # simulate selecting the agent radio
    dialog._use_radios["agent"].setChecked(True)
    assert dialog._selected_key == "agent"
    assert "rotorquant" not in dialog._step2_cfg_lbl.text()
    assert "standard" in dialog._step2_cfg_lbl.text()
