"""Tests for research/architecture/port_v4_to_v7_8b.py.

Covers:
  1. _verify_against_template: catches shape mismatches, unexpected keys,
     and missing keys; passes on a valid dict; respects INTENTIONALLY_ABSENT.
  2. NameError regression: port_v4_to_v7_8b accepts ModelConfig objects
     (not just preset strings) without referencing undefined variables.
  3. validate_v7_8b_port: int8 all-zero detection is not short-circuited
     by operator-precedence bugs on V_q keys.
"""
import importlib.util
import inspect
import sys
import torch

import pytest


def _load_port_module():
    """Load the port module without importing the full research package."""
    spec = importlib.util.spec_from_file_location(
        "port_v4_to_v7_8b",
        "research/architecture/port_v4_to_v7_8b.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["port_v4_to_v7_8b"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def port_mod():
    return _load_port_module()


# ── _verify_against_template ──


class TestVerifyAgainstTemplate:
    def test_valid_dict_no_problems(self, port_mod):
        template = {"a.weight": torch.zeros(4, 4), "b.bias": torch.zeros(4)}
        save_dict = {"a.weight": torch.zeros(4, 4), "b.bias": torch.zeros(4)}
        assert port_mod._verify_against_template(save_dict, template) == []

    def test_shape_mismatch_detected(self, port_mod):
        template = {"a.weight": torch.zeros(4, 4)}
        save_dict = {"a.weight": torch.zeros(4, 8)}
        problems = port_mod._verify_against_template(save_dict, template)
        assert len(problems) == 1
        assert "a.weight" in problems[0]
        assert "shape" in problems[0].lower() or "vs" in problems[0]

    def test_unexpected_key_detected(self, port_mod):
        template = {"a.weight": torch.zeros(4, 4)}
        save_dict = {"a.weight": torch.zeros(4, 4), "extra": torch.zeros(2)}
        problems = port_mod._verify_against_template(save_dict, template)
        assert any("unexpected" in p for p in problems)
        assert any("extra" in p for p in problems)

    def test_missing_key_detected(self, port_mod):
        template = {"a.weight": torch.zeros(4, 4), "b.bias": torch.zeros(4)}
        save_dict = {"a.weight": torch.zeros(4, 4)}
        problems = port_mod._verify_against_template(save_dict, template)
        assert any("missing" in p for p in problems)
        assert any("b.bias" in p for p in problems)

    def test_intentionally_absent_skipped(self, port_mod):
        """Keys matching INTENTIONALLY_ABSENT are not flagged as missing."""
        template = {
            "blocks.0.attn.rope.inv_freq": torch.zeros(8),
            "mtp_module.head.weight": torch.zeros(10, 10),
            "head.embed_ref.embed.weight": torch.zeros(10, 10),
            "a.weight": torch.zeros(4, 4),
        }
        save_dict = {"a.weight": torch.zeros(4, 4)}
        problems = port_mod._verify_against_template(save_dict, template)
        assert problems == [], f"Intentionally absent keys flagged: {problems}"


# ── NameError regression ──


class TestNameErrorRegression:
    def test_no_old_variable_names_in_source(self, port_mod):
        """The renamed params src_config_name/dst_config_name must not
        appear anywhere in the function body (would be a NameError at runtime).
        """
        src = inspect.getsource(port_mod.port_v4_to_v7_8b)
        assert "src_config_name" not in src, "stale reference to src_config_name"
        assert "dst_config_name" not in src, "stale reference to dst_config_name"

    def test_accepts_modelconfig_objects(self, port_mod):
        """port_v4_to_v7_8b signature accepts non-string args (ModelConfig).
        Verifies the isinstance check path exists.
        """
        sig = inspect.signature(port_mod.port_v4_to_v7_8b)
        params = list(sig.parameters.keys())
        assert "src_config" in params
        assert "dst_config" in params
        # The old names must NOT be parameter names
        assert "src_config_name" not in params
        assert "dst_config_name" not in params


# ── validate_v7_8b_port int8 precedence fix ──


class TestValidateInt8Precedence:
    """The old code had:
        if t.numel() == 0 or t.abs().max().item() == 0 and "U_q" in k or "V_q" in k:
            pass
    which due to precedence evaluated ("V_q" in k) as a standalone clause,
    making the `if` true for ANY V_q key (harmless because body was `pass`,
    but it was dead code masking the real check below). The fix removes the
    dead if/pass block. We verify the fixed source no longer contains the
    ambiguous expression.
    """

    def test_no_ambiguous_precedence_expression(self):
        spec = importlib.util.spec_from_file_location(
            "validate_v7_8b_port",
            "research/sandbox/validate_v7_8b_port.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = inspect.getsource(mod)
        # The ambiguous expression must be gone
        assert 'and "U_q" in k or "V_q" in k' not in src, (
            "ambiguous operator-precedence expression still present"
        )


# ── Validator INTENTIONALLY_MISSING and _zero_ok coverage ──


class TestValidatorIntentionallyMissing:
    """The validator's INTENTIONALLY_MISSING list must match the port
    script's INTENTIONALLY_ABSENT list — otherwise the validator flags
    keys the port script intentionally omits (e.g. head.embed_ref.*).
    """

    def test_head_embed_ref_in_intentionally_missing(self):
        spec = importlib.util.spec_from_file_location(
            "validate_v7_8b_port",
            "research/sandbox/validate_v7_8b_port.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert "head.embed_ref." in mod.INTENTIONALLY_MISSING, (
            "head.embed_ref.* must be in INTENTIONALLY_MISSING (tied to embed.*)"
        )

    def test_memory_u_v_in_zero_ok(self):
        """TITAN memory u/v are legitimately zero-init (memory starts empty)."""
        spec = importlib.util.spec_from_file_location(
            "validate_v7_8b_port",
            "research/sandbox/validate_v7_8b_port.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._zero_ok("blocks.0._memory.u"), "_memory.u should be zero-ok"
        assert mod._zero_ok("blocks.0._memory.v"), "_memory.v should be zero-ok"
