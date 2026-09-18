"""Tests for master prompt generator and chat features."""
import pytest

from forge.config import get_config
from forge_gui.api.master_prompt import (
    generate_master_prompt,
    get_default_prompt_for_config,
    _detect_model_name,
    _format_arch,
    _format_params,
)


class TestMasterPrompt:
    def test_v2_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "ForgeLM V2" in prompt
        assert "ForgeAI" in prompt
        assert "TEXT-ONLY" in prompt
        assert "cannot see images" in prompt
        assert "coding" in prompt.lower()

    def test_v12_jamba_prompt(self):
        cfg = get_config("forgelm_v12_jamba")
        prompt = generate_master_prompt(cfg, "forgelm_v12_jamba")
        assert "ForgeLM V12 Jamba" in prompt
        assert "MULTIMODAL" not in prompt

    def test_tools_enabled_in_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2", tools_enabled=True)
        assert "tools" in prompt.lower()
        assert "remember" in prompt.lower()

    def test_tools_disabled_no_tool_section(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2", tools_enabled=False)
        assert "load_lora" not in prompt

    def test_thinking_enabled(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2", thinking_enabled=True)
        assert "step-by-step" in prompt.lower()

    def test_arch_in_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "layers" in prompt
        assert "d_model" in prompt

    def test_params_in_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        # should have B or M suffix
        assert any(s in prompt for s in ("B ", "M "))

    def test_donor_model(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "Jamba" in prompt


class TestDetectModelName:
    def test_v2(self):
        assert _detect_model_name("forgelm_v2") == "ForgeLM V2"

    def test_v12_jamba(self):
        assert _detect_model_name("forgelm_v12_jamba") == "ForgeLM V12 Jamba"

    def test_jamba(self):
        assert _detect_model_name("jamba_x") == "ForgeLM V2"

    def test_unknown(self):
        assert _detect_model_name("some_model") == "Some Model"


class TestFormatArch:
    def test_v2_arch(self):
        cfg = get_config("forgelm_v2")
        arch = _format_arch(cfg)
        assert "28 layers" in arch
        assert "d_model=2560" in arch
        assert "GQA" in arch


class TestFormatParams:
    def test_v2_params(self):
        cfg = get_config("forgelm_v2")
        params = _format_params(cfg)
        assert "B" in params or "M" in params


class TestGetDefaultPrompt:
    def test_v2_default(self):
        prompt = get_default_prompt_for_config("forgelm_v2")
        assert "ForgeLM V2" in prompt
        assert "TEXT-ONLY" in prompt

    def test_fallback_on_bad_config(self):
        prompt = get_default_prompt_for_config("nonexistent_config")
        assert "ForgeAI" in prompt
        assert "helpful" in prompt


class TestChatStoreImage:
    def test_append_message_with_image(self):
        from forge_gui.api.chat_store import ChatStore
        store = ChatStore()
        conv = store.create()
        idx = store.append_message(conv["id"], "user", "look at this", image="/tmp/test.png")
        msg = store.get(conv["id"])["messages"][idx]
        assert msg["image"] == "/tmp/test.png"

    def test_append_message_without_image(self):
        from forge_gui.api.chat_store import ChatStore
        store = ChatStore()
        conv = store.create()
        idx = store.append_message(conv["id"], "user", "hello")
        msg = store.get(conv["id"])["messages"][idx]
        assert "image" not in msg
