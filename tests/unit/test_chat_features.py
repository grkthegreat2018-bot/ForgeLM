"""Tests for master prompt generator and chat features."""
import pytest

from research.config import get_config
from forge_gui.api.master_prompt import (
    generate_master_prompt,
    get_default_prompt_for_config,
    _detect_model_name,
    _format_arch,
    _format_params,
)


class TestMasterPrompt:
    def test_v2_light_prompt(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light")
        assert "ForgeLM V2 Light" in prompt
        assert "ForgeAI" in prompt
        assert "TEXT-ONLY" in prompt
        assert "cannot see images" in prompt
        assert "coding" in prompt.lower()

    def test_v2_pro_prompt(self):
        cfg = get_config("forgelm_v2_pro")
        prompt = generate_master_prompt(cfg, "forgelm_v2_pro")
        assert "ForgeLM V2 Pro" in prompt
        assert "MULTIMODAL" in prompt
        assert "can see and analyze images" in prompt

    def test_pro_has_vision_light_doesnt(self):
        light = generate_master_prompt(get_config("forgelm_v2_light"), "forgelm_v2_light")
        pro = generate_master_prompt(get_config("forgelm_v2_pro"), "forgelm_v2_pro")
        assert "MULTIMODAL" not in light
        assert "MULTIMODAL" in pro

    def test_tools_enabled_in_prompt(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light", tools_enabled=True)
        assert "tools" in prompt.lower()
        assert "remember" in prompt.lower()

    def test_tools_disabled_no_tool_section(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light", tools_enabled=False)
        assert "load_lora" not in prompt

    def test_thinking_enabled(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light", thinking_enabled=True)
        assert "step-by-step" in prompt.lower()

    def test_arch_in_prompt(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light")
        assert "layers" in prompt
        assert "d_model" in prompt

    def test_params_in_prompt(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light")
        # should have B or M suffix
        assert any(s in prompt for s in ("B ", "M "))

    def test_donor_model_light(self):
        cfg = get_config("forgelm_v2_light")
        prompt = generate_master_prompt(cfg, "forgelm_v2_light")
        assert "LFM 2.5" in prompt

    def test_donor_model_pro(self):
        cfg = get_config("forgelm_v2_pro")
        prompt = generate_master_prompt(cfg, "forgelm_v2_pro")
        assert "LFM 2.5-VL" in prompt


class TestDetectModelName:
    def test_v2_light(self):
        assert _detect_model_name("forgelm_v2_light") == "ForgeLM V2 Light"

    def test_v2_pro(self):
        assert _detect_model_name("forgelm_v2_pro") == "ForgeLM V2 Pro"

    def test_v10_alias(self):
        assert _detect_model_name("forgelm_v10_1.2b") == "ForgeLM V2 Light"

    def test_v11_alias(self):
        assert _detect_model_name("forgelm_v11_3b_vl") == "ForgeLM V2 Pro"

    def test_lfm25(self):
        assert _detect_model_name("lfm25_tiny") == "LFM 2.5"


class TestFormatArch:
    def test_light_arch(self):
        cfg = get_config("forgelm_v2_light")
        arch = _format_arch(cfg)
        assert "16 layers" in arch
        assert "d_model=2048" in arch
        assert "GQA" in arch
        assert "QK-norm" in arch
        assert "IRI-FP4" in arch
        assert "SpectralKV" in arch

    def test_pro_arch_has_vision(self):
        cfg = get_config("forgelm_v2_pro")
        arch = _format_arch(cfg)
        assert "30 layers" in arch
        assert "vision=siglip2" in arch


class TestFormatParams:
    def test_light_params(self):
        cfg = get_config("forgelm_v2_light")
        params = _format_params(cfg)
        assert "B" in params or "M" in params

    def test_pro_params_larger(self):
        light = _format_params(get_config("forgelm_v2_light"))
        pro = _format_params(get_config("forgelm_v2_pro"))
        # pro should be larger (3B vs 1.2B)
        light_num = float(light.rstrip("BM"))
        pro_num = float(pro.rstrip("BM"))
        assert pro_num > light_num


class TestGetDefaultPrompt:
    def test_light_default(self):
        prompt = get_default_prompt_for_config("forgelm_v2_light")
        assert "ForgeLM V2 Light" in prompt
        assert "TEXT-ONLY" in prompt

    def test_pro_default(self):
        prompt = get_default_prompt_for_config("forgelm_v2_pro")
        assert "ForgeLM V2 Pro" in prompt
        assert "MULTIMODAL" in prompt

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
