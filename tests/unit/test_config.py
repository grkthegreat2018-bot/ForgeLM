"""Tests for research.config — ModelConfig dataclass and get_config factory."""

import pytest

from research.config import MODEL_CONFIGS, ModelConfig, get_config


class TestModelConfigDefaults:
    """Default ModelConfig values and validation."""

    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.vocab_size == 151665
        assert cfg.d_model == 1024
        assert cfg.n_layers == 16
        assert cfg.n_heads == 16
        assert cfg.attn_type == "mla"
        assert cfg.ffn_type == "swiglu"
        assert cfg.norm_type == "layernorm"
        assert cfg.dtype == "bfloat16"

    def test_device_is_string(self):
        cfg = ModelConfig()
        assert isinstance(cfg.device, str)
        assert cfg.device in ("cuda", "cpu")

    def test_n_kv_heads_defaults_none(self):
        cfg = ModelConfig()
        assert cfg.n_kv_heads is None

    def test_intermediate_size_defaults_none(self):
        cfg = ModelConfig()
        assert cfg.intermediate_size is None


class TestModelConfigValidation:
    """__post_init__ validation logic."""

    def test_d_model_must_divide_n_heads(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            ModelConfig(d_model=100, n_heads=3)

    def test_diff_attn_requires_d_model_div_2n_heads(self):
        with pytest.raises(ValueError, match="Differential Attention"):
            ModelConfig(d_model=100, n_heads=7, attn_type="diff")

    def test_diff_attn_valid(self):
        cfg = ModelConfig(d_model=768, n_heads=12, attn_type="diff")
        assert cfg.attn_type == "diff"

    def test_standard_attn_skips_diff_check(self):
        cfg = ModelConfig(d_model=100, n_heads=4, attn_type="standard")
        assert cfg.d_model == 100


class TestModelConfigs:
    """Pre-defined MODEL_CONFIGS registry."""

    def test_configs_exist(self):
        expected = {
            "360m_mla",
            "250m_diff",
            "135m_mla",
            "tiny_test",
            "tiny_draft",
            "qwen25_coder_1.5b",
            "qwen25_coder_0.5b_svd",
            "xp_1.5b_mqa",
            "xp_1.5b_mla_moe",
            "forgelm_v1",
            "forgelm_v2",
        }
        assert expected.issubset(set(MODEL_CONFIGS.keys()))

    def test_tiny_test_is_small(self):
        cfg = MODEL_CONFIGS["tiny_test"]
        assert cfg.d_model == 256
        assert cfg.n_layers == 2
        assert cfg.n_heads == 4

    def test_qwen25_coder_1_5b_matches_architecture(self):
        cfg = MODEL_CONFIGS["qwen25_coder_1.5b"]
        assert cfg.vocab_size == 151936
        assert cfg.d_model == 1536
        assert cfg.n_layers == 28
        assert cfg.n_heads == 12
        assert cfg.n_kv_heads == 2
        assert cfg.intermediate_size == 8960
        assert cfg.attn_type == "gqa"
        assert cfg.attn_bias is True
        assert cfg.norm_type == "rmsnorm"
        assert cfg.rope_base == 1_000_000.0

    def test_forgelm_v2_has_qk_norm(self):
        cfg = MODEL_CONFIGS["forgelm_v2"]
        assert cfg.use_qk_norm is True

    def test_forgelm_v1_no_qk_norm(self):
        cfg = MODEL_CONFIGS["forgelm_v1"]
        assert cfg.use_qk_norm is False

    def test_all_configs_pass_validation(self):
        """Every pre-defined config should pass __post_init__ without error."""
        for name, cfg in MODEL_CONFIGS.items():
            assert isinstance(cfg, ModelConfig), f"{name} is not ModelConfig"
            assert cfg.d_model % cfg.n_heads == 0, f"{name}: d_model not divisible by n_heads"


class TestGetConfig:
    """get_config factory function."""

    def test_get_named_config(self):
        cfg = get_config("tiny_test")
        assert cfg.d_model == 256
        assert cfg.n_layers == 2

    def test_get_default_config(self):
        cfg = get_config(None)
        assert cfg.d_model == 1024
        assert cfg.n_layers == 16

    def test_get_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown config"):
            get_config("nonexistent_model")

    def test_get_config_with_overrides(self):
        cfg = get_config("tiny_test", d_model=512, n_layers=4)
        assert cfg.d_model == 512
        assert cfg.n_layers == 4
        # Other fields preserved
        assert cfg.n_heads == 4
        assert cfg.attn_type == "mla"

    def test_get_config_override_creates_new_instance(self):
        original = get_config("tiny_test")
        modified = get_config("tiny_test", d_model=512)
        assert original.d_model == 256  # original unchanged
        assert modified.d_model == 512

    def test_get_config_override_validation(self):
        """Overrides should still trigger __post_init__ validation."""
        with pytest.raises(ValueError, match="divisible by n_heads"):
            get_config("tiny_test", d_model=100, n_heads=3)
