"""Tests for research.config — ModelConfig dataclass and get_config factory."""

import pytest

from research.config import MODEL_CONFIGS, ModelConfig, get_config


class TestModelConfigDefaults:
    """Default ModelConfig values and validation."""

    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.vocab_size == 65536
        assert cfg.d_model == 2048
        assert cfg.n_layers == 16
        assert cfg.n_heads == 32
        assert cfg.attn_type == "gqa"
        assert cfg.ffn_type == "swiglu"
        assert cfg.norm_type == "rmsnorm"
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


class TestModelConfigs:
    """Pre-defined MODEL_CONFIGS registry."""

    def test_configs_exist(self):
        expected = {
            "lfm25_1.2b",
            "lfm25_tiny",
        }
        assert expected.issubset(set(MODEL_CONFIGS.keys()))

    def test_lfm25_tiny_is_small(self):
        cfg = MODEL_CONFIGS["lfm25_tiny"]
        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4

    def test_lfm25_1_2b_matches_architecture(self):
        cfg = MODEL_CONFIGS["lfm25_1.2b"]
        assert cfg.vocab_size == 65536
        assert cfg.d_model == 2048
        assert cfg.n_layers == 16
        assert cfg.n_heads == 32
        assert cfg.n_kv_heads == 8
        assert cfg.intermediate_size == 8192
        assert cfg.attn_type == "gqa"  # reference LFM2.5 port (no keys)
        assert cfg.attn_bias is False
        assert cfg.norm_type == "rmsnorm"
        assert cfg.rope_base == 1_000_000.0

    def test_lfm25_1_2b_has_qk_norm(self):
        cfg = MODEL_CONFIGS["lfm25_1.2b"]
        assert cfg.use_qk_norm is True

    def test_lfm25_1_2b_has_conv_layers(self):
        cfg = MODEL_CONFIGS["lfm25_1.2b"]
        assert cfg.layer_types is not None
        assert cfg.layer_types.count("conv") == 10
        assert cfg.layer_types.count("attention") == 6

    def test_forgelm_v3_is_labeled_full_stack(self):
        """ForgeLM V3 = LFM2.5 base + all 2025/2026 architecture keys."""
        cfg = MODEL_CONFIGS["forgelm_v3"]
        assert cfg.d_model == 2048 and cfg.n_layers == 16
        assert cfg.attn_type == "diff"          # differential attention
        assert cfg.use_bitnet is True           # BitNet b1.58 QAT
        assert cfg.use_titan_memory is True     # TITAN neural memory
        assert cfg.titan_memory_rank == 64      # low-rank (VRAM-friendly)
        assert cfg.use_mod is True              # Mixture-of-Depths
        assert cfg.mod_keep_fraction == 1.0     # lossless start

    def test_forgelm_v3_same_base_arch(self):
        """Same LFM2.5 skeleton (heads, dims, layers) as the reference port."""
        a = MODEL_CONFIGS["lfm25_1.2b"]
        b = MODEL_CONFIGS["forgelm_v3"]
        for f in ("d_model", "n_layers", "n_heads", "n_kv_heads",
                  "intermediate_size", "vocab_size", "rope_base",
                  "max_seq_len", "use_qk_norm"):
            assert getattr(a, f) == getattr(b, f), f
        assert a.layer_types == b.layer_types

    def test_all_configs_pass_validation(self):
        """Every pre-defined config should pass __post_init__ without error."""
        for name, cfg in MODEL_CONFIGS.items():
            assert isinstance(cfg, ModelConfig), f"{name} is not ModelConfig"
            assert cfg.d_model % cfg.n_heads == 0, f"{name}: d_model not divisible by n_heads"


class TestGetConfig:
    """get_config factory function."""

    def test_get_named_config(self):
        cfg = get_config("lfm25_tiny")
        assert cfg.d_model == 128
        assert cfg.n_layers == 4

    def test_get_default_config(self):
        cfg = get_config(None)
        assert cfg.d_model == 2048
        assert cfg.n_layers == 16

    def test_get_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown config"):
            get_config("nonexistent_model")

    def test_get_config_with_overrides(self):
        cfg = get_config("lfm25_tiny", d_model=256, n_layers=8)
        assert cfg.d_model == 256
        assert cfg.n_layers == 8
        # Other fields preserved
        assert cfg.n_heads == 4
        assert cfg.attn_type == "gqa"

    def test_get_config_override_creates_new_instance(self):
        original = get_config("lfm25_tiny")
        modified = get_config("lfm25_tiny", d_model=256)
        assert original.d_model == 128  # original unchanged
        assert modified.d_model == 256

    def test_get_config_override_validation(self):
        """Overrides should still trigger __post_init__ validation."""
        with pytest.raises(ValueError, match="divisible by n_heads"):
            get_config("lfm25_tiny", d_model=100, n_heads=3)
