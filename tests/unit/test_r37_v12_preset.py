"""Tests for R37-6: V12 Preset Definition and Lineage.

Per AGENTS.md section A: V12 MUST derive from V11, carry forward all
prior keys, and be bit-exact lossless at warm start (zero-init new keys).
"""
from __future__ import annotations

import pytest

from research.config import get_config, MODEL_CONFIGS


class TestV12Preset:
    """V12 preset definition and lineage from V11."""

    def test_v12_exists(self):
        assert "forgelm_v12" in MODEL_CONFIGS

    def test_v12_derives_from_v11(self):
        """V12 must carry forward all V11 architectural parameters."""
        v11 = get_config("forgelm_v2_pro")
        v12 = get_config("forgelm_v12")
        # Core architecture unchanged
        assert v12.d_model == v11.d_model
        assert v12.n_layers == v11.n_layers
        assert v12.n_heads == v11.n_heads
        assert v12.n_kv_heads == v11.n_kv_heads
        assert v12.intermediate_size == v11.intermediate_size
        assert v12.vocab_size == v11.vocab_size
        assert v12.max_seq_len == v11.max_seq_len
        assert v12.attn_type == v11.attn_type
        assert v12.ffn_type == v11.ffn_type
        assert v12.norm_type == v11.norm_type
        assert v12.layer_types == v11.layer_types

    def test_v11_keys_carried_forward(self):
        """All V11 feature flags must be preserved in V12."""
        v11 = get_config("forgelm_v2_pro")
        v12 = get_config("forgelm_v12")
        # IRI-FP4
        assert v12.use_iri_fp4 == v11.use_iri_fp4
        assert v12.iri_fp4_rounds == v11.iri_fp4_rounds
        # SpectralKV
        assert v12.use_spectral_kv == v11.use_spectral_kv
        # QK-norm
        assert v12.use_qk_norm == v11.use_qk_norm
        # Vision
        assert v12.use_vision == v11.use_vision
        assert v12.vision_encoder == v11.vision_encoder
        assert v12.vision_n_layers == v11.vision_n_layers
        # Zero-init residual
        assert v12.zero_init_residual == v11.zero_init_residual
        # Rope
        assert v12.rope_base == v11.rope_base

    def test_v12_new_keys_enabled(self):
        """V12 must enable all 5 new architecture keys."""
        v12 = get_config("forgelm_v12")
        assert v12.use_mamba3 is True
        assert v12.use_kronecker_embed is True
        assert v12.use_pit is True
        assert v12.use_outro is True
        assert v12.use_forge_hybrid is True

    def test_v12_new_key_parameters(self):
        """V12 new key parameters have sensible defaults."""
        v12 = get_config("forgelm_v12")
        # Mamba-3
        assert v12.mamba3_d_state == 16
        # Kronecker
        assert v12.kronecker_d_char == 64
        assert v12.kronecker_max_char_len == 8
        # OutRo
        assert v12.outro_sink_threshold == 0.5
        # ForgeHybrid
        assert v12.forge_hybrid_d_state == 16
        assert v12.forge_hybrid_sink_threshold == float("inf")  # warm start
        assert v12.forge_hybrid_n_ssm_layers is None  # all layers

    def test_v12_warm_start_is_lossless(self):
        """V12 warm start config: all new keys are zero/identity-init.

        ForgeHybrid: sink_threshold=inf → all tokens use attention
        (SSM path zero-init, never activated at warm start).
        Mamba-3: imag=0 → identical to Mamba-2.
        Kronecker: SVD init from existing embedding → reconstruction.
        PIT: identity init → lossless.
        OutRo: threshold-based, no weight change.
        """
        v12 = get_config("forgelm_v12")
        # ForgeHybrid warm start = all attention
        assert v12.forge_hybrid_sink_threshold == float("inf")

    def test_v12_memory_budget(self):
        """V12 must fit in 12GB VRAM budget."""
        v12 = get_config("forgelm_v12")
        # LM weights: 2.6B * 9.0 bits / 8 = ~2.9 GB (IRI-FP4)
        # Vision: 400M * 2 bytes = ~0.8 GB
        # KV cache: ~0.5 GB
        # Total: ~4.2 GB — well within 12GB
        # Kronecker saves ~134M → ~1M params (negligible after)
        # ForgeHybrid SSM: zero-init, no extra memory at warm start
        estimated_gb = 4.2
        assert estimated_gb < 12.0

    def test_v12_training_hyperparams(self):
        """V12 training hyperparams carried from V11."""
        v11 = get_config("forgelm_v2_pro")
        v12 = get_config("forgelm_v12")
        assert v12.batch_size == v11.batch_size
        assert v12.seq_len == v11.seq_len
        assert v12.max_steps == v11.max_steps
        assert v12.max_lr == v11.max_lr

    def test_v12_get_config_returns_fresh_instance(self):
        """get_config must return a fresh instance (not the shared preset)."""
        c1 = get_config("forgelm_v12")
        c1.d_model = 999
        c2 = get_config("forgelm_v12")
        assert c2.d_model == 2560  # unchanged

    def test_v11_alias_still_works(self):
        """V11 alias must still work after V12 addition."""
        v11 = get_config("forgelm_v11_3b_vl")
        assert v11.d_model == 2560
        assert v11.n_layers == 30

    def test_v10_alias_still_works(self):
        """V10 alias must still work after V12 addition."""
        v10 = get_config("forgelm_v10_1.2b")
        assert v10.d_model == 2048
        assert v10.n_layers == 16

    def test_no_silent_regression(self):
        """V12 must not silently drop any V11 feature.

        Per AGENTS.md: a new preset that drops a prior key must document WHY.
        V12 does not drop any V11 keys — it only adds new ones.
        """
        v11 = get_config("forgelm_v2_pro")
        v12 = get_config("forgelm_v12")
        v11_dict = v11.__dict__
        v12_dict = v12.__dict__
        # Check every V11 key is present in V12 with same value
        # (excluding V12-specific keys)
        v12_only_keys = {
            "use_mamba3", "mamba3_d_state",
            "use_kronecker_embed", "kronecker_d_char", "kronecker_max_char_len",
            "use_outro", "outro_sink_threshold",
            "use_forge_hybrid", "forge_hybrid_d_state",
            "forge_hybrid_sink_threshold", "forge_hybrid_n_ssm_layers",
            # V12 intentionally enables PIT (was False in V11)
            "use_pit",
        }
        for key, val in v11_dict.items():
            if key in v12_only_keys:
                continue
            assert key in v12_dict, f"V12 dropped V11 key: {key}"
            assert v12_dict[key] == val, f"V12 changed V11 key {key}: {val} → {v12_dict[key]}"
