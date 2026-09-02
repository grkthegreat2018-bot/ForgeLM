"""Unit tests for ForgeLM V11 config and vision modules.

Tests V11 config preset, vision tower, projector, and connector.
Uses CPU-only small configs for fast testing.
"""
import pytest
import torch

from research.config import ModelConfig, get_config


# ── V11 config ──────────────────────────────────────────────────────────

class TestV11Config:
    def test_v11_exists(self):
        c = get_config("forgelm_v2_pro")
        assert c is not None

    def test_v11_carries_v10_keys(self):
        c = get_config("forgelm_v2_pro")
        # V10 carried keys
        assert c.use_iri_fp4 is True
        assert c.iri_fp4_rounds == 2
        assert c.use_spectral_kv is True
        assert c.use_qk_norm is True
        assert c.zero_init_residual is True
        assert c.attn_type == "gqa"
        assert c.ffn_type == "swiglu"
        assert c.norm_type == "rmsnorm"
        assert c.rope_base == 1_000_000.0

    def test_v11_vision_keys(self):
        c = get_config("forgelm_v2_pro")
        assert c.use_vision is True
        assert c.vision_encoder == "siglip2"
        assert c.vision_hidden_size == 1152
        assert c.vision_image_size == 384
        assert c.vision_patch_size == 14
        assert c.vision_n_layers == 27
        assert c.vision_n_heads == 16
        assert c.vision_projector_dim == 2560  # matches d_model
        assert c.vision_n_queries == 128

    def test_v11_architecture(self):
        c = get_config("forgelm_v2_pro")
        assert c.vocab_size == 131072  # 128K
        assert c.d_model == 2560
        assert c.n_layers == 30
        assert c.n_heads == 40
        assert c.n_kv_heads == 8
        assert c.max_seq_len == 131072  # 128K context

    def test_v11_layer_types_count(self):
        c = get_config("forgelm_v2_pro")
        assert len(c.layer_types) == 30

    def test_v11_d_model_divisible_by_heads(self):
        c = get_config("forgelm_v2_pro")
        assert c.d_model % c.n_heads == 0

    def test_v11_heads_divisible_by_kv_heads(self):
        c = get_config("forgelm_v2_pro")
        assert c.n_heads % c.n_kv_heads == 0

    def test_v11_projector_dim_matches_d_model(self):
        c = get_config("forgelm_v2_pro")
        assert c.vision_projector_dim == c.d_model

    def test_v11_n_patches(self):
        """SigLIP2 384/14 = 27.4 → 27×27 = 729 patches."""
        c = get_config("forgelm_v2_pro")
        n_patches = (c.vision_image_size // c.vision_patch_size) ** 2
        assert n_patches == 729


# ── Vision module (small config for CPU testing) ────────────────────────

class TestVisionModules:
    @pytest.fixture()
    def small_vision_config(self):
        return ModelConfig(
            vocab_size=256, d_model=128, n_layers=4, n_heads=4,
            n_kv_heads=2, intermediate_size=256,
            use_vision=True,
            vision_encoder="siglip2",
            vision_hidden_size=64,
            vision_image_size=28,
            vision_patch_size=14,
            vision_n_layers=2,
            vision_n_heads=4,
            vision_intermediate_size=128,
            vision_projector_dim=128,
            vision_projector_type="mlp",
            vision_n_queries=8,
        )

    def test_patch_embed(self, small_vision_config):
        from research.vision import PatchEmbed
        c = small_vision_config
        pe = PatchEmbed(c.vision_image_size, c.vision_patch_size,
                        3, c.vision_hidden_size)
        images = torch.randn(2, 3, c.vision_image_size, c.vision_image_size)
        out = pe(images)
        n_patches = (c.vision_image_size // c.vision_patch_size) ** 2
        assert out.shape == (2, n_patches, c.vision_hidden_size)

    def test_siglip2_tower(self, small_vision_config):
        from research.vision import SigLIP2VisionTower
        c = small_vision_config
        tower = SigLIP2VisionTower(
            hidden_size=c.vision_hidden_size,
            image_size=c.vision_image_size,
            patch_size=c.vision_patch_size,
            n_layers=c.vision_n_layers,
            n_heads=c.vision_n_heads,
            intermediate_size=c.vision_intermediate_size,
        )
        images = torch.randn(1, 3, c.vision_image_size, c.vision_image_size)
        out = tower(images)
        n_patches = (c.vision_image_size // c.vision_patch_size) ** 2
        assert out.shape == (1, 1 + n_patches, c.vision_hidden_size)

    def test_vision_projector_mlp(self, small_vision_config):
        from research.vision import VisionProjector
        c = small_vision_config
        proj = VisionProjector(
            vision_hidden_size=c.vision_hidden_size,
            lm_hidden_size=c.vision_projector_dim,
            projector_type="mlp",
        )
        x = torch.randn(2, 10, c.vision_hidden_size)
        out = proj(x)
        assert out.shape == (2, 10, c.vision_projector_dim)

    def test_vision_projector_linear(self, small_vision_config):
        from research.vision import VisionProjector
        c = small_vision_config
        proj = VisionProjector(
            vision_hidden_size=c.vision_hidden_size,
            lm_hidden_size=c.vision_projector_dim,
            projector_type="linear",
        )
        x = torch.randn(2, 10, c.vision_hidden_size)
        out = proj(x)
        assert out.shape == (2, 10, c.vision_projector_dim)

    def test_query_pooler(self, small_vision_config):
        from research.vision import QueryPooler
        c = small_vision_config
        pooler = QueryPooler(
            hidden_size=c.vision_hidden_size,
            n_queries=c.vision_n_queries,
            n_heads=4,
        )
        x = torch.randn(2, 50, c.vision_hidden_size)  # 50 visual tokens
        out = pooler(x)
        assert out.shape == (2, c.vision_n_queries, c.vision_hidden_size)

    def test_vision_language_connector(self, small_vision_config):
        from research.vision import VisionLanguageConnector
        c = small_vision_config
        connector = VisionLanguageConnector(c)
        images = torch.randn(1, 3, c.vision_image_size, c.vision_image_size)
        out = connector(images)
        assert out.shape == (1, c.vision_n_queries, c.vision_projector_dim)

    def test_freeze_tower(self, small_vision_config):
        from research.vision import VisionLanguageConnector
        c = small_vision_config
        connector = VisionLanguageConnector(c)
        connector.freeze_tower()
        for p in connector.tower.parameters():
            assert not p.requires_grad

    def test_param_count(self, small_vision_config):
        from research.vision import VisionLanguageConnector
        c = small_vision_config
        connector = VisionLanguageConnector(c)
        counts = connector.param_count()
        assert counts["tower"] > 0
        assert counts["pooler"] > 0
        assert counts["projector"] > 0
        assert counts["total"] == counts["tower"] + counts["pooler"] + counts["projector"]

    def test_preprocess_image(self):
        from research.vision import preprocess_image
        # uint8 image
        img = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)
        out = preprocess_image(img, image_size=384)
        assert out.shape == (1, 3, 384, 384)
        assert out.dtype == torch.float32

    def test_preprocess_image_batch(self):
        from research.vision import preprocess_image
        img = torch.randn(2, 3, 50, 50)
        out = preprocess_image(img, image_size=28)
        assert out.shape == (2, 3, 28, 28)


# ── Port script helpers ─────────────────────────────────────────────────

class TestPortHelpers:
    def test_expand_embedding(self):
        from research.vision.port_v10_to_v11 import expand_embedding
        old = torch.randn(100, 64)
        new = expand_embedding(old, 200, 100)
        assert new.shape == (200, 64)
        # first 100 rows should be copied
        assert torch.equal(new[:100], old)
        # last 100 rows should be zero
        assert torch.all(new[100:] == 0)

    def test_expand_d_model_1d(self):
        from research.vision.port_v10_to_v11 import expand_d_model
        old = torch.randn(128)
        new = expand_d_model(old, 256, 128)
        assert new.shape == (256,)
        assert torch.equal(new[:128], old)
        assert torch.all(new[128:] == 0)

    def test_expand_d_model_2d_square(self):
        from research.vision.port_v10_to_v11 import expand_d_model
        old = torch.randn(128, 128)
        new = expand_d_model(old, 256, 128)
        assert new.shape == (256, 256)
        assert torch.equal(new[:128, :128], old)

    def test_expand_d_model_2d_input(self):
        from research.vision.port_v10_to_v11 import expand_d_model
        old = torch.randn(512, 128)  # (out, in=d_model)
        new = expand_d_model(old, 256, 128)
        assert new.shape == (512, 256)
        assert torch.equal(new[:, :128], old)

    def test_expand_d_model_2d_output(self):
        from research.vision.port_v10_to_v11 import expand_d_model
        old = torch.randn(128, 512)  # (out=d_model, in)
        new = expand_d_model(old, 256, 128)
        assert new.shape == (256, 512)
        assert torch.equal(new[:128, :], old)
