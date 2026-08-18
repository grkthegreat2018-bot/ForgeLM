"""Tests for the four recommended improvements:

1. SnapKV + 4-bit KV combined cache (kv_backend.py)
2. Golden trajectory injection (GRPOTrainer + ReplayBuffer)
3. ELO-driven curriculum (elo_tracker.py + InfiniteCurriculum)
4. Fused QK-Norm + RoPE Triton kernel (fused_rope_qknorm.py)

All CPU-only, fast, no CUDA required.
"""
import os
import sys

import numpy as np
import pytest
import torch

# ── 1. SnapKV + 4-bit KV Cache ─────────────────────────────────────────────

class TestSnapKV4BitCache:
    """Test the combined SnapKV eviction + Hadamard INT4 quantization cache."""

    def test_factory_creates_snapkv_4bit(self):
        from research.inference.kv_backend import build_kv_cache, SnapKV4BitCache
        cache = build_kv_cache("snapkv_4bit")
        assert isinstance(cache, SnapKV4BitCache)

    def test_init_and_append(self):
        from research.inference.kv_backend import build_kv_cache
        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2,
                   max_seq_len=256, device="cpu", dtype=torch.float32)
        # Append a small sequence
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.append(k, v, position=0)
        assert cache.seq_len == 4

    def test_get_returns_correct_shape(self):
        from research.inference.kv_backend import build_kv_cache
        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2,
                   max_seq_len=256, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        cache.append(k, v, position=0)
        k_out, v_out = cache.get(None)
        assert k_out.shape == (1, 2, 8, 64)
        assert v_out.shape == (1, 2, 8, 64)

    def test_eviction_triggers_at_capacity(self):
        from research.inference.kv_backend import build_kv_cache
        cache = build_kv_cache("snapkv_4bit")
        # Small budget to trigger eviction quickly
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2,
                   max_seq_len=256, device="cpu", dtype=torch.float32)
        cache.obs_window = 4
        cache.budget = 8
        cache.max_capacity = 12

        # Append tokens one at a time to trigger eviction
        for i in range(20):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            cache.append(k, v, position=i)

        # After eviction, seq_len should not exceed max_capacity
        assert cache.seq_len <= cache.max_capacity

    def test_info_reports_compression(self):
        from research.inference.kv_backend import build_kv_cache
        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2,
                   max_seq_len=256, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.append(k, v, position=0)
        info = cache.info()
        assert info["type"] == "snapkv_4bit"
        assert info["bits"] == 4
        assert info["compression"] > 1.0
        assert "eviction_ratio" in info
        assert "bit_ratio" in info

    def test_clear(self):
        from research.inference.kv_backend import build_kv_cache
        cache = build_kv_cache("snapkv_4bit")
        cache.init(n_heads=8, head_dim=64, n_kv_heads=2,
                   max_seq_len=256, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.append(k, v, position=0)
        assert cache.seq_len == 4
        cache.clear()
        assert cache.seq_len == 0


# ── 2. Golden Trajectory Injection ──────────────────────────────────────────

class TestGoldenTrajectoryInjection:
    """Test GRPOTrainer replay buffer integration."""

    def test_replay_buffer_accepted_in_constructor(self):
        """GRPOTrainer should accept a replay_buffer parameter."""
        from research.self_play.grpo_trainer import GRPOTrainer, GRPOConfig
        from research.self_play.replay_buffer import ReplayBuffer

        # Create dummy model, tokenizer, ref_model
        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type("MockTok", (), {
            "__call__": lambda self, text, **kw: type("Enc", (), {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]])})()
        })()

        buf = ReplayBuffer(max_size=100)
        trainer = GRPOTrainer(model, tokenizer, ref_model,
                              device="cpu", replay_buffer=buf)
        assert trainer.replay_buffer is buf

    def test_inject_golden_replays_skips_small_buffer(self):
        """Should not inject when buffer is below min size."""
        from research.self_play.grpo_trainer import GRPOTrainer, GRPOConfig
        from research.self_play.replay_buffer import ReplayBuffer

        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type("MockTok", (), {
            "__call__": lambda self, text, **kw: type("Enc", (), {
                "input_ids": torch.tensor([[1, 2, 3]])})()
        })()

        buf = ReplayBuffer(max_size=100)
        # Add only a few items — below replay_min_buffer_size (50)
        for i in range(10):
            buf.add({"prompt": f"p{i}", "solution": f"s{i}", "quality": 1.0})

        trainer = GRPOTrainer(model, tokenizer, ref_model,
                              device="cpu", replay_buffer=buf)
        prompts = ["test prompt"]
        completions = [["comp1", "comp2"]]
        rewards = [[1.0, 0.0]]
        p, c, r, n = trainer._inject_golden_replays(prompts, completions, rewards)
        assert n == 0  # should not inject
        assert len(p) == 1  # unchanged

    def test_inject_golden_replays_adds_trajectories(self):
        """Should inject golden trajectories when buffer is large enough."""
        from research.self_play.grpo_trainer import GRPOTrainer, GRPOConfig
        from research.self_play.replay_buffer import ReplayBuffer

        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type("MockTok", (), {
            "__call__": lambda self, text, **kw: type("Enc", (), {
                "input_ids": torch.tensor([[1, 2, 3]])})()
        })()

        buf = ReplayBuffer(max_size=1000)
        # Fill buffer above min size
        for i in range(60):
            buf.add({"prompt": f"prompt_{i}", "solution": f"solution_{i}",
                     "quality": 1.0, "test_passed": True})

        config = GRPOConfig(replay_ratio=0.2, replay_min_buffer_size=50)
        trainer = GRPOTrainer(model, tokenizer, ref_model,
                              device="cpu", config=config, replay_buffer=buf)

        prompts = ["p1", "p2", "p3"]
        completions = [["c1a", "c1b"], ["c2a", "c2b"], ["c3a", "c3b"]]
        rewards = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        p, c, r, n = trainer._inject_golden_replays(prompts, completions, rewards)
        assert n > 0  # should inject some
        assert len(p) == 3 + n  # original + injected
        # Injected trajectories should have reward 1.0
        for i in range(n):
            assert r[3 + i] == [1.0, 1.0]

    def test_record_golden_trajectories(self):
        """Should record successful trajectories into replay buffer."""
        from research.self_play.grpo_trainer import GRPOTrainer, GRPOConfig
        from research.self_play.replay_buffer import ReplayBuffer

        model = torch.nn.Linear(10, 10)
        ref_model = torch.nn.Linear(10, 10)
        tokenizer = type("MockTok", (), {
            "__call__": lambda self, text, **kw: type("Enc", (), {
                "input_ids": torch.tensor([[1, 2, 3]])})()
        })()

        buf = ReplayBuffer(max_size=1000)
        trainer = GRPOTrainer(model, tokenizer, ref_model,
                              device="cpu", replay_buffer=buf)

        prompts = ["p1", "p2"]
        completions = [["good_sol", "bad_sol"], ["ok_sol", "great_sol"]]
        rewards = [[1.0, 0.0], [0.5, 1.0]]

        trainer._record_golden_trajectories(prompts, completions, rewards)

        # Should have recorded 3 successful trajectories (1.0, 1.0, and 0.5 < 0.99)
        # Actually 0.5 < 0.99 so not recorded. Only reward >= 0.99 recorded.
        # good_sol (1.0) + great_sol (1.0) = 2 recorded
        assert len(buf) == 2


# ── 3. ELO-Driven Curriculum ────────────────────────────────────────────────

class TestEloTracker:
    """Test the ELO matchmaking system."""

    def test_initial_state(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        assert elo.model_rating == 1200.0
        assert len(elo.prompt_ratings) == 0

    def test_register_prompt(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker()
        elo.register_prompt("task_1")
        assert "task_1" in elo.prompt_ratings
        assert elo.prompt_ratings["task_1"].rating == 1200.0

    def test_expected_win_prob_equal_rating(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        prob = elo.expected_win_prob(1200.0)
        assert abs(prob - 0.5) < 1e-6  # equal rating = 50%

    def test_expected_win_prob_higher_model(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1400.0)  # model is 200 points higher
        prob = elo.expected_win_prob(1200.0)
        assert prob > 0.5  # model favored

    def test_update_model_rating_success(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        elo.register_prompt("task_1", rating=1200.0)
        elo.update_model_rating("task_1", success=True)
        # Model won → rating should increase
        assert elo.model_rating > 1200.0
        # Prompt lost → rating should decrease
        assert elo.prompt_ratings["task_1"].rating < 1200.0

    def test_update_model_rating_failure(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        elo.register_prompt("task_1", rating=1200.0)
        elo.update_model_rating("task_1", success=False)
        # Model lost → rating should decrease
        assert elo.model_rating < 1200.0
        # Prompt won → rating should increase
        assert elo.prompt_ratings["task_1"].rating > 1200.0

    def test_zero_sum_update(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        elo.register_prompt("task_1", rating=1200.0)
        initial_sum = elo.model_rating + elo.prompt_ratings["task_1"].rating
        elo.update_model_rating("task_1", success=True)
        final_sum = elo.model_rating + elo.prompt_ratings["task_1"].rating
        assert abs(final_sum - initial_sum) < 1e-4  # zero-sum

    def test_select_prompts_goldilocks(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        # Register prompts at various difficulty levels
        for i, rating in enumerate([800, 1000, 1200, 1400, 1600]):
            elo.register_prompt(f"task_{i}", rating=rating)
        # Model at 1200 → should prefer tasks near 1200 (50% win prob)
        selected = elo.select_prompts(n=3)
        # The task at 1200 should be selected (closest to 50%)
        assert "task_2" in selected  # rating 1200 = exact match

    def test_select_mixed_prompts(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        for i in range(20):
            elo.register_prompt(f"task_{i}", rating=1000 + i * 50)
        selected = elo.select_mixed_prompts(n=10, exploration_ratio=0.2)
        assert len(selected) == 10

    def test_k_factor_decreases_with_attempts(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0, k_factor=32.0, k_factor_min=8.0)
        elo.register_prompt("task_1", rating=1200.0)
        # First attempt: full K-factor
        k1 = elo._effective_k("task_1")
        assert k1 == 32.0
        # After 20 attempts: minimum K-factor
        for _ in range(20):
            elo.update_model_rating("task_1", success=True)
        k20 = elo._effective_k("task_1")
        assert k20 == 8.0  # fully decayed

    def test_domain_stats(self):
        from research.self_play.elo_tracker import EloTracker
        elo = EloTracker(initial_rating=1200.0)
        for i in range(5):
            elo.register_prompt(f"task_{i}", rating=1200.0 + i * 50)
            elo.update_model_rating(f"task_{i}", success=(i % 2 == 0))
        stats = elo.domain_stats()
        assert stats["n_prompts"] == 5
        assert "model_rating" in stats
        assert "mean_success_rate" in stats


# ── 4. Fused QK-Norm + RoPE ─────────────────────────────────────────────────

class TestFusedQKNormRoPE:
    """Test the fused RMSNorm + RoPE kernel (PyTorch fallback on CPU)."""

    def test_pytorch_fallback_matches_separate_ops(self):
        """The PyTorch fallback should produce the same result as separate
        RMSNorm + RoPE ops."""
        from research.decoding.fused_rope_qknorm import fused_qk_norm_rope

        B, n_heads, T, head_dim = 2, 4, 8, 64
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.ones(head_dim)
        k_weight = torch.ones(head_dim)
        cos = torch.randn(T, head_dim)
        sin = torch.randn(T, head_dim)
        eps = 1e-6

        # Fused (will use PyTorch fallback on CPU)
        q_fused, k_fused = fused_qk_norm_rope(
            q, k, q_weight, k_weight, cos, sin, eps, use_triton=False)

        # Reference: separate ops
        q_rms = q.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        q_normed = q * q_rms * q_weight
        k_rms = k.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        k_normed = k * k_rms * k_weight

        def rotate_half(x):
            d = x.shape[-1]
            return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)

        cos_b = cos.unsqueeze(0).unsqueeze(0)
        sin_b = sin.unsqueeze(0).unsqueeze(0)
        q_ref = (q_normed * cos_b) + (rotate_half(q_normed) * sin_b)
        k_ref = (k_normed * cos_b) + (rotate_half(k_normed) * sin_b)

        assert torch.allclose(q_fused, q_ref, atol=1e-5)
        assert torch.allclose(k_fused, k_ref, atol=1e-5)

    def test_identity_weights_preserve_input_up_to_rope(self):
        """With identity RMSNorm weights (all 1.0), the norm should just
        scale by 1/rms, then RoPE is applied."""
        from research.decoding.fused_rope_qknorm import fused_qk_norm_rope

        B, n_heads, T, head_dim = 1, 2, 4, 64
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.ones(head_dim)
        k_weight = torch.ones(head_dim)
        cos = torch.ones(T, head_dim)  # identity rotation
        sin = torch.zeros(T, head_dim)
        eps = 1e-6

        q_out, k_out = fused_qk_norm_rope(
            q, k, q_weight, k_weight, cos, sin, eps, use_triton=False)

        # With cos=1, sin=0: output = normed * 1 + rotate_half(normed) * 0 = normed
        # So output should just be the RMSNormed input
        q_rms = q.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        q_expected = q * q_rms  # identity weight
        assert torch.allclose(q_out, q_expected, atol=1e-5)

    def test_non_identity_weights(self):
        """Non-identity weights should scale the normed output."""
        from research.decoding.fused_rope_qknorm import fused_qk_norm_rope

        B, n_heads, T, head_dim = 1, 2, 4, 64
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.randn(head_dim) * 0.1 + 1.0  # near-identity
        k_weight = torch.randn(head_dim) * 0.1 + 1.0
        cos = torch.ones(T, head_dim)
        sin = torch.zeros(T, head_dim)
        eps = 1e-6

        q_out, _ = fused_qk_norm_rope(
            q, k, q_weight, k_weight, cos, sin, eps, use_triton=False)

        q_rms = q.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        q_expected = q * q_rms * q_weight
        assert torch.allclose(q_out, q_expected, atol=1e-5)

    def test_single_tensor_wrapper(self):
        """fused_norm_rope_single should work for a single tensor."""
        from research.decoding.fused_rope_qknorm import fused_norm_rope_single

        B, n_heads, T, head_dim = 1, 4, 8, 64
        x = torch.randn(B, n_heads, T, head_dim)
        weight = torch.ones(head_dim)
        cos = torch.ones(T, head_dim)
        sin = torch.zeros(T, head_dim)

        out = fused_norm_rope_single(x, weight, cos, sin, use_triton=False)
        # With identity cos/sin and identity weight: just RMSNorm
        x_rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        expected = x * x_rms
        assert torch.allclose(out, expected, atol=1e-5)

    def test_head_dim_must_be_power_of_2(self):
        """Triton path requires power-of-2 head_dim. PyTorch path works
        with any head_dim."""
        from research.decoding.fused_rope_qknorm import fused_qk_norm_rope

        # head_dim=48 (not power of 2) — should work with PyTorch fallback
        B, n_heads, T, head_dim = 1, 2, 4, 48
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)
        q_weight = torch.ones(head_dim)
        k_weight = torch.ones(head_dim)
        cos = torch.ones(T, head_dim)
        sin = torch.zeros(T, head_dim)

        # Should work (uses PyTorch fallback)
        q_out, k_out = fused_qk_norm_rope(
            q, k, q_weight, k_weight, cos, sin, use_triton=False)
        assert q_out.shape == q.shape
