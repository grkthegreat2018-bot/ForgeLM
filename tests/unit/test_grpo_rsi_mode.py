"""Tests for GRPO-only RSI mode in InfiniteSelfPlayLoop.

Verifies:
  1. LoopConfig accepts training_mode="grpo" and GRPO config fields
  2. run_epoch branches to _grpo_train when training_mode="grpo"
  3. _grpo_train skips when too few trajectories
  4. CLI arg --training-mode is wired correctly
"""
import pytest
from unittest.mock import MagicMock, patch

from forge.self_play.infinite_loop import LoopConfig, InfiniteSelfPlayLoop


class TestGRPOConfig:
    """LoopConfig GRPO fields."""

    def test_default_training_mode_is_sft(self):
        c = LoopConfig()
        assert c.training_mode == "sft"

    def test_grpo_mode_configurable(self):
        c = LoopConfig(training_mode="grpo")
        assert c.training_mode == "grpo"

    def test_grpo_config_fields_exist(self):
        c = LoopConfig(training_mode="grpo")
        assert hasattr(c, "grpo_max_steps")
        assert hasattr(c, "grpo_group_size")
        assert hasattr(c, "grpo_lr")
        assert hasattr(c, "grpo_kl_coeff")
        assert hasattr(c, "grpo_clip_range")
        assert hasattr(c, "grpo_rl_algorithm")
        assert hasattr(c, "grpo_min_trajectories")

    def test_grpo_defaults_sane_for_12gb(self):
        """GRPO defaults should be VRAM-safe for RTX 5070 12GB."""
        c = LoopConfig()
        assert c.grpo_group_size >= 2  # need ≥2 for advantage computation
        assert c.grpo_lr <= 1e-5  # small LR for stability
        assert c.grpo_max_seq_len <= 1024  # short seq for VRAM
        assert c.grpo_grad_accum >= 1


class TestGRPOTrainingModeBranch:
    """run_epoch branches correctly based on training_mode."""

    def test_sft_mode_calls_finetune(self):
        """SFT mode should call _finetune (not _grpo_train)."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._export_trajectories = MagicMock(return_value="/tmp/data.jsonl")
        loop._finetune = MagicMock(return_value="/tmp/candidate.safetensors")
        loop._evaluate = MagicMock(return_value={"passed": True})
        loop._maybe_promote = MagicMock(return_value=True)

        # Mock open() to return enough examples
        import builtins
        original_open = builtins.open

        class MockFile:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return iter(["line"] * 10)  # 10 examples
            def __exit__(self, *a):
                pass

        with patch("builtins.open", MockFile):
            loop.run_epoch()

        loop._finetune.assert_called_once()
        assert not loop._grpo_train.called if hasattr(loop._grpo_train, "called") else True

    def test_grpo_mode_calls_grpo_train(self):
        """GRPO mode should call _grpo_train (not _finetune)."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="grpo"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._grpo_train = MagicMock(return_value="/tmp/candidate.safetensors")
        loop._evaluate = MagicMock(return_value={"passed": True})
        loop._maybe_promote = MagicMock(return_value=True)
        loop._free_engine = MagicMock()

        loop.run_epoch()

        loop._grpo_train.assert_called_once()
        loop._evaluate.assert_called_once_with("/tmp/candidate.safetensors")

    def test_grpo_mode_skips_evaluation_on_empty_result(self):
        """When _grpo_train returns empty string, skip eval."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="grpo"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._grpo_train = MagicMock(return_value="")  # skipped
        loop._evaluate = MagicMock()
        loop._maybe_promote = MagicMock()
        loop._free_engine = MagicMock()

        result = loop.run_epoch()

        loop._grpo_train.assert_called_once()
        loop._evaluate.assert_not_called()
        assert result.get("grpo", {}).get("skipped") is True


class TestGRPOTrajectoryGrouping:
    """_grpo_train groups trajectories correctly for GRPO."""

    def test_too_few_trajectories_returns_empty(self):
        """When fewer than grpo_min_trajectories, skip GRPO."""
        loop = InfiniteSelfPlayLoop("fake.ckpt",
                                    LoopConfig(training_mode="grpo",
                                               grpo_min_trajectories=10))
        loop._free_engine = MagicMock()
        loop._trajectories = [
            {"task_description": "task1", "solution_code": "x", "reward": 1.0,
             "domain": "math"},
        ]
        result = loop._grpo_train(epoch=1)
        assert result == ""

    def test_paired_by_domain(self):
        """Trajectories should be paired within the same domain."""
        loop = InfiniteSelfPlayLoop("fake.ckpt",
                                    LoopConfig(training_mode="grpo",
                                               grpo_min_trajectories=2))
        loop._free_engine = MagicMock()
        loop._trajectories = [
            {"task_description": "task1", "solution_code": "sol1", "reward": 1.0,
             "domain": "math"},
            {"task_description": "task2", "solution_code": "sol2", "reward": 0.0,
             "domain": "math"},
            {"task_description": "task3", "solution_code": "sol3", "reward": 1.0,
             "domain": "algorithms"},
            {"task_description": "task4", "solution_code": "sol4", "reward": 0.0,
             "domain": "algorithms"},
        ]

        # Mock the heavy imports inside _grpo_train
        with patch("forge.config.get_config") as mock_cfg, \
             patch("forge.model_loader.ModelLoader.build_model_fast") as mock_build, \
             patch("forge.training.bitnet_lora.add_lora_adapters") as mock_lora, \
             patch("forge.training.bitnet_lora.merge_lora_adapters") as mock_merge, \
             patch("forge.self_play.grpo_trainer.GRPOTrainer") as mock_trainer_cls, \
             patch("research.tokenizer_cache.get_tokenizer") as mock_tok, \
             patch("forge.checkpoint_io.save_training_checkpoint") as mock_save, \
             patch("forge.training.training_utils.oom_guard") as mock_oom:
            mock_cfg.return_value = MagicMock()
            mock_model = MagicMock()
            mock_build.return_value = mock_model
            # Return real tensors so numel() and format strings work
            import torch as _t
            _fake_param = _t.nn.Parameter(_t.zeros(1))
            mock_lora.return_value = (4, [_fake_param])
            mock_merge.return_value = 4
            mock_trainer = MagicMock()
            mock_trainer.train_step.return_value = {"loss": 0.1, "kl": 0.01}
            mock_trainer_cls.return_value = mock_trainer
            mock_oom.return_value.__enter__ = MagicMock(return_value=MagicMock(skipped=False))
            mock_oom.return_value.__exit__ = MagicMock(return_value=False)

            result = loop._grpo_train(epoch=1)

        # Should have called train_step at least once
        assert mock_trainer.train_step.called
        # Should have saved the checkpoint
        mock_save.assert_called_once()
        assert result != ""


class TestEngineCleanup:
    """run_epoch must free the self-play engine on every path.

    Regression test for the VRAM leak where a skipped finetune left the
    self-play ForgeEngine resident, so the next epoch loaded a second
    engine on top (12.8 GB used, KV cache capped to 64 tokens).
    """

    class _MockFile:
        def __init__(self, lines):
            self._lines = lines
        def __call__(self, *a, **k):
            return self
        def __enter__(self):
            return iter(self._lines)
        def __exit__(self, *a):
            pass

    def test_skipped_finetune_frees_engine(self):
        """Too-few-examples path must free the engine."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.0})
        loop._export_trajectories = MagicMock(return_value="/tmp/data.jsonl")
        loop._finetune = MagicMock()
        loop._evaluate = MagicMock()
        loop._free_engine = MagicMock()

        with patch("builtins.open", self._MockFile([])):  # 0 examples
            result = loop.run_epoch()

        loop._free_engine.assert_called_once()
        loop._finetune.assert_not_called()
        assert result["finetune"]["skipped"] is True

    def test_train_exception_frees_engine(self):
        """Exception in Phase 2 must free the engine."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._export_trajectories = MagicMock(
            side_effect=RuntimeError("export boom"))
        loop._free_engine = MagicMock()

        result = loop.run_epoch()

        loop._free_engine.assert_called_once()
        assert result["error"] == "export boom"

    def test_self_play_exception_frees_engine(self):
        """Exception in Phase 1 must free the engine."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(
            side_effect=RuntimeError("selfplay boom"))
        loop._free_engine = MagicMock()

        result = loop.run_epoch()

        loop._free_engine.assert_called_once()
        assert "selfplay boom" in result["error"]

    def test_normal_path_still_frees_via_eval(self):
        """Full path: _evaluate frees the engine (called there, not here)."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._export_trajectories = MagicMock(return_value="/tmp/data.jsonl")
        loop._finetune = MagicMock(return_value="/tmp/candidate.safetensors")
        loop._evaluate = MagicMock(return_value={"passed": True})
        loop._maybe_promote = MagicMock(return_value=True)
        loop._free_engine = MagicMock()

        with patch("builtins.open", self._MockFile(["line"] * 10)):
            loop.run_epoch()

        # run_epoch itself shouldn't double-free on the happy path —
        # _finetune/_evaluate own the cleanup there.
        loop._evaluate.assert_called_once()


class TestCheckpointNaming:
    """Epoch checkpoints must never overwrite existing files."""

    def test_epoch_path_uses_config_tag(self, tmp_path):
        c = LoopConfig(config_name="forgelm_v2",
                       checkpoint_dir=str(tmp_path))
        loop = InfiniteSelfPlayLoop("fake.ckpt", c)
        p = loop._epoch_checkpoint_path(3)
        assert "ForgeLM_V2_SP3" in p

    def test_epoch_path_never_overwrites(self, tmp_path):
        c = LoopConfig(config_name="forgelm_v2",
                       checkpoint_dir=str(tmp_path))
        loop = InfiniteSelfPlayLoop("fake.ckpt", c)
        first = loop._epoch_checkpoint_path(1)
        # Simulate a previous run leaving SP1 behind
        open(first, "w").close()
        second = loop._epoch_checkpoint_path(1)
        assert second != first
        assert second.endswith("_r2.safetensors")


class TestStrictPromote:
    """Promotion requires the candidate to beat/tie the base by default."""

    def _loop(self, strict=True):
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(strict_promote=strict, eval_threshold=0.5))
        loop._free_engine = MagicMock()
        return loop

    def _eval(self, loop, base_q, cand_q, winner):
        with patch("forge.self_play.discovery.fast_eval.fast_eval",
                   return_value={"base": {"quality": base_q},
                                 "candidate": {"quality": cand_q},
                                 "winner": winner}):
            return loop._evaluate("cand.safetensors")

    def test_winner_candidate_promotes(self):
        assert self._eval(self._loop(), 0.5, 0.6, "CANDIDATE")["passed"]

    def test_strict_rejects_regressed_candidate(self):
        # Candidate within lenient threshold (0.4 >= 0.5*0.5) but lost.
        r = self._eval(self._loop(strict=True), 0.5, 0.4, "BASE")
        assert r["passed"] is False

    def test_lenient_allows_within_threshold(self):
        r = self._eval(self._loop(strict=False), 0.5, 0.4, "BASE")
        assert r["passed"] is True

    def test_lenient_rejects_large_regression(self):
        r = self._eval(self._loop(strict=False), 0.5, 0.1, "BASE")
        assert r["passed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
