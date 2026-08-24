"""Tests for ForgeEngine refactoring fixes.

Validates the specific bugs fixed in the refactor commit:
1. Thread-safe checkpoint caches (_ckpt_cache_lock)
2. _clear_cuda_cache helper exists and is callable
3. _release_acceleration_resources cleans up acceleration slots
4. _active_kv_bits is initialized and tracked
5. _read_checkpoint_metadata handles missing/corrupt files gracefully
6. continue_session doesn't use 'generated_ids' in dir() hack
"""
import sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import pytest
import torch
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCheckpointCacheThreadSafety:
    """The module-level caches must be guarded by a lock."""

    def test_lock_exists(self):
        from research.inference import forge_engine
        assert hasattr(forge_engine, "_ckpt_cache_lock")
        assert isinstance(forge_engine._ckpt_cache_lock, type(threading.Lock()))

    def test_concurrent_cache_access_no_crash(self, tmp_path):
        """Multiple threads writing to _checkpoint_size_cache simultaneously."""
        from research.inference import forge_engine
        # Clear cache for clean test
        with forge_engine._ckpt_cache_lock:
            forge_engine._checkpoint_size_cache.clear()

        errors = []

        def worker(i):
            try:
                # Simulate the cache pattern from from_checkpoint
                path = f"fake_path_{i}.safetensors"
                ckpt_size = forge_engine._checkpoint_size_cache.get(path)
                if ckpt_size is None:
                    ckpt_size = 1024 * i  # fake size
                    with forge_engine._ckpt_cache_lock:
                        forge_engine._checkpoint_size_cache[path] = ckpt_size
                        while len(forge_engine._checkpoint_size_cache) > forge_engine._CKPT_CACHE_MAX:
                            forge_engine._checkpoint_size_cache.popitem(last=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # Cache should be bounded
        assert len(forge_engine._checkpoint_size_cache) <= forge_engine._CKPT_CACHE_MAX


class TestClearCudaCache:
    """_clear_cuda_cache helper should exist and be callable."""

    def test_helper_exists(self):
        from research.inference.forge_engine import ForgeEngine
        assert hasattr(ForgeEngine, "_clear_cuda_cache")

    def test_safe_on_cpu(self, forge_engine_fixture):
        """Should not crash even on CPU (no CUDA)."""
        engine = forge_engine_fixture
        # Should be a no-op on CPU
        engine._clear_cuda_cache()


class TestReleaseAccelerationResources:
    """_release_acceleration_resources should nil out all acceleration slots."""

    def test_helper_exists(self):
        from research.inference.forge_engine import ForgeEngine
        assert hasattr(ForgeEngine, "_release_acceleration_resources")

    def test_releases_all_slots(self, forge_engine_fixture):
        engine = forge_engine_fixture
        # Set fake acceleration objects
        engine._graph_runner = MagicMock()
        engine._megakernel = MagicMock()
        engine._flex_decoding = MagicMock()
        engine._chunked_prefill = MagicMock()
        engine.acceleration = "cuda_graph"

        engine._release_acceleration_resources()

        assert engine._graph_runner is None
        assert engine._megakernel is None
        assert engine._flex_decoding is None
        assert engine._chunked_prefill is None
        assert engine.acceleration is None

    def test_calls_release_method_if_present(self, forge_engine_fixture):
        engine = forge_engine_fixture
        mock_obj = MagicMock()
        mock_obj.release = MagicMock()
        engine._graph_runner = mock_obj
        engine._release_acceleration_resources()
        mock_obj.release.assert_called_once()

    def test_safe_with_none_slots(self, forge_engine_fixture):
        engine = forge_engine_fixture
        # All slots already None — should not crash
        engine._release_acceleration_resources()
        assert engine._graph_runner is None


class TestActiveKvBits:
    """_active_kv_bits should be initialized in __init__ and tracked in activation."""

    def test_initialized_in_init(self, forge_engine_fixture):
        engine = forge_engine_fixture
        assert hasattr(engine, "_active_kv_bits")
        assert isinstance(engine._active_kv_bits, int)

    def test_kv_cache_name_tracked(self, forge_engine_fixture):
        engine = forge_engine_fixture
        assert hasattr(engine, "_active_kv_cache_name")


class TestReadCheckpointMetadataGraceful:
    """_read_checkpoint_metadata should handle missing/corrupt files."""

    def test_missing_file_returns_empty(self):
        from research.inference.forge_engine import ForgeEngine
        metadata = ForgeEngine._read_checkpoint_metadata("nonexistent_file.safetensors")
        assert isinstance(metadata, dict)
        assert len(metadata) == 0  # empty dict, not crash


class TestContinueSessionNoDirHack:
    """continue_session should not use 'generated_ids' in dir() hack."""

    def test_no_dir_hack_in_source(self):
        path = Path(__file__).parent.parent.parent / "research" / "inference" / "forge_engine.py"
        source = path.read_text(encoding="utf-8")
        assert "'generated_ids' in dir()" not in source, (
            "The 'generated_ids' in dir() hack should have been removed in the refactor"
        )
        assert "'generated_ids' in dir()" not in source


# ── Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def forge_engine_fixture():
    """Minimal ForgeEngine on CPU for testing helper methods.

    Uses a mock tokenizer to avoid filesystem dependency on tokenizer location.
    """
    from research.config import get_config
    from research.model_loader import ConfigurableResearchLLM
    from research.inference.forge_engine import ForgeEngine

    cfg = get_config("lfm25_tiny")
    cfg.vocab_size = 256
    cfg.device = "cpu"
    cfg.dtype = "float32"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    # Mock tokenizer — these tests only exercise helper methods, not generation
    tok = MagicMock()
    tok.vocab_size = 256
    tok.eos_token_id = 0
    tok.bos_token_id = 0
    tok.pad_token_id = 0
    tok.encode = MagicMock(return_value=[1, 2, 3])
    tok.decode = MagicMock(return_value="test")
    tok.__call__ = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})

    engine = ForgeEngine(model, tok, device="cpu")
    yield engine
    del engine
    del model
