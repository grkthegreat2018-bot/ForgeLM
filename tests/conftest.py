"""Shared pytest fixtures for ForgeAI test suite."""

import pytest
import torch

from research.config import ModelConfig, get_config

# Small vocab for CPU tests — avoids 151665×d_model embedding spike.
_CPU_VOCAB = 256


@pytest.fixture
def tiny_config():
    """Minimal config for fast CPU-only testing. Uses small vocab to avoid CPU spike."""
    cfg = get_config("tiny_test")
    return ModelConfig(**{**cfg.__dict__, "device": "cpu", "vocab_size": _CPU_VOCAB})


@pytest.fixture
def tiny_config_cpu(tiny_config):
    """Tiny config forced to CPU (alias for clarity)."""
    return tiny_config


@pytest.fixture
def tiny_config_gpu():
    """Tiny config for GPU tests. Uses small vocab. Mark with @pytest.mark.gpu."""
    cfg = get_config("tiny_test")
    return ModelConfig(**{**cfg.__dict__, "device": "cuda", "vocab_size": _CPU_VOCAB})


@pytest.fixture
def gpu_available():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Clean temporary directory for checkpoint I/O tests."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture
def small_state_dict():
    """Small state dict with mixed tensors and metadata for checkpoint tests."""
    return {
        "weight_a": torch.randn(4, 8, dtype=torch.float32),
        "weight_b": torch.randn(16, dtype=torch.float32),
        "step": 100,
        "config": {"lr": 1e-4, "epochs": 10},
    }


@pytest.fixture
def bf16_state_dict():
    """State dict with bf16 tensors (common in ForgeAI)."""
    return {
        "weight_a": torch.randn(4, 8, dtype=torch.bfloat16),
        "weight_b": torch.randn(16, dtype=torch.bfloat16),
        "step": 50,
    }
