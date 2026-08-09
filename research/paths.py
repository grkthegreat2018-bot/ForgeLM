"""Central path management — replaces 17 hardcoded Windows paths.

All paths are relative to the project root (the parent of `research/`).
This makes the codebase portable across machines and operating systems.
"""
from pathlib import Path

# Project root = parent of research/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common directories used across the codebase.
DATA_DIR = PROJECT_ROOT / "research" / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "research" / "checkpoints"
TMP_DIR = PROJECT_ROOT / ".devin" / "tmp"
HF_CACHE_DIR = PROJECT_ROOT / ".devin" / "hf_cache"
BENCH_CACHE_DIR = PROJECT_ROOT / ".devin" / "bench_cache"
TORCH_CACHE_DIR = PROJECT_ROOT / ".devin" / "torch_cache"

# Specific subdirectories.
CURRICULUM_DIR = DATA_DIR / "curriculum"
EXPERT_TRAINING_DIR = DATA_DIR / "expert_training"
LCB_EVAL_DIR = DATA_DIR / "lcb_eval"
REASONING_BENCH_DIR = DATA_DIR / "reasoning_bench"
FORGELM_V4_DIR = CHECKPOINTS_DIR / "forgelm_v4"
AIRMOE_MODULES_DIR = CHECKPOINTS_DIR / "airmoe_modules"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_str(path: Path) -> str:
    """Convert Path to forward-slash string (for cross-platform compatibility)."""
    return str(path).replace("\\", "/")
