"""Central path management — replaces hardcoded Windows paths.

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
LIBRARY_DIR = DATA_DIR / "library"

# Specific subdirectories.
CURRICULUM_DIR = DATA_DIR / "curriculum"
EXPERT_TRAINING_DIR = DATA_DIR / "expert_training"
LCB_EVAL_DIR = DATA_DIR / "lcb_eval"
REASONING_BENCH_DIR = DATA_DIR / "reasoning_bench"
AIRMOE_MODULES_DIR = CHECKPOINTS_DIR / "airmoe_modules"

# Tokenizer (shared by all ForgeLM models — Qwen-style, from LFM2.5).
LFM25_HF_DIR = CHECKPOINTS_DIR / "lfm25_tokenizer"

# ForgeLM V10-1.2B: the sole base model (lossless LFM2.5 port + V10 features).
# All tests and training use this as the default checkpoint.
V10_CHECKPOINT = CHECKPOINTS_DIR / "ForgeLM_V10_1.2B.safetensors"

# Backward-compatible aliases — V7/V9 checkpoints were deleted; all point to V10.
LFM25_CHECKPOINT = V10_CHECKPOINT
V9_CHECKPOINT = V10_CHECKPOINT

# LM Studio model paths (for GGUF inference / data generation).
LMSTUDIO_MODELS_ROOT = Path("D:/LMstudio/Models/lmstudio-community")
LMSTUDIO_GGUF = LMSTUDIO_MODELS_ROOT / "LFM2.5-1.2B-Instruct-GGUF" / "LFM2.5-1.2B-Instruct-Q8_0.gguf"
LMSTUDIO_API = "http://localhost:1234/v1"

# Additional artifacts used by scripts/.
EXPERTS_DIR = EXPERT_TRAINING_DIR / "experts"
HF_DATASETS_DIR = EXPERT_TRAINING_DIR / "hf_datasets"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
ABLATION_RESULTS_DIR = RESULTS_DIR / "ablation"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_str(path: Path) -> str:
    """Convert Path to forward-slash string (for cross-platform compatibility)."""
    return str(path).replace("\\", "/")
