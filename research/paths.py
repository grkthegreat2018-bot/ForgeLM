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
AIRMOE_MODULES_DIR = CHECKPOINTS_DIR / "airmoe_modules"

# V2 expert library — spectrally-injected experts built on V2 checkpoint.
FORGELM_V2_EXPERTS_DIR = EXPERT_TRAINING_DIR / "experts_injected"
FORGELM_V2_CHECKPOINT = CHECKPOINTS_DIR / "forgelm_v2.safetensors"
# Backward compat alias
FORGELM_V4_DIR = FORGELM_V2_EXPERTS_DIR

# Additional artifacts used by scripts/ (previously hardcoded there).
QWEN_HF_TOKENIZER_DIR = CHECKPOINTS_DIR / "qwen_hf"
DSPARK_HEAD_PATH = CHECKPOINTS_DIR / "dspark_head.pt"
VOCAB_PACK_DIR = CHECKPOINTS_DIR / "vocab_packs"
EXPERTS_DIR = EXPERT_TRAINING_DIR / "experts"
HF_DATASETS_DIR = EXPERT_TRAINING_DIR / "hf_datasets"
ALL_TEACHERS_V2_SCORED = DATA_DIR / "all_teachers_v2_scored.jsonl"
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
