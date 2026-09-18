"""Launch presets — process templates for the Launch page.

The process-runner half of this module moved to
``forge_gui_server.services.procs.ProcessService`` (asyncio subprocesses
feeding the WebSocket event stream). What remains here is the Qt-free
preset catalog + command builder used by the REST API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .status_reader import project_root


@dataclass
class ProcessPreset:
    """A launchable process template with editable arguments."""
    name: str
    script: str
    description: str
    args: list[str] = field(default_factory=list)
    arg_defaults: dict[str, str] = field(default_factory=dict)


def get_presets() -> list[ProcessPreset]:
    """Return all available process presets."""
    return [
        ProcessPreset(
            name="Train Expert (Supervised)",
            script="scripts/train_expert.py",
            description="Fine-tune an AirMoE expert on external data (math, science, etc.)",
            arg_defaults={"--topic": "python_algorithms", "--data": "", "--epochs": "3"},
        ),
        ProcessPreset(
            name="Train Expert (Self-Play)",
            script="scripts/train_expert.py",
            description="Self-play mode — model generates + verifies code solutions",
            arg_defaults={"--topic": "python_algorithms", "--mode": "selfplay", "--epochs": "3"},
        ),
        ProcessPreset(
            name="Ablation Benchmark",
            script="scripts/ablation_benchmark.py",
            description="Run ablation benchmark suite across model configurations",
            arg_defaults={},
        ),
        ProcessPreset(
            name="Download HF Datasets",
            script="scripts/download_hf_datasets.py",
            description="Download HuggingFace datasets for training",
            arg_defaults={},
        ),
        ProcessPreset(
            name="Extract Vocab Packs",
            script="scripts/extract_vocab_packs.py",
            description="Extract vocabulary packs from tokenizer",
            arg_defaults={},
        ),
    ]


def _venv_python() -> str:
    root = project_root()
    venv_python = str(root / "venv" / "Scripts" / "python.exe")
    return venv_python if Path(venv_python).is_file() else "python"


def build_command(preset: ProcessPreset, arg_overrides: dict[str, str]) -> list[str]:
    """Build the full command list from a preset + arg overrides."""
    root = project_root()
    cmd = [_venv_python(), str(root / preset.script)]
    merged = {**preset.arg_defaults, **arg_overrides}
    for flag, val in merged.items():
        if val.strip():
            cmd.extend([flag, val.strip()])
    return cmd
