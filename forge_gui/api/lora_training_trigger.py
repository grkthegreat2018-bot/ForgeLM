"""LoRA training trigger — allows the agent to kick off LoRA training.

Provides a tool definition and execution wrapper that the unified harness
dispatches. The agent can request training of a new LoRA adapter for a
specific skill category, using chat history or a dataset as training data.

The training runs as a subprocess via ProcessManager (same as the FineTune
page), so it doesn't block the agent loop. The agent can poll the training
status using the `check_training` tool.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def lora_training_tool_defs() -> list[dict]:
    """Tool definitions for LoRA training triggers."""
    return [
        {
            "type": "function",
            "function": {
                "name": "train_lora",
                "description": (
                    "Launch a LoRA training run to create a new skill "
                    "adapter. Training runs in the background. Use this "
                    "when you identify a skill gap and want to improve "
                    "the model for a specific category (coding, math, "
                    "reasoning, etc.). Training data can come from chat "
                    "history (good-rated turns) or a JSONL file."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": (
                                "Skill category for the adapter (coding, "
                                "math, reasoning, tool_use, agentic, "
                                "chat_assist, self_play, vision)"),
                            "enum": ["coding", "math", "reasoning",
                                     "tool_use", "agentic", "chat_assist",
                                     "self_play", "vision"],
                        },
                        "rank": {
                            "type": "integer",
                            "description": "LoRA rank (default 32)",
                        },
                        "data_source": {
                            "type": "string",
                            "enum": ["chat_history", "file"],
                            "description": (
                                "Source of training data: 'chat_history' "
                                "uses good-rated chat turns, 'file' uses "
                                "a JSONL file path"),
                        },
                        "data_file": {
                            "type": "string",
                            "description": (
                                "Path to JSONL training data (required "
                                "if data_source='file')"),
                        },
                        "epochs": {
                            "type": "integer",
                            "description": "Training epochs (default 3)",
                        },
                        "lr": {
                            "type": "number",
                            "description": "Learning rate (default 5e-5)",
                        },
                    },
                    "required": ["category"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_training",
                "description": (
                    "Check the status of a LoRA training run. Returns "
                    "progress, current epoch, loss, and whether it's "
                    "still running."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID from train_lora response",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
    ]


class LoraTrainingTrigger:
    """Executes LoRA training tool calls via ProcessManager.

    Args:
        proc_mgr: ProcessManager instance for launching subprocesses.
        chat_store: ChatStore for extracting good-rated turns.
        checkpoint: Base checkpoint path for training.
    """

    def __init__(self, proc_mgr=None, chat_store=None,
                 checkpoint: str = "") -> None:
        self.proc_mgr = proc_mgr
        self.chat_store = chat_store
        self.checkpoint = checkpoint
        self._training_tasks: dict[str, dict] = {}  # task_id → metadata

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch a training tool call."""
        if name == "train_lora":
            return self._train_lora(args)
        elif name == "check_training":
            return self._check_training(args)
        return {"ok": False, "result": {"error": f"unknown training tool: {name}"}}

    def _train_lora(self, args: dict) -> dict:
        """Launch a LoRA training run."""
        category = args.get("category", "")
        if not category:
            return {"ok": False, "result": {"error": "category required"}}

        rank = args.get("rank", 32)
        epochs = args.get("epochs", 3)
        lr = args.get("lr", 5e-5)
        data_source = args.get("data_source", "chat_history")

        # prepare training data
        if data_source == "chat_history":
            if self.chat_store is None:
                return {"ok": False, "result": {"error": "no chat_store available"}}
            data_path = self._export_chat_data(category)
            if data_path is None:
                return {"ok": False, "result": {
                    "error": "no good-rated chat turns found for training"}}
        else:
            data_file = args.get("data_file", "")
            if not data_file:
                return {"ok": False, "result": {
                    "error": "data_file required when data_source='file'"}}
            data_path = Path(data_file)
            if not data_path.is_file():
                return {"ok": False, "result": {
                    "error": f"data file not found: {data_file}"}}

        # build output adapter name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        adapter_name = f"ForgeLM_V11_{category}_R{rank}_{timestamp}_lora"
        output_dir = Path("research/checkpoints/lora") / adapter_name

        # build training command
        venv_py = str(Path("venv/Scripts/python.exe"))
        cmd = [
            venv_py, "-u", "-m", "research.training.runners.sft_train",
            "--checkpoint", self.checkpoint,
            "--data", str(data_path),
            "--save-lora-adapter",
            "--lora-rank", str(rank),
            "--lora-alpha", str(rank * 2),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--output", str(output_dir / "adapter.safetensors"),
        ]

        # launch via ProcessManager
        if self.proc_mgr is None:
            return {"ok": False, "result": {
                "error": "no ProcessManager available to launch training"}}

        task_id = self.proc_mgr.launch(f"LoRA Training ({category})", cmd)
        self._training_tasks[task_id] = {
            "category": category,
            "rank": rank,
            "adapter_name": adapter_name,
            "output_dir": str(output_dir),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_path": str(data_path),
        }

        return {"ok": True, "result": {
            "task_id": task_id,
            "category": category,
            "adapter_name": adapter_name,
            "rank": rank,
            "epochs": epochs,
            "message": (
                f"LoRA training launched for '{category}' category. "
                f"Adapter will be saved as {adapter_name}. "
                f"Use check_training(task_id='{task_id}') to monitor progress.")}}

    def _check_training(self, args: dict) -> dict:
        """Check training status."""
        task_id = args.get("task_id", "")
        if task_id not in self._training_tasks:
            return {"ok": False, "result": {"error": f"unknown task_id: {task_id}"}}

        meta = self._training_tasks[task_id]
        # check if process is still running
        running = False
        if self.proc_mgr is not None:
            procs = self.proc_mgr.processes()
            running = any(p.get("id") == task_id and p.get("running")
                          for p in procs)

        return {"ok": True, "result": {
            "task_id": task_id,
            "running": running,
            "category": meta["category"],
            "adapter_name": meta["adapter_name"],
            "started": meta["started"],
            "output_dir": meta["output_dir"],
            "message": (
                "Training in progress..." if running else
                "Training completed (or stopped). Check output directory.")}}

    def _export_chat_data(self, category: str) -> Optional[Path]:
        """Export good-rated chat turns as JSONL for training."""
        if self.chat_store is None:
            return None
        try:
            export_path = Path("data/sft") / f"lora_{category}_{int(time.time())}.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            n = self.chat_store.export_training_data(str(export_path))
            if n == 0:
                return None
            return export_path
        except Exception as e:
            logger.warning("chat data export failed: %s", e)
            return None

    @property
    def training_tasks(self) -> dict[str, dict]:
        return self._training_tasks
