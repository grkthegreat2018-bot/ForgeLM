"""Model boot worker — loads a checkpoint and runs a quick test generation.

Runs in a QThread so the UI doesn't freeze during model loading.
Emits status signals + test output for the Models page.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QThread, Signal

from .status_reader import project_root

logger = logging.getLogger(__name__)


class ModelBootWorker(QThread):
    """Boots a model from checkpoint, runs a test prompt, reports results."""

    status = Signal(str)       # human-readable status
    output = Signal(str)       # test generation output
    result = Signal(dict)      # final result dict
    error = Signal(str)

    def __init__(self, checkpoint: str, config_name: str,
                 test_prompt: str = "def fibonacci(n):",
                 max_new_tokens: int = 64,
                 parent=None) -> None:
        super().__init__(parent)
        self.checkpoint = checkpoint
        self.config_name = config_name
        self.test_prompt = test_prompt
        self.max_new_tokens = max_new_tokens
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.status.emit("Loading model…")
            engine = self._load_engine()
            if engine is None:
                self.error.emit("Failed to load engine — check checkpoint path and config name.")
                return
            if self._cancel:
                return

            self.status.emit("Model loaded · running test generation…")
            t0 = time.perf_counter()
            try:
                out = engine.generate(
                    self.test_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7, top_k=50, top_p=0.95,
                )
            except Exception as e:
                self.error.emit(f"Generation failed: {type(e).__name__}: {e}")
                return
            dt = max(1e-6, time.perf_counter() - t0)
            tps = len(out.split()) / dt

            self.output.emit(out)
            self.result.emit({
                "output": out,
                "tokens": len(out.split()),
                "time_s": dt,
                "tps": tps,
                "prompt": self.test_prompt,
            })
            self.status.emit(f"Done · {tps:.1f} tok/s · {dt:.2f}s")
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

    def _load_engine(self):
        if not self.checkpoint or not self.config_name:
            return None
        try:
            from research.inference.forge_engine import ForgeEngine  # type: ignore
            root = project_root()
            ckpt = self.checkpoint
            if not __import__("os").path.isabs(ckpt):
                ckpt = str(root / ckpt)
            engine = ForgeEngine.from_checkpoint(
                checkpoint=ckpt, config_name=self.config_name,
            )
            engine.activate()
            return engine
        except Exception as e:
            logger.warning("model boot failed (ckpt=%s cfg=%s): %s",
                           self.checkpoint, self.config_name, e)
            return None
