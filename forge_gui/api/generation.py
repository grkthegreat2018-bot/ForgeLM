"""Live generation bridge — streams tokens from the SHARED EngineRuntime.

The old behavior loaded a fresh engine per generation (a second 5GB model
in VRAM next to whatever the Engine/Chat pages had resident). Now the
worker borrows the resident engine; if none is loaded it fails with a
hint instead of silently loading another copy.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QThread, Signal

from .engine_runtime import EngineRuntime

logger = logging.getLogger(__name__)


class GenerationWorker(QThread):
    """Runs in a background thread; emits tokens + completion stats."""

    token = Signal(str)        # one decoded token/chunk
    done = Signal(str, float)  # full_text, tok/s
    error = Signal(str)
    status = Signal(str)       # human-readable status line

    def __init__(self, runtime: EngineRuntime, prompt: str,
                 max_new_tokens: int = 128, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.95,
                 parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: C901
        try:
            if not self.runtime.is_ready():
                self.error.emit(
                    "No model resident — load one on the Engine page first "
                    "(one shared engine serves every page).")
                return
            self.status.emit("Generating…")
            t0 = time.perf_counter()
            chunks: list[str] = []
            with self.runtime.acquire(timeout_s=60.0) as engine:
                for tok in engine.generate_stream(
                        self.prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                ):
                    if self._cancel:
                        break
                    chunks.append(tok)
                    self.token.emit(tok)
            dt = max(1e-6, time.perf_counter() - t0)
            full = "".join(chunks)
            tps = len(full.split()) / dt
            self.done.emit(full, tps)
        except RuntimeError as e:
            # engine busy / not loaded
            self.error.emit(str(e))
        except Exception as e:
            logger.warning("generation failed: %s", e, exc_info=True)
            self.error.emit(f"{type(e).__name__}: {e}")
