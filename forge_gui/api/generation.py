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


class AdaptiveGenWorker(QThread):
    """Adaptive thinking generation — uses generate_adaptive().

    The model's root token decides whether to think (long generation with
    think_prefix) or answer directly (short generation). Emits the result
    and whether thinking was triggered.
    """

    done = Signal(str, bool, float)   # text, did_think, tok/s
    error = Signal(str)
    status = Signal(str)

    def __init__(self, runtime: EngineRuntime, prompt: str,
                 think_max_tokens: int = 512, no_think_max_tokens: int = 256,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 80, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.prompt = prompt
        self.think_max_tokens = think_max_tokens
        self.no_think_max_tokens = no_think_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def run(self) -> None:
        try:
            if not self.runtime.is_ready():
                self.error.emit("No model resident — load one on the Engine page first.")
                return
            self.status.emit("Adaptive generating…")
            t0 = time.perf_counter()
            with self.runtime.acquire(timeout_s=120.0) as engine:
                text, did_think = engine.generate_adaptive(
                    self.prompt,
                    think_max_tokens=self.think_max_tokens,
                    no_think_max_tokens=self.no_think_max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
            dt = max(1e-6, time.perf_counter() - t0)
            tps = len(text.split()) / dt
            self.done.emit(text, did_think, tps)
        except RuntimeError as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.warning("adaptive gen failed: %s", e, exc_info=True)
            self.error.emit(f"{type(e).__name__}: {e}")


class BatchGenWorker(QThread):
    """Batched multi-prompt generation — uses generate_batch().

    Processes multiple prompts in a single batched forward pass for 3-5x
    throughput vs serial. Each line of the input is a separate prompt.
    """

    done = Signal(list, float)   # list[str] results, total tok/s
    error = Signal(str)
    status = Signal(str)

    def __init__(self, runtime: EngineRuntime, prompts: list[str],
                 max_new_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 80, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def run(self) -> None:
        try:
            if not self.runtime.is_ready():
                self.error.emit("No model resident — load one on the Engine page first.")
                return
            self.status.emit(f"Batch generating {len(self.prompts)} prompts…")
            t0 = time.perf_counter()
            with self.runtime.acquire(timeout_s=300.0) as engine:
                results = engine.generate_batch(
                    self.prompts,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
            dt = max(1e-6, time.perf_counter() - t0)
            total_words = sum(len(r.split()) for r in results)
            tps = total_words / dt
            self.done.emit(results, tps)
        except RuntimeError as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.warning("batch gen failed: %s", e, exc_info=True)
            self.error.emit(f"{type(e).__name__}: {e}")


class RawGenWorker(QThread):
    """Raw generation — uses generate_raw() with advanced sampling controls.

    Supports min_p / min_k sampling and configurable skip_special_tokens.
    For self-play / agentic loops that need fine-grained decode control.
    """

    done = Signal(str, float)   # text, tok/s
    error = Signal(str)
    status = Signal(str)

    def __init__(self, runtime: EngineRuntime, prompt: str,
                 max_new_tokens: int = 256, temperature: float = 0.2,
                 top_p: float = 1.0, top_k: int = 80,
                 repetition_penalty: float = 1.05,
                 min_p: float = 0.0, min_k: float = 0.0,
                 skip_special_tokens: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.min_k = min_k
        self.skip_special_tokens = skip_special_tokens

    def run(self) -> None:
        try:
            if not self.runtime.is_ready():
                self.error.emit("No model resident — load one on the Engine page first.")
                return
            self.status.emit("Raw generating…")
            t0 = time.perf_counter()
            with self.runtime.acquire(timeout_s=120.0) as engine:
                text = engine.generate_raw(
                    self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    min_p=self.min_p,
                    min_k=self.min_k,
                    skip_special_tokens=self.skip_special_tokens,
                )
            dt = max(1e-6, time.perf_counter() - t0)
            tps = len(text.split()) / dt
            self.done.emit(text, tps)
        except RuntimeError as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.warning("raw gen failed: %s", e, exc_info=True)
            self.error.emit(f"{type(e).__name__}: {e}")
