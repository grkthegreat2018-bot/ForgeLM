"""Model boot worker — loads into the SHARED EngineRuntime and runs a test
generation. All GUI engine loads go through EngineRuntime so exactly one
model is ever resident in VRAM (the old behavior spawned a second engine
here, which could exhaust the 12GB card when another page had one loaded).
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QThread, Signal

from .engine_runtime import EngineRuntime

logger = logging.getLogger(__name__)


class ModelBootWorker(QThread):
    """Ensures the runtime has the requested checkpoint, then test-generates."""

    status = Signal(str)       # human-readable status
    output = Signal(str)       # test generation output
    result = Signal(dict)      # final result dict
    error = Signal(str)

    def __init__(self, runtime: EngineRuntime, checkpoint: str,
                 config_name: str, test_prompt: str = "def fibonacci(n):",
                 max_new_tokens: int = 64, use_compile: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.checkpoint = checkpoint
        self.config_name = config_name
        self.test_prompt = test_prompt
        self.max_new_tokens = max_new_tokens
        self.use_compile = use_compile
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            need_reload = self._needs_reload()
            if need_reload:
                self.status.emit("Loading model into shared runtime…")
                self.runtime.load(self.checkpoint, self.config_name,
                                  use_compile=self.use_compile)
                t0 = time.time()
                while (self.runtime.state == "loading"
                       and time.time() - t0 < 300):
                    self.msleep(400)
                if self._cancel:
                    return
                if not self.runtime.is_ready():
                    self.error.emit(self.runtime.error
                                    or "load failed — see Engine page / gui.log")
                    return
            else:
                self.status.emit("Using resident model…")

            if self._cancel:
                return
            self.status.emit("Model loaded · running test generation…")
            t0 = time.perf_counter()
            try:
                with self.runtime.acquire(timeout_s=120.0) as engine:
                    out = engine.generate(
                        self.test_prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7, top_k=50, top_p=0.95,
                    )
            except RuntimeError as e:
                self.error.emit(f"Generation failed: {e}")
                return
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
            logger.warning("model boot failed: %s", e, exc_info=True)
            self.error.emit(f"{type(e).__name__}: {e}")

    def _needs_reload(self) -> bool:
        if not self.runtime.is_ready():
            return True
        info = self.runtime.info
        resident = str(info.get("checkpoint", "")).replace("\\", "/").lower()
        wanted = self.checkpoint.replace("\\", "/").lower()
        if wanted and not resident.endswith(wanted.split("/")[-1]):
            return True
        # same checkpoint but a different config → reload
        return info.get("config_name") != self.config_name
