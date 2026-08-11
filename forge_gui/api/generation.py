"""Live generation bridge — wraps ForgeEngine in a QThread and streams tokens.

Emits Qt signals so the Generations page can render token-by-token without
blocking the UI thread. Falls back to a stub generator when the engine or
checkpoint is unavailable, so the page is always demonstrable.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from .status_reader import project_root

logger = logging.getLogger(__name__)


class GenerationWorker(QThread):
    """Runs in a background thread; emits tokens + completion stats."""

    token = Signal(str)        # one decoded token/chunk
    done = Signal(str, float)  # full_text, tok/s
    error = Signal(str)
    status = Signal(str)       # human-readable status line

    def __init__(self, prompt: str, checkpoint: str, config_name: str,
                 max_new_tokens: int = 128, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.95,
                 parent=None) -> None:
        super().__init__(parent)
        self.prompt = prompt
        self.checkpoint = checkpoint
        self.config_name = config_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: C901
        try:
            self.status.emit("Loading model…")
            engine = _load_engine(self.checkpoint, self.config_name)
            if engine is None:
                self.status.emit("Engine unavailable — running stub stream")
                self._stub_stream()
                return
            self.status.emit("Generating…")
            t0 = time.perf_counter()
            chunks: list[str] = []
            try:
                stream = engine.stream_generate(
                    self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
            except AttributeError:
                stream = None
            if stream is not None:
                for tok in stream:
                    if self._cancel:
                        break
                    chunks.append(tok)
                    self.token.emit(tok)
            else:
                # Fallback: non-streaming generate, then chunk the output.
                out = engine.generate(
                    self.prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                # Emit in small word chunks for a streaming feel.
                buf = ""
                for word in out.split(" "):
                    if self._cancel:
                        break
                    piece = (buf + " " + word).strip()
                    chunks.append(piece if not buf else " " + word)
                    self.token.emit(piece if not buf else " " + word)
                    buf = word
            dt = max(1e-6, time.perf_counter() - t0)
            full = "".join(chunks)
            tps = len(full.split()) / dt
            self.done.emit(full, tps)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

    def _stub_stream(self) -> None:
        """Deterministic placeholder stream when no model is loaded."""
        sample = (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        yield a\n"
            "        a, b = b, a + b\n\n"
            "# ForgeAI stub generation — load a checkpoint to run the real engine.\n"
        )
        t0 = time.perf_counter()
        chunks: list[str] = []
        for ch in sample:
            if self._cancel:
                break
            chunks.append(ch)
            self.token.emit(ch)
            self.msleep(12)
        dt = max(1e-6, time.perf_counter() - t0)
        self.done.emit("".join(chunks), len(sample.split()) / dt)


def _load_engine(checkpoint: str, config_name: str):
    """Try to build a ForgeEngine; return None on any failure."""
    if not checkpoint or not config_name:
        return None
    try:
        from research.inference.forge_engine import ForgeEngine  # type: ignore
        root = project_root()
        ckpt = checkpoint
        if not __import__("os").path.isabs(ckpt):
            ckpt = str(root / ckpt)
        engine = ForgeEngine.from_checkpoint(
            checkpoint=ckpt, config_name=config_name,
        )
        engine.activate()
        return engine
    except Exception as e:
        logger.warning("engine load failed (checkpoint=%s config=%s): %s",
                       checkpoint, config_name, e)
        return None


def list_checkpoints() -> list[tuple[str, str]]:
    """Return [(relative_path, name), ...] of safetensors/pt under research/checkpoints."""
    root = project_root()
    ckpt_dir = root / "research" / "checkpoints"
    out: list[tuple[str, str]] = []
    if not ckpt_dir.is_dir():
        return out
    for p in sorted(ckpt_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".safetensors", ".pt", ".bin", ".gguf"):
            rel = str(p.relative_to(root)).replace("\\", "/")
            out.append((rel, p.name))
    return out
