"""EngineRuntime — one resident ForgeEngine shared by Chat / Agent / Console.

The engine is heavy (torch + checkpoint in VRAM) so exactly one instance is
kept alive and every page borrows it through ``acquire()`` (a re-entrant
generation lock that serializes forward passes across worker threads).
Loading runs in a QThread; state changes are emitted as Qt signals:

    idle ──load()──▶ loading ──▶ ready
                       │            │
                       ▼            ▼
                     error     (unload → idle)

Activation control:
    load(..., activation=None)   → the engine's own optimal auto-activation
                                   runs once at load time (no second reset —
                                   this used to silently downgrade
                                   rotorquant→paged / nvfp4→bf16).
    load(..., activation={...})  → exact feature set from the catalog
                                   (forge_gui.api.activation_catalog).
    reactivate({...})            → live re-activation of the resident engine
                                   (swap KV cache / quant / decoding without
                                   reloading weights).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from PySide6.QtCore import QObject, QThread, Signal

from .status_reader import project_root

logger = logging.getLogger(__name__)


def _set_no_compile(use_compile: bool | None) -> None:
    """Set/clear FORGE_NO_COMPILE before the engine import (opt-out of
    torch.compile, which is broken on some triton/SM120 stacks)."""
    if use_compile is True:
        os.environ.pop("FORGE_NO_COMPILE", None)
    elif use_compile is False:
        os.environ["FORGE_NO_COMPILE"] = "1"


class _LoadWorker(QThread):
    """Imports torch + ForgeEngine and loads the checkpoint off the UI thread."""

    progress = Signal(str)
    finished_ok = Signal(object, dict)   # engine, info
    failed = Signal(str)

    def __init__(self, checkpoint: str, config_name: str,
                 activation: Optional[dict] = None,
                 use_compile: Optional[bool] = None, parent=None) -> None:
        super().__init__(parent)
        self.checkpoint = checkpoint
        self.config_name = config_name
        self.activation = activation
        self.use_compile = use_compile

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            # Fast load ⇒ opt out of torch.compile before the engine is
            # imported (from_checkpoint auto-activates the optimal preset,
            # which compiles — broken on some triton/GPU stacks).
            if self.activation is not None:
                _set_no_compile(bool(self.activation.get("use_compile", False)))
            else:
                _set_no_compile(self.use_compile if self.use_compile is not None
                                else False)
            self.progress.emit("importing torch / ForgeEngine…")
            from research.inference.forge_engine import ForgeEngine  # type: ignore
            root = project_root()
            ckpt = self.checkpoint
            if not ckpt:
                ckpt = str(root / "research" / "checkpoints" /
                           "ForgeLM_V2_Light.safetensors")
            elif not _isabs(ckpt):
                ckpt = str(root / ckpt)

            # VRAM pre-flight: fail fast with a clear message instead of
            # silently degrading to AirLLM meta-device streaming.
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                need = os.path.getsize(ckpt) * 2.5  # weights + activation + warmup
                if free < need:
                    self.failed.emit(
                        f"not enough free VRAM: {free / 1e9:.1f} GB free, "
                        f"loading this checkpoint needs ~{need / 1e9:.1f} GB. "
                        f"Close other GPU apps / GUI instances (or unload the "
                        f"current model) and retry.")
                    return

            self.progress.emit(f"loading {ckpt} ({self.config_name})…")
            if self.activation is not None:
                # Explicit activation: skip auto-activation and activate the
                # exact feature set requested by the user.
                engine = ForgeEngine.from_checkpoint(
                    checkpoint=ckpt, config_name=self.config_name,
                    auto_activate=False)
                self.progress.emit("activating features (manual preset)…")
                engine.activate(**self.activation)
            else:
                # No explicit config: let the engine pick its optimal preset
                # exactly once (honors FORGE_NO_COMPILE). No second activate()
                # call — that used to reset every optimal feature back to
                # defaults (paged KV, no quant, no fusion).
                engine = ForgeEngine.from_checkpoint(
                    checkpoint=ckpt, config_name=self.config_name)
                self.progress.emit("activating features (optimal preset)…")

            active: dict = {}
            try:
                ac = getattr(engine, "active_config", None)
                if ac is not None:
                    active = ac.to_dict()
            except Exception:
                pass
            info = {
                "checkpoint": ckpt,
                "config_name": self.config_name,
                "load_s": round(time.perf_counter() - t0, 2),
                "device": str(getattr(getattr(engine, "device", None), "type", "?")),
                "dtype": str(getattr(engine, "dtype", "?")),
                "use_compile": bool(active.get("use_compile",
                                               self.use_compile or False)),
                "activation": active,
            }
            self.finished_ok.emit(engine, info)
        except Exception as e:
            logger.warning("engine load failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _ReactivateWorker(QThread):
    """Re-activates runtime strategies on the RESIDENT engine (no reload)."""

    progress = Signal(str)
    finished_ok = Signal(dict)     # new active config
    failed = Signal(str)

    def __init__(self, runtime: "EngineRuntime", activation: dict,
                 parent=None) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self.activation = activation

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            _set_no_compile(bool(self.activation.get("use_compile", False)))
            with self._runtime.acquire(timeout_s=120.0) as engine:
                self.progress.emit("applying features to resident engine…")
                engine.activate(**self.activation)
                active: dict = {}
                try:
                    ac = getattr(engine, "active_config", None)
                    if ac is not None:
                        active = ac.to_dict()
                except Exception:
                    pass
            self.progress.emit(
                f"features applied in {time.perf_counter() - t0:.1f}s")
            self.finished_ok.emit(active)
        except Exception as e:
            logger.warning("reactivate failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


def _isabs(p: str) -> bool:
    return os.path.isabs(p)


class EngineRuntime(QObject):
    """Owns the resident engine; serializes generation across threads."""

    state_changed = Signal(str)          # idle | loading | ready | error
    progress = Signal(str)
    ready = Signal(dict)                 # info
    failed = Signal(str)
    reactivating = Signal()              # live feature re-activation started
    reactivated = Signal(dict)           # new active config

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = None
        self._state = "idle"
        self._info: dict[str, Any] = {}
        self._error = ""
        self._worker: Optional[_LoadWorker] = None
        self._reworker: Optional[_ReactivateWorker] = None
        self._lock = threading.Lock()

    # ── state ─────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    @property
    def error(self) -> str:
        return self._error

    def is_ready(self) -> bool:
        return self._state == "ready" and self._engine is not None

    def is_busy(self) -> bool:
        """Loading or re-activating right now."""
        return (self._state == "loading"
                or (self._reworker is not None and self._reworker.isRunning()))

    def _set_state(self, s: str) -> None:
        self._state = s
        self.state_changed.emit(s)

    # ── load / unload ─────────────────────────────────────────────────
    def load(self, checkpoint: str, config_name: str,
             use_compile: bool | None = None,
             activation: dict | None = None) -> None:
        """Load a checkpoint.

        ``activation=None`` → engine's optimal auto-activation (fast-load
        toggle still honored). ``activation={...}`` → the exact feature set
        (see forge_gui.api.activation_catalog).
        """
        if self._state == "loading":
            return
        if self.is_ready():
            self.unload()
        self._error = ""
        self._set_state("loading")
        self._worker = _LoadWorker(checkpoint, config_name, activation,
                                   use_compile, parent=self)
        self._worker.progress.connect(self.progress)
        self._worker.finished_ok.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_loaded(self, engine, info: dict) -> None:
        self._engine = engine
        self._info = info or {}
        self._set_state("ready")
        self.ready.emit(self._info)

    def _on_failed(self, err: str) -> None:
        self._error = err
        self._engine = None
        self._set_state("error")
        self.failed.emit(err)

    def unload(self) -> None:
        w = self._worker
        if w is not None and w.isRunning():
            w.wait(5000)
        self._worker = None
        rw = self._reworker
        if rw is not None and rw.isRunning():
            rw.wait(5000)
        self._reworker = None
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.sleep()
            except Exception as e:
                logger.warning("engine sleep failed during unload: %s", e)
            del engine
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._info = {}
        self._set_state("idle")

    def shutdown(self, wait_s: float = 60.0) -> None:
        """Wait for an in-flight load so the QThread is never destroyed
        while still running (called from MainWindow.closeEvent)."""
        for w in (self._worker, self._reworker):
            if w is not None and w.isRunning():
                if not w.wait(int(wait_s * 1000)):
                    logger.warning("worker did not finish within %ss", wait_s)

    # ── live re-activation ────────────────────────────────────────────
    def reactivate(self, activation: dict) -> None:
        """Apply a new feature set to the resident engine without reloading
        weights (swap KV cache / quantization / decoding strategies)."""
        if not self.is_ready():
            self.failed.emit("no resident engine to re-activate — load a model first")
            return
        if self.is_busy():
            self.failed.emit("engine is busy (loading / re-activating)")
            return
        self.reactivating.emit()
        self._reworker = _ReactivateWorker(self, activation, parent=self)
        self._reworker.progress.connect(self.progress)
        self._reworker.finished_ok.connect(self._on_reactivated)
        self._reworker.failed.connect(self.failed)
        self._reworker.start()

    def _on_reactivated(self, active: dict) -> None:
        self._info = dict(self._info, activation=active,
                          use_compile=bool(active.get("use_compile", False)))
        self.reactivated.emit(active)

    # ── borrowing the engine ──────────────────────────────────────────
    def acquire(self, timeout_s: float = 600.0):
        """Context manager: exclusive access to the engine while generating.

        Raises RuntimeError if the engine is not ready. Blocks while another
        thread holds the engine (chat streaming, agent round, benchmark…).
        """
        return _EngineLease(self, timeout_s)

    def try_engine(self):
        """Non-blocking peek — returns the engine or None (no lock held)."""
        return self._engine if self.is_ready() else None


class _EngineLease:
    def __init__(self, rt: EngineRuntime, timeout_s: float) -> None:
        self._rt = rt
        self._timeout = timeout_s

    def __enter__(self):
        if not self._rt._lock.acquire(timeout=self._timeout):
            raise RuntimeError("engine busy — another generation is running")
        if not self._rt.is_ready():
            self._rt._lock.release()
            raise RuntimeError("engine not loaded — load a model first "
                               "(Engine Console)")
        return self._rt._engine

    def __exit__(self, *exc) -> None:
        self._rt._lock.release()
        return None
