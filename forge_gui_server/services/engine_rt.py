"""EngineService — async port of forge_gui.api.engine_runtime.EngineRuntime.

One resident ForgeEngine shared by chat / agent / tools. The engine is
heavy (torch + checkpoint in VRAM) so exactly one instance lives here and
callers borrow it through ``acquire()`` — a threading.Lock lease identical
in semantics to the Qt version (engine calls run in executor threads, so
a threading lock is correct and preserves blocking-acquire behavior).

Load/reactivate run in a dedicated single-worker ThreadPoolExecutor;
progress is published to the event hub instead of Qt signals.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from forge_gui.api.status_reader import project_root

logger = logging.getLogger(__name__)


def _set_no_compile(use_compile: bool | None) -> None:
    if use_compile is True:
        os.environ.pop("FORGE_NO_COMPILE", None)
    elif use_compile is False:
        os.environ["FORGE_NO_COMPILE"] = "1"


class EngineService:
    """Owns the resident engine; serializes generation across tasks."""

    def __init__(self, hub) -> None:
        self._hub = hub
        self._engine = None
        self._state = "idle"           # idle | loading | ready | error
        self._info: dict[str, Any] = {}
        self._error = ""
        self._lock = threading.Lock()  # generation lease
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engine")
        self._load_task = None
        self._reactivating = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

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
        return self._state == "loading" or self._reactivating

    def snapshot(self) -> dict[str, Any]:
        return {"state": self._state, "info": dict(self._info),
                "error": self._error, "busy": self.is_busy()}

    def _set_state(self, s: str) -> None:
        self._state = s
        self._hub.publish("engine", self.snapshot())

    def _progress(self, msg: str) -> None:
        self._hub.publish("engine_progress", {"message": msg})

    # ── load / unload ─────────────────────────────────────────────────
    def load(self, checkpoint: str, config_name: str,
             use_compile: bool | None = None,
             activation: dict | None = None,
             config_overrides: dict | None = None) -> None:
        """Kick off an async load. Progress/result arrives via hub events."""
        if self._state == "loading":
            return
        if self.is_ready():
            self.unload()
        self._error = ""
        self._set_state("loading")
        loop = self._loop or asyncio.get_event_loop()
        coro = self._load_async(
            loop, checkpoint, config_name, use_compile, activation,
            config_overrides)
        if loop.is_running():
            self._load_task = asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            self._load_task = asyncio.ensure_future(coro)

    async def _load_async(self, loop, checkpoint, config_name, use_compile,
                          activation, config_overrides) -> None:
        try:
            engine, info = await loop.run_in_executor(
                self._pool, self._load_blocking, checkpoint, config_name,
                use_compile, activation, config_overrides)
        except Exception as e:
            self._error = str(e)
            self._engine = None
            self._set_state("error")
            return
        self._engine = engine
        self._info = info
        self._set_state("ready")

    def _load_blocking(self, checkpoint, config_name, use_compile,
                       activation, config_overrides):
        """Blocking load — runs in the engine executor thread."""
        t0 = time.perf_counter()
        if activation is not None:
            _set_no_compile(bool(activation.get("use_compile", False)))
        else:
            _set_no_compile(use_compile if use_compile is not None else False)
        self._progress("importing torch / ForgeEngine…")
        from forge.engine.forge_engine import ForgeEngine  # type: ignore
        root = project_root()
        ckpt = checkpoint
        if not ckpt:
            ckpt = str(root / "research" / "checkpoints" /
                       "ForgeLM_V2.safetensors")
        elif not os.path.isabs(ckpt):
            ckpt = str(root / ckpt)
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"checkpoint not found: {ckpt}")

        import torch
        from forge.engine.engine_common import _fast_load_vram_required
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            need = _fast_load_vram_required(os.path.getsize(ckpt))
            if free < need:
                raise RuntimeError(
                    f"not enough free VRAM: {free / 1e9:.1f} GB free, "
                    f"loading this checkpoint needs ~{need / 1e9:.1f} GB. "
                    f"Close other GPU apps / GUI instances (or unload the "
                    f"current model) and retry.")

        self._progress(f"loading {ckpt} ({config_name})…")
        if activation is not None:
            engine = ForgeEngine.from_checkpoint(
                checkpoint=ckpt, config_name=config_name,
                auto_activate=False, config_overrides=config_overrides)
            self._progress("activating features (manual preset)…")
            engine.activate(**activation)
        else:
            engine = ForgeEngine.from_checkpoint(
                checkpoint=ckpt, config_name=config_name,
                config_overrides=config_overrides)
            self._progress("activating features (optimal preset)…")

        active: dict = {}
        try:
            ac = getattr(engine, "active_config", None)
            if ac is not None:
                active = ac.to_dict()
        except Exception:
            logger.debug("Failed to get active config dict", exc_info=True)
        info = {
            "checkpoint": ckpt,
            "config_name": config_name,
            "load_s": round(time.perf_counter() - t0, 2),
            "device": str(getattr(getattr(engine, "device", None), "type", "?")),
            "dtype": str(getattr(engine, "dtype", "?")),
            "use_compile": bool(active.get("use_compile",
                                         use_compile or False)),
            "activation": active,
        }
        return engine, info

    def unload(self) -> None:
        self._load_task = None
        self._reactivating = False
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
                logger.debug("Failed to clear CUDA cache", exc_info=True)
        self._info = {}
        self._set_state("idle")

    # ── live re-activation ────────────────────────────────────────────
    def reactivate(self, activation: dict) -> None:
        if not self.is_ready():
            self._hub.publish("engine_error",
                              {"error": "no resident engine to re-activate"})
            return
        if self.is_busy():
            self._hub.publish("engine_error",
                              {"error": "engine is busy"})
            return
        self._reactivating = True
        self._set_state(self._state)  # republish busy flag
        loop = self._loop or asyncio.get_event_loop()
        coro = self._reactivate_async(loop, activation)
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            asyncio.ensure_future(coro)

    async def _reactivate_async(self, loop, activation) -> None:
        t0 = time.perf_counter()
        try:
            active = await loop.run_in_executor(
                self._pool, self._reactivate_blocking, activation)
        except Exception as e:
            self._reactivating = False
            self._hub.publish("engine_error", {"error": str(e)})
            self._set_state(self._state)
            return
        self._progress(f"features applied in {time.perf_counter() - t0:.1f}s")
        self._info = dict(self._info, activation=active,
                          use_compile=bool(active.get("use_compile", False)))
        self._reactivating = False
        self._set_state(self._state)

    def _reactivate_blocking(self, activation):
        _set_no_compile(bool(activation.get("use_compile", False)))
        with self.acquire(timeout_s=120.0) as engine:
            self._progress("applying features to resident engine…")
            engine.activate(**activation)
            try:
                ac = getattr(engine, "active_config", None)
                return ac.to_dict() if ac is not None else {}
            except Exception:
                return {}

    # ── borrowing the engine ──────────────────────────────────────────
    def acquire(self, timeout_s: float = 600.0):
        """Sync context manager — call from executor threads (blocking)."""
        return _EngineLease(self, timeout_s)

    def try_engine(self):
        return self._engine if self.is_ready() else None

    async def shutdown(self) -> None:
        if self._load_task and not self._load_task.done():
            try:
                await asyncio.wait_for(self._load_task, timeout=60)
            except Exception:
                pass
        self._pool.shutdown(wait=False)


class _EngineLease:
    def __init__(self, rt: EngineService, timeout_s: float) -> None:
        self._rt = rt
        self._timeout = timeout_s

    def __enter__(self):
        if not self._rt._lock.acquire(timeout=self._timeout):
            raise RuntimeError("engine busy — another generation is running")
        if not self._rt.is_ready():
            self._rt._lock.release()
            raise RuntimeError("engine not loaded — load a model first")
        return self._rt._engine

    def __exit__(self, *exc) -> None:
        self._rt._lock.release()
        return None


# singleton — created in app.py lifespan
engine_service: EngineService | None = None
