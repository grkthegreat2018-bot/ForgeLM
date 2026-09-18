"""LoraService — async port of LoraManager + LoraHarness.

Pure helpers (scan_lora_adapters, read_adapter_header, parse_category,
TARGET_PRESETS, MODE_CATEGORY_MAP) are imported from the original module
— importing forge_gui.api.lora_store pulls PySide6 class definitions but
they are never instantiated here.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from forge_gui.api.lora_store import (
    LORA_CATEGORIES,
    MODE_CATEGORY_MAP,
    TARGET_PRESETS,
    LoRAEntry,
    parse_category,
    read_adapter_header,
    scan_lora_adapters,
)
from forge_gui.api.status_reader import project_root

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lora")


class LoraService:
    def __init__(self, hub, runtime) -> None:
        self._hub = hub
        self._runtime = runtime
        self._busy = False
        self._mode = "chat"
        self._pinned: str | None = None
        self._current: str | None = None

    # ── scan / query ──────────────────────────────────────────────────
    def scan(self) -> list[dict]:
        return [vars(e) for e in scan_lora_adapters()]

    def adapters_by_category(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {c: [] for c in LORA_CATEGORIES}
        out["uncategorized"] = []
        for e in scan_lora_adapters():
            out.setdefault(e.category, []).append(vars(e))
        return out

    def status(self) -> dict[str, Any]:
        return {"mode": self._mode, "pinned": self._pinned,
                "current": self._current, "busy": self._busy,
                "target_presets": sorted(TARGET_PRESETS.keys()),
                "modes": sorted(MODE_CATEGORY_MAP.keys())}

    def _emit(self, kind: str, data: Any = None) -> None:
        self._hub.publish("lora", {"kind": kind, "data": data,
                                   "status": self.status()})

    def _status(self, msg: str) -> None:
        self._emit("status", msg)

    # ── engine hot-swap ───────────────────────────────────────────────
    async def load_on_engine(self, path: str, rank: int = 32,
                             alpha: int | None = None,
                             target_key: str = "default") -> dict:
        loop = asyncio.get_running_loop()
        self._busy = True
        try:
            return await loop.run_in_executor(
                _pool, self._action_blocking, "load", path, rank, alpha,
                target_key)
        finally:
            self._busy = False

    async def unload_from_engine(self) -> dict:
        loop = asyncio.get_running_loop()
        self._busy = True
        try:
            return await loop.run_in_executor(
                _pool, self._action_blocking, "unload", "", 32, None,
                "default")
        finally:
            self._busy = False

    async def refresh_info(self) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _pool, self._action_blocking, "info", "", 32, None, "default")

    def _action_blocking(self, action: str, path: str, rank: int,
                         alpha: int | None, target_key: str) -> dict:
        try:
            with self._runtime.acquire(timeout_s=60.0) as engine:
                if action == "load":
                    full = project_root() / path
                    p = str(full if full.is_file() else path)
                    self._status(f"loading adapter {Path(p).name}…")
                    targets = TARGET_PRESETS.get(target_key)
                    engine.load_lora(p, rank=rank, alpha=alpha,
                                     target_modules=targets)
                    info = engine.lora_info() or {}
                    self._current = info.get("path", "") or path
                    self._emit("loaded", info)
                    return {"ok": True, "info": info}
                elif action == "unload":
                    self._status("unloading adapter…")
                    engine.unload_lora()
                    self._current = None
                    self._emit("unloaded")
                    return {"ok": True}
                else:
                    info = engine.lora_info()
                    self._current = (info or {}).get("path") or None
                    self._emit("loaded" if info else "unloaded", info)
                    return {"ok": True, "info": info}
        except Exception as e:
            logger.warning("lora action %s failed: %s", action, e,
                           exc_info=True)
            self._emit("failed", f"{type(e).__name__}: {e}")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── merge (CPU) ───────────────────────────────────────────────────
    async def merge(self, base_checkpoint: str, config_name: str,
                    adapter_path: str, rank: int, alpha: int | None,
                    out_path: str) -> dict:
        loop = asyncio.get_running_loop()
        self._busy = True
        try:
            return await loop.run_in_executor(
                _pool, self._merge_blocking, base_checkpoint, config_name,
                adapter_path, rank, alpha, out_path)
        finally:
            self._busy = False

    def _merge_blocking(self, base, config_name, adapter, rank, alpha,
                        out) -> dict:
        engine = None
        try:
            self._status("loading base model on CPU…")
            from forge.engine.forge_engine import ForgeEngine  # type: ignore
            root = project_root()
            if base and not Path(base).is_absolute():
                cand = root / base
                base = str(cand if cand.is_file() else base)
            engine = ForgeEngine.from_checkpoint(
                checkpoint=base, config_name=config_name,
                device="cpu", auto_activate=False)

            self._status("attaching + loading LoRA adapter…")
            engine.load_lora(adapter, rank=rank, alpha=alpha)

            self._status("merging adapters into base weights…")
            from forge.training.bitnet_lora import merge_lora_adapters  # type: ignore
            n = merge_lora_adapters(engine.model)

            self._status("saving merged checkpoint…")
            from forge.checkpoint_io import save_training_checkpoint  # type: ignore
            cfg = getattr(engine, "config", None)
            meta = {"lora_merged": True, "adapter": Path(adapter).name,
                    "merged_adapters": n,
                    "t": time.strftime("%Y-%m-%d %H:%M")}
            if cfg is not None:
                meta["config"] = getattr(cfg, "__dict__", {})
            save_training_checkpoint(engine.model, out, meta=meta)
            self._emit("merged", out)
            return {"ok": True, "out": out}
        except Exception as e:
            logger.warning("lora merge failed: %s", e, exc_info=True)
            self._emit("failed", f"{type(e).__name__}: {e}")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            engine = None
            import gc
            gc.collect()

    # ── mode harness ──────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._mode

    async def set_mode(self, mode: str) -> dict:
        if mode not in MODE_CATEGORY_MAP:
            return {"ok": False, "error": f"unknown mode: {mode}"}
        self._mode = mode
        self._emit("mode", mode)
        if self._pinned is not None:
            self._status(f"mode={mode} (pinned: {Path(self._pinned).name})")
            return {"ok": True, "pinned": True}
        return await self._auto_swap()

    async def pin_adapter(self, path: str) -> dict:
        self._pinned = path
        self._status(f"pinned {Path(path).name}")
        if path != self._current:
            return await self._load_auto(path)
        return {"ok": True}

    async def unpin(self) -> dict:
        self._pinned = None
        self._status("unpinned — auto-selecting for mode")
        return await self._auto_swap()

    async def _auto_swap(self) -> dict:
        entries = scan_lora_adapters()
        if not entries:
            self._status(f"mode={self._mode} (no adapters available)")
            if self._current:
                await self.unload_from_engine()
            return {"ok": True}
        preferred = MODE_CATEGORY_MAP.get(self._mode, [])
        best = self._best_for_categories(entries, preferred)
        if best is None:
            self._status(f"mode={self._mode} (no matching adapter)")
            if self._current:
                await self.unload_from_engine()
            return {"ok": True}
        if best.path == self._current:
            self._status(f"mode={self._mode} (already loaded: {best.name})")
            return {"ok": True}
        if self._current:
            await self.unload_from_engine()
        return await self._load_auto(best.path)

    @staticmethod
    def _best_for_categories(entries: list[LoRAEntry],
                             categories: list[str]) -> LoRAEntry | None:
        for cat in categories:
            matching = [e for e in entries if e.category == cat
                        and not e.header_error]
            if matching:
                matching.sort(key=lambda e: e.modified, reverse=True)
                return matching[0]
        return None

    async def _load_auto(self, path: str) -> dict:
        full = project_root() / path
        p = str(full if full.is_file() else path)
        hdr = read_adapter_header(p)
        rank = hdr.get("rank") or 32
        alpha = rank * 2
        self._status(f"loading {Path(path).name} (rank {rank})…")
        return await self.load_on_engine(path, rank, alpha)

    def recommend_for_mode(self, mode: str) -> dict | None:
        entries = scan_lora_adapters()
        best = self._best_for_categories(
            entries, MODE_CATEGORY_MAP.get(mode, []))
        return vars(best) if best else None


lora_service: LoraService | None = None
