"""LoRA store — adapter discovery + hot-load / unload / merge workers + categories.

Adapter files follow the project convention ``*lora*.safetensors``
(e.g. ``ForgeLM_V2_Light_R31_lora.safetensors``, ``epoch3_lora.safetensors``)
under ``research/checkpoints/``.

**Skill-based category system**: adapters are tagged with a skill category
parsed from the filename. The naming convention is::

    ForgeLM_V11_<category>_R<rank>_lora.safetensors
    ForgeLM_V11_coding_R32_lora.safetensors
    ForgeLM_V11_agentic_R64_lora.safetensors

Categories drive the harness auto-load/unload: when the user enters chat
mode, the ``chat_assist`` adapter loads; when they switch to the Agent
page, ``agentic`` loads; etc. The user can also manually override which
adapters are active from the LoRA Manager page.

The safetensors header is parsed with pure stdlib (8-byte length prefix +
JSON) so rank / param count / dtype can be shown in the GUI and the correct
rank auto-filled — no torch import needed for browsing.

Merging runs on CPU (device="cpu") so it never competes with the resident
engine for the 12GB of VRAM (mixed CPU/GPU mandate).
"""
from __future__ import annotations

import json
import logging
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from .status_reader import project_root

logger = logging.getLogger(__name__)

# target-module presets understood by ForgeEngine.load_lora
TARGET_PRESETS = {
    "default": None,  # engine default: FFN + attention
    "ffn": ["w_gate", "w_up", "w_down"],
    "attention": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"],
}

# ── skill-based LoRA categories ─────────────────────────────────────────
# Each category maps to a mode of the harness. The harness auto-loads the
# best adapter for the active mode and unloads it when switching modes.
LORA_CATEGORIES = (
    "coding",        # Python/code generation + tool use
    "math",          # Arithmetic, algorithms, reasoning
    "reasoning",     # General chain-of-thought / planning
    "tool_use",      # Agentic tool calling proficiency
    "vision",        # Image understanding (V11+)
    "self_play",     # Self-play discovery / training tasks
    "agentic",       # Full agentic loop (coding + tools + planning)
    "chat_assist",   # General chat assistant personality
)

# Map harness mode → preferred LoRA category (in priority order)
MODE_CATEGORY_MAP = {
    "chat": ["chat_assist", "reasoning", "coding"],
    "agent": ["agentic", "tool_use", "coding", "reasoning"],
    "self_play": ["self_play", "reasoning", "coding"],
    "training": ["coding", "math"],
}

# Category display metadata (label, description, icon)
CATEGORY_INFO = {
    "coding":       ("Coding", "Python code generation + debugging", "⌨"),
    "math":         ("Math", "Arithmetic, algorithms, numerical reasoning", "∑"),
    "reasoning":    ("Reasoning", "Chain-of-thought, planning, logic", "§"),
    "tool_use":     ("Tool Use", "Agentic tool calling proficiency", "⚙"),
    "vision":       ("Vision", "Image understanding (V11+)", "◉"),
    "self_play":    ("Self-Play", "Discovery + training task generation", "⚡"),
    "agentic":      ("Agentic", "Full agentic loop: coding + tools + planning", "⌘"),
    "chat_assist":  ("Chat Assist", "General chat assistant personality", "✉"),
}

# Regex to parse category from filename: looks for _<category>_ before _lora
_CATEGORY_RE = re.compile(
    r"_(" + "|".join(re.escape(c) for c in LORA_CATEGORIES) + r")_.*lora",
    re.IGNORECASE,
)


def parse_category(name: str) -> str:
    """Extract the skill category from an adapter filename.

    Returns the category string (e.g. "coding") or "uncategorized" if no
    recognized category is found in the filename.
    """
    m = _CATEGORY_RE.search(name.lower())
    if m:
        return m.group(1)
    # also check for category anywhere in the name (less strict)
    for cat in LORA_CATEGORIES:
        if cat in name.lower():
            return cat
    return "uncategorized"


@dataclass
class LoRAEntry:
    name: str
    path: str
    size_bytes: int = 0
    size_label: str = ""
    modified: float = 0.0
    rank: Optional[int] = None
    n_tensors: int = 0
    n_params: int = 0
    dtype: str = ""
    base_hint: str = ""     # guessed base checkpoint from filename
    header_error: str = ""
    category: str = "uncategorized"  # skill-based category (coding, math, ...)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def read_adapter_header(path: str | Path) -> dict:
    """Parse a safetensors header with stdlib only.

    Returns {rank, n_tensors, n_params, dtype} — best effort, never raises
    (errors come back as {"error": ...}).
    """
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 100_000_000:
                return {"error": "bad header length"}
            header = json.loads(f.read(n).decode("utf-8"))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    rank = None
    n_params = 0
    dtypes: set[str] = set()
    n_tensors = 0
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        n_tensors += 1
        n_params += int(__import__("math").prod(shape)) if shape else 0
        dtypes.add(str(info.get("dtype", "")))
        if rank is None and "lora_A" in name and len(shape) == 2:
            rank = int(shape[0])
    return {
        "rank": rank,
        "n_tensors": n_tensors,
        "n_params": n_params,
        "dtype": "/".join(sorted(d for d in dtypes if d)) or "?",
    }


def _base_hint(name: str) -> str:
    """Guess the base checkpoint from an adapter filename
    (ForgeLM_V2_Light_R31_lora → ForgeLM_V2_Light)."""
    stem = name
    for marker in ("_lora", ".lora"):
        if marker in stem:
            stem = stem.split(marker)[0]
            break
    return stem


def scan_lora_adapters() -> list[LoRAEntry]:
    """All LoRA adapter checkpoints under research/checkpoints/."""
    root = project_root()
    ckpt_dir = root / "research" / "checkpoints"
    out: list[LoRAEntry] = []
    if not ckpt_dir.is_dir():
        return out
    for p in sorted(ckpt_dir.rglob("*.safetensors")):
        if "lora" not in p.name.lower():
            continue
        try:
            st = p.stat()
        except OSError as e:
            logger.warning("stat failed for adapter %s: %s", p, e)
            continue
        hdr = read_adapter_header(p)
        out.append(LoRAEntry(
            name=p.name, path=str(p.relative_to(root)).replace("\\", "/"),
            size_bytes=st.st_size, size_label=_human_bytes(st.st_size),
            modified=st.st_mtime,
            rank=hdr.get("rank"), n_tensors=hdr.get("n_tensors", 0),
            n_params=hdr.get("n_params", 0), dtype=hdr.get("dtype", "?"),
            base_hint=_base_hint(p.name),
            header_error=hdr.get("error", ""),
            category=parse_category(p.name),
        ))
    out.sort(key=lambda e: e.modified, reverse=True)
    return out


# ── workers ────────────────────────────────────────────────────────────

class _LoraActionWorker(QThread):
    """Hot-load / unload an adapter on the resident engine."""

    status = Signal(str)
    loaded = Signal(dict)      # engine.lora_info()
    unloaded = Signal()
    failed = Signal(str)

    def __init__(self, runtime, action: str, path: str = "",
                 rank: int = 32, alpha: Optional[int] = None,
                 target_key: str = "default", parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.action = action          # "load" | "unload" | "info"
        self.path = path
        self.rank = rank
        self.alpha = alpha
        self.target_key = target_key

    def run(self) -> None:
        try:
            with self.runtime.acquire(timeout_s=60.0) as engine:
                if self.action == "load":
                    full = project_root() / self.path
                    p = str(full if full.is_file() else self.path)
                    self.status.emit(f"loading adapter {Path(p).name}…")
                    targets = TARGET_PRESETS.get(self.target_key)
                    engine.load_lora(p, rank=self.rank, alpha=self.alpha,
                                     target_modules=targets)
                    self.loaded.emit(engine.lora_info() or {})
                elif self.action == "unload":
                    self.status.emit("unloading adapter…")
                    engine.unload_lora()
                    self.unloaded.emit()
                else:
                    info = engine.lora_info()
                    if info:
                        self.loaded.emit(info)
                    else:
                        self.unloaded.emit()
        except Exception as e:
            logger.warning("lora action %s failed: %s", self.action, e,
                           exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


class _LoraMergeWorker(QThread):
    """Merge an adapter into a base checkpoint — entirely on CPU so the
    resident GPU engine (if any) is untouched."""

    status = Signal(str)
    merged = Signal(str)       # output path
    failed = Signal(str)

    def __init__(self, base_checkpoint: str, config_name: str,
                 adapter_path: str, rank: int, alpha: Optional[int],
                 out_path: str, parent=None) -> None:
        super().__init__(parent)
        self.base = base_checkpoint
        self.config_name = config_name
        self.adapter = adapter_path
        self.rank = rank
        self.alpha = alpha
        self.out = out_path

    def run(self) -> None:
        engine = None
        try:
            self.status.emit("loading base model on CPU…")
            from research.inference.forge_engine import ForgeEngine  # type: ignore
            root = project_root()
            base = self.base
            if base and not Path(base).is_absolute():
                cand = root / base
                base = str(cand if cand.is_file() else base)
            engine = ForgeEngine.from_checkpoint(
                checkpoint=base, config_name=self.config_name,
                device="cpu", auto_activate=False)

            self.status.emit("attaching + loading LoRA adapter…")
            engine.load_lora(self.adapter, rank=self.rank, alpha=self.alpha)

            self.status.emit("merging adapters into base weights…")
            from research.training.bitnet_lora import merge_lora_adapters  # type: ignore
            n = merge_lora_adapters(engine.model)

            self.status.emit("saving merged checkpoint…")
            from research.checkpoint_io import save_training_checkpoint  # type: ignore
            cfg = getattr(engine, "config", None)
            meta = {"lora_merged": True, "adapter": Path(self.adapter).name,
                    "merged_adapters": n, "t": time.strftime("%Y-%m-%d %H:%M")}
            if cfg is not None:
                meta["config"] = getattr(cfg, "__dict__", {})
            save_training_checkpoint(engine.model, self.out, meta=meta)
            self.merged.emit(self.out)
        except Exception as e:
            logger.warning("lora merge failed: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            # Free CPU memory promptly (model can be ~5GB in RAM).
            engine = None
            try:
                import gc
                gc.collect()
            except Exception:
                pass


class LoraManager(QObject):
    """UI-facing manager for LoRA adapters (browse / hot-load / merge)."""

    busy_changed = Signal(bool)
    status = Signal(str)
    lora_loaded = Signal(dict)
    lora_unloaded = Signal()
    merge_done = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker: Optional[QThread] = None
        self._busy = False

    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _start(self, worker: QThread) -> None:
        if self.is_busy():
            self.failed.emit("LoRA operation already running")
            return
        self._worker = worker
        self._set_busy(True)
        for sig, slot in ((worker.status, self.status),
                          (worker.failed, self._on_failed)):
            sig.connect(slot)
        if hasattr(worker, "loaded"):
            worker.loaded.connect(self.lora_loaded)
        if hasattr(worker, "unloaded"):
            worker.unloaded.connect(self.lora_unloaded)
        if hasattr(worker, "merged"):
            worker.merged.connect(self.merge_done)
        worker.start()

    def _set_busy(self, on: bool) -> None:
        self._busy = on
        self.busy_changed.emit(on)

    def _on_failed(self, err: str) -> None:
        self._set_busy(False)
        self.failed.emit(err)

    # ── public actions ────────────────────────────────────────────────
    def load_on_engine(self, runtime, path: str, rank: int,
                       alpha: Optional[int] = None,
                       target_key: str = "default") -> None:
        """Hot-load an adapter onto the resident engine."""
        self._start(_LoraActionWorker(runtime, "load", path, rank, alpha,
                                      target_key, parent=self))

    def unload_from_engine(self, runtime) -> None:
        self._start(_LoraActionWorker(runtime, "unload", parent=self))

    def refresh_info(self, runtime) -> None:
        """Peek at the resident engine's LoRA state (emits loaded/unloaded)."""
        self._start(_LoraActionWorker(runtime, "info", parent=self))

    def merge(self, base_checkpoint: str, config_name: str, adapter_path: str,
              rank: int, alpha: Optional[int], out_path: str) -> None:
        """Merge adapter → base on CPU, writing a standalone checkpoint."""
        self._start(_LoraMergeWorker(base_checkpoint, config_name, adapter_path,
                                     rank, alpha, out_path, parent=self))


# ── LoraHarness — auto-load/unload by mode ──────────────────────────────

class LoraHarness(QObject):
    """Mode-aware LoRA auto-manager.

    When the harness switches mode (chat → agent → self_play), it:
    1. Finds the best adapter for the new mode's preferred categories.
    2. Unloads the current adapter (if any) and loads the best one.
    3. Falls back gracefully if no adapter exists for a category.

    The user can manually override by calling ``pin_adapter()`` — a pinned
    adapter stays loaded regardless of mode changes until ``unpin()``.
    """

    mode_changed = Signal(str)          # new mode
    adapter_changed = Signal(str)       # new adapter path ("" = unloaded)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, manager: LoraManager, runtime,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mgr = manager
        self._runtime = runtime
        self._mode = "chat"
        self._pinned: Optional[str] = None   # pinned adapter path
        self._current: Optional[str] = None   # currently loaded adapter path
        # track manager signals
        manager.lora_loaded.connect(self._on_loaded)
        manager.lora_unloaded.connect(self._on_unloaded)
        manager.failed.connect(self._on_failed)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def current_adapter(self) -> Optional[str]:
        return self._current

    @property
    def pinned_adapter(self) -> Optional[str]:
        return self._pinned

    # ── mode switching ────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        """Switch the harness mode and auto-load the best adapter.

        Modes: "chat", "agent", "self_play", "training".
        If an adapter is pinned, it stays loaded (mode change is noted
        but no swap happens).
        """
        if mode not in MODE_CATEGORY_MAP:
            logger.warning("unknown lora harness mode: %s", mode)
            return
        self._mode = mode
        self.mode_changed.emit(mode)
        if self._pinned is not None:
            self.status.emit(f"mode={mode} (pinned: {Path(self._pinned).name})")
            return
        self._auto_swap()

    def pin_adapter(self, path: str) -> None:
        """Pin a specific adapter — stays loaded across mode changes."""
        self._pinned = path
        self.status.emit(f"pinned {Path(path).name}")
        if path != self._current:
            self._load(path)

    def unpin(self) -> None:
        """Remove pin and re-evaluate for the current mode."""
        self._pinned = None
        self.status.emit("unpinned — auto-selecting for mode")
        self._auto_swap()

    # ── auto-swap logic ───────────────────────────────────────────────
    def _auto_swap(self) -> None:
        """Find the best adapter for the current mode and swap to it."""
        entries = scan_lora_adapters()
        if not entries:
            self.status.emit(f"mode={self._mode} (no adapters available)")
            if self._current:
                self._mgr.unload_from_engine(self._runtime)
            return

        preferred = MODE_CATEGORY_MAP.get(self._mode, [])
        best = self._best_for_categories(entries, preferred)
        if best is None:
            # no matching adapter — unload if something is loaded
            self.status.emit(f"mode={self._mode} (no matching adapter)")
            if self._current:
                self._mgr.unload_from_engine(self._runtime)
            return

        if best.path == self._current:
            self.status.emit(f"mode={self._mode} (already loaded: {best.name})")
            return

        # unload current, then load best
        if self._current:
            self._mgr.unload_from_engine(self._runtime)
        self._load(best.path)

    def _best_for_categories(self, entries: list[LoRAEntry],
                             categories: list[str]) -> Optional[LoRAEntry]:
        """Pick the best adapter from the preferred categories.

        Selection criteria (in order):
        1. First category in the priority list that has an adapter
        2. Within that category, the most recently modified adapter
        """
        for cat in categories:
            matching = [e for e in entries if e.category == cat
                        and not e.header_error]
            if matching:
                # most recently modified wins
                matching.sort(key=lambda e: e.modified, reverse=True)
                return matching[0]
        return None

    def _load(self, path: str) -> None:
        """Load an adapter by path (auto-detects rank from header)."""
        full = project_root() / path
        p = str(full if full.is_file() else path)
        hdr = read_adapter_header(p)
        rank = hdr.get("rank") or 32
        alpha = rank * 2
        self.status.emit(f"loading {Path(path).name} (rank {rank})…")
        self._mgr.load_on_engine(self._runtime, path, rank, alpha)

    # ── signal handlers ───────────────────────────────────────────────
    def _on_loaded(self, info: dict) -> None:
        path = info.get("path", "")
        self._current = path if path else self._current
        self.adapter_changed.emit(path or "")
        self.status.emit(f"loaded: {Path(path).name if path else '?'}")

    def _on_unloaded(self) -> None:
        self._current = None
        self.adapter_changed.emit("")
        self.status.emit("adapter unloaded")

    def _on_failed(self, err: str) -> None:
        self.failed.emit(err)
        self.status.emit(f"error: {err[:120]}")

    # ── query ─────────────────────────────────────────────────────────
    def adapters_by_category(self) -> dict[str, list[LoRAEntry]]:
        """Group all scanned adapters by category."""
        entries = scan_lora_adapters()
        out: dict[str, list[LoRAEntry]] = {c: [] for c in LORA_CATEGORIES}
        out["uncategorized"] = []
        for e in entries:
            out.setdefault(e.category, []).append(e)
        return out

    def recommend_for_mode(self, mode: str) -> Optional[LoRAEntry]:
        """Return the recommended adapter for a mode (without loading)."""
        entries = scan_lora_adapters()
        return self._best_for_categories(
            entries, MODE_CATEGORY_MAP.get(mode, []))
