"""Progressive tensor loading — load essential tensors first, stream rest.

R35-6: Enhances ForgeLoader's GGUF/safetensors path with progressive
tensor loading. Load only the tensors needed for the first forward pass,
stream the rest in background. Reduces time-to-first-token.

VRAM budget: Only essential tensors (embedding + first 25% of layers +
final layer + unembedding) are loaded to GPU initially. Remaining
tensors stream in background via a daemon thread.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import torch


class ProgressiveLoader:
    """Progressive tensor loader with background streaming."""

    DEFAULT_ESSENTIAL_FRACTION = 0.25

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        essential_tensors: list[str] | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._essential_names = essential_tensors
        self._tensors: dict[str, torch.Tensor] = {}
        self._all_names: list[str] = []
        self._loaded_names: set[str] = set()
        self._essential_loaded: set[str] = set()
        self._lock = threading.RLock()
        self._bg_thread: threading.Thread | None = None
        self._bg_done = threading.Event()
        self._load_fn: Callable | None = None

    def set_load_fn(self, fn: Callable[[str], torch.Tensor]):
        """Set the function that loads a single tensor by name."""
        self._load_fn = fn

    def set_tensor_names(self, names: list[str]):
        """Set the full list of tensor names in the checkpoint."""
        self._all_names = list(names)
        if self._essential_names is None:
            self._essential_names = self._compute_essential(names)

    def _compute_essential(self, names: list[str]) -> list[str]:
        """Heuristic: embedding + first 25% of layers + last layer + unembedding."""
        essential = []
        for n in names:
            lower = n.lower()
            if any(k in lower for k in ("embed", "wte", "input_embed")):
                essential.append(n)
            if any(k in lower for k in ("head", "lm_head", "unembed", "output")):
                essential.append(n)
            if any(k in lower for k in ("rope", "cos", "sin", "position")):
                essential.append(n)
        # Add first 25% and last of decoder layers
        layer_names = [n for n in names if "layers." in n or "h." in n]
        if layer_names:
            layer_indices = {}
            for n in layer_names:
                parts = n.replace("layers.", ".").replace("h.", ".").split(".")
                for p in parts:
                    if p.isdigit():
                        idx = int(p)
                        layer_indices.setdefault(idx, []).append(n)
                        break
            sorted_indices = sorted(layer_indices.keys())
            if sorted_indices:
                n_essential = max(1, int(len(sorted_indices) * self.DEFAULT_ESSENTIAL_FRACTION))
                for idx in sorted_indices[:n_essential]:
                    essential.extend(layer_indices[idx])
                essential.extend(layer_indices[sorted_indices[-1]])
        return list(set(essential))

    def load_essential(self) -> dict[str, torch.Tensor]:
        """Load only essential tensors to GPU."""
        if self._load_fn is None:
            raise RuntimeError("set_load_fn must be called first")
        result = {}
        for name in (self._essential_names or []):
            if name in self._loaded_names:
                continue
            try:
                t = self._load_fn(name)
                if self.device == "cuda" and torch.cuda.is_available():
                    t = t.to(self.device)
                with self._lock:
                    self._tensors[name] = t
                    self._loaded_names.add(name)
                    self._essential_loaded.add(name)
                result[name] = t
            except Exception:
                pass
        return result

    def load_remaining_background(self, callback: Callable | None = None):
        """Start background thread to load remaining tensors."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return

        def _bg():
            for name in self._all_names:
                if name in self._loaded_names:
                    continue
                try:
                    t = self._load_fn(name)
                    if self.device == "cuda" and torch.cuda.is_available():
                        t = t.to(self.device)
                    with self._lock:
                        self._tensors[name] = t
                        self._loaded_names.add(name)
                except Exception:
                    pass
            self._bg_done.set()
            if callback:
                callback()

        self._bg_thread = threading.Thread(target=_bg, daemon=True)
        self._bg_thread.start()

    def get_tensor(self, name: str) -> torch.Tensor | None:
        """Get a tensor by name. Loads on-demand if not yet loaded."""
        with self._lock:
            if name in self._tensors:
                return self._tensors[name]
        # On-demand load (blocking)
        if self._load_fn is None or name not in self._all_names:
            return None
        try:
            t = self._load_fn(name)
            if self.device == "cuda" and torch.cuda.is_available():
                t = t.to(self.device)
            with self._lock:
                self._tensors[name] = t
                self._loaded_names.add(name)
            return t
        except Exception:
            return None

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._loaded_names

    def loading_progress(self) -> float:
        if not self._all_names:
            return 0.0
        with self._lock:
            return len(self._loaded_names) / len(self._all_names)

    def wait_until_loaded(self, timeout: float | None = None) -> bool:
        return self._bg_done.wait(timeout=timeout)

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_total": len(self._all_names),
                "n_loaded": len(self._loaded_names),
                "n_essential": len(self._essential_loaded),
                "loading_progress": self.loading_progress(),
                "bytes_loaded": sum(
                    t.numel() * t.element_size()
                    for t in self._tensors.values()
                ),
            }
