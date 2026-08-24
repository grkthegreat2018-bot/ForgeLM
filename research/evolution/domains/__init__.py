"""Base domain interface. Each domain defines a search space + evaluator.

To add a new search domain (quant, kernel, training, boot, etc.),
subclass BaseDomain and implement:
  - name(): domain name
  - output_dim(): number of continuous params generators produce
  - decode(params): map [0,1] params → concrete config dict
  - encode(config): map config dict → [0,1] params (for context)
  - evaluate(config): run the real evaluation, return {score, behavioral, metadata}
  - behavioral_dims(): MAP-Elites grid spec
  - discrete_choices(): optional discrete params for template generator
"""
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseDomain(ABC):
    """Abstract base for search domains."""

    def __init__(self):
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def _t(self, *args, **kwargs):
        """Create a tensor on this domain's device."""
        import torch
        kwargs.setdefault('device', self._device)
        return torch.tensor(*args, **kwargs)

    def _randn(self, *args, **kwargs):
        """Create randn tensor on this domain's device."""
        import torch
        kwargs.setdefault('device', self._device)
        return torch.randn(*args, **kwargs)

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def output_dim(self) -> int:
        """Number of continuous [0,1] params the generators produce."""

    @abstractmethod
    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        """Map [0,1] params → concrete config dict."""

    @abstractmethod
    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        """Map config dict → [0,1] params (for context/surrogate)."""

    @abstractmethod
    def evaluate(self, config: dict[str, Any]) -> dict:
        """Evaluate a config. Return {score, behavioral, metadata}.
        score: float, higher = better
        behavioral: tuple of floats (for MAP-Elites grid)
        metadata: dict with any extra info
        """

    @abstractmethod
    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        """MAP-Elites grid: [(name, n_bins, min, max), ...]"""

    def discrete_choices(self) -> dict[str, list] | None:
        """Optional discrete params for template generator."""
        return None

    def seed_configs(self) -> list[dict[str, Any]]:
        """Known-good configs to pre-evaluate before search begins.
        Override in subclass to bootstrap the surrogate + archive.
        """
        return []

    def to_cpu(self) -> "BaseDomain":
        """Create a CPU-side copy of this domain for parallel evaluation.

        Override in subclasses that hold GPU tensors. The CPU copy should
        have all tensors moved to CPU so it can run in a separate process
        without touching the GPU.
        """
        return self


# ── Domain registry ──
# Import all domain modules so they're available via DOMAINS dict.
import importlib
import inspect

def _discover_domains() -> dict[str, type[BaseDomain]]:
    """Auto-discover all BaseDomain subclasses across all domain modules."""
    modules = [
        "synthetic", "quant", "kv_eviction", "sparse_attn", "hqe_kv", "kara",
        "attention_domains", "kv_domains", "quant_domains",
        "training_domains", "memory_domains", "decoding_domains", "arch_domains",
    ]
    registry = {}
    for mod_name in modules:
        try:
            mod = importlib.import_module(f".{mod_name}", package=__name__)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (issubclass(obj, BaseDomain) and obj is not BaseDomain
                        and obj.__module__ == mod.__name__):
                    registry[obj.__name__] = obj
        except ImportError as e:
            pass
    return registry

DOMAINS: dict[str, type[BaseDomain]] = _discover_domains()

def get_domain(name: str, **kwargs) -> BaseDomain:
    """Instantiate a domain by class name."""
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain '{name}'. Available: {list(DOMAINS.keys())}")
    return DOMAINS[name](**kwargs)

def list_domains() -> list[str]:
    """List all available domain class names."""
    return sorted(DOMAINS.keys())
