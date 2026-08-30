"""Simulator registry — raw metric computation for each domain.

A simulator is a function: simulate(config, domain) -> dict[str, float]
It computes raw metrics (sqnr, compression, convergence_speed, etc.) and
returns them as a dict. Scoring is handled by RewardGuard, NOT here.

This separation is the core of the refactor: simulators are pure metric
computers, scoring is declarative in JSON. No more mixing the two.

Register a simulator with @register("name"). Look up with get_simulator("name").
Simulator functions live in this package (one module per category).
"""
from __future__ import annotations

import importlib
import pkgutil
import threading
from typing import Any, Callable

# Type: simulate(config: dict, domain: BaseDomain) -> dict[str, float]
SimulatorFn = Callable[[dict, Any], dict]

_REGISTRY: dict[str, SimulatorFn] = {}
_LOADED = False
_LOAD_LOCK = threading.Lock()


def register(name: str) -> Callable[[SimulatorFn], SimulatorFn]:
    """Decorator to register a simulator function."""
    def deco(fn: SimulatorFn) -> SimulatorFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def get_simulator(name: str) -> SimulatorFn:
    """Look up a simulator by name. Auto-loads simulators/ on first call."""
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"No simulator '{name}' registered. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_simulators() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


def _ensure_loaded() -> None:
    """Import all simulator modules so @register decorators run."""
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        pkg = "research.evolution.simulators"
        try:
            pkg_mod = importlib.import_module(pkg)
        except ImportError:
            _LOADED = True
            return
        for finder, name, ispkg in pkgutil.iter_modules(pkg_mod.__path__):
            if name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{pkg}.{name}")
            except Exception as e:
                print(f"[simulators] Warning: failed to load {name}: {e}")
        _LOADED = True


def clear_registry() -> None:
    """Clear all registered simulators (for tests)."""
    global _LOADED
    _REGISTRY.clear()
    _LOADED = False
