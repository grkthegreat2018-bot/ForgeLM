"""Forge engine package — re-exports from research.inference during R40 migration.

Phase 1: This file re-exports key engine symbols from research.inference
so that `from forge.engine import ForgeEngine` works.
Phase 2: Files will be physically moved here with compatibility shims.
"""
from __future__ import annotations

# Re-export the main engine symbols from the current location
try:
    from research.inference.forge_engine import ForgeEngine
    from research.inference.forge_server import ForgeServer
except ImportError:
    pass

try:
    from research.inference.kv_backend import build_kv_cache, KVCacheStrategy
except ImportError:
    pass
