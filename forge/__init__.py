"""ForgeAI main package — re-exports from research/ during R40 migration.

This is the Phase 1 skeleton of the forge/ package. During the R40
restructuring, code will be migrated from research/ to forge/ in phases.
This file currently re-exports from the existing research/ paths so that
`from forge import X` works alongside `from research import X`.

Phase 1: Create skeleton (this file) — no code moved, just wrappers.
Phase 2: Move engine files (research/inference/ → forge/engine/).
Phase 3: Move keys, training, quantization, etc.
Phase 4: Move GUI (forge_gui/ → gui/).
Phase 5: Clean up scripts and root.
Phase 6: Remove compatibility shims.
"""
from __future__ import annotations
