"""Discovery-based self-play package.

Autonomous, goal-free exploration where the LLM uses tools (scripting,
theorizing, sudo-thinking, web research, DB queries, schema refactoring)
and decides for itself what to save to its own SQLite database — which it
may also refactor via audited DDL.

When the DB is deemed high-quality and large enough, the loop auto-triggers
fine-tuning (saved as epoch#). New epochs are compared to the best on
quality + skill + compute; winner stays, loser is archived. Every 12 epochs
a distill run filters bloat so the model remembers only DB-curated content.
Anti-regression blocks exact repeats and rolls back stuck bursts.

Public API:
    DiscoveryLoop   — the agentic loop (discovery_loop)
    DiscoveryDB     — the LLM-modifiable SQLite memory (discovery_db)
    ToolRegistry    — bound tools for one session (discovery_tools)
    EpochManager    — fine-tune / distill / best-vs-loser orchestration (epoch_manager)
    quality_eval    — epoch scoring (quality_eval)
    finetune        — SFT from DB content (finetune)
    distill         — bloat-filtering self-distillation (distill)
    anti_regression — fingerprint block + stuck rollback (anti_regression)
"""
from research.self_play.discovery.anti_regression import (
    FingerprintSet, StuckDetector)
from research.self_play.discovery.discovery_db import DiscoveryDB
from research.self_play.discovery.discovery_loop import DiscoveryLoop
from research.self_play.discovery.discovery_tools import ToolRegistry
from research.self_play.discovery.epoch_manager import EpochManager

__all__ = ["DiscoveryDB", "DiscoveryLoop", "ToolRegistry", "EpochManager",
           "FingerprintSet", "StuckDetector"]
