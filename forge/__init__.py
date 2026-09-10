"""ForgeAI main package — model architecture, training, and inference.

Subpackages:
    config          — ModelConfig, presets
    model_loader    — ModelLoader, ConfigurableResearchLLM, ModularBlock
    checkpoint_io   — checkpoint I/O, cleanup
    engine          — ForgeEngine, KV backend, attention, schedulers
    keys            — 75+ weight transform and runtime keys
    training        — SFT/DPO/RLVR runners, optimizers, losses
    decoding        — Medusa, Eagle, MTP speculative decoding
    quant           — RotorQuant, FP8, INT4, KV compress
    moe             — MoE layer, AirMoE infinite expert library
    self_play       — GRPO, discovery loop, infinite curriculum
    evolution       — ForgeEvolve engine, domains, simulators
    runtime         — VRAM manager, CUDA graphs, forward cache
    distillation    — agentic distillation, distill client
    evaluation      — checkpoint testing, goal scoring
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Apply centralized runtime configuration (CUDA env vars, cache dirs, TF32
# toggles, orphaned-tmp cleanup).  This used to be a direct
# ``cleanup_orphaned_tmp()`` call plus scattered side effects in
# ``forge.model_loader``; it now lives in one idempotent place.  See critique
# finding F17 and ``forge/runtime/configure.py``.
try:
    from forge.runtime.configure import configure
    configure()
except Exception:
    logger.debug("Runtime configuration failed", exc_info=True)
