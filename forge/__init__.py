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

# Clean up orphaned .tmp checkpoint files from crashed writes.
try:
    from forge.checkpoint_io import cleanup_orphaned_tmp
    cleanup_orphaned_tmp()
except Exception:
    pass
