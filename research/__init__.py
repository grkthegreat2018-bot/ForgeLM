"""ForgeAI research package — model architecture, training, and inference.

Subpackages:
    core (root)     — config, model_loader, checkpoint_io
    training        — self-play expert training, DPO, training utils
    decoding        — DSpark, Medusa, MTP speculative decoding
    quantization    — RotorQuant, FP8, INT4, KV compress
    evaluation      — reasoning benchmarks, prompt tests, goal scoring
    moe             — MoE conversion, AirMoE infinite expert library
    runtime         — VRAM manager, CUDA graphs, forward cache, self-model
    architecture    — MTP heads, LFM2.5 porting reference
    self_play       — infinite curriculum, recursive self-play, sandbox
    keys            — 75+ weight transform and runtime keys
    inference       — forge engine, KV backend, decoding strategies
"""

# Clean up orphaned .tmp checkpoint files from crashed writes.
# These accumulate when a checkpoint save is interrupted (kill -9, OOM, etc).
# Safe to call at import — only removes .tmp files in checkpoint directories.
try:
    from research.checkpoint_io import cleanup_orphaned_tmp
    cleanup_orphaned_tmp()
except Exception:
    pass  # don't fail import if cleanup has issues

