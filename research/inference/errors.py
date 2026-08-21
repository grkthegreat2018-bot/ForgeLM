"""Structured exception hierarchy for ForgeEngine.

Replaces bare RuntimeError/ValueError with typed exceptions that carry
error codes, context, and recovery suggestions.

Hierarchy:
    ForgeEngineError
    ├── ActivationError      — strategy activation failed
    ├── GenerationError      — generation failed (OOM, decode error, timeout)
    ├── CheckpointError      — checkpoint loading failed
    ├── ConfigurationError   — invalid config or parameters
    └── RecoveryError        — crash recovery failed

Each exception includes:
    - ``code``: machine-readable error code (for logging/retry logic)
    - ``context``: dict of relevant values (model, device, vram, etc.)
    - ``suggestion``: human-readable recovery suggestion
"""
from __future__ import annotations

from typing import Any


class ForgeEngineError(RuntimeError):
    """Base exception for all ForgeEngine errors.

    Inherits from RuntimeError so existing ``pytest.raises(RuntimeError)``
    tests continue to work.
    """

    code: str = "FORGE_UNKNOWN"
    suggestion: str = "Check engine logs with read_log() or diagnose()."

    def __init__(self, message: str, *, context: dict[str, Any] | None = None,
                 suggestion: str | None = None):
        super().__init__(message)
        self.context = context or {}
        if suggestion:
            self.suggestion = suggestion

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "context": self.context,
            "suggestion": self.suggestion,
        }


class ActivationError(ForgeEngineError):
    """Strategy activation failed (KV cache, decoding, quantization, etc.)."""
    code = "ACTIVATION_FAILED"
    suggestion = "Check feature compatibility with your model config."


class GenerationError(ForgeEngineError):
    """Generation failed (decode error, timeout, logits error)."""
    code = "GENERATION_FAILED"
    suggestion = "Try reducing max_new_tokens or check model state."


class GenerationOOMError(GenerationError):
    """Out-of-memory during generation."""
    code = "GENERATION_OOM"
    suggestion = ("Reduce max_new_tokens, use kv_cache='s4r' with kv_bits=4, "
                  "or quantize='int4'. Call engine.sleep(1) to free VRAM.")


class GenerationTimeoutError(GenerationError):
    """Generation exceeded time limit."""
    code = "GENERATION_TIMEOUT"
    suggestion = "Increase timeout or reduce max_new_tokens."


class CheckpointError(ForgeEngineError):
    """Checkpoint loading or metadata reading failed."""
    code = "CHECKPOINT_ERROR"
    suggestion = "Verify checkpoint path exists and is a valid safetensors file."


class ConfigurationError(ForgeEngineError):
    """Invalid configuration or parameters."""
    code = "CONFIG_ERROR"
    suggestion = "Check config_name, device, and model parameters."


class RecoveryError(ForgeEngineError):
    """Crash recovery failed."""
    code = "RECOVERY_FAILED"
    suggestion = "Recovery files may be corrupted. Call clear_recovery() and retry."
