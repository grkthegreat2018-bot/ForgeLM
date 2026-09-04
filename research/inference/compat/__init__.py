"""Compatibility adapters for external model architectures."""
from .arch_adapters import (
    convert_qwen3_checkpoint,
    convert_gemma3_checkpoint,
    convert_llama4_checkpoint,
    convert_checkpoint,
    detect_architecture,
    detect_n_layers,
    gemma3_layer_types,
)

__all__ = [
    "convert_qwen3_checkpoint",
    "convert_gemma3_checkpoint",
    "convert_llama4_checkpoint",
    "convert_checkpoint",
    "detect_architecture",
    "detect_n_layers",
    "gemma3_layer_types",
]
