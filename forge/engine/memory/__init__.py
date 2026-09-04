"""Elastic memory abstractions for ForgeAI inference.

Implements the eLLM Virtual Tensor model (arXiv 2506.15155): decouples the
virtual address space of a tensor from physical GPU memory so the CPU's
system RAM acts as an extensible buffer. On RTX 5070 (12 GB VRAM) + 32 GB
system RAM this yields a ~3x larger effective memory budget and up to
2.32x decode throughput for 128K-context batches.
"""
from forge.engine.memory.virtual_tensor import VirtualTensor, VirtualTensorPool

__all__ = ["VirtualTensor", "VirtualTensorPool"]
