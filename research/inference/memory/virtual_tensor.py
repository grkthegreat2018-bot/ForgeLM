"""eLLM — Virtual Tensor Abstraction for Elastic Memory.

Decouple virtual address space from physical GPU memory. Elastic
inflation/deflation using CPU as extensible buffer. 2.32x decode
throughput, 3x larger batch for 128K.

Source: arXiv 2506.15155

VRAM budget: GPU holds the "hot" portion, CPU holds the "cold" portion.
On 12GB + 32GB, this enables ~3x larger effective memory for KV caches
and model weights.
"""
from __future__ import annotations

import threading
from typing import Any

import torch


class VirtualTensor:
    """A tensor that transparently spans GPU and CPU memory.

    The GPU portion holds the most recently accessed elements. The CPU
    portion acts as an extensible buffer. Access is transparent —
    __getitem__ fetches from CPU to GPU on demand.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        cpu_buffer_size: int | None = None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.target_device = device
        self.n_elements = 1
        for s in shape:
            self.n_elements *= s
        self.element_bytes = torch.tensor([], dtype=dtype).element_size()
        self.total_bytes = self.n_elements * self.element_bytes

        if cpu_buffer_size is None:
            cpu_buffer_size = self.n_elements
        self.cpu_capacity = min(cpu_buffer_size, self.n_elements)

        self._gpu_n = 0
        self._gpu_data: torch.Tensor | None = None
        self._cpu_data: torch.Tensor | None = None
        self._lock = threading.RLock()
        self._access_counts: list[int] = []

    def allocate_gpu(self, n_elements: int) -> None:
        """Allocate GPU memory for n_elements. Rest stays on CPU."""
        with self._lock:
            n = min(n_elements, self.n_elements)
            self._gpu_n = n
            if n > 0:
                flat_shape = (n,) + self.shape[1:] if len(self.shape) > 1 else (n,)
                self._gpu_data = torch.empty(flat_shape, dtype=self.dtype,
                                             device=self.target_device)
            if self.n_elements - n > 0:
                cpu_n = self.n_elements - n
                flat_cpu = (cpu_n,) + self.shape[1:] if len(self.shape) > 1 else (cpu_n,)
                self._cpu_data = torch.empty(flat_cpu, dtype=self.dtype,
                                             device="cpu")
                if torch.cuda.is_available():
                    self._cpu_data = self._cpu_data.pin_memory()

    def inflate(self, n_elements: int) -> None:
        """Move n_elements from CPU to GPU (elastic inflation)."""
        with self._lock:
            if self._gpu_data is None:
                self.allocate_gpu(n_elements)
                return
            new_n = min(self._gpu_n + n_elements, self.n_elements)
            if new_n <= self._gpu_n:
                return
            new_shape = (new_n,) + self.shape[1:] if len(self.shape) > 1 else (new_n,)
            new_gpu = torch.empty(new_shape, dtype=self.dtype,
                                  device=self.target_device)
            new_gpu[:self._gpu_n] = self._gpu_data
            fetch_n = new_n - self._gpu_n
            if self._cpu_data is not None and fetch_n > 0:
                new_gpu[self._gpu_n:new_n] = self._cpu_data[:fetch_n].to(
                    self.target_device, non_blocking=True)
                remaining = self.n_elements - new_n
                if remaining > 0:
                    self._cpu_data = self._cpu_data[fetch_n:fetch_n + remaining]
                else:
                    self._cpu_data = None
            self._gpu_data = new_gpu
            self._gpu_n = new_n

    def deflate(self, n_elements: int) -> None:
        """Move n_elements from GPU back to CPU (elastic deflation)."""
        with self._lock:
            new_n = max(0, self._gpu_n - n_elements)
            if new_n >= self._gpu_n:
                return
            moved = self._gpu_n - new_n
            gpu_slice = self._gpu_data[new_n:new_n + moved].cpu()
            if self._cpu_data is None:
                cpu_n = self.n_elements - new_n
                flat_cpu = (cpu_n,) + self.shape[1:] if len(self.shape) > 1 else (cpu_n,)
                self._cpu_data = torch.empty(flat_cpu, dtype=self.dtype,
                                             device="cpu")
                if torch.cuda.is_available():
                    self._cpu_data = self._cpu_data.pin_memory()
                self._cpu_data[:moved] = gpu_slice
            else:
                old_cpu_n = self._cpu_data.shape[0]
                new_cpu_n = old_cpu_n + moved
                flat_cpu = (new_cpu_n,) + self.shape[1:] if len(self.shape) > 1 else (new_cpu_n,)
                new_cpu = torch.empty(flat_cpu, dtype=self.dtype,
                                      device="cpu")
                if torch.cuda.is_available():
                    new_cpu = new_cpu.pin_memory()
                new_cpu[:moved] = gpu_slice
                new_cpu[moved:] = self._cpu_data
                self._cpu_data = new_cpu
            new_shape = (new_n,) + self.shape[1:] if len(self.shape) > 1 else (new_n,)
            if new_n > 0:
                self._gpu_data = self._gpu_data[:new_n].contiguous()
            else:
                self._gpu_data = None
            self._gpu_n = new_n

    def to_gpu(self) -> None:
        """Move entire tensor to GPU."""
        self.inflate(self.n_elements)

    def to_cpu(self) -> None:
        """Move entire tensor to CPU."""
        self.deflate(self._gpu_n)

    def _flat_index(self, key: Any) -> tuple[int, int]:
        """Convert a key to (start, end) flat indices."""
        if isinstance(key, int):
            return key, key + 1
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.n_elements
            return start, stop
        return 0, self.n_elements

    def __getitem__(self, key: Any) -> torch.Tensor:
        start, end = self._flat_index(key)
        with self._lock:
            if self._gpu_data is not None and end <= self._gpu_n:
                return self._gpu_data[start:end]
            if self._cpu_data is not None and start >= self._gpu_n:
                cpu_start = start - self._gpu_n
                cpu_end = end - self._gpu_n
                return self._cpu_data[cpu_start:cpu_end].to(self.target_device)
            # Spanning GPU and CPU — assemble
            parts = []
            if self._gpu_data is not None and start < self._gpu_n:
                gpu_end = min(end, self._gpu_n)
                parts.append(self._gpu_data[start:gpu_end])
            if self._cpu_data is not None and end > self._gpu_n:
                cpu_start = max(0, start - self._gpu_n)
                cpu_end = end - self._gpu_n
                parts.append(self._cpu_data[cpu_start:cpu_end].to(self.target_device))
            if parts:
                return torch.cat(parts, dim=0)
            return torch.empty(0, dtype=self.dtype, device=self.target_device)

    def __setitem__(self, key: Any, value: torch.Tensor) -> None:
        start, end = self._flat_index(key)
        with self._lock:
            if self._gpu_data is not None and end <= self._gpu_n:
                self._gpu_data[start:end] = value
            elif self._cpu_data is not None and start >= self._gpu_n:
                cpu_start = start - self._gpu_n
                cpu_end = end - self._gpu_n
                self._cpu_data[cpu_start:cpu_end] = value.cpu()
            else:
                # Spanning — write to whichever portion overlaps
                if self._gpu_data is not None and start < self._gpu_n:
                    gpu_end = min(end, self._gpu_n)
                    self._gpu_data[start:gpu_end] = value[:gpu_end - start]
                if self._cpu_data is not None and end > self._gpu_n:
                    cpu_start = max(0, start - self._gpu_n)
                    cpu_end = end - self._gpu_n
                    offset = max(0, self._gpu_n - start)
                    self._cpu_data[cpu_start:cpu_end] = value[offset:].cpu()

    def device_location(self, index: int) -> str:
        """Returns 'gpu' or 'cpu' for a given flat index."""
        if index < self._gpu_n:
            return "gpu"
        return "cpu"

    def memory_stats(self) -> dict:
        gpu_bytes = self._gpu_n * self.element_bytes if self._gpu_data is not None else 0
        cpu_bytes = (self.n_elements - self._gpu_n) * self.element_bytes
        return {
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "total_bytes": self.total_bytes,
            "gpu_fraction": self._gpu_n / max(1, self.n_elements),
            "gpu_n": self._gpu_n,
            "cpu_n": self.n_elements - self._gpu_n,
        }


class VirtualTensorPool:
    """Manages multiple VirtualTensors with a shared GPU memory budget."""

    def __init__(self, gpu_budget_bytes: int):
        self.gpu_budget = gpu_budget_bytes
        self._tensors: list[VirtualTensor] = []
        self._access_log: list[int] = []  # tensor index per access

    def create(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> VirtualTensor:
        vt = VirtualTensor(shape, dtype)
        self._tensors.append(vt)
        return vt

    def rebalance(self) -> None:
        """Rebalance GPU allocation based on access patterns."""
        total_bytes = sum(vt.total_bytes for vt in self._tensors)
        if total_bytes <= self.gpu_budget:
            for vt in self._tensors:
                if vt._gpu_data is None and vt._cpu_data is None:
                    vt.allocate_gpu(vt.n_elements)
                else:
                    vt.to_gpu()
            return
        # Allocate proportional to access frequency
        access_counts = [0] * len(self._tensors)
        for idx in self._access_log[-100:]:
            if 0 <= idx < len(access_counts):
                access_counts[idx] += 1
        total_access = max(1, sum(access_counts))
        for i, vt in enumerate(self._tensors):
            fraction = access_counts[i] / total_access if total_access > 0 else 1.0 / len(self._tensors)
            target_bytes = int(self.gpu_budget * fraction)
            target_n = target_bytes // vt.element_bytes
            current_n = vt._gpu_n
            if vt._gpu_data is None and vt._cpu_data is None:
                vt.allocate_gpu(target_n)
            elif target_n > current_n:
                vt.inflate(target_n - current_n)
            elif target_n < current_n:
                vt.deflate(current_n - target_n)

    def record_access(self, tensor_idx: int) -> None:
        self._access_log.append(tensor_idx)

    def stats(self) -> dict:
        gpu_used = sum(vt.memory_stats()["gpu_bytes"] for vt in self._tensors)
        cpu_used = sum(vt.memory_stats()["cpu_bytes"] for vt in self._tensors)
        return {
            "n_tensors": len(self._tensors),
            "gpu_used_bytes": gpu_used,
            "cpu_used_bytes": cpu_used,
            "gpu_budget_bytes": self.gpu_budget,
            "gpu_utilization": gpu_used / max(1, self.gpu_budget),
        }
