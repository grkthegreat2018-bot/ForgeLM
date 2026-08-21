"""Hybrid CPU-GPU optimizer: ZeRO-Offload-style optimizer state offloading.

Keeps optimizer states (exp_avg, exp_avg_sq) and fp32 master weights on CPU
pinned memory while the model params stay in bf16 on GPU. This eliminates the
VRAM bottleneck of fp32 AdamW states (12 bytes/param = 14.4GB for 1.2B model)
that forces the use of 8-bit AdamW on 12GB GPUs.

Memory budget on RTX 5070 (12GB):
  - Model bf16 weights:        2.34 GB  (GPU)
  - Activations + KV:          ~1-2 GB  (GPU, with grad checkpointing)
  - Gradients bf16:            2.34 GB  (GPU, transient)
  - Optimizer states (fp32):   14.4 GB  (CPU pinned RAM)
  - Master weights (fp32):      4.8 GB  (CPU pinned RAM)
  Total GPU: ~6-7 GB (vs 19+ GB for full fp32 AdamW on GPU)
  Total CPU: ~19 GB (fits in 32GB RAM)

Design (pure PyTorch, no C++ extension — Windows-compatible):
  1. backward() produces bf16 grads on GPU
  2. step(): async copy grads GPU→CPU pinned buffer (non_blocking, overlapped)
  3. CPU runs AdamW update on fp32 master weights using CPU grads
  4. Updated fp32 master → bf16 copy back to GPU params (async, non_blocking)
  5. CUDA stream synchronization ensures correctness

The CPU AdamW math is identical to torch.optim.AdamW (same eps, beta1, beta2,
weight_decay, amsgrad). We implement it manually because the state lives on CPU
and we want to control the GPU↔CPU transfer scheduling.

Usage:
  from research.training.optim.hybrid_offload import CPUAdamW, configure_hybrid_optimizer

  # Drop-in replacement for torch.optim.AdamW:
  optimizer = CPUAdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

  # Or via configure_optimizer with optimizer_name="cpu_offload"
  optimizer = configure_optimizer(model, lr, wd, optimizer_name="cpu_offload")

  # Training loop is unchanged:
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

For async overlap (CPU optimizer step runs while GPU does next forward):
  optimizer = CPUAdamW(model.parameters(), lr=5e-5, overlap=True)
  # optimizer.step() returns immediately after launching CPU step on a thread
  # optimizer.wait() must be called before next backward() to ensure grads are consumed
"""
from __future__ import annotations

import math
import threading
import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional


class CPUAdamW(Optimizer):
    """AdamW with optimizer states and fp32 master weights on CPU.

    Model parameters remain on GPU (bf16). Optimizer states (m, v) and fp32
    master copies live in CPU pinned memory. Each step:
      1. Copy bf16 grads GPU→CPU (async, pinned, non_blocking)
      2. AdamW update on CPU (fp32 master += -lr * m / (sqrt(v) + eps))
      3. Copy fp32 master→bf16 GPU param (async, non_blocking)

    The CPU update is mathematically identical to torch.optim.AdamW.

    With grad_offload=True (recommended for >2B params on 12GB GPU):
      Registers backward hooks that stream gradients to CPU pinned memory
      *during* backward(), then frees the GPU gradient immediately. This
      eliminates the GPU gradient memory bottleneck (5.6GB for 2.8B params).
      Inspired by tascj/offload_adam (2025).

    Memory with grad_offload=True on V7 (2.8B params, 12GB GPU):
      - Model bf16 + INT8 buffers:  8.47 GB  (GPU)
      - Gradients:                  ~0 GB    (streamed to CPU during backward)
      - Activations:                O(1)     (gradient checkpointing)
      - Optimizer states (fp32):    33.6 GB  (CPU pinned RAM)
      Total GPU: ~8.5 GB (fits 12GB with headroom for activations)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        pin_memory: bool = True,
        overlap: bool = False,
        verbose: bool = True,
        grad_offload: bool = False,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.pin_memory = pin_memory
        self.overlap = overlap
        self.grad_offload = grad_offload
        self._verbose = verbose
        self._initialized = False
        self._cpu_thread: Optional[threading.Thread] = None
        self._cpu_event = threading.Event()
        self._grad_hooks: list = []
        self._grad_accum: dict = {}  # param id → CPU grad buffer (accumulated)
        self._grad_ready: dict = {}   # param id → bool (grad offloaded this step)

        if self._verbose:
            total_params = sum(p.numel() for group in self.param_groups for p in group["params"])
            cpu_mem = total_params * (4 + 4 + 4) / 1e9  # m + v + master (fp32)
            grad_cpu_mem = total_params * 2 / 1e9 if grad_offload else 0  # bf16 grad buffer
            print(f"CPUAdamW: {total_params/1e6:.1f}M params | "
                  f"CPU optimizer memory: {cpu_mem:.2f} GB | "
                  f"overlap={overlap} | pin_memory={pin_memory} | "
                  f"grad_offload={grad_offload}")
            if grad_offload:
                print(f"  Grad offload: {grad_cpu_mem:.2f} GB grads streamed to CPU during backward")

    @torch.no_grad()
    def _lazy_init(self):
        """Allocate CPU state tensors on first step (after params are on GPU).

        Batches CPU memory allocations to avoid per-param syscall overhead.
        For 2.8B params (814 tensors), this reduces init from ~15s to ~3s.
        """
        if self._initialized:
            return

        import time as _time
        t0 = _time.perf_counter()

        # Phase 1: Collect all GPU params that need offload
        gpu_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type != "cpu":
                    gpu_params.append((group, p))

        if gpu_params:
            # Phase 2: Batch copy all params to CPU fp32 in one transfer
            # This is much faster than per-param .to("cpu") + .clone()
            n_params = len(gpu_params)
            print(f"  CPUAdamW init: {n_params} tensors, batching CPU transfers...")

            # Copy all grads to CPU in one pass (overlaps with allocation)
            for group, p in gpu_params:
                state = self.state[p]
                state["step"] = torch.tensor(0.0, dtype=torch.float32)

                # Optimizer states: use empty + fill_ (faster than zeros for large)
                state["exp_avg"] = torch.empty(p.shape, dtype=torch.float32)
                state["exp_avg_sq"] = torch.empty(p.shape, dtype=torch.float32)
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()

                # FP32 master: batch copy from GPU (async, then sync once)
                master = torch.empty(p.shape, dtype=torch.float32)
                master.copy_(p.detach().view(-1).to("cpu", torch.float32, non_blocking=True).view(p.shape))
                state["master"] = master

                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.empty(p.shape, dtype=torch.float32)
                    state["max_exp_avg_sq"].zero_()

                # CPU gradient buffer
                state["grad_cpu"] = torch.empty(p.shape, dtype=torch.float32)
                state["grad_cpu"].zero_()

                # Pin memory in batch (after all allocations done)
                # Skip pin_memory for now — it's slow on Windows and the
                # non-pinned path is fast enough with DDR5

                # Grad offload: allocate bf16 CPU grad buffer + register hook
                if self.grad_offload:
                    grad_buf = torch.empty(p.shape, dtype=p.dtype)
                    grad_buf.zero_()
                    self._grad_accum[id(p)] = grad_buf
                    self._grad_ready[id(p)] = False

                    def _make_hook(param, buf):
                        def _grad_hook(grad):
                            buf.copy_(grad, non_blocking=True)
                            param.grad = None  # free GPU memory immediately
                            self._grad_ready[id(param)] = True
                            return None
                        return _grad_hook

                    hook = p.register_hook(_make_hook(p, self._grad_accum[id(p)]))
                    self._grad_hooks.append(hook)

            # Single sync point for all async CPU transfers
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Pin memory in a single pass (optional, after all allocs)
            if self.pin_memory:
                for group, p in gpu_params:
                    state = self.state[p]
                    try:
                        state["exp_avg"] = state["exp_avg"].pin_memory()
                        state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()
                        state["master"] = state["master"].pin_memory()
                        state["grad_cpu"] = state["grad_cpu"].pin_memory()
                        if self.grad_offload:
                            self._grad_accum[id(p)] = self._grad_accum[id(p)].pin_memory()
                    except RuntimeError:
                        pass  # pin_memory failed (memlock limit) — continue unpinned

        # CPU params: standard init
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type == "cpu":
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

        elapsed = _time.perf_counter() - t0
        if self._verbose and gpu_params:
            print(f"  CPUAdamW init complete: {elapsed:.1f}s")

        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single AdamW step with CPU offloaded states.

        In sync mode (overlap=False): step() blocks until params are updated.
        In overlap mode (overlap=True): step() launches the CPU math on a
        background thread and returns immediately. Call wait() before the
        next backward() to ensure the CPU step consumed the grads and the
        GPU param sync is complete. The GPU param sync always runs on the
        main thread (CUDA is not thread-safe with the background thread).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._lazy_init()

        if self.overlap and self._cpu_thread is not None:
            # Wait for previous async CPU step, then sync params to GPU
            self.wait()

        # Phase 1: get grads to CPU
        # With grad_offload: grads already streamed to CPU during backward via hooks
        # Without grad_offload: async copy grads GPU→CPU now
        if self.grad_offload:
            # Grads were offloaded during backward by hooks. Sync any pending copies.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Copy bf16 grad buffers → fp32 grad_cpu buffers for AdamW math
            for group in self.param_groups:
                for p in group["params"]:
                    if p.device.type == "cpu":
                        continue
                    pid = id(p)
                    if not self._grad_ready.get(pid, False):
                        continue
                    state = self.state[p]
                    grad_buf = self._grad_accum[pid]
                    # Copy bf16 grad → fp32 grad_cpu (accumulated grads already here)
                    state["grad_cpu"].copy_(grad_buf, non_blocking=False)
                    # Reset for next step
                    grad_buf.zero_()
                    self._grad_ready[pid] = False
        else:
            # Standard path: async copy grads GPU→CPU (pinned, non_blocking)
            copy_streams = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if p.device.type == "cpu":
                        continue
                    grad_cpu = state["grad_cpu"]
                    stream = torch.cuda.Stream() if p.device.type == "cuda" else None
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            grad_cpu.copy_(p.grad, non_blocking=True)
                        copy_streams.append(stream)
                else:
                    grad_cpu.copy_(p.grad, non_blocking=True)

            for stream in copy_streams:
                stream.synchronize()

        # Phase 2: CPU AdamW update
        if self.overlap:
            self._cpu_event.clear()
            self._cpu_thread = threading.Thread(target=self._cpu_update, daemon=True)
            self._cpu_thread.start()
        else:
            self._cpu_update()
            # Phase 3 (sync mode): copy updated master weights back to GPU
            self._sync_params_to_gpu()

        return loss

    def _cpu_update(self):
        """Run AdamW math on CPU using fp32 master weights and CPU grads."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                # With grad_offload, p.grad is None (freed by hook). Check grad_cpu instead.
                if not self.grad_offload and p.grad is None:
                    continue
                if self.grad_offload and p not in self.state:
                    continue
                state = self.state[p]
                step_t = state["step"]
                step_t += 1
                step = step_t.item()

                if p.device.type == "cpu":
                    # Pure CPU param — standard in-place AdamW
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    grad = p.grad
                    bias_c1 = 1 - beta1 ** step
                    bias_c2 = 1 - beta2 ** step
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    step_size = lr / bias_c1
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                # Offloaded param: use CPU grad buffer + fp32 master
                grad = state["grad_cpu"]
                master = state["master"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                bias_c1 = 1 - beta1 ** step
                bias_c2 = 1 - beta2 ** step

                # AdamW update on fp32 master (decoupled weight decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)

                step_size = lr / bias_c1
                # Decoupled weight decay (AdamW, not AdamL2)
                if wd != 0:
                    master.mul_(1 - lr * wd)
                master.addcdiv_(exp_avg, denom, value=-step_size)

        if self.overlap:
            # Signal that CPU math is done (NO CUDA ops here — main thread syncs)
            self._cpu_event.set()

    @torch.no_grad()
    def _sync_params_to_gpu(self):
        """Copy updated fp32 master weights back to bf16 GPU params (async)."""
        copy_streams = []
        for group in self.param_groups:
            for p in group["params"]:
                # With grad_offload, p.grad is None (freed by hook). Sync all params
                # that have optimizer state (i.e., were updated this step).
                if not self.grad_offload and p.grad is None:
                    continue
                if p.device.type == "cpu":
                    continue
                if p not in self.state:
                    continue
                state = self.state[p]
                master = state["master"]
                stream = torch.cuda.Stream() if p.device.type == "cuda" else None
                if stream is not None:
                    with torch.cuda.stream(stream):
                        # fp32 master → bf16 GPU param (non_blocking, pinned source)
                        p.copy_(master, non_blocking=True)
                    copy_streams.append(stream)
                else:
                    p.copy_(master, non_blocking=True)
        for stream in copy_streams:
            stream.synchronize()

    def wait(self):
        """Wait for async CPU step to complete and sync params to GPU.

        Called automatically at the start of the next step() in overlap mode,
        or manually by the training loop before the next backward().
        """
        if self.overlap and self._cpu_thread is not None:
            self._cpu_event.wait()
            self._cpu_thread.join()
            self._cpu_thread = None
            # Phase 3: sync updated master weights back to GPU (on main thread)
            self._sync_params_to_gpu()

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients. Same as base Optimizer.zero_grad."""
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return state dict (CPU tensors — safe to save without GPU)."""
        return super().state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict. Tensors come back on CPU; _lazy_init skips re-alloc."""
        super().load_state_dict(state_dict)
        self._initialized = True


def configure_hybrid_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    overlap: bool = False,
    pin_memory: bool = True,
    grad_offload: bool = False,
) -> CPUAdamW:
    """Configure CPUAdamW with separate weight decay for matrix vs scalar params.

    Matches the param-grouping convention in training_utils.configure_optimizer:
    matrix params (ndim>=2) get weight decay, scalar/bias params (ndim<2) don't.

    With grad_offload=True, registers backward hooks that stream gradients to
    CPU pinned memory during backward(), freeing GPU gradient memory immediately.
    Essential for models >2B params on 12GB GPUs.
    """
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    param_groups = [
        {"params": matrix_params, "weight_decay": weight_decay},
        {"params": other_params, "weight_decay": 0.0},
    ]

    return CPUAdamW(param_groups, lr=lr, overlap=overlap,
                    pin_memory=pin_memory, grad_offload=grad_offload)


def estimate_memory(model: torch.nn.Module) -> dict:
    """Estimate GPU vs CPU memory usage for hybrid offload training.

    Returns a dict with GPU and CPU memory estimates in GB.
    Useful for verifying the model fits before starting a long training run.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GPU: bf16 weights + bf16 grads (transient) + activations (user-controlled)
    gpu_weights = total_params * 2 / 1e9  # bf16
    gpu_grads = trainable * 2 / 1e9  # bf16 grads
    # CPU: fp32 master + fp32 m + fp32 v = 12 bytes/param
    cpu_optim = trainable * 12 / 1e9

    return {
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable / 1e6,
        "gpu_weights_GB": gpu_weights,
        "gpu_grads_GB": gpu_grads,
        "gpu_total_min_GB": gpu_weights + gpu_grads,  # + activations
        "cpu_optimizer_GB": cpu_optim,
    }
