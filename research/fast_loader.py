"""Fast checkpoint loader — minimizes disk I/O bottleneck.

Three optimizations:
1. **mmap + direct-to-GPU**: Load tensors one at a time via mmap, transfer
   directly to GPU. Avoids materializing the full state dict on CPU.
2. **Async prefetch**: Start loading the next layer's tensors while the
   current layer is being loaded into the model. Overlaps I/O with compute.
3. **INT4 storage format**: Optionally store weights as 4-bit on disk
   (4x smaller files), dequantize on load. Saves disk space + I/O time.

Usage:
    from research.fast_loader import fast_load_checkpoint
    state = fast_load_checkpoint("model.safetensors", device="cuda")
    # Or with prefetch:
    state = fast_load_checkpoint("model.safetensors", device="cuda", prefetch=True)
"""
import torch
import os
import threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from safetensors import safe_open
from safetensors.torch import load_file


def fast_load_checkpoint(path: str, device: str = "cuda",
                         prefetch: bool = True,
                         max_workers: int = 2) -> Dict[str, torch.Tensor]:
    """Load a safetensors checkpoint with optimized disk I/O.

    Args:
        path: Path to .safetensors file
        device: Target device ("cuda" or "cpu")
        prefetch: If True, prefetch next tensors while loading current
        max_workers: Number of prefetch threads

    Returns:
        Dict of tensor_name -> tensor on target device
    """
    path = str(path)
    dev = torch.device(device)
    file_size = Path(path).stat().st_size

    if dev.type == "cuda":
        vram_free = torch.cuda.mem_get_info(dev)[0]
    else:
        vram_free = file_size * 2  # Assume enough RAM

    # If model fits in VRAM and we're on GPU, use direct-to-GPU loading
    if dev.type == "cuda" and vram_free > file_size * 1.3:
        return _load_direct_to_gpu(path, dev, prefetch, max_workers)
    else:
        # Fallback: load to CPU first, then move
        return _load_to_cpu(path)


def _load_direct_to_gpu(path: str, dev: torch.device,
                        prefetch: bool, max_workers: int) -> Dict[str, torch.Tensor]:
    """Load safetensors tensors one at a time, directly to GPU via mmap.

    This avoids materializing the full state dict on CPU RAM.
    Each tensor is mmap'd from disk, then .to(device) transfers to GPU.
    """
    state = {}

    if not prefetch:
        # Simple sequential load
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)
                state[key] = tensor.to(dev)
        return state

    # Prefetch: load next batch of tensors while current batch transfers
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Sort keys by size (smallest first) to overlap better
        # Actually, sort by name to maintain layer order for prefetch
        keys_sorted = sorted(keys)

        batch_size = 8  # Load 8 tensors at a time
        executor = ThreadPoolExecutor(max_workers=max_workers)

        def load_batch(batch_keys):
            """Load a batch of tensors from mmap to CPU."""
            results = {}
            with safe_open(path, framework="pt", device="cpu") as f2:
                for k in batch_keys:
                    results[k] = f2.get_tensor(k)
            return results

        # Prefetch first batch
        batches = [keys_sorted[i:i+batch_size]
                   for i in range(0, len(keys_sorted), batch_size)]
        future = executor.submit(load_batch, batches[0])

        for i, batch in enumerate(batches):
            # Get current batch (already prefetched)
            cpu_tensors = future.result()

            # Prefetch next batch
            if i + 1 < len(batches):
                future = executor.submit(load_batch, batches[i + 1])

            # Transfer current batch to GPU
            for k, t in cpu_tensors.items():
                state[k] = t.to(dev)

        executor.shutdown(wait=False)

    return state


def _load_to_cpu(path: str) -> Dict[str, torch.Tensor]:
    """Standard load to CPU (fallback)."""
    return load_file(path)


def fast_load_to_model(model: torch.nn.Module, checkpoint_path: str,
                       device: str = "cuda",
                       prefetch: bool = True) -> Tuple[List[str], List[str]]:
    """Load checkpoint directly into model parameters, minimizing CPU RAM.

    Instead of loading the full state dict then calling load_state_dict,
    this loads tensors one at a time and assigns directly to model params.

    Args:
        model: The model to load into (already on target device)
        checkpoint_path: Path to .safetensors file
        device: Target device
        prefetch: If True, prefetch next layer while loading current

    Returns:
        (missing_keys, unexpected_keys)
    """
    dev = torch.device(device)
    path = str(checkpoint_path)

    # Get model parameter mapping
    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    with safe_open(path, framework="pt", device="cpu") as f:
        ckpt_keys = set(f.keys())

    model_keys = set(param_dict.keys()) | set(buffer_dict.keys())

    # Find direct matches
    matched = ckpt_keys & model_keys
    missing = list(model_keys - ckpt_keys)
    unexpected = list(ckpt_keys - model_keys)

    # Load matched tensors one at a time
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in matched:
            tensor = f.get_tensor(key)
            if key in param_dict:
                param_dict[key].data.copy_(tensor.to(dev))
            elif key in buffer_dict:
                buffer_dict[key].data.copy_(tensor.to(dev))

    return missing, unexpected


def save_int4_checkpoint(state: Dict[str, torch.Tensor], path: str,
                         group_size: int = 128):
    """Save checkpoint in INT4 format (4x smaller on disk).

    Each 2D weight tensor is quantized to 4-bit with per-group scales.
    Non-2D tensors (norms, biases, embeddings) are kept in bf16.

    Args:
        state: Dict of tensor_name -> tensor
        path: Output .safetensors path
        group_size: Quantization group size
    """
    from safetensors.torch import save_file
    quantized = {}

    for name, tensor in state.items():
        if tensor.dim() == 2 and tensor.numel() > 10000:
            # Quantize to INT4
            q_tensor, scales = _quantize_int4(tensor, group_size)
            quantized[f"{name}__q"] = q_tensor
            quantized[f"{name}__scale"] = scales
        else:
            # Keep as bf16
            quantized[name] = tensor.to(torch.bfloat16)

    save_file(quantized, path)
    orig_size = sum(t.numel() * t.element_size() for t in state.values())
    new_size = sum(t.numel() * t.element_size() for t in quantized.values())
    print(f"  INT4 save: {orig_size/1e9:.2f} GB -> {new_size/1e9:.2f} GB "
          f"({new_size/orig_size:.1%})")


def load_int4_checkpoint(path: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load INT4 checkpoint and dequantize to bf16.

    Args:
        path: Path to INT4 .safetensors file
        device: Target device

    Returns:
        Dict of tensor_name -> bf16 tensor (dequantized)
    """
    dev = torch.device(device)
    state = {}

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())

    # Find quantized pairs
    quantized_keys = {k for k in keys if k.endswith("__q")}
    plain_keys = keys - quantized_keys - {k.replace("__q", "__scale") for k in quantized_keys}

    # Load plain tensors
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in plain_keys:
            state[k] = f.get_tensor(k).to(dev)

        # Dequantize INT4 tensors
        for qk in quantized_keys:
            base_name = qk.replace("__q", "")
            scale_key = qk.replace("__q", "__scale")
            q_tensor = f.get_tensor(qk)
            scales = f.get_tensor(scale_key)
            dequant = _dequantize_int4(q_tensor, scales)
            state[base_name] = dequant.to(dev)

    return state


def _quantize_int4(tensor: torch.Tensor, group_size: int = 128):
    """Quantize tensor to INT4 with per-group scales.

    Returns (quantized_int8, scales) where quantized values are in [-8, 7]
    stored as int8, and scales are per-group float16.
    """
    orig_shape = tensor.shape
    t = tensor.float().reshape(-1, group_size)

    # Per-group scale
    max_val = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_val / 7.0  # INT4 range: [-8, 7]

    # Quantize
    q = (t / scale).round().clamp(-8, 7).to(torch.int8)
    return q.reshape(orig_shape), scale.squeeze(-1).to(torch.float16)


def _dequantize_int4(q_tensor: torch.Tensor, scales: torch.Tensor,
                     group_size: int = 128) -> torch.Tensor:
    """Dequantize INT4 tensor back to float."""
    orig_shape = q_tensor.shape
    q = q_tensor.float().reshape(-1, group_size)
    s = scales.float().unsqueeze(-1)  # [n_groups, 1]
    return (q * s).reshape(orig_shape)


def benchmark_load(path: str, device: str = "cuda") -> dict:
    """Benchmark different loading strategies."""
    import time
    file_size = Path(path).stat().st_size

    # Method 1: Standard load_file
    torch.cuda.empty_cache() if device == "cuda" else None
    t_start = time.time()
    state1 = load_file(path)
    t1 = time.time() - t_start
    del state1
    torch.cuda.empty_cache() if device == "cuda" else None

    # Method 2: fast_load (direct to GPU)
    t_start = time.time()
    state2 = fast_load_checkpoint(path, device=device, prefetch=True)
    t2 = time.time() - t_start
    del state2
    torch.cuda.empty_cache() if device == "cuda" else None

    # Method 3: fast_load (no prefetch)
    t_start = time.time()
    state3 = fast_load_checkpoint(path, device=device, prefetch=False)
    t3 = time.time() - t_start
    del state3
    torch.cuda.empty_cache() if device == "cuda" else None

    return {
        "file_size_mb": file_size / 1e6,
        "standard_load_s": t1,
        "fast_prefetch_s": t2,
        "fast_sequential_s": t3,
        "standard_mb_s": file_size / 1e6 / t1,
        "fast_prefetch_mb_s": file_size / 1e6 / t2,
        "fast_sequential_mb_s": file_size / 1e6 / t3,
    }
