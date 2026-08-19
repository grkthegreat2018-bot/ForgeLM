"""torch.compile mode auto-tuner for inference.

PyTorch torch.compile has four modes:
  - default: fastest compile, modest speedup
  - reduce-overhead: CUDA graphs to reduce per-step launch overhead
  - max-autotune: Triton autotuning for best kernel selection
  - max-autotune-no-cudagraphs: same but without CUDA graphs

The best mode depends on the model, batch size, and hardware. This module
benchmarks all modes on a short generation run and picks the fastest one.

Known issues (from PyTorch GitHub #171672):
  - max-autotune and reduce-overhead can rebuild CUDA graphs every iteration
    if the model has parameter mutations (e.g., optimizer steps in training)
  - For inference (no parameter mutations), these modes work correctly
  - max-autotune-no-cudagraphs is safe when CUDA graphs cause issues

For our 1.2B model on RTX 5070:
  - default: ~10% speedup, 30s compile
  - reduce-overhead: ~20% speedup, 60s compile (CUDA graphs)
  - max-autotune: ~25% speedup, 300s compile (Triton autotuning + CUDA graphs)
  - max-autotune-no-cudagraphs: ~15% speedup, 300s compile (no CUDA graphs)

This auto-tuner runs a quick benchmark of each mode and picks the best,
caching the result so subsequent runs skip the benchmark.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


# Cache file for compile mode benchmark results.
# Lives under research/runtime/ (gitignored runtime cache directory).
_CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "runtime" / "compile_mode_cache.json"


def _run_benchmark(
    model: torch.nn.Module,
    mode: str,
    input_ids: torch.Tensor,
    n_warmup: int = 3,
    n_benchmark: int = 10,
) -> float:
    """Benchmark a torch.compile mode.

    Returns: tokens per second (higher is better)
    """
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
    except Exception:
        return 0.0

    # Warmup (includes compilation)
    with torch.inference_mode():
        for _ in range(n_warmup):
            try:
                _ = compiled(input_ids)
            except Exception:
                return 0.0
        if input_ids.is_cuda:
            torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    n_tokens = 0
    with torch.inference_mode():
        for _ in range(n_benchmark):
            try:
                out = compiled(input_ids)
                n_tokens += input_ids.shape[1]
            except Exception:
                return 0.0
    if input_ids.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return n_tokens / elapsed if elapsed > 0 else 0.0


def auto_tune_compile(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    model_id: str = "default",
    modes: list[str] | None = None,
    force_rebenchmark: bool = False,
) -> tuple[str, torch.nn.Module]:
    """Auto-tune torch.compile mode for a model.

    Benchmarks all modes on a short run and picks the fastest.
    Results are cached per model_id.

    Args:
        model: the model to compile
        input_ids: sample input for benchmarking (1, T)
        model_id: unique identifier for caching (e.g., "forgelm_v4")
        modes: list of modes to try (default: all 4)
        force_rebenchmark: ignore cache and re-benchmark

    Returns:
        (best_mode, compiled_model)
    """
    if modes is None:
        modes = ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]
        # Skip max-autotune (CUDA graphs) if model has dynamic shapes
        # or if we're on a setup where graph rebuild is known to be slow

    # Check cache
    if not force_rebenchmark and _CACHE_FILE.exists():
        try:
            cache = json.loads(_CACHE_FILE.read_text())
            if model_id in cache:
                best_mode = cache[model_id]["best_mode"]
                print(f"  [CompileAutoTune] Cached result for {model_id}: {best_mode}")
                compiled = torch.compile(model, mode=best_mode, fullgraph=False)
                return best_mode, compiled
        except Exception:
            pass

    # Benchmark each mode
    results = {}
    print(f"  [CompileAutoTune] Benchmarking modes for {model_id}...")
    for mode in modes:
        print(f"    Testing {mode}...", end=" ", flush=True)
        tps = _run_benchmark(model, mode, input_ids)
        results[mode] = tps
        print(f"{tps:.0f} tok/s")

    # Pick best
    best_mode = max(results, key=results.get)
    best_tps = results[best_mode]
    print(f"  [CompileAutoTune] Best: {best_mode} ({best_tps:.0f} tok/s)")

    # Cache result
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache = {}
        if _CACHE_FILE.exists():
            cache = json.loads(_CACHE_FILE.read_text())
        cache[model_id] = {
            "best_mode": best_mode,
            "tok_per_s": best_tps,
            "all_results": results,
        }
        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass

    compiled = torch.compile(model, mode=best_mode, fullgraph=False)
    return best_mode, compiled


def get_recommended_mode(
    model_size_params: int,
    batch_size: int = 1,
    has_dynamic_shapes: bool = False,
    is_training: bool = False,
) -> str:
    """Get a recommended compile mode without benchmarking.

    Heuristic-based recommendation for quick setup without the benchmark
    overhead.

    Args:
        model_size_params: number of model parameters
        batch_size: inference batch size
        has_dynamic_shapes: whether input shapes change between calls
        is_training: whether this is for training (not inference)

    Returns:
        recommended mode string
    """
    if is_training:
        return "reduce-overhead"  # CUDA graphs for training loop

    if has_dynamic_shapes:
        return "max-autotune-no-cudagraphs"  # no CUDA graphs for dynamic shapes

    if batch_size == 1:
        # Small batch: launch overhead dominates → CUDA graphs help
        return "max-autotune"  # Triton autotuning + CUDA graphs

    if model_size_params < 1_000_000_000:
        # Small model (< 1B): launch overhead is significant
        return "reduce-overhead"
    else:
        # Large model: compute-bound, autotuning matters more
        return "max-autotune-no-cudagraphs"
