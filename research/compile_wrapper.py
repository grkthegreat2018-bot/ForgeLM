"""torch.compile wrapper for inference and training speed.

torch.compile with mode="reduce-overhead" uses CUDA graphs under the hood
+ operator fusion for 30-50% speedup. This module wraps it with proper
error handling and fallbacks for Windows/Triton issues.

Also provides:
- Static shape inference (avoid recompilation on shape changes)
- Compile cache (avoid recompilation across runs)
- Mode comparison benchmark

Usage:
    from research.compile_wrapper import compile_model, compile_for_inference

    # Compile for training
    model = compile_model(model, mode="reduce-overhead")

    # Compile for inference (more aggressive)
    model = compile_for_inference(model)
"""
import torch
import torch.nn as nn
import time
from typing import Dict


def compile_model(model: nn.Module, mode: str = "reduce-overhead",
                  fullgraph: bool = False, dynamic: bool = None,
                  backend: str = "inductor") -> nn.Module:
    """Wrap model with torch.compile.

    Args:
        model: the model to compile
        mode: "default", "reduce-overhead", or "max-autotune"
            - default: basic fusion, safe
            - reduce-overhead: CUDA graphs + fusion, 30-50% faster
            - max-autotune: auto-tune kernels, slow first run, fastest after
        fullgraph: if True, compile entire graph (faster but less compatible)
        dynamic: if True, support dynamic shapes (slower). None=auto.
        backend: "inductor" (default) or "cudagraphs"

    Returns:
        compiled model (or original if compilation fails)
    """
    if not hasattr(torch, "compile"):
        print("  [Compile] torch.compile not available (PyTorch < 2.0)")
        return model

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
        )
        print(f"  [Compile] compiled with mode={mode}, backend={backend}")
        return compiled
    except Exception as e:
        print(f"  [Compile] WARNING: compilation failed: {e}")
        print(f"  [Compile] falling back to eager mode")
        return model


def compile_for_inference(model: nn.Module,
                          mode: str = "reduce-overhead") -> nn.Module:
    """Compile model for inference (most aggressive settings).

    Args:
        model: the model
        mode: "reduce-overhead" (default) or "max-autotune"

    Returns:
        compiled model
    """
    model.eval()
    # For inference: no gradients, full graph, static shapes.
    compiled = compile_model(model, mode=mode, fullgraph=True, dynamic=False)

    # Warmup compilation (first call triggers compile).
    print("  [Compile] warming up (first call triggers compilation)...")
    with torch.no_grad():
        # Create dummy input.
        if hasattr(model, "config"):
            cfg = model.config
            dummy = torch.randint(0, cfg.vocab_size, (1, min(cfg.max_seq_len, 64)),
                                 device=next(model.parameters()).device)
        else:
            dummy = torch.randn(1, 64, 128, device=next(model.parameters()).device)
        t0 = time.time()
        _ = compiled(dummy)
        compile_time = time.time() - t0
        print(f"  [Compile] compilation took {compile_time:.1f}s")

    return compiled


def compile_for_training(model: nn.Module,
                         mode: str = "reduce-overhead") -> nn.Module:
    """Compile model for training (conservative settings).

    Args:
        model: the model
        mode: "default" (safe) or "reduce-overhead" (faster but may break)

    Returns:
        compiled model
    """
    # For training: need gradients, dynamic shapes (variable batch sizes).
    compiled = compile_model(model, mode=mode, fullgraph=False, dynamic=True)
    return compiled


class ShapeCache:
    """Cache compiled models by input shape to avoid recompilation.

    torch.compile recompiles when input shapes change. This cache keeps
    separate compiled versions for each shape, avoiding recompilation.

    Usage:
        cache = ShapeCache(model)
        output = cache.forward(input_ids)  # compiles once per shape
    """

    def __init__(self, model: nn.Module, mode: str = "reduce-overhead"):
        self.base_model = model
        self.mode = mode
        self.compiled_cache: Dict[tuple, nn.Module] = {}
        self.compile_count = 0

    def _get_key(self, input_ids: torch.Tensor) -> tuple:
        """Get cache key from input shape."""
        return (input_ids.shape, input_ids.dtype, input_ids.device)

    def _get_or_compile(self, input_ids: torch.Tensor) -> nn.Module:
        """Get compiled model for this shape, compiling if needed."""
        key = self._get_key(input_ids)

        if key not in self.compiled_cache:
            # Clone model and compile for this shape.
            # Note: torch.compile wraps the model, so we need a fresh wrapper
            # for each shape. In practice, the compiled graph is cached
            # internally by torch, so this is fast after the first compile.
            self.compiled_cache[key] = torch.compile(
                self.base_model, mode=self.mode, fullgraph=True, dynamic=False
            )
            self.compile_count += 1
            print(f"  [ShapeCache] compiled for shape {input_ids.shape} "
                  f"(total compilations: {self.compile_count})")

        return self.compiled_cache[key]

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass with shape-specific compiled model."""
        compiled = self._get_or_compile(input_ids)
        return compiled(input_ids, **kwargs)

    def __call__(self, input_ids: torch.Tensor, **kwargs):
        return self.forward(input_ids, **kwargs)


def benchmark_compile_modes(model, input_ids, n_warmup=5, n_runs=20,
                            device="cuda") -> Dict:
    """Benchmark different torch.compile modes.

    Args:
        model: the model
        input_ids: test input
        n_warmup: warmup iterations
        n_runs: benchmark iterations

    Returns:
        dict with timing for each mode
    """
    results = {}

    modes = ["eager", "default", "reduce-overhead", "max-autotune"]

    for mode in modes:
        print(f"\n  [Compile Benchmark] mode={mode}")

        if mode == "eager":
            test_model = model
        else:
            try:
                test_model = torch.compile(model, mode=mode, fullgraph=True)
            except Exception as e:
                print(f"    skipped: {e}")
                continue

        # Warmup.
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = test_model(input_ids)
            if device.type == "cuda" if hasattr(device, 'type') else False:
                torch.cuda.synchronize()

        # Benchmark.
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = test_model(input_ids)
        if hasattr(device, 'type') and device.type == "cuda":
            torch.cuda.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.time() - t0) / n_runs
        results[mode] = elapsed
        print(f"    {elapsed*1000:.2f}ms/iter")

    # Summary.
    if "eager" in results:
        base = results["eager"]
        print(f"\n  [Compile Benchmark] Summary:")
        for mode, t in results.items():
            speedup = base / t
            print(f"    {mode}: {t*1000:.2f}ms ({speedup:.2f}x)")

    return results


def check_compile_compatibility() -> Dict:
    """Check if torch.compile will work in this environment.

    Returns:
        dict with compatibility info
    """
    info = {
        "torch_version": torch.__version__,
        "compile_available": hasattr(torch, "compile"),
        "triton_available": False,
        "cuda_available": torch.cuda.is_available(),
        "issues": [],
    }

    # Check Triton (required for inductor backend).
    try:
        import triton
        info["triton_available"] = True
        info["triton_version"] = triton.__version__
    except ImportError:
        info["issues"].append("Triton not installed (required for inductor backend)")

    # Check CUDA.
    if not info["cuda_available"]:
        info["issues"].append("CUDA not available (compile works on CPU but slower)")

    # Check for Windows path issue.
    import platform
    if platform.system() == "Windows":
        try:
            import triton.runtime.cache as cache_mod
            # Check if the patch is applied (short temp dir names).
            import inspect
            src = inspect.getsource(cache_mod.FileCacheManager.put)
            if "uuid4()[:8]" in src or "[:8]" in src:
                info["triton_path_patch"] = True
            else:
                info["issues"].append("Triton Windows path patch not applied (see AGENTS.md)")
                info["triton_path_patch"] = False
        except Exception:
            info["triton_path_patch"] = "unknown"

    # Check for sm_120 (consumer Blackwell) issue.
    if info["cuda_available"]:
        cap = torch.cuda.get_device_capability()
        if cap[0] == 12 and cap[1] == 0:
            info["gpu"] = "Consumer Blackwell (sm_120)"
            info["issues"].append(
                "Consumer Blackwell detected — ensure Triton sm_120 patch is applied"
            )

    return info
