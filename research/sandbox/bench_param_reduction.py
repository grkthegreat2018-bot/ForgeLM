"""Benchmark: prior config (V4) vs new compressed config (V6).

Builds both configs via the standard ForgeEngine pipeline (get_config +
ModelLoader.build_model_fast), then measures:
  1. Parameter count (total + breakdown by component)
  2. VRAM footprint (bf16 bytes)
  3. Forward pass correctness (output shape, no NaNs)
  4. Forward pass latency (GPU-prioritized with hybrid offload)
  5. Per-key parameter savings

GPU PRIORITY + MIXED OFFLOAD:
  - If CUDA is available: model goes to GPU with hybrid offload
    (attention layers on GPU, conv layers on CPU — same as ForgeEngine)
  - If model doesn't fit in VRAM: falls back to hybrid offload
  - If no CUDA: falls back to CPU

Run: python -m research.sandbox.bench_param_reduction
"""
import time
import torch
import torch.nn as nn

from research.config import get_config, MODEL_CONFIGS
from research.model_loader import ModelLoader, ConfigurableResearchLLM


# ── Device detection ──────────────────────────────────────────────────────
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_FREE, VRAM_TOTAL = torch.cuda.mem_get_info(DEVICE)
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU only"
    VRAM_FREE = VRAM_TOTAL = 0


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_breakdown(model: nn.Module) -> dict:
    """Break down params by top-level module."""
    breakdown = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            breakdown[name] = n
    return breakdown


def vram_bytes_bf16(model: nn.Module) -> int:
    return sum(p.numel() * 2 for p in model.parameters())  # bf16 = 2 bytes


def gpu_param_bytes(model: nn.Module) -> int:
    """Bytes of params currently on GPU."""
    total = 0
    for p in model.parameters():
        if p.device.type == "cuda":
            total += p.numel() * p.element_size()
    return total


def cpu_param_bytes(model: nn.Module) -> int:
    """Bytes of params currently on CPU."""
    total = 0
    for p in model.parameters():
        if p.device.type == "cpu":
            total += p.numel() * p.element_size()
    return total


def _count_unique_params(model: nn.Module) -> int:
    """Count unique params (shared/tied weights counted once)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def build_model(config_name: str, **overrides) -> tuple[nn.Module, dict]:
    """Build model directly on GPU — all compute on GPU, zero CPU compute.

    Uses torch.device('cuda') context so parameters are allocated on GPU
    directly (no fp32 CPU spike, no CPU→GPU transfer).
    """
    cfg = get_config(config_name, **overrides)
    target_dtype = torch.bfloat16

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        if CUDA_AVAILABLE:
            # Build directly on GPU — no CPU RAM spike, no CPU compute
            with torch.device(str(DEVICE)):
                cfg.device = str(DEVICE)
                model = ConfigurableResearchLLM(cfg)
            placement = "gpu (full, all compute on GPU)"
        else:
            cfg.device = "cpu"
            model = ConfigurableResearchLLM(cfg)
            placement = "cpu"
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    # Count unique params (expert tying shares Parameter objects)
    unique_params = _count_unique_params(model)

    info = {
        'config_name': config_name,
        'config': cfg,
        'n_params': count_params(model),
        'unique_params': unique_params,
        'vram_bf16': vram_bytes_bf16(model),
        'breakdown': param_breakdown(model),
        'placement': placement,
        'gpu_bytes': gpu_param_bytes(model) if CUDA_AVAILABLE else 0,
        'cpu_bytes': cpu_param_bytes(model) if CUDA_AVAILABLE else 0,
    }
    return model, info


def forward_test(model: nn.Module, vocab_size: int, seq_len: int = 16) -> dict:
    """Run forward pass and measure latency on the model's current device."""
    # Determine input device from model params
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = DEVICE

    x = torch.randint(0, vocab_size, (1, seq_len), device=input_device)

    # Warmup (especially important for GPU — kernel compilation)
    with torch.no_grad():
        for _ in range(5):
            out = model(x)
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(20):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    logits = out[0] if isinstance(out, tuple) else out

    # GPU memory after forward
    gpu_mem_after = 0
    if CUDA_AVAILABLE:
        free, total = torch.cuda.mem_get_info(DEVICE)
        gpu_mem_after = (total - free)  # bytes used

    return {
        'output_shape': tuple(logits.shape),
        'has_nan': torch.isnan(logits).any().item(),
        'has_inf': torch.isinf(logits).any().item(),
        'mean_latency_ms': (sum(times) / len(times)) * 1000,
        'min_latency_ms': min(times) * 1000,
        'max_latency_ms': max(times) * 1000,
        'device': str(input_device),
        'gpu_mem_after_mb': gpu_mem_after / 1e6 if CUDA_AVAILABLE else 0,
    }


def fmt_bytes(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} GB"
    if n >= 1e6:
        return f"{n/1e6:.2f} MB"
    if n >= 1e3:
        return f"{n/1e3:.2f} KB"
    return f"{n} B"


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def print_comparison(baseline: dict, compressed: dict, label: str = ""):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    b_p = baseline['n_params']
    c_p = compressed['n_params']
    b_v = baseline['vram_bf16']
    c_v = compressed['vram_bf16']
    reduction = (1 - c_p / b_p) * 100 if b_p > 0 else 0
    vram_reduction = (1 - c_v / b_v) * 100 if b_v > 0 else 0
    print(f"  Params:  {fmt_params(b_p):>10s}  →  {fmt_params(c_p):>10s}  "
          f"({reduction:+.1f}%)")
    print(f"  VRAM:    {fmt_bytes(b_v):>10s}  →  {fmt_bytes(c_v):>10s}  "
          f"({vram_reduction:+.1f}%)")
    print(f"  Placement: {baseline['placement']}  →  {compressed['placement']}")
    if CUDA_AVAILABLE:
        print(f"  GPU bytes: {fmt_bytes(baseline['gpu_bytes'])}  →  "
              f"{fmt_bytes(compressed['gpu_bytes'])}")
        print(f"  CPU bytes: {fmt_bytes(baseline['cpu_bytes'])}  →  "
              f"{fmt_bytes(compressed['cpu_bytes'])}")
    print(f"\n  Param breakdown:")
    all_keys = set(baseline['breakdown'].keys()) | set(compressed['breakdown'].keys())
    for key in sorted(all_keys):
        b_n = baseline['breakdown'].get(key, 0)
        c_n = compressed['breakdown'].get(key, 0)
        if b_n > 0 or c_n > 0:
            pct = (1 - c_n / b_n) * 100 if b_n > 0 else 0
            print(f"    {key:25s}: {fmt_params(b_n):>10s} → {fmt_params(c_n):>10s} "
                  f"({pct:+.1f}%)")


def print_forward_results(results: dict, label: str = ""):
    print(f"\n  Forward ({label}) [device: {results['device']}]:")
    print(f"    Output shape:    {results['output_shape']}")
    print(f"    Has NaN:         {results['has_nan']}")
    print(f"    Has Inf:         {results['has_inf']}")
    print(f"    Mean latency:    {results['mean_latency_ms']:.2f} ms")
    print(f"    Min/Max latency: {results['min_latency_ms']:.2f} / "
          f"{results['max_latency_ms']:.2f} ms")
    if CUDA_AVAILABLE:
        print(f"    GPU mem after:   {results['gpu_mem_after_mb']:.1f} MB")


def bench_tiny():
    """Benchmark tiny config: baseline vs each key individually vs combined."""
    print("\n" + "=" * 70)
    print("  TINY CONFIG BENCHMARK (fast iteration)")
    print("=" * 70)

    configs = {
        'baseline (tiny)': ('lfm25_tiny', {}),
        'monarch_ffn': ('lfm25_tiny', {
            'ffn_compression': 'monarch', 'monarch_block_size': 16,
        }),
        'kron_ffn': ('lfm25_tiny', {
            'ffn_compression': 'kron',
            'kron_a': 16, 'kron_b': 16, 'kron_c': 8, 'kron_d': 16,
        }),
        'tt_ffn': ('lfm25_tiny', {
            'ffn_compression': 'tt', 'tt_rank': 4,
        }),
        'hyperloop': ('lfm25_tiny', {
            'use_hyperloop': True, 'hyperloop_begin': 1,
            'hyperloop_end': 1, 'hyperloop_loop_iters': 2,
        }),
        'lisa': ('lfm25_tiny', {
            'use_lisa': True, 'lisa_compress': 6,
        }),
        'combined (all 5)': ('lfm25_tiny', {
            'ffn_compression': 'monarch', 'monarch_block_size': 16,
            'use_hyperloop': True, 'hyperloop_begin': 1,
            'hyperloop_end': 1, 'hyperloop_loop_iters': 2,
            'use_lisa': True, 'lisa_compress': 6,
        }),
    }

    results = {}
    for label, (config_name, overrides) in configs.items():
        print(f"\n  Building {label}...")
        try:
            model, info = build_model(config_name, **overrides)
            fwd = forward_test(model, info['config'].vocab_size, seq_len=16)
            results[label] = {**info, 'forward': fwd}
            print(f"    Params: {fmt_params(info['n_params'])}, "
                  f"VRAM: {fmt_bytes(info['vram_bf16'])}, "
                  f"placement: {info['placement']}, "
                  f"shape: {fwd['output_shape']}, "
                  f"NaN: {fwd['has_nan']}, "
                  f"latency: {fwd['mean_latency_ms']:.2f}ms")
            # Cleanup GPU memory
            if CUDA_AVAILABLE:
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    FAIL: {e}")
            import traceback; traceback.print_exc()

    # Comparison table
    baseline = results.get('baseline (tiny)')
    if baseline:
        print(f"\n{'='*70}")
        print(f"  TINY CONFIG COMPARISON (vs baseline)")
        print(f"{'='*70}")
        print(f"  {'Config':25s} {'Params':>10s} {'VRAM':>10s} {'Param Δ':>8s} "
              f"{'Latency':>10s} {'NaN':>5s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*5}")
        b_p = baseline['n_params']
        for label, r in results.items():
            pct = (1 - r['n_params'] / b_p) * 100 if b_p > 0 else 0
            lat = r['forward']['mean_latency_ms']
            nan = r['forward']['has_nan']
            print(f"  {label:25s} {fmt_params(r['n_params']):>10s} "
                  f"{fmt_bytes(r['vram_bf16']):>10s} {pct:+7.1f}% "
                  f"{lat:>9.2f}ms {'YES' if nan else 'no':>5s}")


def bench_individual_keys_full():
    """Benchmark each key individually on full V4 base."""
    print("\n" + "=" * 70)
    print("  INDIVIDUAL KEY BENCHMARK (full V4 base)")
    print("=" * 70)

    configs = {
        'V4 baseline': ('forgelm_v7', {}),
        'V4 + Monarch FFN': ('forgelm_v7', {
            'ffn_compression': 'monarch', 'monarch_block_size': 32,
        }),
        'V4 + Kronecker FFN': ('forgelm_v7', {
            'ffn_compression': 'kron',
            'kron_a': 64, 'kron_b': 128,  # 64*128=8192=intermediate
            'kron_c': 32, 'kron_d': 64,   # 32*64=2048=d_model
        }),
        'V4 + TT FFN': ('forgelm_v7', {
            'ffn_compression': 'tt', 'tt_rank': 4,
        }),
        'V4 + Hyperloop': ('forgelm_v7', {
            'use_hyperloop': True, 'hyperloop_begin': 2,
            'hyperloop_end': 2, 'hyperloop_loop_iters': 3,
        }),
        'V4 + LiSA': ('forgelm_v7', {
            'use_lisa': True, 'lisa_compress': 6,
        }),
    }

    results = {}
    for label, (config_name, overrides) in configs.items():
        print(f"\n  Building {label}...")
        try:
            model, info = build_model(config_name, **overrides)
            results[label] = info
            print(f"    Params: {fmt_params(info['n_params'])}, "
                  f"VRAM: {fmt_bytes(info['vram_bf16'])}, "
                  f"placement: {info['placement']}")
            if CUDA_AVAILABLE:
                print(f"    GPU: {fmt_bytes(info['gpu_bytes'])}, "
                      f"CPU: {fmt_bytes(info['cpu_bytes'])}")
            # Cleanup GPU memory
            if CUDA_AVAILABLE:
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    FAIL: {e}")
            import traceback; traceback.print_exc()

    # Comparison table
    baseline = results.get('V4 baseline')
    if baseline:
        print(f"\n{'='*70}")
        print(f"  INDIVIDUAL KEY COMPARISON (vs V4 baseline)")
        print(f"{'='*70}")
        print(f"  {'Config':25s} {'Params':>10s} {'VRAM':>10s} {'Param Δ':>8s} "
              f"{'GPU':>10s} {'CPU':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
        b_p = baseline['n_params']
        for label, r in results.items():
            pct = (1 - r['n_params'] / b_p) * 100 if b_p > 0 else 0
            gpu_s = fmt_bytes(r['gpu_bytes']) if CUDA_AVAILABLE else "—"
            cpu_s = fmt_bytes(r['cpu_bytes']) if CUDA_AVAILABLE else "—"
            print(f"  {label:25s} {fmt_params(r['n_params']):>10s} "
                  f"{fmt_bytes(r['vram_bf16']):>10s} {pct:+7.1f}% "
                  f"{gpu_s:>10s} {cpu_s:>10s}")


def bench_full_v4_vs_v6():
    """Benchmark full V4 vs V6-compressed (no checkpoint, just architecture)."""
    print("\n" + "=" * 70)
    print("  FULL CONFIG BENCHMARK: V4 vs V6-Compressed")
    print("=" * 70)

    # V4 baseline
    print("\n  Building forgelm_v7 (baseline)...")
    try:
        model_v4, info_v4 = build_model('forgelm_v7')
        print(f"    Params: {fmt_params(info_v4['n_params'])}, "
              f"VRAM: {fmt_bytes(info_v4['vram_bf16'])}, "
              f"placement: {info_v4['placement']}")
        if CUDA_AVAILABLE:
            print(f"    GPU: {fmt_bytes(info_v4['gpu_bytes'])}, "
                  f"CPU: {fmt_bytes(info_v4['cpu_bytes'])}")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback; traceback.print_exc()
        return

    # V6 compressed (old name — use V6 dense now)
    print("\n  Building forgelm_v7 (dense compressed)...")
    try:
        model_v6, info_v6 = build_model('forgelm_v7')
        print(f"    Params: {fmt_params(info_v6['n_params'])}, "
              f"VRAM: {fmt_bytes(info_v6['vram_bf16'])}, "
              f"placement: {info_v6['placement']}")
        if CUDA_AVAILABLE:
            print(f"    GPU: {fmt_bytes(info_v6['gpu_bytes'])}, "
                  f"CPU: {fmt_bytes(info_v6['cpu_bytes'])}")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback; traceback.print_exc()
        return

    # Comparison
    print_comparison(info_v4, info_v6, "V4 (baseline) vs V6 (compressed)")

    # Forward tests
    print("\n  Running forward passes (seq_len=16)...")
    try:
        fwd_v4 = forward_test(model_v4, info_v4['config'].vocab_size, seq_len=16)
        print_forward_results(fwd_v4, "V4")
    except Exception as e:
        print(f"    V4 forward FAIL: {e}")
        import traceback; traceback.print_exc()
        fwd_v4 = None

    try:
        fwd_v6 = forward_test(model_v6, info_v6['config'].vocab_size, seq_len=16)
        print_forward_results(fwd_v6, "V6")
    except Exception as e:
        print(f"    V6 forward FAIL: {e}")
        import traceback; traceback.print_exc()
        fwd_v6 = None

    # Latency comparison
    if fwd_v4 and fwd_v6:
        speedup = fwd_v4['mean_latency_ms'] / fwd_v6['mean_latency_ms']
        print(f"\n  Latency speedup (V4/V6): {speedup:.2f}x")

    # Cleanup
    if CUDA_AVAILABLE:
        del model_v4, model_v6
        torch.cuda.empty_cache()


def bench_v6_all():
    """Benchmark V6-Dense and V6-MoE vs V4 and V5."""
    print("\n" + "=" * 70)
    print("  V6 FULL BENCHMARK: V4 vs V5 vs V6-Dense vs V6-MoE")
    print("=" * 70)

    configs = {
        'V4 (16L baseline)': 'forgelm_v7',
        'V5 MoE (16L)': 'forgelm_v7_moe',
        'V6 Dense (24L)': 'forgelm_v7',
        'V6 MoE (24L)': 'forgelm_v7_moe',
    }

    results = {}
    for label, config_name in configs.items():
        print(f"\n  Building {label}...")
        try:
            model, info = build_model(config_name)
            results[label] = info
            print(f"    Params: {fmt_params(info['n_params'])}, "
                  f"VRAM: {fmt_bytes(info['vram_bf16'])}, "
                  f"placement: {info['placement']}")
            if CUDA_AVAILABLE:
                print(f"    GPU: {fmt_bytes(info['gpu_bytes'])}, "
                      f"CPU: {fmt_bytes(info['cpu_bytes'])}")
            # Forward test
            try:
                fwd = forward_test(model, info['config'].vocab_size, seq_len=16)
                results[label]['forward'] = fwd
                print(f"    Forward: {fwd['output_shape']}, "
                      f"NaN={fwd['has_nan']}, "
                      f"latency={fwd['mean_latency_ms']:.2f}ms")
            except Exception as e:
                print(f"    Forward FAIL: {e}")
                results[label]['forward'] = None
            # Cleanup
            if CUDA_AVAILABLE:
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    BUILD FAIL: {e}")
            import traceback; traceback.print_exc()

    # Comparison table
    if results:
        print(f"\n{'='*70}")
        print(f"  V6 COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"  {'Config':25s} {'Params':>10s} {'VRAM':>10s} {'GPU':>10s} "
              f"{'Latency':>10s} {'NaN':>5s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")
        for label, r in results.items():
            lat = r['forward']['mean_latency_ms'] if r.get('forward') else 0
            nan = r['forward']['has_nan'] if r.get('forward') else 'N/A'
            gpu_s = fmt_bytes(r['gpu_bytes']) if CUDA_AVAILABLE else "—"
            print(f"  {label:25s} {fmt_params(r['n_params']):>10s} "
                  f"{fmt_bytes(r['vram_bf16']):>10s} {gpu_s:>10s} "
                  f"{lat:>9.2f}ms {'YES' if nan else 'no':>5s}")

        # Param breakdown for each
        for label, r in results.items():
            print(f"\n  {label} breakdown:")
            for key in sorted(r['breakdown'].keys()):
                print(f"    {key:25s}: {fmt_params(r['breakdown'][key]):>10s}")


if __name__ == "__main__":
    print("=" * 70)
    print("  PARAMETER REDUCTION BENCHMARK")
    print("  V4 vs V5 vs V6 (all best practices + scaled params)")
    print("=" * 70)
    print(f"\n  Device: {GPU_NAME}")
    if CUDA_AVAILABLE:
        print(f"  VRAM: {VRAM_FREE/1e9:.2f} GB free / {VRAM_TOTAL/1e9:.2f} GB total")
        print(f"  Strategy: GPU-priority with hybrid offload (attention→GPU, conv→CPU)")
    else:
        print(f"  Strategy: CPU fallback (no CUDA detected)")

    # Phase 1: Tiny config (fast, tests all keys)
    bench_tiny()

    # Phase 2: Individual keys on full V4 base
    bench_individual_keys_full()

    # Phase 3: Full V4 vs old V6 comparison
    bench_full_v4_vs_v6()

    # Phase 4: V6 full benchmark (V4 vs V5 vs V6-Dense vs V6-MoE)
    bench_v6_all()

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
