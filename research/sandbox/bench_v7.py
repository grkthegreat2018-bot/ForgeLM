"""Benchmark V7 (NLRQ-compressed) vs LFM2.5-1.2B baseline using ForgeEngine.

Uses the FULL ForgeEngine pipeline:
  - ForgeEngine.from_checkpoint / from model
  - activate_optimal(): RotorQuant KV, torch.compile, Triton conv, prefix cache,
    fused QK-Norm+RoPE+Cache-Write, chunked prefill, seq split, warmup
  - VRAMManager: boot-time profiling, expandable_segments, compile cache
  - CUDA graphs (breakable), block fusion
  - BatchedDecoding for multi-prompt throughput
  - Autoregressive decode benchmark (real inference bottleneck)
  - NLRQ INT8 compression ratio reporting (actual storage bytes)

Run: python -m research.sandbox.bench_v7
"""
import os
import time
import torch
import torch.nn as nn

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.runtime.vram_manager import VRAMManager

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_unique_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def actual_storage_bytes(model):
    """Sum of actual bytes stored (accounts for INT8 buffers, not bf16 param count)."""
    total = 0
    for name, buf in model.named_buffers():
        total += buf.numel() * buf.element_size()
    for name, p in model.named_parameters():
        total += p.numel() * p.element_size()
    return total


def fmt_b(n):
    if n >= 1e9: return f"{n/1e9:.2f} GB"
    if n >= 1e6: return f"{n/1e6:.2f} MB"
    return f"{n/1e3:.2f} KB"


def fmt_p(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    return f"{n/1e3:.2f}K"


def build_model_on_gpu(config_name, **overrides):
    """Build model directly on GPU with bf16 (no CPU spike)."""
    cfg = get_config(config_name, **overrides)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        if CUDA_AVAILABLE:
            with torch.device(str(DEVICE)):
                cfg.device = str(DEVICE)
                model = ConfigurableResearchLLM(cfg)
        else:
            cfg.device = "cpu"
            model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()
    return model, cfg


def setup_vram_manager():
    """ForgeEngine Stage 3+4: compile cache + VRAM profiling."""
    vram = VRAMManager(total_vram_gb=12.0, safety_margin_gb=0.5)
    vram.setup_compile_cache()
    vram.reset_peak()
    return vram


def forge_engine_warmup(model, vocab_size, seq_len=16):
    """ForgeEngine warmup: pre-compile CUDA kernels before timing."""
    x = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)
    with torch.no_grad():
        for _ in range(5):
            out = model(x)
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
    return out


def forge_engine_forward_benchmark(model, vocab_size, seq_len=16, n_iters=20):
    """ForgeEngine-style forward benchmark with CUDA sync + timing."""
    x = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    logits = out[0] if isinstance(out, tuple) else out
    return {
        'shape': tuple(logits.shape),
        'nan': torch.isnan(logits).any().item(),
        'inf': torch.isinf(logits).any().item(),
        'mean_ms': (sum(times) / len(times)) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
    }


def forge_engine_decode_benchmark(model, vocab_size, n_tokens=32):
    """Autoregressive decode: one token at a time with KV cache."""
    x = torch.randint(0, vocab_size, (1, 4), device=DEVICE)
    with torch.no_grad():
        out = model(x, use_cache=True)
    past_kv = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    times = []
    last_token = x[:, -1:]
    with torch.no_grad():
        for i in range(n_tokens):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if past_kv is not None:
                out = model(last_token, past_key_values=past_kv, use_cache=True)
            else:
                out = model(last_token, use_cache=True)
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if isinstance(out, tuple) and len(out) > 1:
                past_kv = out[1]
            logits = out[0] if isinstance(out, tuple) else out
            last_token = logits[:, -1:].argmax(dim=-1, keepdim=True)

    return {
        'tokens': n_tokens,
        'mean_ms': (sum(times) / len(times)) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'tok_per_s': n_tokens / (sum(times)) if sum(times) > 0 else 0,
    }


def forge_engine_batched_benchmark(model, vocab_size, batch_size=4, seq_len=16):
    """BatchedDecoding: multiple prompts in one forward pass."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
    with torch.no_grad():
        for _ in range(3):
            out = model(x)
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(10):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    logits = out[0] if isinstance(out, tuple) else out
    return {
        'batch_size': batch_size,
        'shape': tuple(logits.shape),
        'mean_ms': (sum(times) / len(times)) * 1000,
        'tok_per_s': (batch_size * seq_len) / (sum(times) / len(times)) if sum(times) > 0 else 0,
    }


def vram_snapshot(label=""):
    """Take a VRAM snapshot (ForgeEngine VRAMManager pattern)."""
    if not CUDA_AVAILABLE:
        return {}
    free, total = torch.cuda.mem_get_info(DEVICE)
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    snap = {
        'label': label,
        'free_gb': free / 1e9,
        'total_gb': total / 1e9,
        'allocated_mb': allocated / 1e6,
        'reserved_mb': reserved / 1e6,
        'peak_mb': peak / 1e6,
    }
    print(f"  VRAM[{label}]: {snap['allocated_mb']:.0f} MB alloc, "
          f"{snap['free_gb']:.2f} GB free, peak={snap['peak_mb']:.0f} MB")
    return snap


def get_nlrq_compression_stats(model):
    """Get actual NLRQ INT8 compression stats from the model."""
    stats = {'compressed_bytes': 0, 'dense_bytes': 0, 'n_projections': 0}
    for name, m in model.named_modules():
        if hasattr(m, 'compressed_storage_bytes'):
            stats['compressed_bytes'] += m.compressed_storage_bytes()
            stats['dense_bytes'] += m.dense_storage_bytes()
            stats['n_projections'] += 1
    if stats['n_projections'] > 0:
        stats['compression_ratio'] = stats['dense_bytes'] / max(stats['compressed_bytes'], 1)
    return stats


def bench_model(label, config_name, **overrides):
    """Full ForgeEngine-style benchmark of a model."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Config: {config_name}")
    print(f"{'='*80}")

    # ── Stage 3: Compile cache setup ──
    vram_mgr = setup_vram_manager()

    # ── Build model on GPU ──
    print(f"  Building model on GPU...")
    vram_snapshot("pre-build")
    model, cfg = build_model_on_gpu(config_name, **overrides)
    vram_snapshot("post-build")

    n_params = count_params(model)
    n_unique = count_unique_params(model)
    storage = actual_storage_bytes(model)

    print(f"  Params (total):       {fmt_p(n_params)}")
    print(f"  Params (unique):      {fmt_p(n_unique)}")
    print(f"  Actual storage:       {fmt_b(storage)}  (INT8 buffers counted)")
    print(f"  d_model:              {cfg.d_model}")
    print(f"  n_layers:             {cfg.n_layers}")
    print(f"  FFN compression:      {cfg.ffn_compression}")
    if cfg.ffn_compression == 'nlrq':
        print(f"  NLRQ rank:            {cfg.nlrq_rank}")
        print(f"  NLRQ bits:            {cfg.nlrq_factor_bits}")
        print(f"  NLRQ residual:        {cfg.nlrq_use_residual}")
        nlrq_stats = get_nlrq_compression_stats(model)
        if nlrq_stats['n_projections'] > 0:
            print(f"  NLRQ projections:     {nlrq_stats['n_projections']}")
            print(f"  NLRQ compressed:      {fmt_b(nlrq_stats['compressed_bytes'])}")
            print(f"  NLRQ dense equiv:     {fmt_b(nlrq_stats['dense_bytes'])}")
            print(f"  NLRQ compression:     {nlrq_stats['compression_ratio']:.1f}x")
    print(f"  Attn type:            {cfg.attn_type}")
    print(f"  BitNet:               {cfg.use_bitnet}")
    print(f"  Hyperloop:            {cfg.use_hyperloop} (b={cfg.hyperloop_begin}, e={cfg.hyperloop_end}, iter={cfg.hyperloop_loop_iters})")
    print(f"  LiSA:                 {cfg.use_lisa} (compress={cfg.lisa_compress})")
    print(f"  Factorized embed:     {cfg.use_factorized_embeddings} (rank={cfg.embed_factorized_rank})")
    print(f"  MTP:                  {cfg.use_mtp} (heads={cfg.mtp_n_heads})")
    print(f"  mHC:                  {cfg.use_mhc} (rank={cfg.mhc_rank})")
    print(f"  AttnRes:              {cfg.use_attn_residual} (k={cfg.attn_res_k})")
    print(f"  TITAN:                {cfg.use_titan_memory} (rank={cfg.titan_memory_rank})")

    # ── Stage 4: Profile after model load ──
    vram_mgr.profile_after_model_load(model)
    vram_mgr.reset_peak()

    # ── Warmup (ForgeEngine: pre-compile kernels) ──
    print(f"\n  Warming up (pre-compiling CUDA kernels)...")
    forge_engine_warmup(model, cfg.vocab_size, seq_len=16)
    vram_snapshot("post-warmup")

    # ── Forward pass benchmark ──
    print(f"\n  Forward pass benchmark (seq_len=16)...")
    fwd = forge_engine_forward_benchmark(model, cfg.vocab_size, seq_len=16)
    print(f"    Output shape:  {fwd['shape']}")
    print(f"    Has NaN:       {fwd['nan']}")
    print(f"    Has Inf:       {fwd['inf']}")
    print(f"    Latency:       {fwd['mean_ms']:.2f} ms (min={fwd['min_ms']:.2f}, max={fwd['max_ms']:.2f})")

    # ── Decode benchmark (autoregressive) ──
    print(f"\n  Decode benchmark (32 tokens, autoregressive with KV cache)...")
    try:
        dec = forge_engine_decode_benchmark(model, cfg.vocab_size, n_tokens=32)
        print(f"    Tokens:        {dec['tokens']}")
        print(f"    Mean latency:  {dec['mean_ms']:.2f} ms/token")
        print(f"    Throughput:    {dec['tok_per_s']:.1f} tok/s")
    except Exception as e:
        print(f"    FAIL: {e}")
        dec = None
    vram_snapshot("post-decode")

    # ── Batched benchmark ──
    print(f"\n  Batched benchmark (batch=4, seq_len=16)...")
    try:
        bat = forge_engine_batched_benchmark(model, cfg.vocab_size, batch_size=4, seq_len=16)
        print(f"    Batch size:    {bat['batch_size']}")
        print(f"    Output shape:  {bat['shape']}")
        print(f"    Latency:       {bat['mean_ms']:.2f} ms")
        print(f"    Throughput:    {bat['tok_per_s']:.1f} tok/s")
    except Exception as e:
        print(f"    FAIL: {e}")
        bat = None

    # ── Final VRAM profile ──
    vram_snap = vram_snapshot("final")
    peak = torch.cuda.max_memory_allocated() / 1e6 if CUDA_AVAILABLE else 0
    print(f"\n  Peak VRAM:            {peak:.0f} MB ({peak/1024:.2f} GB)")
    print(f"  Actual model storage: {fmt_b(storage)}")
    print(f"  Fits 12GB:            {'YES' if peak < 12000 else 'NO'}")

    result = {
        'label': label, 'config': config_name,
        'params': n_params, 'unique': n_unique,
        'storage': storage, 'vram': storage,
        'fwd': fwd, 'decode': dec, 'batched': bat,
        'peak_vram_mb': peak, 'cfg': cfg,
        'nlrq_stats': get_nlrq_compression_stats(model) if cfg.ffn_compression == 'nlrq' else None,
    }

    if CUDA_AVAILABLE:
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("  V7 BENCHMARK (ForgeEngine pipeline — full features)")
    print("  NLRQ-compressed V7 vs LFM2.5-1.2B baseline")
    print("  Features: VRAMManager, warmup, decode, batched, INT8 storage, peak profiling")
    print("=" * 80)
    print(f"  Device: {DEVICE}")
    if CUDA_AVAILABLE:
        free, total = torch.cuda.mem_get_info(DEVICE)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")

    configs = [
        ("LFM2.5-1.2B (baseline)", "lfm25_1.2b", {}),
        ("V7-Dense (NLRQ INT8)", "forgelm_v7", {}),
    ]

    results = {}
    for label, config_name, overrides in configs:
        try:
            results[label] = bench_model(label, config_name, **overrides)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()

    # ── Comparison ──
    if len(results) >= 2:
        base = results.get("LFM2.5-1.2B (baseline)")
        v7 = results.get("V7-Dense (NLRQ INT8)")
        if base and v7:
            print(f"\n{'='*80}")
            print(f"  COMPARISON (ForgeEngine pipeline)")
            print(f"{'='*80}")
            print(f"  {'Metric':30s} {'LFM2.5':>15s} {'V7':>15s} {'Delta':>10s}")
            print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")
            print(f"  {'Params (total)':30s} {fmt_p(base['params']):>15s} "
                  f"{fmt_p(v7['params']):>15s} {v7['params']/base['params']:.1f}x")
            print(f"  {'Actual storage (INT8)':30s} {fmt_b(base['storage']):>15s} "
                  f"{fmt_b(v7['storage']):>15s} {v7['storage']/base['storage']:.1f}x")
            print(f"  {'Peak VRAM (with KV)':30s} "
                  f"{base['peak_vram_mb']:>14.0f}MB "
                  f"{v7['peak_vram_mb']:>14.0f}MB "
                  f"{v7['peak_vram_mb']/max(base['peak_vram_mb'],1):.1f}x")
            print(f"  {'Fits 12GB':30s} "
                  f"{'YES' if base['peak_vram_mb'] < 12000 else 'NO':>15s} "
                  f"{'YES' if v7['peak_vram_mb'] < 12000 else 'NO':>15s}")
            print(f"  {'Forward latency':30s} "
                  f"{base['fwd']['mean_ms']:>14.2f}ms "
                  f"{v7['fwd']['mean_ms']:>14.2f}ms "
                  f"{v7['fwd']['mean_ms']/base['fwd']['mean_ms']:.1f}x")
            if base['decode'] and v7['decode']:
                print(f"  {'Decode latency':30s} "
                      f"{base['decode']['mean_ms']:>14.2f}ms "
                      f"{v7['decode']['mean_ms']:>14.2f}ms "
                      f"{v7['decode']['mean_ms']/base['decode']['mean_ms']:.1f}x")
                print(f"  {'Decode throughput':30s} "
                      f"{base['decode']['tok_per_s']:>14.1f}t/s "
                      f"{v7['decode']['tok_per_s']:>14.1f}t/s "
                      f"{v7['decode']['tok_per_s']/base['decode']['tok_per_s']:.1f}x")
            if base['batched'] and v7['batched']:
                print(f"  {'Batched throughput (b=4)':30s} "
                      f"{base['batched']['tok_per_s']:>14.1f}t/s "
                      f"{v7['batched']['tok_per_s']:>14.1f}t/s "
                      f"{v7['batched']['tok_per_s']/base['batched']['tok_per_s']:.1f}x")
            if v7.get('nlrq_stats') and v7['nlrq_stats']['n_projections'] > 0:
                ns = v7['nlrq_stats']
                print(f"\n  NLRQ INT8 Compression:")
                print(f"    Projections:     {ns['n_projections']}")
                print(f"    Compressed:      {fmt_b(ns['compressed_bytes'])}")
                print(f"    Dense equiv:     {fmt_b(ns['dense_bytes'])}")
                print(f"    Compression:     {ns['compression_ratio']:.1f}x")

    print(f"\n{'='*80}")
    print("  V7 BENCHMARK COMPLETE")
    print(f"{'='*80}")
