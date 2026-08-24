"""Benchmark FreeToken (CPUAdamW) vs BAdam on the actual V7 8B model.

Tests:
1. BAdam baseline (current production path) — 1 layer at a time, fits VRAM
2. CPUAdamW + FreeToken with bf16 optimizer states — all layers, CPU offload
3. CPUAdamW + FreeToken + FP4 checkpointing — larger batch possible

Measures: tok/s, VRAM peak, CPU RAM usage, step time breakdown.
"""
import os, sys, time, argparse
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def log(msg):
    print(f"  {msg}", flush=True)


def vram_snapshot(label=""):
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    free, total = torch.cuda.mem_get_info()
    log(f"VRAM[{label}]: {alloc:.2f} GB alloc, {peak:.2f} GB peak, "
        f"{free/1e9:.2f} GB free / {total/1e9:.2f} GB total")
    return {"alloc_gb": alloc, "peak_gb": peak, "free_gb": free/1e9}


def build_8b_model(config_name, device, dtype, use_checkpointing=True,
                   factor_training=False):
    from research.config import get_config
    from research.model_loader import ConfigurableResearchLLM
    from research.sandbox.train_8b_all import (
        materialize_meta_, reset_nlrq_layers_, enable_factor_training_all_,
        freeze_dead_params_, enable_checkpointing_, compute_loss,
    )
    
    cfg = get_config(config_name)
    cfg.device = "meta"
    cfg.dtype = "bfloat16"
    cfg.use_gradient_checkpointing = use_checkpointing
    cfg.selective_gradient_checkpointing = "all"
    
    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg)
    materialize_meta_(model, device, dtype)
    cfg.device = str(device)
    
    reset_nlrq_layers_(model)
    if factor_training:
        enable_factor_training_all_(model)
    
    # Freeze MTP + dead params
    mtp = getattr(model, "mtp_module", None)
    if mtp is not None:
        for p in mtp.parameters():
            p.requires_grad_(False)
            p._forge_frozen = True
    
    use_flce = True  # Fused Linear-CE, always beneficial for 8B
    n_dead = freeze_dead_params_(model, device, use_flce)
    log(f"Froze {n_dead} dead param tensors")
    
    if use_checkpointing:
        enable_checkpointing_(model)
    
    # Critical: empty cache after build to reclaim transient build memory
    # The materialize_meta_ + freeze_dead_params_ probe transiently
    # materializes all grads, peaking at ~18GB. Empty cache to reclaim.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    return model, cfg, use_flce


def make_synthetic_data(seq_len, n_seqs, vocab_size, device):
    """Generate synthetic training data."""
    data = torch.randint(0, vocab_size, (n_seqs, seq_len), dtype=torch.long)
    return data


def bench_badam(model, cfg, device, seq_len, n_steps, use_flce, batch_size=1):
    """Benchmark BAdam (current production path)."""
    from research.sandbox.train_8b_all import compute_loss
    from research.training.optim.badam import configure_badam
    from research.keys.compression.nlrq_ffn_key import NLRQLinear
    
    factor_on = any(m.factor_training_enabled() for m in model.modules()
                    if isinstance(m, NLRQLinear))
    
    optimizer = configure_badam(
        model, lr=3e-4, weight_decay=0.01,
        blocks_per_layer=1, switch_every=10,
        switch_mode="descending",
        bf16_large_states=(factor_on and True),
    )
    
    vram_snapshot("BAdam init")
    data = make_synthetic_data(seq_len, n_steps * batch_size + 10, cfg.vocab_size, device)
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        idx = (step - 1) * batch_size
        input_ids = data[idx:idx + batch_size].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = compute_loss(model, input_ids, use_flce) / batch_size
        loss.backward()
        
        for group in optimizer.param_groups:
            group["lr"] = 3e-4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        del loss, input_ids
    
    vram_snapshot("BAdam post-run")
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    # Warmup-adjusted throughput
    warmup = min(3, len(times) // 3)
    steady_times = times[warmup:]
    avg_step = sum(steady_times) / len(steady_times)
    tps = (batch_size * seq_len) / avg_step
    
    return {
        "mode": "BAdam",
        "tok_s": tps,
        "ms_per_step": avg_step * 1000,
        "peak_vram_gb": peak,
        "batch_size": batch_size,
    }


def bench_cpuadamw_freetoken(model, cfg, device, seq_len, n_steps, use_flce,
                              batch_size=1, bf16_states=True, fp4_checkpoint=False):
    """Benchmark CPUAdamW + FreeToken."""
    from research.sandbox.train_8b_all import compute_loss
    from research.training.optim.hybrid_offload import CPUAdamW
    
    # Apply FP4 checkpointing if requested
    if fp4_checkpoint:
        from research.training.fp4_checkpoint import enable_fp4_checkpointing
        enable_fp4_checkpointing(model)
        log("FP4 gradient checkpointing enabled")
    
    # Create CPUAdamW with FreeToken enhancements
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                     "lr": 3e-4, "weight_decay": 0.01}]
    
    optimizer = CPUAdamW(
        param_groups, lr=3e-4, weight_decay=0.01,
        grad_offload=True,
        double_buffer=True,
        bandwidth_adaptive=True,
        bf16_state=bf16_states,  # Use bf16 m,v to fit 32GB RAM
    )
    
    vram_snapshot("CPUAdamW init")
    data = make_synthetic_data(seq_len, n_steps * batch_size + 10, cfg.vocab_size, device)
    
    model.train()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    for step in range(1, n_steps + 1):
        # Wait for previous CPU step (FreeToken overlap)
        if hasattr(optimizer, 'wait') and optimizer.overlap:
            optimizer.wait()
        
        t0 = time.perf_counter()
        idx = (step - 1) * batch_size
        input_ids = data[idx:idx + batch_size].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = compute_loss(model, input_ids, use_flce) / batch_size
        loss.backward()
        
        # Record bandwidth sample for FreeToken predictor
        if hasattr(optimizer, 'record_bandwidth_sample'):
            vram_gb = torch.cuda.memory_allocated() / 1e9
            optimizer.record_bandwidth_sample(vram_gb=vram_gb)
        
        for group in optimizer.param_groups:
            group["lr"] = 3e-4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        del loss, input_ids
    
    # Final wait for any pending CPU step
    if hasattr(optimizer, 'wait') and optimizer.overlap:
        optimizer.wait()
    
    vram_snapshot("CPUAdamW post-run")
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    warmup = min(3, len(times) // 3)
    steady_times = times[warmup:]
    avg_step = sum(steady_times) / len(steady_times)
    tps = (batch_size * seq_len) / avg_step
    
    return {
        "mode": f"CPUAdamW+FreeToken{'+FP4' if fp4_checkpoint else ''}{' bf16' if bf16_states else ''}",
        "tok_s": tps,
        "ms_per_step": avg_step * 1000,
        "peak_vram_gb": peak,
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FreeToken on V7 8B")
    parser.add_argument("--config", default="forgelm_v7_8b_b")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("all", "badam", "cpuadamw", "fp4"),
                        default="all")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for 8B benchmark")
        sys.exit(1)
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    print("=" * 70)
    print(f"  V7 8B FreeToken Benchmark")
    print(f"  Config: {args.config}, seq_len={args.seq_len}, "
          f"batch={args.batch_size}, steps={args.steps}")
    print("=" * 70)
    
    results = []
    
    if args.mode in ("all", "badam"):
        print("\n--- BAdam Baseline ---")
        try:
            model, cfg, use_flce = build_8b_model(args.config, device, dtype,
                                                  factor_training=True)
            r = bench_badam(model, cfg, device, args.seq_len, args.steps,
                           use_flce, args.batch_size)
            results.append(r)
            print(f"  Result: {r['tok_s']:.0f} tok/s, {r['ms_per_step']:.0f} ms/step, "
                  f"{r['peak_vram_gb']:.2f} GB peak VRAM")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  BAdam FAILED: {e}")
            import traceback; traceback.print_exc()
    
    if args.mode in ("all", "cpuadamw"):
        print("\n--- CPUAdamW + FreeToken (bf16 states, no factor training) ---")
        try:
            model, cfg, use_flce = build_8b_model(args.config, device, dtype,
                                                  factor_training=False)
            r = bench_cpuadamw_freetoken(model, cfg, device, args.seq_len, args.steps,
                                         use_flce, args.batch_size, bf16_states=True)
            results.append(r)
            print(f"  Result: {r['tok_s']:.0f} tok/s, {r['ms_per_step']:.0f} ms/step, "
                  f"{r['peak_vram_gb']:.2f} GB peak VRAM")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  CPUAdamW FAILED: {e}")
            import traceback; traceback.print_exc()
    
    if args.mode in ("all", "fp4"):
        print("\n--- CPUAdamW + FreeToken + FP4 Checkpointing ---")
        try:
            model, cfg, use_flce = build_8b_model(args.config, device, dtype,
                                                  factor_training=False)
            r = bench_cpuadamw_freetoken(model, cfg, device, args.seq_len, args.steps,
                                         use_flce, args.batch_size, bf16_states=True,
                                         fp4_checkpoint=True)
            results.append(r)
            print(f"  Result: {r['tok_s']:.0f} tok/s, {r['ms_per_step']:.0f} ms/step, "
                  f"{r['peak_vram_gb']:.2f} GB peak VRAM")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FP4 Checkpoint FAILED: {e}")
            import traceback; traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Mode':<35} {'tok/s':>8} {'ms/step':>10} {'VRAM GB':>10} {'Speedup':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    baseline_tps = results[0]['tok_s'] if results else 1
    for r in results:
        speedup = r['tok_s'] / baseline_tps
        print(f"  {r['mode']:<35} {r['tok_s']:>8.0f} {r['ms_per_step']:>10.0f} "
              f"{r['peak_vram_gb']:>10.2f} {speedup:>7.2f}x")
    
    # ETA estimate
    if results:
        best = max(results, key=lambda r: r['tok_s'])
        total_tokens = 516.5e6  # v7 + fineweb_edu
        eta_1epoch = total_tokens / best['tok_s'] / 3600
        eta_2epoch = eta_1epoch * 2
        print(f"\n  Best: {best['mode']} at {best['tok_s']:.0f} tok/s")
        print(f"  ETA 1 epoch (516M tokens): {eta_1epoch:.1f}h ({eta_1epoch/24:.1f} days)")
        print(f"  ETA 2 epochs (1.03B tokens): {eta_2epoch:.1f}h ({eta_2epoch/24:.1f} days)")


if __name__ == "__main__":
    main()
