"""Benchmark NVFP4 / AS-FP4 / R-FP4 vs INT8 / bf16 on realistic LLM weights.

Tests on:
1. Simulated LLM weights (large matrices with outliers)
2. ForgeLM V4 1.2B model (if checkpoint available)
3. ForgeLM V7 8B model (if checkpoint available)

Measures: compression, Frobenius error, forward pass error, speed.
"""
import os, sys, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
import torch.nn.functional as F


def frob_err(ref, q):
    return (ref.float() - q.float()).norm().item() / ref.float().norm().clamp(min=1e-8).item()


def simulate_llm_weights(out_f, in_f, seed=42):
    """Simulate realistic LLM weights: small std with occasional outliers."""
    torch.manual_seed(seed)
    W = torch.randn(out_f, in_f) * 0.02  # typical LLM weight std
    # Add outliers (~1% of weights are 5x larger)
    outlier_mask = torch.rand(out_f, in_f) < 0.01
    W[outlier_mask] *= 5.0
    return W


def bench_matrix_sizes():
    """Benchmark all quant methods across matrix sizes."""
    from research.inference.quant.nvfp4_quant import (
        NVFP4Linear, _quantize_to_fp4, _dequantize_fp4,
    )
    from research.inference.quant.novel_quant import (
        ASFP4Linear, ResidualFP4Linear,
        _quantize_to_fp4_adaptive, _quantize_to_fp4_residual,
    )
    from research.quantization.inference_quant import QuantizedLinear, FastINT8Linear

    print("=" * 80)
    print("  Quantization Benchmark: NVFP4 vs AS-FP4 vs R-FP4 vs INT8 vs bf16")
    print("=" * 80)

    sizes = [
        (2048, 8192),   # typical FFN up-proj
        (4096, 4096),   # typical attention proj
        (8192, 2048),   # typical FFN down-proj
    ]

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    for out_f, in_f in sizes:
        print(f"\n  Matrix: ({out_f}, {in_f}) — {out_f*in_f*2/1024**2:.1f} MB bf16")

        W = simulate_llm_weights(out_f, in_f)
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight.data.copy_(W)

        if use_cuda:
            lin = lin.cuda().to(dtype)

        x = torch.randn(32, in_f, dtype=dtype, device=device)
        with torch.no_grad():
            y_ref = lin(x)

        # bf16 baseline
        bf16_bytes = out_f * in_f * 2

        results = {}

        # NVFP4
        nvfp4 = NVFP4Linear.from_linear(lin, block_size=32)
        if use_cuda: nvfp4 = nvfp4.cuda()
        with torch.no_grad():
            y_nvfp4 = nvfp4(x)
        nvfp4_bytes = nvfp4.weight_packed.numel() + nvfp4.weight_scales.numel()
        results["NVFP4"] = (frob_err(y_ref, y_nvfp4), bf16_bytes / nvfp4_bytes)

        # AS-FP4
        asfp4 = ASFP4Linear.from_linear(lin, block_size=32)
        if use_cuda: asfp4 = asfp4.cuda()
        with torch.no_grad():
            y_asfp4 = asfp4(x)
        asfp4_bytes = asfp4.weight_packed.numel() + asfp4.weight_scales.numel()
        results["AS-FP4"] = (frob_err(y_ref, y_asfp4), bf16_bytes / asfp4_bytes)

        # R-FP4 (5% residual)
        rfp4 = ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.05)
        if use_cuda: rfp4 = rfp4.cuda()
        with torch.no_grad():
            y_rfp4 = rfp4(x)
        rfp4_bytes = (rfp4.weight_packed.numel() + rfp4.weight_scales.numel() +
                      rfp4.residual_indices.numel() * 4 +
                      rfp4.residual_values.numel() +
                      rfp4.residual_scale.numel() * 4)
        results["R-FP4 5%"] = (frob_err(y_ref, y_rfp4), bf16_bytes / rfp4_bytes)

        # R-FP4 (10% residual)
        rfp4_10 = ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.10)
        if use_cuda: rfp4_10 = rfp4_10.cuda()
        with torch.no_grad():
            y_rfp4_10 = rfp4_10(x)
        rfp4_10_bytes = (rfp4_10.weight_packed.numel() + rfp4_10.weight_scales.numel() +
                         rfp4_10.residual_indices.numel() * 4 +
                         rfp4_10.residual_values.numel() +
                         rfp4_10.residual_scale.numel() * 4)
        results["R-FP4 10%"] = (frob_err(y_ref, y_rfp4_10), bf16_bytes / rfp4_10_bytes)

        # INT8 (weight-only)
        int8_lin = QuantizedLinear(lin, bits=8)
        if use_cuda: int8_lin = int8_lin.cuda()
        with torch.no_grad():
            y_int8 = int8_lin(x)
        int8_bytes = out_f * in_f * 1  # 1 byte per weight
        results["INT8"] = (frob_err(y_ref, y_int8), bf16_bytes / int8_bytes)

        # INT4 (weight-only, group=128)
        int4_lin = QuantizedLinear(lin, bits=4, group_size=128)
        if use_cuda: int4_lin = int4_lin.cuda()
        with torch.no_grad():
            y_int4 = int4_lin(x)
        int4_bytes = out_f * in_f * 0.5 + out_f * (in_f // 128) * 2  # 0.5 byte + scales
        results["INT4"] = (frob_err(y_ref, y_int4), bf16_bytes / int4_bytes)

        # Print results
        print(f"  {'Method':>12}  {'Frob Err':>10}  {'Compression':>12}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*12}")
        for name, (err, comp) in results.items():
            print(f"  {name:>12}  {err:>10.4f}  {comp:>11.1f}x")

    # Speed benchmark on CUDA
    if use_cuda:
        print(f"\n  Speed (CUDA, bf16, batch=32):")
        print(f"  {'Method':>12}  {'ms/call':>10}  {'vs bf16':>10}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

        out_f, in_f = 8192, 2048
        W = simulate_llm_weights(out_f, in_f)
        lin = nn.Linear(in_f, out_f, bias=False).cuda().to(dtype)
        lin.weight.data.copy_(W.to(dtype))
        x = torch.randn(32, in_f, dtype=dtype, device="cuda")

        modules = {
            "bf16": lin,
            "NVFP4": NVFP4Linear.from_linear(lin, block_size=32).cuda(),
            "AS-FP4": ASFP4Linear.from_linear(lin, block_size=32).cuda(),
            "R-FP4 5%": ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.05).cuda(),
        }

        # Warmup
        for _ in range(20):
            with torch.no_grad():
                for m in modules.values():
                    _ = m(x)
        torch.cuda.synchronize()

        bf16_ms = None
        for name, mod in modules.items():
            t0 = time.perf_counter()
            for _ in range(200):
                with torch.no_grad():
                    _ = mod(x)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 5  # ms per call
            if name == "bf16":
                bf16_ms = ms
            ratio = f"{ms/bf16_ms:.2f}x" if bf16_ms else "1.00x"
            print(f"  {name:>12}  {ms:>10.2f}  {ratio:>10}")


def bench_v7_8b_estimate():
    """Estimate VRAM savings for V7 8B with NVFP4."""
    print("\n" + "=" * 80)
    print("  V7 8B VRAM Estimate with NVFP4")
    print("=" * 80)

    # V7 8B _b variant: ~8.05B params
    n_params = 8.05e9

    bf16_gb = n_params * 2 / 1024**3
    nvfp4_gb = n_params * 0.53 / 1024**3  # 0.53 bytes/weight (FP4 + FP8 scales)
    int8_gb = n_params * 1 / 1024**3
    int4_gb = n_params * 0.56 / 1024**3  # 0.5 + scales

    print(f"  Params:       {n_params/1e9:.2f}B")
    print(f"  bf16:         {bf16_gb:.2f} GB  (doesn't fit 12GB)")
    print(f"  NVFP4:        {nvfp4_gb:.2f} GB  (fits 12GB with {12-nvfp4_gb:.1f}GB for KV+activations)")
    print(f"  INT8:         {int8_gb:.2f} GB  (fits 12GB with {12-int8_gb:.1f}GB for KV+activations)")
    print(f"  INT4:         {int4_gb:.2f} GB  (fits 12GB with {12-int4_gb:.1f}GB for KV+activations)")
    print(f"\n  NVFP4 frees ~{bf16_gb - nvfp4_gb:.1f} GB vs bf16")
    print(f"  At 32K context, KV cache needs ~2-3 GB (with rotorquant 4-bit)")
    print(f"  NVFP4 + rotorquant KV: {nvfp4_gb + 3:.1f} GB total -> fits 12GB comfortably")


def main():
    bench_matrix_sizes()
    bench_v7_8b_estimate()
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
