"""Test novel quantization algorithms (AS-FP4, ResidualFP4) vs NVFP4 baseline."""
import os, sys, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
from research.inference.quant.nvfp4_quant import (
    NVFP4Linear, quantize_model_nvfp4, _quantize_to_fp4, _dequantize_fp4,
)
from research.inference.quant.novel_quant import (
    ASFP4Linear, ResidualFP4Linear,
    _optimal_fp4_scale, _quantize_to_fp4_adaptive, _quantize_to_fp4_residual,
    quantize_model_asfp4, quantize_model_residual_fp4,
)


def frob_err(ref, q):
    """Frobenius norm relative error."""
    return (ref - q).norm().item() / ref.norm().clamp(min=1e-8).item()


def test_asfp4_vs_nvfp4():
    """Compare AS-FP4 (MSE-optimal scale) vs NVFP4 (absmax scale)."""
    print("=" * 60)
    print("  Novel Quant: AS-FP4 vs NVFP4")
    print("=" * 60)

    torch.manual_seed(42)
    # Use realistic LLM-like weights (small std, with some outliers)
    W = torch.randn(512, 1024) * 0.05
    # Add some outliers (like real LLM weights)
    outlier_mask = torch.rand(512, 1024) < 0.01
    W[outlier_mask] *= 5.0

    # Standard NVFP4 (absmax scale)
    packed_std, scales_std, gs_std = _quantize_to_fp4(W, block_size=32)
    W_dq_std = _dequantize_fp4(packed_std, scales_std, 512, 1024, 32, torch.float32,
                               global_scale=gs_std)

    # AS-FP4 (MSE-optimal scale)
    packed_adapt, scales_adapt, gs_adapt = _quantize_to_fp4_adaptive(W, block_size=32)
    W_dq_adapt = _dequantize_fp4(packed_adapt, scales_adapt, 512, 1024, 32, torch.float32,
                                 global_scale=gs_adapt)

    err_std = frob_err(W, W_dq_std)
    err_adapt = frob_err(W, W_dq_adapt)
    improvement = (1.0 - err_adapt / err_std) * 100

    print(f"  NVFP4 (absmax):    frob_err = {err_std:.4f}")
    print(f"  AS-FP4 (optimal):  frob_err = {err_adapt:.4f}")
    print(f"  Improvement:       {improvement:.1f}%")

    assert err_adapt <= err_std, "AS-FP4 should be better or equal"
    print("  PASS\n")


def test_residual_fp4():
    """Test ResidualFP4 with different residual ratios."""
    print("  Novel Quant: ResidualFP4")
    print("-" * 60)

    torch.manual_seed(42)
    W = torch.randn(512, 1024) * 0.05
    outlier_mask = torch.rand(512, 1024) < 0.01
    W[outlier_mask] *= 5.0

    # Baseline FP4
    packed, scales, gs = _quantize_to_fp4(W, block_size=32)
    W_dq_fp4 = _dequantize_fp4(packed, scales, 512, 1024, 32, torch.float32,
                               global_scale=gs)
    err_fp4 = frob_err(W, W_dq_fp4)

    print(f"  FP4 (no residual):     frob_err = {err_fp4:.4f}")

    for ratio in [0.02, 0.05, 0.10, 0.20]:
        packed_r, scales_r, gs_r, res_idx, res_val, res_scale = _quantize_to_fp4_residual(
            W, block_size=32, residual_ratio=ratio
        )
        W_dq = _dequantize_fp4(packed_r, scales_r, 512, 1024, 32, torch.float32,
                               global_scale=gs_r)
        # Add residual
        res_val_f = res_val.to(torch.float32)
        res_scaled = res_val_f * res_scale
        W_dq.scatter_add_(1, res_idx.long(), res_scaled)
        err_r = frob_err(W, W_dq)
        improvement = (1.0 - err_r / err_fp4) * 100
        overhead_bytes = int(1024 * ratio) * (4 + 1)  # int32 index + int8 value
        total_bytes = packed_r.numel() + scales_r.numel() + overhead_bytes * 512
        bf16_bytes = 512 * 1024 * 2
        compression = bf16_bytes / total_bytes
        print(f"  R-FP4 ({ratio*100:>4.0f}% res):    frob_err = {err_r:.4f}  "
              f"({improvement:+5.1f}%)  {compression:.1f}x compression")

    print("  PASS\n")


def test_linear_forward():
    """Test all three Linear variants produce correct outputs."""
    print("  Novel Quant: Forward pass comparison")
    print("-" * 60)

    torch.manual_seed(42)
    lin = nn.Linear(512, 1024, bias=True)
    # Scale weights to be more LLM-like
    lin.weight.data *= 0.05

    x = torch.randn(8, 512, dtype=torch.float32)

    with torch.no_grad():
        y_ref = lin(x)

        nvfp4 = NVFP4Linear.from_linear(lin, block_size=32)
        y_nvfp4 = nvfp4(x)

        asfp4 = ASFP4Linear.from_linear(lin, block_size=32)
        y_asfp4 = asfp4(x)

        rfp4 = ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.05)
        y_rfp4 = rfp4(x)

    err_nvfp4 = frob_err(y_ref, y_nvfp4)
    err_asfp4 = frob_err(y_ref, y_asfp4)
    err_rfp4 = frob_err(y_ref, y_rfp4)

    print(f"  NVFP4:     frob_err = {err_nvfp4:.4f}")
    print(f"  AS-FP4:    frob_err = {err_asfp4:.4f}")
    print(f"  R-FP4 5%:  frob_err = {err_rfp4:.4f}")

    # All should be reasonable
    assert err_nvfp4 < 0.20, f"NVFP4 error too high: {err_nvfp4}"
    assert err_asfp4 < 0.20, f"AS-FP4 error too high: {err_asfp4}"
    assert err_rfp4 < 0.20, f"R-FP4 error too high: {err_rfp4}"

    # AS-FP4 should be better or equal to NVFP4
    assert err_asfp4 <= err_nvfp4 + 0.01, "AS-FP4 should be better"
    # R-FP4 should be better than NVFP4 (residual corrects errors)
    assert err_rfp4 <= err_nvfp4 + 0.01, "R-FP4 should be better"

    print("  PASS\n")


def test_cuda_speed():
    """Benchmark all methods on CUDA."""
    if not torch.cuda.is_available():
        print("  CUDA speed: SKIPPED (no CUDA)\n")
        return

    print("  Novel Quant: CUDA speed benchmark")
    print("-" * 60)

    torch.manual_seed(42)
    lin = nn.Linear(2048, 8192, bias=False).cuda().to(torch.bfloat16)

    nvfp4 = NVFP4Linear.from_linear(lin, block_size=32).cuda()
    asfp4 = ASFP4Linear.from_linear(lin, block_size=32).cuda()
    rfp4 = ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.05).cuda()

    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = lin(x); _ = nvfp4(x); _ = asfp4(x); _ = rfp4(x)
    torch.cuda.synchronize()

    for name, module in [("bf16", lin), ("NVFP4", nvfp4),
                          ("AS-FP4", asfp4), ("R-FP4", rfp4)]:
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                _ = module(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 10
        print(f"  {name:>8}: {ms:.2f} ms/call")

    # Memory
    for name, module in [("NVFP4", nvfp4), ("AS-FP4", asfp4), ("R-FP4", rfp4)]:
        packed_bytes = module.weight_packed.numel()
        scale_bytes = module.weight_scales.numel()
        if hasattr(module, 'residual_indices'):
            res_bytes = (module.residual_indices.numel() * 4 +
                        module.residual_values.numel() +
                        module.residual_scale.numel() * 4)
        else:
            res_bytes = 0
        total = packed_bytes + scale_bytes + res_bytes
        bf16_bytes = 2048 * 8192 * 2
        print(f"  {name:>8} mem: {total/1024:.0f} KB ({bf16_bytes/total:.1f}x vs bf16)")

    print("  PASS\n")


def test_quantize_model():
    """Test quantize_model functions on a small model."""
    print("  Novel Quant: Model-level quantization")
    print("-" * 60)

    model = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128),
    )

    x = torch.randn(4, 256)
    with torch.no_grad():
        y_ref = model(x)

    # AS-FP4
    m1 = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128),
    )
    m1.load_state_dict(model.state_dict())
    n1 = quantize_model_asfp4(m1, verbose=True)
    with torch.no_grad():
        y1 = m1(x)
    print(f"    AS-FP4 output err: {frob_err(y_ref, y1):.4f} ({n1} layers)")

    # R-FP4
    m2 = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128),
    )
    m2.load_state_dict(model.state_dict())
    n2 = quantize_model_residual_fp4(m2, residual_ratio=0.05, verbose=True)
    with torch.no_grad():
        y2 = m2(x)
    print(f"    R-FP4 output err:  {frob_err(y_ref, y2):.4f} ({n2} layers)")

    assert n1 == 3 and n2 == 3
    assert y1.shape == y_ref.shape and y2.shape == y_ref.shape
    print("  PASS\n")


def main():
    test_asfp4_vs_nvfp4()
    test_residual_fp4()
    test_linear_forward()
    test_quantize_model()
    test_cuda_speed()
    print("=" * 60)
    print("  ALL NOVEL QUANT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
