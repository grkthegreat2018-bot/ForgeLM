"""Tests for R&D Round 23: DLoRA (DoRA + LoRA) for warm-starting V8 from V7-8B-B.

DLoRA decomposes weights into magnitude + direction (DoRA), then applies LoRA
to the direction component. This allows efficient adaptation of a pre-trained
model with fewer trainable parameters than full fine-tuning while preserving
the magnitude-direction structure that DoRA showed improves quality.
Will be implemented at research/training/dlora.py.
"""
import os, sys, tempfile, math
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── R23a: DLoRA imports & structure ─────────────────────────────────────────

def test_dlora_imports():
    """Import research.training.dlora, verify DLoRAAdapter class exists."""
    from research.training.dlora import DLoRAAdapter

    assert DLoRAAdapter is not None, "DLoRAAdapter should be importable"
    assert isinstance(DLoRAAdapter, type), "DLoRAAdapter should be a class"
    print("  dlora_imports: PASS")


def test_dlora_adapter_creation():
    """Create DLoRAAdapter, verify it has magnitude, direction, and LoRA A/B."""
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8)

    # Magnitude parameter (per-output scalar or vector)
    assert hasattr(adapter, "magnitude"), "DLoRAAdapter must have magnitude"
    # Direction parameter (the base direction matrix)
    assert hasattr(adapter, "direction"), "DLoRAAdapter must have direction"
    # LoRA matrices A and B
    assert hasattr(adapter, "lora_A"), "DLoRAAdapter must have lora_A"
    assert hasattr(adapter, "lora_B"), "DLoRAAdapter must have lora_B"

    # Check shapes
    assert adapter.lora_A.shape == (8, 128), f"lora_A shape wrong: {adapter.lora_A.shape}"
    assert adapter.lora_B.shape == (128, 8), f"lora_B shape wrong: {adapter.lora_B.shape}"
    print("  dlora_adapter_creation: PASS")


def test_dlora_zero_init():
    """At init, DLoRA should be a no-op (LoRA B=0, magnitude=1).

    The forward output should equal the base weight forward at initialization.
    """
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8).to(_DEV)

    # LoRA B should be zero at init (no-op)
    assert torch.all(adapter.lora_B == 0), "LoRA B should be zero-initialized"

    # Magnitude should be 1 at init
    mag = adapter.magnitude
    if mag.dim() == 0:
        assert abs(mag.item() - 1.0) < 1e-6, f"magnitude should be 1, got {mag.item()}"
    else:
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-6), \
            "magnitude should be all ones at init"

    # Forward should match base direction forward (since LoRA=0, mag=1)
    x = torch.randn(4, 128, device=_DEV)
    with torch.no_grad():
        out_dlora = adapter(x)
        # Base direction forward: x @ direction.T
        base_out = F_linear_dir(adapter, x)
    assert torch.allclose(out_dlora, base_out, atol=1e-5), \
        "DLoRA forward at init should equal base direction forward"
    print("  dlora_zero_init: PASS")


def F_linear_dir(adapter, x):
    """Helper: plain linear using direction weight."""
    d = adapter.direction
    if d.dim() == 2:
        return F.linear(x, d)
    return x @ d.T


def test_dlora_forward():
    """Forward pass through DLoRAAdapter, verify output shape and finite values."""
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8).to(_DEV)
    x = torch.randn(4, 32, 128, device=_DEV)

    with torch.no_grad():
        out = adapter(x)

    assert out.shape == (4, 32, 128), f"Output shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "Output should be finite"
    print("  dlora_forward: PASS")


def test_dlora_backward():
    """Forward + backward, verify gradients exist for LoRA but NOT base weight."""
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8).to(_DEV)

    # Freeze the base direction (it's the pre-trained weight)
    if hasattr(adapter, "direction") and isinstance(adapter.direction, nn.Parameter):
        adapter.direction.requires_grad = False

    x = torch.randn(4, 128, device=_DEV)
    out = adapter(x)
    loss = out.sum()
    loss.backward()

    # LoRA params should have gradients
    assert adapter.lora_A.grad is not None, "lora_A should have gradient"
    assert adapter.lora_B.grad is not None, "lora_B should have gradient"

    # Magnitude should have gradient (it's trainable in DoRA)
    if isinstance(adapter.magnitude, nn.Parameter):
        assert adapter.magnitude.grad is not None, "magnitude should have gradient"

    # Base direction should NOT have gradient (frozen)
    if isinstance(adapter.direction, nn.Parameter):
        assert adapter.direction.grad is None, \
            "Base direction should be frozen (no gradient)"
    print("  dlora_backward: PASS")


def test_dlora_merge():
    """After training, merge DLoRA back into base weight.

    Merged weight = base + magnitude * (direction + LoRA_delta).
    Forward output should match before and after merge.
    """
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8).to(_DEV)

    # Simulate some training: perturb LoRA B and magnitude
    with torch.no_grad():
        adapter.lora_B.add_(torch.randn_like(adapter.lora_B) * 0.01)
        if adapter.magnitude.dim() == 0:
            adapter.magnitude.add_(0.05)
        else:
            adapter.magnitude.add_(torch.randn_like(adapter.magnitude) * 0.05)

    x = torch.randn(4, 128, device=_DEV)
    with torch.no_grad():
        out_before = adapter(x)

    # Merge: compute the effective weight
    with torch.no_grad():
        lora_delta = adapter.scale * (adapter.lora_B @ adapter.lora_A)  # (out, in)
        direction = adapter.direction
        if direction.dim() == 2:
            effective_w = adapter.magnitude * (direction + lora_delta)
        else:
            effective_w = adapter.magnitude * (direction + lora_delta)

    # Verify forward with effective weight matches adapter forward
    with torch.no_grad():
        out_merged = F.linear(x, effective_w)
    assert torch.allclose(out_before, out_merged, atol=1e-4), \
        "Merged weight forward should match adapter forward"
    print("  dlora_merge: PASS")


def test_dlora_with_bitnet():
    """Add DLoRA to a BitNetLinear layer, verify BitNet quantize + DLoRA works."""
    from research.training.dlora import DLoRAAdapter
    from research.keys.quantization.bitnet_b158_key import BitNetLinear

    torch.manual_seed(42)
    bitnet_layer = BitNetLinear(128, 128, quantize=True).to(_DEV)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8).to(_DEV)

    x = torch.randn(4, 128, device=_DEV)
    with torch.no_grad():
        base_out = bitnet_layer(x)
        lora_out = adapter(x)
        combined = base_out + lora_out

    assert combined.shape == (4, 128), "Combined output shape wrong"
    assert torch.isfinite(combined).all(), "Combined output should be finite"
    print("  dlora_with_bitnet: PASS")


def test_dlora_param_count():
    """Verify DLoRA trainable params << base model params.

    For 128x128 layer with rank=8:
      LoRA params = 8*128 + 128*8 = 2048 vs base 16384.
      Ratio should be < 15%.
    """
    from research.training.dlora import DLoRAAdapter

    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8)

    lora_params = adapter.lora_A.numel() + adapter.lora_B.numel()
    base_params = 128 * 128
    ratio = lora_params / base_params

    print(f"  LoRA params: {lora_params}, base: {base_params}, ratio: {ratio:.2%}")
    assert lora_params == 2048, f"Expected 2048 LoRA params, got {lora_params}"
    assert ratio < 0.15, f"LoRA param ratio should be < 15%, got {ratio:.2%}"
    print("  dlora_param_count: PASS")


def test_dlora_reduces_loss():
    """Train 20 steps with DLoRA on MSE task, verify loss decreases.

    Base weights frozen, only LoRA + magnitude train.
    """
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    adapter = DLoRAAdapter(in_features=128, out_features=128, rank=8).to(_DEV)

    # Freeze base direction
    if isinstance(adapter.direction, nn.Parameter):
        adapter.direction.requires_grad = False

    # Collect trainable params (LoRA + magnitude)
    params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2)

    x = torch.randn(16, 128, device=_DEV)
    y = torch.randn(16, 128, device=_DEV)

    initial_loss = nn.MSELoss()(adapter(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = nn.MSELoss()(adapter(x), y)
        loss.backward()
        opt.step()
    final_loss = loss.item()

    print(f"  DLoRA loss: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease with DLoRA training"
    print("  dlora_reduces_loss: PASS")


def test_dlora_warmstart_v7_to_v8():
    """Build tiny V7 and V8 models (same dims for test), port V7 weights to V8,
    add DLoRA to V8, verify forward works and DLoRA starts as no-op (lossless port).
    """
    from research.training.dlora import DLoRAAdapter

    torch.manual_seed(42)
    d = 128

    # Tiny V7 model (just a linear layer for this test)
    v7 = nn.Linear(d, d, bias=False).to(_DEV)
    # Tiny V8 model (same dims for this test — real V8 is 2x wider)
    v8 = nn.Linear(d, d, bias=False).to(_DEV)

    # Port V7 weights to V8 (identity port since same dims)
    with torch.no_grad():
        v8.weight.copy_(v7.weight)

    # Add DLoRA to V8
    adapter = DLoRAAdapter(in_features=d, out_features=d, rank=8).to(_DEV)

    x = torch.randn(4, d, device=_DEV)
    with torch.no_grad():
        # V7 forward
        out_v7 = v7(x)
        # V8 forward (ported weights)
        out_v8_base = v8(x)
        # V8 + DLoRA (should be no-op at init)
        out_v8_dlora = v8(x) + adapter(x)

    # Ported V8 should match V7 (lossless port)
    assert torch.allclose(out_v7, out_v8_base, atol=1e-6), \
        "V8 ported weights should match V7"
    # DLoRA at init should be no-op (V8+DLoRA == V8 == V7)
    assert torch.allclose(out_v8_dlora, out_v8_base, atol=1e-5), \
        "DLoRA should start as no-op (lossless warm start)"
    print("  dlora_warmstart_v7_to_v8: PASS")


# ── Main ────────────────────────────────────────────────────────────────────

def main_r23_dlora():
    print("=" * 70)
    print("  R&D ROUND 23: DLoRA (DoRA + LoRA) for V8 Warm-Start")
    print("=" * 70)

    print("\n  R23a: DLoRA imports & structure")
    test_dlora_imports()
    test_dlora_adapter_creation()
    test_dlora_zero_init()

    print("\n  R23b: DLoRA forward & backward")
    test_dlora_forward()
    test_dlora_backward()

    print("\n  R23c: DLoRA merge")
    test_dlora_merge()

    print("\n  R23d: DLoRA with BitNet")
    test_dlora_with_bitnet()

    print("\n  R23e: DLoRA efficiency")
    test_dlora_param_count()
    test_dlora_reduces_loss()

    print("\n  R23f: DLoRA warm-start V7 -> V8")
    test_dlora_warmstart_v7_to_v8()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 DLoRA TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_dlora()
