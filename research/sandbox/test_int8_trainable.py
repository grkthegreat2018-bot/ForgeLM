"""Quick test: BitNet int8 trainable storage (R&D round 15).

Verifies:
1. enable_int8_training() converts BitNetLinear correctly
2. Forward pass works with int8 weights
3. Backward pass sends gradients to CPU master
4. requantize_from_master() refreshes the int8 buffer
5. Memory: int8 on GPU, bf16 master on CPU
"""
import torch
import sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

from research.keys.quantization.bitnet_b158_key import BitNetLinear, enable_int8_training

def test_int8_trainable():
    print("=== Test: BitNet int8 trainable storage ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a BitNetLinear layer
    lin = BitNetLinear(256, 512, quantize=True, learned_scale=True).to(device)
    print(f"  Before: weight on {lin.weight.device}, dtype={lin.weight.dtype}, "
          f"numel={lin.weight.numel()}")

    # Enable int8 training
    lin.enable_int8_training()
    print(f"  After:  weight on {lin.weight.device}, dtype={lin.weight.dtype}")
    print(f"          weight_int8 on {lin.weight_int8.device}, dtype={lin.weight_int8.dtype}")
    print(f"          _int8_trainable={lin._int8_trainable}")

    # Verify storage split
    assert lin._int8_trainable, "Should be int8 trainable"
    assert lin.weight_int8.dtype == torch.int8, "int8 buffer should be int8"
    assert lin.weight.dtype == torch.bfloat16, "master should be bf16"
    if device == "cuda":
        assert lin.weight_int8.is_cuda, "int8 should be on GPU"
        assert not lin.weight.is_cuda, "master should be on CPU"
    print("  [PASS] Storage split correct")

    # Forward pass
    x = torch.randn(4, 256, device=device, dtype=torch.float32)
    y = lin(x)
    print(f"  Forward: input {x.shape} -> output {y.shape}")
    assert y.shape == (4, 512), f"Expected (4, 512), got {y.shape}"
    print("  [PASS] Forward pass works")

    # Backward pass — gradient should flow to CPU master
    loss = y.sum()
    loss.backward()
    print(f"  Backward: master grad on {lin.weight.grad.device if lin.weight.grad is not None else None}")
    assert lin.weight.grad is not None, "Master weight should have gradient"
    assert not lin.weight.grad.is_cuda, "Gradient should be on CPU (master device)"
    print("  [PASS] Gradient flows to CPU master")

    # Requantize from master
    old_int8 = lin.weight_int8.clone()
    # Perturb the master weight slightly
    with torch.no_grad():
        lin.weight.add_(torch.randn_like(lin.weight) * 0.01)
    lin.requantize_from_master()
    new_int8 = lin.weight_int8
    changed = (old_int8 != new_int8).sum().item()
    print(f"  Requantize: {changed} int8 values changed after master update")
    assert changed > 0, "Requantize should update some int8 values"
    print("  [PASS] requantize_from_master() works")

    # Memory check (GPU only)
    if device == "cuda":
        gpu_bytes = lin.weight_int8.numel()  # 1 byte/param
        cpu_bytes = lin.weight.numel() * 2   # 2 bytes/param (bf16)
        print(f"  Memory: GPU={gpu_bytes/1024:.0f}KB (int8), CPU={cpu_bytes/1024:.0f}KB (bf16 master)")
        print(f"  vs standard QAT: GPU={(lin.weight.numel()*4)/1024:.0f}KB (fp32 master on GPU)")
        print(f"  GPU savings: {(lin.weight.numel()*4 - gpu_bytes)/1024:.0f}KB ({4/1:.0f}x)")

    print("\n=== All tests passed! ===")

def test_enable_int8_training_on_model():
    """Test enable_int8_training() on a small model with multiple BitNetLinear."""
    print("\n=== Test: enable_int8_training on model ===")
    import torch.nn as nn

    model = nn.Sequential(
        BitNetLinear(128, 256, quantize=True),
        BitNetLinear(256, 128, quantize=True),
        BitNetLinear(128, 64, quantize=True),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n = enable_int8_training(model)
    print(f"  Converted {n} modules")
    assert n == 3, f"Expected 3, got {n}"

    # Forward + backward through full model
    x = torch.randn(2, 128, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # All master weights should have gradients
    for i, mod in enumerate(model):
        assert mod.weight.grad is not None, f"Module {i} master weight should have grad"
        assert mod.weight_int8.dtype == torch.int8, f"Module {i} int8 buffer wrong dtype"
    print("  [PASS] Full model forward+backward works")

    print("\n=== All model tests passed! ===")

if __name__ == "__main__":
    test_int8_trainable()
    test_enable_int8_training_on_model()
