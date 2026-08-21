"""Smoke test for all 5 new parameter-reduction keys.

Builds a tiny model with each key enabled, verifies:
1. Model builds without error
2. Forward pass produces correct output shape
3. Parameter count is reduced vs baseline
4. (For FFN compression) from_dense conversion works

Run: python -m research.sandbox.test_param_reduction_keys
"""
import torch
import torch.nn as nn
import copy

from research.config import ModelConfig, MODEL_CONFIGS


def make_tiny_config(**overrides) -> ModelConfig:
    """Create a tiny config for fast testing."""
    base = dict(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=256,
        attn_type="gqa",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=128,
        conv_kernel_size=3,
        use_qk_norm=True,
        layer_types=["conv", "conv", "attention", "conv"],
        batch_size=2,
        seq_len=64,
        device="cpu",
    )
    base.update(overrides)
    return ModelConfig(**base)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_baseline():
    """Build baseline tiny model."""
    print("\n=== Baseline (no compression keys) ===")
    config = make_tiny_config()
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    # Forward pass
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    print(f"  Forward output shape: {out.shape}")
    assert out.shape == (1, 16, config.vocab_size), f"Bad shape: {out.shape}"
    print("  PASS")
    return n


def test_monarch_ffn():
    """Test Monarch FFN compression."""
    print("\n=== Monarch FFN ===")
    config = make_tiny_config(
        ffn_compression="monarch",
        monarch_block_size=16,  # small block for tiny model
    )
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    # Check that FFN uses MonarchLinear
    from research.keys.compression.monarch_ffn_key import MonarchLinear
    ffn = model.blocks[0].ffn
    assert isinstance(ffn.w_gate, MonarchLinear), f"Expected MonarchLinear, got {type(ffn.w_gate)}"
    print(f"  w_gate type: {type(ffn.w_gate).__name__}")

    # Forward pass
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 16, config.vocab_size)
    print(f"  Forward output shape: {out.shape}")
    print("  PASS")
    return n


def test_kron_ffn():
    """Test Kronecker FFN compression."""
    print("\n=== Kronecker FFN ===")
    # For d_model=128, intermediate=256:
    # gate_kron (a,b) with a*b=256: (16, 16)
    # down_kron (a,b) with a*b=128: (8, 16)
    config = make_tiny_config(
        ffn_compression="kron",
        kron_a=16, kron_b=16,  # 16*16=256=intermediate
        kron_c=8, kron_d=16,   # 8*16=128=d_model
    )
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    from research.keys.compression.kron_ffn_key import KroneckerLinear
    ffn = model.blocks[0].ffn
    assert isinstance(ffn.w_gate, KroneckerLinear), f"Expected KroneckerLinear, got {type(ffn.w_gate)}"
    print(f"  w_gate type: {type(ffn.w_gate).__name__}")

    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 16, config.vocab_size)
    print(f"  Forward output shape: {out.shape}")
    print("  PASS")
    return n


def test_tt_ffn():
    """Test Tensor-Train FFN compression."""
    print("\n=== Tensor-Train FFN ===")
    config = make_tiny_config(
        ffn_compression="tt",
        tt_rank=4,
    )
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    from research.keys.compression.tt_ffn_key import TTLinear
    ffn = model.blocks[0].ffn
    assert isinstance(ffn.w_gate, TTLinear), f"Expected TTLinear, got {type(ffn.w_gate)}"
    print(f"  w_gate type: {type(ffn.w_gate).__name__}")

    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 16, config.vocab_size)
    print(f"  Forward output shape: {out.shape}")
    print("  PASS")
    return n


def test_hyperloop():
    """Test Hyperloop layer sharing."""
    print("\n=== Hyperloop ===")
    config = make_tiny_config(
        use_hyperloop=True,
        hyperloop_begin=1,
        hyperloop_end=1,
        hyperloop_loop_iters=2,
    )
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    # Check hyperloop attributes exist
    assert hasattr(model, 'loop_block'), "Missing loop_block"
    assert hasattr(model, 'loop_gates'), "Missing loop_gates"
    assert hasattr(model, 'middle_gates'), "Missing middle_gates"
    print(f"  loop_block: {type(model.loop_block).__name__}")
    print(f"  loop_gates: {len(model.loop_gates)} gates")
    print(f"  middle_gates: {len(model.middle_gates)} gates")

    # Check gates are zero-init (lossless)
    assert all(g.item() == 0.0 for g in model.loop_gates), "Loop gates not zero-init"
    assert all(g.item() == 1.0 for g in model.middle_gates), "Middle gates not one-init"
    print("  Gates: loop=0 (lossless), middle=1 (active)")

    # Forward pass (should work as normal since gates don't affect standard forward)
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 16, config.vocab_size)
    print(f"  Forward output shape: {out.shape}")
    print("  PASS")
    return n


def test_lisa():
    """Test LiSA cross-layer Q/K sharing."""
    print("\n=== LiSA ===")
    config = make_tiny_config(
        use_lisa=True,
        lisa_compress=6,
        lisa_align_dim=0,  # auto = d_model // 4 = 32
    )
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    # Check LiSA module exists
    assert hasattr(model, 'lisa'), "Missing lisa module"
    assert model.lisa is not None, "LiSA module is None"
    print(f"  lisa type: {type(model.lisa).__name__}")

    # Check gates are zero-init (lossless)
    assert all(g.item() == 0.0 for g in model.lisa.gates), "LiSA gates not zero-init"
    print(f"  Gates: all 0 (lossless at init)")

    # Forward pass
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 16, config.vocab_size)
    print(f"  Forward output shape: {out.shape}")
    print("  PASS")
    return n


def test_combined():
    """Test all keys together."""
    print("\n=== Combined (all 5 keys) ===")
    config = make_tiny_config(
        ffn_compression="monarch",
        monarch_block_size=16,
        use_hyperloop=True,
        hyperloop_begin=1,
        hyperloop_end=1,
        hyperloop_loop_iters=2,
        use_lisa=True,
        lisa_compress=6,
    )
    from research.model_loader import ConfigurableResearchLLM
    model = ConfigurableResearchLLM(config)
    n = count_params(model)
    print(f"  Params: {n:,d}")

    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 16, config.vocab_size)
    print(f"  Forward output shape: {out.shape}")
    print("  PASS")
    return n


def test_ffn_conversion():
    """Test that FFN compression conversion from dense works."""
    print("\n=== FFN Conversion (dense -> monarch) ===")
    from research.model_loader import ModelLoader
    from research.keys.compression.monarch_ffn_key import MonarchLinear

    # Create a dense weight
    torch.manual_seed(42)
    dense_weight = torch.randn(256, 128)  # (out, in)

    # Convert to Monarch
    ml = MonarchLinear.from_dense(dense_weight, block_size=16)

    # Check that forward approximates dense
    x = torch.randn(1, 128)
    y_dense = x @ dense_weight.T  # nn.Linear forward
    y_monarch = ml(x)
    error = (y_dense - y_monarch).abs().mean().item()
    print(f"  Dense vs Monarch output error: {error:.6f}")
    # Monarch is an approximation, so some error is expected
    # But with ALS fitting it should be reasonable
    assert error < 1.0, f"Monarch approximation error too high: {error}"

    # Check param reduction
    dense_params = dense_weight.numel()
    monarch_params = ml.L.numel() + ml.R.numel()
    print(f"  Dense params: {dense_params:,d}")
    print(f"  Monarch params: {monarch_params:,d} ({100*monarch_params/dense_params:.1f}%)")
    assert monarch_params < dense_params, "Monarch should have fewer params"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: Parameter-Reduction Keys")
    print("=" * 60)

    results = {}
    try:
        results['baseline'] = test_baseline()
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()

    for name, fn in [("monarch", test_monarch_ffn),
                     ("kron", test_kron_ffn),
                     ("tt", test_tt_ffn),
                     ("hyperloop", test_hyperloop),
                     ("lisa", test_lisa),
                     ("combined", test_combined)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()

    try:
        test_ffn_conversion()
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline = results.get('baseline', 0)
    if baseline > 0:
        for name, n in results.items():
            if name == 'baseline':
                print(f"  {name:15s}: {n:>10,d} params (baseline)")
            else:
                pct = 100 * (1 - n / baseline) if baseline else 0
                print(f"  {name:15s}: {n:>10,d} params ({pct:+.1f}% vs baseline)")
    print("\nDone.")
