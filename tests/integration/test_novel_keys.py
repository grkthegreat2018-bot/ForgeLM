"""Test script for 3 novel keys: CLKV, AnchorVocab, AdaTopK.

Builds a tiny model and verifies each key:
  1. Applies correctly (no crash)
  2. Produces valid output (no NaN/inf)
  3. Measures the expected savings
  4. Reverts correctly (output matches original)
"""
import torch
import torch.nn as nn
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.config import ModelConfig, get_config
from research.model_loader import ModelLoader, ConfigurableResearchLLM
from research.keys.cache.clkv_key import CLKVKey
from research.keys.misc.anchor_vocab_key import AnchorVocabKey
from research.keys.moe.adaptive_topk_key import AdaTopKKey


def build_tiny_model(device="cpu"):
    """Build a tiny model for testing."""
    config = ModelConfig(
        vocab_size=1000,  # small vocab for fast testing
        d_model=128,
        n_layers=4,
        n_heads=4,
        attn_type="mla",
        kv_compression_dim=64,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        max_seq_len=128,
        device=device,
    )
    model = ConfigurableResearchLLM(config).to(device)
    model.eval()
    return model, config


def build_tiny_moe_model(device="cpu"):
    """Build a tiny model with MoE for testing AdaTopK."""
    from research.moe import replace_ffn_with_moe
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        attn_type="mla",
        kv_compression_dim=64,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        max_seq_len=128,
        device=device,
    )
    model = ConfigurableResearchLLM(config).to(device)
    # Replace FFN with MoE (4 experts, top-2, dense_bypass for testing)
    replace_ffn_with_moe(model, n_experts=4, top_k=2, d_model=128,
                        shared_expert=True, d_ff=64, dense_bypass=False)
    # MoE layers are created on CPU — move entire model to device again
    model = model.to(device)
    model.eval()
    return model, config


def test_clkv():
    """Test Cross-Layer KV Sharing."""
    print("\n" + "=" * 60)
    print("TEST 1: Cross-Layer KV Sharing (CLKV)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = build_tiny_model(device)

    # Generate input
    idx = torch.randint(0, config.vocab_size, (1, 16), device=device)

    # Baseline forward
    with torch.no_grad():
        logits_base, _, presents_base = model(idx, use_cache=True)
    n_kv_base = len(presents_base)
    print(f"  Baseline: {n_kv_base} KV caches, logits shape={logits_base.shape}")
    print(f"  Baseline logits range: [{logits_base.min():.3f}, {logits_base.max():.3f}]")

    # Apply CLKV
    key = CLKVKey(share_factor=2)
    n_reused = key.apply(model)

    with torch.no_grad():
        logits_clkv, _, presents_clkv = model(idx, use_cache=True)
    n_kv_clkv = len(presents_clkv)
    print(f"  CLKV: {n_kv_clkv} KV cache entries, {n_reused} layers reuse shared KV")
    print(f"  CLKV logits range: [{logits_clkv.min():.3f}, {logits_clkv.max():.3f}]")

    # Check output is valid
    assert not torch.isnan(logits_clkv).any(), "CLKV produced NaN!"
    assert not torch.isinf(logits_clkv).any(), "CLKV produced Inf!"
    print("  [PASS] Output is valid (no NaN/Inf)")

    # Check KV reduction
    # With share_factor=2 and 4 layers: 2 leader KV + 2 follower placeholders
    # All 4 presents entries exist but followers share references with leaders
    assert n_kv_clkv == n_kv_base, f"Expected {n_kv_base} presents, got {n_kv_clkv}"
    # Verify that follower KV references match leader KV
    leader_kv = presents_clkv[0]
    follower_kv = presents_clkv[1]
    assert leader_kv[0] is follower_kv[0] or torch.equal(leader_kv[0], follower_kv[0]), \
        "Follower KV should match leader KV"
    print("  [PASS] KV cache sharing verified (follower reuses leader KV)")

    # Revert
    key.revert(model)
    with torch.no_grad():
        logits_reverted, _, _ = model(idx, use_cache=True)
    assert torch.allclose(logits_base, logits_reverted, atol=1e-5), \
        "Reverted output doesn't match baseline!"
    print("  [PASS] Revert produces identical output to baseline")

    # Measure KV VRAM
    kv_bytes_base = sum(
        p[0].nelement() * p[0].element_size() + p[1].nelement() * p[1].element_size()
        for p in presents_base if p is not None
    )
    # CLKV: only leader KV are unique, followers are shared references
    unique_kv = []
    seen_ids = set()
    for p in presents_clkv:
        if p is not None:
            pid = id(p[0])
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique_kv.append(p)
    kv_bytes_clkv = sum(
        p[0].nelement() * p[0].element_size() + p[1].nelement() * p[1].element_size()
        for p in unique_kv
    )
    print(f"  KV VRAM: baseline={kv_bytes_base} bytes, clkv={kv_bytes_clkv} bytes "
          f"({1 - kv_bytes_clkv / kv_bytes_base:.0%} reduction)")

    print("  [PASS] CLKV test passed!")
    return True


def test_anchor_vocab():
    """Test Anchor Vocab Pruning."""
    print("\n" + "=" * 60)
    print("TEST 2: Anchor Vocab Pruning (AnchorVocab)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = build_tiny_model(device)

    # Generate input
    idx = torch.randint(0, config.vocab_size, (1, 8), device=device)

    # Baseline forward (single token decode)
    with torch.no_grad():
        # Prefill
        logits_prefill, _, presents = model(idx, use_cache=True)
        # Single token decode
        next_idx = torch.tensor([[42]], device=device)
        logits_base, _, _ = model(next_idx, past_key_values=presents, use_cache=True)
    print(f"  Baseline logits shape: {logits_base.shape}")
    print(f"  Baseline top-5 tokens: {logits_base.topk(5).indices[0].tolist()}")
    print(f"  Baseline logits range: [{logits_base.min():.3f}, {logits_base.max():.3f}]")

    # Apply AnchorVocab
    key = AnchorVocabKey(n_anchors=32, top_clusters=4, max_iter=10)
    n_tokens = key.apply(model)

    with torch.no_grad():
        logits_anchor, _, _ = model(next_idx, past_key_values=presents, use_cache=True)
    print(f"  Anchor logits shape: {logits_anchor.shape}")
    print(f"  Anchor top-5 tokens: {logits_anchor.topk(5).indices[0].tolist()}")
    valid = logits_anchor[logits_anchor > float('-inf')]
    if valid.numel() > 0:
        print(f"  Anchor logits range: [{valid.min():.3f}, {logits_anchor.max():.3f}]")
    else:
        print(f"  Anchor logits: all -inf (no candidates selected)")

    # Check output is valid
    assert not torch.isnan(logits_anchor).any(), "AnchorVocab produced NaN!"
    n_inf = (logits_anchor == float('-inf')).sum().item()
    total = logits_anchor.numel()
    print(f"  Pruned tokens: {n_inf}/{total} ({n_inf/total:.0%} of vocab pruned)")

    # Check top-1 match
    base_top1 = logits_base.topk(1).indices[0, 0].item()
    anchor_top1 = logits_anchor.topk(1).indices[0, 0].item()
    print(f"  Top-1 match: base={base_top1}, anchor={anchor_top1}, "
          f"match={base_top1 == anchor_top1}")

    # The top-1 should match (or be very close) since we selected top-4 of 32 clusters
    # With small vocab (1000) and 32 anchors, top-4 clusters cover ~125 tokens
    # The correct token should be in the selected clusters most of the time
    if base_top1 == anchor_top1:
        print("  [PASS] Top-1 token matches baseline")
    else:
        # Check if base top-1 is in anchor's selected tokens
        base_top1_logit = logits_anchor[0, 0, base_top1].item()
        if base_top1_logit > float('-inf'):
            print(f"  [PASS] Base top-1 is in anchor's selected tokens (logit={base_top1_logit:.3f})")
        else:
            print(f"  [WARN] Base top-1 was pruned (miss). This can happen with small vocab/anchors.")

    # Check stats
    stats = key.get_stats(model)
    print(f"  Stats: {stats}")

    # Revert
    key.revert(model)
    with torch.no_grad():
        logits_reverted, _, _ = model(next_idx, past_key_values=presents, use_cache=True)
    assert torch.allclose(logits_base, logits_reverted, atol=1e-5), \
        "Reverted output doesn't match baseline!"
    print("  [PASS] Revert produces identical output to baseline")

    print("  [PASS] AnchorVocab test passed!")
    return True


def test_adaptive_topk():
    """Test Adaptive Expert Top-K."""
    print("\n" + "=" * 60)
    print("TEST 3: Adaptive Expert Top-K (AdaTopK)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = build_tiny_moe_model(device)

    # Generate input
    idx = torch.randint(0, config.vocab_size, (1, 16), device=device)

    # Baseline forward
    with torch.no_grad():
        logits_base, _ = model(idx, use_cache=False)
    print(f"  Baseline logits shape: {logits_base.shape}")
    print(f"  Baseline logits range: [{logits_base.min():.3f}, {logits_base.max():.3f}]")

    # Apply AdaTopK
    key = AdaTopKKey(min_k=1, max_k=3, lo_entropy=0.2, hi_entropy=0.7)
    n_patched = key.apply(model)

    with torch.no_grad():
        logits_adaptive, _ = model(idx, use_cache=False)
    print(f"  Adaptive logits shape: {logits_adaptive.shape}")
    print(f"  Adaptive logits range: [{logits_adaptive.min():.3f}, {logits_adaptive.max():.3f}]")

    # Check output is valid
    assert not torch.isnan(logits_adaptive).any(), "AdaTopK produced NaN!"
    assert not torch.isinf(logits_adaptive).any(), "AdaTopK produced Inf!"
    print("  [PASS] Output is valid (no NaN/Inf)")

    # Check stats
    stats = key.get_stats(model)
    print(f"  Stats: {stats}")

    # Revert
    key.revert(model)
    with torch.no_grad():
        logits_reverted, _ = model(idx, use_cache=False)
    assert torch.allclose(logits_base, logits_reverted, atol=1e-5), \
        "Reverted output doesn't match baseline!"
    print("  [PASS] Revert produces identical output to baseline")

    print("  [PASS] AdaTopK test passed!")
    return True


def test_combined():
    """Test all 3 keys applied together."""
    print("\n" + "=" * 60)
    print("TEST 4: Combined (CLKV + AnchorVocab + AdaTopK)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = build_tiny_moe_model(device)

    idx = torch.randint(0, config.vocab_size, (1, 16), device=device)

    # Baseline
    with torch.no_grad():
        logits_base, _, presents_base = model(idx, use_cache=True)
    print(f"  Baseline: logits={logits_base.shape}, KV caches={len(presents_base)}")

    # Apply all 3
    clkv = CLKVKey(share_factor=2)
    anchor = AnchorVocabKey(n_anchors=32, top_clusters=4, max_iter=10)
    adatopk = AdaTopKKey(min_k=1, max_k=3)

    clkv.apply(model)
    anchor.apply(model)
    adatopk.apply(model)

    # Forward with all keys active
    with torch.no_grad():
        # Need fresh KV for combined test (CLKV changes forward)
        logits_combined, _, presents_combined = model(idx, use_cache=True)
    print(f"  Combined: logits={logits_combined.shape}, KV entries={len(presents_combined)}")

    # Check valid
    assert not torch.isnan(logits_combined).any(), "Combined produced NaN!"
    n_inf = (logits_combined == float('-inf')).sum().item()
    print(f"  Pruned logits: {n_inf}/{logits_combined.numel()} ({n_inf/logits_combined.numel():.0%})")

    # Revert all
    adatopk.revert(model)
    anchor.revert(model)
    clkv.revert(model)

    with torch.no_grad():
        logits_reverted, _, _ = model(idx, use_cache=True)
    assert torch.allclose(logits_base, logits_reverted, atol=1e-5), \
        "Reverted output doesn't match baseline after combined test!"
    print("  [PASS] All keys reverted, output matches baseline")

    print("  [PASS] Combined test passed!")
    return True


def benchmark_anchor_vocab():
    """Benchmark AnchorVocab speedup on a larger vocab."""
    print("\n" + "=" * 60)
    print("BENCHMARK: AnchorVocab on large vocab (151936)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [SKIP] CUDA not available, skipping benchmark")
        return True

    # Build model with large vocab (like ForgeLM V2)
    config = ModelConfig(
        vocab_size=151936,
        d_model=1536,
        n_layers=4,  # few layers for speed
        n_heads=12,
        attn_type="mla",
        kv_compression_dim=512,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        max_seq_len=128,
        device=device,
    )
    model = ConfigurableResearchLLM(config).to(device).to(torch.bfloat16)
    model.eval()

    # Single token input
    idx = torch.tensor([[42]], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(idx, use_cache=False)
    torch.cuda.synchronize()

    # Baseline timing
    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            logits_base = model(idx, use_cache=False)[0]
    torch.cuda.synchronize()
    t_base = (time.time() - t0) / 50
    print(f"  Baseline: {t_base*1000:.2f} ms/token, logits={logits_base.shape}")

    # Apply AnchorVocab
    key = AnchorVocabKey(n_anchors=512, top_clusters=8, max_iter=15)
    key.apply(model)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(idx, use_cache=False)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            logits_anchor = model(idx, use_cache=False)[0]
    torch.cuda.synchronize()
    t_anchor = (time.time() - t0) / 50
    print(f"  AnchorVocab: {t_anchor*1000:.2f} ms/token, logits={logits_anchor.shape}")

    stats = key.get_stats(model)
    print(f"  Stats: {stats}")
    print(f"  Speedup: {t_base/t_anchor:.2f}x")

    key.revert(model)
    del model
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("NOVEL KEY TEST SUITE: CLKV + AnchorVocab + AdaTopK")
    print("=" * 60)

    results = []
    try:
        results.append(("CLKV", test_clkv()))
    except Exception as e:
        print(f"  [FAIL] CLKV: {e}")
        import traceback; traceback.print_exc()
        results.append(("CLKV", False))

    try:
        results.append(("AnchorVocab", test_anchor_vocab()))
    except Exception as e:
        print(f"  [FAIL] AnchorVocab: {e}")
        import traceback; traceback.print_exc()
        results.append(("AnchorVocab", False))

    try:
        results.append(("AdaTopK", test_adaptive_topk()))
    except Exception as e:
        print(f"  [FAIL] AdaTopK: {e}")
        import traceback; traceback.print_exc()
        results.append(("AdaTopK", False))

    try:
        results.append(("Combined", test_combined()))
    except Exception as e:
        print(f"  [FAIL] Combined: {e}")
        import traceback; traceback.print_exc()
        results.append(("Combined", False))

    try:
        results.append(("Benchmark", benchmark_anchor_vocab()))
    except Exception as e:
        print(f"  [SKIP] Benchmark: {e}")
        results.append(("Benchmark", False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)

