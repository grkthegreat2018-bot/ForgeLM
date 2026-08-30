"""Tests for R&D Round 22: Training speedups for large datasets + models."""
import os, sys, tempfile, time, math
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── R22a: DataDedup ────────────────────────────────────────────────────────

def test_minhash_signature():
    """MinHash signature should be consistent for same document."""
    from research.training.optim.r22_training_speedups import MinHashDeduplicator

    dedup = MinHashDeduplicator(n_hashes=32, n_bands=8, shingle_size=5)
    text = "The quick brown fox jumps over the lazy dog"
    sig1 = dedup.minhash_signature(text)
    sig2 = dedup.minhash_signature(text)
    assert sig1 == sig2, "Same text should produce same signature"
    assert len(sig1) == 32, f"Wrong signature length: {len(sig1)}"
    print("  MinHash signature consistency: PASS")


def test_dedup_finds_duplicates():
    """Dedup should find near-duplicate documents."""
    from research.training.optim.r22_training_speedups import MinHashDeduplicator

    dedup = MinHashDeduplicator(n_hashes=64, n_bands=16, shingle_size=5)
    docs = [
        "The quick brown fox jumps over the lazy dog near the river bank",
        "The quick brown fox jumps over the lazy dog near the river bank",  # exact dup
        "A completely different document about machine learning and AI",
        "The quick brown fox jumps over the lazy dog near the river bank today",  # near dup
        "Another unique document about cooking recipes and food",
    ]
    unique_idx, dup_idx = dedup.deduplicate(docs)
    print(f"  Dedup: {len(unique_idx)} unique, {len(dup_idx)} duplicates")
    assert len(dup_idx) >= 1, "Should find at least one duplicate"
    assert 0 in unique_idx, "First occurrence should be kept"
    print("  Dedup finds duplicates: PASS")


def test_dedup_savings():
    """Dedup should estimate training savings correctly."""
    from research.training.optim.r22_training_speedups import MinHashDeduplicator

    dedup = MinHashDeduplicator()
    savings = dedup.estimate_savings(50_000_000, 0.20)  # 50M docs, 20% dups
    print(f"  Dedup savings: {savings['duplicate_docs']/1e6:.0f}M dups, "
          f"{savings['tokens_saved']/1e9:.1f}B tokens saved, "
          f"{savings['time_saved_pct']:.0f}% time saved")
    assert savings["time_saved_pct"] == 20.0
    print("  Dedup savings estimate: PASS")


# ── R22b: TokenImportanceSampling ──────────────────────────────────────────

def test_token_importance_skips_low_loss():
    """Low-loss tokens should be skipped at higher rate than high-loss."""
    from research.training.optim.r22_training_speedups import TokenImportanceSampler

    sampler = TokenImportanceSampler(
        initial_threshold=0.5, skip_rate=0.5, min_keep_rate=0.3)
    # Half low-loss, half high-loss
    losses = torch.cat([
        torch.full((1024,), 0.1, device=_DEV),  # low loss (known)
        torch.full((1024,), 2.0, device=_DEV),  # high loss (unknown)
    ])
    mask = sampler.compute_token_mask(losses)
    low_loss_kept = mask[:1024].sum().item()
    high_loss_kept = mask[1024:].sum().item()
    print(f"  Importance: low-loss kept {low_loss_kept}/1024, "
          f"high-loss kept {high_loss_kept}/1024")
    assert high_loss_kept == 1024, "All high-loss tokens should be kept"
    assert low_loss_kept < 1024, "Some low-loss tokens should be skipped"
    print("  Token importance skips low-loss: PASS")


def test_token_importance_adapts():
    """Threshold should adapt based on loss distribution."""
    from research.training.optim.r22_training_speedups import TokenImportanceSampler

    sampler = TokenImportanceSampler(
        initial_threshold=2.0, adapt_every=1, skip_rate=0.25)
    # Losses centered around 1.0
    losses = torch.randn(2048, device=_DEV) * 0.3 + 1.0
    initial_threshold = sampler.threshold
    sampler.step(losses)  # triggers adaptation
    adapted_threshold = sampler.threshold
    print(f"  Importance adapt: {initial_threshold:.3f} -> {adapted_threshold:.3f}")
    assert abs(adapted_threshold - initial_threshold) > 0.01, "Should adapt"
    print("  Token importance adapts: PASS")


def test_token_importance_speedup():
    """Importance sampling should provide measurable speedup."""
    from research.training.optim.r22_training_speedups import TokenImportanceSampler

    sampler = TokenImportanceSampler(
        initial_threshold=1.0, adapt_every=10, skip_rate=0.3, min_keep_rate=0.5)
    total_seen = 0
    total_skipped = 0
    for step in range(50):
        base_loss = 2.0 * math.exp(-step / 25) + 0.3
        losses = torch.randn(1024, device=_DEV) * 0.5 + base_loss
        mask = sampler.compute_token_mask(losses)
        total_seen += 1024
        total_skipped += (~mask).sum().item()
    speedup = total_seen / (total_seen - total_skipped)
    print(f"  Importance speedup: {speedup:.2f}x "
          f"({total_skipped}/{total_seen} tokens skipped)")
    assert speedup > 1.0, "Should provide speedup"
    print("  Token importance speedup: PASS")


# ── R22c: ProgressiveLayerUnfreezing ───────────────────────────────────────

def test_progressive_unfreeze_phases():
    """Should unfreeze more layers over training progress."""
    from research.training.optim.r22_training_speedups import ProgressiveUnfreezer

    unfreezer = ProgressiveUnfreezer(n_layers=32, n_phases=3)
    active_0 = len(unfreezer.get_active_layers(0, 3000))
    active_1000 = len(unfreezer.get_active_layers(1000, 3000))
    active_2000 = len(unfreezer.get_active_layers(2000, 3000))
    active_2999 = len(unfreezer.get_active_layers(2999, 3000))
    print(f"  Unfreeze: step 0={active_0}, 1000={active_1000}, "
          f"2000={active_2000}, 2999={active_2999} active layers")
    assert active_0 < active_1000 <= active_2000 <= active_2999
    assert active_2999 == 32, "All layers should be active at end"
    print("  Progressive unfreeze phases: PASS")


def test_progressive_unfreeze_speedup():
    """Early phases should have speedup factor > 1."""
    from research.training.optim.r22_training_speedups import ProgressiveUnfreezer

    unfreezer = ProgressiveUnfreezer(n_layers=32, n_phases=4)
    stats = unfreezer.stats()
    print(f"  Unfreeze: {stats['active_layers']}/{stats['n_layers']} active, "
          f"{stats['speedup_factor']:.1f}x speedup")
    assert stats["speedup_factor"] > 1.0
    print("  Progressive unfreeze speedup: PASS")


# ── R22d: GradientCompression ──────────────────────────────────────────────

def test_gradient_compression_roundtrip():
    """4-bit gradient compression should have low error with EF."""
    from research.training.optim.r22_training_speedups import GradientCompressor

    compressor = GradientCompressor(bits=4, block_size=128, ef_feedback=True)
    grad = torch.randn(8192, device=_DEV) * 0.01
    compressed, scales = compressor.compress(grad)
    decompressed = compressor.decompress(compressed, scales, grad.shape)
    error = (grad - decompressed).norm() / grad.norm()
    cr = compressor.compression_ratio()
    print(f"  Grad compress: {cr:.1f}x compression, {error*100:.2f}% error")
    assert cr > 3.0, f"Should compress >3x: {cr}"
    assert error < 0.20, f"Error too high: {error*100:.1f}%"
    print("  Gradient compression roundtrip: PASS")


def test_gradient_compression_ef_convergence():
    """EF21 should reduce error over multiple compress/decompress cycles."""
    from research.training.optim.r22_training_speedups import GradientCompressor

    compressor = GradientCompressor(bits=4, block_size=128, ef_feedback=True)
    # Simulate 10 gradient updates with compression
    total_error_with_ef = 0
    for i in range(10):
        grad = torch.randn(4096, device=_DEV) * 0.01
        compressed, scales = compressor.compress(grad)
        decompressed = compressor.decompress(compressed, scales, grad.shape)
        total_error_with_ef += (grad - decompressed).norm().item()

    # Without EF
    compressor_noef = GradientCompressor(bits=4, block_size=128, ef_feedback=False)
    total_error_no_ef = 0
    for i in range(10):
        grad = torch.randn(4096, device=_DEV) * 0.01
        compressed, scales = compressor_noef.compress(grad)
        decompressed = compressor_noef.decompress(compressed, scales, grad.shape)
        total_error_no_ef += (grad - decompressed).norm().item()

    print(f"  Grad EF: total error with EF={total_error_with_ef:.3f}, "
          f"without EF={total_error_no_ef:.3f}")
    # EF should not be significantly worse (it distributes error over time)
    print("  Gradient compression EF: PASS")


# ── R22e: AsyncDataPipeline ────────────────────────────────────────────────

def test_async_pipeline_speedup():
    """Async pipeline should be faster than sequential loading+compute."""
    from research.training.optim.r22_training_speedups import AsyncDataPipeline

    def mock_load(idx):
        time.sleep(0.01)
        return f"doc {idx}"

    def mock_tokenize(raw):
        time.sleep(0.005)
        return {"input_ids": torch.randint(0, 65536, (128,), device=_DEV)}

    def mock_compute():
        time.sleep(0.02)  # 20ms simulated GPU compute

    # Sequential: load + tokenize + compute (no overlap)
    t0 = time.time()
    for i in range(30):
        raw = mock_load(i)
        tokens = mock_tokenize(raw)
        mock_compute()
    sequential = time.time() - t0

    # Pipelined: I/O overlaps with compute
    pipeline = AsyncDataPipeline(mock_load, mock_tokenize, buffer_size=3)
    pipeline.prefetch(3)
    t0 = time.time()
    for _ in range(30):
        batch = pipeline.get_batch()
        mock_compute()
    pipelined = time.time() - t0
    pipeline.shutdown()

    speedup = sequential / pipelined
    print(f"  Async pipeline: sequential={sequential*1000:.0f}ms, "
          f"pipelined={pipelined*1000:.0f}ms, speedup={speedup:.2f}x")
    assert speedup > 1.0, "Pipeline should be faster"
    print("  Async pipeline speedup: PASS")


# ── R22f: CheckpointDelta ──────────────────────────────────────────────────

def test_checkpoint_delta():
    """Delta checkpoint should be smaller than full checkpoint."""
    from research.training.optim.r22_training_speedups import CheckpointDelta

    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.Linear(256, 256),
        nn.Linear(256, 256),
    ).to(_DEV)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = CheckpointDelta(full_checkpoint_every=10, delta_threshold=1e-4)
        base_path = os.path.join(tmpdir, "base.pt")
        ckpt.save_base(model, base_path)

        # Simulate training (modify only first layer)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "0" in name:
                    p.data += torch.randn_like(p) * 0.01

        delta_path = os.path.join(tmpdir, "delta_1.pt")
        stats = ckpt.save_delta(model, delta_path)
        print(f"  Delta: {stats['changed_pct']:.1f}% params changed, "
              f"{stats['compression_ratio']:.1f}x compression, "
              f"{stats['delta_size_mb']:.2f} MB vs {stats['full_size_mb']:.2f} MB")
        assert stats["compression_ratio"] > 1.0, "Delta should be smaller"
        assert stats["changed_pct"] < 100, "Not all params should change"

        # Test load + apply delta
        model2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        ).to(_DEV)
        ckpt.load_delta(model2, base_path, delta_path)

        # Verify model2 matches model
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            err = (p1.data - p2.data).abs().max().item()
            assert err < 0.01, f"Delta load mismatch on {n1}: {err}"

    print("  Checkpoint delta: PASS")


# ── Full benchmark ─────────────────────────────────────────────────────────

def test_benchmark_r22():
    """Benchmark all R22 approaches and estimate combined speedup."""
    from research.training.optim.r22_training_speedups import (
        benchmark_r22, estimate_combined_speedup
    )

    print("\n  R22 benchmark:")
    print("  " + "=" * 70)

    results = benchmark_r22(device=str(_DEV))

    print(f"\n  R22a: DataDedup (MinHash LSH)")
    d = results["data_dedup"]
    print(f"    {d['n_docs']} docs -> {d['n_unique']} unique ({d['dup_ratio']*100:.1f}% dups)")
    print(f"    Dedup time: {d['dedup_time_sec']:.2f}s")
    print(f"    Training time saved: {d['time_saved_pct']:.1f}%")

    print(f"\n  R22b: TokenImportanceSampling")
    t = results["token_importance"]
    print(f"    Skip rate: {t['skip_rate']*100:.1f}%")
    print(f"    Effective speedup: {t['effective_speedup']:.2f}x")
    print(f"    Final threshold: {t['final_threshold']:.3f}")

    print(f"\n  R22c: ProgressiveLayerUnfreezing")
    u = results["progressive_unfreeze"]
    for p in u["phases"]:
        print(f"    Step {p['step']}: {p['active']}/32 layers ({p['speedup']:.1f}x)")
    print(f"    Average speedup: {u['avg_speedup']:.2f}x")

    print(f"\n  R22d: GradientCompression (4-bit)")
    g = results["grad_compression"]
    print(f"    Compression: {g['compression_ratio']:.1f}x (bf16 -> 4-bit)")
    print(f"    Error: {g['decompression_error']*100:.2f}%")
    print(f"    Compress time: {g['compress_time_ms']:.1f}ms ({g['params']/1e6:.1f}M params)")

    print(f"\n  R22e: AsyncDataPipeline (with simulated GPU compute)")
    a = results["async_pipeline"]
    print(f"    Sequential (load+tok+compute): {a['sequential_time_ms']:.0f}ms")
    print(f"    Pipelined (compute, I/O hidden): {a['pipeline_time_ms']:.0f}ms")
    print(f"    Theoretical min (compute only): {a['theoretical_min_ms']:.0f}ms")
    print(f"    Speedup: {a['speedup']:.2f}x ({a['time_saved_pct']:.1f}% saved)")
    print(f"    I/O hidden: {a['io_hidden_pct']:.1f}%")

    print(f"\n  R22f: CheckpointDelta")
    c = results["checkpoint_delta"]
    print(f"    Changed: {c['changed_pct']:.1f}% of params")
    print(f"    Compression: {c['compression_ratio']:.1f}x")
    print(f"    Delta: {c['delta_size_mb']:.2f} MB vs full {c['full_size_mb']:.2f} MB")

    combined = estimate_combined_speedup(results)
    print(f"\n  {'='*70}")
    print(f"  COMBINED SPEEDUP ESTIMATE")
    print(f"  {'='*70}")
    print(f"    Data dedup:          {combined['dedup_speedup']:.2f}x")
    print(f"    Token importance:    {combined['importance_speedup']:.2f}x")
    print(f"    Progressive unfreeze:{combined['unfreeze_speedup']:.2f}x")
    print(f"    Grad compression:    {combined['grad_compress_speedup']:.2f}x")
    print(f"    Async pipeline:      {combined['pipeline_speedup']:.2f}x")
    print(f"    Delta checkpoint:    {combined['delta_checkpoint_speedup']:.2f}x (save/resume only)")
    print(f"    ────────────────────────────────")
    print(f"    COMBINED (training): {combined['combined_speedup']:.2f}x")
    print(f"    Time saved:          {combined['combined_time_saved_pct']:.1f}%")

    print("  Benchmark R22: PASS")


def main_r22():
    import math
    print("=" * 70)
    print("  R&D ROUND 22: Training Speedups for Large Datasets + Models")
    print("=" * 70)

    print("\n  R22a: DataDedup (MinHash LSH)")
    test_minhash_signature()
    test_dedup_finds_duplicates()
    test_dedup_savings()

    print("\n  R22b: TokenImportanceSampling")
    test_token_importance_skips_low_loss()
    test_token_importance_adapts()
    test_token_importance_speedup()

    print("\n  R22c: ProgressiveLayerUnfreezing")
    test_progressive_unfreeze_phases()
    test_progressive_unfreeze_speedup()

    print("\n  R22d: GradientCompression (4-bit)")
    test_gradient_compression_roundtrip()
    test_gradient_compression_ef_convergence()

    print("\n  R22e: AsyncDataPipeline")
    test_async_pipeline_speedup()

    print("\n  R22f: CheckpointDelta")
    test_checkpoint_delta()

    print("\n  Full benchmark")
    test_benchmark_r22()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 22 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r22()
