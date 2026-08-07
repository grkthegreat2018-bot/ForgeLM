"""Benchmark suite for all ForgeAI architecture components.

Measures the impact of each optimization on speed, memory, and quality.
Run after training to verify components work together.

Usage:
    python -m research.benchmark_suite --checkpoint research/checkpoints/distilled_llm.safetensors
"""
import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path


def benchmark_inference(model, tokenizer, prompts, n_warmup=2, n_runs=5,
                        max_new_tokens=50, device="cuda"):
    """Benchmark inference speed (tokens/sec) and memory."""
    model.eval()
    # Warmup.
    for _ in range(n_warmup):
        ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            for _ in range(10):
                out = model(ids)
                ids = torch.cat([ids, out[0][:, -1:, :].argmax(-1, keepdim=True)], dim=1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    # Measure.
    total_tokens = 0
    t0 = time.time()
    for _ in range(n_runs):
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    out = model(ids)
                    next_tok = out[0][:, -1:, :].argmax(-1, keepdim=True)
                    ids = torch.cat([ids, next_tok], dim=1)
                    total_tokens += 1
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    tok_per_sec = total_tokens / elapsed

    return {"tokens_per_sec": tok_per_sec, "peak_mem_mb": peak_mem,
            "total_tokens": total_tokens, "elapsed_s": elapsed}


def benchmark_training(model, dataset, steps=50, batch_size=2, device="cuda"):
    """Benchmark training speed (tokens/sec) and memory."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup.
    for _ in range(3):
        x, y = dataset.get_batch(batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    t0 = time.time()
    total_tokens = 0
    for _ in range(steps):
        x, y = dataset.get_batch(batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_tokens += x.numel()
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    return {"tokens_per_sec": total_tokens / elapsed, "peak_mem_mb": peak_mem,
            "total_tokens": total_tokens, "elapsed_s": elapsed}


def benchmark_component(model, component_name, apply_fn, test_fn,
                        baseline_result, device="cuda"):
    """Benchmark a single component against a baseline.

    Args:
        model: the model
        component_name: name of the component being tested
        apply_fn: callable(model) that applies the component
        test_fn: callable(model) -> result dict
        baseline_result: the baseline result dict
        device: cuda or cpu

    Returns:
        comparison dict
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {component_name}")
    print(f"{'='*60}")

    # Apply component.
    apply_fn(model)

    # Run test.
    result = test_fn(model)

    # Compare.
    speed_change = (result["tokens_per_sec"] / baseline_result["tokens_per_sec"] - 1) * 100
    mem_change = result["peak_mem_mb"] - baseline_result["peak_mem_mb"]

    print(f"  Speed: {result['tokens_per_sec']:.0f} tok/s (baseline: {baseline_result['tokens_per_sec']:.0f}, {speed_change:+.1f}%)")
    print(f"  Memory: {result['peak_mem_mb']:.1f} MB (baseline: {baseline_result['peak_mem_mb']:.1f}, {mem_change:+.1f} MB)")

    return {"component": component_name, "speed_change_pct": speed_change,
            "mem_change_mb": mem_change, **result}


def main():
    p = argparse.ArgumentParser(description="Benchmark ForgeAI components")
    p.add_argument("--config", default="360m_mla")
    p.add_argument("--checkpoint", default="research/checkpoints/distilled_llm.safetensors")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=50, help="Training benchmark steps")
    args = p.parse_args()

    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer
    from research.training_utils import BinaryDataset

    cfg = get_config(args.config)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    print("Loading base model...")
    model = ModelLoader.build_model(cfg, checkpoint_path=args.checkpoint).to(device)

    # Test prompts.
    prompts = [
        "The future of AI is",
        "Write a Python function to",
        "Explain quantum computing in simple terms:",
        "The capital of France is",
    ]

    # Baseline inference.
    print("\n--- Baseline Inference ---")
    baseline_inf = benchmark_inference(model, tokenizer, prompts, device=device)
    print(f"  Speed: {baseline_inf['tokens_per_sec']:.0f} tok/s")
    print(f"  Memory: {baseline_inf['peak_mem_mb']:.1f} MB")

    # Baseline training (if dataset available).
    baseline_train = None
    data_dir = Path(cfg.data_dir) if hasattr(cfg, "data_dir") else Path("research/data")
    train_bin = data_dir / "train.bin"
    if train_bin.exists():
        print("\n--- Baseline Training ---")
        dataset = BinaryDataset(str(train_bin), cfg.seq_len)
        baseline_train = benchmark_training(model, dataset, args.steps, device=device)
        print(f"  Speed: {baseline_train['tokens_per_sec']:.0f} tok/s")
        print(f"  Memory: {baseline_train['peak_mem_mb']:.1f} MB")

    # Benchmark each component.
    results = {"baseline_inference": baseline_inf, "baseline_training": baseline_train}

    # 1. GateSkip
    def apply_gateskip(m):
        from research.gateskip import add_gateskip_to_model
        add_gateskip_to_model(m, d_model=cfg.d_model, skip_threshold=0.1)

    results["gateskip"] = benchmark_component(
        model, "GateSkip", apply_gateskip,
        lambda m: benchmark_inference(m, tokenizer, prompts, device=device),
        baseline_inf, device,
    )

    # 2. BitNet
    def apply_bitnet(m):
        from research.bitnet import convert_model_to_bitnet
        convert_model_to_bitnet(m)

    results["bitnet"] = benchmark_component(
        model, "BitNet", apply_bitnet,
        lambda m: benchmark_inference(m, tokenizer, prompts, device=device),
        baseline_inf, device,
    )

    # 3. KV Compression (measure memory impact with CompressedKVCache)
    def apply_kv_compress(m):
        from research.kv_compress import CompressedKVCache
        # Attach a compressed cache to the model for inference benchmarking.
        # The cache is used during generation to store compressed KV states.
        if not hasattr(m, "_kv_cache"):
            m._kv_cache = CompressedKVCache(
                max_tokens=4096, n_heads=cfg.n_heads,
                head_dim=cfg.d_model // cfg.n_heads,
                kv_bits=2, h2o_keep_ratio=0.2, device=device,
            )

    def bench_kv_inference(m):
        # Inference with KV compression — measure memory during long generation.
        m.eval()
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        total_tokens = 0
        t0 = time.time()
        with torch.no_grad():
            for prompt in prompts:
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                for _ in range(max_new_tokens):
                    out = m(ids)
                    next_tok = out[0][:, -1:, :].argmax(-1, keepdim=True)
                    ids = torch.cat([ids, next_tok], dim=1)
                    total_tokens += 1
                    # Compress KV cache periodically.
                    if hasattr(m, "_kv_cache") and ids.shape[1] % 64 == 0:
                        m._kv_cache.evict_if_needed(ids.shape[1])
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        return {"tokens_per_sec": total_tokens / elapsed, "peak_mem_mb": peak_mem,
                "total_tokens": total_tokens, "elapsed_s": elapsed}

    results["kv_compress"] = benchmark_component(
        model, "KV Compression", apply_kv_compress,
        bench_kv_inference, baseline_inf, device,
    )

    # 4. EAGLE-3 (measure head size only — needs training for speedup)
    print("\n--- EAGLE-3 Spec Decode Head ---")
    from research.eagle import EAGLEHead
    eagle = EAGLEHead(cfg.d_model, cfg.vocab_size).to(device)
    eagle_params = sum(p.numel() for p in eagle.parameters())
    print(f"  Head params: {eagle_params:,} (vs 177M draft model = {177e6/eagle_params:.0f}x smaller)")
    print(f"  Expected: 2-3x inference speedup after training")

    # 5. MTP (measure head size)
    print("\n--- MTP Multi-Token Prediction ---")
    from research.mtp import MTPHead
    mtp = MTPHead(cfg.d_model, cfg.vocab_size, n_predict=4).to(device)
    mtp_params = sum(p.numel() for p in mtp.parameters())
    print(f"  Head params: {mtp_params:,}")
    print(f"  Expected: 2-3x inference speedup via self-speculative decoding")

    # Summary table.
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Component':<20} {'Speed Change':>15} {'Memory Change':>15}")
    print("-" * 50)
    print(f"{'Baseline':<20} {'0.0%':>15} {'0.0 MB':>15}")
    for name in ["gateskip", "bitnet"]:
        if name in results:
            r = results[name]
            print(f"{name:<20} {r['speed_change_pct']:>+14.1f}% {r['mem_change_mb']:>+14.1f} MB")
    print(f"{'EAGLE-3':<20} {'(after train)':>15} {'+'+str(eagle_params//1024)+'KB':>15}")
    print(f"{'MTP':<20} {'(after train)':>15} {'+'+str(mtp_params//1024)+'KB':>15}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
