"""Benchmark: FreeToken-inspired training enhancements vs baseline.

Compares training throughput (samples/s, tok/s) across:
1. Baseline: CPUAdamW (sync, no overlap) — current production
2. Overlap: CPUAdamW with overlap=True (CPU step overlaps with next forward)
3. FreeToken: CPUAdamW with double_buffer + bandwidth_adaptive + overlap
4. FreeToken+chunked: Same as 3 but with explicit chunk_size_mb

Measures: forward ms, backward ms, optimizer ms, total ms, tok/s, VRAM.
Runs N steps on the ForgeLM V4 1.2B model with synthetic data.

Usage:
  python research/sandbox/bench_freetoken_training.py [--steps 50]
"""
import os, sys, time, argparse, json
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn

from research.config import get_config
from research.model_loader import ModelLoader
from research.training.optim.hybrid_offload import CPUAdamW


def build_model(config_name="lfm25_tiny", device="cuda", n_params_m=1300):
    """Build a model with ~n_params_m parameters for benchmarking.

    Uses a simple transformer-like architecture with enough parameters to
    stress the CPUAdamW optimizer (the thing we're benchmarking).
    For 1300M params: d=2048, 4 layers, intermediate=8192.
    """
    dev = torch.device(device)
    d = 2048
    n_layers = max(1, n_params_m // 50)  # ~50M per layer
    intermediate = 8192
    vocab = 65536

    class BenchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(d, intermediate), nn.GELU(),
                              nn.Linear(intermediate, d))
                for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d)
            self.lm_head = nn.Linear(d, vocab, bias=False)
            self.lm_head.weight = self.embed.weight  # tie

        def forward(self, idx):
            x = self.embed(idx)
            for layer in self.layers:
                x = x + layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    model = BenchModel().to(dev).to(torch.bfloat16)
    model.train()

    actual_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Built BenchModel: {n_layers} layers, {actual_params:.0f}M params")
    return model, None


def run_benchmark(model, optimizer, steps, seq_len, batch_size, device, label, cfg=None):
    """Run training benchmark and return timing stats."""
    dev = torch.device(device)
    vocab_size = 65536

    # Synthetic data
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
    labels = input_ids.clone()

    # Warmup (3 steps)
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
                labels[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        if hasattr(optimizer, 'wait'):
            optimizer.wait()
        del out, logits, loss
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed run
    fwd_times = []
    bwd_times = []
    opt_times = []
    wait_times = []

    for step in range(steps):
        optimizer.zero_grad()

        # Wait for previous overlap step (3-stage pipeline)
        t0 = time.perf_counter()
        if hasattr(optimizer, 'wait') and optimizer.overlap:
            optimizer.wait()
        torch.cuda.synchronize()
        wait_ms = (time.perf_counter() - t0) * 1000

        # Forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
                labels[:, 1:].contiguous().view(-1))
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000

        # Backward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

        del out, logits, loss

        # Free fragmentation between steps (12GB VRAM is tight)
        torch.cuda.empty_cache()

        # Optimizer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000

        fwd_times.append(fwd_ms)
        bwd_times.append(bwd_ms)
        opt_times.append(opt_ms)
        wait_times.append(wait_ms)

        if step < 3 or step % 10 == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            total = fwd_ms + bwd_ms + opt_ms + wait_ms
            print(f"  [{label}] step {step}: fwd={fwd_ms:.0f}ms bwd={bwd_ms:.0f}ms "
                  f"opt={opt_ms:.0f}ms wait={wait_ms:.0f}ms total={total:.0f}ms "
                  f"vram={vram:.2f}GB")

    # Stats
    import statistics
    avg_fwd = statistics.mean(fwd_times[2:])  # skip first 2 (warmup)
    avg_bwd = statistics.mean(bwd_times[2:])
    avg_opt = statistics.mean(opt_times[2:])
    avg_wait = statistics.mean(wait_times[2:])
    avg_total = avg_fwd + avg_bwd + avg_opt + avg_wait
    tok_s = (seq_len * batch_size) / (avg_total / 1000)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "label": label,
        "avg_fwd_ms": avg_fwd,
        "avg_bwd_ms": avg_bwd,
        "avg_opt_ms": avg_opt,
        "avg_wait_ms": avg_wait,
        "avg_total_ms": avg_total,
        "tok_s": tok_s,
        "samples_s": batch_size / (avg_total / 1000),
        "peak_vram_gb": peak_vram,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--config", default="lfm25_tiny")
    parser.add_argument("--params-m", type=int, default=100,
                        help="Target model size in millions of params")
    args = parser.parse_args()

    print(f"FreeToken Training Benchmark")
    print(f"  Model: {args.config}, seq_len={args.seq_len}, batch={args.batch_size}")
    print(f"  Steps: {args.args if hasattr(args, 'args') else args.steps}")
    print()

    device = "cuda"
    results = []

    # Test configurations
    configs = [
        ("baseline", {"overlap": False, "double_buffer": False, "bandwidth_adaptive": False}),
        ("overlap", {"overlap": True, "double_buffer": False, "bandwidth_adaptive": False}),
        ("freetoken", {"overlap": True, "double_buffer": True, "bandwidth_adaptive": True}),
        ("freetoken_chunked", {"overlap": True, "double_buffer": True, "bandwidth_adaptive": True, "chunk_size_mb": 64}),
    ]

    for label, opt_kwargs in configs:
        print(f"\n{'='*60}")
        print(f"  Testing: {label}")
        print(f"{'='*60}")

        model, cfg = build_model(args.config, device, n_params_m=args.params_m)
        matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
        other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
        param_groups = [
            {"params": matrix_params, "weight_decay": 0.01},
            {"params": other_params, "weight_decay": 0.0},
        ]
        optimizer = CPUAdamW(param_groups, lr=5e-5, verbose=True, **opt_kwargs)

        result = run_benchmark(model, optimizer, args.steps, args.seq_len,
                               args.batch_size, device, label, cfg=cfg)
        results.append(result)

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Label':<22} {'Fwd':>7} {'Bwd':>7} {'Opt':>7} {'Wait':>7} {'Total':>7} "
          f"{'tok/s':>8} {'VRAM':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<22} {r['avg_fwd_ms']:>6.0f} {r['avg_bwd_ms']:>6.0f} "
              f"{r['avg_opt_ms']:>6.0f} {r['avg_wait_ms']:>6.0f} {r['avg_total_ms']:>6.0f}ms "
              f"{r['tok_s']:>7.0f} {r['peak_vram_gb']:>6.2f}G")

    # Speedup vs baseline
    baseline_tps = results[0]['tok_s']
    print(f"\n  Speedup vs baseline:")
    for r in results[1:]:
        speedup = r['tok_s'] / baseline_tps
        print(f"    {r['label']:<22} {speedup:.2f}x ({r['tok_s']:.0f} vs {baseline_tps:.0f} tok/s)")

    # Save results
    out_path = "research/results/bench_freetoken_training.json"
    os.makedirs("research/results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
