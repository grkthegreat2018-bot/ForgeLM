"""Comprehensive ForgeAI benchmark — finds bottlenecks across all subsystems.

Tests:
  1. Model load time (cold + warm cache)
  2. Inference speed (tokens/sec) across decoding strategies
  3. KV cache strategies (memory + speed)
  4. Quantization impact (INT4, INT8, FP8)
  5. EAGLE-3 draft head forward pass speed
  6. Training throughput (tokens/sec, samples/sec)
  7. Tool call parsing speed
  8. Context manager compression speed
  9. VRAM usage per configuration
 10. torch.compile impact

Usage:
  $env:PYTHONPATH="D:\windsurf\ForgeAI"
  D:\windsurf\ForgeAI\venv\Scripts\python.exe D:\windsurf\ForgeAI\.devin\bench_all.py
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Set PYTHONPATH
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn

# ─── Helpers ──────────────────────────────────────────────────────────────────

def vram_mb():
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e6


def vram_free_mb():
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return free / 1e6


def fmt_mb(mb):
    if mb > 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.0f} MB"


def time_fn(fn, n_warmup=1, n_runs=3, **kwargs):
    """Time a function, return avg ms and result."""
    for _ in range(n_warmup):
        result = fn(**kwargs)
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    return avg, result, times


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─── Benchmarks ───────────────────────────────────────────────────────────────

CHECKPOINT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_BSP.safetensors"
TOKENIZER_PATH = r"D:\windsurf\ForgeAI\research\checkpoints\lfm25_tokenizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = {}


def bench_model_load():
    """Benchmark model loading time."""
    print("\n" + "="*70)
    print("1. MODEL LOAD TIME")
    print("="*70)
    from research.model_loader import ModelLoader
    from research.config import get_config

    config = get_config("lfm25_1.2b")

    # Cold load (clear cache first)
    clear_gpu()
    t0 = time.perf_counter()
    model = ModelLoader.build_model_fast(config, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16)
    model = model.to(DEVICE)
    load_time = time.perf_counter() - t0
    vram_after = vram_mb()

    print(f"  Load time: {load_time:.2f}s")
    print(f"  VRAM after load: {fmt_mb(vram_after)}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    RESULTS["model_load"] = {
        "time_s": load_time,
        "vram_mb": vram_after,
        "params": sum(p.numel() for p in model.parameters()),
    }
    return model


def bench_inference_strategies(model, tokenizer):
    """Benchmark inference speed across decoding strategies."""
    print("\n" + "="*70)
    print("2. INFERENCE SPEED (tokens/sec)")
    print("="*70)
    from research.inference.forge_engine import ForgeEngine

    prompt = "You are a helpful assistant. Explain how binary search works in Python."
    max_new_tokens = 50

    strategies = ["standard"]
    # Add more if available
    try:
        from research.inference.decoding import build_decoding
        # Test if MTP self-spec is available
        if hasattr(model, "mtp_head"):
            strategies.append("mtp_selfspec")
    except Exception:
        pass

    results = {}
    for strat in strategies:
        print(f"\n  --- {strat} ---")
        clear_gpu()

        engine = ForgeEngine(model=model, tokenizer=tokenizer, device=DEVICE)
        try:
            engine.activate(
                kv_cache="standard",
                decoding=strat,
                warmup=True,
                use_prefix_cache=False,
            )
        except Exception as e:
            print(f"    SKIP: {e}")
            results[strat] = {"error": str(e)}
            continue

        vram_before = vram_mb()

        # Warmup
        try:
            engine.generate(prompt, max_new_tokens=10)
        except Exception as e:
            print(f"    Warmup failed: {e}")
            results[strat] = {"error": str(e)}
            continue

        # Benchmark
        times = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = engine.generate(prompt, max_new_tokens=max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        tps = max_new_tokens / avg_time
        vram_after = vram_mb()

        print(f"    {tps:.1f} tok/s | {avg_time*1000:.0f}ms for {max_new_tokens} tokens")
        print(f"    VRAM: {fmt_mb(vram_after)} (delta: {fmt_mb(vram_after - vram_before)})")

        results[strat] = {
            "tokens_per_sec": tps,
            "latency_ms": avg_time * 1000,
            "vram_mb": vram_after,
            "vram_delta_mb": vram_after - vram_before,
        }

    RESULTS["inference_strategies"] = results


def bench_kv_cache(model, tokenizer):
    """Benchmark KV cache strategies."""
    print("\n" + "="*70)
    print("3. KV CACHE STRATEGIES")
    print("="*70)
    from research.inference.forge_engine import ForgeEngine

    prompt = "Write a Python function that sorts a list using quicksort."
    max_new_tokens = 50

    strategies = ["standard", "paged", "streaming", "snapkv"]
    results = {}

    for strat in strategies:
        print(f"\n  --- {strat} ---")
        clear_gpu()
        engine = ForgeEngine(model=model, tokenizer=tokenizer, device=DEVICE)
        try:
            engine.activate(kv_cache=strat, decoding="standard", warmup=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            results[strat] = {"error": str(e)}
            continue

        vram_before = vram_mb()
        try:
            engine.generate(prompt, max_new_tokens=10)  # warmup
        except Exception as e:
            print(f"    Warmup failed: {e}")
            results[strat] = {"error": str(e)}
            continue

        times = []
        for _ in range(3):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            engine.generate(prompt, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        tps = max_new_tokens / avg
        vram_after = vram_mb()
        print(f"    {tps:.1f} tok/s | {avg*1000:.0f}ms | VRAM: {fmt_mb(vram_after)}")
        results[strat] = {"tokens_per_sec": tps, "latency_ms": avg*1000, "vram_mb": vram_after}

    RESULTS["kv_cache"] = results


def bench_quantization(tokenizer):
    """Benchmark quantization impact. Reloads fresh model each test."""
    print("\n" + "="*70)
    print("4. QUANTIZATION IMPACT")
    print("="*70)
    from research.inference.forge_engine import ForgeEngine
    from research.model_loader import ModelLoader
    from research.config import get_config

    prompt = "Explain the difference between TCP and UDP."
    max_new_tokens = 50

    configs = [
        ("bf16 (baseline)", None),
        ("int8", "int8"),
        ("int4", "int4"),
    ]
    results = {}

    for name, quant in configs:
        print(f"\n  --- {name} ---")
        # Reload fresh model each test (quantization is in-place)
        clear_gpu()
        ModelLoader.clear_cache()
        config = get_config("lfm25_1.2b")
        fresh_model = ModelLoader.build_model_fast(config, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16)
        fresh_model = fresh_model.to(DEVICE)

        engine = ForgeEngine(model=fresh_model, tokenizer=tokenizer, device=DEVICE)
        try:
            engine.activate(kv_cache="standard", decoding="standard", quantize=quant, warmup=True)
        except Exception as e:
            print(f"    SKIP: {e}")
            results[name] = {"error": str(e)}
            continue

        vram_before = vram_mb()
        try:
            engine.generate(prompt, max_new_tokens=10)
        except Exception as e:
            print(f"    Warmup failed: {e}")
            results[name] = {"error": str(e)}
            continue

        times = []
        for _ in range(3):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            engine.generate(prompt, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        tps = max_new_tokens / avg
        vram_after = vram_mb()
        print(f"    {tps:.1f} tok/s | {avg*1000:.0f}ms | VRAM: {fmt_mb(vram_after)}")
        results[name] = {"tokens_per_sec": tps, "latency_ms": avg*1000, "vram_mb": vram_after}

    RESULTS["quantization"] = results


def bench_eagle3(model, tokenizer):
    """Benchmark EAGLE-3 draft head forward pass speed."""
    print("\n" + "="*70)
    print("5. EAGLE-3 DRAFT HEAD")
    print("="*70)
    from research.decoding.eagle import Eagle3Head, add_eagle3_to_model, extract_hidden_states

    results = {}

    # Create head
    try:
        head = add_eagle3_to_model(model)
        head = head.to(DEVICE).to(torch.bfloat16)
        n_params = sum(p.numel() for p in head.parameters())
        unique_params = n_params - sum(p.numel() for n, p in head.named_parameters() if 'embed' in n or 'lm_head' in n)
        print(f"  Head params: {n_params:,} (unique: {unique_params:,})")
        print(f"  Head size: {unique_params * 2 / 1024**3:.3f} GB (bf16)")

        input_ids = tokenizer("Hello world, this is a test.", return_tensors="pt").input_ids.to(DEVICE)
        extract_layers = [head.low_layer, head.mid_layer, head.high_layer]

        # Benchmark hidden state extraction (no_grad, not inference_mode for compat)
        def do_extract():
            with torch.no_grad():
                return extract_hidden_states(model, input_ids, extract_layers)

        avg_extract, _, _ = time_fn(do_extract, n_warmup=2, n_runs=5)
        print(f"  Hidden state extraction (3 layers): {avg_extract:.1f}ms")

        # Benchmark fuse
        with torch.no_grad():
            hidden_list, final_hidden, _ = extract_hidden_states(model, input_ids, extract_layers)

        avg_fuse, _, _ = time_fn(
            head.fuse_hidden_states, n_warmup=2, n_runs=5,
            hidden_states_list=hidden_list,
        )
        print(f"  Feature fusion (3*d -> d): {avg_fuse:.2f}ms")

        fused = head.fuse_hidden_states(hidden_list)
        draft_input_ids = input_ids[:, :-1]

        def do_draft():
            with torch.no_grad():
                return head.draft_forward(fused[:, :-1], draft_input_ids)

        avg_draft, _, _ = time_fn(do_draft, n_warmup=2, n_runs=5)
        print(f"  Draft forward pass: {avg_draft:.1f}ms")

        # Compare to target model forward pass
        def do_target():
            with torch.no_grad():
                return model(input_ids)

        avg_target, _, _ = time_fn(do_target, n_warmup=2, n_runs=5)
        print(f"  Target forward pass: {avg_target:.1f}ms")
        print(f"  Draft/Target ratio: {avg_draft/avg_target:.2f}x (should be <0.3x for speedup)")

        vram_head = vram_mb()
        print(f"  VRAM with head: {fmt_mb(vram_head)}")

        results = {
            "head_unique_params": unique_params,
            "head_size_gb": unique_params * 2 / 1024**3,
            "extract_ms": avg_extract,
            "fuse_ms": avg_fuse,
            "draft_forward_ms": avg_draft,
            "target_forward_ms": avg_target,
            "draft_target_ratio": avg_draft / avg_target,
            "vram_mb": vram_head,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results = {"error": str(e)}

    RESULTS["eagle3"] = results


def bench_training(tokenizer):
    """Benchmark training throughput. Reloads fresh model (quantization corrupts it)."""
    print("\n" + "="*70)
    print("6. TRAINING THROUGHPUT")
    print("="*70)
    from research.model_loader import ModelLoader
    from research.config import get_config

    # Reload fresh model (previous benchmarks may have quantized it)
    clear_gpu()
    # Clear architecture cache to avoid getting quantized clone
    ModelLoader.clear_cache()
    config = get_config("lfm25_1.2b")
    model = ModelLoader.build_model_fast(config, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16)
    model = model.to(DEVICE)

    results = {}

    # Create dummy training data
    batch_size = 1
    seq_len = 512
    input_ids = torch.randint(0, 65536, (batch_size, seq_len), device=DEVICE)
    targets = torch.randint(0, 65536, (batch_size, seq_len), device=DEVICE)

    # Standard forward + backward
    model.train()
    model.gradient_checkpointing_enable() if hasattr(model, 'gradient_checkpointing_enable') else None

    # Warmup
    model.zero_grad()
    out = model(input_ids, targets=targets)
    loss = out[0] if isinstance(out, tuple) else out
    loss.mean().backward() if loss.dim() > 0 else loss.backward()
    model.zero_grad()

    # Benchmark forward only
    avg_fwd, _, _ = time_fn(
        model, n_warmup=2, n_runs=5,
        idx=input_ids, targets=targets,
    )
    print(f"  Forward pass (B={batch_size}, T={seq_len}): {avg_fwd:.1f}ms")
    print(f"  Throughput: {batch_size * seq_len / avg_fwd * 1000:.0f} tokens/s (forward only)")

    # Benchmark forward + backward
    def fwd_bwd():
        model.zero_grad()
        out = model(input_ids, targets=targets)
        loss = out[0] if isinstance(out, tuple) else out
        loss = loss.mean() if loss.dim() > 0 else loss
        loss.backward()
        return loss.item()

    avg_fwb, _, _ = time_fn(fwd_bwd, n_warmup=2, n_runs=5)
    print(f"  Forward+backward: {avg_fwb:.1f}ms")
    print(f"  Throughput: {batch_size * seq_len / avg_fwb * 1000:.0f} tokens/s (fwd+bwd)")

    vram_after = vram_mb()
    print(f"  VRAM during training: {fmt_mb(vram_after)}")

    # Test with different batch sizes
    for bs in [1, 2, 4]:
        try:
            ids = torch.randint(0, 65536, (bs, seq_len), device=DEVICE)
            tgts = torch.randint(0, 65536, (bs, seq_len), device=DEVICE)
            model.zero_grad()

            def fwd_bwd_bs():
                model.zero_grad()
                out = model(ids, targets=tgts)
                loss = out[0] if isinstance(out, tuple) else out
                loss = loss.mean() if loss.dim() > 0 else loss
                loss.backward()
                return loss.item()

            avg, _, _ = time_fn(fwd_bwd_bs, n_warmup=1, n_runs=3)
            vram_bs = vram_mb()
            tps = bs * seq_len / avg * 1000
            print(f"  B={bs}: {avg:.0f}ms | {tps:.0f} tok/s | VRAM: {fmt_mb(vram_bs)}")
            results[f"batch_{bs}"] = {"time_ms": avg, "tokens_per_sec": tps, "vram_mb": vram_bs}
        except torch.cuda.OutOfMemoryError:
            print(f"  B={bs}: OOM")
            results[f"batch_{bs}"] = {"error": "OOM"}
            clear_gpu()
            break

    model.eval()
    results["forward_ms"] = avg_fwd
    results["fwd_bwd_ms"] = avg_fwb
    results["forward_tps"] = batch_size * seq_len / avg_fwd * 1000
    results["fwd_bwd_tps"] = batch_size * seq_len / avg_fwb * 1000
    results["vram_mb"] = vram_after

    RESULTS["training"] = results


def bench_tool_parsing(model, tokenizer):
    """Benchmark tool call parsing speed."""
    print("\n" + "="*70)
    print("7. TOOL CALL PARSING")
    print("="*70)
    from research.self_play.discovery.qwen_adapter import qwen_parse_tool_calls

    # Generate test texts with tool calls
    start_marker = bytes.fromhex("3c7c746f6f6c5f63616c6c5f73746172747c3e").decode("ascii")
    end_marker = bytes.fromhex("3c7c746f6f6c5f63616c6c5f656e647c3e").decode("ascii")

    test_texts = [
        f"Let me search for that.\n{start_marker}\n{{\"name\": \"web_search\", \"arguments\": {{\"query\": \"python asyncio\"}}}}\n{end_marker}",
        f"I'll run a script.\n{start_marker}\n{{\"name\": \"run_script\", \"arguments\": {{\"code\": \"print('hello')\"}}}}\n{end_marker}",
        "Just a normal response without any tool calls.",
        f"{start_marker}\n{{\"name\": \"think\", \"arguments\": {{\"content\": \"test\", \"confidence\": 0.8}}}}\n{end_marker}",
    ]

    # Warmup
    for text in test_texts:
        qwen_parse_tool_calls(text)

    # Benchmark
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        for text in test_texts:
            qwen_parse_tool_calls(text)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    per_call = avg / len(test_texts) * 1000
    print(f"  Parse {len(test_texts)} calls x100 iterations: {avg*1000:.1f}ms total")
    print(f"  Per call: {per_call:.3f}ms")
    print(f"  Calls/sec: {len(test_texts) / avg:.0f}")

    RESULTS["tool_parsing"] = {
        "per_call_ms": per_call,
        "calls_per_sec": len(test_texts) / avg,
    }


def bench_context_manager(model, tokenizer):
    """Benchmark context manager compression."""
    print("\n" + "="*70)
    print("8. CONTEXT MANAGER COMPRESSION")
    print("="*70)
    from research.self_play.discovery.context_manager import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(max_seq_len=32768, reserved_for_generation=4096)
    ctx = ContextManager(config, tokenizer=tokenizer)

    # Build a long conversation
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(30):
        messages.append({"role": "user", "content": f"Question {i}: What is {i}*{i}?"})
        messages.append({"role": "assistant", "content": f"The answer is {i*i}."})

    # Warmup
    ctx.maybe_compress(messages)

    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        compressed, was_compressed = ctx.maybe_compress(messages)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    from research.self_play.discovery.context_manager import count_tokens
    orig_tokens = count_tokens(str(messages), tokenizer)
    comp_tokens = count_tokens(str(compressed), tokenizer) if compressed != messages else orig_tokens

    print(f"  Original messages: {len(messages)}")
    print(f"  Original tokens: {orig_tokens}")
    print(f"  Compressed tokens: {comp_tokens}")
    print(f"  Compression time: {avg*1000:.1f}ms")
    print(f"  Was compressed: {was_compressed}")

    RESULTS["context_manager"] = {
        "time_ms": avg * 1000,
        "orig_tokens": orig_tokens,
        "comp_tokens": comp_tokens,
        "was_compressed": was_compressed,
    }


def bench_vram_breakdown(model, tokenizer):
    """Detailed VRAM breakdown."""
    print("\n" + "="*70)
    print("9. VRAM BREAKDOWN")
    print("="*70)

    clear_gpu()
    baseline = vram_mb()
    print(f"  Baseline (CUDA context): {fmt_mb(baseline)}")

    # Model on GPU
    model = model.to(DEVICE)
    vram_model = vram_mb()
    model_mb = vram_model - baseline
    print(f"  Model weights: {fmt_mb(model_mb)}")

    # KV cache
    from research.inference.kv_backend import build_kv_cache
    n_layers = 16
    n_kv_heads = 8
    head_dim = 64
    max_seq = 32768
    kv = build_kv_cache("standard")
    kv.init(n_kv_heads, head_dim, n_layers, max_seq, DEVICE, torch.bfloat16)
    vram_kv = vram_mb()
    kv_mb = vram_kv - vram_model
    print(f"  KV cache (32K tokens): {fmt_mb(kv_mb)}")

    # Training memory (gradients + optimizer)
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    # Simulate gradient allocation
    input_ids = torch.randint(0, 65536, (1, 512), device=DEVICE)
    targets = torch.randint(0, 65536, (1, 512), device=DEVICE)
    out = model(input_ids, targets=targets)
    loss = out[0] if isinstance(out, tuple) else out
    loss = loss.mean() if loss.dim() > 0 else loss
    loss.backward()
    vram_grads = vram_mb()
    grad_mb = vram_grads - vram_kv
    print(f"  Gradients (B=1, T=512): {fmt_mb(grad_mb)}")

    # Optimizer state (AdamW = 2x params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.step()
    vram_opt = vram_mb()
    opt_mb = vram_opt - vram_grads
    print(f"  Optimizer state (AdamW): {fmt_mb(opt_mb)}")

    total = vram_opt
    free = vram_free_mb()
    print(f"  Total used: {fmt_mb(total)}")
    print(f"  Free: {fmt_mb(free)}")

    model.eval()

    RESULTS["vram_breakdown"] = {
        "baseline_mb": baseline,
        "model_mb": model_mb,
        "kv_cache_mb": kv_mb,
        "gradients_mb": grad_mb,
        "optimizer_mb": opt_mb,
        "total_mb": total,
        "free_mb": free,
    }


def bench_forward_layers(model, tokenizer):
    """Per-layer forward pass timing to find slow layers."""
    print("\n" + "="*70)
    print("10. PER-LAYER FORWARD TIMING")
    print("="*70)

    input_ids = torch.randint(0, 65536, (1, 256), device=DEVICE)
    model.eval()

    # Get embedding
    with torch.inference_mode():
        x = model.embed(input_ids)
        position_ids = torch.arange(256, device=DEVICE).unsqueeze(0)

        layer_times = []
        for i, block in enumerate(model.blocks):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Run block
            x, _ = block(x, past_key_value=None, use_cache=False, layer_idx=i,
                        attention_bias=None, position_ids=position_ids)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            layer_times.append(dt)

            layer_type = "attn" if i in [2, 5, 8, 10, 12, 14] else "conv"
            print(f"  Layer {i:2d} ({layer_type}): {dt:.2f}ms")

        # Final ln + head
        t0 = time.perf_counter()
        x = model.ln_f(x)
        logits = model.head(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"  ln_f + head: {dt:.2f}ms")

    attn_times = [layer_times[i] for i in [2, 5, 8, 10, 12, 14]]
    conv_times = [layer_times[i] for i in range(16) if i not in [2, 5, 8, 10, 12, 14]]
    total_attn = sum(attn_times)
    total_conv = sum(conv_times)
    total = total_attn + total_conv + dt

    print(f"\n  Total: {total:.1f}ms")
    print(f"  Attention layers (6): {total_attn:.1f}ms ({total_attn/total*100:.0f}%)")
    print(f"  Conv layers (10): {total_conv:.1f}ms ({total_conv/total*100:.0f}%)")
    print(f"  ln_f + head: {dt:.1f}ms ({dt/total*100:.0f}%)")

    RESULTS["per_layer"] = {
        "layer_times_ms": layer_times,
        "total_ms": total,
        "attn_total_ms": total_attn,
        "conv_total_ms": total_conv,
        "head_ms": dt,
        "attn_pct": total_attn / total * 100,
        "conv_pct": total_conv / total * 100,
    }


def bench_compile(model, tokenizer):
    """Benchmark torch.compile impact."""
    print("\n" + "="*70)
    print("11. TORCH.COMPILE IMPACT")
    print("="*70)

    input_ids = torch.randint(0, 65536, (1, 128), device=DEVICE)
    model.eval()

    # Without compile
    def fwd():
        with torch.inference_mode():
            return model(input_ids)

    avg_no_compile, _, _ = time_fn(fwd, n_warmup=3, n_runs=5)
    print(f"  Without compile: {avg_no_compile:.1f}ms")

    # With compile (mode='reduce-overhead' for inference)
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        def fwd_compiled():
            with torch.inference_mode():
                return compiled(input_ids)

        # Extra warmup for compile
        avg_compiled, _, _ = time_fn(fwd_compiled, n_warmup=5, n_runs=5)
        print(f"  With compile (reduce-overhead): {avg_compiled:.1f}ms")
        print(f"  Speedup: {avg_no_compile / avg_compiled:.2f}x")

        RESULTS["torch_compile"] = {
            "without_ms": avg_no_compile,
            "with_ms": avg_compiled,
            "speedup": avg_no_compile / avg_compiled,
        }
    except Exception as e:
        print(f"  Compile failed: {e}")
        RESULTS["torch_compile"] = {"error": str(e), "without_ms": avg_no_compile}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("ForgeAI Comprehensive Benchmark")
    print(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM: {fmt_mb((total-free)/1e6)} used / {fmt_mb(total/1e6)} total")
    print("="*70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"Tokenizer loaded: vocab={tokenizer.vocab_size}")

    # Load model
    model = bench_model_load()

    # Run all benchmarks
    bench_forward_layers(model, tokenizer)
    bench_inference_strategies(model, tokenizer)
    bench_kv_cache(model, tokenizer)
    bench_quantization(tokenizer)

    # Reload fresh model for EAGLE-3 (quantization may have corrupted it)
    clear_gpu()
    from research.model_loader import ModelLoader
    from research.config import get_config
    ModelLoader.clear_cache()
    config = get_config("lfm25_1.2b")
    model = ModelLoader.build_model_fast(config, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16)
    model = model.to(DEVICE)

    bench_eagle3(model, tokenizer)
    bench_training(tokenizer)
    bench_tool_parsing(model, tokenizer)
    bench_context_manager(model, tokenizer)
    bench_vram_breakdown(model, tokenizer)
    bench_compile(model, tokenizer)

    # Save results
    out_path = r"D:\windsurf\ForgeAI\.devin\benchmark_results_all.json"
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"{'='*70}")

    # Print summary
    print("\n" + "="*70)
    print("BOTTLENECK SUMMARY")
    print("="*70)

    if "per_layer" in RESULTS:
        pl = RESULTS["per_layer"]
        print(f"  Per-layer: attn={pl['attn_pct']:.0f}% | conv={pl['conv_pct']:.0f}% | head={pl['head_ms']:.1f}ms")

    if "inference_strategies" in RESULTS:
        for strat, data in RESULTS["inference_strategies"].items():
            if "tokens_per_sec" in data:
                print(f"  Inference ({strat}): {data['tokens_per_sec']:.1f} tok/s | VRAM: {fmt_mb(data['vram_mb'])}")

    if "training" in RESULTS:
        tr = RESULTS["training"]
        if "fwd_bwd_tps" in tr:
            print(f"  Training: {tr['fwd_bwd_tps']:.0f} tok/s (fwd+bwd) | VRAM: {fmt_mb(tr['vram_mb'])}")

    if "eagle3" in RESULTS and "draft_forward_ms" in RESULTS["eagle3"]:
        e = RESULTS["eagle3"]
        print(f"  EAGLE-3: draft={e['draft_forward_ms']:.1f}ms vs target={e['target_forward_ms']:.1f}ms (ratio={e['draft_target_ratio']:.2f}x)")

    if "vram_breakdown" in RESULTS:
        v = RESULTS["vram_breakdown"]
        print(f"  VRAM: model={fmt_mb(v['model_mb'])} | kv={fmt_mb(v['kv_cache_mb'])} | grads={fmt_mb(v['gradients_mb'])} | opt={fmt_mb(v['optimizer_mb'])}")
        print(f"  VRAM total: {fmt_mb(v['total_mb'])} | free: {fmt_mb(v['free_mb'])}")


if __name__ == "__main__":
    main()
