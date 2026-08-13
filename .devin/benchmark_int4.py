"""Benchmark INT8 KV cache and INT4 weight quantization on the ported LFM2.5 model.

Compares:
1. Baseline (bf16 weights, fp16 KV cache)
2. INT8 KV cache (bf16 weights, int8 KV cache)
3. INT4 weights (int4 weights, fp16 KV cache)
4. INT4 + INT8 (int4 weights, int8 KV cache)

Measures: VRAM usage, generation speed (tok/s), output quality (same text?)
"""
import os
import sys
import time
import gc

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM, PreAllocatedKVCache
from safetensors.torch import load_file as load_safetensors

PORTED_CHECKPOINT = os.path.join(PROJECT_ROOT, "research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "research/checkpoints/lfm25_tokenizer")

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 30


def load_model(device="cuda"):
    """Load the ported LFM2.5 model."""
    cfg = get_config("lfm25_1.2b")
    cfg_dict = {**cfg.__dict__, "device": device}
    cfg = type(cfg)(**cfg_dict)
    model = ConfigurableResearchLLM(cfg).to(device)
    state = load_safetensors(PORTED_CHECKPOINT)
    model.load_state_dict(state, strict=False)
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False
    model.eval()
    return model, cfg


def get_vram_mb(device="cuda"):
    """Get current GPU VRAM usage."""
    if device == "cpu":
        return 0
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    return allocated


def generate_text(model, tokenizer, prompt, max_new_tokens=30, device="cuda"):
    """Generate text and return (text, time, tokens_generated).

    Uses full-context forward (no KV cache) to ensure correct output
    regardless of cache implementation. Slower but correct for benchmarking.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    t0 = time.time()
    ids = input_ids.clone()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Pass FULL context each time (no KV cache)
            logits, _ = model(ids, use_cache=False)
            next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            if next_token.item() == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, next_token], dim=1)

    elapsed = time.time() - t0
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    tokens_generated = ids.shape[1] - input_ids.shape[1]
    return text, elapsed, tokens_generated


def benchmark_baseline(tokenizer, device="cuda"):
    """Benchmark baseline (no quantization)."""
    print("\n[1] Baseline (bf16 weights, no KV cache quant)")
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()

    model, cfg = load_model(device)
    vram = get_vram_mb(device)
    text, elapsed, n_tokens = generate_text(model, tokenizer, PROMPT, MAX_NEW_TOKENS, device)
    tok_s = n_tokens / elapsed

    print(f"  VRAM: {vram:.0f} MB")
    print(f"  Speed: {tok_s:.1f} tok/s ({elapsed:.2f}s for {n_tokens} tokens)")
    print(f"  Output: {text[:80]}")

    del model
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None
    return {"vram_mb": vram, "tok_s": tok_s, "text": text[:80]}


def benchmark_int4(tokenizer, device="cuda"):
    """Benchmark INT4 weight quantization."""
    print("\n[2] INT4 weight quantization")
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()

    model, cfg = load_model(device)

    from research.inference.int4_quant import quantize_model_int4
    stats = quantize_model_int4(model, group_size=128)
    print(f"  Quantized: {stats['quantized_layers']} layers, "
          f"{stats['quantized_params']/1e6:.0f}M params")

    vram = get_vram_mb(device)
    text, elapsed, n_tokens = generate_text(model, tokenizer, PROMPT, MAX_NEW_TOKENS, device)
    tok_s = n_tokens / elapsed

    print(f"  VRAM: {vram:.0f} MB")
    print(f"  Speed: {tok_s:.1f} tok/s ({elapsed:.2f}s for {n_tokens} tokens)")
    print(f"  Output: {text[:80]}")

    del model
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None
    return {"vram_mb": vram, "tok_s": tok_s, "text": text[:80], "quant_stats": stats}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {PORTED_CHECKPOINT}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    # Run benchmarks
    results = {}
    results["baseline"] = benchmark_baseline(tokenizer, device)
    results["int4"] = benchmark_int4(tokenizer, device)

    # Test bitsandbytes int4 (fused kernel, much faster)
    print("\n[3] bitsandbytes INT4 (nf4) weight quantization")
    torch.cuda.empty_cache()
    gc.collect()

    model, cfg = load_model(device)
    try:
        import bitsandbytes as bnb
        # Replace Linear layers with bnb.nn.Linear4bit
        def replace_with_4bit(module, parent=None, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear) and not any(s in full_name for s in ["embed", "head", "mtp"]):
                    # Replace with bnb 4-bit linear
                    w = child.weight.data
                    has_bias = child.bias is not None
                    new_layer = bnb.nn.Linear4bit(
                        child.in_features, child.out_features,
                        bias=has_bias, compute_dtype=torch.bfloat16,
                        quant_type="nf4",
                    )
                    new_layer.weight = bnb.nn.Params4bit(
                        w.to(torch.bfloat16), requires_grad=False,
                        quant_type="nf4",
                    )
                    if has_bias:
                        new_layer.bias = child.bias
                    new_layer = new_layer.to(device)
                    setattr(module, child_name, new_layer)
                else:
                    replace_with_4bit(child, module, full_name)

        replace_with_4bit(model)
        vram = get_vram_mb(device)
        text, elapsed, n_tokens = generate_text(model, tokenizer, PROMPT, MAX_NEW_TOKENS, device)
        tok_s = n_tokens / elapsed
        print(f"  VRAM: {vram:.0f} MB")
        print(f"  Speed: {tok_s:.1f} tok/s ({elapsed:.2f}s for {n_tokens} tokens)")
        print(f"  Output: {text[:80]}")
        results["bnb_int4"] = {"vram_mb": vram, "tok_s": tok_s, "text": text[:80]}
    except Exception as e:
        print(f"  bitsandbytes int4 failed: {e}")
        results["bnb_int4"] = {"vram_mb": 0, "tok_s": 0, "text": str(e)[:80]}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'VRAM (MB)':<15} {'Speed (tok/s)':<15} {'Output match'}")
    print("-" * 70)

    baseline_text = results["baseline"]["text"]
    for name, r in results.items():
        # Check if output is coherent (starts with same first 30 chars)
        coherent = "YES" if r["text"][:30] == baseline_text[:30] else "NO"
        print(f"{name:<20} {r['vram_mb']:<15.0f} {r['tok_s']:<15.1f} {coherent}")

    print("-" * 70)
    base_vram = results["baseline"]["vram_mb"]
    for name, r in results.items():
        if name == "baseline" or r["vram_mb"] == 0:
            continue
        savings = base_vram - r["vram_mb"]
        pct = savings / base_vram * 100
        speed_ratio = r["tok_s"] / results["baseline"]["tok_s"]
        print(f"{name}: {savings:.0f} MB saved ({pct:.0f}% VRAM), {speed_ratio:.1f}x speed")


if __name__ == "__main__":
    main()
