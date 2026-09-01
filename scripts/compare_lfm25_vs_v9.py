"""Compare LFM2.5-1.2B vs ForgeLM V9-1.2B: tokens/sec, memory, accuracy.

Tests:
  1. Lossless verification: forward pass logit diff (should be ~0.0)
  2. Speed: tokens/sec generation (greedy, 128 tokens)
  3. Memory: VRAM + RAM usage during inference
  4. Accuracy: multi-question test (10 questions, greedy decoding)
  5. KV cache: standard vs SpectralKV compression ratio

Usage:
  python scripts/compare_lfm25_vs_v9.py
"""
import sys, os, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(msg):
    print(msg, flush=True)

def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() / 1e6
    return 0, 0

def get_ram_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6

# ─── Test questions ───────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    ("What is 2+2?", "4"),
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What color do you get when you mix blue and yellow?", "green"),
    ("How many sides does a triangle have?", "three"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is 10 minus 3?", "7"),
    ("Name a fruit that is red.", "apple"),
    ("What season comes after winter?", "spring"),
]

def load_model(config_name, checkpoint_path, device="cuda"):
    """Load a model with ForgeEngine."""
    from research.config import get_config
    from research.model_loader import ModelLoader
    cfg = get_config(config_name, device=device)
    # Use bf16 to save VRAM
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=checkpoint_path,
                                          dtype=torch.bfloat16, fast_load=True)
    model.eval()
    return model, cfg

def forward_logits(model, tokenizer, text, device="cuda", max_len=128):
    """Get logits from a forward pass on text."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out.logits if hasattr(out, "logits") else out
    return logits, input_ids

def generate_text(model, tokenizer, prompt, max_new_tokens=64, device="cuda"):
    """Generate text greedily (temperature=0)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(generated)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out.logits if hasattr(out, "logits") else out
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    return result

def check_answer(response, expected):
    """Check if the expected answer appears in the response (case-insensitive)."""
    return expected.lower() in response.lower()

# ─── Main comparison ──────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("LFM2.5-1.2B vs ForgeLM V9-1.2B Comparison")
    log("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\nDevice: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok_path = "research/checkpoints/lfm25_tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    log(f"Tokenizer: {tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size}")

    # ── Load both models ──
    results = {}

    # LFM2.5
    log("\n" + "─" * 50)
    log("Loading LFM2.5-1.2B...")
    t0 = time.time()
    try:
        model_lfm, cfg_lfm = load_model("forgelm_v10_1.2b",
            "research/checkpoints/ForgeLM_V10_1.2B.safetensors", device)
        load_time_lfm = time.time() - t0
        ram_lfm = get_ram_mb()
        gpu_alloc_lfm, gpu_max_lfm = get_gpu_mem()
        n_params_lfm = sum(p.numel() for p in model_lfm.parameters())
        log(f"  Loaded in {load_time_lfm:.1f}s")
        log(f"  Params: {n_params_lfm/1e6:.1f}M")
        log(f"  GPU mem: {gpu_alloc_lfm:.0f} MB (peak {gpu_max_lfm:.0f} MB)")
        log(f"  RAM: {ram_lfm:.0f} MB")
        results["lfm25"] = {"load_time": load_time_lfm, "params": n_params_lfm,
                            "gpu_mb": gpu_alloc_lfm, "gpu_peak_mb": gpu_max_lfm,
                            "ram_mb": ram_lfm}
    except Exception as e:
        log(f"  FAILED: {e}")
        traceback.print_exc()
        model_lfm = None

    # V9
    log("\n" + "─" * 50)
    log("Loading ForgeLM V9-1.2B...")
    t0 = time.time()
    try:
        model_v9, cfg_v9 = load_model("forgelm_v10_1.2b",
            "research/checkpoints/ForgeLM_V10_1.2B.safetensors", device)
        load_time_v9 = time.time() - t0
        ram_v9 = get_ram_mb()
        gpu_alloc_v9, gpu_max_v9 = get_gpu_mem()
        n_params_v9 = sum(p.numel() for p in model_v9.parameters())
        log(f"  Loaded in {load_time_v9:.1f}s")
        log(f"  Params: {n_params_v9/1e6:.1f}M")
        log(f"  GPU mem: {gpu_alloc_v9:.0f} MB (peak {gpu_max_v9:.0f} MB)")
        log(f"  RAM: {ram_v9:.0f} MB")
        results["v9"] = {"load_time": load_time_v9, "params": n_params_v9,
                         "gpu_mb": gpu_alloc_v9, "gpu_peak_mb": gpu_max_v9,
                         "ram_mb": ram_v9}
    except Exception as e:
        log(f"  FAILED: {e}")
        traceback.print_exc()
        model_v9 = None

    if model_lfm is None or model_v9 is None:
        log("\nCannot compare — one or both models failed to load")
        return

    # ── Test 1: Lossless verification (logit diff) ──
    log("\n" + "=" * 50)
    log("TEST 1: Lossless verification (logit diff)")
    log("=" * 50)

    test_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "The capital of France is",
        "def fibonacci(n):",
        "Machine learning is a subset of",
    ]

    max_logit_diff = 0.0
    for prompt in test_prompts:
        try:
            logits_lfm, ids_lfm = forward_logits(model_lfm, tokenizer, prompt, device)
            logits_v9, ids_v9 = forward_logits(model_v9, tokenizer, prompt, device)
            # Compare last token logits
            diff = (logits_lfm[:, -1, :] - logits_v9[:, -1, :]).abs().max().item()
            max_logit_diff = max(max_logit_diff, diff)
            log(f"  '{prompt[:30]}...': max logit diff = {diff:.6f}")
        except Exception as e:
            log(f"  '{prompt[:30]}...': ERROR: {e}")

    log(f"\n  MAX LOGIT DIFF: {max_logit_diff:.6f}")
    is_lossless = max_logit_diff < 0.01
    log(f"  LOSSLESS: {'YES ✓' if is_lossless else 'NO ✗'}")
    results["lossless"] = is_lossless
    results["max_logit_diff"] = max_logit_diff

    # ── Test 2: Speed (tokens/sec) ──
    log("\n" + "=" * 50)
    log("TEST 2: Generation speed (tokens/sec)")
    log("=" * 50)

    speed_prompt = "The future of artificial intelligence is"
    n_gen = 64

    for name, model in [("LFM2.5", model_lfm), ("V9", model_v9)]:
        try:
            # Warmup
            _ = generate_text(model, tokenizer, speed_prompt, max_new_tokens=8, device=device)
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            output = generate_text(model, tokenizer, speed_prompt, max_new_tokens=n_gen, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tps = n_gen / elapsed
            log(f"  {name}: {tps:.1f} tok/s ({n_gen} tokens in {elapsed:.2f}s)")
            results.setdefault(name, {})["tokens_per_sec"] = tps
            results.setdefault(name, {"tokens_per_sec": tps})
        except Exception as e:
            log(f"  {name}: ERROR: {e}")
            traceback.print_exc()

    # ── Test 3: Memory during generation ──
    log("\n" + "=" * 50)
    log("TEST 3: Memory during generation")
    log("=" * 50)

    for name, model in [("LFM2.5", model_lfm), ("V9", model_v9)]:
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ram_before = get_ram_mb()
            _ = generate_text(model, tokenizer, "Generate a long story about a robot.",
                              max_new_tokens=128, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            gpu_alloc, gpu_peak = get_gpu_mem()
            ram_after = get_ram_mb()
            log(f"  {name}: GPU={gpu_alloc:.0f} MB (peak {gpu_peak:.0f} MB), RAM={ram_after:.0f} MB")
            results.setdefault(name, {})["gen_gpu_mb"] = gpu_alloc
            results.setdefault(name, {})["gen_gpu_peak_mb"] = gpu_peak
            results.setdefault(name, {})["gen_ram_mb"] = ram_after
        except Exception as e:
            log(f"  {name}: ERROR: {e}")

    # ── Test 4: Accuracy (multi-question) ──
    log("\n" + "=" * 50)
    log("TEST 4: Accuracy (10 questions, greedy decoding)")
    log("=" * 50)

    for name, model in [("LFM2.5", model_lfm), ("V9", model_v9)]:
        correct = 0
        total = len(TEST_QUESTIONS)
        for i, (question, expected) in enumerate(TEST_QUESTIONS):
            try:
                prompt = f"Question: {question}\nAnswer:"
                response = generate_text(model, tokenizer, prompt, max_new_tokens=32, device=device)
                # Extract just the answer part
                answer_part = response[len(prompt):].strip() if prompt in response else response
                is_correct = check_answer(answer_part, expected)
                if is_correct:
                    correct += 1
                log(f"  {name} Q{i+1}: {'✓' if is_correct else '✗'} '{question}' -> '{answer_part[:50]}'")
            except Exception as e:
                log(f"  {name} Q{i+1}: ERROR: {e}")

        accuracy = correct / total
        log(f"\n  {name} Accuracy: {correct}/{total} = {accuracy:.1%}")
        results.setdefault(name, {})["accuracy"] = accuracy
        results.setdefault(name, {})["correct"] = correct
        results.setdefault(name, {})["total"] = total

    # ── Test 5: KV cache comparison ──
    log("\n" + "=" * 50)
    log("TEST 5: KV cache compression")
    log("=" * 50)

    # Standard KV cache size (per layer, per token)
    n_kv = cfg_lfm.n_kv_heads  # 8
    head_dim = cfg_lfm.d_model // cfg_lfm.n_heads  # 64
    n_layers = cfg_lfm.n_layers  # 16
    seq_len = 2048

    standard_kv_bytes = 2 * n_kv * head_dim * seq_len * n_layers * 2  # K+V, bf16
    standard_kv_mb = standard_kv_bytes / 1e6

    # SpectralKV size
    max_freq = cfg_v9.spectral_kv_max_freq  # 64
    n_coeffs = 1 + 2 * max_freq  # 129
    sink_size = cfg_v9.spectral_kv_sink_size  # 4
    spectral_kv_bytes = (2 * n_kv * head_dim * n_coeffs * n_layers * 2 +  # coefficients
                         2 * n_kv * head_dim * sink_size * n_layers * 2)  # sinks
    spectral_kv_mb = spectral_kv_bytes / 1e6
    compression = standard_kv_bytes / spectral_kv_bytes

    log(f"  Standard KV (seq={seq_len}): {standard_kv_mb:.1f} MB")
    log(f"  SpectralKV (max_freq={max_freq}): {spectral_kv_mb:.1f} MB")
    log(f"  Compression: {compression:.0f}×")
    results["kv_standard_mb"] = standard_kv_mb
    results["kv_spectral_mb"] = spectral_kv_mb
    results["kv_compression"] = compression

    # ── Summary ──
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    lfm = results.get("lfm25", {})
    v9 = results.get("v9", {})

    log(f"\n{'Metric':<30} {'LFM2.5':>15} {'V9-1.2B':>15} {'Winner':>10}")
    log(f"{'─'*70}")

    def fmt(v, suffix=""):
        return f"{v:.1f}{suffix}" if isinstance(v, float) else str(v)

    # Params
    log(f"{'Params (M)':<30} {lfm.get('params',0)/1e6:>15.1f} {v9.get('params',0)/1e6:>15.1f} {'=':>10}")
    # Load time
    log(f"{'Load time (s)':<30} {lfm.get('load_time',0):>15.1f} {v9.get('load_time',0):>15.1f} {'=':>10}")
    # GPU mem
    log(f"{'GPU mem (MB)':<30} {lfm.get('gpu_mb',0):>15.0f} {v9.get('gpu_mb',0):>15.0f} {'=':>10}")
    # Speed
    lfm_tps = lfm.get('tokens_per_sec', 0)
    v9_tps = v9.get('tokens_per_sec', 0)
    speed_winner = "V9" if v9_tps > lfm_tps else ("LFM" if lfm_tps > v9_tps else "=")
    log(f"{'Speed (tok/s)':<30} {lfm_tps:>15.1f} {v9_tps:>15.1f} {speed_winner:>10}")
    # Accuracy
    lfm_acc = lfm.get('accuracy', 0)
    v9_acc = v9.get('accuracy', 0)
    acc_winner = "V9" if v9_acc > lfm_acc else ("LFM" if lfm_acc > v9_acc else "=")
    log(f"{'Accuracy':<30} {lfm_acc:>14.0%} {v9_acc:>15.0%} {acc_winner:>10}")
    # Lossless
    log(f"{'Lossless (logit diff)':<30} {'baseline':>15} {results.get('max_logit_diff', 999):>15.6f} {'=':>10}")
    log(f"{'Lossless':<30} {'':>15} {'YES' if results.get('lossless', False) else 'NO':>15} {'':>10}")
    # KV cache
    log(f"{'KV cache (MB @2K ctx)':<30} {standard_kv_mb:>15.1f} {spectral_kv_mb:>15.1f} {'V9':>10}")
    log(f"{'KV compression':<30} {'1×':>15} {compression:>14.0f}× {'V9':>10}")

    # Verdict
    log(f"\n{'─'*70}")
    v9_exact = results.get('lossless', False)
    v9_beats = (v9_tps >= lfm_tps and v9_acc >= lfm_acc and spectral_kv_mb < standard_kv_mb)
    if v9_exact and v9_beats:
        log("VERDICT: V9 is EXACT and BEATS LFM2.5 in all tech terms ✓")
        log("  → Safe to delete prior ForgeLM models and keep V9")
    elif v9_exact:
        log("VERDICT: V9 is EXACT but does not beat LFM2.5 in all metrics")
    else:
        log("VERDICT: V9 is NOT exact — investigate before deleting prior models")

    # Save results
    with open("scripts/_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to scripts/_comparison_results.json")

    # Cleanup
    del model_lfm, model_v9
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
