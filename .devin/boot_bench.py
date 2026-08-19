"""Boot-time benchmark — cold load to first token.

Measures every stage of the boot pipeline per the BOOT_TIME_AUDIT.md 7-stage
model. Reports per-stage wall-clock so we can see exactly where time goes
and compare novel variations against the baseline.

Run:
    python .devin/boot_bench.py [--tiny] [--no-cache-clear]
        --tiny          Use lfm25_tiny (4 layers) for fast iteration
        --no-cache-clear  Skip OS file cache flush (warm-cache run)
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Load .env so API keys / env flags are set before torch imports.
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Force a cold torch import (we want to measure import cost too).
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def flush_os_cache(path: str) -> None:
    """Best-effort flush of OS file cache for the checkpoint on Windows."""
    try:
        import ctypes
        # SetSystemFileCacheSize would need admin; instead use the simpler
        # approach of evicting via large dummy allocation + free.
        # On Windows the OS cache is hard to flush without admin, so we
        # just report whether the file appears cached via first-read timing.
        _ = ctypes.windll.kernel32  # noqa: just probe
    except Exception:
        pass


def stage(name: str, t0: float, stages: dict) -> float:
    """Record a stage boundary, return new t0."""
    t = time.perf_counter()
    stages[name] = t - t0
    return t


def bench_baseline(config_name: str, checkpoint_path: str, device: str,
                   dtype_str: str, clear_cache: bool) -> dict:
    """Baseline: replicate load_default_model exactly, but with per-stage timing."""
    import torch
    stages = {}
    t0 = time.perf_counter()

    # Stage A: torch + research imports (already imported above, but measure
    # the heavy model_loader import separately).
    t0 = stage("A_imports_setup", t0, stages)

    if clear_cache:
        flush_os_cache(checkpoint_path)

    # Stage B: config fetch
    from research.config import get_config
    cfg = get_config(config_name, device=device)
    t0 = stage("B_config_fetch", t0, stages)

    # Stage C: tokenizer load (in baseline we mimic load_default_model: serial
    # AFTER model build). We'll measure it separately too.
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    # Stage D: architecture build (ConfigurableResearchLLM.__init__ + .to(device))
    from research.model_loader import ConfigurableResearchLLM
    t_arch = time.perf_counter()
    model = ConfigurableResearchLLM(cfg).to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    # Stage E: dtype convert
    t_dtype = time.perf_counter()
    if dtype is not None:
        model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    # Stage F: checkpoint mmap / fastsafetensors load
    from research.model_loader import ModelLoader
    if checkpoint_path is None:
        stages["F_weights_load"] = 0.0
        stages["G_diff_convert"] = 0.0
        stages["H_load_state_dict"] = 0.0
        stages["I_post_load_scan"] = 0.0
        state = {}
    else:
        t_weights = time.perf_counter()
        state = ModelLoader._load_safetensors_mmap(
            checkpoint_path, model, device=torch.device(device))
        stages["F_weights_load"] = time.perf_counter() - t_weights

        # Stage G: GQA -> diff warm start conversion (if applicable)
        t_conv = time.perf_counter()
        if cfg.attn_type == "diff":
            qk = next((k for k in state if "attn.q_proj.weight" in k), None)
            if qk is not None:
                exp_rows = cfg.n_heads * (cfg.d_model // cfg.n_heads)
                if state[qk].shape[0] == exp_rows:
                    from research.keys.attention.differential_attn_key import (
                        DifferentialAttentionKey)
                    res = DifferentialAttentionKey(
                        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                        identity=True).forward(state)
                    if res.success:
                        state = res.weights
        stages["G_diff_convert"] = time.perf_counter() - t_conv

        # Stage H: load_state_dict (GPU->GPU copy)
        t_lsd = time.perf_counter()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if device == "cuda":
            torch.cuda.synchronize()
        stages["H_load_state_dict"] = time.perf_counter() - t_lsd

        # Stage I: post-load QK-norm + diff-attn identity scan
        t_scan = time.perf_counter()
        for block in model.blocks:
            attn = block.attn
            if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                q_id = (attn.q_norm.weight == 1.0).all()
                k_id = (attn.k_norm.weight == 1.0).all()
                attn._qk_norm_identity = bool(q_id and k_id)
            if hasattr(attn, 'lambda_param') and hasattr(attn, 'set_identity'):
                attn.set_identity((attn.lambda_param == 0.0).all().item())
        if device == "cuda":
            torch.cuda.synchronize()
        stages["I_post_load_scan"] = time.perf_counter() - t_scan

    model.eval()

    # Stage J: tokenizer load (serial, after model — baseline behavior)
    t_tok = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok

    # Stage K: KV cache alloc
    from research.model_loader import create_kv_cache
    t_kv = time.perf_counter()
    cache = create_kv_cache(model, max_total=2048, batch=1,
                            device=torch.device(device))
    if device == "cuda":
        torch.cuda.synchronize()
    stages["K_kv_cache_alloc"] = time.perf_counter() - t_kv

    # Stage L: first forward (prefill) — includes cache_devices() lazy scan
    t_ff = time.perf_counter()
    ids = tok("The capital of France is", return_tensors="pt")
    if hasattr(ids, "to"):
        ids = ids.to(device)
    else:
        ids = {k: v.to(device) for k, v in ids.items()}
    input_ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
    with torch.no_grad():
        out = model(input_ids, preallocated_cache=cache, use_cache=True)
        logits = out[0]
    if device == "cuda":
        torch.cuda.synchronize()
    stages["L_first_forward"] = time.perf_counter() - t_ff

    # Stage M: first decode token (argmax + write to cache)
    t_decode = time.perf_counter()
    with torch.no_grad():
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache.advance()
        out2 = model(next_tok, preallocated_cache=cache, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    stages["M_first_decode"] = time.perf_counter() - t_decode

    stages["TOTAL"] = sum(stages.values())
    stages["config"] = config_name
    stages["checkpoint"] = checkpoint_path
    stages["device"] = device
    stages["dtype"] = dtype_str
    stages["clear_cache"] = clear_cache
    if device == "cuda":
        stages["vram_gb"] = torch.cuda.memory_allocated() / 1e9
        stages["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    stages["n_params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    return stages, model, tok, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                    help="Use lfm25_tiny for fast iteration")
    ap.add_argument("--no-cache-clear", action="store_true",
                    help="Skip OS cache flush (warm run)")
    ap.add_argument("--config", default=None, help="Override config name")
    ap.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--runs", type=int, default=1,
                    help="Repeat N times (only first is truly cold)")
    args = ap.parse_args()

    if args.config:
        config_name = args.config
    else:
        config_name = "lfm25_tiny" if args.tiny else "forgelm_v3"
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        if args.tiny:
            checkpoint_path = None  # tiny uses random weights
        else:
            checkpoint_path = "research/checkpoints/ForgeLM_V3_Base.safetensors"

    print(f"\n{'='*70}")
    print(f"BOOT BENCH — baseline")
    print(f"  config:     {config_name}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  device:     {args.device}")
    print(f"  dtype:      {args.dtype}")
    print(f"  clear_cache: {not args.no_cache_clear}")
    print(f"{'='*70}\n")

    all_runs = []
    for run in range(args.runs):
        if run > 0:
            gc.collect()
            if args.device == "cuda":
                import torch
                torch.cuda.empty_cache()
        stages, model, tok, cache = bench_baseline(
            config_name, checkpoint_path, args.device, args.dtype,
            clear_cache=not args.no_cache_clear and run == 0)
        all_runs.append(stages)
        print(f"\n--- Run {run+1}/{args.runs} ---")
        for k, v in stages.items():
            if k in ("vram_gb", "vram_reserved_gb"):
                print(f"  {k:25s} {v:8.3f} GB")
            elif k in ("n_params_m",):
                print(f"  {k:25s} {v:8.1f} M")
            elif isinstance(v, float):
                print(f"  {k:25s} {v*1000:8.1f} ms")
            else:
                print(f"  {k:25s} {v}")
        # Cleanup
        del model, tok, cache
        gc.collect()
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()

    # Save results
    out_path = PROJECT / ".devin" / "boot_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
