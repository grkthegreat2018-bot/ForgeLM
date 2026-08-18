"""ForgeLM V3 vs V2 benchmark — compute, speed, tokens/s, VRAM, RAM, I/O, size.

V2 = lfm25_1.2b (plain GQA) | V3 = forgelm_v3 (diff-attn + BitNet QAT +
TITAN + MoD), both loaded from the same ForgeLM_V2_BSP checkpoint (lossless).
Run: python .devin/bench_v3_vs_v2.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())

CKPT = r"research/checkpoints/ForgeLM_V2_BSP.safetensors"


def load_model(config_name, mod_keep=None):
    from research.config import get_config
    from research.model_loader import ConfigurableResearchLLM
    from research.checkpoint_io import load_checkpoint

    cfg = get_config(config_name, device="cuda", dtype="bfloat16")
    if mod_keep is not None:
        cfg.use_mod = True
        cfg.mod_keep_fraction = mod_keep
    m = ConfigurableResearchLLM(cfg).to("cuda").to(torch.bfloat16)

    tensors = {k: v for k, v in load_checkpoint(CKPT, map_location="cpu").items()
               if isinstance(v, torch.Tensor)}
    if cfg.attn_type == "diff":
        from research.keys.attention.differential_attn_key import (
            DifferentialAttentionKey)
        qk = next((k for k in tensors if "attn.q_proj.weight" in k), None)
        if qk is not None and tensors[qk].shape[0] == cfg.n_heads * (
                cfg.d_model // cfg.n_heads):
            tensors = DifferentialAttentionKey(
                n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                identity=True).forward(tensors).weights
    m.load_state_dict({k: v.to("cuda") for k, v in tensors.items()},
                      strict=False)
    for b in m.blocks:
        a = b.attn
        if hasattr(a, "set_identity"):
            a.set_identity((a.lambda_param == 0).all().item())
    return m


def _run_prefill_decode(m, device, n_prefill=128, n_decode=100):
    from research.model_loader import create_kv_cache
    ids = torch.randint(0, 65536, (1, n_prefill), device=device)
    cache = create_kv_cache(m, n_prefill + n_decode, batch=1, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _ = m(ids, preallocated_cache=cache)
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_decode):
            tok = logits[0, -1].argmax().view(1, 1)
            logits, _ = m(tok, preallocated_cache=cache)
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0
    return n_prefill / t_prefill, n_decode / t_decode


def bench_prefill_decode(m, device, n_prefill=128, n_decode=100, reps=3):
    """Warmup + best-of-N (kills cold-kernel / clock variance)."""
    m.eval()
    _run_prefill_decode(m, device, 64, 20)  # warmup (discarded)
    pps = []; tps = []
    for _ in range(reps):
        p, t = _run_prefill_decode(m, device, n_prefill, n_decode)
        pps.append(p); tps.append(t)
    return max(pps), max(tps)


def bench_decode_compiled(m, device, n_decode=100):
    """torch.compile the decode step (CUDA graphs) — removes CPU launch
    overhead, the dominant cost for B=1 autoregressive decode."""
    from research.model_loader import create_kv_cache
    try:
        compiled = m.compile_for_inference(mode="reduce-overhead")
    except Exception as e:
        print(f"    compile failed: {e}")
        return None
    ids = torch.randint(0, 65536, (1, 128), device=device)
    cache = create_kv_cache(m, 228, batch=1, device=device)
    with torch.no_grad():
        logits, _ = compiled(ids, preallocated_cache=cache)
        tok = logits[0, -1].argmax().view(1, 1)
        for _ in range(5):  # warmup compile
            logits, _ = compiled(tok, preallocated_cache=cache)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_decode):
            tok = logits[0, -1].argmax().view(1, 1)
            logits, _ = compiled(tok, preallocated_cache=cache)
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return n_decode / t


def bench_train(m, device, steps=4, batch=2, seq=64):
    m.train()
    x = torch.randint(0, 65536, (batch, seq), device=device)
    y = torch.randint(0, 65536, (batch, seq), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = m(x, targets=y)[1]
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e6
    return t, peak, loss.item()


def fmt(x):
    return f"{x:,.1f}"


def main():
    dev = "cuda"
    print("=" * 78)
    print("ForgeLM V3 vs V2 (RTX 5070, bf16, same BSP checkpoint)")
    print("=" * 78)

    rows = []
    for label, cfg, mod_keep in (("V2 (gqa)", "lfm25_1.2b", None),
                                 ("V3 (diff+bitnet+titan+mod)", "forgelm_v3", None),
                                 ("V3-MoD0.5 (skip in train)", "forgelm_v3", 0.5)):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        m = load_model(cfg, mod_keep=mod_keep)
        torch.cuda.synchronize()
        t_load = time.perf_counter() - t0

        n_params = sum(p.numel() for p in m.parameters())
        wbytes = sum(p.numel() * 2 for p in m.parameters())
        vram_weights = torch.cuda.memory_allocated() / 1e6

        pps, tps = bench_prefill_decode(m, dev)
        decode_compiled = None
        if "V3" in label and "MoD0.5" not in label:
            decode_compiled = bench_decode_compiled(m, dev)
        tt, peak_train, last_loss = bench_train(m, dev)

        rows.append(dict(label=label, load_s=t_load, params_m=n_params / 1e6,
                         size_mb=wbytes / 1e6, vram_w_mb=vram_weights,
                         prefill_tps=pps,
                         decode_tps=tps, decode_spt=1000.0 / tps,
                         decode_compiled=decode_compiled,
                         train_s=tt, train_peak_mb=peak_train,
                         loss=last_loss))
        print(f"  [{label}] done: load {t_load:.1f}s, "
              f"prefill {pps:.0f} tok/s, decode {tps:.0f} tok/s"
              + (f", decode-compiled {decode_compiled:.0f} tok/s"
                 if decode_compiled else "")
              + f", train {tt:.2f}s/step (peak {peak_train:.0f}MB)")

    print()
    print(f"{'metric':<34}{'V2':>12}{'V3':>14}{'V3-MoD0.5':>14}{'V3 vs V2':>12}")
    keys = [("params_m", "params (M)", "{:.0f}"),
            ("size_mb", "bf16 size (MB)", "{:.0f}"),
            ("vram_w_mb", "VRAM weights (MB)", "{:.0f}"),
            ("load_s", "load+build (s)", "{:.1f}"),
            ("prefill_tps", "prefill (tok/s)", "{:.0f}"),
            ("decode_tps", "decode (tok/s)", "{:.0f}"),
            ("decode_spt", "decode (ms/tok)", "{:.1f}"),
            ("decode_compiled", "decode compiled (tok/s)", "{:.0f}"),
            ("train_s", "train (s/step)", "{:.2f}"),
            ("train_peak_mb", "train peak VRAM (MB)", "{:.0f}"),
            ("loss", "train loss", "{:.3f}")]
    def fmtv(v):
        return f.format(v) if v is not None else "-"

    for key, name, f in keys:
        v2, v3, v3m = rows[0][key], rows[1][key], rows[2][key]
        delta = f"{100 * (v3 - v2) / v2:+.1f}%" if v2 else "-"
        print(f"{name:<34}{fmtv(v2):>12}{fmtv(v3):>14}"
              f"{fmtv(v3m):>14}{delta:>12}")

    # File size on disk (same checkpoint for both; V3 adds ~72MB of new
    # params when SAVED).
    print()
    print(f"On-disk checkpoint (shared): {os.path.getsize(CKPT)/1e6:.0f} MB")
    extra = (rows[1]["size_mb"] - rows[0]["size_mb"])
    print(f"V3 saved checkpoint delta: +{extra:.0f} MB "
          f"(TITAN+MoD+diff params)")


if __name__ == "__main__":
    main()
