"""Validate a ported V7-8B checkpoint against the target model definition.

Checks (all must pass for a 'valid' checkpoint):
  1. Every model state_dict key exists in the checkpoint (except intentionally
     skipped: RoPE buffers, MTP head).
  2. No unexpected checkpoint keys.
  3. All shapes match.
  4. Critical tensors are non-degenerate (embed not all-zero, NLRQ S positive,
     norms not all-zero, attention projections not all-zero).
  5. Forward pass runs and produces finite logits (CUDA, seq=8).

Usage:
  python research/sandbox/validate_v7_8b_port.py [--ckpt path] [--config forgelm_v7_8b_b] [--no-forward]
"""
import sys, os, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM

# Keys excluded from the "must be in checkpoint" check: RoPE inv-freq buffers
# are recomputed at build time; the MTP head is trained separately;
# head.embed_ref.* is the same module object as embed.* (factorized embedding
# tying), so loading embed.* writes through to the head.
INTENTIONALLY_MISSING = ("rope.inv_freq", "mtp_module.", "head.embed_ref.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="research/checkpoints/ForgeLM_V7_8B_B_ported.safetensors")
    ap.add_argument("--config", default="forgelm_v7_8b_b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-forward", action="store_true")
    args = ap.parse_args()

    print(f"[1] Building {args.config} model on CPU (template)...")
    cfg = get_config(args.config)
    cfg.device = "cpu"
    model = ConfigurableResearchLLM(cfg)
    template = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {len(template)} template keys, {n_params/1e9:.3f}B params")

    print(f"[2] Reading checkpoint: {args.ckpt}")
    ckpt = safe_open(args.ckpt, framework="pt")
    ckpt_keys = set(ckpt.keys())
    print(f"    {len(ckpt_keys)} checkpoint keys, "
          f"{os.path.getsize(args.ckpt)/1e9:.2f} GB")

    failures = []

    # ── Key diff ──
    missing = []
    for k in template:
        if k in ckpt_keys:
            continue
        if any(s in k for s in INTENTIONALLY_MISSING):
            continue
        missing.append(k)
    unexpected = sorted(k for k in ckpt_keys if k not in template)

    print(f"\n[3] Key diff:")
    print(f"    missing from ckpt (non-intentional): {len(missing)}")
    for k in missing[:40]:
        print(f"      MISSING {k}  template shape={tuple(template[k].shape)}")
    print(f"    unexpected in ckpt: {len(unexpected)}")
    for k in unexpected[:40]:
        info = ckpt.get_slice(k)
        print(f"      UNEXPECTED {k}  shape={tuple(info.get_shape())}")
    if missing:
        failures.append(f"{len(missing)} missing keys")
    if unexpected:
        failures.append(f"{len(unexpected)} unexpected keys")

    # ── Shape diff ──
    shape_mismatch = []
    for k in template:
        if k not in ckpt_keys:
            continue
        sl = ckpt.get_slice(k)
        if tuple(sl.get_shape()) != tuple(template[k].shape):
            shape_mismatch.append((k, tuple(sl.get_shape()), tuple(template[k].shape)))
    print(f"    shape mismatches: {len(shape_mismatch)}")
    for k, cs, ts in shape_mismatch[:40]:
        print(f"      SHAPE {k}: ckpt={cs} model={ts}")
    if shape_mismatch:
        failures.append(f"{len(shape_mismatch)} shape mismatches")

    # ── Content sanity (lazy per-tensor load; skip int8 NLRQ factors for speed) ──
    print(f"\n[4] Content sanity:")
    bad_content = []
    checked = 0
    for k in sorted(ckpt_keys):
        if any(s in k for s in INTENTIONALLY_MISSING):
            continue
        t = ckpt.get_tensor(k)
        if t.dtype in (torch.int8, torch.uint8):
            # int8 NLRQ factors (U_q/V_q) are checked only for all-zero below.
            if t.numel() == 0 or t.abs().max().item() == 0:
                bad_content.append((k, "all-zero int8"))
            checked += 1
            continue
        tf = t.float()
        mx = tf.abs().max().item() if tf.numel() else 0.0
        isnan = bool(torch.isnan(tf).any().item()) if tf.numel() else False
        if isnan:
            bad_content.append((k, "NaN"))
        elif mx == 0.0 and not _zero_ok(k):
            bad_content.append((k, "all-zero"))
        checked += 1
    print(f"    checked {checked} tensors")
    for k, why in bad_content[:40]:
        print(f"      CONTENT {k}: {why}")
    if bad_content:
        failures.append(f"{len(bad_content)} degenerate tensors")

    # ── NLRQ S positive ──
    n_neg_s = 0
    for k in sorted(ckpt_keys):
        if k.endswith(".S"):
            s = ckpt.get_tensor(k).float()
            if (s <= 0).any().item():
                n_neg_s += 1
    print(f"    NLRQ S tensors with non-positive values: {n_neg_s}")
    if n_neg_s:
        failures.append(f"{n_neg_s} NLRQ S tensors non-positive")

    # ── Forward pass ──
    if not args.no_forward:
        print(f"\n[5] Forward pass ({args.device}, seq=8)...")
        dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        dtype_map = {}
        for k in template:
            if k in ckpt_keys:
                dtype_map[k] = ckpt.get_slice(k).get_dtype()
        # load strictly-checked subset; missing keys stay at init.
        # Move tensors to dev BEFORE load_state_dict(assign=True), otherwise
        # CPU tensors from safetensors replace model params while RoPE
        # inv_freq (not in ckpt) stays on dev → device mismatch in forward.
        sd = {}
        for k in ckpt_keys & set(template.keys()):
            sd[k] = ckpt.get_tensor(k).to(dev)
        missing_l, unexpected_l = model.load_state_dict(sd, strict=False, assign=True)
        print(f"    load: {len(missing_l)} missing, {len(unexpected_l)} unexpected")
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)
        with torch.no_grad():
            out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        if isinstance(logits, tuple):
            logits = logits[0]
        finite = bool(torch.isfinite(logits.float()).all().item())
        print(f"    logits shape={tuple(logits.shape)}, finite={finite}, "
              f"std={logits.float().std().item():.4f}")
        if not finite:
            failures.append("non-finite logits")

    print("\n" + "=" * 60)
    if failures:
        print(f"INVALID: {failures}")
        sys.exit(1)
    print("VALID: all checks passed")
    sys.exit(0)


def _zero_ok(k: str) -> bool:
    """Keys where all-zero is the intended lossless init."""
    return any(s in k for s in (
        "loop_gate", "middle_gate", "lisa.gates", "lisa.align",
        "v_mix_gate", "sinks", "_attn_res.gates", "_v0_gates",
        "freq_scale", "_mhc.gate", "_memory.gate", "_memory.u",
        "_memory.v", "router",
    ))


if __name__ == "__main__":
    main()
