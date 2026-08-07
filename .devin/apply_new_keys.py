"""Apply new KeyStack transforms to ForgeLM v1 checkpoint.

Creates ForgeLM v2 with:
  - QK-Norm for MLA (identity init, lossless)
  - DenseFormer DWA (identity init, lossless)
  - SandwichNorm (identity init, lossless)
  - Logit Cap (runtime flag, near-lossless)
  - SwiGLU Clamp (runtime flag, near-lossless)
  - WQ Elimination (OPTIONAL, lossy without fine-tuning)

Usage:
    python -u .devin\\apply_new_keys.py
    python -u .devin\\apply_new_keys.py --wq-elim  # include WQ elimination
"""
import sys, os, time, argparse
sys.path.insert(0, '.')

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

SRC = "research/checkpoints/forgelm_v1.safetensors"
DST = "research/checkpoints/forgelm_v2.safetensors"

# ForgeLM v1 config
N_LAYERS = 28
D_MODEL = 1536
N_HEADS = 12
HEAD_DIM = D_MODEL // N_HEADS  # 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wq-elim", action="store_true",
                        help="Include WQ elimination (lossy without fine-tuning)")
    parser.add_argument("--src", default=SRC)
    parser.add_argument("--dst", default=DST)
    args = parser.parse_args()

    print("=" * 60)
    print("Apply New KeyStack Transforms to ForgeLM v1")
    print("=" * 60)
    print(f"  Source: {args.src}")
    print(f"  Output: {args.dst}")
    print(f"  Config: {N_LAYERS} layers, d_model={D_MODEL}, heads={N_HEADS}, head_dim={HEAD_DIM}")

    # Load all tensors from source
    print("\n[1] Loading source checkpoint...")
    t0 = time.time()
    state = {}
    with safe_open(args.src, framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    print(f"  Loaded {len(state)} tensors in {time.time()-t0:.1f}s")

    # Apply QK-Norm for MLA (lossless, identity init)
    print("\n[2] Applying QK-Norm for MLA...")
    from research.keys.qk_norm_mla_key import apply_qk_norm_mla
    state = apply_qk_norm_mla(state, N_LAYERS, HEAD_DIM)

    # Apply DenseFormer (lossless, identity init)
    print("\n[3] Applying DenseFormer DWA...")
    from research.keys.denseformer_key import DenseFormerKey
    df_key = DenseFormerKey()
    df_result = df_key.forward({"n_layers": N_LAYERS, "dilation": 1})
    if df_result.success:
        for i, w in enumerate(df_result.weights["dwa_weights"]):
            state[f"dwa_weights.{i}"] = w.to(torch.bfloat16)
        print(f"  DenseFormer: {df_result.metadata['total_params']} DWA params (identity init)")

    # Apply SandwichNorm (lossless, identity init)
    print("\n[4] Applying SandwichNorm...")
    for i in range(N_LAYERS):
        state[f"blocks.{i}.post_attn_norm.weight"] = torch.ones(D_MODEL, dtype=torch.bfloat16)
        state[f"blocks.{i}.post_ffn_norm.weight"] = torch.ones(D_MODEL, dtype=torch.bfloat16)
    print(f"  SandwichNorm: {2*N_LAYERS} post-sublayer norms (identity init)")

    # Apply Logit Cap + SwiGLU Clamp (runtime flags)
    print("\n[5] Applying runtime flags (Logit Cap + SwiGLU Clamp)...")
    state["_runtime_flags"] = torch.tensor([1], dtype=torch.uint8)
    print(f"  Runtime flags: logit_cap(+-30), swiglu_clamp(a=1.702,limit=7)")

    # Optionally apply WQ Elimination (LOSSY without fine-tuning)
    if args.wq_elim:
        print("\n[6] Applying WQ Elimination (LOSSY - needs fine-tuning)...")
        from research.keys.wq_elim_key import apply_wq_elim
        state = apply_wq_elim(state, N_LAYERS, D_MODEL)
    else:
        print("\n[6] WQ Elimination: SKIPPED (use --wq-elim to enable)")

    # Save
    print(f"\n[7] Saving to {args.dst}...")
    t0 = time.time()
    save_file(state, args.dst, metadata={
        "source": args.src,
        "pipeline": "apply_new_keys",
        "n_layers": str(N_LAYERS),
        "d_model": str(D_MODEL),
        "transforms": "qk_norm_mla,denseformer,sandwich_norm,logit_cap,swiglu_clamp"
                      + (",wq_elim" if args.wq_elim else ""),
        "tensors": str(len(state)),
    })
    size_mb = Path(args.dst).stat().st_size / 1e6
    print(f"  Saved {len(state)} tensors, {size_mb:.0f} MB in {time.time()-t0:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: ForgeLM v2 created")
    print(f"  Tensors: {len(state)} (source: {len(state) - 6})")
    print(f"  New keys applied:")
    print(f"    - QK-Norm MLA (lossless, {2*N_LAYERS*HEAD_DIM} params)")
    print(f"    - DenseFormer DWA (lossless, {df_result.metadata['total_params']} params)")
    print(f"    - SandwichNorm (lossless, {2*N_LAYERS} norms)")
    print(f"    - Logit Cap (runtime, 0 params)")
    print(f"    - SwiGLU Clamp (runtime, 0 params)")
    if args.wq_elim:
        print(f"    - WQ Elim (LOSSY, saves {N_LAYERS*D_MODEL**2/1e6:.1f}M params)")
    print(f"  Size: {size_mb:.0f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
