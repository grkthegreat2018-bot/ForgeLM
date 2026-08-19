"""Convert a trained V3 checkpoint (Differential Attention) to V4 (GTA).

Pipeline:
  1. Reverse DifferentialAttentionKey: average doubled q/k rows → GQA shapes,
     remove lambda_param and diff-specific rms_norm.
  2. Forward GTAKey: set v_proj = k_proj (V=K warm start), add v_mix_gate=0.
  3. Keep all other weights unchanged (BitNet qscale, TITAN, MoD, MHC, AttnRes).

The result is a V4 checkpoint that loads bit-exact into forgelm_v4 config.

CLI:
  python -m research.architecture.v3_to_v4 \
    --input research/checkpoints/ForgeLM_V3_SFT.safetensors \
    --output research/checkpoints/ForgeLM_V4.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file


def convert_v3_to_v4(
    input_path: str,
    output_path: str,
    n_layers: int = 16,
    n_heads: int = 32,
    n_kv_heads: int = 8,
):
    """Convert V3 (diff attention) checkpoint to V4 (GTA attention).

    Steps:
      1. Reverse diff: average doubled q/k rows, remove lambda_param + rms_norm
      2. Forward GTA: v_proj = k_proj, add v_mix_gate = 0
      3. Save
    """
    print(f"  [V3→V4] Loading: {input_path}")
    t0 = time.time()
    state = load_file(input_path)
    print(f"  [V3→V4] Loaded {len(state)} keys in {time.time()-t0:.1f}s")

    # Find attention layers (layers with q_proj)
    attn_layers = []
    for k in state:
        if k.startswith("blocks.") and ".attn.q_proj.weight" in k:
            layer_idx = int(k.split(".")[1])
            attn_layers.append(layer_idx)
    attn_layers.sort()
    print(f"  [V3→V4] Attention layers: {attn_layers}")

    head_dim = 64  # d_model // n_heads = 2048 // 32

    # Step 1: Reverse DifferentialAttention
    #   - Average doubled q_proj rows: [2*n_heads*hd, d] → [n_heads*hd, d]
    #   - Average doubled k_proj rows: [2*n_kv*hd, d] → [n_kv*hd, d]
    #   - Remove lambda_param
    #   - Remove diff-specific rms_norm (per-head RMSNorm on diff output)
    print(f"  [V3→V4] Step 1: Reverse DifferentialAttention...")
    keys_to_remove = []
    for layer_idx in attn_layers:
        base = f"blocks.{layer_idx}.attn"

        # Average q_proj: [4096, 2048] → [2048, 2048]
        qk = f"{base}.q_proj.weight"
        if qk in state:
            w = state[qk]  # [2*n_heads*hd, d]
            if w.shape[0] == 2 * n_heads * head_dim:
                # Average the two groups per head
                # _avg_rows: view(n_heads, 2, hd, d).mean(dim=1) → [n_heads*hd, d]
                w_avg = (w.view(n_heads, 2, head_dim, -1).mean(dim=1)
                         .reshape(n_heads * head_dim, -1))
                state[qk] = w_avg.contiguous()
                print(f"    {base}.q_proj: {list(w.shape)} → {list(w_avg.shape)}")

        # Average k_proj: [1024, 2048] → [512, 2048]
        kk = f"{base}.k_proj.weight"
        if kk in state:
            w = state[kk]
            if w.shape[0] == 2 * n_kv_heads * head_dim:
                w_avg = (w.view(n_kv_heads, 2, head_dim, -1).mean(dim=1)
                         .reshape(n_kv_heads * head_dim, -1))
                state[kk] = w_avg.contiguous()
                print(f"    {base}.k_proj: {list(w.shape)} → {list(w_avg.shape)}")

        # Remove lambda_param (diff-specific)
        lam_key = f"{base}.lambda_param"
        if lam_key in state:
            keys_to_remove.append(lam_key)

        # Remove diff-specific rms_norm (GTA doesn't have this)
        rms_key = f"{base}.rms_norm.weight"
        if rms_key in state:
            keys_to_remove.append(rms_key)

    for k in keys_to_remove:
        del state[k]
    print(f"    Removed {len(keys_to_remove)} diff-specific keys "
          f"(lambda_param + rms_norm)")

    # Step 2: Forward GTAKey
    #   - Set v_proj = k_proj (V=K warm start)
    #   - Add v_mix_gate = 0 (lossless)
    print(f"  [V3→V4] Step 2: Forward GTA (V=K, v_mix_gate=0)...")
    for layer_idx in attn_layers:
        base = f"blocks.{layer_idx}.attn"
        kk = f"{base}.k_proj.weight"
        vk = f"{base}.v_proj.weight"

        if kk in state and vk in state:
            # Set v_proj = k_proj (V=K warm start)
            state[vk] = state[kk].clone()
            print(f"    {base}.v_proj = k_proj ({list(state[kk].shape)})")

        # Copy k_proj.qscale to v_proj.qscale if BitNet
        kq = f"{base}.k_proj.qscale"
        vq = f"{base}.v_proj.qscale"
        if kq in state and vq in state:
            state[vq] = state[kq].clone()

        # Add v_mix_gate = 0 (lossless)
        state[f"{base}.v_mix_gate"] = torch.zeros(1, dtype=torch.float32)

    # Save
    print(f"  [V3→V4] Saving: {output_path}")
    t1 = time.time()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(state, output_path)
    print(f"  [V3→V4] Saved {len(state)} keys in {time.time()-t1:.1f}s")
    print(f"  [V3→V4] Done! Total: {time.time()-t0:.1f}s")

    # Verify key shapes
    print(f"\n  [V3→V4] Verification (layer 2):")
    for k in sorted(state.keys()):
        if "blocks.2.attn" in k:
            print(f"    {k}: {list(state[k].shape)}")


def main():
    parser = argparse.ArgumentParser(description="Convert V3 checkpoint to V4")
    parser.add_argument("--input", "-i", required=True,
                        help="Input V3 checkpoint (.safetensors)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output V4 checkpoint (.safetensors)")
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    args = parser.parse_args()

    convert_v3_to_v4(
        input_path=args.input,
        output_path=args.output,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
    )


if __name__ == "__main__":
    main()
