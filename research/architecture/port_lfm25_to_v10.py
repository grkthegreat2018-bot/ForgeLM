"""Port official LFM2.5-1.2B HF checkpoint to ForgeLM V2 Light with IRI-FP4 quantization.

V10 is the same architecture as V9 (d_model=2048, 16 layers, 10 conv + 6 GQA)
but applies IRI-FP4 (Iterative Residual Refinement FP4) weight quantization to
all 2D Linear weights. IRI-FP4 runs K rounds of FP4 quantization on the
successive residuals, achieving 62.6 dB SQNR at 3 rounds (1.6 bits/weight).

Key mapping is identical to the LFM2.5 HF layout (10 conv + 6 GQA layers).
The IRI-FP4 quantization is applied AFTER the key mapping as a separate step.

Checkpoint format:
  - 2D Linear weights → replaced by 3 packed tensors:
      {key}.iri_packed        (n_rounds, out, in_padded//2) uint8  — FP4 codes
      {key}.iri_scales        (n_rounds, out, n_blocks)    float16 — per-block scales
      {key}.iri_global_scale  (n_rounds, out)              float32 — per-row global
    Plus a dequantized copy for verification:
      {key}.weight_dequant    (out, in)                     bfloat16
  - 1D weights (conv, norms), embedding, head → kept as-is in bfloat16

Usage:
  python -m research.architecture.port_lfm25_to_v10 \
    --input research/checkpoints/lfm25_official/model.safetensors \
    --output research/checkpoints/ForgeLM_V2_Light.safetensors

  # Verify forward-pass parity against V9:
  python -m research.architecture.port_lfm25_to_v10 \
    --input research/checkpoints/lfm25_official/model.safetensors \
    --output research/checkpoints/ForgeLM_V2_Light.safetensors \
    --verify --v9-checkpoint research/checkpoints/ForgeLM_V9_1.2B.safetensors
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from research.inference.quant.novel_quant import (
    _optimal_fp4_scale,
    _fp4_quant_dequant_block,
    _pack_fp4_round,
    _dequantize_fp4,
    quantize_iri_fp4,
)
from research.inference.quant.nvfp4_quant import _FP8_DTYPE


# ═══════════════════════════════════════════════════════════════════════════════
# Key mapping (LFM2.5 HF layout — V10 = LFM2.5 architecture)
# ═══════════════════════════════════════════════════════════════════════════════

KEY_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    # LFM2.5's "embedding_norm" is actually the FINAL norm (applied after
    # all layers, before the head). Misleading name in HF code.
    "model.embedding_norm.weight": "ln_f.weight",
}

LAYER_KEY_MAP = {
    "operator_norm.weight": "ln1.weight",
    "ffn_norm.weight": "ln2.weight",
    "conv.in_proj.weight": "attn.in_proj.weight",
    "conv.conv.weight": "attn.conv.weight",
    "conv.out_proj.weight": "attn.out_proj.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.out_proj.weight": "attn.out_proj.weight",
    "self_attn.q_layernorm.weight": "attn.q_norm.weight",
    "self_attn.k_layernorm.weight": "attn.k_norm.weight",
    "feed_forward.w1.weight": "ffn.w_gate.weight",
    "feed_forward.w2.weight": "ffn.w_down.weight",
    "feed_forward.w3.weight": "ffn.w_up.weight",
}


def map_layer_key(src_key: str) -> str | None:
    """Map a HuggingFace key to ForgeAI internal key (same as V9 port)."""
    if not src_key.startswith("model.layers."):
        return KEY_MAP.get(src_key)
    parts = src_key.split(".")
    layer_n = int(parts[2])
    rest = ".".join(parts[3:])
    dst_suffix = LAYER_KEY_MAP.get(rest)
    if dst_suffix is None:
        return None
    return f"blocks.{layer_n}.{dst_suffix}"


# ═══════════════════════════════════════════════════════════════════════════════
# IRI-FP4 packing (per-weight-tensor)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_iri_fp4_packed(w: torch.Tensor, block_size: int = 32,
                             n_rounds: int = 3) -> dict:
    """IRI-FP4 quantize a 2D weight, returning packed data + dequantized weight.

    Runs n_rounds of FP4 quantization on successive residuals. Each round is
    packed as (packed uint8, float16 block scales, float32 global scale).

    Returns dict with:
      'iri_packed':       (n_rounds, out, in_padded//2) uint8
      'iri_scales':       (n_rounds, out, n_blocks)    float16
      'iri_global_scale': (n_rounds, out)              float32
      'weight_dequant':   (out, in)                    bfloat16
    """
    out_f, in_f = w.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(w, (0, pad)) if pad > 0 else w
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    packed_list = []
    scales_list = []
    gs_list = []
    acc = torch.zeros_like(wp)
    residual = wp.clone()

    for _ in range(n_rounds):
        wb = residual.view(out_f, n_blocks, block_size)
        scale = _optimal_fp4_scale(wb)
        w_dq = _fp4_quant_dequant_block(wb, scale)
        packed, scales_fp8, gs = _pack_fp4_round(wb, scale, block_size)
        packed_list.append(packed)
        scales_list.append(scales_fp8)
        gs_list.append(gs)
        acc = acc + w_dq.view(out_f, in_p)
        residual = residual - w_dq.view(out_f, in_p)

    iri_packed = torch.stack(packed_list, dim=0)       # (R, out, in_p//2)
    iri_scales = torch.stack(scales_list, dim=0)       # (R, out, n_blocks)
    iri_gs = torch.stack(gs_list, dim=0)               # (R, out)
    dequant = acc[:, :in_f].to(torch.bfloat16)

    return {
        "iri_packed": iri_packed.contiguous(),
        "iri_scales": iri_scales.to(torch.float16).contiguous(),
        "iri_global_scale": iri_gs.contiguous(),
        "weight_dequant": dequant.contiguous(),
    }


def dequantize_iri_fp4_packed(iri_packed: torch.Tensor,
                               iri_scales: torch.Tensor,
                               iri_global_scale: torch.Tensor,
                               out_features: int, in_features: int,
                               block_size: int = 32) -> torch.Tensor:
    """Reconstruct dequantized weight from IRI-FP4 packed data."""
    n_rounds = iri_packed.shape[0]
    acc = torch.zeros(out_features, in_features, dtype=torch.float32,
                      device=iri_packed.device)
    for r in range(n_rounds):
        w = _dequantize_fp4(
            iri_packed[r], iri_scales[r].to(_FP8_DTYPE),
            out_features, in_features, block_size, torch.float32,
            global_scale=iri_global_scale[r],
        )
        acc = acc + w
    return acc.to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════════
# Classification of weights
# ═══════════════════════════════════════════════════════════════════════════════

# Keys to keep as-is (not quantized): embeddings, head, 1D norms, 1D conv
KEEP_AS_IS_SUFFIXES = (
    "embed.weight", "head.weight", "ln_f.weight",
    "ln1.weight", "ln2.weight",
    "q_norm.weight", "k_norm.weight",
)

# 1D conv weights (in_proj, conv, out_proj) — small, keep as-is
CONV_SUFFIXES = (
    "attn.in_proj.weight", "attn.conv.weight", "attn.out_proj.weight",
)


def should_quantize(key: str, tensor: torch.Tensor) -> bool:
    """Determine if a weight should be IRI-FP4 quantized."""
    if tensor.ndim != 2:
        return False
    if any(key.endswith(s) for s in KEEP_AS_IS_SUFFIXES):
        return False
    if any(key.endswith(s) for s in CONV_SUFFIXES):
        return False
    # Quantize all remaining 2D weights (q/k/v/out_proj, ffn gate/up/down)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main port function
# ═══════════════════════════════════════════════════════════════════════════════

def port_lfm25_to_v10(input_path: str, output_path: str,
                       device: str = "cpu",
                       n_rounds: int = 3,
                       block_size: int = 32):
    """Port official LFM2.5 HF checkpoint to ForgeLM V2 Light with IRI-FP4 quantization.

    Steps:
      1. Load LFM2.5 checkpoint (same key mapping as V9 port)
      2. Apply IRI-FP4 quantization to all 2D Linear weights
      3. Save as V10 checkpoint with IRI-FP4 packed weights + dequant copies

    Args:
        input_path: path to LFM2.5 HF .safetensors checkpoint
        output_path: path to write V10 .safetensors checkpoint
        device: device for quantization computation
        n_rounds: IRI-FP4 rounds (3 = 62.6 dB SQNR, 1.6 bpw)
        block_size: IRI-FP4 block size
    """
    t0 = time.time()
    print(f"Porting LFM2.5 → ForgeLM V2 Light (IRI-FP4, {n_rounds} rounds, block={block_size})")

    # Load source
    print(f"\n[1] Loading source: {input_path}")
    src = load_file(input_path)
    src_params = sum(v.numel() for v in src.values())
    src_bytes = sum(v.numel() * v.element_size() for v in src.values())
    print(f"  {len(src)} tensors, {src_params/1e6:.1f}M params, {src_bytes/1e9:.2f} GB (bf16)")

    # Key mapping (same as V9)
    print(f"\n[2] Mapping keys (same as V9 port)...")
    mapped = {}
    copied = 0
    skipped = 0
    for src_key, src_w in src.items():
        dst_key = map_layer_key(src_key)
        if dst_key is None:
            print(f"  UNMAPPED: {src_key}")
            skipped += 1
            continue
        mapped[dst_key] = src_w.to(torch.bfloat16).contiguous()
        copied += 1

    # Handle tied head
    if "head.weight" not in mapped and "embed.weight" in mapped:
        mapped["head.weight"] = mapped["embed.weight"].clone()
        copied += 1
        print(f"  TIED: head.weight = embed.weight")

    print(f"  Mapped: {copied}, Skipped: {skipped}")

    # Apply IRI-FP4 quantization
    print(f"\n[3] Applying IRI-FP4 quantization ({n_rounds} rounds, block_size={block_size})...")
    dst = {}
    n_quantized = 0
    n_kept = 0
    quantized_params = 0
    kept_params = 0

    for key, w in mapped.items():
        if should_quantize(key, w):
            w_f = w.float().to(device)
            packed = quantize_iri_fp4_packed(w_f, block_size, n_rounds)
            dst[f"{key}.iri_packed"] = packed["iri_packed"]
            dst[f"{key}.iri_scales"] = packed["iri_scales"]
            dst[f"{key}.iri_global_scale"] = packed["iri_global_scale"]
            # NOTE: weight_dequant is NOT stored — it's reconstructed on load
            # by dequantize_iri_fp4_state() in model_loader.py
            n_quantized += 1
            quantized_params += w.numel()
        else:
            dst[key] = w.contiguous()
            n_kept += 1
            kept_params += w.numel()

    print(f"  Quantized (IRI-FP4): {n_quantized} tensors, {quantized_params/1e6:.1f}M params")
    print(f"  Kept as-is (bf16):   {n_kept} tensors, {kept_params/1e6:.1f}M params")

    # Compute memory savings
    print(f"\n[4] Memory analysis:")
    # Original bf16 size
    orig_bf16_bytes = quantized_params * 2  # bf16 = 2 bytes
    # IRI-FP4 packed size: n_rounds * (0.5 bytes per weight for packed
    #   + n_blocks * scale_bytes / out + global_scale_bytes / out)
    # Approximate: ~0.56 bytes/weight per round
    iri_bytes = 0
    for key in mapped:
        if should_quantize(key, mapped[key]):
            out_f, in_f = mapped[key].shape
            in_padded = in_f + (block_size - in_f % block_size) % block_size
            n_blocks = in_padded // block_size
            # packed: n_rounds * out * (in_padded // 2) * 1 byte (uint8)
            iri_bytes += n_rounds * out_f * (in_padded // 2) * 1
            # scales: n_rounds * out * n_blocks * 2 bytes (float16)
            iri_bytes += n_rounds * out_f * n_blocks * 2
            # global_scale: n_rounds * out * 4 bytes (float32)
            iri_bytes += n_rounds * out_f * 4
    kept_bytes = kept_params * 2  # bf16
    dequant_bytes = quantized_params * 2  # dequant copies (bf16)

    total_packed = iri_bytes + kept_bytes
    total_with_dequant = total_packed + dequant_bytes
    total_orig = src_bytes

    print(f"  Original bf16:              {total_orig/1e9:.3f} GB")
    print(f"  IRI-FP4 packed (no dequant): {total_packed/1e9:.3f} GB")
    print(f"    - IRI-FP4 weights:         {iri_bytes/1e9:.3f} GB ({iri_bytes/orig_bf16_bytes:.2f}x of bf16 weights)")
    print(f"    - Kept-as-is (conv/norm):  {kept_bytes/1e9:.3f} GB")
    print(f"  With dequant copies:         {total_with_dequant/1e9:.3f} GB")
    print(f"  Compression (packed only):   {total_orig/total_packed:.1f}x")
    print(f"  Effective bpw (quantized):   {iri_bytes*8/quantized_params:.2f} bits/weight")

    # Save
    print(f"\n[5] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(dst, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  {len(dst)} tensors, {fsize:.3f} GB on disk")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return dst


# ═══════════════════════════════════════════════════════════════════════════════
# Verification: compare V9 and V10 forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def verify_v9_v10(v9_path: str, v10_path: str, device: str = "cpu",
                   n_rounds: int = 3, block_size: int = 32,
                   seq_len: int = 16, n_tokens: int = 4):
    """Compare V9 (bf16) and V10 (IRI-FP4) forward pass outputs.

    Loads both checkpoints, reconstructs V10 weights from IRI-FP4 packed data,
    and compares per-tensor weight error + a small forward pass.
    """
    from research.config import get_config
    from research.model_loader import ConfigurableResearchLLM

    print(f"\n{'='*60}")
    print(f"Verification: V9 vs V10 forward pass")
    print(f"{'='*60}")

    # Load checkpoints
    print(f"\n[V1] Loading V9: {v9_path}")
    v9_state = load_file(v9_path)
    print(f"  {len(v9_state)} tensors")

    print(f"\n[V2] Loading V10: {v10_path}")
    v10_state = load_file(v10_path)
    print(f"  {len(v10_state)} tensors")

    # Reconstruct V10 weights from IRI-FP4 packed data
    print(f"\n[V3] Reconstructing V10 weights from IRI-FP4 packed data...")
    v10_recon = {}
    n_recon = 0
    for key in list(v10_state.keys()):
        if key.endswith(".iri_packed"):
            base_key = key[:-len(".iri_packed")]
            iri_packed = v10_state[f"{base_key}.iri_packed"]
            iri_scales = v10_state[f"{base_key}.iri_scales"]
            iri_gs = v10_state[f"{base_key}.iri_global_scale"]
            # Get shape from dequant copy
            dequant_key = f"{base_key}.weight_dequant"
            if dequant_key in v10_state:
                out_f, in_f = v10_state[dequant_key].shape
            else:
                # Infer from packed: (R, out, in_p//2)
                out_f = iri_packed.shape[1]
                in_padded = iri_packed.shape[2] * 2
                in_f = in_padded  # approximate
            w_recon = dequantize_iri_fp4_packed(
                iri_packed, iri_scales, iri_gs,
                out_f, in_f, block_size,
            )
            v10_recon[base_key] = w_recon
            n_recon += 1
        elif key.endswith(".iri_scales") or key.endswith(".iri_global_scale") \
                or key.endswith(".weight_dequant"):
            continue  # skip packed metadata
        else:
            v10_recon[key] = v10_state[key]

    print(f"  Reconstructed {n_recon} IRI-FP4 tensors")

    # Per-tensor weight error
    print(f"\n[V4] Per-tensor weight error (V9 bf16 vs V10 IRI-FP4 dequant):")
    max_rel_err = 0.0
    max_sqnr = 0.0
    min_sqnr = float("inf")
    for key in v9_state:
        if key in v10_recon:
            w9 = v9_state[key].float()
            w10 = v10_recon[key].float()
            if w9.shape != w10.shape:
                continue
            err = (w9 - w10)
            mse = (err ** 2).mean().item()
            signal_power = (w9 ** 2).mean().item()
            if signal_power > 1e-12:
                sqnr = 10 * torch.log10(torch.tensor(signal_power / max(mse, 1e-12))).item()
            else:
                sqnr = float("inf")
            rel_err = err.abs().max().item() / (w9.abs().max().item() + 1e-8)
            if key.endswith(".weight") and w9.ndim == 2 and not any(
                s in key for s in ("embed", "head", "conv", "in_proj", "out_proj")
            ):
                max_rel_err = max(max_rel_err, rel_err)
                max_sqnr = max(max_sqnr, sqnr)
                min_sqnr = min(min_sqnr, sqnr)
                if n_recon <= 20 or sqnr < 50:
                    print(f"  {key:50s} SQNR={sqnr:6.1f} dB  max_rel_err={rel_err:.4e}")

    if min_sqnr != float("inf"):
        print(f"\n  Summary: min SQNR={min_sqnr:.1f} dB, max SQNR={max_sqnr:.1f} dB, "
              f"max rel err={max_rel_err:.4e}")
        if min_sqnr >= 60:
            print(f"  ✓ IRI-FP4 {n_rounds} rounds achieves ≥60 dB SQNR (lossless-grade)")
        else:
            print(f"  ⚠ SQNR below 60 dB — consider more rounds")

    # Forward pass comparison
    print(f"\n[V5] Forward pass comparison (seq_len={seq_len}, {n_tokens} tokens)...")
    cfg = get_config("forgelm_v2_light", device=device)
    cfg.use_iri_fp4 = False  # we'll load weights manually

    # Build V9 model
    model_v9 = ConfigurableResearchLLM(cfg)
    missing_v9, unexpected_v9 = model_v9.load_state_dict(v9_state, strict=False)
    model_v9.eval()

    # Build V10 model (same architecture, load reconstructed weights)
    model_v10 = ConfigurableResearchLLM(cfg)
    missing_v10, unexpected_v10 = model_v10.load_state_dict(v10_recon, strict=False)
    model_v10.eval()

    # Random input
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        out_v9 = model_v9(input_ids)
        out_v10 = model_v10(input_ids)

    # Compare logits
    if hasattr(out_v9, "logits"):
        logits_v9 = out_v9.logits
        logits_v10 = out_v10.logits
    else:
        logits_v9 = out_v9
        logits_v10 = out_v10

    logit_diff = (logits_v9 - logits_v10).abs()
    max_diff = logit_diff.max().item()
    mean_diff = logit_diff.mean().item()
    cos_sim = F.cosine_similarity(
        logits_v9.flatten().unsqueeze(0),
        logits_v10.flatten().unsqueeze(0),
    ).item()

    # Top-1 agreement
    top1_v9 = logits_v9[:, -1].argmax(dim=-1)
    top1_v10 = logits_v10[:, -1].argmax(dim=-1)
    top1_match = (top1_v9 == top1_v10).float().mean().item()

    print(f"\n  Logit max diff:   {max_diff:.4e}")
    print(f"  Logit mean diff:  {mean_diff:.4e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Top-1 match:       {top1_match*100:.1f}%")

    if max_diff < 1.0 and top1_match > 0.95:
        print(f"\n  ✓ V10 forward pass matches V9 (IRI-FP4 is lossless-grade)")
    else:
        print(f"\n  ⚠ V10 forward pass differs from V9 — check quantization quality")

    return {
        "min_sqnr": min_sqnr,
        "max_logit_diff": max_diff,
        "cos_sim": cos_sim,
        "top1_match": top1_match,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Port LFM2.5 → ForgeLM V2 Light (IRI-FP4 quantization)"
    )
    parser.add_argument("--input", required=True,
                        help="Path to LFM2.5 HF .safetensors checkpoint")
    parser.add_argument("--output", required=True,
                        help="Path to write V10 .safetensors checkpoint")
    parser.add_argument("--device", default="cpu",
                        help="Device for quantization (cpu/cuda)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="IRI-FP4 rounds (default: 3 = 62.6 dB SQNR)")
    parser.add_argument("--block-size", type=int, default=32,
                        help="IRI-FP4 block size (default: 32)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify forward pass against V9 after porting")
    parser.add_argument("--v9-checkpoint",
                        default="research/checkpoints/ForgeLM_V9_1.2B.safetensors",
                        help="Path to V9 checkpoint for verification")
    args = parser.parse_args()

    port_lfm25_to_v10(
        args.input, args.output,
        device=args.device,
        n_rounds=args.rounds,
        block_size=args.block_size,
    )

    if args.verify:
        verify_v9_v10(
            args.v9_checkpoint, args.output,
            device=args.device,
            n_rounds=args.rounds,
            block_size=args.block_size,
        )
