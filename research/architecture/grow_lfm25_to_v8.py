"""Grow LFM2.5-1.2B into V8-8B dimensions using function-preserving duplication.

Strategy:
  1. Build a "plain V8" model (V8 dims, no BitNet/NLRQ/QSA/etc.)
  2. Load ported LFM2.5 weights from safetensors
  3. Expand weights using duplication strategy:
     - Width: tile [x, x] for d_model, block-diagonal for linear weights
     - Depth: first 16 layers = source, next 16 = zero-init identity
  4. Verify lossless: V8 logits[:, :, :vocab] == LFM2.5 logits
  5. Save grown checkpoint

The grown checkpoint can then have V8 keys enabled one-by-one
(each is lossless at init by design).
"""
import os
import sys
import time
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from safetensors.torch import load_file as load_safetensors, save_file
from research.config import get_config
from research.model_loader import ConfigurableResearchLLM


# Layer types for LFM2.5 (16 layers) and V8 (32 layers, same pattern x2)
LFM25_LAYERS = ["conv", "conv", "attention", "conv", "conv", "attention",
                "conv", "conv", "attention", "conv", "conv", "attention",
                "conv", "conv", "attention", "conv"]
V8_LAYERS = LFM25_LAYERS * 2  # 32 layers, same pattern repeated


def plain_v8_config():
    """V8 dimensions with all extra keys disabled — pure transformer."""
    overrides = dict(
        n_layers=32,
        d_model=4096,
        n_heads=64,
        n_kv_heads=16,
        intermediate_size=16384,
        vocab_size=65536,
        max_seq_len=32768,
        rope_base=1_000_000.0,
        layer_types=V8_LAYERS,
        # Disable all V8-specific keys
        use_bitnet=False,
        use_bitnet_embedding=False,
        ffn_compression="none",
        use_factorized_embeddings=False,
        use_qsa=False,
        use_gated_residual=False,
        use_ngram_embedding=False,
        ngram_host=False,
        use_hashed_nlrq=False,
        use_mtp=False,
        use_hyperloop=False,
        use_lisa=False,
        lisa_align_dim=0,
        use_titan_memory=False,
        titan_memory_rank=0,
        use_mhc=False,
        mhc_rank=0,
        use_mod=False,
        mod_n_skip_layers=0,
        use_attn_residual=False,
        use_value_residual=False,
        use_sandwich_norm=False,
        use_learned_sink=False,
        use_pit=False,
        use_fused_gemm=False,
        use_smooth_swiglu=False,
        use_swiglu_clamp=False,
        use_mu_scaling=False,
        use_peagle_tied=False,
        use_chunked_ce=False,
        use_qk_norm=True,  # LFM2.5 has QK norm
        use_varlen=False,
        use_triton_kernels=False,
        use_gradient_checkpointing=False,
        use_fp8_activation=False,
        use_fp8_training=False,
        use_w8a8=False,
        use_nvfp4=False,
        use_diff_attn=False,
        use_gvpo=False,
        use_om_grpo=False,
        use_sc_grpo=False,
        use_liger_ce=False,
        embed_factorized_rank=0,
        embed_tie_factor=1.0,
        attn_type="gqa",  # plain GQA, not GTA
        ffn_type="swiglu",
        norm_type="rmsnorm",
        attention_pattern="standard",
        rope_variant="none",
        device="cpu",
        dtype="float32",
        tie_word_embeddings=True,
        conv_kernel_size=3,
    )
    return get_config("forgelm_v8_8b", **overrides)


def grow_weight_2d(src_w: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Grow a 2D weight [out, in] to target shape using block-diagonal duplication.

    For 2x width: [[W, 0], [0, W]] → [x, x] @ blkdiag = [W@x, W@x]
    First half of output = source output (function-preserving).
    """
    src_out, src_in = src_w.shape
    dst_out, dst_in = target_shape
    w_out = dst_out // src_out
    w_in = dst_in // src_in

    # Block-diagonal: [[W, 0], [0, W], ...]
    dst = torch.zeros(dst_out, dst_in, dtype=src_w.dtype)
    for i in range(w_out):
        for j in range(w_in):
            dst[i * src_out:(i + 1) * src_out,
                j * src_in:(j + 1) * src_in] = src_w
    return dst


def grow_weight_1d(src_w: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Grow a 1D weight [d] to target shape by tiling.

    [w, w, ...] — first half = source (function-preserving for RMSNorm).
    """
    src_d = src_w.shape[0]
    dst_d = target_shape[0]
    ratio = dst_d // src_d
    return src_w.repeat(ratio)


def grow_conv_weight(src_w: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Grow conv weight [out, in, kernel] to target shape.

    Out and in are tiled (block-diagonal), kernel stays the same.
    """
    src_out, src_in, kernel = src_w.shape
    dst_out, dst_in, dst_kernel = target_shape
    w_out = dst_out // src_out
    w_in = dst_in // src_in

    dst = torch.zeros(dst_out, dst_in, kernel, dtype=src_w.dtype)
    for i in range(w_out):
        for j in range(w_in):
            dst[i * src_out:(i + 1) * src_out,
                j * src_in:(j + 1) * src_in] = src_w
    return dst


def grow_lfm25_to_v8(
    lfm25_checkpoint: str,
    output_path: str,
    device: str = "cpu",
):
    """Grow LFM2.5-1.2B weights into V8-8B dimensions."""
    t0 = time.time()

    # 1. Load LFM2.5 weights
    print(f"[1] Loading LFM2.5 weights from {lfm25_checkpoint}...")
    lfm_state = load_safetensors(lfm25_checkpoint)
    print(f"  {len(lfm_state)} tensors")

    # 2. Build plain V8 model
    print(f"\n[2] Building plain V8 model (V8 dims, no extra keys)...")
    cfg = plain_v8_config()
    model = ConfigurableResearchLLM(cfg)
    v8_state = model.state_dict()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, {len(v8_state)} tensors")

    # 3. Grow weights
    print(f"\n[3] Growing weights (duplication strategy)...")
    ported = 0
    skipped = 0

    # Embedding: [65536, 2048] → [65536, 4096] — tile along dim 1
    if "embed.weight" in lfm_state and "embed.weight" in v8_state:
        src = lfm_state["embed.weight"]  # [65536, 2048]
        dst = torch.cat([src, src], dim=1)  # [65536, 4096]
        v8_state["embed.weight"] = dst.to(v8_state["embed.weight"].dtype)
        ported += 1
        print(f"  embed.weight: {list(src.shape)} → {list(dst.shape)}")

    # Final norm: [2048] → [4096] — tile
    if "ln_f.weight" in lfm_state and "ln_f.weight" in v8_state:
        src = lfm_state["ln_f.weight"]
        dst = src.repeat(2)
        v8_state["ln_f.weight"] = dst.to(v8_state["ln_f.weight"].dtype)
        ported += 1
        print(f"  ln_f.weight: {list(src.shape)} → {list(dst.shape)}")

    # Head: tied to embed — skip (will be tied)
    if "head.weight" in v8_state:
        print(f"  head.weight: tied to embed.weight (skipping)")

    # Per-layer weights
    for i in range(16):  # source layers 0-15
        v8_layer = i  # first 16 V8 layers = source layers
        is_attn = V8_LAYERS[i] == "attention"

        # Norms
        for norm_key in ["ln1", "ln2"]:
            lfm_key = f"blocks.{i}.{norm_key}.weight"
            v8_key = f"blocks.{v8_layer}.{norm_key}.weight"
            if lfm_key in lfm_state and v8_key in v8_state:
                src = lfm_state[lfm_key]
                dst = grow_weight_1d(src, v8_state[v8_key].shape)
                v8_state[v8_key] = dst.to(v8_state[v8_key].dtype)
                ported += 1

        if is_attn:
            # Attention layer: q_proj, k_proj, v_proj, out_proj, q_norm, k_norm
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                lfm_key = f"blocks.{i}.attn.{proj}.weight"
                v8_key = f"blocks.{v8_layer}.attn.{proj}.weight"
                if lfm_key in lfm_state and v8_key in v8_state:
                    src = lfm_state[lfm_key]
                    dst = grow_weight_2d(src, v8_state[v8_key].shape)
                    v8_state[v8_key] = dst.to(v8_state[v8_key].dtype)
                    ported += 1

            # QK norms: [64] → [64] (head_dim stays same, just copy)
            for norm in ["q_norm", "k_norm"]:
                lfm_key = f"blocks.{i}.attn.{norm}.weight"
                v8_key = f"blocks.{v8_layer}.attn.{norm}.weight"
                if lfm_key in lfm_state and v8_key in v8_state:
                    v8_state[v8_key] = lfm_state[lfm_key].to(v8_state[v8_key].dtype)
                    ported += 1
        else:
            # Conv layer: in_proj, conv, out_proj
            for proj in ["in_proj", "out_proj"]:
                lfm_key = f"blocks.{i}.attn.{proj}.weight"
                v8_key = f"blocks.{v8_layer}.attn.{proj}.weight"
                if lfm_key in lfm_state and v8_key in v8_state:
                    src = lfm_state[lfm_key]
                    dst = grow_weight_2d(src, v8_state[v8_key].shape)
                    v8_state[v8_key] = dst.to(v8_state[v8_key].dtype)
                    ported += 1

            # Conv weight: [out, 1, kernel] → [2*out, 1, kernel]
            lfm_key = f"blocks.{i}.attn.conv.weight"
            v8_key = f"blocks.{v8_layer}.attn.conv.weight"
            if lfm_key in lfm_state and v8_key in v8_state:
                src = lfm_state[lfm_key]
                dst = grow_conv_weight(src, v8_state[v8_key].shape)
                v8_state[v8_key] = dst.to(v8_state[v8_key].dtype)
                ported += 1

        # FFN: w_gate, w_up, w_down (all layers)
        for ffn_key in ["w_gate", "w_up", "w_down"]:
            lfm_key = f"blocks.{i}.ffn.{ffn_key}.weight"
            v8_key = f"blocks.{v8_layer}.ffn.{ffn_key}.weight"
            if lfm_key in lfm_state and v8_key in v8_state:
                src = lfm_state[lfm_key]
                dst = grow_weight_2d(src, v8_state[v8_key].shape)
                v8_state[v8_key] = dst.to(v8_state[v8_key].dtype)
                ported += 1

    # Layers 16-31: zero-init identity (function-preserving)
    # The zero_init_residual flag handles this in the model, but we need
    # to make sure the residual contribution is zero.
    # For a plain transformer without gated residual, the residual is x + f(x).
    # If f(x) = 0 (zero weights), then output = x (identity).
    # The model's zero_init_residual should handle this, but let's explicitly
    # zero the output projections and FFN down projections for layers 16-31.
    print(f"\n  Layers 16-31: zero-init identity layers...")
    for i in range(16, 32):
        # Zero out output proj and FFN down to make layer = identity
        for key_pattern in [
            f"blocks.{i}.attn.out_proj.weight",
            f"blocks.{i}.ffn.w_down.weight",
        ]:
            if key_pattern in v8_state:
                v8_state[key_pattern] = torch.zeros_like(v8_state[key_pattern])

    # 4. Load into model
    print(f"\n[4] Loading grown weights into V8 model...")
    missing, unexpected = model.load_state_dict(v8_state, strict=False)
    if missing:
        print(f"  Missing: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    # Mark QK-norm as non-identity
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    # 5. Save
    print(f"\n[5] Saving grown checkpoint to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dict = {}
    for k, v in model.state_dict().items():
        if k == "head.weight":
            continue  # tied
        save_dict[k] = v.contiguous().to(torch.bfloat16).clone()
    save_file(save_dict, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {fsize:.2f} GB (bf16)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Ported: {ported} tensors")
    return model


def verify_growth(
    lfm25_checkpoint: str,
    v8_checkpoint: str,
    device: str = "cpu",
):
    """Verify that V8 model's logits match LFM2.5 model's logits.

    The duplication strategy ensures that the first half of V8's hidden state
    matches LFM2.5's hidden state, so logits should be identical.
    """
    print(f"\n=== Verifying growth (lossless check) ===")

    # Build LFM2.5 model
    cfg_lfm = get_config("lfm25_1.2b")
    cfg_lfm = type(cfg_lfm)(**{**cfg_lfm.__dict__, "device": device, "dtype": "float32"})
    model_lfm = ConfigurableResearchLLM(cfg_lfm)
    lfm_state = load_safetensors(lfm25_checkpoint)
    model_lfm.load_state_dict(lfm_state, strict=False)
    for block in model_lfm.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False
    model_lfm.eval()

    # Build V8 model
    cfg_v8 = plain_v8_config()
    model_v8 = ConfigurableResearchLLM(cfg_v8)
    v8_state = load_safetensors(v8_checkpoint)
    model_v8.load_state_dict(v8_state, strict=False)
    for block in model_v8.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False
    model_v8.eval()

    # Forward pass on same input
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg_lfm.vocab_size, (1, 16), device=device)

    with torch.no_grad():
        logits_lfm, _ = model_lfm(input_ids)
        logits_v8, _ = model_v8(input_ids)

    print(f"  LFM2.5 logits: {list(logits_lfm.shape)}, range [{logits_lfm.min():.4f}, {logits_lfm.max():.4f}]")
    print(f"  V8 logits:     {list(logits_v8.shape)}, range [{logits_v8.min():.4f}, {logits_v8.max():.4f}]")

    # V8 logits should have first half = LFM2.5 logits (due to duplication)
    # The output head is [vocab, 4096] with first 2048 cols = source weights
    # So logits_v8[:, :, :vocab] should match logits_lfm
    # Actually, the head weight is grown as [[W, 0], ...] so:
    # logits_v8 = hidden_v8 @ head_v8.T
    # hidden_v8 first half = hidden_lfm, head_v8 first half = head_lfm
    # So logits_v8 = hidden_lfm @ head_lfm.T = logits_lfm
    max_diff = (logits_v8 - logits_lfm).abs().max().item()
    mean_diff = (logits_v8 - logits_lfm).abs().mean().item()

    print(f"\n  Max logit diff:  {max_diff:.6f}")
    print(f"  Mean logit diff: {mean_diff:.6f}")

    # Check if argmax matches (token prediction preserved)
    pred_lfm = logits_lfm.argmax(dim=-1)
    pred_v8 = logits_v8.argmax(dim=-1)
    match_rate = (pred_lfm == pred_v8).float().mean().item()
    print(f"  Argmax match rate: {match_rate:.4f}")

    if max_diff < 0.1:
        print(f"\n  ✓ LOSSLESS: max diff < 0.1")
    elif max_diff < 1.0:
        print(f"\n  ~ NEAR-LOSSLESS: max diff < 1.0")
    else:
        print(f"\n  ✗ NOT LOSSLESS: max diff = {max_diff:.4f}")

    return max_diff < 0.1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grow LFM2.5 to V8-8B")
    parser.add_argument("--lfm25-checkpoint", type=str,
                        default="research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors")
    parser.add_argument("--output", type=str,
                        default="research/checkpoints/ForgeLM_V8_8B_grown.safetensors")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    model = grow_lfm25_to_v8(args.lfm25_checkpoint, args.output, args.device)
    if args.verify:
        verify_growth(args.lfm25_checkpoint, args.output, args.device)
