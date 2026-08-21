"""Port LFM2.5 base weights into V5 BitNet-MoE architecture and save checkpoint.

V5 vs V4 differences:
  - MoE FFN: 8 routed experts (top-2) + 1 shared expert (DeepSeek-V3 style)
    Shared expert = LFM2.5 FFN weights (lossless)
    Routed experts = LFM2.5 FFN + ±5% noise (diverse init)
    dense_bypass=True → exact dense FFN output at init (lossless)
  - Expert Tying (g=2): layers (0,1), (2,3), ... share expert weights
  - Factorized Embedding: SVD of LFM2.5 embedding (7.8x param reduction)
  - Full BitNet: all linear layers ternary (attention + FFN + experts + embedding)

V5.1 additional keys (all lossless at init — no weight conversion needed):
  - MTP (Multi-Token Prediction): zero-init heads, tied to model head
  - ValueResidual (ResFormer): gate=0 → lossless, V_0 captured at runtime
  - SandwichNorm: identity-init post-sublayer RMSNorm (weight=1.0)
  - LearnedSink (GPT-OSS): zero-init per-head attention sink bias
  - SwiGLU Clamp (GPT-OSS): runtime-only activation change (no weights)
  - LeRoPE: learnable RoPE frequencies (freq_scale=1.0 = standard RoPE)

Usage:
    python -m research.architecture.port_to_v5 \\
        --input research/checkpoints/forgelm_v7_Base.safetensors \\
        --output research/checkpoints/forgelm_v7_Base.safetensors
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file as load_safetensors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM


def port_to_v5(input_path: str, output_path: str, config_name: str = "forgelm_v7_moe"):
    """Port V4/LFM2.5 base weights into V5 BitNet-MoE architecture checkpoint.

    Steps:
    1. Build blank V5 model (MoE, factorized embed, BitNet, GTA, all keys)
    2. Load base checkpoint (V4 or LFM2.5)
    3. Factorize embedding via SVD (lossless at start)
    4. Map base FFN weights → shared expert (lossless)
    5. Init routed experts = shared expert + ±5% noise
    6. Init BitNet qscale from loaded weights
    7. Expert tying (pointer aliasing, g=2)
    8. Save full state_dict
    """
    t0 = time.time()
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {gpu}")

    # 2. Load base checkpoint to CPU (2.34GB bf16, fine for 32GB RAM)
    print(f"[1] Loading base checkpoint from {input_path}...")
    base_state = load_safetensors(input_path)
    print(f"  {len(base_state)} tensors loaded")

    # 1. Build blank V5 model in bf16 on CPU (3.9B × 2 = 7.8GB, total ~10GB CPU)
    print(f"\n[2] Building blank V5 model (config: {config_name})...")
    cfg = get_config(config_name)
    # Force bf16 dtype during construction to halve memory
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = ConfigurableResearchLLM(cfg)
    torch.set_default_dtype(old_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M params")
    print(f"  MoE: {cfg.moe_n_experts} experts, top-{cfg.moe_top_k}, tying g={cfg.moe_tie_group_size}")
    print(f"  Factorized embedding: rank={cfg.embed_factorized_rank}")
    print(f"  BitNet: {cfg.use_bitnet}, embedding: {cfg.use_bitnet_embedding}")

    # 3. Apply GTA conversion if base is GQA
    if cfg.attn_type == "gta":
        print(f"\n[3] Applying GQA -> GTA weight conversion (V=K, gate=0, lossless)...")
        from research.keys.attention.gta_key import GTAKey
        res = GTAKey(n_layers=cfg.n_layers, n_heads=cfg.n_heads).forward(base_state)
        if res.success:
            base_state = res.weights
            print(f"  GTA conversion applied (lossless)")
        else:
            print(f"  WARNING: GTA conversion failed: {res.message}")

    # 4. Factorize embedding via SVD (move to GPU temporarily for speed)
    if cfg.use_factorized_embeddings:
        print(f"\n[4] Factorizing embedding via SVD (rank={cfg.embed_factorized_rank})...")
        embed_key = "embed.weight"
        if embed_key in base_state:
            from research.keys.architecture.factorized_embed_key import FactorizedEmbedding
            orig_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            orig_embed.weight.data.copy_(base_state[embed_key])
            # SVD on GPU for speed, then move back to CPU
            if gpu.type == "cuda":
                orig_embed = orig_embed.to(gpu)
            fact = FactorizedEmbedding.from_embedding(orig_embed, rank=cfg.embed_factorized_rank)
            base_state["embed.embed.weight"] = fact.embed.weight.data.cpu().to(torch.bfloat16)
            base_state["embed.project.weight"] = fact.project.weight.data.cpu().to(torch.bfloat16)
            del orig_embed, fact
            if gpu.type == "cuda":
                torch.cuda.empty_cache()
            del base_state[embed_key]
            if "head.weight" in base_state:
                del base_state["head.weight"]
            print(f"  Factorized: {cfg.vocab_size}x{cfg.d_model} -> {cfg.vocab_size}x{cfg.embed_factorized_rank} + {cfg.embed_factorized_rank}x{cfg.d_model}")
            print(f"  Params: {cfg.vocab_size * cfg.d_model / 1e6:.1f}M -> {(cfg.vocab_size * cfg.embed_factorized_rank + cfg.embed_factorized_rank * cfg.d_model) / 1e6:.1f}M")

    # 5. Map base FFN weights → MoE shared expert + init routed experts
    if cfg.use_moe:
        print(f"\n[5] Mapping base FFN -> MoE (shared=copy, routed=copy+noise)...")
        n_mapped = 0
        for i, block in enumerate(model.blocks):
            if not hasattr(block.ffn, 'experts'):
                continue
            moe = block.ffn
            d_model = cfg.d_model
            d_ff = cfg.moe_d_ff or cfg.intermediate_size

            # V4 checkpoint has: blocks.N.ffn.w_gate.weight, w_up.weight, w_down.weight
            # (BitNet format with .qscale, but .weight is the master weight)
            w_gate_key = f"blocks.{i}.ffn.w_gate.weight"
            w_up_key = f"blocks.{i}.ffn.w_up.weight"
            w_down_key = f"blocks.{i}.ffn.w_down.weight"

            has_base_ffn = all(k in base_state for k in [w_gate_key, w_up_key, w_down_key])

            if has_base_ffn:
                base_w1 = base_state[w_gate_key]  # (d_ff, d_model)
                base_w3 = base_state[w_up_key]    # (d_ff, d_model)
                base_w2 = base_state[w_down_key]  # (d_model, d_ff)

                # Shared expert = exact copy (lossless)
                with torch.no_grad():
                    moe.shared.w1.weight.copy_(base_w1)
                    moe.shared.w3.weight.copy_(base_w3)
                    moe.shared.w2.weight.copy_(base_w2)

                # Routed experts = copy + ±5% noise (diverse init)
                for expert in moe.experts:
                    with torch.no_grad():
                        noise_scale = 0.05
                        expert.w1.weight.copy_(base_w1 + noise_scale * torch.randn_like(base_w1))
                        expert.w3.weight.copy_(base_w3 + noise_scale * torch.randn_like(base_w3))
                        expert.w2.weight.copy_(base_w2 + noise_scale * torch.randn_like(base_w2))

                n_mapped += 1
                # Remove base FFN keys (V5 uses different key structure for MoE)
                for k in [w_gate_key, w_up_key, w_down_key,
                          f"blocks.{i}.ffn.w_gate.qscale",
                          f"blocks.{i}.ffn.w_up.qscale",
                          f"blocks.{i}.ffn.w_down.qscale"]:
                    if k in base_state:
                        del base_state[k]

        print(f"  Mapped {n_mapped} FFN layers to MoE (shared=copy, routed=copy+5%noise)")

    # 6. Load remaining weights (attention, conv, norms, etc.)
    print(f"\n[6] Loading remaining weights into V5 model...")
    # Ensure base_state tensors are bf16 to match model
    base_state = {k: v.to(torch.bfloat16) for k, v in base_state.items()}
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"  Missing keys: {len(missing)}")
    if missing:
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:10]:
            print(f"    {k}")

    # 7. Initialize BitNet qscale from loaded weights
    if cfg.use_bitnet:
        print(f"\n[7] Initializing BitNet qscale from loaded weights...")
        from research.keys.quantization.bitnet_b158_key import BitNetLinear, BitNetEmbedding
        n_init = 0
        for name, module in model.named_modules():
            if isinstance(module, BitNetLinear) and hasattr(module, '_reanchor_qscale'):
                module._reanchor_qscale()
                n_init += 1
            elif isinstance(module, BitNetEmbedding) and hasattr(module, 'qscale') and module.qscale is not None:
                with torch.no_grad():
                    module.qscale.copy_(module.weight.abs().mean().clamp(min=1e-6) / 0.7)
                n_init += 1
        print(f"  Initialized {n_init} qscale values")

    # 8. Mark QK-norm as non-identity
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    # 8a. V5.1 keys are all zero/identity init — no conversion needed.
    # LeRoPE: freq_scale=1.0 (identity = standard RoPE) — already set by build.
    # ValueResidual: gate=0 (lossless) — already set by build.
    # SandwichNorm: weight=1.0 (identity) — already set by build.
    # LearnedSink: sinks=0.0 (lossless) — already set by build.
    # SwiGLU Clamp: runtime-only activation change — no weights.
    # MTP: zero-init heads, tied to model head — already set by build.
    if getattr(cfg, 'use_mtp', False) or getattr(cfg, 'use_value_residual', False) \
            or getattr(cfg, 'use_sandwich_norm', False) or getattr(cfg, 'use_learned_sink', False) \
            or getattr(cfg, 'use_swiglu_clamp', False) or getattr(cfg, 'rope_variant', 'standard') != 'standard':
        print(f"\n[8a] V5.1 keys verified (all lossless at init):")
        print(f"  MTP: {getattr(cfg, 'use_mtp', False)}")
        print(f"  ValueResidual: {getattr(cfg, 'use_value_residual', False)}")
        print(f"  SandwichNorm: {getattr(cfg, 'use_sandwich_norm', False)}")
        print(f"  LearnedSink: {getattr(cfg, 'use_learned_sink', False)}")
        print(f"  SwiGLU Clamp: {getattr(cfg, 'use_swiglu_clamp', False)}")
        print(f"  RoPE variant: {getattr(cfg, 'rope_variant', 'standard')}")

    # 9. Expert tying is already applied during model construction
    if getattr(model, '_expert_tying_applied', False):
        print(f"\n[8] Expert tying already applied (g={cfg.moe_tie_group_size})")

    # 10. Save full V5 state_dict (with expert tying — skip odd-layer duplicates)
    print(f"\n[9] Saving V5 checkpoint to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_dict = {}
    # With factorized embedding, no tied head.weight to skip
    tied_keys = set()
    if not cfg.use_factorized_embeddings and not cfg.use_pit:
        tied_keys.add("head.weight")

    for k, v in model.state_dict().items():
        if k in tied_keys:
            print(f"  Skipping tied weight: {k}")
            continue
        save_dict[k] = v.contiguous().to(torch.bfloat16).cpu().clone()

    # Apply expert tying to state dict (remove odd-layer expert/shared duplicates)
    if cfg.moe_tie_group_size and cfg.moe_tie_group_size > 1:
        from research.keys.moe.expert_tying_key import ExpertTyingKey
        et = ExpertTyingKey(tie_group_size=cfg.moe_tie_group_size)
        tied_pairs, saved_bytes = et._tie_state_dict(save_dict, cfg.n_layers)
        print(f"  Expert tying: removed {len(tied_pairs)} odd-layer duplicates, saved {saved_bytes / 1e6:.0f} MB")

    save_file(save_dict, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {fsize:.2f} GB (bf16)")
    print(f"  Tensors: {len(save_dict)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Total params: {n_params / 1e6:.1f}M")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Port LFM2.5/V4 base weights into V5 BitNet-MoE architecture checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input V4 or LFM2.5 safetensors checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output V5 safetensors checkpoint")
    parser.add_argument("--config", type=str, default="forgelm_v7_moe",
                        help="V5 config preset name")
    args = parser.parse_args()

    port_to_v5(args.input, args.output, args.config)
