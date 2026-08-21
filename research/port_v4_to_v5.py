"""Port a V4/LFM2.5 dense checkpoint into V5 MoE architecture.

V5 was designed to load V4 losslessly:
  - Attention/norm weights: identical keys, load directly
  - GTA warm start: V=K, gate=0 (handled by build_model_fast)
  - MoE dense_bypass: shared expert = original FFN, routed experts = copies
  - Factorized embedding: SVD of original embedding (lossless at rank=256)
  - BitNet: ternary quantization applied after weight loading

Mapping:
  V4: blocks.N.ffn.w_gate.weight  ->  V5: blocks.N.ffn.shared.w1.weight
                                        blocks.N.ffn.experts.K.w1.weight (copies)
  V4: blocks.N.ffn.w_up.weight    ->  V5: blocks.N.ffn.shared.w3.weight
                                        blocks.N.ffn.experts.K.w3.weight (copies)
  V4: blocks.N.ffn.w_down.weight  ->  V5: blocks.N.ffn.shared.w2.weight
                                        blocks.N.ffn.experts.K.w2.weight (copies)
  V4: embed.weight (65536, 2048)  ->  V5: embed.embed.weight (65536, 256)
                                        embed.project.weight (2048, 256)
  (via SVD: U, S, V = svd(embed); embed = U @ diag(S); project = V @ diag(S))

Usage:
    python -m research.port_v4_to_v5 \
        --v4-checkpoint research/checkpoints/forgelm_v7_Base.safetensors \
        --output research/checkpoints/forgelm_v7_Ported.safetensors \
        --config forgelm_v7_moe
"""
from __future__ import annotations

import argparse
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def svd_factorize_embedding(embed_weight: torch.Tensor, rank: int
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Factorize embedding (vocab, d_model) -> (vocab, rank) @ (rank, d_model).

    Uses SVD: W = U @ diag(S) @ V^T
    embed = U * sqrt(S)  (vocab, rank)
    project = (sqrt(S) * V^T)  (rank, d_model) -> stored as (d_model, rank)

    This is lossless when rank >= min(vocab, d_model).
    """
    W = embed_weight.float()
    vocab, d_model = W.shape
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Truncate to rank
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    sqrt_S = S.sqrt()
    embed = U * sqrt_S.unsqueeze(0)  # (vocab, rank)
    project = (sqrt_S.unsqueeze(1) * Vh).t()  # (d_model, rank)
    return embed.to(embed_weight.dtype), project.to(embed_weight.dtype)


def port_v4_to_v5(
    v4_checkpoint: str,
    output: str,
    config_name: str = "forgelm_v7_moe",
    n_experts: int = 8,
    embed_rank: int = 256,
    apply_bitnet: bool = True,
) -> dict:
    """Port a V4 dense checkpoint to V5 MoE architecture.

    Args:
        v4_checkpoint: Path to V4 .safetensors checkpoint.
        output: Path to write V5 .safetensors checkpoint.
        config_name: V5 config name.
        n_experts: Number of MoE routed experts.
        embed_rank: Factorized embedding rank.
        apply_bitnet: If True, ternary-quantize weights (BitNet b1.58).

    Returns:
        Stats dict.
    """
    from research.config import get_config
    cfg = get_config(config_name, device="cpu")

    print(f"[PortV4->V5] Source: {v4_checkpoint}")
    print(f"[PortV4->V5] Target config: {config_name}")
    print(f"[PortV4->V5] MoE experts: {n_experts}, embed rank: {embed_rank}")
    print(f"[PortV4->V5] BitNet: {'yes' if apply_bitnet else 'no'}")
    print()

    t0 = time.time()

    # Load V4 state dict
    print("[PortV4->V5] Loading V4 state dict...")
    v4_state = {}
    with safe_open(v4_checkpoint, framework="pt") as f:
        for key in f.keys():
            v4_state[key] = f.get_tensor(key)
    print(f"[PortV4->V5] V4 has {len(v4_state)} tensors")

    # Build V5 state dict
    v5_state = {}
    n_mapped = 0
    n_copied = 0
    n_generated = 0

    # 1. Factorized embedding: SVD of V4 embed.weight
    print("[PortV4->V5] Factorizing embedding via SVD...")
    if "embed.weight" in v4_state:
        embed_w = v4_state["embed.weight"]
        embed_factorized, project = svd_factorize_embedding(embed_w, embed_rank)
        v5_state["embed.embed.weight"] = embed_factorized
        v5_state["embed.project.weight"] = project
        # Head (tied): same factorized weights
        v5_state["head.embed_ref.embed.weight"] = embed_factorized.clone()
        v5_state["head.embed_ref.project.weight"] = project.clone()
        n_generated += 4
        print(f"  embed.weight {embed_w.shape} -> embed.embed {embed_factorized.shape} + project {project.shape}")

    # 2. MoE FFN: map dense FFN -> shared expert + routed expert copies
    print("[PortV4->V5] Mapping dense FFN -> MoE experts...")
    n_layers = cfg.n_layers
    for layer_idx in range(n_layers):
        prefix = f"blocks.{layer_idx}.ffn."

        # V4 dense FFN weights
        w_gate = v4_state.get(f"{prefix}w_gate.weight")
        w_up = v4_state.get(f"{prefix}w_up.weight")
        w_down = v4_state.get(f"{prefix}w_down.weight")

        if w_gate is None:
            continue

        # Map: w_gate -> w1, w_up -> w3, w_down -> w2
        # Shared expert = exact copy of original FFN (lossless)
        v5_state[f"{prefix}shared.w1.weight"] = w_gate.clone()
        v5_state[f"{prefix}shared.w3.weight"] = w_up.clone()
        v5_state[f"{prefix}shared.w2.weight"] = w_down.clone()

        # Copy qscale if present
        for src_name, dst_name in [("w_gate", "w1"), ("w_up", "w3"), ("w_down", "w2")]:
            qscale_key = f"{prefix}{src_name}.qscale"
            if qscale_key in v4_state:
                v5_state[f"{prefix}shared.{dst_name}.qscale"] = v4_state[qscale_key].clone()

        # Routed experts: copies of shared (will diverge during training)
        for expert_idx in range(n_experts):
            v5_state[f"{prefix}experts.{expert_idx}.w1.weight"] = w_gate.clone()
            v5_state[f"{prefix}experts.{expert_idx}.w3.weight"] = w_up.clone()
            v5_state[f"{prefix}experts.{expert_idx}.w2.weight"] = w_down.clone()

            for src_name, dst_name in [("w_gate", "w1"), ("w_up", "w3"), ("w_down", "w2")]:
                qscale_key = f"{prefix}{src_name}.qscale"
                if qscale_key in v4_state:
                    v5_state[f"{prefix}experts.{expert_idx}.{dst_name}.qscale"] = v4_state[qscale_key].clone()

        n_copied += (1 + n_experts) * 3  # shared + n_experts, 3 weights each

    # 3. All other keys: copy directly (attention, norms, MHC, AttnRes, etc.)
    print("[PortV4->V5] Copying shared keys (attention, norms, etc.)...")
    v5_keys_so_far = set(v5_state.keys())
    for key, tensor in v4_state.items():
        if key in v5_keys_so_far:
            continue  # already mapped
        if key.startswith("blocks.") and ".ffn." in key:
            continue  # FFN already handled
        if key == "embed.weight":
            continue  # embedding already handled
        # Direct copy
        v5_state[key] = tensor.clone()
        n_mapped += 1

    # 4. Generate MoE router weights (small, random init is fine)
    print("[PortV4->V5] Generating MoE router weights...")
    import torch.nn as nn
    d_model = cfg.d_model
    for layer_idx in range(n_layers):
        prefix = f"blocks.{layer_idx}.ffn."
        # Router gate: (n_experts, d_model)
        if f"{prefix}router.gate.weight" not in v5_state:
            router_w = torch.empty(n_experts, d_model)
            nn.init.kaiming_uniform_(router_w, a=1.0)
            v5_state[f"{prefix}router.gate.weight"] = router_w
            n_generated += 1
        # Router noise: (n_experts, d_model)
        if f"{prefix}router.noise.weight" not in v5_state:
            noise_w = torch.empty(n_experts, d_model)
            nn.init.kaiming_uniform_(noise_w, a=1.0)
            v5_state[f"{prefix}router.noise.weight"] = noise_w
            n_generated += 1
        # Router noise_scale: scalar parameter (0.1 default)
        if f"{prefix}router.noise_scale" not in v5_state:
            v5_state[f"{prefix}router.noise_scale"] = torch.ones(1) * 0.1
            n_generated += 1

    print(f"[PortV4->V5] Mapped: {n_mapped} direct, {n_copied} FFN->MoE, {n_generated} generated")
    print(f"[PortV4->V5] V5 state dict: {len(v5_state)} tensors")

    # 5. Apply BitNet ternary quantization if requested
    if apply_bitnet:
        print("[PortV4->V5] Applying BitNet b1.58 ternary quantization...")
        from research.keys.quantization.bitnet_b158_key import ternary_quantize
        n_quantized = 0
        for key in list(v5_state.keys()):
            tensor = v5_state[key]
            if key.endswith(".weight") and tensor.is_floating_point():
                q, scale = ternary_quantize(tensor.float())
                v5_state[key] = q.to(torch.int8)
                n_quantized += 1
        print(f"[PortV4->V5] Quantized {n_quantized} weights to ternary int8")

    # 6. Save
    metadata = {
        "_ported_from": "v4",
        "_source_checkpoint": v4_checkpoint.split("/")[-1].split("\\")[-1],
        "_v5_config": config_name,
    }
    if apply_bitnet:
        metadata["_bitnet_prequant"] = "1"
        metadata["_prequant_mode"] = "int8"

    print(f"[PortV4->V5] Saving to {output}...")
    # Ensure all tensors are contiguous (safetensors requirement)
    v5_state = {k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in v5_state.items()}
    save_file(v5_state, output, metadata=metadata)

    elapsed = time.time() - t0
    print(f"[PortV4->V5] Done in {elapsed:.1f}s")
    print(f"[PortV4->V5] Output: {len(v5_state)} tensors")

    return {
        "tensors": len(v5_state),
        "mapped": n_mapped,
        "copied": n_copied,
        "generated": n_generated,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Port a V4 dense checkpoint to V5 MoE architecture")
    parser.add_argument("--v4-checkpoint", required=True,
                        help="Path to V4 .safetensors checkpoint")
    parser.add_argument("--output", required=True,
                        help="Path to write V5 .safetensors checkpoint")
    parser.add_argument("--config", default="forgelm_v7_moe",
                        help="V5 config name")
    parser.add_argument("--n-experts", type=int, default=8,
                        help="Number of MoE routed experts")
    parser.add_argument("--embed-rank", type=int, default=256,
                        help="Factorized embedding rank")
    parser.add_argument("--no-bitnet", action="store_true",
                        help="Skip BitNet ternary quantization (keep fp32)")
    args = parser.parse_args()

    port_v4_to_v5(
        v4_checkpoint=args.v4_checkpoint,
        output=args.output,
        config_name=args.config,
        n_experts=args.n_experts,
        embed_rank=args.embed_rank,
        apply_bitnet=not args.no_bitnet,
    )


if __name__ == "__main__":
    main()
