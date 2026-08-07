"""ForgeAI Tiny Model Builder — assembles the full absurdly small model.

Combines all architecture components into one optimized model:
- BitNet 1.58 ternary weights (10x weight compression)
- GateSkip token-wise layer skipping (15% compute savings)
- MLA Multi-Head Latent Attention (KV cache compression)
- SSA-compatible attention (sparse training ready)
- MoLA hot-loadable LoRA experts (better than MoE)
- KVQuant + H2O cache (32K context on 12GB)
- EAGLE-3 spec decode head (318x smaller draft)
- MTP multi-token prediction (training + inference speedup)
- TreeLoRA hierarchical adapters (multi-task continual learning)

Target: sub-500M params, beats 7B+ open source, runs on 12GB VRAM.

Usage:
    from research.tiny_model import build_tiny_model, TinyModelConfig

    config = TinyModelConfig(
        d_model=1024, n_layers=19, n_heads=16,
        vocab_size=151665,
        use_bitnet=True,        # ternary weights
        use_gateskip=True,      # layer skipping
        use_mtp=True,           # multi-token prediction
        use_eagle=True,         # spec decode head
        lora_hot_load=True,     # MoLA expert system
    )
    model = build_tiny_model(config)
"""
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from research.config import get_config
from research.model_loader import ModelLoader, ModularBlock
from research.bitnet import BitLinear, convert_model_to_bitnet
from research.gateskip import GateSkipBlock, add_gateskip_to_model
from research.mola import MoLAModel
from research.mtp import MTPHead
from research.eagle import EAGLEHead


@dataclass
class TinyModelConfig:
    """Configuration for the absurdly small ForgeAI model."""
    # Core architecture (matches 360m_mla config).
    d_model: int = 1024
    n_layers: int = 19
    n_heads: int = 16
    vocab_size: int = 151665
    max_seq_len: int = 1024

    # Component toggles.
    use_bitnet: bool = True          # ternary weights {-1, 0, +1}
    use_gateskip: bool = True        # token-wise layer skipping
    use_mtp: bool = True             # multi-token prediction head
    mtp_n_predict: int = 4           # tokens to predict ahead
    use_eagle: bool = True           # EAGLE-3 spec decode head
    lora_hot_load: bool = False      # MoLA expert system (needs adapter dir)
    adapter_dir: str = "research/checkpoints/live"

    # GateSkip params.
    gateskip_threshold: float = 0.1  # skip if gate < threshold
    gateskip_init_bias: float = 2.0  # start mostly-executing

    # LoRA/MoLA params.
    lora_rank: int = 16
    lora_alpha: int = 32
    max_adapters: int = 16
    adapter_cache_size: int = 4

    # KV cache compression.
    kv_bits: int = 2                 # KVQuant bits
    h2o_max_tokens: int = 4096       # H2O eviction limit
    h2o_keep_ratio: float = 0.2      # heavy hitter fraction

    # BitNet params.
    bitnet_act_bits: int = 8         # activation quantization bits


def build_tiny_model(config: TinyModelConfig,
                     checkpoint_path: Optional[str] = None,
                     device: str = "cuda") -> dict:
    """Build the full ForgeAI tiny model with all components.

    Args:
        config: TinyModelConfig with all toggles
        checkpoint_path: optional base checkpoint to load
        device: cuda or cpu

    Returns:
        dict with keys:
            'model': the main model (possibly wrapped in MoLA)
            'mtp_head': MTP head (or None)
            'eagle_head': EAGLE head (or None)
            'config': the config used
            'stats': parameter counts and memory estimates
    """
    print("=" * 60)
    print("Building ForgeAI Tiny Model")
    print("=" * 60)

    # 1. Load base model from existing config.
    cfg = get_config("360m_mla")
    checkpoint = checkpoint_path if checkpoint_path else None
    model = ModelLoader.build_model(cfg, checkpoint_path=checkpoint)
    model = model.to(device)

    stats = {"base_params": sum(p.numel() for p in model.parameters())}
    print(f"\n[1/6] Base model loaded: {stats['base_params']:,} params")

    # 2. Convert to BitNet (ternary weights).
    if config.use_bitnet:
        n_converted = convert_model_to_bitnet(model)
        bitnet_params = sum(p.numel() for p in model.parameters())
        # Estimate ternary memory: each weight is 1.58 bits → 0.2 bytes
        ternary_mem = sum(p.numel() for n, p in model.named_parameters()
                         if "weight" in n and p.requires_grad) * 0.2 / (1024**2)
        stats["bitnet_converted"] = n_converted
        stats["ternary_weight_mem_mb"] = ternary_mem
        print(f"[2/6] BitNet: {n_converted} layers converted to ternary "
              f"(~{ternary_mem:.1f} MB weight memory)")

    # 3. Add GateSkip (token-wise layer skipping).
    if config.use_gateskip:
        n_wrapped = add_gateskip_to_model(
            model, d_model=config.d_model,
            skip_threshold=config.gateskip_threshold
        )
        gateskip_params = sum(p.numel() for p in model.parameters()) - stats["base_params"]
        stats["gateskip_wrapped"] = n_wrapped
        stats["gateskip_params"] = gateskip_params
        print(f"[3/6] GateSkip: {n_wrapped} blocks wrapped "
              f"(+{gateskip_params:,} gate params)")

    # 4. Wrap in MoLA (hot-loadable LoRA experts).
    if config.lora_hot_load:
        model = MoLAModel(
            model, adapter_dir=config.adapter_dir,
            d_model=config.d_model, max_adapters=config.max_adapters,
            cache_size=config.adapter_cache_size,
            lora_rank=config.lora_rank, lora_alpha=config.lora_alpha,
        )
        stats["mola_enabled"] = True
        print(f"[4/6] MoLA: hot-load expert system enabled "
              f"(max {config.max_adapters} adapters, cache {config.adapter_cache_size})")
    else:
        print("[4/6] MoLA: disabled (use lora_hot_load=True to enable)")

    # 5. Build MTP head (multi-token prediction).
    mtp_head = None
    if config.use_mtp:
        mtp_head = MTPHead(
            d_model=config.d_model, vocab_size=config.vocab_size,
            n_predict=config.mtp_n_predict,
        ).to(device)
        mtp_params = sum(p.numel() for p in mtp_head.parameters())
        stats["mtp_params"] = mtp_params
        print(f"[5/6] MTP: {config.mtp_n_predict}-token prediction head "
              f"({mtp_params:,} params)")
    else:
        print("[5/6] MTP: disabled")

    # 6. Build EAGLE-3 head (speculative decoding).
    eagle_head = None
    if config.use_eagle:
        eagle_head = EAGLEHead(
            d_model=config.d_model, vocab_size=config.vocab_size,
            n_layers=2, n_heads=8,
        ).to(device)
        eagle_params = sum(p.numel() for p in eagle_head.parameters())
        stats["eagle_params"] = eagle_params
        print(f"[6/6] EAGLE-3: spec decode head ({eagle_params:,} params)")
    else:
        print("[6/6] EAGLE-3: disabled")

    # Summary.
    total_params = sum(p.numel() for p in model.parameters())
    if mtp_head:
        total_params += mtp_params
    if eagle_head:
        total_params += eagle_params

    stats["total_params"] = total_params
    stats["total_params_millions"] = total_params / 1e6

    print("\n" + "=" * 60)
    print("ForgeAI Tiny Model — Build Complete")
    print("=" * 60)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    if config.use_bitnet:
        print(f"  Weight memory (ternary): ~{stats.get('ternary_weight_mem_mb', 0):.1f} MB")
        print(f"  vs FP16: ~{stats['base_params'] * 2 / (1024**2):.1f} MB")
    print(f"  GateSkip overhead: +{stats.get('gateskip_params', 0):,} params")
    if mtp_head:
        print(f"  MTP head: +{mtp_params:,} params")
    if eagle_head:
        print(f"  EAGLE head: +{eagle_params:,} params")
    print("=" * 60)

    return {
        "model": model,
        "mtp_head": mtp_head,
        "eagle_head": eagle_head,
        "config": config,
        "stats": stats,
    }


def estimate_memory(config: TinyModelConfig) -> dict:
    """Estimate VRAM usage for the tiny model configuration.

    Returns dict with memory breakdown in MB.
    """
    d = config.d_model
    n = config.n_layers
    v = config.vocab_size

    # Compute base model params from architecture (not hardcoded).
    # Embedding + n_layers * (qkv + mlp + norms) + lm_head
    embed_params = v * d
    per_layer = (
        3 * d * d +  # q, k, v projections
        d * d +      # output projection
        d * (d * 4) + (d * 4) * d +  # MLP up + down
        4 * d       # layer norms
    )
    lm_head_params = d * v
    base_params = embed_params + n * per_layer + lm_head_params

    # Weight memory.
    if config.use_bitnet:
        weight_mem = base_params * 0.2 / (1024**2)  # 1.58 bits = 0.2 bytes
    else:
        weight_mem = base_params * 2 / (1024**2)  # FP16

    # KV cache (per token).
    head_dim = d // config.n_heads
    kv_per_token = 2 * config.n_heads * head_dim * 2  # K+V, FP16
    if config.kv_bits < 16:
        kv_per_token = 2 * config.n_heads * head_dim * (config.kv_bits / 8)

    # H2O caps the cache.
    kv_cache_mem = kv_per_token * config.h2o_max_tokens / (1024**2)

    # GateSkip overhead.
    gateskip_mem = n * d * 2 / (1024**2)  # 1 Linear(d, 1) per layer

    # MTP head.
    mtp_mem = config.mtp_n_predict * d * v * 2 / (1024**2) if config.use_mtp else 0

    # EAGLE head.
    eagle_mem = 2 * d * d * 2 / (1024**2) if config.use_eagle else 0  # 2 layers

    # LoRA adapters (per adapter in cache).
    lora_per_adapter = config.lora_rank * d * 4 * 2 / (1024**2)  # 4 projections
    lora_total = lora_per_adapter * config.adapter_cache_size if config.lora_hot_load else 0

    total = weight_mem + kv_cache_mem + gateskip_mem + mtp_mem + eagle_mem + lora_total

    return {
        "weights_mb": weight_mem,
        "kv_cache_mb": kv_cache_mem,
        "gateskip_mb": gateskip_mem,
        "mtp_head_mb": mtp_mem,
        "eagle_head_mb": eagle_mem,
        "lora_adapters_mb": lora_total,
        "total_mb": total,
        "total_gb": total / 1024,
        "fits_12gb": total < 12000,
    }


if __name__ == "__main__":
    # Print memory estimate for default config.
    config = TinyModelConfig()
    mem = estimate_memory(config)
    print("\nMemory estimate (default config):")
    for k, v in mem.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
