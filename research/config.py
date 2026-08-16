"""Config-driven model specifications for ForgeAI architecture research."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """All hyperparameters needed to build a ConfigurableResearchLLM."""

    # === Architecture ===
    vocab_size: int = 65536
    d_model: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    max_seq_len: int = 32768
    attn_type: str = "gqa"            # "gqa" (only supported type now)
    ffn_type: str = "swiglu"          # "swiglu" (only supported type now)
    norm_type: str = "rmsnorm"        # "rmsnorm" (only supported type now)
    n_kv_heads: int | None = None     # KV heads for GQA (None = MHA)
    intermediate_size: int | None = None  # FFN hidden dim (None = 8*d/3)
    attn_bias: bool = False
    rope_base: float = 1_000_000.0
    # YaRN RoPE scaling for context extension. None = no scaling.
    rope_scaling: dict | None = None
    # RoPE variant: "standard", "lerope", or "adarope" (both learnable variants
    # init to identity = standard RoPE, lossless at start).
    rope_variant: str = "standard"

    # === Weight sharing ===
    # Tie input embeddings to output head (default True for most small models).
    # Set False for models with separate embed/head (Qwen2.5, some larger models).
    tie_word_embeddings: bool = True

    # === Hybrid layer types ===
    # Per-layer type list. None = all attention (default).
    # Each entry: "attention" (or "attn"), "conv", "mamba", or "moe".
    layer_types: list[str] | None = None
    conv_kernel_size: int = 3

    # === Our keys (all zero/identity init = lossless at start) ===
    # QK-norm: RMSNorm on Q and K before RoPE (LFM2/Gemma3/Qwen3 style).
    use_qk_norm: bool = False
    # Zero-init output projections: attn.out_proj and ffn.w_down start at zero.
    zero_init_residual: bool = False
    # PIT (Pseudo-Inverse Tying): replaces weight tying with orthonormal shared
    # memory + learned SPD transform. Identity init = lossless.
    use_pit: bool = False
    # AttnRes (Attention Residuals, Kimi K3): cross-layer retrieval, gated.
    use_attn_residual: bool = False
    attn_res_k: int = 4
    # mHC (Manifold-Constrained Hyper-Connections, DeepSeek-V4): low-rank
    # residual projection. Gate init=0 = lossless.
    use_mhc: bool = False
    mhc_rank: int = 0  # 0 = auto (d_model // 4)
    # CSA (Compressed Sparse Attention, DeepSeek-V4): top-k position selection.
    attention_pattern: str = "standard"  # "standard", "csa", "csa_hca_hybrid"
    csa_top_k: int = 256
    # MTP (Multi-Token Prediction): shared-weight heads for speculative decode.
    use_mtp: bool = False
    mtp_n_heads: int = 2
    mtp_loss_weight: float = 0.3
    # BitNet b1.58: ternary QAT on all linear layers ({-1,0,1} weights, STE).
    # Training = ternary forward (STE, master weights fp); eval = full
    # precision unless bitnet_force_quant (deploy after QAT converges).
    use_bitnet: bool = False
    bitnet_force_quant: bool = False
    bitnet_learned_scale: bool = True  # per-layer learnable scale (QAT)
    # Differential Attention (Diff-Transformer): dual softmax maps subtracted
    # to cancel attention noise. Parameter-efficient; opt-in (not lossless).
    use_diff_attn: bool = False
    diff_attn_lambda_init: float | None = None  # None = paper formula
    # TITAN neural memory: gated long-term memory, zero-init gate = lossless.
    use_titan_memory: bool = False
    titan_memory_rank: int = 0  # 0 = d_model
    # Mixture-of-Depths: per-block token routing; keep_fraction=1.0 = lossless
    # (all tokens processed). <1.0 skips tokens in training (FLOPs ceiling).
    use_mod: bool = False
    mod_keep_fraction: float = 1.0

    # === Training ===
    dropout: float = 0.0
    device: str = "cpu"
    dtype: str = "bfloat16"
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data"
    batch_size: int = 2
    seq_len: int = 1024
    max_steps: int = 500
    warmup_steps: int = 200
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 250
    save_every: int = 500
    eval_batches: int = 10

    # === Memory optimizations ===
    # Chunked cross-entropy: fuses head + CE without materializing full logits.
    use_chunked_ce: bool = False
    ce_chunk_size: int = 256
    # Liger-Kernel fused linear cross-entropy (requires liger-kernel).
    use_liger_ce: bool = False
    # Gradient checkpointing: recompute forward during backward to save VRAM.
    use_gradient_checkpointing: bool = False
    # Selective checkpoint strategy: "all" (full block), "ffn" (recompute only
    # the FFN — largest activation consumer, minimal compute penalty), "attn",
    # "none". Only applies when use_gradient_checkpointing is True.
    selective_gradient_checkpointing: str = "all"
    # Fixed attention scale (0.12 from NanoGPT speedrun) instead of head_dim**-0.5.
    attn_scale: float = None

    # === Training extensions ===
    # OEC (Output Embedding Centering): suppresses anisotropic common-mode shift.
    oec_mode: str = "none"
    oec_lambda: float = 1e-3
    # SC-GRPO / OM-GRPO / GVPO: RL training extensions.
    use_sc_grpo: bool = False
    use_om_grpo: bool = False
    use_gvpo: bool = False
    gvpo_lambda: float = 0.3

    # === Legacy compat (kept for keys that reference these) ===
    kv_compression_dim: int = 128  # used by MLA keys (no longer in core)
    enable_draft_head: bool = False

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).")


# Pre-defined architecture targets.
MODEL_CONFIGS = {
    # LFM2.5-1.2B-Instruct port: exact architectural match to LiquidAI's model.
    #   - 16 layers: 10 double-gated conv + 6 GQA attention (layers 2,5,8,10,12,14)
    #   - d_model=2048, 32 heads, 8 KV heads (GQA 4x), head_dim=64
    #   - SwiGLU FFN (intermediate=8192), RMSNorm, QK-layernorm on attention
    #   - RoPE theta=1M, 128K context (32K for VRAM budget)
    #   - Vocab=65536, tied embeddings
    # Weights ported directly from LFM2.5-1.2B-Instruct safetensors — 100%
    # weight preservation, no SVD resize, no dimension mismatch.
    # Our keys (mHC, MTP, Safety, PIT) plug in with zero/identity init.
    "lfm25_1.2b": ModelConfig(
        vocab_size=65536,
        d_model=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        intermediate_size=8192,
        attn_type="gqa",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        conv_kernel_size=3,
        use_qk_norm=True,
        layer_types=["conv", "conv", "attention", "conv", "conv", "attention",
                     "conv", "conv", "attention", "conv", "attention", "conv",
                     "attention", "conv", "attention", "conv"],
        batch_size=2,
        seq_len=1024,
    ),
    # ForgeLM V3 — labeled evolution of the LFM2.5 port with the full
    # 2025/2026 architecture stack:
    #   - Differential Attention (identity warm start; GQA checkpoints
    #     auto-convert losslessly at load)
    #   - BitNet b1.58 QAT on the FFN (ternary in training, fp eval)
    #   - TITAN neural memory (low-rank, zero-init gate)
    #   - Mixture-of-Depths router (keep_fraction=1.0 = lossless start)
    # ALL mechanisms are lossless at load: ForgeLM_V2_BSP.safetensors loads
    # bit-exact (verified max logit diff 0.0, incl. KV-cached decode).
    "forgelm_v3": ModelConfig(
        vocab_size=65536,
        d_model=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        intermediate_size=8192,
        attn_type="diff",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        conv_kernel_size=3,
        use_qk_norm=True,
        use_bitnet=True,
        bitnet_learned_scale=True,
        layer_types=["conv", "conv", "attention", "conv", "conv", "attention",
                     "conv", "conv", "attention", "conv", "attention", "conv",
                     "attention", "conv", "attention", "conv"],
        use_titan_memory=True,
        titan_memory_rank=64,
        use_mod=True,
        mod_keep_fraction=1.0,
        batch_size=2,
        seq_len=1024,
    ),
    # Qwen2.5-3B-Instruct: standard GQA transformer.
    #   - 36 layers, all attention (no conv/mamba)
    #   - d_model=2048, 16 Q heads, 2 KV heads (GQA 8x), head_dim=128
    #   - SwiGLU FFN (intermediate=11008), RMSNorm
    #   - RoPE theta=1M, 32K context, Q/K/V projection bias
    #   - Vocab=151936, no tied embeddings
    "qwen25_3b": ModelConfig(
        vocab_size=151936,
        d_model=2048,
        n_layers=36,
        n_heads=16,
        n_kv_heads=2,
        intermediate_size=11008,
        attn_type="gqa",
        attn_bias=True,  # Qwen2.5 uses bias in Q/K/V projections
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        use_qk_norm=False,
        layer_types=["attention"] * 36,  # all attention, no conv
        tie_word_embeddings=True,  # Qwen2.5-3B ties embed_tokens to lm_head
        batch_size=2,
        seq_len=1024,
    ),
    # Tiny LFM2.5 for fast testing
    "lfm25_tiny": ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=256,
        attn_type="gqa",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=128,
        conv_kernel_size=3,
        use_qk_norm=True,
        layer_types=["conv", "conv", "attention", "conv"],
        batch_size=2,
        seq_len=64,
        max_steps=20,
        warmup_steps=5,
    ),
}


def get_config(name: str | None = None, **overrides) -> ModelConfig:
    """Fetch a named config and apply optional overrides.

    Always returns a FRESH ModelConfig instance (never the shared preset),
    so callers can safely mutate fields (device, dtype, ...) without
    corrupting MODEL_CONFIGS for other users of the same preset.
    """
    if name is None:
        base = ModelConfig()
    else:
        if name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Available: {list(MODEL_CONFIGS)}")
        base = MODEL_CONFIGS[name]
    return ModelConfig(**{**base.__dict__, **overrides})
