"""Config-driven model specifications for ForgeAI architecture research."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """All hyperparameters needed to build a ConfigurableResearchLLM."""

    vocab_size: int = 151665
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    max_seq_len: int = 2048
    attn_type: str = "mla"           # "diff", "mla", "standard", "gqa"
    ffn_type: str = "swiglu"         # "swiglu", "standard"
    norm_type: str = "layernorm"     # "layernorm", "rmsnorm"
    n_kv_heads: Optional[int] = None  # KV heads for GQA (None = MHA)
    intermediate_size: Optional[int] = None  # FFN hidden dim (None = 8*d/3)
    attn_bias: bool = False  # Add bias to q/k/v projections (Qwen2 style)
    enable_draft_head: bool = False
    kv_compression_dim: int = 128
    rope_base: float = 10000.0
    # YaRN RoPE scaling for context extension. None = no scaling.
    # Dict form: {"type":"yarn", "factor":4.0, "original_max_position_embeddings":1024,
    #             "attention_factor":1.0, "beta_fast":32.0, "beta_slow":1.0}
    rope_scaling: Optional[dict] = None
    dropout: float = 0.0
    device: str = "cuda" if __import__("torch", fromlist=["cuda"]).cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data"

    # Chunked cross-entropy: fuses head + CE without materializing full
    # [B*T, vocab] logits. Saves ~2.8 GB at batch 2 / seq 1024 / vocab 151665,
    # enabling larger batch sizes. ~20% slower per step but net throughput
    # improves when the saved memory allows batch_size to double.
    use_chunked_ce: bool = False
    ce_chunk_size: int = 256
    # Liger-Kernel fused linear cross-entropy: fuses head matmul + CE into one
    # Triton kernel, avoiding materializing [B*T, vocab] logits. ~20% throughput
    # + 60% memory reduction vs stock F.cross_entropy. Requires liger-kernel.
    use_liger_ce: bool = False
    # Architectural improvements from NanoGPT speedrun / IMU-1 (2026):
    # QK-norm: RMSNorm on Q and K before RoPE — improves stability, allows
    # higher LR, ~1-2% lower val loss (OLMoE, Gemma3, Qwen3 all use it).
    use_qk_norm: bool = False
    # Zero-init output projections: attn.out_proj and ffn.w_down start at zero
    # so residual stream is unchanged at init — cleaner gradient flow early.
    zero_init_residual: bool = False
    # Fixed attention scale (0.12 from NanoGPT speedrun) instead of head_dim**-0.5.
    # Only used when set; None = default head_dim**-0.5.
    attn_scale: float = None
    # Activation checkpointing: recompute block forward during backward to save
    # ~50% activation VRAM at the cost of ~20% more compute. Enables larger
    # batch sizes or longer sequences on memory-constrained GPUs.
    use_gradient_checkpointing: bool = False

    # Training defaults
    batch_size: int = 4
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

    def __post_init__(self):
        if self.attn_type == "diff" and self.d_model % (2 * self.n_heads) != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by 2*n_heads ({2 * self.n_heads}) for Differential Attention."
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).")


# Pre-defined architecture targets for the RTX 5070 12 GB budget.
MODEL_CONFIGS = {
    "360m_mla": ModelConfig(
        d_model=1024,
        n_layers=19,
        n_heads=16,
        attn_type="mla",
        ffn_type="swiglu",
        batch_size=2,
        seq_len=1024,
    ),
    "250m_diff": ModelConfig(
        d_model=768,
        n_layers=16,
        n_heads=12,
        attn_type="diff",
        ffn_type="swiglu",
        batch_size=6,
        seq_len=1024,
    ),
    "135m_mla": ModelConfig(
        d_model=768,
        n_layers=8,
        n_heads=12,
        attn_type="mla",
        ffn_type="swiglu",
        batch_size=8,
        seq_len=1024,
    ),
    "tiny_test": ModelConfig(
        d_model=256,
        n_layers=2,
        n_heads=4,
        attn_type="mla",
        ffn_type="swiglu",
        batch_size=2,
        seq_len=128,
        max_steps=20,
        warmup_steps=5,
    ),
    "tiny_draft": ModelConfig(
        d_model=1024,
        n_layers=2,
        n_heads=16,
        attn_type="mla",
        ffn_type="swiglu",
        kv_compression_dim=128,
        batch_size=4,
        seq_len=1024,
        max_steps=500,
        warmup_steps=50,
    ),
    # Exact architectural replica of Qwen2.5-Coder-1.5B-Instruct.
    # Used for weight porting: load Qwen safetensors directly into our model.
    "qwen25_coder_1.5b": ModelConfig(
        vocab_size=151936,
        d_model=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,
        intermediate_size=8960,
        attn_type="gqa",
        attn_bias=True,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        batch_size=2,
        seq_len=1024,
    ),
    # SVD-resized variant: smaller d_model (1024) and fewer layers (20).
    # Target for Key #2 (SVD resize from qwen25_coder_1.5b).
    "qwen25_coder_0.5b_svd": ModelConfig(
        vocab_size=151936,
        d_model=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        intermediate_size=4096,
        attn_type="gqa",
        attn_bias=True,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        batch_size=2,
        seq_len=1024,
    ),
    # XP model: Qwen2.5-Coder-1.5B after KeyStack transforms (GQA→MQA applied).
    # Same architecture as qwen25_coder_1.5b but with n_kv_heads=1 (MQA).
    "xp_1.5b_mqa": ModelConfig(
        vocab_size=151936,
        d_model=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=1,
        intermediate_size=8960,
        attn_type="gqa",
        attn_bias=True,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        batch_size=2,
        seq_len=1024,
    ),
    # XP MLA+MoE: KeyStack-transformed Qwen with MLA attention + MoE FFN.
    # MLA: 4x KV compression (d_c=512 vs 2*128=256 GQA KV dim, but cached as 512 not 2048)
    # MoE: 4 routed + 1 shared expert, d_ff=1792 each, top-2 routing
    "xp_1.5b_mla_moe": ModelConfig(
        vocab_size=151936,
        d_model=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,  # kept for compat; MLA ignores this
        intermediate_size=8960,
        attn_type="mla",
        kv_compression_dim=512,
        attn_bias=True,
        ffn_type="swiglu",  # converted to MoE at load time
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        batch_size=2,
        seq_len=1024,
    ),
    # ForgeLM v1: The full KeyStack-transformed model.
    # Qwen2.5-Coder-1.5B base with all lossless transforms:
    #   - MLA attention (d_c=512, 4x KV compression, 100% energy)
    #   - MoE FFN (4 routed + 1 shared, lossless with top-4)
    #   - MRL (matryoshka dimension reordering)
    #   - QuaRot (Hadamard rotation for KV quantization)
    #   - ValueResidual (V0 stored + 28 gates at 0)
    #   - RotorQuant (rotation matrices for KV compression)
    #   - MTP (4 prediction heads for speculative decoding)
    #   - AirLLM (streamable for low-VRAM inference)
    # Runtime features: StreamingLLM, SnapKV, prefix caching, torch.compile
    "forgelm_v1": ModelConfig(
        vocab_size=151936,
        d_model=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,  # kept for compat; MLA ignores this
        intermediate_size=8960,
        attn_type="mla",
        kv_compression_dim=512,
        attn_bias=True,
        ffn_type="swiglu",  # converted to MoE at load time
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        batch_size=2,
        seq_len=1024,
    ),
    # ForgeLM v2: v1 + new training-free keys (all lossless/identity-init):
    #   - QK-Norm for MLA (RMSNorm on Q/K, identity init → no-op)
    #   - DenseFormer DWA (depth-weighted averaging, identity init → no-op)
    #   - SandwichNorm (post-sublayer RMSNorm, identity init → no-op)
    #   - Logit Cap (runtime clamp ±30)
    #   - SwiGLU Clamp (runtime, GPT-OSS style)
    # All keys are lossless at init — model produces identical output to v1.
    # Fine-tuning can then learn the new parameters for quality gains.
    "forgelm_v2": ModelConfig(
        vocab_size=151936,
        d_model=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,
        intermediate_size=8960,
        attn_type="mla",
        kv_compression_dim=512,
        attn_bias=True,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=32768,
        batch_size=2,
        seq_len=1024,
        use_qk_norm=True,  # QK-Norm for MLA (identity init)
    ),
}


def get_config(name: Optional[str] = None, **overrides) -> ModelConfig:
    """Fetch a named config and apply optional overrides."""
    if name is None:
        base = ModelConfig()
    else:
        if name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Available: {list(MODEL_CONFIGS)}")
        base = MODEL_CONFIGS[name]
    if overrides:
        # Build a new dataclass instance with overrides.
        cfg = ModelConfig(**{**base.__dict__, **overrides})
        return cfg
    return base
