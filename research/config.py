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
    norm_eps: float = 1e-6  # RMSNorm epsilon (LFM2.5 uses 1e-5)
    use_embed_norm: bool = False  # Norm after embedding (LFM2.5 has this, standard models don't)
    use_final_norm: bool = True  # Final norm before head (LFM2.5 doesn't have this)
    rope_base: float = 1_000_000.0  # LFM2.5 base: 1M (evolution 10M was synthetic-only, reverted)
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
    mtp_n_heads: int = 4  # evolution: 4 heads (was 2), score 27.69 vs 23.6
    mtp_loss_weight: float = 0.495  # evolution: 0.495 (was 0.3), near-0.5 optimum
    # mtp_weight: CLI/train-runner alias for mtp_loss_weight. When set (not
    # None), it overrides mtp_loss_weight at MTPModule construction time.
    # mtp_weight=0.0 disables the MTP loss contribution (params become dead,
    # frozen by freeze_dead_params_ before BAdam partitioning).
    mtp_weight: float | None = None
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
    mod_aux_loss_weight: float = 1e-8  # near-zero aux loss (prevents router collapse)
    mod_n_skip_layers: int = 0  # 0 = no skipping (safe default; >0 skips FFN in selected layers)
    # FFN-SkipLLM: skip FFN blocks during inference when input norm is small
    # (saturated layers contribute little). 0.0 = skip none, 0.3 = skip ~30%.
    # Only active during eval (not training). Based on EMNLP 2024 paper.
    # NOTE: Calibration on ForgeLM V3 (1.2B, 16 layers) shows NO FFN saturation
    # — cosine similarities are all low/negative (FFN actively transforms in all
    # layers). This technique requires 32+ layer models to have a non-cold region.
    # Kept in config for future larger models. See docs/FFN_RESEARCH.md.
    ffn_skip_threshold: float = 0.0  # 0.0 = disabled

    # === V4: Hardware-efficient inference keys (2026-08) ===
    # GLA (Grouped Latent Attention, arXiv 2505.21487): latent-compressed KV,
    # shifts decode from BW-bound to compute-bound. Identity warm start (lossless).
    # attn_type="gla" activates this; gla_latent_dim sets compression (0=full=lossless).
    gla_latent_dim: int = 0  # 0 = full KV dim (lossless), >0 = compressed latent
    # GTA (Grouped-Tied Attention, arXiv 2505.21487): ties V to K, halves KV cache.
    # attn_type="gta" activates this. V=K at init (lossless), training unties.
    # Fused QKV + Gate-Up GEMM: single GEMM for Q/K/V and for gate/up projections.
    # Halves kernel launches. Lossless (same math, just fused).
    use_fused_gemm: bool = False
    # BitNet int8 trainable storage (R&D round 15): stores ternary weights as
    # int8 on GPU (1 byte/param) with bf16 master weights on CPU. STE gradients
    # flow GPU→CPU, BAdam updates CPU master, re-quantizes active block to int8.
    # Enables 8B param training on 12GB VRAM (8GB weights + 2.5GB optim + 1GB act).
    bitnet_int8_training: bool = False
    # W8A8 INT8 quantization: weight + activation INT8 with tensor-core GEMM.
    # 2-3x decode speedup at batch=1. Applied at inference (not training).
    use_w8a8: bool = False
    w8a8_mode: str = "int8"  # "int8" or "fp8"
    # NVFP4 quantization (R&D 14): FP4 E2M1 weights with FP8 block scales.
    # Native on Blackwell SM120 (RTX 5070). 3.8x compression, ~99% quality.
    # Auto-selected as default quant on Blackwell by forge_engine.
    use_nvfp4: bool = False
    nvfp4_block_size: int = 32  # NVFP4 standard: 32 elements per FP8 scale
    nvfp4_w4a8: bool = False  # W4A8 mode: FP4 weights + FP8 activations

    # === FP8 Training (Blackwell/Hopper) ===
    # Smooth-SwiGLU: per-channel RMSNorm on FFN gate output to prevent FP8
    # overflow from SiLU outliers. Required for stable FP8 training.
    use_smooth_swiglu: bool = False
    # μScaling: unit-variance weight init (std=1/sqrt(d_in)) keeps all tensors
    # within FP8 range without dynamic scaling overhead. Applied at init only.
    use_mu_scaling: bool = False
    # FP8 autocast: enables torch.autocast with FP8 (E4M3) for forward pass.
    # Requires SM90+ (Hopper/Blackwell). Falls back to BF16 on older GPUs.
    use_fp8_training: bool = False

    # === V5: MoE + Factorized Embedding + Full BitNet (2026-09) ===
    # Mixture-of-Experts FFN: replace dense FFN with shared expert + routed
    # experts. Active params stay ~same; total params scale with n_experts.
    # AirMoE hotloads routed experts from disk at inference (only top-k in VRAM).
    use_moe: bool = False
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_shared_expert: bool = True   # always-active shared expert (DeepSeek-V3 style)
    moe_dense_bypass: bool = True    # skip router at init (lossless = dense FFN)
    # Disable dense_bypass after this many training steps so the router
    # activates (experts start specializing). 0 = never disable (stays dense).
    # DeepSeek-V3 uses a warmup where the router gradually takes over.
    moe_dense_bypass_warmup_steps: int = 0
    moe_noisy_gating: bool = True    # add noise during training for exploration
    moe_load_balance_weight: float = 0.01
    # Router mode: "switch" (Switch-Transformer aux loss, backward compat) or
    # "aux_free" (DeepSeek-V3 auxiliary-loss-free load balancing with per-expert
    # bias + sequence-wise balance loss from full softmax). "aux_free" is
    # lossless at init (bias=0).
    moe_router_mode: str = "aux_free"
    moe_d_ff: int | None = None      # expert hidden dim (None = intermediate_size)
    # Expert Tying (arXiv 2606.16825): share expert weights across consecutive
    # layer groups. g=2: layers (0,1) share, (2,3) share, etc. 2x expert param
    # reduction, near-lossless. Applied as pointer aliasing after model build.
    moe_expert_tying: bool = False
    moe_tie_group_size: int = 2      # consecutive layers that share experts
    # Factorized Embedding (ALBERT pattern): decompose vocab×d_model into
    # vocab×rank × rank×d_model. 7.8x param reduction for 65536×2048.
    # Init via SVD of original embedding (lossless at start).
    use_factorized_embeddings: bool = False
    embed_factorized_rank: int = 256  # factorization rank (<< d_model)
    embed_tie_factor: float = 0.75  # evolution: high tying (was implicit 0.5)
    # BitNet on embeddings: ternary quantize embedding weights (QAT).
    # Saves ~8x on embedding VRAM. Combined with factorized = ~60x reduction.
    use_bitnet_embedding: bool = False

    # === V5.1: Additional architecture keys (2026-08) ===
    # All lossless at init (zero/identity), activated during training.

    # ValueResidual (ResFormer, arXiv 2410.17897): add V_0 residual to all
    # layers. Gate=0 at init → lossless. Training opens the gate.
    # mode: "resformer" (additive V_i + gate*V_0) or "svformer" (shared V_0).
    use_value_residual: bool = False
    value_residual_mode: str = "resformer"  # "resformer" or "svformer"
    value_residual_gate_init: float = 0.0  # 0 = lossless at start

    # SandwichNorm: post-sublayer RMSNorm (identity init = lossless).
    # Stabilizes MoE training by bounding post-FFN/post-attn activations.
    use_sandwich_norm: bool = False

    # LearnedSink (GPT-OSS): per-head attention sink bias logits.
    # init_method="zero" = lossless (no bias added). "constant" adds a
    # small positive bias to stabilize attention.
    use_learned_sink: bool = False
    learned_sink_init: float = 0.0  # 0 = lossless; 1.0 = GPT-OSS default
    learned_sink_init_method: str = "zero"  # "zero", "constant", "random"

    # SwiGLU Clamp (GPT-OSS): clamped SwiGLU activation with scaled sigmoid.
    # Prevents outlier activations from destabilizing quantized inference.
    # This is a runtime-only change (no weights), but changes the activation
    # function — NOT lossless in the strict sense, but close at init since
    # clamping only affects outliers (|x| > limit=7.0).
    use_swiglu_clamp: bool = False
    swiglu_clamp_alpha: float = 1.702  # sigmoid scale (approximates GELU)
    swiglu_clamp_limit: float = 7.0  # clamp value (prevents outliers)

    # === V5.2: Parameter-reduction keys (2026-08) ===
    # All decomposition-based keys convert dense weights on first checkpoint
    # load, then save the compressed checkpoint (no re-conversion on subsequent
    # loads). See .devin/scratchpad.md for full analysis.

    # Monarch FFN (Dao et al. 2022): W ≈ L @ R @ P with L,R block-diagonal.
    # 70% FFN param reduction, near-lossless (matches dense expressivity).
    # ffn_compression="monarch" activates this.
    # block_size: size of block-diagonal blocks (must divide d_model and intermediate).
    ffn_compression: str = "none"  # "none", "monarch", "kron", "tt", "nlrq"
    monarch_block_size: int = 32
    # NLRQ FFN (R&D winner): SVD + quantized factors + optional INT4 residual.
    # 12.8x param reduction (rank=256, 8-bit), 3x with INT4 residual at 0.15% error.
    nlrq_rank: int = 256             # SVD truncation rank
    nlrq_factor_bits: int = 8        # bits per U/V factor element
    nlrq_use_residual: bool = False  # add INT4 group-quantized residual
    nlrq_residual_group_size: int = 128
    nlrq_use_hadamard: bool = False       # HINT4: Hadamard rotation on INT4 factors
    use_peagle_tied: bool = False         # Tied PEAGLE draft head (7x param reduction)
    peagle_lora_rank: int = 32            # LoRA rank for tied PEAGLE heads
    # Kronecker FFN: W = A ⊗ B. 71% FFN param reduction, lossy-recoverable.
    # Factorization shapes for (d_model, intermediate): (a*b, c*d) = (d_model, intermediate).
    kron_a: int = 64   # A is (a, c)
    kron_b: int = 32   # B is (b, d)
    kron_c: int = 32
    kron_d: int = 256
    # Tensor-Train FFN: TT decomposition of weight matrix. 71% reduction, lossy-recoverable.
    tt_rank: int = 4   # TT rank (higher = less compression, less loss)

    # Hyperloop Transformers (arXiv 2604.21254): looped middle blocks.
    # begin_layers unique + middle shared (looped n_loop_iters times) + end_layers unique.
    # 44% layer param reduction, near-lossless (paper: BETTER quality).
    # Lossless at init: all layers unique, loop gate=0. Training opens gate.
    use_hyperloop: bool = False
    hyperloop_begin: int = 2    # unique begin layers
    hyperloop_end: int = 2      # unique end layers
    hyperloop_loop_iters: int = 3  # how many times to loop the shared middle block

    # LiSA (TACL 2026): cross-layer Q/K sharing with alignment FFN.
    # 6x Q/K compression, 19-40% throughput improvement.
    # Lossless at init: per-layer Q/K loaded from checkpoint, shared Q/K gate=0.
    # Training opens gate, then per-layer Q/K can be pruned.
    use_lisa: bool = False
    lisa_compress: int = 6     # Q/K compression factor across attention layers
    lisa_align_dim: int = 0    # alignment FFN hidden dim (0 = d_model // 4)

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
    # Entropy-weighted CE alpha (WeFT/VCORE 2025). When > 0, the chunked CE
    # path uses ChunkedEntropyWeightedCE instead of plain ChunkedLinearCrossEntropy.
    # 0 = disabled (plain CE), 0.5 = production default.
    entropy_alpha: float = 0.0
    # Liger-Kernel fused linear cross-entropy (requires liger-kernel).
    use_liger_ce: bool = False
    # === R&D round 14: training speedup features (2026-08-25) ===
    # Varlen attention: FlashAttention varlen path for packed sequences.
    # Eliminates cross-example attention contamination + padding-mask compute
    # waste. Requires cu_seqlens from PackedSequenceDataset (emit_cu_seqlens).
    # Community: Unsloth 2.1x faster padding-free, 50% less VRAM.
    use_varlen: bool = False
    # Triton fused training kernels: fused RMSNorm + SwiGLU as single Triton
    # kernels (Liger-Kernel-style). Reduces kernel launches and intermediate
    # materialization. Tuned for SM120 (RTX 5070, GDDR7 672 GB/s).
    # Community: Liger-Kernel 20% throughput, 60% VRAM; Unsloth 2-5x.
    use_triton_kernels: bool = False
    # Triton kernel block sizes (tuned for SM120 GDDR7 bandwidth).
    # RMSNorm: block = d_model (2048 for V10).
    # SwiGLU: block = intermediate_size (8192 for V10).
    triton_rms_block_size: int = 4096
    triton_swiglu_block_size: int = 16384
    # APOLLO optimizer: SVD-free random-projection gradient scaling.
    # SGD-like memory with AdamW-level performance. No SVD overhead vs GaLore.
    # Community: arXiv 2412.05183, ICLR 2025. APOLLO-Mini rank=1 = SGD memory.
    apollo_rank: int = 8  # auxiliary subspace rank (1 = APOLLO-Mini)
    apollo_scale: str = "tensor"  # "tensor", "channel"
    # BREAD: landscape correction for BAdam. Applies memory-efficient SGD
    # updates to inactive blocks during the same backward pass, preventing
    # optimization landscape narrowing. Microsoft BlockOptimizers.
    # Community: OpenReview zs6bRl05g8, accelerates BAdam convergence.
    bread_sgd_correction: str = "partial"  # "all", "partial", "disabled"
    bread_sgd_lr_scale: float = 5.0  # SGD lr = base_lr * this (typical 5x)
    # FlashOptim: companded 8-bit optimizer states (7 bytes/param vs 16).
    # Companding on momentum/variance with tight error bounds.
    # Community: arXiv 2602.23349, >50% per-param memory reduction.
    flashoptim_bits: int = 8  # 8 = companded uint8, 4 = companded uint4

    # === R&D Round 19: Qwen3.8-Flash-Next keys (2026-08-29) ===
    # QSA (Qwen Sparse Attention): top-k sparse position selection with
    # averaged attention scores. Reduces attention from O(n²) to O(n*k).
    # Lossless at init (k = seq_len = full attention).
    use_qsa: bool = False
    qsa_top_k: int = 256  # number of positions to keep (0 = full attention)
    # Gated Residual: per-layer learnable gate on the residual stream.
    # Gate init=1.0 = lossless (passes residual through unchanged).
    use_gated_residual: bool = False
    gated_residual_init: float = 1.0  # 1.0 = identity (lossless)
    # N-gram Embedding: host-side n-gram lookup table for knowledge
    # augmentation. 15.3 GB host RAM for 3-gram table (vocab=65536).
    use_ngram_embedding: bool = False
    ngram_n: int = 3  # n-gram order (2=bigram, 3=trigram)
    ngram_dim: int = 256  # embedding dimension per n-gram
    ngram_host: bool = True  # store on host RAM (True) or GPU (False)

    # === R&D Round 20-21: Memory-efficient training + cross-domain formats ===
    # Optimizer selection: "adamw_8bit", "adamw_4bit", "nvme_badam",
    # "muon_bitnet_4bit", "ternary", "nvme_muon_4bit" (best combo)
    optimizer_type: str = "adamw_8bit"
    # NVMe streaming: optimizer states on NVMe, 1 layer active in RAM
    nvme_streaming: bool = False
    nvme_path: str = ""  # path for NVMe optimizer state files
    # FP8 activation storage: quantize activations to FP8 during forward
    # (2x activation memory reduction). GEMMs still run in bf16.
    use_fp8_activation: bool = False
    # GradTopK: top-K% gradient sparsification with EF21 error feedback.
    # 10x fewer gradient transfers per step.
    grad_topk_ratio: float = 1.0  # 1.0 = disabled, 0.1 = top 10%
    grad_topk_ef: bool = True  # error feedback (prevents staleness)
    # HashedNLRQ: hash the NLRQ low-rank factors for additional compression.
    # NLRQ (12.8x) × hash (4-16x) = 50-100x total FFN compression.
    use_hashed_nlrq: bool = False
    hashed_nlrq_compression: float = 8.0  # hash compression factor

    # === R&D Round 24: SpectralKV + BitNetResidual (2026-08-30) ===
    # SpectralKV: Fourier-basis KV cache with O(1) memory per token.
    # Validated on real LFM2.5 weights: 63× compression at 0.095 error.
    # Selected via ActivationConfig.kv_cache="spectral" at inference time.
    use_spectral_kv: bool = False  # config flag (also selectable at inference)
    spectral_kv_max_freq: int = 64  # Fourier frequencies (more = lower error)
    spectral_kv_sink_size: int = 4  # full-precision sink tokens
    # BitNetResidual: ternary weights + element-level dense residual.
    # Validated on real LFM2.5 weights: 0.33 error at 1.8× compression.
    # Replaces pure BitNet (0.80 error) and NLRQ rank-1024 (1.4 error).
    use_bitnet_residual: bool = False
    bitnet_residual_frac: float = 0.10  # fraction of elements kept dense
    bitnet_residual_type: str = "element"  # "element" (validated best)

    # === R&D Round 26: IRI-FP4 Lossless Weight Quantization (2026-08-31) ===
    # IRI-FP4: Iterative Residual FP4. Per-block MSE-optimal scale + iterative
    # residual refinement. Validated on real LFM2.5 + Qwen 2.5 weights:
    #   - x1 (4.5 bits/w, 20.7 dB SQNR): +50.8% PPL — too lossy
    #   - x2 (9.0 bits/w, 41.6 dB SQNR): -0.4% PPL — LOSSLESS, 3.5x vs fp32
    #   - x3 (13.6 bits/w, 62.6 dB SQNR): +0.1% PPL — overkill
    # V10 uses x2: 3.5x compression, near-lossless, no QAT needed.
    use_iri_fp4: bool = False
    iri_fp4_rounds: int = 2        # 2 rounds = 9.0 bits/w, 41.6 dB SQNR, lossless
    iri_fp4_block_size: int = 32   # block size for per-block scale
    # Gradient checkpointing: recompute forward during backward to save VRAM.
    use_gradient_checkpointing: bool = False
    # Selective checkpoint strategy: "all" (full block), "ffn" (recompute only
    # the FFN — largest activation consumer, minimal compute penalty), "attn",
    # "none". Only applies when use_gradient_checkpointing is True.
    # Evolution-discovered: "selective" with all 16 layers + block_size=512
    # gives best quality (0.92) at 50%+ VRAM savings for 32K context.
    selective_gradient_checkpointing: str = "all"
    # Checkpoint block size (evolution-discovered: 512 matches typical seq len)
    checkpoint_block_size: int = 512
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
        # GQA: n_heads must be a multiple of n_kv_heads (or n_kv_heads == n_heads).
        n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        if n_kv <= 0 or self.n_heads % n_kv != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a positive multiple of "
                f"n_kv_heads ({n_kv}) for GQA."
            )
        # Per-layer type list must match n_layers when provided.
        # If n_layers is overridden but layer_types isn't, auto-resize by
        # truncating or cycling the existing pattern to match n_layers.
        if self.layer_types is not None:
            n_lt = len(self.layer_types)
            if n_lt != self.n_layers:
                if n_lt > self.n_layers:
                    # Truncate to first n_layers entries.
                    self.layer_types = list(self.layer_types[: self.n_layers])
                else:
                    # Cycle existing pattern to fill n_layers.
                    base = list(self.layer_types)
                    self.layer_types = [
                        base[i % n_lt] for i in range(self.n_layers)
                    ]
        # FFN intermediate size: positive when explicitly set.
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be > 0."
            )
        # head_dim must be consistent (d_model / n_heads) and >= 1.
        head_dim = self.d_model // self.n_heads
        if head_dim < 1:
            raise ValueError(
                f"head_dim (d_model/n_heads = {head_dim}) must be >= 1."
            )


# Pre-defined architecture targets.
MODEL_CONFIGS = {
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
        rope_base=1_000_000.0,          # LFM2.5 original: 1M (reference port)
        max_seq_len=128,
        conv_kernel_size=3,
        use_qk_norm=True,
        layer_types=["conv", "conv", "attention", "conv"],
        batch_size=2,
        seq_len=64,
        max_steps=20,
        warmup_steps=5,
    ),

    # Ultra-compact ForgeLM V7 gen model for ForgeEvolve.
    # Uses V7 architecture keys (BitNet b1.58, GTA attention, NLRQ FFN
    # compression, RoPE) at a tiny size for use as an LLM-based gen model
    # in evolutionary search. Built from scratch (random init) — no checkpoint.
    # Reuses LFM2.5 tokenizer (vocab=65536). Configurable size for grow/shrink.
    "gen_model_tiny": ModelConfig(
        vocab_size=65536,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,                   # GQA 4x (8/2)
        intermediate_size=512,
        attn_type="gta",                # Grouped-Tied Attention (V7)
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,          # LFM2.5 base: 1M
        max_seq_len=512,
        use_qk_norm=True,
        # ── V7 keys (minimal set for tiny gen model) ──
        use_bitnet=True,                # BitNet b1.58 ternary QAT
        bitnet_learned_scale=True,
        ffn_compression="nlrq",         # NLRQ FFN compression
        nlrq_rank=64,                   # small rank for tiny intermediate=512
        nlrq_factor_bits=8,
        nlrq_use_residual=False,
        # ── Training hyperparams (for fine-tuning) ──
        batch_size=1,
        seq_len=512,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    ),


    # ForgeLM V10-1.2B: LFM2.5-1.2B with R26 IRI-FP4 lossless weight quant.
    # Same dimensions as V9 (d_model=2048, 16 layers) for 1:1 architecture compat.
    # V10 replaces V9's BitNetResidual (ternary, catastrophic post-training PPL)
    # with IRI-FP4 x2 (9.0 bits/w, 41.6 dB SQNR, -0.4% PPL — near-lossless).
    #
    # V10 advantages over V9:
    #   - 3.5x weight compression vs fp32 (IRI-FP4 x2 at 9.0 bits/w)
    #   - NEAR-LOSSLESS: 41.6 dB SQNR, -0.4% PPL delta (V9 ternary = catastrophic)
    #   - No QAT/fine-tuning needed (V9 BitNetResidual needs QAT to recover)
    #   - Same SpectralKV 63x KV cache compression (carried from V9)
    #
    # Memory: 1.2B params * 9.0 bits/w / 8 = ~1.35 GB weights (vs 2.61 GB bf16)
    # Full model with KV cache: ~1.8 GB total (fits any GPU)
    # ──────────────────────────────────────────────────────────────────────
    "forgelm_v10_1.2b": ModelConfig(
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
        norm_eps=1e-5,
        use_embed_norm=False,
        use_final_norm=True,
        rope_base=1_000_000.0,
        max_seq_len=32768,
        conv_kernel_size=3,
        use_qk_norm=True,
        layer_types=["conv", "conv", "attention", "conv", "conv", "attention",
                     "conv", "conv", "attention", "conv", "attention",
                     "conv", "attention", "conv", "attention", "conv"],
        # ── V10 NEW: IRI-FP4 lossless weight quantization (R26) ──
        use_iri_fp4=True,
        iri_fp4_rounds=2,              # 2 rounds = 9.0 bits/w, 41.6 dB SQNR, lossless
        iri_fp4_block_size=32,
        # ── V9 carried: SpectralKV (inference-time, no checkpoint change) ──
        use_spectral_kv=True,
        spectral_kv_max_freq=64,
        spectral_kv_sink_size=4,
        # ── No BitNet/BitNetResidual (replaced by IRI-FP4) ──
        use_bitnet_residual=False,
        use_bitnet=False,
        use_bitnet_embedding=False,
        ffn_compression="none",
        nlrq_rank=0,
        use_hashed_nlrq=False,
        use_factorized_embeddings=False,
        embed_factorized_rank=0,
        use_pit=False,
        zero_init_residual=True,
        # ── Training hyperparams ──
        batch_size=1,
        seq_len=2048,
        max_steps=50000,
        warmup_steps=2000,
        max_lr=3e-4,
        min_lr=3e-5,
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
