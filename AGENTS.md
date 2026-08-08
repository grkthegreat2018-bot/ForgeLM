# ForgeAI Research — Agent Notes

## Weight Strategy (Qwen2.5-Coder-1.5B)

**Two-pronged approach** — see `docs/keys/KEY_DEFINITION.md` for full details.

### Prong 1: Direct Weight Transforms (cross-arch port — the viable path)
Transform weights directly between architectures, no extraction needed:
- **BitNet**: bf16 → ternary {-1,0,+1} via absmean — `research/convert_keys.py`
- **SVD resize**: large → small via shared SVD projection — `research/convert_key_svd.py`
- **MLA**: GQA K/V → low-rank compression — **WORKING** (cos=0.9999 at d_c=512, 4/5 top-5 match)
  - Key insight: SVD on UNEXPANDED K/V (preserves GQA head structure for RoPE)
  - Key insight: copy K/V biases from GQA to k_up_proj/v_up_proj (bias dominates K signal, norm=1240)
  - d_c=256 retains 73% energy, d_c=512 retains 100% (rank=2*n_kv*head_dim=512)
- **MoE**: dense FFN → routed experts — **WORKING** (weight split EXACT, cos=1.0 with all experts)
  - Splits intermediate_size=8960 into 5 slices (4 routed + 1 shared, d_ff=1792 each)
  - Same total params, 60% active FLOPs (top-2 of 4 + shared)
  - w_down scaled by n_experts to compensate for softmax gating (1/n per expert)
  - Router initialized to uniform (zeros), needs fine-tuning (cos=0.73 with top-2)
- **BitNet**: bf16 → ternary {-1,0,+1} — **WORKING** (2.29 GB vs 3.09 GB, 26% smaller)
  - Quantizes all 2D Linear weights (attention + MoE experts), keeps embed/norm/bias/router bf16
- **Liquid (LFM2)**: hybrid conv+attention blocks — **IMPLEMENTED** (`liquid.py`)
  - Double-gated short conv replaces ~60% of attention (O(T*k*d) vs O(T²*d)), 2x faster on CPU
- **DSpark**: semi-autoregressive speculative decoding — **IMPLEMENTED** (`dspark.py`)
  - Parallel backbone + sequential RNN + confidence-scheduled verification, 60-85% speedup
- **RotorQuant**: KV cache compression via Givens rotations — **IMPLEMENTED** (`rotorquant.py`)
  - 3.88x compression, 0.94% error, 1.12ms/token, O(d) per vector
- **QK-Norm for MLA**: RMSNorm on Q/K with projection absorption — **IMPLEMENTED** (`keys/qk_norm_mla_key.py`)
  - 2026 paper: absorb static norm weight into k_up_proj/q_proj, dynamic scalar at runtime
  - Identity init = lossless, verified cos=1.0 vs v1
- **WQ Elimination**: Replace Q projection with identity — **IMPLEMENTED** (`keys/wq_elim_key.py`)
  - 2026 paper: saves 25% attention params (66M for ForgeLM), needs fine-tuning to recover
- **DenseFormer**: Depth-weighted averaging (DWA) — **IMPLEMENTED** (`keys/denseformer_key.py`)
  - Identity init = lossless, cross-layer dense connections
- **SandwichNorm**: Post-sublayer RMSNorm — **IMPLEMENTED** (`keys/sandwich_norm_key.py`)
  - Identity init = lossless, combines Pre-Norm + Post-Norm stability
- **Logit Cap**: Clamp logits to ±30 — **IMPLEMENTED** (`keys/logit_cap_key.py`)
  - Runtime flag, prevents extreme logits (Gemma 2 style)
- **SwiGLU Clamp**: GPT-OSS clamped SwiGLU — **IMPLEMENTED** (`keys/swiglu_clamp_key.py`)
  - Runtime flag, α=1.702 sigmoid + value clamping + +1 residual
- **ForgeLM v2**: v1 + 5 new lossless keys (QK-Norm MLA, DenseFormer, SandwichNorm, Logit Cap, SwiGLU Clamp)
  - Apply: `python -u .devin\apply_new_keys.py` (lossless, cos=1.0 vs v1)
  - Config: `forgelm_v2` (use_qk_norm=True)
  - Checkpoint: `research/checkpoints/forgelm_v2.safetensors` (928 tensors, 3643 MB)
  - **WARNING**: `apply_safe_keys.py` corrupted v2 on 2026-08-06 by overwriting it in-place
    with MD5-hash-based "fact injection" (random noise, not real facts). w2 weights in
    layers 20-27 expert 0 were destroyed (cos=0.07). Rebuilt from v1 with apply_new_keys.py.
    NEVER run apply_safe_keys.py on v2 — it uses hash-based fake fact vectors, not real model
    forward passes. The "LOSSLESS" label was wrong.
- **Norm Folding**: Fold ALL RMSNorm into adjacent Linear weights — **IMPLEMENTED** (`keys/norm_folding_key.py`)
  - TaperNorm 2026: static norm scale γ absorbed into q_proj, kv_down_proj, w_gate, w_up, head
  - Eliminates 113 norm operations per forward pass (57 ln + 56 qk_norm)
  - Lossless (verified), applied to v3
- **Expert Consolidation**: Merge similar MoE experts — **IMPLEMENTED** (`keys/expert_consolidation_key.py`)
  - Novel: no published method for MoE→fewer-experts merging
  - Cosine similarity threshold + weighted averaging + router redirect
  - Applied post-load (experts created at MoE conversion time)
- **GRAIL Compensation**: Heal lossy transforms via ridge regression — **IMPLEMENTED** (`keys/grail_key.py`)
  - GRAIL 2026: Gram matrix + ridge regression reconstruction map
  - Makes WQ Elim, Wanda, Expert Consolidation near-lossless
  - Verified: error reduction from 2.19 → 0.00008 on test data
- **Activation Transmutation**: Swap SwiGLU → ReGLU/GeGLU — **IMPLEMENTED** (`keys/activation_transmute_key.py`)
  - Novel: per-channel affine transform (α, β) via grid search
  - SwiGLU → ReGLU: 0.7% relative error (near-lossless)
  - Enables faster activation functions without retraining
- **ForgeLM v3**: v2 + Norm Folding (lossless) — UNPUBLISHED (still V2 in public)
  - Apply: `python -u .devin\apply_novel_keys.py` (lossless)
  - Checkpoint: `research/checkpoints/forgelm_v3.safetensors` (815 tensors, 3643 MB)
  - 113 fewer tensors than v2 (norms folded into adjacent weights)
- **Lossless Quant Chain**: SpinQuant→QuaRot→int4 — **IMPLEMENTED** (`keys/lossless_quant_key.py`)
  - Chains Hadamard rotation + int4 GPTQ for near-lossless 4x compression
  - Target: sub-8GB VRAM (3.4GB bf16 → ~1.2GB int4 + ~0.8GB KV/activations = ~2GB total)
  - Rotation smooths outliers → int4 quantization error drops 3-5x
- **AirMoE**: AirLLM + MoE hybrid — **IMPLEMENTED** (`keys/airmoe_key.py`)
  - Novel arch: only active experts loaded to VRAM, inactive on disk
  - LRU cache + router-based prefetch + SVD expert compression
  - Infinite params, limited load — each task calls relevant experts from disk
  - Expert compression: SVD low-rank (90% energy) reduces disk transfer 4-10x
  - Inspired by FlashMoE/FluxMoE/FineMoE (2026)
- **Knowledge Pack**: Zero-token knowledge via KV cache injection — **IMPLEMENTED** (`keys/knowledge_pack_key.py`)
  - 2026 paper: KV-Prefix Equivalence — KV cache from standalone pass = joint pass KV
  - Pre-compute KV caches for knowledge domains, inject at inference (zero token cost)
  - Contrastive steering: value deltas nudge behavior (mid-layers 33-66%)
- **Closed-Form Fact Injection**: Write facts into MLP weights — **IMPLEMENTED** (`keys/fact_injection_key.py`)
  - COLM 2026 (Stanford): closed-form recipe for fact-storing MLPs, no gradient descent
  - Rank-1 weight update per fact using unused hidden dimensions
  - Multiple facts stored in parallel using different hidden dims
- **Context-to-Weight Patch**: Convert ICL to rank-1 weight patches — **IMPLEMENTED** (`keys/context_patch_key.py`)
  - 2026 paper: context effect = rank-1 patch to MLP weights + RMSNorm scale
  - Extract patches from few-shot examples, apply permanently
  - Proven for MoE, gating, pre/post-norm architectures
- **Self-Play Knowledge**: Generate data → closed-form weight update — **IMPLEMENTED** (`keys/self_play_key.py`)
  - Novel: combines self-play generation with closed-form fact injection
  - Pipeline: generate Q→A pairs → confidence filter → fact vectors → inject into MLP
  - No gradient descent, no loss function, no training loop
- **Binary g128**: Binary {±1} quantization with group-wise FP16 scales — **IMPLEMENTED** (`keys/binary_g128_key.py`)
  - Inspired by Bonsai-27B (prism-ml): 1.125 bits/weight, 14.2x compression vs FP16
  - Every 128 weights share one FP16 scale factor
  - Novel twist: group scale absorption — fold FP16 scales into adjacent RMSNorm/Linear weights
  - LOSSY — do NOT apply to ForgeLM V2 or expert packs
- **4-bit KV Cache**: Group-wise 4-bit KV quantization — **IMPLEMENTED** (`keys/kv4bit_key.py`)
  - Complementary to RotorQuant (simpler, faster), per-channel FP16 scales
  - Novel twist: KV scale absorption — fold scales into QK-Norm (ForgeLM already has QK-Norm)
  - LOSSY — do NOT apply to ForgeLM V2 or expert packs
- **Hybrid Linear Attention**: 75% linear / 25% full attention — **IMPLEMENTED** (`keys/hybrid_linear_key.py`)
  - Inspired by Bonsai-27B: O(T·d) not O(T²·d), only 25% of layers grow KV cache
  - Linear attention via elu+1 feature map (Katharopoulos et al. 2020)
  - Novel twist: adaptive split ratio based on attention entropy per layer
  - LOSSY (architecture change) — do NOT apply to ForgeLM V2 or expert packs
- **Test-Gated Fact Injection**: Only inject test-verified facts — **IMPLEMENTED** (`keys/test_gated_injection_key.py`)
  - Combines FactInjectionKey + test verification from self-play pipeline
  - Only injects solutions where test_passed=True, scaled by quality score
  - LOSSLESS — safe for ForgeLM V2 and expert packs
- **CoT Knowledge Pack**: Reasoning traces as KV cache packs — **IMPLEMENTED** (`keys/cot_knowledge_pack_key.py`)
  - Combines KnowledgePackKey + chain-of-thought traces from self-play
  - Pre-computes KV from reasoning, injects at inference (zero token cost)
  - LOSSLESS — safe for ForgeLM V2 and expert packs
- **Self-Play Context Patch**: Rank-1 patches from self-play solutions — **IMPLEMENTED** (`keys/selfplay_context_patch_key.py`)
  - Combines ContextPatchKey + self-play (prompt→solution) pairs
  - Test-passed → positive patches, failed → negative anti-patches
  - LOSSLESS — safe for ForgeLM V2 and expert packs
- **MAC-Attention**: Match-Amend-Complete attention reuse — **IMPLEMENTED** (`keys/attn_reuse_key.py`)
  - arxiv 2604.00235: reuse attention for similar queries, 14.3x attention speedup
  - Caches (pre-RoPE query, unnormalized attention, sum_scores) per layer
  - On match: amend cached attention with only NEW KV entries (O(new) not O(total))
  - 99% hit rate in self-play, lossless at init (cache empty = all misses)
  - TRIVIAL class: runtime cache, training-free, model-agnostic
  - Conditional O(1) — O(1) on hit, O(N) on miss. Not true O(1).
- **Gated DeltaNet-2**: Fixed-state recurrent attention — **IMPLEMENTED** (`keys/gated_deltanet_key.py`)
  - arxiv 2605.22791: used in Qwen3.5, Kimi, Olmo Hybrid (production-ready)
  - Channel-wise erase + write gates (decoupled), fixed (d×d) state per head
  - TRUE O(1) per token — state NEVER grows, context length irrelevant
  - Verified: 0.02x ratio (perfectly O(1)), 192 KB fixed state per layer
  - PARTIAL class: needs fine-tuning (lossy at init). NOT for ForgeLM V2.
- **SATA**: Symmetry-Aware Taylor Attention — **IMPLEMENTED** (`keys/sata_key.py`)
  - arxiv 2602.00294: PROVES softmax attention is O(1) per token via Taylor expansion
  - P=4 terms = Float16 precision (mathematical identity, not approximation)
  - Fixed hidden state: (dV+1) × C(dK+P-1, P-1) per head, never grows
  - Verified: 0.95x ratio (O(1)), cos=0.9997 vs standard attention at P=2
  - PARTIAL class: near-lossless at P=4, needs fine-tuning for quality
- **Context Phase Transition**: Bake context into weights, O(1) generation — **IMPLEMENTED** (`context_phase_transition.py`)
  - Combines Context Patch + Knowledge Pack + Fact Injection (3 existing keys)
  - 3-phase: Ingest (O(N), one-time) → Transition (apply patches) → Generate (O(1)/token)
  - Context is IN THE WEIGHTS — no KV cache for context at generation
  - PARTIAL class: approximation (rank-1 patches), best for structured context
- **Stateful Transformers**: Async pre-compute, O(|query|) latency — **IMPLEMENTED** (`stateful_transformer.py`)
  - arxiv 2605.13784: decouple data plane (async) from query plane (O(|q|))
  - Process context in background thread, queries consume pre-computed KV
  - 2.4-5.9x speedup, preserves full attention quality
  - TRIVIAL class: runtime orchestration, training-free
- **Forward Cache**: Cache forward passes for repeated inputs — **IMPLEMENTED** (`forward_cache.py`)
  - SYSTEMS_IDEATION.md C2: 20-40% fewer forward passes in self-play
  - LRU cache keyed on input token ID hash, returns cached logits on hit
  - TRIVIAL class: runtime cache, training-free
- **Self-Modeling**: Model predicts own errors, retries — **IMPLEMENTED** (`self_model.py`)
  - SYSTEMS_IDEATION.md D5: confidence-based retry policy
  - ConfidenceScorer (logit gap + entropy + top-1 prob) + RetryPolicy
  - Retry with higher temperature on low confidence
  - TRIVIAL class: runtime policy, training-free
- **Expert Tying (V3)**: Tie expert weights across consecutive layer pairs — **IMPLEMENTED** (`keys/expert_tying_key.py`)
  - arxiv 2606.16825: share MoE experts between layer pairs (0,1), (2,3), etc.
  - 2x FFN VRAM reduction (924.8M -> 462.4M, saved 882 MB)
  - LOSSY: cos=0.53 (experts between layers are NOT similar, cos=0.01)
  - PARTIAL class: needs fine-tuning to recover quality
- **WiSparse (C2)**: Weight-aware activation sparsity — **IMPLEMENTED** (`keys/wisparse_key.py`)
  - Skip neurons where |activation| * |weight| < percentile threshold
  - Training-free, cos=0.99995 at 3% target (14% actual sparsity)
  - Precomputes weight importance (w1_norm * w3_norm * w2_norm per neuron)
  - TRIVIAL class: runtime sparsity, no weight changes
- **SharQ FP4 (C3)**: FP4 weights + sparse residual — **IMPLEMENTED** (`keys/sharq_fp4_key.py`)
  - Decompose: weight = FP4_quantize(weight) + sparse_residual
  - Per-channel scaling (better than per-tensor)
  - cos=0.97 at 15% residual (2.5x compression), cos=0.99 at 40% (1.5x)
  - TRIVIAL class: runtime quantization, training-free
- **L1 Speculative Attention**: Draft low-rank attn, verify with full — **IMPLEMENTED** (`keys/speculative_keys.py`)
  - Entropy-based accept/reject: low-entropy attention = easy to approximate
  - cos=1.0 (LOSSLESS), 71% accept rate, 56.8% attention compute saved
  - TRIVIAL class: runtime optimization, training-free
- **L6 Speculative FFN**: Draft top-1 expert, verify with full — **IMPLEMENTED** (`keys/speculative_keys.py`)
  - cos=1.0 (LOSSLESS), 0% accept rate with dense_bypass (experts differ)
  - Would work with routed MoE (top-2 of 4) where top-1 dominates
  - TRIVIAL class: runtime optimization, training-free
- **L7 Redundant Layer Skip**: Skip near-identity layers — **IMPLEMENTED** (`keys/speculative_keys.py`)
  - Greedy calibration: try each layer individually, skip only if output unchanged
  - cos=1.0 (LOSSLESS) at 0% skip — this model has no truly redundant layers
  - At 0.99 threshold: 9/28 skippable but cos=0.916 (lossy)
  - TRIVIAL class: runtime optimization, training-free
- **Tensor Deduplication**: Dedup exact-same tensors — **IMPLEMENTED** (`keys/tensor_dedup_key.py`)
  - LOSSLESS: identical bytes → shared storage, zero information loss
  - V2 has 165 duplicate tensors (467.3 MB): embed==head (466.75 MB tied but stored separately),
    56x post_attn_norm==post_ffn_norm (all identity), 56x q_norm==k_norm (all identity),
    28x router.gate==router.noise (all zeros)
  - FULL class: reversible (dedup map records aliases), composable
  - Applied to V2: 928 → 735 tensors, 3643 → 3176 MB (12.8% smaller, bit-exact)
- **Attention Scale Folding**: Fold 1/sqrt(head_dim) into q_proj/k_up_proj — **IMPLEMENTED** (`keys/attn_scale_fold_key.py`)
  - Math identity: fold^2 = scale, RoPE is linear so scale propagates through
  - **NOT APPLIED to V2**: PyTorch's `F.scaled_dot_product_attention` bakes in 1/sqrt(head_dim)
    internally, so folding into weights causes double-scaling. Would need model code change
    to pass `scale=1.0` to SDPA. Key is correct math but incompatible with SDPA's default.
  - FULL class: reversible (divide back), composable
- **Dead Weight Pruning**: Remove all-zero tensors — **IMPLEMENTED** (`keys/dead_weight_key.py`)
  - LOSSLESS: zero tensors contribute nothing to forward pass
  - V2 has 28 dead tensors (0.34 MB): router.noise (all zeros from init)
  - Skips router.gate (all-zero but IS used by MoE router forward pass)
  - FULL class: reversible (record what was removed), composable
- **RoPE Buffer Sharing**: Share cos/sin buffers across all layers — **IMPLEMENTED** (`keys/rope_share_key.py`)
  - LOSSLESS: all layers use same RoPE (same head_dim, base, max_seq_len)
  - Saves (n_layers-1) × 2 × max_seq_len × head_dim × 4 bytes VRAM
  - TRIVIAL class: runtime optimization, no weight changes
  - `apply_rope_sharing(model)` — call after model load, before inference
- **V2 Optimized Checkpoint**: v2 + Dedup + DeadWeight (bit-exact lossless)
  - Apply: `py -3.13 .devin\optimize_v2.py --apply --no-scale-fold` (bit-exact lossless)
  - Load: `from research.optimized_loader import load_optimized_v2, load_optimized_v2_fast`
  - Checkpoint: `research/checkpoints/forgelm_v2_opt.safetensors` (735 tensors, 3176 MB)
  - Metadata: `forgelm_v2_opt.safetensors.meta.json` (dedup map)
  - **Verified: logit cos=1.0, top-1 match=100%, KL=0.0 — PERFECTLY LOSSLESS**
  - **Save: 467.6 MB (12.8%), 193 fewer tensors (20.8%)**
- **KeyStack**: 61 keys total (13 FULL, 5 BI, 26 PARTIAL, 17 TRIVIAL)
- **Version naming**: V1 = published. V2+ = unpublished dev versions until published to HF.
- **Pipeline**: `python -m research.serving.forge_pipeline` chains MLA → MoE → BitNet

### Directory Structure (refactored)
```
research/
  __init__.py, config.py, model_loader.py, checkpoint_io.py,
  convert_keys.py, convert_key_svd.py    # core
  training/         # self_play_expert_training, dpo_align, bake_v4, training_utils, chunked_ce
  decoding/         # dspark, eagle, medusa, mtp
  quantization/     # bitnet, spinquant, rotorquant, wanda_prune, inference_quant, kv_compress, paged_kv
  evaluation/       # reasoning_benchmarks, prompt_tests*, goal_scorer, goal_tasks, livecodebench_eval
  moe/              # moe (core conversion), airmoe_infinite, airmoe_hotswap, mola
  serving/          # forge_pipeline, serve, chat_ui, fast_infer
  runtime/          # vram_manager, cuda_graph, forward_cache, self_model, signal_capture, task_logger
  architecture/     # dora, gateskip, live_learn, thinking_model
  o1_generation/    # context_phase_transition, stateful_transformer
  self_play/        # infinite_curriculum, recursive_self_play, self_play_sandbox
  keys/             # 75 key implementations (weight transforms + runtime keys)
  inference/        # forge_engine, kv_backend, decoding, streaming_llm, snapkv
  checkpoints/      # forgelm_v2.safetensors, qwen_hf/ (tokenizer), forgelm_v4/ (AirMoE experts)
  data/             # expert_training/ (training results)
```

### Prong 2: Extraction + Fine-tune (same-arch, reduce training)
Extract what's exact, approximate the rest, copy the impossible, then fine-tune:
- **Exact (float64 lstsq):** RMSNorm, V projection, W_up
- **Approximate (Gauss-Newton):** W_gate (silu non-monotonic — 60% ambiguous)
- **NOT recoverable (proven):** O (head-channel non-identifiability, 2025),
  W_down (underdetermined), Q/K multi-head (NOT identifiable, 2026)
- **Trivial (direct copy):** Embedding, LM Head (tied), RoPE, Causal mask
- **Numerical:** use `lstsq` (QR) not `pinv` (SVD); float64 for solve; center data with bias

**Code:** `research/convert_keys.py` + `convert_key_svd.py` (Prong 1, core)
**Docs:** `docs/keys/KEY_DEFINITION.md` — full classification with literature
**Technique survey:** `docs/research/LLM_TECHNIQUES_SURVEY.md` — 320+ techniques across 6 parts (weights, runtime, KV cache, architecture, + 2026 addendum: MISA/Alloc-MoE/SpecMoE/LightMoE, TEMPO/PoT/ORCA/DiSCTT, task-vector=gradient/Pico, FOREVER/self-generated-replay/OAKS)
**Novel ideation:** `docs/research/NOVEL_SOLUTION_IDEATION.md` — the AirMoE-style "combine techniques to solve each other's weaknesses" methodology + 12 novel idea candidates (GhostMoE, TTT-Pack, ExpertGenesis, FoldedHeadMoE, ConfSpec, FastExpertMerge, BudgetGuard, PocketExpert, DoRAPatch, CompressedPack, UncertainLearn, MTP-EAGLE-DSpark) + residual-problem map + research gaps. Use this to generate new keys.
**Component atlas:** `docs/research/LLM_COMPONENT_ATLAS.md` — per-component (embedding, RoPE, attention, FFN, norm, residual, MoE router, KV cache, output, loss) mechanics + optimizations + what-if ideation ([PUSH]/[FLIP]/[TRY] tags) + cross-component ideas + test-when-free priority. Use this for component-level refinement ideas.
**First-principles ideation:** `docs/research/FIRST_PRINCIPLES_IDEATION.md` — original theorizing (no literature aggregation). 12 ideas (F1-F12) each derived by naming a hidden structural assumption, showing why it's wasteful, and deriving a fix from mechanics. Includes the generative rule: "find what the architecture treats as uniform/global/binary, make it structured/local/spectral." Ideas: indexed attention, residual memory hierarchy, weight/activation spectrum, defactoring (separate knowledge from computation), per-query adaptive temperature, continuous-time RoPE, mechanistic distillation, shared basis layers, working memory registers, task-adaptive precision, bidirectional generation, per-head learned kernels.
**Key mapping master:** `docs/research/KEY_MAPPING_MASTER.md` — maps ALL 58 ideas across the 3 ideation docs to concrete keys (class, training-need, forward/reverse sketch, filename). 38 no-training keys, 9 minimal-training, 7 NOT-A-KEY (arch redesigns for config.py). 3 FULL lossless (FoldedHeadMoE, AsymmetricTying, FoldQKNormMLA). 5-tier implementation priority. Each ideation doc has a Key Mapping section pointing here.
**Key novelty audit:** `docs/research/KEY_NOVELTY_AUDIT.md` — online research checking which of the 47 proposed keys already exist. Result: 30 PUBLISHED (implement + cite, verified to work), 8 ADJACENT (close work, our twist differs), 9 NOVEL (not found — ForgeAI research contributions). WARNING: FoldedHeadMoE reclassified FULL→PARTIAL (routing absorption paper shows end-to-end gates fail; must be post-hoc). Implement published keys first (safe wins), research the 9 novel ones.
**Key novelty audit Part 2:** `docs/research/KEY_NOVELTY_AUDIT_PART2.md` — extends audit with 11 more published keys (SAE steering, MoH, fast weights, RadixAttention, test-time scaling, model merging, EAGLE-3) and first-principles ideation on 8 uncovered components (tokenizer, sampling, batching, prompt structure, compute-as-resource, SAE features, cross-attention injection, model merging at inference). 8 NEW novel keys identified (vocab_pack, conformal_sampler, conformal_batch, prompt_to_pack, compute_futures, sae_pack, cross_attn_pack, per_query_interp). Total: 41 PUBLISHED + 8 ADJACENT + 17 NOVEL + 11 NOT-A-KEY = 77 ideas mapped. The "pack" pattern (extract knowledge from fine-tuned model as portable pack, inject into base without training) is ForgeAI's signature — 6 of 17 novel keys are pack keys. Conformal prediction is underexploited — 3 novel keys use it for coverage guarantees.
**Boot-time audit:** `docs/research/BOOT_TIME_AUDIT.md` — covers the 7-stage cold-start pipeline (tokenize → load → compile → profile KV → CUDA graphs → prefix warm → first token). ForgeAI has 3 assets (Gigatoken, DSpark, torch.compile). 8 published techniques to adopt: TokTier (2.1x faster than Gigatoken), fastsafetensors (4.8-7.5x weight load), persistent compile+autotune cache (env vars), growable KV cache (eliminates OOM), CUDA graph capture (22ms→14ms/token), Foundry templates (99% cold-start reduction), RadixAttention (prefix sharing), Lookahead decoding (zero-infra 1.2-1.5x). 2 novel boot-time keys: boot_pipeline (orchestrator) + graph_template (portable CUDA graph pack). Boot-time is systems-level (all TRIVIAL keys), not weight-level.
**Verification:** `python .devin/test_keystack_verify.py`

## Environment

- Windows 11, NVIDIA RTX 5070 (12 GB VRAM), CUDA 13.1 driver, Python 3.11 (hermes venv base).
- Project virtual environment: `D:\windsurf\ForgeAI\venv` (Python 3.11.15).
- Install command: `D:\windsurf\ForgeAI\venv\Scripts\pip.exe install -r D:\windsurf\ForgeAI\requirements.txt`
  - Uses PyTorch 2.8.0+cu128 for Blackwell/RTX 5070.
  - bitsandbytes 0.49.2 is installed; `research.train` and `research.sft_align` default to 8-bit AdamW (`bnb`).
  - `torch.compile` with Inductor works with `triton-windows==3.4.0.post21` plus a one-line patch to `triton/runtime/cache.py` (shortens temp dir names to avoid Windows MAX_PATH). When patched, it raises throughput from ~10,000 to ~13,300 tok/s and drops VRAM to ~7.5 GB.
  - `prep_data.py` loads `.env` and sets `HF_HUB_DOWNLOAD_TIMEOUT=60` automatically. Set `HF_TOKEN` in your environment or `.env` to use your Hugging Face account for higher rate limits.

## Verified Commands

```powershell
# Activate environment
. D:\windsurf\ForgeAI\venv\Scripts\Activate.ps1

# Port Qwen2.5-Coder-1.5B weights into our model format (zero training, exact replica)
# Downloads original bf16 safetensors from HuggingFace (~3GB, lossless)
# Then maps all 338 tensors 1:1 into our ConfigurableResearchLLM
# Result: our model with 100% of Qwen's quality, verified cosine sim = 1.0
python -m research.port_weights --src ".devin/hf_cache/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a/model.safetensors" --config qwen25_coder_1.5b --out research/checkpoints/qwen25_coder_1.5b_ported.safetensors

# Verify ported model matches HF Qwen2.5-Coder-1.5B (logit cosine sim should be 1.0)
python .devin/verify_ported.py

# Pre-tokenize 100M training mix (adjust --train-tokens as needed)
python -m research.prep_data --train-tokens 100000000 --val-tokens 2000000

# Pre-tokenize with gigatoken (compat mode, ~30-40x faster on desktop CPUs, exact HF parity)
python -m research.prep_data --train-tokens 100000000 --val-tokens 2000000 --tokenizer-backend gigatoken-compat

# Pre-tokenize with gigatoken (native batched mode, largest speedup on many-core CPUs)
python -m research.prep_data --train-tokens 100000000 --val-tokens 2000000 --tokenizer-backend gigatoken-native --encode-batch 512

# Pre-train 360M MLA baseline (fastest config: add --compile if triton is patched)
python -m research.train --config 360m_mla --steps 50000 --compile

# Pre-train with chunked CE (saves ~1.86 GB VRAM, enables batch 3-4; not compatible with --compile)
python -m research.train --config 360m_mla --steps 50000 --batch-size 4 --chunked-ce --ce-chunk-size 256

# Pre-train with activation checkpointing + chunked CE (batch 32 fits at 8.61 GB)
python -m research.train --config 360m_mla --steps 50000 --batch-size 32 --gradient-checkpointing --chunked-ce --ce-chunk-size 256

# Pre-train with YaRN 4x context extension (1024 -> 4096 tokens)
python -m research.train --config 360m_mla --steps 50000 --batch-size 1 --seq-len 4096 --yarn-factor 4.0 --yarn-orig-len 1024 --gradient-checkpointing --chunked-ce

# Pre-train with GaLore optimizer (for large models; slower on 360M)
python -m research.train --config 360m_mla --steps 50000 --optimizer galore

# Pre-train with safetensors checkpoint format (pickle-safe, mmap load)
python -m research.train --config 360m_mla --steps 50000 --checkpoint-format safetensors

# Pre-train with EMA weight averaging (free 15% quality boost)
python -m research.train --config 360m_mla --steps 50000 --ema-decay 0.999 --ema-eval --checkpoint-format safetensors

# Pre-train with MoE (Mixture of Experts, 4 experts top-2, 3x params same FLOPs)
python -m research.train --config 360m_mla --steps 50000 --moe --moe-experts 4 --moe-topk 2

# Pre-train with RMSNorm (faster than LayerNorm)
python -m research.train --config 360m_mla --steps 50000 --norm-type rmsnorm

# Pre-train with all optimizations (BitNet + GateSkip + MTP + MoE + RMSNorm)
python -m research.train --config 360m_mla --steps 50000 --bitnet --gateskip --mtp --moe --norm-type rmsnorm --compile

# DoRA fine-tuning (weight-decomposed LoRA, better than vanilla LoRA)
python -c "from research.dora import apply_dora_to_model; from research.model_loader import ModelLoader; from research.config import get_config; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); apply_dora_to_model(m, rank=16, alpha=32)"

# Score and filter synthetic data quality (6-dimension scoring)
python -c "from research.quality_score import score_file; score_file('research/data/all_teachers.jsonl', output_path='research/data/all_teachers_scored.jsonl', min_score=0.3)"

# Curriculum ordering for distillation (easy → hard)
python -c "from research.quality_score import curriculum_order; curriculum_order('research/data/all_teachers_scored.jsonl', 'research/data/all_teachers_curriculum.jsonl', strategy='difficulty')"

# Distill from Qwen 2.5-0.5B teacher (60% ppl reduction)
python -m research.distill --config 360m_mla --checkpoint research/checkpoints/pretrained_llm.safetensors --steps 5000 --temperature 2.0 --alpha 0.5

# Extended distillation with top-K KL (1500x faster) + periodic checkpoints (pause/resume)
# Saves every 100 steps to {save}.step{N}.safetensors, auto-deletes old (keeps last 5)
# Writes status.json every 10 steps for the GUI monitor
# Safeguards: VRAM watchdog (--vram-limit-gb), NaN detection, crash recovery, heartbeat file
$env:PYTHONUNBUFFERED="1"; python -u -m research.distill --config 360m_mla --checkpoint research/checkpoints/distilled_llm.safetensors --steps 30000 --batch-size 2 --seq-len 512 --temperature 4.0 --alpha 0.7 --lr 3e-4 --top-k-kl 100 --save-every 100 --keep-checkpoints 5 --status-file research/checkpoints/distill_status.json --save research/checkpoints/distilled_v2.safetensors --vram-limit-gb 11.0

# Resume distillation from a periodic checkpoint (e.g. after pausing)
$env:PYTHONUNBUFFERED="1"; python -u -m research.distill --config 360m_mla --resume research/checkpoints/distilled_v2.step5000.safetensors --steps 30000 --batch-size 2 --seq-len 512 --temperature 4.0 --alpha 0.7 --lr 3e-4 --top-k-kl 100 --save-every 100 --keep-checkpoints 5 --status-file research/checkpoints/distill_status.json --save research/checkpoints/distilled_v2.safetensors --vram-limit-gb 11.0

# Distillation with batch-size 1 (lower VRAM, ~7.3 GB, ~4500 tok/s, ETA ~1h20m for 21K steps)
$env:PYTHONUNBUFFERED="1"; python -u -m research.distill --config 360m_mla --checkpoint research/checkpoints/pretrained_llm.safetensors --resume research/checkpoints/distilled_v2.step8400.safetensors --steps 30000 --temperature 2.0 --alpha 0.5 --top-k-kl 100 --save-every 50 --keep-checkpoints 5 --status-file research/checkpoints/distill_status.json --vram-limit-gb 11.0 --batch-size 1

# Launch GUI monitor (reads status.json, 30fps continuous render loop, heart-monitor chart animation)
python -m research.monitor

# Distill safeguards reference:
#   --vram-limit-gb 11.0    Abort before OOM freeze (12 GB card)
#   --save-every 50         Checkpoint frequency (50 = more frequent saves)
#   --keep-checkpoints 5    Auto-delete old checkpoints (keep last 5)
#   --status-file PATH      Write progress JSON every 10 steps (for GUI)
#   --heartbeat-file PATH   Write timestamp every 10 steps (hang detection)
# Crash recovery: on KeyboardInterrupt/CUDA error, saves .interrupt_stepN.safetensors or .emergency_stepN.safetensors
# NaN detection: checks weights every 50 steps, saves .nan_stepN.safetensors if corrupted

# Unified training safeguards (2026-07 refactor — ALL training scripts: train, distill,
# distill_synthetic, distill_multi, sft_align, dpo_align, online_learn):
#   All support --save-every/--keep-checkpoints/--status-file/--heartbeat-file/--vram-limit-gb/--resume.
#   Periodic saves now store FULL training state: weights (.safetensors) + optimizer/EMA/RNG
#   sidecar (<ckpt>.train.pt) + step in <ckpt>.meta.json (research/checkpoint_io.py:
#   save_training_checkpoint / load_training_state / step_checkpoint_path /
#   cleanup_step_checkpoints / emergency_save; guards in research/training_utils.py:
#   has_nan_params, vram_exceeded, write_status_json, write_heartbeat, add_safeguard_args).
#   --resume <ckpt> restores weights + optimizer momentum + EMA + RNG + step counter and
#   continues the loop where it left off (works with .stepN/.interrupt_stepN/.nan_stepN files).
#   Ctrl-C now saves an .interrupt_stepN checkpoint before exiting in every script.
# Notes:
#   - train.py: periodic saves write both pretrained_llm.stepN.<ext> (kept last N) and the
#     canonical pretrained_llm.<ext>; duplicate final save removed.
#   - sft_align.py default output changed to sft_llm.safetensors (was unsafe torch.save .pt).
#   - distill_synthetic.py: --save path now holds the resumable state; the EMA-merged
#     deliverable goes to <save-stem>.ema.safetensors.
#   - Loading a full-state checkpoint shows "Unexpected keys: ['step', ...]" — harmless.

# Distillation speed/quality overhaul (2026-07, research/distill.py):
#   - Hidden-state loss path (default): student never materializes [B*T, 151936] logits.
#     KL runs on gathered top-K logits (chunked bmm against head weight); CE uses fused
#     chunked CE (chunked_ce.py). Big VRAM + speed win. --no-hidden-loss restores legacy path.
#   - Teacher runs under torch.inference_mode + SDPA; --teacher-compile for ~1.3x teacher fwd.
#   - Quality bug fixed: gradients were clipped TWICE per step (now once).
#   - --ema-decay 0.999: EMA weights saved to <save-stem>.ema.safetensors at the end.
#   - Periodic saves now store full training state (optimizer/EMA/step) and resume restores them.
#   - BUG FIX: the periodic-checkpoint block was dedented out of the training loop, so
#     --save-every NEVER actually saved mid-run checkpoints. Fixed; .stepN saves work again.
#   - torch.cuda.empty_cache() interval relaxed 50 -> 200 steps (allocator stalls).
# Fast distillation command:
$env:PYTHONUNBUFFERED="1"; python -u -m research.distill --config 360m_mla --checkpoint research/checkpoints/pretrained_llm.safetensors --steps 30000 --batch-size 2 --seq-len 512 --temperature 4.0 --alpha 0.7 --lr 3e-4 --top-k-kl 100 --save-every 100 --keep-checkpoints 5 --status-file research/checkpoints/distill_status.json --save research/checkpoints/distilled_v2.safetensors --vram-limit-gb 11.0 --teacher-compile --ema-decay 0.999

# Train tiny draft model for speculative decoding (2-layer, 177M params)
python -m research.train --config tiny_draft --steps 5000 --batch-size 4 --checkpoint-format safetensors

# Speculative decoding inference (1.5-3x speedup, INT8 quantized main model by default)
python -m research.speculative_decode --main-model research/checkpoints/distilled_llm.safetensors --draft-model research/checkpoints/draft_llm.safetensors --k 4 --temperature 0.8

# Online speculative distillation (draft learns during inference)
python -m research.speculative_decode --main-model research/checkpoints/distilled_llm.safetensors --draft-model research/checkpoints/draft_llm.safetensors --online --learn-rate 3e-4 --save-draft research/checkpoints/draft_learned.safetensors

# Multi-teacher distillation with offline logit caching
python -m research.distill_multi --config 360m_mla --checkpoint research/checkpoints/pretrained_llm.safetensors --teacher Qwen/Qwen2.5-0.5B --steps 5000 --cache-size 10000

# Full online learning with replay buffer + EMA safety (24.3% ppl reduction)
python -m research.online_learn --config 360m_mla --checkpoint research/checkpoints/distilled_llm.safetensors --steps 1000 --replay-buffer-size 1000 --ema-decay 0.999

# Sequence-level distillation from synthetic data (with pretrain mixing to prevent overfitting)
python -m research.distill_synthetic --student-ckpt research/checkpoints/distilled_llm.safetensors --data "research/data/synthetic_*.jsonl" --steps 500 --lr 2e-4 --ema-decay 0.999 --quality-check --mix-pretrain 0.5 --save research/checkpoints/synthetic_mixed.safetensors

# Merge all multi-teacher synthetic data into one file (with dedup + domain inference)
python -m research.merge_synthetic --dedup --output research/data/all_teachers.jsonl

# Distill from all merged multi-teacher data (V1: 1541 samples, 58.1% ppl reduction, but catastrophic forgetting)
python -m research.distill_synthetic --student-ckpt research/checkpoints/distilled_llm.safetensors --data research/data/all_teachers.jsonl --steps 1000 --lr 2e-4 --ema-decay 0.999 --quality-check --mix-pretrain 0.3 --save research/checkpoints/multi_teacher_distilled.safetensors

# V2 distillation (6776 samples, 72.3% ppl reduction, 2654 -> 736, no catastrophic forgetting)
# Uses lower LR (5e-5), more steps (2000), higher pretrain mix (0.7) than V1
python -m research.distill_synthetic --student-ckpt research/checkpoints/distilled_llm.safetensors --data "research/data/all_teachers_v2_scored.jsonl" --steps 2000 --lr 5e-5 --ema-decay 0.999 --quality-check --mix-pretrain 0.7 --save research/checkpoints/synthetic_v2.safetensors

# V3 chat-template distillation (CRITICAL for instruction tuning)
# Reformat data to Qwen chat format (<|im_start|>user/assistant) + CoT for reasoning/math
python -m research.reformat_chat --input research/data/all_teachers_v2_scored.jsonl --output research/data/all_teachers_v2_chat.jsonl --cot-domains reasoning math coding
# Curriculum order (easy -> hard)
python -c "from research.quality_score import curriculum_order; curriculum_order('research/data/all_teachers_v2_chat.jsonl', 'research/data/all_teachers_v2_chat_curriculum.jsonl', strategy='difficulty')"
# Distill with chat template + assistant-only loss masking (use PYTHONUNBUFFERED=1 for live progress)
# NOTE: EMA 0.99 (100-step window) is better than 0.999 here — chat format converges fast
# NOTE: Don't use --quality-check with chat template (val loss on pretrain data rises as model learns chat format)
$env:PYTHONUNBUFFERED="1"; python -u -m research.distill_synthetic --student-ckpt research/checkpoints/distilled_llm.safetensors --data "research/data/all_teachers_v2_chat_curriculum.jsonl" --steps 1200 --lr 5e-5 --ema-decay 0.99 --mix-pretrain 0.7 --save research/checkpoints/synthetic_v3b_chat.safetensors --use-chat-template

# DPO self-reward alignment (run AFTER SFT chat distillation; uses LLM-as-judge, no preference data needed)
$env:PYTHONUNBUFFERED="1"; python -u -m research.dpo_align --config 360m_mla --checkpoint research/checkpoints/synthetic_v3b_chat.safetensors --method self-reward --self-reward-n 100 --self-reward-samples 2 --max-steps 200 --lr 5e-6 --max-seq-length 768 --save research/checkpoints/dpo_v3b.safetensors --use-chat-template

# Merge all teachers including LM Studio Liquid 5232-sample run
python -c "from research.merge_synthetic import merge_files; merge_files(['research/data/lmstudio_liquid.jsonl','research/data/lmstudio_gemma4.jsonl','research/data/lmstudio_glm5.2.jsonl','research/data/lmstudio_grok3.jsonl','research/data/lmstudio_lfm2.5.jsonl','research/data/synthetic_coding.jsonl','research/data/synthetic_knowledge.jsonl','research/data/synthetic_reasoning.jsonl'], 'research/data/all_teachers_v2.jsonl')"

# Benchmark all architecture components (speed + memory comparison)
python -m research.benchmark_suite --checkpoint research/checkpoints/distilled_llm.safetensors

# Generate synthetic data via LM Studio API (requires LM Studio server running on port 1234)
# Teacher lineup: liquid/lfm2.5-1.2b (general), gemma-4-12b-obliterated (coding/reasoning),
# adi-qwen2.5-14b-glm5.2-general (GLM-5.2 distill, coding+reasoning)
python -m research.generate_synthetic --output research/data/lmstudio_lfm2.5.jsonl --num-samples 500 --domains coding reasoning knowledge math writing --base-url http://localhost:1234/v1 --model liquid/lfm2.5-1.2b --temperature 0.8 --max-tokens 512 --concurrency 4

# Gemma-4-12B requires thinking disabled (buggy in LM Studio):
python -m research.generate_synthetic --output research/data/lmstudio_gemma4.jsonl --num-samples 300 --domains coding reasoning knowledge math writing --base-url http://localhost:1234/v1 --model gemma-4-12b-obliterated --temperature 0.8 --max-tokens 512 --concurrency 4

# Load a model via LM Studio API (JIT loading also works on first request):
# POST http://localhost:1234/api/v1/models/load  body: {"model":"gemma-4-12b-obliterated","context_length":4096,"flash_attention":true}

# SFT for tool-calling
python -m research.sft_align --config 360m_mla --max-samples 10000 --max-steps 500 --max-seq-length 1024

# SFT with LISA (train only top-k important layers per step; saves optimizer VRAM)
python -m research.sft_align --config 360m_mla --max-samples 10000 --max-steps 500 --lisa --lisa-k 8 --lisa-interval 20

# DPO alignment (with frozen reference model; needs pretrained checkpoint)
python -m research.dpo_align --config 360m_mla --method dpo --checkpoint research/checkpoints/pretrained_llm.safetensors --beta 0.1 --max-steps 500

# ORPO alignment (no reference model needed; saves ~1.5 GB VRAM vs DPO)
python -m research.dpo_align --config 360m_mla --method orpo --checkpoint research/checkpoints/pretrained_llm.safetensors --max-steps 500

# Self-rewarding alignment (no preference data needed — model judges itself)
python -m research.dpo_align --config 360m_mla --method self-reward --checkpoint research/checkpoints/pretrained_llm.safetensors --self-reward-n 200 --max-steps 500

# Pre-train with BitNet ternary weights (10x weight compression)
python -m research.train --config 360m_mla --steps 50000 --bitnet --checkpoint-format safetensors

# Pre-train with GateSkip (15% compute savings at inference)
python -m research.train --config 360m_mla --steps 50000 --gateskip --checkpoint-format safetensors

# Pre-train with MTP (multi-token prediction, enables self-speculative decoding)
python -m research.train --config 360m_mla --steps 50000 --mtp --mtp-n 4 --checkpoint-format safetensors

# Pre-train with all optimizations (BitNet + GateSkip + MTP)
python -m research.train --config 360m_mla --steps 50000 --bitnet --gateskip --mtp --mtp-n 4 --checkpoint-format safetensors

# Wanda pruning (20% sparsity, no retraining needed)
python -m research.wanda_prune --config 360m_mla --checkpoint research/checkpoints/pretrained_llm.safetensors --sparsity 0.2 --n-samples 64

# Model merging (SLERP / TIES / DARE)
python -m research.merge_models --method slerp --alpha 0.5 --model-a research/checkpoints/pretrained_llm.safetensors --model-b research/checkpoints/pruned_llm.safetensors --out research/checkpoints/merged_llm

# Differentiable DARE-TIES merging (gradient-based, 10x faster than evolutionary)
python -m research.merge_models --method diff-dare-ties --model-a research/checkpoints/pretrained_llm.safetensors --model-b research/checkpoints/pruned_llm.safetensors --model-c research/checkpoints/dpo_llm.pt --out research/checkpoints/merged_optimized.safetensors

# Deduplicate synthetic data files (MinHash + LSH, cross-file)
python -c "from research.dedup import dedup_directory; dedup_directory('research/data/', pattern='lmstudio_*.jsonl', threshold=0.85)"

# Context extension with 100 samples (Entropy-ABF, 4x context)
python -c "from research.entropy_abf import EntropyABF; from research.model_loader import ModelLoader; from research.config import get_config; from transformers import AutoTokenizer; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); abf = EntropyABF(m, t); abf.measure_entropy(['sample text']); abf.compute_scaling(4.0); abf.apply_to_model(); abf.finetune(['long text sample'], steps=100)"

# Progressive distillation from teacher checkpoint series
python -c "from research.progressive_distill import ProgressiveDistiller; from research.model_loader import ModelLoader; from research.config import get_config; from transformers import AutoTokenizer; s = ModelLoader.build_model(get_config('360m_mla')); t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); pd = ProgressiveDistiller(s, ['ckpt1.pt','ckpt2.pt','ckpt3.pt'], t); pd.train(data, steps=5000, model_builder=lambda: ModelLoader.build_model(get_config('360m_mla')))"

# Evaluate checkpoint
python -m research.eval_suite --config 360m_mla --checkpoint research/checkpoints/sft_llm.pt --batch-size 2

# Serve the model with OpenAI-compatible API on port 8080 (INT8 quantized by default, 4-8x speedup)
python -m research.serve --config 360m_mla --checkpoint research/checkpoints/sft_llm.pt --port 8080

# Serve WITHOUT quantization (slower, exact precision)
python -m research.serve --config 360m_mla --checkpoint research/checkpoints/sft_llm.pt --port 8080 --quantize none

# Serve with INT4 quantization (3x speedup, ~1-2% quality loss)
python -m research.serve --config 360m_mla --checkpoint research/checkpoints/sft_llm.pt --port 8080 --quantize int4

# Serve WITH live signal capture (enables /v1/feedback and /v1/code_result endpoints)
python -m research.serve --config 360m_mla --checkpoint research/checkpoints/distilled_llm.safetensors --port 8080 --signal-log research/data/live_training.jsonl

# Fast inference with INT8 quantization (2x speedup, <1% quality loss)
python -c "from research.fast_infer import FastInferenceEngine; from research.model_loader import ModelLoader; from research.config import get_config; from transformers import AutoTokenizer; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); e = FastInferenceEngine(m, t, quantize='int8'); print(e.generate('Hello, world!', max_new_tokens=100))"

# Fast inference with INT8 + CUDA graphs + Medusa (5-10x speedup)
python -c "from research.fast_infer import FastInferenceEngine; from research.model_loader import ModelLoader; from research.config import get_config; from transformers import AutoTokenizer; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); e = FastInferenceEngine(m, t, quantize='int8', use_cuda_graph=True, use_medusa=True); print(e.generate('Hello, world!', max_new_tokens=100))"

# INT4 weight-only quantization (3x speedup, ~1-2% quality loss)
python -c "from research.inference_quant import quantize_model_int4; from research.model_loader import ModelLoader; from research.config import get_config; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); quantize_model_int4(m, group_size=128)"

# Medusa speculative decoding training (add heads, short fine-tune)
python -c "from research.medusa import add_medusa_to_model, MedusaTrainer; from research.model_loader import ModelLoader; from research.config import get_config; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); heads = add_medusa_to_model(m, n_heads=4); trainer = MedusaTrainer(m, heads, lr=1e-4)"

# Benchmark CUDA graph inference speedup
python -c "from research.cuda_graph import benchmark_cuda_graph; from research.model_loader import ModelLoader; from research.config import get_config; import torch; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors').cuda(); ids = torch.randint(0, 151665, (1, 64)).cuda(); benchmark_cuda_graph(m, ids)"

# Compare all inference methods (baseline vs INT8 vs CUDA graph)
python -c "from research.fast_infer import compare_inference_methods; from research.model_loader import ModelLoader; from research.config import get_config; from transformers import AutoTokenizer; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); compare_inference_methods(m, t, 'Hello, world!', max_new_tokens=50)"

# Live LoRA training on captured signals (run alongside or after serve.py)
python -m research.live_learn --base-ckpt research/checkpoints/distilled_llm.safetensors --signals research/data/live_training.jsonl --steps 100 --lr 1e-4 --lora-rank 16

# Rollback to a previous LoRA version (deletes newer versions)
python -m research.live_learn --rollback v003

# Merge a LoRA version into base weights (folds adapter into model)
python -m research.live_learn --merge v005 --base-ckpt research/checkpoints/distilled_llm.safetensors --out research/checkpoints/merged_live.safetensors

# BitNet: convert model to ternary weights (1.58-bit, 10x memory reduction)
python -c "from research.bitnet import convert_model_to_bitnet; from research.model_loader import ModelLoader; from research.config import get_config; m = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); n = convert_model_to_bitnet(m); print(f'Converted {n} layers to ternary')"

# GateSkip: add token-wise layer skipping to model (15% compute savings)
python -c "from research.gateskip import add_gateskip_to_model; from research.model_loader import ModelLoader; from research.config import get_config; m = ModelLoader.build_model(get_config('360m_mla')); n = add_gateskip_to_model(m, d_model=1024); print(f'Wrapped {n} blocks with GateSkip')"

# MoLA: serve with hot-loadable LoRA adapter experts (better than MoE)
# (First train adapters via live_learn.py, then serve with MoLA routing)
python -c "from research.mola import MoLAModel; from research.model_loader import ModelLoader; from research.config import get_config; base = ModelLoader.build_model(get_config('360m_mla'), checkpoint_path='research/checkpoints/distilled_llm.safetensors'); mola = MoLAModel(base, adapter_dir='research/checkpoints/live', d_model=1024, max_adapters=8, cache_size=4); print('MoLA model ready')"

# KV compression: enable 2-bit KV cache + H2O eviction for 32K context
# (Used inside attention forward — see kv_compress.py CompressedKVCache)

# SSA training: train with sparse + full attention alignment
# (Wrap your trainer: ssa = SSATrainer(model); loss = ssa.compute_loss(batch, task_loss_fn))

# EAGLE-3: train lightweight spec decode head (557K params, 318x smaller than draft model)
python -c "from research.eagle import train_eagle_head; from research.model_loader import ModelLoader; from research.config import get_config; from transformers import AutoTokenizer; m = ModelLoader.build_model(get_config('360m_mla')); tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); head = train_eagle_head(m, tok, ['example text'], steps=100); print('EAGLE head trained')"

# Run the existing ForgeAI UI (serves static files + /tool/* on default port)
python launch.py
```

## Key Architecture Details

- Tokenizer: `Qwen/Qwen2.5-0.5B` — real vocab size is `151665` (not 151643; 151643 is the EOS token ID).
- 360M MLA config: `d_model=1024`, `n_layers=19`, `n_heads=16`, `kv_compression_dim=128` — ~362M parameters.
- Training dtype: BF16 AMP; `seq_len=1024`, `batch_size=2` is the safe first baseline on 12 GB.
- Optimizer: `bitsandbytes` 8-bit AdamW is the default and best Triton-free option. It drops VRAM to ~9.35 GB and pushes throughput to ~10,000 tok/s on batch 2. `fused=True` AdamW is used as fallback. `bnb.optim.Lion8bit` is also available via `--optimizer lion` and uses ~8.98 GB, but throughput is similar.
- Compile: with a patched `triton-windows 3.4.0` and `torch.compile`, throughput rises to ~13,300 tok/s and VRAM drops to ~7.5 GB on batch 2. The first ~20 steps are slow while Inductor compiles kernels.
- Data mix: 50% `fineweb-edu` `sample-10BT`, 30% `cosmopedia-v2` `cosmopedia-v2`, 20% `Yxanul/python-finest-pretrain`.
  - Original `HuggingFaceTB/stack-edu-python` does not expose `text`; `python-finest-pretrain` is used as the Python code source.

## Gigatoken (fast bulk tokenizer)

`research/prep_data.py` supports three tokenizer backends via `--tokenizer-backend`:

| Backend | Speedup (this box) | Parity | Notes |
|---|---|---|---|
| `hf` (default) | 1x | — | HuggingFace `AutoTokenizer`, no extra deps. |
| `gigatoken-compat` | ~35x | exact HF | `gt.Tokenizer(hf).as_hf()` wrapper, per-sample `encode`. Drop-in. |
| `gigatoken-native` | ~17x* | exact HF | `gt.Tokenizer(model).encode_batch_list` on `--encode-batch` strings, parallelized in Rust. |

\* Measured on Intel Core Ultra 7 265F for 5k short samples. On EPYC-class CPUs the
native path scales much higher (up to ~691x for Qwen 2/2.5 per upstream benchmarks).
The compat path is faster on this desktop because it avoids Python-list-to-Rust
conversion overhead; the native path wins on many-core servers and very large batches.

- Dependency: `gigatoken==0.9.0` (added to `requirements.txt`). Optional — the `hf`
  backend works without it. Install: `D:\windsurf\ForgeAI\venv\Scripts\pip.exe install gigatoken==0.9.0`.
- All three backends produce **byte-identical** `train.bin`/`val.bin` for the same
  input stream (verified via smoke test). Per-document EOS insertion is preserved
  in every backend.
- `metadata.json` now records `tokenizer_backend`, `eos_token_id`, and `vocab_size`.
- **Why not `gt.encode_files` (disk-spill)?** `encode_files` returns a flat token
  stream with no document boundaries, which makes per-document EOS insertion
  impossible without fragile delimiter-recovery. `encode_batch_list` preserves
  per-doc boundaries trivially and is used instead.

## Known Issues / Warnings

- Qwen's chat template does not support `return_assistant_tokens_mask`; SFT trains on the full conversation.
- Hugging Face caches emit symlink warnings on Windows; non-critical.
- `torch.compile` with Inductor requires `triton-windows==3.4.0.post21` on PyTorch 2.8. Unpatched `triton-windows` crashes during kernel cache writes because Windows' default 260-character path limit is exceeded. The fix is to shorten the temp dir name in `venv\Lib\site-packages\triton\runtime\cache.py` (see repo notes).
- `torch.cuda.make_graphed_callables` works for CUDA graph capture but was ~40-50% *slower* on this 360M MLA model than the eager fused-AdamW path, so it is left as an experimental flag (`--cuda-graph`) and not recommended.

## Triton Windows Path-Length Patch

To use `--compile` on Windows:

```powershell
pip install "triton-windows==3.4.0.post21"
```

Then shorten the temp dir names in `venv\Lib\site-packages\triton\runtime\cache.py`:

```python
# In FileCacheManager.put(...)
# Random ID to avoid any collisions; keep it short for Windows MAX_PATH.
rnd_id = str(uuid.uuid4())[:8]
# use temp dir to be robust against program interruptions
temp_dir = os.path.join(self.cache_dir, f"tmp.{rnd_id}")
os.makedirs(temp_dir, exist_ok=True)
temp_path = os.path.join(temp_dir, filename)
```

This avoids `FileNotFoundError` during Inductor's kernel cache writes and lets `torch.compile` deliver ~30% faster training.

## Triton Consumer Blackwell (sm_120) Patch

`triton-windows==3.4.0.post21` has a bug where it generates `sm_120a` PTX targets for
consumer Blackwell GPUs (RTX 5070/5080/5090). Consumer Blackwell has **no tensor memory
(tcgen05)**, so `sm_120a` causes `illegal memory access` in every Triton kernel. The fix
(from upstream PR #9734, which was reverted) is applied to
`venv\Lib\site-packages\triton\backends\nvidia\compiler.py`:

1. **`sm_arch_from_capability`** — only add "a" suffix for `90 <= capability < 120`
   (Hopper sm_90a, datacenter Blackwell sm_100a). Consumer Blackwell (sm_120) gets no suffix.
2. **PTX `.target` regex** — changed `r'\.target sm_\d+'` to `r'\.target sm_\d+a?'`
   so `sm_120a` targets are correctly replaced.
3. **`make_ttgir` pipeline** — route sm_120 away from tensor memory passes
   (`add_hoist_tmem_alloc`, `add_promote_lhs_to_tmem`, `add_warp_specialize`,
   `add_optimize_tmem_layouts`, `add_interleave_tmem`). These generate tcgen05
   instructions that crash on consumer hardware.

After patching, clear the Triton cache (`~/.triton/cache`) so kernels recompile with
the correct `sm_120` target. Basic Triton kernels (elementwise, reduction, softmax)
work correctly after the patch. `torch.compile` throughput is unchanged (~13,240 tok/s).

**Liger Kernel note:** Liger v0.8.1's Triton CE kernel still crashes on sm_120 even
after this patch, due to additional Triton issues beyond sm_120a. The CuTe DSL backend
(`LIGER_KERNEL_IMPL=cutedsl`) requires `cutlass` which is not installed. Liger is left
installed (`--no-deps`) for future use but is not wired into the training loop.

## Chunked Cross-Entropy (--chunked-ce)

`research/chunked_ce.py` implements a pure-PyTorch fused linear + cross-entropy that
chunks the token dimension and never materializes the full `[B*T, vocab]` logits tensor.
Activated via `--chunked-ce` in `train.py` and `sft_align.py`.

**Measured results (360M MLA, RTX 5070, 40 steps):**

| Config | tok/s | VRAM | Notes |
|---|---|---|---|
| Batch 2, no CE | 10,314 | 9.35 GB | Baseline |
| Batch 2, `--compile` | 13,240 | 7.53 GB | **Best throughput** (existing) |
| Batch 2, `--chunked-ce` | 7,832 | 7.49 GB | -24% tok/s, -1.86 GB |
| Batch 3, `--chunked-ce` | 8,706 | 8.44 GB | Fits in 12 GB |
| Batch 4, `--chunked-ce` | 8,900 | 9.39 GB | Fits in 12 GB |
| Batch 4, no CE | 1,069 | 14.35 GB | CPU spill, unusable |

**When to use:** The chunked CE saves ~1.86 GB VRAM, enabling batch 3-4 without CPU
spill. However, per-step throughput is lower than the batch 2 `--compile` baseline
because chunking creates many small GEMMs. Use `--chunked-ce` when you need memory
headroom (e.g., longer sequences, gradient checkpointing) rather than for speed.
Not compatible with `--compile` (custom autograd Function causes graph breaks).

## Training Speed Optimization (2026-07, RTX 5070 / Blackwell)

**Profile (360M MLA, batch 2, seq 1024, fused+compile, warm cache):**
```
fwd   : 205 ms (62%)   ← pure matmul FLOPs, the bottleneck
bwd   :  83 ms (25%)
opt   :  40 ms (12%)
data  :   0 ms ( 0%)   ← solved with pinned memory + non_blocking H2D
```

**What works (ranked by impact):**
| Optimization | tok/s gain | Notes |
|---|---|---|
| `--optimizer fused` (default) | **6x over bnb** | bnb 8-bit causes graph breaks in Inductor; fused AdamW is fully compile-compatible |
| `--compile` (mode=default) | **6x over eager** | Inductor kernel fusion; warm cache gives ~12,000 tok/s |
| `--val-every 0` | +5-10% wall time | Skip mid-training validation; only run at final step |
| `--fused-clip` | ~4% (noise) | Folds grad clip into optimizer step; marginal |
| Pinned memory + non_blocking | data load = 0% | Already in BinaryDataset by default |
| Persistent compile cache | -warmup only | `TORCHINDUCTOR_FX_GRAPH_CACHE=1` + `TORCHINDUCTOR_AUTOGRAD_CACHE=1` set automatically by `patch_triton_cache_for_windows()`; cache populates in `.tc/` but AOTAutograd cache key includes tensor metadata that changes between runs, so warmup speedup is modest |

**What doesn't work (tested on RTX 5070 / triton-windows 3.4.0):**
| Optimization | Result | Why |
|---|---|---|
| `--compile-mode reduce-overhead` | Crash | triton-windows `OverflowError: Python int too large to convert to C long` in static_cuda_launcher.py |
| `--compile-mode max-autotune` | Crash | Same triton-windows bug |
| `--cuda-graph` (manual) | 5,942 tok/s (slower) | Captures eager kernels without Inductor fusion; compile(default) is better |
| `--bf16-optimizer` | Fallback to bnb | Stock AdamW can't mix bf16 momentum with fp32 grads; needs custom kernel |
| `--liger-ce` + `--compile` | 2,195 tok/s (5x slower) | Liger's Triton kernel causes graph breaks in Dynamo; `allow_in_graph` fails with FakeTensor dtype error. Use `--liger-ce` only without `--compile` for memory savings |
| rmsnorm vs layernorm | Within noise | ~9,000-9,200 tok/s either way |
| medium vs high precision | Within noise | TF32 already enabled |
| FP8 training | Not attempted | Requires Transformer Engine; marginal gain on 360M, high complexity |

**Liger-Kernel (`--liger-ce`):**
- Without compile: 2,173 tok/s, 8.97 GB VRAM (saves ~0.7 GB vs stock CE)
- With compile: BROKEN (graph breaks, 5x slower)
- Use case: memory-constrained scenarios where you need the 0.7 GB headroom and can't use `--compile`

**Optimal pretraining command (12,000 tok/s, 9.67 GB VRAM):**
```powershell
$env:PYTHONUNBUFFERED="1"; python -u -m research.train --config 360m_mla --steps 50000 --batch-size 2 --seq-len 1024 --optimizer fused --compile --fused-clip --val-every 0 --ema-decay 0.999 --train-bin research/data/pretrain_1b/train.bin --val-bin research/data/pretrain_1b/val.bin --checkpoint-format safetensors --save-every 500 --vram-limit-gb 11.0
```
At 12,000 tok/s: 50K steps × 2 × 1024 = 102.4M tokens, ~8.5 hours total.

**New train.py flags:**
- `--val-every N`: validation cadence (0 = only at final step, saves 5-10% wall time)
- `--fused-clip`: fold grad clipping into optimizer step (marginal)
- `--bf16-optimizer`: falls back to bnb 8-bit (stock AdamW can't do bf16 state)
- `--cuda-graph`: manual CUDA graph capture (slower than --compile, use --compile instead)
- `--compile-mode`: default (recommended), reduce-overhead/max-autotune crash on triton-windows

## New Architecture Components (from research)

| Module | File | Purpose | Memory/Speed |
|---|---|---|---|
| BitNet 1.58 | bitnet.py | Ternary weights {-1,0,1} | 10x weight compression |
| GateSkip | gateskip.py | Token-wise layer skip | -15% compute |
| MoLA | mola.py | LoRA adapter hot-load experts | SSD-streamed experts |
| KVQuant+H2O | kv_compress.py | 2-bit KV cache + eviction | 32K context on 12GB |
| SSA | ssa.py | Sparse+full attention training | Sparse inference w/o loss |
| EAGLE-3 | eagle.py | Lightweight spec decode head | 318x smaller draft |

## Novel Runtime Keys (2026-08-08)

Three new TRIVIAL-class keys for inference speed/memory optimization:

- **CLKV (Cross-Layer KV Sharing)**: Share KV cache across consecutive layer pairs -- `keys/clkv_key.py`
  - Novel: odd layers reuse preceding even layer's KV (skip KV compute entirely)
  - 50% KV cache VRAM reduction (28 layers -> 14 unique KV caches)
  - At T=32768: saves ~2.8 GB KV VRAM for ForgeLM V2
  - TRIVIAL class: runtime cache sharing, training-free, reversible
  - LOSSY (adjacent layer KV correlated but not identical, cos~0.7-0.9)
  - Tested: 50% KV VRAM reduction verified, reversible (output matches baseline)
  - Usage: `CLKVKey(share_factor=2).apply(model)` / `.revert(model)`

- **AnchorVocab (Anchor Vocab Pruning)**: Cluster-based top-K logit projection -- `keys/anchor_vocab_key.py`
  - Novel: k-means cluster vocab embeddings into K anchors, only compute exact
    logits for top-C highest-scoring clusters (skip 98%+ of vocab)
  - For ForgeLM V2 (vocab=151936): K=512, C=8 -> ~2374 candidates (1.6% of vocab)
  - 46.8x head FLOPs reduction, 1.92x wall-clock speedup measured on RTX 5070
  - Near-lossless: top-1 token matches baseline (correct token always in top clusters)
  - TRIVIAL class: runtime optimization, training-free, reversible
  - One-time k-means clustering at apply() (~2s for 151936 embeddings)
  - Multi-token prefill uses full projection (fallback), single-token decode uses pruning
  - Usage: `AnchorVocabKey(n_anchors=512, top_clusters=8).apply(model)` / `.revert(model)`

- **AdaTopK (Adaptive Expert Top-K)**: Entropy-based dynamic expert routing -- `keys/adaptive_topk_key.py`
  - Novel: per-token adaptive k based on router logit entropy
  - Confident tokens (low entropy) -> top-1 (50% FFN FLOPs saved)
  - Uncertain tokens (high entropy) -> top-3 (more experts, better quality)
  - ~15% average FFN FLOPs reduction (avg k=1.7 vs fixed k=2)
  - TRIVIAL class: runtime routing, training-free, reversible
  - Orthogonal to Expert Tying (weight sharing) and Expert Consolidation (merging)
  - Usage: `AdaTopKKey(min_k=1, max_k=3).apply(model)` / `.revert(model)`

Test script: `python -u test_novel_keys.py` (all 5 tests pass: CLKV, AnchorVocab, AdaTopK, Combined, Benchmark)

## Expert Training System

### convert_to_v4.py — V2 to V4 AirMoE converter
Re-runnable converter: rebuilds base model from V2, preserves trained experts.
```powershell
python convert_to_v4.py                              # Default: preserve trained, skip existing
python convert_to_v4.py --rebuild-all                 # Wipe and rebuild everything
python convert_to_v4.py --no-int4-base --svd-energy 0.99  # bf16 base, higher SVD quality
```
- 5 phases: load V2 -> separate base/experts -> int4 quantize base -> build expert library -> save manifest
- Preserves `.trained_*` marker files (trained expert files not overwritten)
- Skips existing seed expert files (only rebuilds if missing or --no-skip-existing)

### train_expert.py — Simple expert trainer for ANY domain
Dead simple CLI: throw data at a topic, get an improved expert.
```powershell
# Supervised mode (any domain: math, science, history, code, ...)
python train_expert.py --topic physics --data physics.json --keywords "physics,force,energy,momentum"
python train_expert.py --topic math_algebra --data problems.jsonl --epochs 5 --lr 1e-3
python train_expert.py --topic history --data history.txt --label "History Knowledge"

# Self-play mode (code tasks with I/O verification, delegates to existing system)
python train_expert.py --topic python_algorithms --mode selfplay --epochs 3
```
Data formats: JSON `[{"prompt":"...","completion":"..."}]`, JSONL, CSV (prompt,completion), TXT (one per paragraph)
- Auto-registers new topics in manifest (no source code edits needed)
- Loads existing trained expert for continual improvement (not re-seed each time)
- Only saves if val accuracy improves over baseline (prevents degradation)
- Early stopping with best-weight restoration
- LoRA fine-tuning on last 8 layers, expert 0 only (0.4M trainable params)
- SVD compressed at 0.99 energy (near-lossless)
- Writes `.trained_{topic}` marker file for the loader

### AirMoE V4 Compatibility Fixes
- Weight names: `w_gate/w_up/w_down` -> `w1/w2/w3` (matches actual model)
- Topic paths: hardcoded 4-topic mapping -> directory scan for any topic name
- Manifest: handles both V4 format (`name/base_model_file/topics`) and AirMoE format (`model_name/base_model`)
- `get_expert_by_topic()`: takes any topic string, not just hardcoded names
- `list_topics()`: scans expert directory for all available topics

### Self-Play Expert Training (existing system)
`research/training/self_play_expert_training.py` — code tasks with I/O verification
- GoalTaskGenerator: programming archetypes (fibonacci, sort, is_prime, etc.)
- RecursiveSelfPlay: generate -> execute -> fix -> retry
- Quality scoring: minimalism + efficiency + diversity
- LoRA fine-tuning on accepted solutions only (quality > 0.7)
- Label smoothing + input dropout (fights self-reinforcing overfitting)
- LiveCodeBench + reasoning benchmarks (contamination-free eval)
- `_save_expert` now uses standalone SVD compression (no bake_v4.py dependency)
- `_save_expert` now updates manifest.json (was missing — new topics invisible to router)
- `train_all_topics` now reads topics from manifest (was hardcoded to 4 topics)
