# ForgeLM V2 — Key Application Candidates

> Analysis of which keys can be applied to ForgeLM V2 to **lower compute cost**, **increase speed**, or **improve quality**. Compiled 2026-08-07 from `KEY_MAPPING_MASTER.md`, `KEY_NOVELTY_AUDIT.md` (+Part 2), `keystack.py`, and the key source files.
>
> V2 currently has: QK-Norm MLA, DenseFormer DWA, SandwichNorm, Logit Cap, SwiGLU Clamp (all lossless/identity-init). V3 adds Norm Folding (lossless).

---

## Current State — What V2 Has

| Key | Status | Effect |
|---|---|---|
| QK-Norm MLA | Applied (identity init) | Quality (trainable γ), enables FoldQKNormMLA |
| DenseFormer DWA | Applied (identity init) | Quality (cross-layer dense connections) |
| SandwichNorm | Applied (identity init) | Stability (Pre+Post norm) |
| Logit Cap | Applied (runtime ±30) | Stability (prevents extreme logits) |
| SwiGLU Clamp | Applied (runtime) | Stability (GPT-OSS clamped SwiGLU) |

V3 (unpublished) adds Norm Folding — eliminates 113 norm ops/forward (lossless).

---

## Tier 1 — ALREADY IMPLEMENTED, LOSSLESS, ready to apply now

These keys have `.py` files, are verified lossless, and are NOT yet in V2. Highest priority — zero quality risk.

### 1A. FoldQKNormMLA (`fold_qknorm_mla_key.py`) — FULL lossless
- **Goal:** Speed (compute cost reduction)
- **Effect:** Absorbs QK-Norm scales γ into `q_proj` and `k_up_proj`. Eliminates **56 per-dimension norm multiplies** per forward pass (28 layers × 2 norms). After folding, norm layers only compute the dynamic RMS scalar.
- **Safety:** Lossless with identity init (γ=1 → absorption is a no-op). Reversible.
- **Composes with:** NormFolding (V3) — both fold norms into weights. Apply FoldQKNormMLA first, then NormFolding.
- **Published:** arxiv 2606.16310 (QK-Normed MLA, verified 400M runs)
- **Verdict:** APPLY. This is the natural next step — V2 already has QK-Norm, this just folds it away.

### 1B. NormFolding (`norm_folding_key.py`) — FULL lossless [already in V3]
- **Goal:** Speed (compute cost reduction)
- **Effect:** Folds ALL RMSNorm into adjacent Linear weights. Eliminates **113 norm operations** per forward pass (57 layer norms + 56 QK-norms).
- **Safety:** Lossless, verified. Already applied to V3.
- **Verdict:** Already in V3. If targeting V2 specifically, fold FoldQKNormMLA + NormFolding together → V3+.

### 1C. Knowledge Pack (`knowledge_pack_key.py`) — LOSSLESS
- **Goal:** Quality (zero-token knowledge injection)
- **Effect:** Pre-compute KV caches for knowledge domains, inject at inference. Zero token cost. Contrastive steering via value deltas (mid-layers 33-66%).
- **Safety:** Lossless — injects pre-computed KV, doesn't modify weights.
- **Verdict:** APPLY for quality. Domain-specific (coding, math packs). Pairs with CoT Knowledge Pack.

### 1D. CoT Knowledge Pack (`cot_knowledge_pack_key.py`) — LOSSLESS
- **Goal:** Quality (reasoning traces as KV packs)
- **Effect:** Combines KnowledgePack + chain-of-thought traces from self-play. Pre-computes KV from reasoning, injects at inference.
- **Safety:** Lossless — safe for V2 and expert packs.
- **Verdict:** APPLY for quality. Best paired with the self-play pipeline.

### 1E. Test-Gated Fact Injection (`test_gated_injection_key.py`) — LOSSLESS
- **Goal:** Quality (inject only verified facts)
- **Effect:** Combines FactInjection + test verification. Only injects solutions where `test_passed=True`, scaled by quality score.
- **Safety:** Lossless — only injects verified facts, no random noise.
- **Verdict:** APPLY for quality. This is the SAFE version of fact injection (vs the corrupted `apply_safe_keys.py`).

### 1F. SelfPlay Context Patch (`selfplay_context_patch_key.py`) — LOSSLESS
- **Goal:** Quality (rank-1 patches from self-play)
- **Effect:** Combines ContextPatch + self-play (prompt→solution) pairs. Test-passed → positive patches, failed → negative anti-patches.
- **Safety:** Lossless — safe for V2 and expert packs.
- **Verdict:** APPLY for quality. Requires self-play pipeline to generate patches.

### 1G. Context Patch (`context_patch_key.py`) — LOSSLESS
- **Goal:** Quality (convert ICL to permanent rank-1 patches)
- **Effect:** Extract patches from few-shot examples, apply permanently. Proven for MoE, gating, pre/post-norm architectures.
- **Safety:** Lossless — closed-form rank-1 update.
- **Verdict:** APPLY for quality. Base for SelfPlay Context Patch.

### 1H. Fact Injection (`fact_injection_key.py`) — LOSSLESS (closed-form)
- **Goal:** Quality (write facts into MLP weights)
- **Effect:** COLM 2026 closed-form recipe. Rank-1 weight update per fact using unused hidden dimensions. No gradient descent.
- **Safety:** Lossless when using real fact vectors (NOT the MD5-hash fake facts from `apply_safe_keys.py`).
- **Verdict:** APPLY for quality. Use Test-Gated version (1E) for safety.

---

## Tier 2 — ALREADY IMPLEMENTED, LOSSY but near-lossless (apply with care)

These have `.py` files but introduce small quality loss. Some can be healed by GRAIL.

### 2A. Activation Transmute (`activation_transmute_key.py`) — near-lossless (0.7%)
- **Goal:** Speed (faster activation functions)
- **Effect:** SwiGLU → ReGLU/GeGLU via per-channel affine transform (α, β) grid search. ReGLU is faster than SwiGLU.
- **Safety:** 0.7% relative error (near-lossless). Novel closed-form method.
- **Verdict:** APPLY for speed if 0.7% error is acceptable. Can be healed by GRAIL.

### 2B. Lossless Quant Chain (`lossless_quant_key.py`) — near-lossless
- **Goal:** Compute cost / memory (4x compression)
- **Effect:** SpinQuant→QuaRot→int4. Hadamard rotation smooths outliers → int4 error drops 3-5x. Target: ~2GB total VRAM.
- **Safety:** Near-lossless (rotation is lossless, int4 GPTQ is near-lossless).
- **Verdict:** APPLY for memory/compute. Biggest single win for VRAM-constrained deployment (RTX 5070 12GB).

### 2C. WQ Elimination (`wq_elim_key.py`) — LOSSY without fine-tune
- **Goal:** Speed / compute (saves 25% attention params = 66M)
- **Effect:** Replace Q projection with identity. Saves 25% attention params.
- **Safety:** LOSSY without fine-tuning. Needs fine-tune to recover quality. Can be healed by GRAIL.
- **Verdict:** APPLY only with fine-tuning or GRAIL compensation. Otherwise skip for V2.

### 2D. Expert Consolidation (`expert_consolidation_key.py`) — LOSSY (healable)
- **Goal:** Speed (fewer experts = fewer FLOPs)
- **Effect:** Merge similar MoE experts via cosine similarity threshold + weighted averaging + router redirect.
- **Safety:** Lossy, but healable by GRAIL (error 2.19 → 0.00008).
- **Verdict:** APPLY with GRAIL compensation for speed. Reduces active expert count.

### 2E. GRAIL Compensation (`grail_key.py`) — LOSSLESS (heals lossy keys)
- **Goal:** Quality (heal lossy transforms)
- **Effect:** Gram matrix + ridge regression reconstruction map. Makes WQ Elim, Wanda, Expert Consolidation near-lossless.
- **Safety:** Lossless — it's a healing key, not a transform itself.
- **Verdict:** APPLY alongside any lossy key (2C, 2D) to make them safe for V2.

---

## Tier 3 — ALREADY IMPLEMENTED, LOSSY (not recommended for V2 without fine-tune)

### 3A. Binary g128 (`binary_g128_key.py`) — LOSSY
- **Goal:** Compute cost (14.2x compression, 1.125 bits/weight)
- **Effect:** Binary {±1} with group-wise FP16 scales. Group scale absorption into adjacent norms.
- **Safety:** LOSSY. AGENTS.md explicitly says "do NOT apply to ForgeLM V2 or expert packs."
- **Verdict:** SKIP for V2. Reserve for extreme-compression deployment variants.

### 3B. 4-bit KV Cache (`kv4bit_key.py`) — LOSSY
- **Goal:** Memory (KV cache compression)
- **Effect:** Group-wise 4-bit KV quantization with scale absorption into QK-Norm.
- **Safety:** LOSSY. AGENTS.md says "do NOT apply to ForgeLM V2 or expert packs."
- **Verdict:** SKIP for V2. Consider for long-context deployment where KV memory dominates.

### 3C. Hybrid Linear Attention (`hybrid_linear_key.py`) — LOSSY
- **Goal:** Speed (O(T·d) not O(T²·d), 75% linear attention)
- **Effect:** 75% linear / 25% full attention. Only 25% of layers grow KV cache. Adaptive split by attention entropy.
- **Safety:** LOSSY (architecture change). AGENTS.md says "do NOT apply to ForgeLM V2."
- **Verdict:** SKIP for V2. Best for long-context (>8K) where quadratic attention dominates.

---

## Tier 4 — RECENTLY IMPLEMENTED (8/6), runtime keys, apply at inference

These were implemented on 2026-08-06 but are NOT yet in the V2 checkpoint or apply scripts. They're runtime patches (TRIVIAL), applied to the model object at load time, not baked into weights.

### 4A. NormGatedMoD (`norm_gated_mod_key.py`) — TRIVIAL, novel
- **Goal:** Speed (skip layers dynamically)
- **Effect:** Skip transformer layers whose residual-delta norm ‖block(x) - x‖ falls below a per-layer threshold. No router parameters — uses free residual signal. Lossless at init (threshold=0.0 = never skip).
- **Safety:** Lossless at init. After calibration, skips layers → speedup with small quality trade-off.
- **Novelty:** NOVEL (ForgeAI research). No published method uses residual-delta norm as gate.
- **Verdict:** APPLY for speed. Calibrate on representative data. Biggest speed win with no weight changes.

### 4B. PerQueryTemp (`per_query_temp_key.py`) — TRIVIAL, lossless at init
- **Goal:** Quality (per-query adaptive attention temperature)
- **Effect:** T(Q) = softplus(w·Q + b). Init w=0, b=√d → standard attention (lossless). Fine-tune w to specialize per head.
- **Safety:** Lossless at init. Quality gains require fine-tuning w.
- **Published:** SSA (NeurIPS 2024), Thermometer of Thoughts (ACL 2026)
- **Verdict:** APPLY (lossless at init). Quality gains come from fine-tuning.

### 4C. Softpick (`softpick_key.py`) — TRIVIAL
- **Goal:** Quality (0% attention sinks, sparse attention, better quantization)
- **Effect:** Replace softmax with softpick = softmax(x) * sigmoid(x). Low-scoring positions → exactly zero. Kills attention sinks.
- **Safety:** Drop-in replacement. Paper says quality may need fine-tune, but 0% sink rate + better quant robustness are immediate.
- **Published:** ACL 2026 Findings (zuhri-etal-2026). Has FlashAttention-2 kernel mod.
- **Verdict:** APPLY for quality + quantization robustness. Pairs well with Lossless Quant Chain (2B) — fewer outliers = better int4.

---

## Tier 5 — PROPOSED but NOT yet implemented (need code first)

From KEY_MAPPING_MASTER + KEY_NOVELTY_AUDIT. These are the highest-value unimplemented keys.

### FULL lossless (Tier 1 priority — implement first):

| Key | Goal | Effect | Published? |
|---|---|---|---|
| `asymmetric_tying_key.py` | Speed/memory | SVD decompose embed+head, tie top-k, free residuals. Lossless at init. | PUBLISHED (PIT, arxiv 2602.04556) |
| `folded_head_moe_key.py` | Speed | Fold router into Q projection (one matmul instead of three). | ADJACENT (needs post-hoc gate distillation → PARTIAL) |

### TRIVIAL runtime (Tier 2 — zero weights, instant):

| Key | Goal | Effect | Published? |
|---|---|---|---|
| `hash_router_key.py` | Speed | Zero router params (hash(token,layer) % n_experts). Poor quality (baseline only). | PUBLISHED (Roller 2021) |
| `ghost_moe_key.py` | Speed/memory | 3-tier expert serving (resident→redirect→prefetch). Extends AirMoE. | NOVEL |
| `conf_spec_key.py` | Speed | Conformal threshold for DSpark verification (adaptive vs fixed). | ADJACENT (ORCA + DSpark) |
| `bidirectional_key.py` | Quality | Native infill mask (MAGNET-style). Bidirectional on known tokens, causal on gap. | PUBLISHED (MAGNET, ACL 2025) |
| `nope_scope_key.py` | Speed | Mask-based positioning, no RoPE arithmetic. | NEW (concept) |
| `conformal_kv_evict_key.py` | Speed/memory | KV eviction with conformal coverage guarantee. | PUBLISHED (KVCalib, PyPI) |
| `task_precision_key.py` | Speed | Dynamic precision routing by task (high-precision for active weights). | PUBLISHED (QuantClaw) |
| `matryoshka_logit_key.py` | Speed | Truncated head (coarse-to-fine vocab projection). | PUBLISHED-adjacent (MRL) |
| `asentmax_key.py` | Quality/Speed | Sparse↔dense temperature knob (1000x extrapolation). | PUBLISHED (ICLR 2026) |
| `radix_kv_key.py` | Speed | Prefix KV sharing across requests (RadixAttention). | PUBLISHED (SGLang) |
| `moa_key.py` | Speed | Mixture of Attention Heads — route to 50-90% heads. | PUBLISHED (MoH, ICML 2025) |
| `dynamic_soup_key.py` | Quality | Model merging at inference (AdaMix-style interpolation). | PUBLISHED |
| `sae_steering_key.py` | Quality | SAE feature steering at hook points. | PUBLISHED (SAELens) |
| `adaptive_ttc_key.py` | Quality | Test-time compute scaling (SOLVE-THEN-LEARN, closed-form). | PUBLISHED |

### PARTIAL closed-form (Tier 3 — need calibration data):

| Key | Goal | Effect | Published? |
|---|---|---|---|
| `shared_basis_key.py` | Speed/memory | SVD decompose weights on layer axis → basis + coefficients. | PUBLISHED (Basis Sharing) |
| `defactoring_key.py` | Quality | Project fact directions out of FFN (decouple knowledge from computation). | ADJACENT (PISCES) |
| `indexed_attn_key.py` | Speed | Sparse attention via halfspace range searching (zero false negatives). | PUBLISHED (Louver) |
| `polyglu_transmute_key.py` | Speed | Per-neuron activation choice via grid search (extends ActivationTransmute). | ADJACENT (PolyGLU) |
| `clamp_ternary_key.py` | Speed | Clamp threshold = BitNet absmean (extends SwiGLUClamp + BitNet). | ADJACENT |
| `pocket_expert_key.py` | Speed/memory | SVD-compress + int4-quantize experts on disk (extends AirMoE). | ADJACENT (ForgeAI composition) |
| `compressed_pack_key.py` | Speed/memory | RotorQuant→4-bit KV pack compression. | ADJACENT (ForgeAI composition) |
| `per_domain_rotor_key.py` | Speed/memory | Per-domain Givens rotation for KV. | PUBLISHED (xKV) |
| `shrink_ffn_key.py` | Speed | Project base FFN to smaller intermediate (closed-form). | PUBLISHED (SliceGPT) |
| `weight_spectrum_key.py` | Quality | Per-token rank-1 ΔW from hidden state (closed-form, FAAST). | PUBLISHED (FAAST) |
| `dora_patch_key.py` | Quality | Rank-1 patch to DoRA direction component. | ADJACENT (Pico + DoRA) |
| `expert_genesis_key.py` | Quality | Spawn expert copy + Fact Injection (continual learning, no forgetting). | NOVEL |
| `uncertain_learn_key.py` | Quality | Uncertainty-gated Fact Injection (consensus→inject). | ADJACENT (DiSCTT) |
| `mech_distill_key.py` | Quality | Extra loss terms (attn-pattern + routing alignment). | PUBLISHED (OISD) |
| `tempo_em_key.py` | Quality | EM framing for TTT with critic recalibration. | PUBLISHED (TEMPO) |

---

## Recommended Application Order for V2

### Phase 1: Lossless speed wins (zero quality risk)
1. **FoldQKNormMLA** (1A) — fold QK-Norm into projections, -56 norm ops/forward
2. **NormFolding** (1B, already V3) — fold all RMSNorm, -113 norm ops/forward
3. **NormGatedMoD** (4A) — calibrate + skip low-contribution layers, ~10-30% speedup
4. **Softpick** (4C) — kill attention sinks, improve quant robustness

### Phase 2: Lossless quality wins (zero quality risk)
5. **PerQueryTemp** (4B) — lossless at init, quality from fine-tune
6. **Knowledge Pack** (1C) + **CoT Knowledge Pack** (1D) — zero-token knowledge injection
7. **Test-Gated Fact Injection** (1E) — inject verified facts only
8. **SelfPlay Context Patch** (1F) — rank-1 patches from self-play

### Phase 3: Near-lossless compression (small quality trade-off)
9. **Lossless Quant Chain** (2B) — SpinQuant→QuaRot→int4, ~2GB VRAM
10. **Activation Transmute** (2A) — SwiGLU→ReGLU, 0.7% error, faster activations
11. **Softpick** (already in Phase 1) — improves int4 quant quality (fewer outliers)

### Phase 4: Lossy with GRAIL healing (needs fine-tune or GRAIL)
12. **WQ Elimination** (2C) + **GRAIL** (2E) — -25% attention params, healed to near-lossless
13. **Expert Consolidation** (2D) + **GRAIL** (2E) — fewer experts, healed to near-lossless

### Phase 5: Implement new keys (highest-value unimplemented)
14. **AsymmetricTying** (Tier 5) — lossless embed/head tying, implement first
15. **MoA/MoH** (Tier 5) — route to 50-90% attention heads, speed
16. **Conformal KV Eviction** (Tier 5) — KV memory with guarantees
17. **Task Precision** (Tier 5) — dynamic precision routing
18. **RadixAttention** (Tier 5) — prefix KV sharing

---

## Synergy Map — Keys that compose

```
Softpick ──► Lossless Quant Chain (fewer outliers → better int4)
     │
     ▼
FoldQKNormMLA ──► NormFolding (fold QK-norm first, then all norms)
     │
     ▼
NormGatedMoD (skip layers → fewer norm ops to fold)
     │
     ▼
Activation Transmute (ReGLU faster than SwiGLU)
     │
     ▼
WQ Elim + GRAIL (heal the lossy Q elimination)
     │
     ▼
Expert Consolidation + GRAIL (heal the lossy expert merge)
     │
     ▼
Knowledge Pack + CoT Pack + Test-Gated Fact Injection (quality layer)
     │
     ▼
SelfPlay Context Patch (continual improvement, no gradient)
```

**The big picture:** Phases 1-2 are pure wins (lossless). Phase 3 is near-lossless compression. Phase 4 needs GRAIL. Phase 5 needs implementation. The self-play pipeline (Knowledge Pack + Fact Injection + Context Patch) is ForgeAI's signature quality path — it adds knowledge without training.

---

## Compute/Speed/Quality Summary

| Key | Compute ↓ | Speed ↑ | Quality ↑ | Risk |
|---|---|---|---|---|
| FoldQKNormMLA | -56 norm ops | ✓ | — | None (lossless) |
| NormFolding | -113 norm ops | ✓ | — | None (lossless) |
| NormGatedMoD | -10-30% layers | ✓✓ | small trade-off | None at init |
| Softpick | — | ✓ (sparse) | ✓ (no sinks) | None (drop-in) |
| PerQueryTemp | — | — | ✓ (adaptive) | None at init |
| Lossless Quant Chain | 4x memory | ✓✓ | — | Near-lossless |
| Activation Transmute | faster act | ✓ | — | 0.7% error |
| WQ Elim + GRAIL | -25% attn | ✓ | — | Healed by GRAIL |
| Expert Cons + GRAIL | fewer experts | ✓ | — | Healed by GRAIL |
| Knowledge Pack | — | — | ✓✓ (zero-token) | None (lossless) |
| Fact Injection | — | — | ✓✓ (closed-form) | None (if verified) |
| Context Patch | — | — | ✓ (rank-1) | None (lossless) |

*Compiled 2026-08-07. Companion to KEY_MAPPING_MASTER.md and KEY_NOVELTY_AUDIT.md.*
