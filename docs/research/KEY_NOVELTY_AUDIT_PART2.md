# Novelty Audit Part 2 — Extended Research + Uncovered Components

> Extends `KEY_NOVELTY_AUDIT.md` with: (1) additional published keys found in a second research round (SAE steering, MoH, fast weights, tokenizer-free, RadixAttention, test-time scaling, model merging, EAGLE-3), and (2) first-principles ideation on components not covered in the original atlas (tokenizer, sampling, batching, prompt structure, compute-as-resource, SAE features, cross-attention injection, model merging at inference).
>
> Compiled 2026-08-06. Companion to `KEY_NOVELTY_AUDIT.md` and `LLM_COMPONENT_ATLAS.md`.

---

## Part A — Additional PUBLISHED Keys (Second Research Round)

### SAE Steering (inference-time feature control)

| Key | Published As | Paper/Code | Notes |
|---|---|---|---|
| `sae_steering_key.py` | **CorrSteer / CRL / FineSteer / SVF / Scalpel / SAE-TS** | arxiv 2508.12535 / 2602.10437 / ACL 2026 / 2607.19364 / github dylanpatriarchi/scalpel / arxiv 2411.0193 | EXACT. SAE decoder vectors = steering directions. Inject at hook points. SAELens library + candle-mi Rust crate have production code. **Implement — use SAELens.** |

### Mixture of Attention Heads (MoA)

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `moa_key.py` | **MoA / SwitchHead / MoH** | EMNLP 2022 / NeurIPS 2024 / ICML 2025 | EXACT. MoA: router selects k heads per token. SwitchHead: 8× fewer attention matrices. MoH: weighted head selection, 50-90% heads, can continue-tune from LLaMA3. **Implement MoH (can fine-tune from existing model).** |

### Fast Weights (dynamic weight generation)

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `fast_weight_key.py` | **FAAST / Ouroboros / SHINE / ReFINE / FwPKM / In-Place TTT** | arxiv 2605.04651 / 2604.02051 / 2602.06358 / 2602.16704 / 2601.00671 / ICLR 2026 | EXACT. FAAST: closed-form fast weights, no gradient. Ouroboros: hypernetwork generates per-step LoRA modulation. SHINE: in-context hypernetwork → LoRA in one pass. In-Place TTT: MLP final projection as fast weights. **FAAST already mapped to `weight_spectrum_key.py`.** |

### Tokenizer-Free (NOT A KEY — needs from-scratch)

| Concept | Published As | Paper | Notes |
|---|---|---|---|
| Tokenizer-free | **T-FREE HAT / BLT / ByteFlow / ATDC** | arxiv 2603.15953 / ACL 2025 / ICLR 2026 / 2605.30080 | PUBLISHED but **NOT A KEY** — all require from-scratch training or major architecture change. Belongs in `config.py` as alternative architectures. |

### Cross-Request KV Sharing (system-level)

| Key | Published As | Paper/Code | Notes |
|---|---|---|---|
| `radix_kv_key.py` | **RadixAttention (SGLang) / Feather** | SGLang / arxiv 2605.06046 | EXACT. Radix tree over KV blocks, automatic prefix reuse. Feather: prefix-homogeneous batching via RL. **Implement as runtime key (TRIVIAL).** |

### Test-Time Compute Scaling

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `adaptive_ttc_key.py` | **THROW / GRACE / SOLVE-THEN-LEARN / Adaptive Sampling** | EACL 2026 / 2606.19354 / 2604.14853 / 2608.03961 | EXACT. THROW: hybrid BoN+beam with PRM-guided branch. GRACE: optimal verification granularity. SOLVE-THEN-LEARN: constrained optimization, closed-form oracle. **Implement SOLVE-THEN-LEARN (has closed-form).** |

### Model Merging at Inference

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `dynamic_soup_key.py` | **Model Soups / Task Arithmetic / Interpolation / AdaMix / LoRA-on-the-Go / UniRoute** | PMLR 2022 / ICLR 2026 / ACL 2026 / 2026 / 2026 / ICLR 2026 | EXACT. AdaMix: dynamic adapter mixing by difficulty. LoRA-on-the-Go: instance-level LoRA selection. UniRoute: route to unseen LLMs. **Implement AdaMix-style (TRIVIAL runtime interpolation).** |

### Speculative Decoding (EAGLE-3)

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `eagle3_key.py` | **EAGLE-3 / EAGLE-3.1** | NeurIPS 2025 / vLLM blog 2026 | EXACT. Multi-layer feature fusion, training-time test, 6.5× speedup. EAGLE-3.1: FC normalization + post-norm feedback. **Already mapped to `mtp_eagle_key.py` — use EAGLE-3 architecture.** |

---

## Part B — First-Principles Ideation on Uncovered Components

The original `LLM_COMPONENT_ATLAS.md` covered 10 components (Embedding, RoPE, Attention, FFN, Norm, Residual, MoE, KV, Output, Loss). These components were **not** covered: Tokenizer, Sampling, Batching, Prompt Structure, Compute-as-Resource, SAE Features, Cross-Attention Injection, Model Merging at Inference. Below: hidden assumptions + what-ifs + key mappings for each.

### Component 0: The Tokenizer

**Hidden assumptions:**
1. Tokenizer is fixed, learned offline (BPE/WordPiece)
2. Vocabulary is a fixed-size discrete set
3. Tokenization is language-agnostic but vocabulary is language-specific
4. Token boundaries are semantic-agnostic (frequency-based)

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Dynamic vocabulary extension per-domain | `vocab_pack_key.py` | PARTIAL | None | NOVEL — extract domain tokens from fine-tuned embedding, inject as new rows |
| Tokenizer-free (byte-level) | — | — | Full | PUBLISHED (T-FREE, BLT, ByteFlow) — NOT A KEY |
| Entropy-based dynamic chunking | — | — | Full | PUBLISHED (BLT) — NOT A KEY |
| Vocabulary pruning (remove unused tokens) | `vocab_prune_key.py` | PARTIAL | None | PUBLISHED-adjacent — count token frequency in calibration, prune zero-use |
| Token embedding as KV cache lookup | — | — | — | Research concept (unifies tokenizer + embedding) |

**Novel key: `vocab_pack_key.py`** — Vocabulary Pack
```
forward(data):
  # data = {"base_embed": E_base, "domain_finetuned_embed": E_domain, "domain_tokens": list}
  # Extract new token embeddings: E_new = E_domain[domain_tokens] - E_base[domain_tokens]
  # The delta captures what the domain fine-tune learned for these tokens
  # Inject as: E_extended = concat(E_base, E_new_rows)
  # Also extend LM head: W_out_extended = concat(W_out_base, E_new_rows)
  return {"embed": E_extended, "lm_head": W_out_extended}

reverse(weights):
  # Extract domain token rows (those beyond base vocab size)
  return {"domain_tokens": ..., "domain_embeds": ...}
```
**Class:** PARTIAL (closed-form, no training). **Novelty:** The *pack* framing (extract from fine-tune, inject into base) is not found. Token embedding extension exists but not as a portable pack.

### Component 11: The Sampling/Decoding

**Hidden assumptions:**
1. Sampling temperature is global (one value for all tokens)
2. Top-k/top-p are fixed hyperparameters
3. All positions get the same sampling budget

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Conformal-calibrated per-query temperature | `conformal_sampler_key.py` | TRIVIAL | None | NOVEL — calibrate T on held-out set via conformal prediction |
| Adaptive sampling budget per query | `adaptive_ttc_key.py` | TRIVIAL | None | PUBLISHED (SOLVE-THEN-LEARN, GRACE) |
| Position-dependent temperature | `pos_temp_key.py` | TRIVIAL | None | NOVEL — hotter at reasoning positions, cooler at factual |
| Conformal early-stop in generation | `conformal_gen_stop_key.py` | TRIVIAL | None | NOVEL — stop when conformal confidence in answer is high |

**Novel key: `conformal_sampler_key.py`** — Conformal Sampler
```
forward(data):
  # data = {"query_features": f, "held_out_scores": [(f_i, score_i)]}
  # Conformal calibration: find T such that P(score > threshold | T) = 1-α
  # Use T for this query's sampling
  return {"temperature": T_calibrated}

reverse(weights):
  # No weights (TRIVIAL)
  return {}
```
**Class:** TRIVIAL (runtime). **Novelty:** Conformal calibration of *sampling temperature* (not just budget) is not found. Adaptive sampling papers calibrate *budget*, not *temperature*.

### Component 12: The Batching/Scheduling

**Hidden assumptions:**
1. Batch is fixed during generation
2. Requests are batched by arrival time or prefix similarity
3. All requests in a batch get the same compute

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Prefix-homogeneous batching | `feather_batch_key.py` | TRIVIAL | None | PUBLISHED (Feather) |
| Cross-request KV sharing | `radix_kv_key.py` | TRIVIAL | None | PUBLISHED (RadixAttention) |
| Conformal compute-budget batching | `conformal_batch_key.py` | TRIVIAL | None | NOVEL — group by conformal-calibrated compute prediction |
| Model routing by difficulty | `model_route_key.py` | TRIVIAL | None | PUBLISHED (UniRoute, TRACE-Router) |

**Novel key: `conformal_batch_key.py`** — Conformal Batch
```
forward(data):
  # data = {"requests": [(query, features)]}
  # Predict compute need per request (token count × layer depth)
  # Conformal calibration: assign each to a compute tier with coverage guarantee
  # Group same-tier requests into batches
  return {"batches": [(tier, [request_ids])]}

reverse(weights):
  return {}
```
**Class:** TRIVIAL (runtime). **Novelty:** Conformal calibration of *batching* (not just routing) is not found. Feather uses RL; UniRoute uses cluster routing. Conformal batching gives a coverage guarantee on compute budget.

### Component -1: The Prompt Structure

**Hidden assumptions:**
1. Prompt is a flat token sequence
2. Prompt is processed once, fully
3. Prompt tokens have uniform importance

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Tree of Thoughts (branching prompt) | — | — | — | PUBLISHED (ToT) — not a key (inference strategy) |
| Soft prompt compression | `soft_prompt_key.py` | PARTIAL | Minimal | PUBLISHED (PIC, PromptEmbedder) |
| Prompt-to-KV-pack conversion | `prompt_to_pack_key.py` | PARTIAL | None | NOVEL — one forward pass → KV pack, reuse across requests |
| Prompt token pruning | `prompt_prune_key.py` | TRIVIAL | None | PUBLISHED (LLMLingua) |

**Novel key: `prompt_to_pack_key.py`** — Prompt-to-Pack
```
forward(data):
  # data = {"prompt_tokens": tokens, "model": M}
  # Run one forward pass: KV = M.encode(tokens)
  # Compress KV via RotorQuant + KV4Bit
  # Store as KnowledgePack
  return {"kv_pack": compressed_KV}

reverse(weights):
  # Decompress pack
  return {"kv": decompressed_KV}
```
**Class:** PARTIAL (one forward pass to create, then TRIVIAL injection). **Novelty:** Knowledge Pack + RadixAttention combination. RadixAttention shares KV *within* a serving session; Prompt-to-Pack makes the KV *portable* across sessions/models. Not found.

### Meta-Component: Compute as a Schedulable Resource

**Hidden assumptions:**
1. Compute is spent uniformly per token
2. Compute is spent once (no speculation)
3. Compute budget is fixed per query

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Adaptive test-time compute | `adaptive_ttc_key.py` | TRIVIAL | None | PUBLISHED (THROW, GRACE, SOLVE-THEN-LEARN) |
| Speculative decoding | `eagle3_key.py` | PARTIAL | Minimal | PUBLISHED (EAGLE-3) |
| Compute futures (skip verification for high-confidence) | `compute_futures_key.py` | TRIVIAL | None | NOVEL — MTP draft + confidence-gated verification skip |
| Layer-wise compute allocation | `norm_gated_mod_key.py` | TRIVIAL | None | NOVEL (from Part 1) — skip by residual-delta norm |

**Novel key: `compute_futures_key.py`** — Compute Futures
```
forward(data):
  # data = {"mtp_head": H, "current_hidden": h, "confidence_threshold": τ}
  # Draft: tokens_draft = H(h)  # MTP predicts next K tokens
  # Score: confidence = max(softmax(tokens_draft))
  # If confidence > τ for position i: skip verification at i
  # Only verify low-confidence positions
  return {"verified_tokens": ..., "skipped_positions": ...}

reverse(weights):
  return {}
```
**Class:** TRIVIAL (runtime). **Novelty:** EAGLE-3 verifies *all* draft tokens. Skipping verification for high-confidence positions (with a calibrated threshold) is not found. This is speculative decoding + conformal gating.

### New Component: SAE Features (Sparse Autoencoder)

**Hidden assumptions:**
1. SAE features are model-specific (not portable)
2. Steering is done at inference time with live SAE encoding
3. Feature selection requires contrastive datasets

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| SAE steering (inference-time) | `sae_steering_key.py` | TRIVIAL | None | PUBLISHED (CorrSteer, CRL, FineSteer, Scalpel) |
| SAE Pack (portable feature pack) | `sae_pack_key.py` | PARTIAL | None | NOVEL — extract features from domain model, store as pack, inject across instances |
| SAE-targeted steering | `sae_targeted_key.py` | PARTIAL | None | PUBLISHED (SAE-TS) |
| SAE feature unlearning | `sae_unlearn_key.py` | PARTIAL | None | PUBLISHED (CRISP) |

**Novel key: `sae_pack_key.py`** — SAE Pack
```
forward(data):
  # data = {"domain_model": M_d, "base_model": M_b, "sae": S, "prompts": P}
  # Encode domain activations: features_d = S.encode(M_d(P))
  # Encode base activations: features_b = S.encode(M_b(P))
  # Delta features: Δf = features_d - features_b (what the domain fine-tune learned)
  # Convert to steering vectors: v = S.decode(Δf)
  # Store as pack: {(layer, feature_idx): coefficient}
  return {"sae_pack": {(layer, idx): coeff}}

reverse(weights):
  # Reconstruct steering vectors from pack
  return {"steering_vectors": ...}
```
**Class:** PARTIAL (closed-form SAE encode/decode, no training). **Novelty:** The *portable pack* framing (extract from domain model, inject into any base model with same SAE) is not found. All existing work does live SAE encoding on the target model.

### New Component: Cross-Attention Knowledge Injection

**Hidden assumptions:**
1. Knowledge enters only through the residual stream (self-attention)
2. Cross-attention requires a separate decoder (encoder-decoder models)

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Cross-attention adapter for frozen LLM | `cross_attn_adapter_key.py` | PARTIAL | Minimal | PUBLISHED (TokenMem) |
| Cross-Attn Pack (extract adapter, inject) | `cross_attn_pack_key.py` | PARTIAL | None | NOVEL — extract adapter from domain fine-tune, inject as pack |
| Hypernetwork-generated LoRA | `hypernet_lora_key.py` | PARTIAL | Minimal | PUBLISHED (SHINE, Ouroboros) |

**Novel key: `cross_attn_pack_key.py`** — Cross-Attention Pack
```
forward(data):
  # data = {"domain_finetuned_model": M_d, "base_model": M_b, "knowledge_passages": K}
  # Extract: the cross-attention adapter weights from M_d
  # Store as pack: {adapter_weights, knowledge_KV}
  return {"cross_attn_pack": {weights, KV}}

reverse(weights):
  return {"adapter_weights": ..., "knowledge_KV": ...}
```
**Class:** PARTIAL (extract adapter weights, no training). **Novelty:** TokenMem trains the adapter; extracting it as a portable pack is not found.

### New Component: Model Merging at Inference

**Hidden assumptions:**
1. Merging happens offline (one-time weight averaging)
2. The merged model is static at inference

**What-ifs:**

| What-If | Key | Class | Train? | Status |
|---|---|---|---|---|
| Model Soups (offline averaging) | `model_soup_key.py` | FULL | None | PUBLISHED |
| Task Arithmetic (offline) | `task_arithmetic_key.py` | FULL | None | PUBLISHED |
| Dynamic adapter mixing | `dynamic_adamix_key.py` | TRIVIAL | None | PUBLISHED (AdaMix) |
| Per-query weight interpolation | `per_query_interp_key.py` | TRIVIAL | None | NOVEL — interpolate full model weights per query (not just adapters) |
| Instance-level LoRA selection | `lora_onthego_key.py` | TRIVIAL | None | PUBLISHED (LoRA-on-the-Go) |

**Novel key: `per_query_interp_key.py`** — Per-Query Interpolation
```
forward(data):
  # data = {"query_features": f, "model_A_weights": W_A, "model_B_weights": W_B}
  # Predict interpolation coefficient: α = router(f)
  # W_mixed = α * W_A + (1-α) * W_B
  # This is a runtime operation (no weight training, just interpolation)
  return {"weights": W_mixed}

reverse(weights):
  # Cannot reverse an interpolation (lossy)
  return {}
```
**Class:** TRIVIAL (runtime). **Novelty:** AdaMix interpolates *adapters* (LoRA); interpolating *full model weights* per query is not found. The "Revisiting Model Interpolation" paper (ACL 2026) does static interpolation; per-query dynamic is a twist. **WARNING:** This requires loading multiple full models → memory cost. Better applied to LoRA adapters (which is AdaMix).

---

## Part C — Updated Novel Key Count

| Source | Novel Keys |
|---|---|
| Part 1 (original audit) | 9 |
| Part 2 (this addendum) | 8 new |
| **Total novel keys** | **17** |

### 8 New Novel Keys from Part 2

| # | Key | Component | Class | Train? | Why Novel |
|---|---|---|---|---|---|
| 13 | `vocab_pack_key.py` | Tokenizer | PARTIAL | None | Vocabulary extension as portable pack (not found) |
| 14 | `conformal_sampler_key.py` | Sampling | TRIVIAL | None | Conformal calibration of temperature (budget is published, T is not) |
| 15 | `conformal_batch_key.py` | Batching | TRIVIAL | None | Conformal batching with compute coverage guarantee (Feather uses RL) |
| 16 | `prompt_to_pack_key.py` | Prompt | PARTIAL | None | Portable KV pack from prompt (RadixAttention is session-only) |
| 17 | `compute_futures_key.py` | Compute | TRIVIAL | None | Confidence-gated verification skip in spec-decode (EAGLE verifies all) |
| 18 | `sae_pack_key.py` | SAE | PARTIAL | None | Portable SAE feature pack (existing work does live encoding) |
| 19 | `cross_attn_pack_key.py` | Cross-Attn | PARTIAL | None | Extract cross-attn adapter as pack (TokenMem trains in-place) |
| 20 | `per_query_interp_key.py` | Merging | TRIVIAL | None | Per-query full-weight interpolation (AdaMix does adapters only) |

---

## Part D — Updated Summary Statistics

| Category | Part 1 | Part 2 | Total |
|---|---|---|---|
| PUBLISHED (implement + cite) | 30 | +11 | 41 |
| ADJACENT (implement, note delta) | 8 | 0 | 8 |
| NOVEL (research contribution) | 9 | +8 | 17 |
| NOT A KEY (arch redesign) | 7 | +4 (tokenizer-free variants) | 11 |
| **Total ideas mapped** | 58 | +19 | 77 |

### By class (all new keys, Part 1 + Part 2):
| Class | Count |
|---|---|
| FULL | 3 (unchanged) |
| TRIVIAL | 25 + 5 = 30 |
| PARTIAL | 19 + 3 = 22 |

### The 17 novel keys (ForgeAI's research output)

**Part 1 (9):** norm_gated_mod, ghost_moe, expert_genesis, ttt_pack, kv_replay, folded_head_moe, inverse_rope, rope_v, per_head_kernel

**Part 2 (8):** vocab_pack, conformal_sampler, conformal_batch, prompt_to_pack, compute_futures, sae_pack, cross_attn_pack, per_query_interp

### Test-when-free priority for the 17 novel keys

| Priority | Key | Why | Est. Cost |
|---|---|---|---|
| 1 | `norm_gated_mod_key.py` | Log residual-delta-norm, no training | 1 hour |
| 2 | `conformal_sampler_key.py` | Calibrate T on held-out set | 1 hour |
| 3 | `compute_futures_key.py` | MTP + confidence threshold | 2 hours |
| 4 | `conformal_batch_key.py` | Group requests by compute prediction | 2 hours |
| 5 | `per_query_interp_key.py` | Interpolate two model weights by query | 2 hours |
| 6 | `vocab_pack_key.py` | Extract domain tokens, extend embedding | 3 hours |
| 7 | `prompt_to_pack_key.py` | One forward pass → KV pack | 3 hours |
| 8 | `sae_pack_key.py` | SAE encode/decode domain delta | 4 hours |
| 9 | `inverse_rope_key.py` | 1/pos rotation formula | 4 hours |
| 10 | `rope_v_key.py` | RoPE on V projection | 4 hours |
| 11 | `ghost_moe_key.py` | 3-tier expert serving | 1 day |
| 12 | `cross_attn_pack_key.py` | Extract cross-attn adapter | 1 day |
| 13 | `kv_replay_key.py` | KV-pack injection for CL replay | 1 day |
| 14 | `expert_genesis_key.py` | Append-only experts | 2 days |
| 15 | `ttt_pack_key.py` | TTT → KV delta → pack | 2 days |
| 16 | `folded_head_moe_key.py` | Post-hoc gate fold | 2 days |
| 17 | `per_head_kernel_key.py` | Per-head kernel family | 3 days |

---

## Part E — Key Implications

### 1. The KeyStack is now ~108 keys (44 existing + 41 published + 8 adjacent + 17 novel - 11 NOT-A-KEY)
Not all are weight-transform keys; many are TRIVIAL runtime. The weight-transform keys (FULL + PARTIAL) are ~28; the rest are runtime/config.

### 2. The "pack" pattern is ForgeAI's signature
6 of the 17 novel keys are "pack" keys: vocab_pack, prompt_to_pack, sae_pack, cross_attn_pack, ttt_pack, kv_replay. The pattern: **extract knowledge from a fine-tuned model as a portable pack, inject into a base model without training.** This is the unifying theme of ForgeAI's novel contributions.

### 3. Conformal prediction is underexploited
3 of the 17 novel keys use conformal prediction: conformal_sampler, conformal_batch, compute_futures (via confidence gating). The conformal framework gives *coverage guarantees* — a mathematical promise that existing heuristic methods lack. This is a research direction worth pursuing as a theme.

### 4. The 11 NOT-A-KEY items belong in `config.py`
- 7 from Part 1 (no-embedding hash, V-only attention, no-norms, shrinking residual, KV-only-memory, F2 memory hierarchy, F9 working memory registers)
- 4 from Part 2 (T-FREE HAT, BLT, ByteFlow, ATDC — all tokenizer-free architectures)
These are alternative architectures, not transforms on existing weights.

---

*Compiled 2026-08-06. Extends `KEY_NOVELTY_AUDIT.md`. New first-principles ideas should be added to `LLM_COMPONENT_ATLAS.md` as Components 0, 11, 12, -1, and meta-component. The 17 novel keys are ForgeAI's research output — verify before publication.*
