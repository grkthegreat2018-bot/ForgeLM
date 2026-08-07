# Key Mapping Master — Every Idea → Key

> Maps every idea from `NOVEL_SOLUTION_IDEATION.md` (12), `LLM_COMPONENT_ATLAS.md` (~40 what-ifs), and `FIRST_PRINCIPLES_IDEATION.md` (12 F-ideas) to a concrete key: its **class** (TRIVIAL/BI/PARTIAL/FULL), **training need** (None / Minimal / Full), **forward/reverse** sketch, and **proposed filename**. Training is avoided wherever possible.
>
> Compiled 2026-08-06. Companion to the three ideation docs and `research/keys/keystack.py`.

---

## Key Class Refresher

| Class | Criteria | Training? | Examples |
|---|---|---|---|
| **TRIVIAL** | Runtime config / identity-init, no weights to learn | None | LogitCap, SandwichNorm, DenseFormer, AirMoE |
| **BI** | Exists in source, exact copy both directions | None | Embedding, RMSNorm, RoPE |
| **PARTIAL** | Weight transform, not reversible (lossy or needs calib) | None (closed-form) or Minimal | Wanda, MoE Router, Fact Injection |
| **FULL** | Reversible + data→weight + composable | None | MTP, SliceGPT, RotorQuant, SpinQuant |

**Priority order for new keys:** FULL > TRIVIAL > PARTIAL(no-train) > PARTIAL(minimal-train) > NOT-A-KEY(full-train).

---

## Master Table — All Ideas → Keys

### From NOVEL_SOLUTION_IDEATION.md (Ideas 1-12)

| # | Idea | Key Name | Class | Train? | File | Status |
|---|---|---|---|---|---|---|
| 1 | GhostMoE | `ghost_moe_key.py` | TRIVIAL | None | NEW | Extends AirMoEKey (resident→redirect→prefetch tiers) |
| 2 | TTT-Pack | `ttt_pack_key.py` | PARTIAL | Minimal¹ | NEW | KV-delta capture + GRAIL ridge map; pack injection is TRIVIAL |
| 3 | ExpertGenesis | `expert_genesis_key.py` | PARTIAL | None | NEW | Spawn expert copy + Fact Injection (closed-form, no gradient) |
| 4 | FoldedHeadMoE | `folded_head_moe_key.py` | FULL | None | NEW | Fold router into Q projection (lossless, composes QKNormMLA+NormFolding) |
| 5 | ConfSpec | `conf_spec_key.py` | TRIVIAL | None | NEW | Conformal threshold replaces DSpark fixed verify threshold |
| 6 | FastExpertMerge | `fast_expert_merge_key.py` | PARTIAL | Minimal² | NEW | 1-epoch LoRA-experts (gradient≈task-vector) → Expert Consolidation merge |
| 7 | BudgetGuard | `budget_guard_key.py` | TRIVIAL | None | NEW | Per-layer expert budget from forgetting profile (config map) |
| 8 | PocketExpert | `pocket_expert_key.py` | PARTIAL | None | NEW | SVD-compress + int4-quantize experts on disk (extends AirMoE+LosslessQuant) |
| 9 | DoRAPatch | `dora_patch_key.py` | PARTIAL | None | NEW | Rank-1 patch to DoRA direction component (closed-form, extends ContextPatch) |
| 10 | CompressedPack | `compressed_pack_key.py` | PARTIAL | None | NEW | RotorQuant→4-bit KV pack compression (extends KnowledgePack+RotorQuant+KV4Bit) |
| 11 | UncertainLearn | `uncertain_learn_key.py` | PARTIAL | None | NEW | Uncertainty-gated Fact Injection (consensus→inject, extends TestGated+SelfPlay) |
| 12 | MTP-EAGLE-DSpark | `mtp_eagle_key.py` | PARTIAL | Minimal³ | NEW | MTP heads as EAGLE draft predictor (wiring config + MTP head fine-tune) |

¹ TTT-Pack: the *pack creation* needs a transient LoRA run (PoT-style), but the pack *injection* is TRIVIAL. The key's forward() captures the KV delta after adaptation; the adaptation itself is external.
² FastExpertMerge: 1-epoch LoRA training per expert (theory proves 1-epoch ≈ converged for merging). The merge is closed-form.
³ MTP-EAGLE-DSpark: MTP heads exist (ForgeAI MTPKey); EAGLE fusion wiring is config; heads may need fine-tune for EAGLE-quality.

### From LLM_COMPONENT_ATLAS.md (Component What-Ifs)

| Comp | What-If | Key Name | Class | Train? | File | Status |
|---|---|---|---|---|---|---|
| 1 Emb | Asymmetric tying | `asymmetric_tying_key.py` | FULL | None | NEW | SVD decompose embed+head, tie top-k, free residuals (lossless at init) |
| 1 Emb | Matryoshka logits | `matryoshka_logit_key.py` | TRIVIAL | None | NEW | Runtime truncated head (coarse-to-fine vocab projection) |
| 1 Emb | Embedding as KV | — | — | — | — | Research concept, not a key yet |
| 1 Emb | No-embedding hash | — | — | Full | — | NOT A KEY (needs from-scratch training) |
| 2 RoPE | NoPE+ScoPE hybrid | `nope_scope_key.py` | TRIVIAL | None | NEW | Mask-based positioning, no arithmetic PE |
| 2 RoPE | Per-layer RoPE base | `per_layer_rope_key.py` | TRIVIAL | None | NEW | Per-layer base hyperparam (config, lossless) |
| 2 RoPE | RoPE on V | `rope_v_key.py` | PARTIAL | Minimal | NEW | Low-freq rotation on V (arch change, fine-tune to recover) |
| 2 RoPE | Inverse RoPE | `inverse_rope_key.py` | PARTIAL | Minimal | NEW | 1/pos rotation on half the bands (formula change, fine-tune) |
| 3 Attn | FoldedHeadMoE (pushed) | `folded_head_moe_key.py` | FULL | None | NEW | (same as Idea 4; QK-Norm scale = gate) |
| 3 Attn | Softpick→delete sinks | `softpick_key.py` | TRIVIAL | None⁴ | NEW | Swap softmax→softpick (drop-in, kills sinks) |
| 3 Attn | Poly-attn as linear feat | `poly_linear_attn_key.py` | PARTIAL | None⁵ | NEW | Kernel trick: polynomial → feature map → linear cost |
| 3 Attn | ASEntmax runtime knob | `asentmax_key.py` | TRIVIAL | None | NEW | Per-prompt sparse↔dense temperature (runtime) |
| 3 Attn | V-only attention | — | — | Full | — | NOT A KEY (needs from-scratch) |
| 3 Attn | Poly-attention one layer | `poly_attn_layer_key.py` | PARTIAL | Minimal | NEW | Add one 3rd-order attention layer (fine-tune) |
| 4 FFN | PolyGLU via transmute | `polyglu_transmute_key.py` | PARTIAL | None | NEW | Per-neuron activation choice via grid search (extends ActivationTransmute) |
| 4 FFN | Shrink base, grow experts | `shrink_ffn_key.py` | PARTIAL | None | NEW | Project base FFN to smaller intermediate (closed-form, extends MoE split) |
| 4 FFN | FFN as KV memory | — | — | — | — | Research concept (unifies FFN+attn KV), not a key yet |
| 4 FFN | Clamp→ternary | `clamp_ternary_key.py` | PARTIAL | None | NEW | Clamp threshold = BitNet absmean (extends SwiGLUClamp+BitNet) |
| 5 Norm | Fold QK-Norm into MLA | `fold_qknorm_mla_key.py` | FULL | None | NEW | Fold scale into kv_down_proj (lossless, extends NormFolding+QKNormMLA) |
| 5 Norm | No norms at all | — | — | Full | — | NOT A KEY (extreme, needs retraining; risky) |
| 5 Norm | Norm γ as pack injection | `norm_gamma_pack_key.py` | PARTIAL | None | NEW | Extract γ delta from domain-FT model, inject as additive γ (closed-form) |
| 6 Res | DenseFormer+MoD | `dynamic_dwa_key.py` | TRIVIAL | None | NEW | Skip-masked DWA (runtime mask on existing DWA weights) |
| 6 Res | Shrinking stream | — | — | Full | — | NOT A KEY (arch taper, needs from-scratch) |
| 6 Res | xHC+AirMoE offload | `xhc_offload_key.py` | TRIVIAL | None | NEW | Stream-level disk offload (runtime, extends AirMoE) |
| 6 Res | MoD from residual norm | `norm_gated_mod_key.py` | TRIVIAL | None | NEW | Skip layers by residual-delta norm (no router params, lossless at init) |
| 7 MoE | Hash-the-token router | `hash_router_key.py` | TRIVIAL | None | NEW | Deterministic hash(token,layer) % n_experts (zero router params) |
| 7 MoE | Conformal router | `conformal_router_key.py` | TRIVIAL | None | NEW | Conformal prediction set → adaptive top-k (runtime) |
| 7 MoE | Router in embedding | `embed_router_key.py` | PARTIAL | Minimal | NEW | Add expert-pref head to embedding (fine-tune to learn) |
| 8 KV | Per-domain rotation | `per_domain_rotor_key.py` | PARTIAL | None | NEW | Per-domain Givens rotation (extends RotorQuant, closed-form fit) |
| 8 KV | KV-as-only-memory | — | — | Full | — | NOT A KEY (extreme research, needs redesign) |
| 8 KV | Conformal KV eviction | `conformal_kv_evict_key.py` | TRIVIAL | None | NEW | Evict by conformal coverage guarantee (runtime) |
| 8 KV | KV pack as replay | `kv_replay_key.py` | TRIVIAL | None | NEW | Inject pack during FT instead of replay tokens (runtime strategy) |
| 9 Out | MTP-head as draft dist | `mtp_draft_key.py` | TRIVIAL | None | NEW | Use MTP head softmax as spec-decode draft distribution (runtime) |
| 9 Out | Logit Cap as quantizer | `logit_cap_quant_key.py` | TRIVIAL | None | NEW | Cap = int4 range (extends LogitCap, runtime) |
| 9 Out | Conformal early-exit | `conformal_exit_key.py` | TRIVIAL | None | NEW | Conformal test at head → emit or re-run deeper (runtime) |
| 10 Loss | TEMPO-EM self-play | `tempo_em_key.py` | PARTIAL | None⁶ | NEW | Add critic recalibration to self-play loop (pipeline key) |
| 10 Loss | Lossless loss (no gradient) | (existing) | PARTIAL | None | — | Already = SelfPlayKey + FactInjectionKey + GRAILKey chain |
| 10 Loss | Conformal loss weighting | `conformal_loss_key.py` | TRIVIAL | None | NEW | Weight CE by conformal confidence (runtime) |

⁴ Softpick: paper says drop-in, but quality may need fine-tune. TRIVIAL swap + optional fine-tune.
⁵ Poly-linear-attn: the kernel-trick expansion is closed-form (no training to derive φ), but matching softmax quality may need fine-tune.
⁶ TEMPO-EM: the recalibration is on a labeled set (no gradient on the model, just critic recalibration). The self-play generation is existing.

### From FIRST_PRINCIPLES_IDEATION.md (F1-F12)

| # | Idea | Key Name | Class | Train? | File | Status |
|---|---|---|---|---|---|---|
| F1 | Indexed Attention | `indexed_attn_key.py` | PARTIAL | None⁷ | NEW | Learn index from attention patterns (calibration clustering, no gradient) |
| F2 | Memory Hierarchy | — | — | Full | — | NOT A KEY (arch redesign, needs from-scratch) |
| F3 | Weight Spectrum | `weight_spectrum_key.py` | PARTIAL | None⁸ | NEW | Per-token rank-1 ΔW from hidden state (closed-form, extends ContextPatch) |
| F4 | Defactoring | `defactoring_key.py` | PARTIAL | None | NEW | Project fact directions out of FFN (closed-form SVD, inverse of FactInjection) |
| F5 | Per-Query Adaptive Temp | `per_query_temp_key.py` | TRIVIAL | None⁹ | NEW | T(Q)=softplus(w·Q+b), init w=0 b=√d (identity=standard, lossless at init) |
| F6 | Continuous-Time RoPE | `continuous_rope_key.py` | TRIVIAL | Minimal¹⁰ | NEW | pos=timestamp (config); needs fine-tune to learn temporal meaning |
| F7 | Mechanistic Distillation | `mech_distill_key.py` | PARTIAL | None¹¹ | NEW | Extra loss terms (attn-pattern + routing + layer-importance); training-time key |
| F8 | Shared Basis Layers | `shared_basis_key.py` | PARTIAL | None | NEW | SVD decompose weight tensor on layer axis → basis + coefficients (closed-form) |
| F9 | Working Memory Registers | — | — | Full | — | NOT A KEY (arch addition, needs from-scratch to learn read/write) |
| F10 | Task-Adaptive Precision | `task_precision_key.py` | TRIVIAL | None | NEW | Profile first tokens → load high-precision for active weights (runtime) |
| F11 | Bidirectional Generation | `bidirectional_key.py` | TRIVIAL | None | NEW | Mode-dependent mask (infill: bidirectional known, causal gap); runtime |
| F12 | Per-Head Learned Kernels | `per_head_kernel_key.py` | PARTIAL | Minimal | NEW | Per-head kernel (RBF/poly), init=linear (identity); fine-tune to specialize |

⁷ Indexed Attention: the index is learned by *clustering* attention patterns on calibration data (k-means, no gradient). At inference it's a lookup.
⁸ Weight Spectrum: ΔW = a(x)·bᵀ where a=x, b=(teacher_target - FFN(x)). Closed-form per-token. If no teacher, b can be a learned predictor (→ minimal training). The closed-form version needs a teacher signal.
⁹ Per-Query Temp: init w=0, b=√d → T(Q)=√d = standard attention. Lossless at init. Fine-tune w to specialize.
¹⁰ Continuous-Time RoPE: the *config* (pos=timestamp) is TRIVIAL. But the model needs fine-tune to learn that temporal gaps carry meaning (unless trained from scratch with timestamps).
¹¹ Mechanistic Distillation: no weight transform, but extra loss terms during distillation. Like SelfPlayKey (a pipeline/training key, not a weight key). No *extra* training — it's added to existing distillation.

---

## Summary Statistics

| Category | Count |
|---|---|
| **NEW keys (no training)** | 38 |
| **NEW keys (minimal training)** | 9 |
| **NOT A KEY (full training needed)** | 7 |
| **Research concepts (not keyable yet)** | 3 |
| **Already exists** | 1 (Lossless loss = SelfPlay+FactInjection+GRAIL chain) |
| **Total ideas mapped** | 58 |

### By class (new keys only):
| Class | Count |
|---|---|
| FULL (lossless, reversible, composable) | 3 (FoldedHeadMoE, Asymmetric Tying, Fold-QKNorm-into-MLA) |
| TRIVIAL (runtime/config, no weights) | 25 |
| PARTIAL (weight transform, closed-form or minimal-train) | 19 |

### NOT A KEY (7 — need full from-scratch training):
- No-embedding hash (Component 1)
- V-only attention (Component 3)
- No norms at all (Component 5)
- Shrinking residual stream (Component 6)
- KV-as-only-memory (Component 8)
- F2 Memory Hierarchy (arch redesign)
- F9 Working Memory Registers (arch addition)

These are architecture redesigns, not weight/runtime transforms. They belong in `config.py` as new architectures, not in `keys/`.

---

## Key Forward/Reverse Sketches (the 3 FULL keys — highest priority)

### FoldedHeadMoEKey (`folded_head_moe_key.py`) — FULL
```
forward(data):
  # data = {"q_proj": W_q, "qk_norm_scale": γ, "router": W_r}
  # Fold router W_r into Q projection via QK-Norm absorption
  # Q_new = (W_q · diag(γ)) concatenated with W_r  [fused projection]
  # The head-selection = top-k of (Q_new[router_dims] · K)
  return {"q_proj": Q_fused}

reverse(weights):
  # Unfold: split Q_fused back into W_q·diag(γ) and W_r
  # SVD or direct split (dimensions are known)
  return {"q_proj": W_q, "qk_norm_scale": γ, "router": W_r}
```
**Lossless:** the fused matmul computes exactly what Q-projection + QK-Norm + router would compute separately. One matmul instead of three.

### AsymmetricTyingKey (`asymmetric_tying_key.py`) — FULL
```
forward(data):
  # data = {"embed": E, "lm_head": W_out}
  # SVD both: E = U_e S_e V_eᵀ, W_out = U_o S_o V_oᵀ
  # Tie top-k shared singular vectors: V_shared = top-k of (V_e + V_o averaged)
  # E = U_e · S_e · [V_shared | V_e_residual]
  # W_out = U_o · S_o · [V_shared | V_o_residual]
  # At init: V_e_residual = 0, V_o_residual = 0 → E = W_out (standard tying)
  return {"embed": E_new, "lm_head": W_out_new, "shared_basis": V_shared}

reverse(weights):
  # Reconstruct full E and W_out from shared + residuals
  return {"embed": E, "lm_head": W_out}
```
**Lossless at init:** residuals = 0 → identical to standard tying. Drift apart during fine-tuning.

### FoldQKNormMLAKey (`fold_qknorm_mla_key.py`) — FULL
```
forward(data):
  # data = {"kv_down_proj": W_kv, "qk_norm_scale": γ}
  # Fold γ into kv_down_proj: W_kv_new = diag(γ) · W_kv
  # The latent is pre-normalized → no runtime QK-Norm needed
  return {"kv_down_proj": W_kv_new}

reverse(weights):
  # Extract γ = row norms of W_kv_new / W_kv
  return {"kv_down_proj": W_kv, "qk_norm_scale": γ}
```
**Lossless:** standard Norm Folding math applied to MLA's down-projection. Extends existing NormFoldingKey.

---

## Implementation Priority (no-training keys first)

### Tier 1 — FULL lossless (implement first, safe for V2/expert packs)
1. `folded_head_moe_key.py` — FoldedHeadMoE (Idea 4 / Atlas 3)
2. `fold_qknorm_mla_key.py` — Fold QK-Norm into MLA (Atlas 5)
3. `asymmetric_tying_key.py` — Asymmetric tying (Atlas 1)

### Tier 2 — TRIVIAL runtime (zero weights, instant apply)
4. `softpick_key.py` — Softpick attention (Atlas 3)
5. `norm_gated_mod_key.py` — MoD from residual norm (Atlas 6)
6. `hash_router_key.py` — Hash router (Atlas 7)
7. `ghost_moe_key.py` — GhostMoE 3-tier (Idea 1)
8. `conf_spec_key.py` — Conformal spec threshold (Idea 5)
9. `bidirectional_key.py` — Native infill mask (F11)
10. `per_query_temp_key.py` — Per-query temperature (F5, lossless at init)
11. `nope_scope_key.py` — NoPE+ScoPE (Atlas 2)
12. `conformal_kv_evict_key.py` — Conformal KV eviction (Atlas 8)
13. `task_precision_key.py` — Task-adaptive precision (F10)
14. `matryoshka_logit_key.py` — Matryoshka logits (Atlas 1)
15. `asentmax_key.py` — ASEntmax runtime knob (Atlas 3)

### Tier 3 — PARTIAL closed-form (no gradient, needs calibration data)
16. `defactoring_key.py` — Defactor facts from FFN (F4)
17. `shared_basis_key.py` — Shared basis layers (F8)
18. `indexed_attn_key.py` — Indexed attention (F1)
19. `polyglu_transmute_key.py` — Per-neuron PolyGLU (Atlas 4)
20. `clamp_ternary_key.py` — Clamp→ternary (Atlas 4)
21. `pocket_expert_key.py` — PocketExpert compression (Idea 8)
22. `compressed_pack_key.py` — Compressed KV pack (Idea 10)
23. `per_domain_rotor_key.py` — Per-domain rotation (Atlas 8)
24. `norm_gamma_pack_key.py` — Norm γ injection (Atlas 5)
25. `dora_patch_key.py` — DoRA direction patch (Idea 9)
26. `expert_genesis_key.py` — ExpertGenesis (Idea 3)
27. `uncertain_learn_key.py` — Uncertainty-gated injection (Idea 11)
28. `weight_spectrum_key.py` — Per-token rank-1 (F3)
29. `shrink_ffn_key.py` — Shrink base FFN (Atlas 4)
30. `mech_distill_key.py` — Mechanistic distillation loss (F7)
31. `tempo_em_key.py` — TEMPO-EM self-play (Atlas 10)

### Tier 4 — PARTIAL minimal-training (1-epoch or fine-tune)
32. `fast_expert_merge_key.py` — 1-epoch expert merge (Idea 6)
33. `mtp_eagle_key.py` — MTP-EAGLE-DSpark (Idea 12)
34. `ttt_pack_key.py` — TTT-Pack (Idea 2)
35. `rope_v_key.py` — RoPE on V (Atlas 2)
36. `inverse_rope_key.py` — Inverse RoPE (Atlas 2)
37. `poly_attn_layer_key.py` — Poly-attention layer (Atlas 3)
38. `embed_router_key.py` — Router in embedding (Atlas 7)
39. `continuous_rope_key.py` — Continuous-time RoPE (F6)
40. `per_head_kernel_key.py` — Per-head kernel (F12)
41. `poly_linear_attn_key.py` — Poly-linear attention (Atlas 3)

### Tier 5 — NOT A KEY (architecture redesigns → config.py, not keys/)
- No-embedding hash, V-only attention, no-norms, shrinking residual, KV-only-memory, F2 memory hierarchy, F9 working memory registers

---

## Registration Plan

When implementing, each new key:
1. Create `research/keys/<name>.py` with class extending `Key`
2. Add to `build_xp_keystack()` in `keystack.py` (in the appropriate section: FULL / BI / PARTIAL / TRIVIAL)
3. Add to `KEY_REGISTRY` in `__init__.py`
4. Add per-key doc to `docs/keys/<name>.md` (format: see `key_swiglu_ffn.md`)
5. Update `AGENTS.md` KeyStack count + key list
6. Add apply script entry to `.devin/apply_new_keys.py` or `.devin/apply_novel_keys.py`

**KeyStack after all additions:** 44 existing + 47 new = 91 keys total (38 no-train + 9 minimal-train).

---

*Compiled 2026-08-06. Source ideas in the three ideation docs. Key framework in `research/keys/base.py` + `keystack.py`.*
